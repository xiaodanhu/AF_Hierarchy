#!/usr/bin/env python
"""K-ablation descriptions (2026-09-25): regenerate the
ordered sub-action list of every THUMOS14 class with K children (K=3, 5)
using the same local LLM as the K=10 ensemble, keeping the stored root
sentence (sentence 0) unchanged.

Output: configs/thumos14_qwen_ensemble_k{K}.json {class: [root, child_1..child_K]}
Run (CPU is fine for 20 classes; the `hierarchy` env has transformers 4.56):
  /data/xiaodan8/anaconda3/envs/hierarchy/bin/python scripts/gen_k_desc.py --k 3 --device cpu
"""
import argparse, json, os, re, time

SYSTEM = ("You are a helpful assistant describing human actions in video for a "
          "computer-vision system. Answer only in the requested format, one "
          "sentence per line, no extra text.")
USER = ("The action is \"{action}\" ({root}).\n"
        "Decompose this action into exactly {k} temporally ordered sub-actions "
        "that together cover the whole action from its beginning to its end. "
        "Describe each sub-action in one sentence of about 15 words naming the "
        "visible body movements and objects, as it would be seen in a video. "
        "Output exactly {k} lines in the form:\n1. <sentence>\n2. <sentence>\n...")
LINE_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def readable(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ").strip().lower()


def load_model(model_dir, device):
    import torch
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForCausalLM as Cls
    tok = AutoTokenizer.from_pretrained(model_dir)
    if device == 'cpu':
        kw = dict(torch_dtype=torch.bfloat16, device_map='cpu')
    elif device == '8bit':
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={'': 0})
    else:
        kw = dict(torch_dtype=torch.bfloat16, device_map={'': 0})
    model = Cls.from_pretrained(model_dir, **kw).eval()
    return tok, model


def chat(tok, model, user, max_new_tokens=220, sample=False, seed=0):
    import torch
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok([text], return_tensors='pt').to(model.device)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=sample)
    if sample:
        torch.manual_seed(seed); gen.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inp, **gen)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def parse(raw, k):
    subs = []
    for line in raw.strip().splitlines():
        m = LINE_RE.match(line)
        if m:
            s = ' '.join(m.group(2).split()).strip().strip('"').rstrip('.').strip()
            if s:
                subs.append(s + '.')
    return subs[:k] if len(subs) >= k else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--ds', default='thumos14')
    ap.add_argument('--device', default='cpu', choices=['cpu', '8bit', 'bf16'])
    ap.add_argument('--model', default='/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots')
    args = ap.parse_args()
    ens = json.load(open(f'configs/{args.ds}_qwen_ensemble.json'))
    out_path = f'configs/{args.ds}_qwen_ensemble_k{args.k}.json'
    md = args.model
    if os.path.basename(md.rstrip('/')) == 'snapshots':
        md = os.path.join(md, sorted(os.listdir(md))[0])
    tok, model = load_model(md, args.device)
    res = json.load(open(out_path)) if os.path.isfile(out_path) else {}
    t0 = time.time()
    for i, (cls, sents) in enumerate(ens.items()):
        if cls in res and len(res[cls]) == args.k + 1:
            continue
        root = sents[0]
        subs = None
        for t in range(6):
            raw = chat(tok, model, USER.format(action=readable(cls), root=root, k=args.k), sample=(t > 0), seed=t)
            subs = parse(raw, args.k)
            if subs:
                break
        assert subs, f'could not parse {cls}: {raw}'
        res[cls] = [root] + subs
        json.dump(res, open(out_path, 'w'), indent=1, ensure_ascii=False)
        print(f'[gen k={args.k}] {i + 1}/{len(ens)} {cls} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    print('done ->', out_path, flush=True)


if __name__ == '__main__':
    main()
