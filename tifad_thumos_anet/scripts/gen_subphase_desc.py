#!/usr/bin/env python
"""Two-level sub-actions (2026-09-25): split every PHASE sentence of the
Qwen ensembles into K' ordered SUB-PHASE sentences with the same local LLM
(Qwen2.5-VL-7B-Instruct, text only) that generated the phases.

Input  : configs/<ds>_qwen_ensemble.json  {class: [node sentence, phase_1..phase_10]}
Output : configs/<ds>_subphase_ensemble.json {class: [[sub_1..sub_K'] x 10]}
         (+ <out>.raw.jsonl with the raw generations for auditing)

K' choice: a phase sentence describes ONE movement
in 2-3 clauses (THUMOS median 3, ActivityNet median 2); K'=3 sub-phases
(onset, execution, completion of that movement) can be described without
inventing content, whereas more slots would force the model to pad.

Run (needs transformers >= 4.49; the `hierarchy` env has 4.56):
  CUDA_VISIBLE_DEVICES=3 /data/xiaodan8/anaconda3/envs/hierarchy/bin/python \
      scripts/gen_subphase_desc.py --ds thumos14 --kp 3
  CUDA_VISIBLE_DEVICES=3 /data/xiaodan8/anaconda3/envs/hierarchy/bin/python \
      scripts/gen_subphase_desc.py --ds anet --kp 3
"""
import argparse
import glob
import json
import os
import re
import sys
import time

SYSTEM = ("You are a helpful assistant describing human actions in video for a "
          "computer-vision system. Answer only in the requested format, one "
          "short sentence per line, no extra text.")

USER = (
    "The action is \"{action}\" ({node}).\n"
    "It unfolds in these ordered phases:\n{phases}\n\n"
    "Phase {k} is: \"{phase}\"\n"
    "Split phase {k} into exactly {kp} shorter consecutive sub-phases that "
    "happen one after another within this phase, from its beginning to its "
    "end. Each sub-phase is one visually observable movement, described in "
    "one sentence of about 10 words that names the body parts or objects "
    "involved. Do not repeat the other phases. Output exactly {kp} lines in "
    "the form:\n" + "\n".join("{}. <sentence>".format(i + 1) for i in range(10)).split("\n")[0]
    + "\n2. <sentence>\n...")

LINE_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def readable(name):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return s.replace("_", " ").strip().lower()


def load_model(model_dir, device='8bit'):
    import torch
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForCausalLM as Cls
    tok = AutoTokenizer.from_pretrained(model_dir)
    if device == '8bit':      # ~8 GB: fits one 16 GB card with headroom (gen_pda_desc.py precedent)
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={'': 0})
    else:
        kw = dict(torch_dtype=torch.bfloat16, device_map={'': 0})
    model = Cls.from_pretrained(model_dir, **kw)
    model.eval()
    return tok, model


def chat(tok, model, user, max_new_tokens=160, sample=False, seed=0):
    import torch
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok([text], return_tensors='pt').to(model.device)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=sample)
    if sample:
        torch.manual_seed(seed)
        gen.update(temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inp, **gen)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def parse(raw, kp):
    subs = []
    for line in raw.strip().splitlines():
        m = LINE_RE.match(line)
        if m:
            s = ' '.join(m.group(2).split()).strip().strip('"').rstrip('.').strip()
            if s:
                subs.append(s + '.')
    if len(subs) >= kp:
        return subs[:kp]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=['thumos14', 'anet'])
    ap.add_argument('--kp', type=int, default=3)
    ap.add_argument('--model', default='/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots')
    ap.add_argument('--device', default='8bit', choices=['8bit', 'bf16'])
    ap.add_argument('--out', default='')
    ap.add_argument('--limit', type=int, default=0, help='debug: first N classes')
    args = ap.parse_args()
    ens = json.load(open(f'configs/{args.ds}_qwen_ensemble.json'))
    out_path = args.out or f'configs/{args.ds}_subphase_ensemble.json'
    model_dir = args.model
    if os.path.basename(model_dir.rstrip('/')) == 'snapshots':
        model_dir = os.path.join(model_dir, sorted(os.listdir(model_dir))[0])
    print(f'[gen] model {model_dir}; {len(ens)} classes; K\'={args.kp}; out {out_path}', flush=True)
    tok, model = load_model(model_dir, args.device)

    result = {}
    if os.path.isfile(out_path):          # resume
        result = json.load(open(out_path))
        print(f'[gen] resuming: {len(result)} classes done', flush=True)
    raw_f = open(out_path + '.raw.jsonl', 'a')
    t0 = time.time()
    classes = list(ens)[:args.limit] if args.limit else list(ens)
    for ci, cls in enumerate(classes):
        if cls in result and len(result[cls]) == len(ens[cls]) - 1:
            continue
        node, phases = ens[cls][0], ens[cls][1:]
        plist = '\n'.join(f'{i + 1}. {p}' for i, p in enumerate(phases))
        per_class = []
        for k, phase in enumerate(phases, start=1):
            user = USER.format(action=readable(cls), node=node, phases=plist, k=k,
                               phase=phase, kp=args.kp)
            subs = None
            for t in range(6):
                raw = chat(tok, model, user, sample=(t > 0), seed=t)
                subs = parse(raw, args.kp)
                raw_f.write(json.dumps({'class': cls, 'k': k, 'try': t, 'raw': raw}) + '\n')
                if subs:
                    break
            if subs is None:
                # last resort: split the phase sentence itself on clause boundaries
                parts = [p.strip() for p in re.split(r',|;', phase.rstrip('.')) if p.strip()]
                while len(parts) < args.kp:
                    parts.append(parts[-1])
                subs = [p + '.' for p in parts[:args.kp]]
                print(f'[gen] WARNING fallback split for {cls} phase {k}', flush=True)
            per_class.append(subs)
        result[cls] = per_class
        json.dump(result, open(out_path, 'w'), indent=1, ensure_ascii=False)
        raw_f.flush()
        el = time.time() - t0
        print(f'[gen] {ci + 1}/{len(classes)} {cls} done ({el / 60:.1f} min elapsed)', flush=True)
    print(f'[gen] done: {len(result)} classes -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
