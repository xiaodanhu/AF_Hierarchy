#!/usr/bin/env python
"""Inference for the hierarchical VLM (C3): generate the JSON tree per test window,
parse it, attach token-log-prob confidences, write predictions.jsonl (resumable).

Greedy decoding plus --samples sampled continuations (temperature 0.7) per window;
the union goes through NMS in the evaluators. --adapter '' runs the base model
zero-shot with the same prompt (the untuned comparison).

  CUDA_VISIBLE_DEVICES=2 python scripts/vlm_hier/infer.py --dataset thumos --data data/vlm_hier/thumos_s0 \
      --adapter ckpt/vlm_hier/thumos_s0/final --out logs/vlm_hier/thumos_s0_pred.jsonl
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import Prompter


def generate(model, proc, inp, max_new, do_sample, n, temperature=0.7):
    import torch
    tok = proc.tokenizer
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=max_new, do_sample=do_sample, num_return_sequences=n,
                           temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None,
                           output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
    L = inp['input_ids'].shape[1]
    seqs = g.sequences[:, L:]
    lp = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)   # (n, T_gen)
    outs = []
    for j in range(seqs.shape[0]):
        ids = seqs[j].tolist(); logps = lp[j].float().cpu().numpy().tolist()
        # cut at EOS / pad
        for stop in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids('<|im_end|>')):
            if stop in ids:
                c = ids.index(stop); ids, logps = ids[:c], logps[:c]
        offs, text = C.token_char_offsets(tok, ids)
        outs.append((text, offs, logps))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['thumos', 'fd'], required=True)
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--adapter', default='', help='LoRA dir; empty = base model zero-shot')
    ap.add_argument('--split', default='test')
    ap.add_argument('--samples', type=int, default=2, help='sampled continuations in addition to greedy')
    ap.add_argument('--max_new', type=int, default=0)
    ap.add_argument('--mode', default='4bit'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--ids', default='', help='file with one window id per line: run only these windows (zero-shot subsample, 11-09)')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--size', type=int, default=0); ap.add_argument('--max_frames', type=int, default=0)
    a = ap.parse_args()
    if not a.size:
        a.size = 280 if a.dataset == 'thumos' else 336   # THUMOS: 280 px (100 tokens / frame pair) keeps dense 64 s windows under ~6.5k tokens; FineDiving 336 (small diver)
    import torch
    nested = True
    max_new = a.max_new or (192 if a.dataset == 'fd' else 640)
    samples = [json.loads(l) for l in open(os.path.join(a.data, f'{a.split}.jsonl'))]
    if a.ids:
        _keep = {l.strip() for l in open(a.ids) if l.strip()}; samples = [s for s in samples if s['id'] in _keep]
        print(f'[infer] subset: {len(samples)} windows from {a.ids}', flush=True)
    if a.limit:
        samples = samples[:a.limit]
    P = Prompter(a.dataset, a.data)
    proc, model = C.load_model(mode=a.mode, lora=bool(a.adapter), adapter=a.adapter or None, device=a.device)
    model.eval()
    dev = model.device if a.device == 'cuda' else 'cpu'
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['id'] for l in open(a.out)}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    f = open(a.out, 'a'); t0 = time.time(); n_done = 0
    for i, s in enumerate(samples):
        if s['id'] in done:
            continue
        frames = C.load_frames(s['frames'][:a.max_frames] if a.max_frames else s['frames'], size=a.size or None)
        prompt = P.test_prompt(s)
        inp = C.video_inputs(proc, C.chat_text(proc, prompt), frames, s['fps'], dev)
        preds, raw = [], []
        runs = [('greedy', False, 1)] + ([('sample', True, a.samples)] if a.samples > 0 else [])
        for tag, do_sample, n in runs:
            try:
                outs = generate(model, proc, inp, max_new, do_sample, n)
            except Exception as ex:
                print(f'  [{s["id"]}] generation failed ({tag}): {ex}', flush=True); continue
            for j, (text, offs, logps) in enumerate(outs):
                inst = C.parse_instances(text, s['win_len'], nested=nested)
                confs = C.instance_confidences(text, offs, logps, inst)
                for o, c in zip(inst, confs):
                    o['conf'] = c; o['src'] = tag if n == 1 else f'{tag}{j}'
                preds += inst; raw.append({'src': tag if n == 1 else f'{tag}{j}', 'text': text})
        f.write(json.dumps({'id': s['id'], 'video': s['video'], 'win_start': s['win_start'], 'win_len': s['win_len'], 'preds': preds, 'raw': raw}) + '\n'); f.flush()
        n_done += 1
        if n_done % 20 == 0 or i == len(samples) - 1:
            print(f'[{i+1}/{len(samples)}] {n_done} windows, {(time.time()-t0)/60:.1f} min, last: {raw[0]["text"][:120] if raw else "-"}', flush=True)
    print('done ->', a.out, flush=True)


if __name__ == '__main__':
    main()
