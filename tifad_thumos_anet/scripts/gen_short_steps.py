#!/usr/bin/env python
"""Compress each THUMOS sub-action sentence to <= 3 words so that root + all K steps fit in ONE CLIP prompt
(77 tokens). Control "root + all sub-actions concatenated, no attention" vs our ASA.
Output: configs/thumos14_qwen_ensemble_concatshort.json = {class: [single sentence]} and the step list.
  /data/xiaodan8/anaconda3/envs/hierarchy/bin/python scripts/gen_short_steps.py --device cpu
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_k_desc import load_model, chat

USER = ("The action is \"{action}\". Its ordered steps are:\n{steps}\n"
        "Rewrite each step as a short phrase of at most 3 words (verb first, no articles), keeping the order. "
        "Output exactly {k} lines in the form:\n1. <phrase>\n2. <phrase>\n...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu', choices=['cpu', '8bit', 'bf16'])
    ap.add_argument('--model', default='/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots')
    a = ap.parse_args()
    md = a.model
    if os.path.basename(md.rstrip('/')) == 'snapshots':
        md = os.path.join(md, sorted(os.listdir(md))[0])
    tok, model = load_model(md, a.device)
    src = json.load(open('configs/thumos14_qwen_ensemble.json'))
    out_path = 'configs/thumos14_qwen_ensemble_concatshort.json'; steps_path = 'configs/thumos14_short_steps.json'
    res = json.load(open(steps_path)) if os.path.isfile(steps_path) else {}
    for i, (c, sents) in enumerate(src.items()):
        if c in res and len(res[c]) == len(sents) - 1:
            continue
        name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', c).lower(); K = len(sents) - 1
        short = None
        for t in range(5):
            raw = chat(tok, model, USER.format(action=name, steps='\n'.join(f'{j+1}. {s}' for j, s in enumerate(sents[1:])), k=K), max_new_tokens=120, sample=(t > 0), seed=t)
            lines = [re.sub(r'^\s*\d+[.)]\s*', '', l).strip().strip('.').lower() for l in raw.strip().splitlines() if re.match(r'^\s*\d+[.)]', l)]
            lines = [' '.join(l.split()[:3]) for l in lines if l]
            if len(lines) >= K:
                short = lines[:K]; break
        if short is None:                                      # fallback: first 3 words of each sentence
            short = [' '.join(re.sub(r'^(the|a|an)\s+', '', s.lower()).split()[:3]) for s in sents[1:]]
        res[c] = short; json.dump(res, open(steps_path, 'w'), indent=1)
        print(f'[short] {i+1}/{len(src)} {c}: {short}', flush=True)
    ens = {c: [f'{src[c][0].rstrip(".")}. Steps: ' + '; '.join(res[c]) + '.'] for c in src}
    json.dump(ens, open(out_path, 'w'), indent=1)
    print('done ->', out_path, flush=True)


if __name__ == '__main__':
    main()
