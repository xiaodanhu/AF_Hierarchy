#!/usr/bin/env python
"""Text-only step-duration priors for the hierarchical VLM (C3): the LLM's common knowledge
of how long each sub-action of an action typically takes, as fractions of the action.
Used by ground.py as the fallback when the video-grounded placement is invalid, so no
child span ever comes from the equal split.

  THUMOS   : configs/thumos14_qwen_ensemble.json  (root + K=10 ordered steps per class)
  FineDiving: data/vlm_hier/fd_s1/meta.json       (dive -> ordered members from the definition tree)

  python scripts/vlm_hier/gen_step_prior.py --dataset thumos --device 8bit   (hierarchy env; cpu also works, slow)
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gen_k_desc import SYSTEM, readable, load_model, chat

USER = ("The action is \"{action}\" ({root}).\nIt unfolds in these ordered steps:\n{steps}\n"
        "Using your knowledge of how this action is performed, estimate what fraction of the action's total duration "
        "each step typically takes (some steps are brief, some long). Output ONLY a JSON object: "
        "{{\"fractions\": [f1, f2, ...]}} with exactly {k} positive numbers that sum to 1.")


def parse(raw, k):
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None
    try:
        f = json.loads(m.group(0)).get('fractions')
        f = [float(x) for x in f]
    except Exception:
        return None
    if len(f) != k or any(x <= 0 for x in f):
        return None
    z = sum(f)
    return [x / z for x in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['thumos', 'fd'], required=True)
    ap.add_argument('--device', default='8bit', choices=['cpu', '8bit', 'bf16'])
    ap.add_argument('--model', default='/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.dataset == 'thumos':
        trees = json.load(open('configs/thumos14_qwen_ensemble.json'))
        items = [(c, readable(c), v[0], v[1:]) for c, v in trees.items()]
        out = a.out or 'configs/thumos14_step_prior.json'
    else:
        meta = json.load(open('data/vlm_hier/fd_s1/meta.json'))
        roots = {re.sub(r'^\([A-Z]+\) ', '', k): v[0] for k, v in json.load(open('/data3/xiaodan8/actionformer6/configs/finediving/fd_dive_ensemble.json')).items()}
        items = [(c['id'], c['name'], roots.get(c['name'], c['name']), c['members']) for c in meta['classes']]
        out = a.out or 'configs/finediving_step_prior.json'
    md = a.model
    if os.path.basename(md.rstrip('/')) == 'snapshots':
        md = os.path.join(md, sorted(os.listdir(md))[0])
    tok, model = load_model(md, a.device)
    res = json.load(open(out)) if os.path.isfile(out) else {}
    n_fallback = 0
    for i, (key, name, root, steps) in enumerate(items):
        if key in res and len(res[key]['fractions']) == len(steps):
            continue
        user = USER.format(action=name, root=root, steps='\n'.join(f'{j+1}. {s}' for j, s in enumerate(steps)), k=len(steps))
        fr = None
        for t in range(4):
            fr = parse(chat(tok, model, user, max_new_tokens=160, sample=(t > 0), seed=t), len(steps))
            if fr:
                break
        src = 'llm'
        if not fr:
            fr = [1.0 / len(steps)] * len(steps); src = 'uniform'; n_fallback += 1
        res[key] = {'fractions': fr, 'steps': steps, 'src': src}
        json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
        print(f'[prior] {i+1}/{len(items)} {name}: ' + ' '.join(f'{x:.2f}' for x in fr) + f' ({src})', flush=True)
    print(f'done -> {out} ({n_fallback} uniform fallbacks)', flush=True)


if __name__ == '__main__':
    main()
