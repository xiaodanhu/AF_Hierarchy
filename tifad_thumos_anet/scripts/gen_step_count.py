#!/usr/bin/env python
"""K-selection statistics how many
temporally ordered sub-action steps does each action class NATURALLY have,
according to the same local LLM that wrote the definition trees, when it is
NOT told a fixed K?

For every class: one greedy decomposition + (n-1) sampled ones (temperature
0.7, seeds 1..n-1); the step count of each is the number of numbered lines.
Output {class: {"counts": [...], "mean": m, "lists": [[...], ...]}} plus a
"_summary" entry with the per-class-mean distribution (mean, sd, min, max,
mean+sd, mean+2sd) -> the K story: K = round(mean), round(mean+sd), round(mean+2sd).

Run (CPU, `hierarchy` env; ~20 s per call):
  /data/xiaodan8/anaconda3/envs/hierarchy/bin/python scripts/gen_step_count.py \
      --ens configs/thumos14_qwen_ensemble.json --out configs/thumos14_step_counts.json --n 5
"""
import argparse, json, os, re, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_k_desc import SYSTEM, LINE_RE, readable, load_model, chat

USER = ("The action is \"{action}\" ({root}).\n"
        "Decompose this action into its natural sequence of temporally ordered sub-actions, "
        "from its beginning to its end. Use as many or as few sub-actions as the action really "
        "has; do not pad the list and do not merge distinct steps. Describe each sub-action in one "
        "sentence of about 15 words naming the visible body movements and objects, as it would be "
        "seen in a video. Output one numbered line per sub-action in the form:\n1. <sentence>\n2. <sentence>\n...")


def parse(raw):
    subs = []
    for line in raw.strip().splitlines():
        m = LINE_RE.match(line)
        if m:
            s = ' '.join(m.group(2).split()).strip().strip('"').rstrip('.').strip()
            if s:
                subs.append(s + '.')
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ens', required=True, help='ensemble json {class: [root, ...]}; only the root sentence is used')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=5, help='decompositions per class (1 greedy + n-1 sampled)')
    ap.add_argument('--device', default='cpu', choices=['cpu', '8bit', 'bf16'])
    ap.add_argument('--model', default='/data3/xiaodan8/.cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots')
    ap.add_argument('--strip_prefix', default=r'^\([A-Z]+\) ', help='regex removed from class keys for the prompt (FineDiving "(FWD) ...")')
    args = ap.parse_args()
    ens = json.load(open(args.ens))
    md = args.model
    if os.path.basename(md.rstrip('/')) == 'snapshots':
        md = os.path.join(md, sorted(os.listdir(md))[0])
    tok, model = load_model(md, args.device)
    res = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    res.pop('_summary', None)
    t0 = time.time()
    for i, (cls, sents) in enumerate(ens.items()):
        if cls in res and len(res[cls]['counts']) >= args.n:
            continue
        root = sents[0]
        name = readable(re.sub(args.strip_prefix, '', cls))
        lists = []
        for t in range(args.n):
            raw = chat(tok, model, USER.format(action=name, root=root), max_new_tokens=400, sample=(t > 0), seed=t)
            subs = parse(raw)
            if not subs:                                   # one retry with sampling, then record 0
                raw = chat(tok, model, USER.format(action=name, root=root), max_new_tokens=400, sample=True, seed=100 + t)
                subs = parse(raw)
            lists.append(subs)
        counts = [len(l) for l in lists]
        res[cls] = {'counts': counts, 'mean': statistics.mean(counts), 'lists': lists}
        json.dump(res, open(args.out, 'w'), indent=1, ensure_ascii=False)
        print(f'[steps] {i + 1}/{len(ens)} {cls}: counts={counts} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    means = [v['mean'] for k, v in res.items() if not k.startswith('_')]
    allc = [c for k, v in res.items() if not k.startswith('_') for c in v['counts']]
    m, sd = statistics.mean(means), statistics.pstdev(means)
    res['_summary'] = {'classes': len(means), 'per_class_mean': {'mean': m, 'sd': sd, 'min': min(means), 'max': max(means)},
                       'all_samples': {'mean': statistics.mean(allc), 'sd': statistics.pstdev(allc)},
                       'K': {'mean': round(m), 'mean+sd': round(m + sd), 'mean+2sd': round(m + 2 * sd)}}
    json.dump(res, open(args.out, 'w'), indent=1, ensure_ascii=False)
    print('summary:', json.dumps(res['_summary']), '->', args.out, flush=True)


if __name__ == '__main__':
    main()
