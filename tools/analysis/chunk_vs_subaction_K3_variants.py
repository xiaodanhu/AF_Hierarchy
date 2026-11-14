#!/usr/bin/env python
"""FineDiving: K = 3 equal chunks per dive against the annotated sub-actions, every
annotated sub-action counted (also the 4th and 5th of the twisting dives). Variants of the chunk <-> sub-action match:
  V1 name   : chunk k <-> k-th child of the dive's text tree (fd_dive_verbalizer member_names) <-> annotated
              sub-action of that name. Children beyond K have no chunk and are LEFT OUT (chunk_vs_subaction_K3_bytext.py).
  V2 name0  : as V1, but every annotated sub-action without a chunk counts as tIoU 0.
  V3 3-part : every dive read as takeoff / flight / entry: chunk 1 <-> annotated takeoff, chunk 2 <-> the span from
              the end of the takeoff to the start of the entry (all somersaults and twists together), chunk 3 <->
              annotated entry. Every annotated sub-action is covered by exactly one chunk.
  V4 cover  : every annotated sub-action <-> the chunk that covers its midpoint (per sub-action tIoU with that chunk,
              every sub-action counted, no text involved).
  V5 best   : every annotated sub-action <-> the chunk with the largest tIoU (upper bound, no text involved).
Mean / median tIoU and % >= 0.5 over the pairs, by number of annotated sub-actions; train+test and test only.
Annotation ids are the sub-action verbalizer raw_id."""
import json, re, statistics, collections
A6 = '/data3/xiaodan8/actionformer6'; FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
K = 3
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))
dv = json.load(open(f'{A6}/configs/finediving/fd_dive_verbalizer.json'))['actions']
sv = json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))['actions']
raw2name = {v['raw_id']: v['name'] for v in sv.values()}
fam = {v['name']: sv[v['id']]['phrase'] for v in sv.values()}            # p00 takeoff .. p04 entry
children = {v['code']: list(v['member_names']) for v in dv.values()}
def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else 0.0
dives = []
for split in ('train', 'test'):
    for l in open(f'{FD}/finediving_{split}_label.txt'):
        if not l.strip(): continue
        for x in json.loads(l)['raw_value']:
            s, e = map(sec, x['span'])
            gt = [(raw2name.get(q['action_id'], q['action_id']), *map(sec, q['span'])) for q in x['actions']]
            if e > s and all(b > a for _, a, b in gt): dives.append((x['activity_id'], s, e, gt, split))
def chunks(s, e): L = e - s; return [(s + k * L / K, s + (k + 1) * L / K) for k in range(K)]
def variants(code, s, e, gt):
    ch = chunks(s, e); out = {}
    # V1 / V2
    names = children[code]; seen = collections.Counter(); v1 = []; matched = set()
    for k in range(min(K, len(names))):
        n = names[k]; occ = seen[n]; seen[n] += 1
        cand = [i for i, g in enumerate(gt) if g[0] == n]
        if occ >= len(cand): continue
        i = cand[occ]; matched.add(i); v1.append(tiou(ch[k], gt[i][1:]))
    out['V1 name (children beyond K left out)'] = v1
    out['V2 name (sub-actions without a chunk count 0)'] = v1 + [0.0] * (len(gt) - len(matched))
    # V3: takeoff / flight / entry
    if len(gt) >= 3:
        mid = (gt[0][2], gt[-1][1])
        out['V3 takeoff / flight / entry'] = [tiou(ch[0], gt[0][1:]), tiou(ch[1], mid), tiou(ch[2], gt[-1][1:])]
    # V4 / V5
    out['V4 chunk covering the sub-action midpoint'] = [tiou(ch[min(K - 1, int((((a + b) / 2) - s) / (e - s) * K))], (a, b)) for _, a, b in gt]
    out['V5 best chunk per sub-action'] = [max(tiou(c, (a, b)) for c in ch) for _, a, b in gt]
    return out
for label, sel in (('train+test', dives), ('test only', [d for d in dives if d[4] == 'test'])):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for code, s, e, gt, _ in sel:
        for name, vals in variants(code, s, e, gt).items():
            agg[name]['all'] += vals; agg[name][len(gt)] += vals
    print(f'\n=== {label}: {len(sel)} dives, annotated sub-actions per dive {dict(sorted(collections.Counter(len(d[3]) for d in sel).items()))}, K = {K} ===')
    for name, d in agg.items():
        v = d['all']
        line = f"  {name:48s} all: mean {statistics.mean(v):.3f} median {statistics.median(v):.3f} >=0.5 {100*sum(x>=0.5 for x in v)/len(v):.0f}% (n={len(v)})"
        for M in (3, 4, 5):
            w = d[M]
            if w: line += f" | {M} sub-actions: {statistics.mean(w):.3f}"
        print(line)
