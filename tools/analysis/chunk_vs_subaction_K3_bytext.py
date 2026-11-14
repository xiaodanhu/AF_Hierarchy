#!/usr/bin/env python
"""Equal split with the deployed K (video_subtree_K = 3), chunk k carrying the k-th
sub-action sentence of the dive's text tree, compared with the annotated sub-action of the SAME NAME.

The FineDiving text tree of a dive (configs/finediving/fd_dive_verbalizer.json, text_nested) has the
dive's sub-actions as ordered children (3, 4 or 5 of them, e.g. back takeoff / 2.5 somersaults pike /
entry into the water). The video branch splits every candidate segment into K = 3 equal chunks.
For every annotated dive (train + test) and every chunk k < K:
  * the chunk's text sub-action = the k-th child of the dive's tree,
  * its ground truth = the annotated sub-action of that name in the same dive (j-th occurrence of the
    name among the children <-> j-th occurrence among the annotations, for the dives that repeat a
    somersault around a twist),
  * tIoU(chunk k, that annotated span).
Children beyond K (4th and 5th sub-actions of the twisting dives) have no chunk and are counted as
uncovered. Dives whose annotated names do not contain the child's name (label errors) are skipped per pair.
Reports mean / median tIoU, % >= 0.5, by number of tree children, for train+test and for test only.
Annotation ids are the sub-action verbalizer's raw_id (see memory finediving-subaction-raw-ids)."""
import json, re, statistics, collections
A6 = '/data3/xiaodan8/actionformer6'; FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
K = 3
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))
dv = json.load(open(f'{A6}/configs/finediving/fd_dive_verbalizer.json'))['actions']
sv = json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))['actions']
raw2name = {v['raw_id']: v['name'] for v in sv.values()}
children = {v['code']: list(v['member_names']) for v in dv.values()}

def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0

dives = []
for split in ('train', 'test'):
    for l in open(f'{FD}/finediving_{split}_label.txt'):
        if not l.strip(): continue
        for x in json.loads(l)['raw_value']:
            s, e = map(sec, x['span'])
            gt = [(raw2name.get(q['action_id'], q['action_id']), *map(sec, q['span'])) for q in x['actions']]
            if e > s and all(b > a for _, a, b in gt): dives.append((x['activity_id'], s, e, gt, split))

def run(sel):
    ious, by_m, unc, skipped, n_d = [], collections.defaultdict(list), 0, 0, 0
    pos = collections.defaultdict(list)
    for code, s, e, gt, split in sel:
        ch = children[code]; M = len(ch); L = e - s; n_d += 1
        unc += max(0, M - K)
        seen_name = collections.Counter()
        for k in range(min(K, M)):
            name = ch[k]; occ = seen_name[name]; seen_name[name] += 1
            cand = [g for g in gt if g[0] == name]
            if occ >= len(cand): skipped += 1; continue
            _, a, b = cand[occ]
            v = tiou((s + k * L / K, s + (k + 1) * L / K), (a, b))
            ious.append(v); by_m[M].append(v); pos[(k, M)].append(v)
    return ious, by_m, unc, skipped, n_d, pos

for label, sel in (('train+test', dives), ('test only', [d for d in dives if d[4] == 'test'])):
    ious, by_m, unc, skipped, n_d, pos = run(sel)
    print(f'\n=== {label}: {n_d} dives, K = {K} equal chunks, chunk k <-> k-th sub-action sentence of the tree <-> annotated sub-action of that name ===')
    print(f'  children per dive tree: {dict(sorted(collections.Counter(len(children[d[0]]) for d in sel).items()))}')
    print(f'  {len(ious)} (chunk, annotated sub-action) pairs, {unc} tree children beyond K without a chunk, {skipped} pairs skipped (name absent in the annotation)')
    print(f'  tIoU mean {statistics.mean(ious):.3f}  median {statistics.median(ious):.3f}  >=0.5: {100*sum(v>=0.5 for v in ious)/len(ious):.1f}%  >=0.3: {100*sum(v>=0.3 for v in ious)/len(ious):.1f}%')
    for M in sorted(by_m):
        v = by_m[M]; nd = sum(1 for d in sel if len(children[d[0]]) == M)
        print(f'    trees with {M} children ({nd} dives, {100*nd/n_d:.0f}%): mean tIoU {statistics.mean(v):.3f}, >=0.5 {100*sum(x>=0.5 for x in v)/len(v):.0f}%  (n={len(v)})')
        for k in range(min(K, M)):
            w = pos[(k, M)]
            print(f'      chunk {k+1} <-> child {k+1} ({"/".join(sorted({children[d[0]][k].split()[-1] if "takeoff" in children[d[0]][k] else children[d[0]][k].split()[-1] for d in sel if len(children[d[0]])==M}))}): mean {statistics.mean(w):.3f}, >=0.5 {100*sum(x>=0.5 for x in w)/len(w):.0f}%')
