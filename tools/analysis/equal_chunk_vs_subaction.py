#!/usr/bin/env python
"""B2.1: how well do equal-length chunks match annotated sub-action spans? (FineDiving)

For every dive: split the dive span into N equal chunks and compare chunk i with the
i-th annotated sub-action span (tIoU). Two settings:
  * N = number of annotated sub-actions of that dive (the count the text tree has)
  * N = 3 fixed, as in the FineDiving config (video_subtree_K: 3)
Also reports how uneven the real sub-action durations are, by position and by type.
"""
import json, re, statistics, collections, sys

FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
verb = json.load(open(f'{FD}/verbalizer.json'))['actions']
name = {k: v['name'] for k, v in verb.items()}
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))

def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0

dives = []
for split in ('train', 'test'):
    for l in open(f'{FD}/finediving_{split}_label.txt'):
        if not l.strip(): continue
        for x in json.loads(l)['raw_value']:
            s, e = map(sec, x['span'])
            subs = [(a['action_id'], *map(sec, a['span'])) for a in x['actions']]
            if e > s and all(b > a for _, a, b in subs):
                dives.append((x['activity_id'], s, e, subs, split))
print(f'{len(dives)} dives (train+test); sub-actions per dive:',
      dict(collections.Counter(len(d[3]) for d in dives)))

def run(fixedK=None):
    ious, by_pos, n_uncovered = [], collections.defaultdict(list), 0
    for code, s, e, subs, _ in dives:
        N = fixedK or len(subs); L = e - s
        for i, (aid, a, b) in enumerate(subs):
            if i >= N: n_uncovered += 1; continue
            ch = (s + i * L / N, s + (i + 1) * L / N)
            v = tiou(ch, (a, b)); ious.append(v); by_pos[(i, len(subs))].append(v)
    return ious, by_pos, n_uncovered

for label, K in (('N = #sub-actions of the dive', None), ('N = 3 fixed (config video_subtree_K)', 3)):
    ious, by_pos, unc = run(K)
    print(f'\n=== equal split, {label} ===')
    print(f'  {len(ious)} (chunk, sub-action) pairs'
          + (f'; {unc} 4th sub-actions have no chunk under K=3' if unc else ''))
    print(f'  tIoU mean {statistics.mean(ious):.3f}  median {statistics.median(ious):.3f}  '
          f'>=0.5: {100*sum(v>=0.5 for v in ious)/len(ious):.1f}%   >=0.3: {100*sum(v>=0.3 for v in ious)/len(ious):.1f}%')
    for (i, n) in sorted(by_pos):
        v = by_pos[(i, n)]
        print(f'    position {i+1}/{n}: mean tIoU {statistics.mean(v):.3f}, >=0.5 {100*sum(x>=0.5 for x in v)/len(v):.0f}%  (n={len(v)})')

print('\n=== how uneven are the real sub-actions? fraction of the dive each occupies ===')
frac = collections.defaultdict(list)
for code, s, e, subs, _ in dives:
    for i, (aid, a, b) in enumerate(subs):
        frac[(i, len(subs))].append((b - a) / (e - s))
for (i, n) in sorted(frac):
    v = frac[(i, n)]
    print(f'  position {i+1}/{n}: mean {statistics.mean(v):.2f}  (equal split would give {1/n:.2f})  '
          f'min {min(v):.2f} max {max(v):.2f}')
bytype = collections.defaultdict(list)
for code, s, e, subs, _ in dives:
    for aid, a, b in subs: bytype[name.get(aid, aid)].append((b - a) / (e - s))
print('  by sub-action type (mean fraction of dive):')
for t, v in sorted(bytype.items(), key=lambda kv: -statistics.mean(kv[1]))[:8]:
    print(f'    {t:28s} {statistics.mean(v):.2f}  (n={len(v)})')
