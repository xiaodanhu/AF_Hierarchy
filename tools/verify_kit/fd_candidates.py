"""FIX: finediving_test_label.txt uses the verbalizer RAW ids (raw_id, 29 classes), not the renumbered id. Names are now mapped through raw_id.
Rank FineDiving test dives by how well the K equal chunks match the annotated
sub-actions (mean tIoU of chunk k vs annotated sub-action k, K = number of annotated sub-actions), so the
verification kit and the figure use dives whose annotation and equal split agree. Held-out codes
(split 1) first. Writes fd_candidates.json: {code: [top dives...]}."""
import json, re, statistics, sys
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))
A6 = '/data3/xiaodan8/actionformer6'; FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
verb = json.load(open(f'{A6}/configs/finediving/fd_dive_verbalizer.json'))
sub = {v['raw_id']: v['name'] for v in json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))['actions'].values()}   # annotation ids are raw ids
held = {verb['actions'][c]['code'] for c in json.load(open(f'{A6}/configs/finediving/fd_dive_attribute_table_s1.json'))['held_out_classes']}
def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else 0.0
dives = {}
for l in open(f'{FD}/finediving_test_label.txt'):
    if not l.strip(): continue
    r = json.loads(l)
    for x in r['raw_value']:
        s, e = map(sec, x['span']); subs = [(sub.get(q['action_id'], q['action_id']), *map(sec, q['span'])) for q in x['actions']]
        K = len(subs); L = e - s
        if K < 2 or L <= 0: continue
        sc = statistics.mean(tiou((s + k * L / K, s + (k + 1) * L / K), (a0, b0)) for k, (_, a0, b0) in enumerate(subs))
        dives.setdefault(x['activity_id'], []).append(dict(video=r['video'], s=s, e=e, dur=L, K=K, tiou=sc, subs=subs))
out = {}
for code, lst in dives.items():
    med = statistics.median(d['dur'] for d in lst)
    for d in lst: d['dur_dev'] = abs(d['dur'] - med) / med
    lst.sort(key=lambda d: (-round(d['tiou'], 2), d['dur_dev']))
    out[code] = lst[:4]
json.dump(out, open('fd_candidates.json', 'w'), indent=1)
rows = sorted(out.items(), key=lambda kv: (kv[0] not in held, kv[0]))
print(f"{'code':6} {'held':5} {'n':4} {'best tIoU':9} {'median tIoU':11} best dive")
for code, lst in rows:
    allt = [d['tiou'] for d in dives[code]]
    b = lst[0]
    print(f"{code:6} {'yes' if code in held else '':5} {len(allt):4} {b['tiou']:9.2f} {statistics.median(allt):11.2f} {b['video']} {b['s']:.1f}-{b['e']:.1f} ({b['dur']:.1f}s, K={b['K']}) " + '; '.join(f"{n} {a0:.1f}-{b0:.1f}" for n, a0, b0 in b['subs']))
