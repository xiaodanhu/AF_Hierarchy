"""FIX: finediving_test_label.txt uses the verbalizer RAW ids (raw_id, 29 classes), not the renumbered id. Names are now mapped through raw_id. The pool filter now requires the annotated sub-action NAMES to equal the dive definition (stricter than the family sequence) and the last sub-action to be an entry: the dataset labels the last phase of the armstand forward dives 612b, 614b, 6142d as '3.5 twists' (raw id c26), which the definition trees inherited, so those three seen codes are out.
Random selection of FineDiving test dives with the badly annotated ones removed.
Pool per dive code: (a) every annotated sub-action has a name in the sub-action verbalizer and the
sequence of families (takeoff, somersault, twist, entry) equals the dive's sub-actions in the
verbalizer (unknown ids, label and order errors out), (b) duration within 30% of the code's median (truncated or
over-long spans out), (c) mean tIoU of the K equal chunks against the K annotated sub-actions >= 0.5
(the alignment statistic, boundary errors out). One dive per code is drawn at random (seed)
from the pool; codes with an empty pool are dropped (listed under 'dropped'). Order of the kit:
held-out codes (split 1) first, then seen, each group by alignment score descending.
Writes fd_selection.json (ordered list) and prints the table."""
import argparse, json, random, re, statistics
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))
A6 = '/data3/xiaodan8/actionformer6'; FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--min_tiou', type=float, default=0.5)
ap.add_argument('--out', default=f'{A6}/tools/verify_kit/fd_selection.json'); a = ap.parse_args()
verb = json.load(open(f'{A6}/configs/finediving/fd_dive_verbalizer.json'))
subv = json.load(open(f'{A6}/configs/finediving/fd_sub_verbalizer.json'))
sub = {v['raw_id']: v['name'] for v in subv['actions'].values()}   # annotation ids are raw ids
fam_of_name = {v['name']: subv['phrases'][v['phrase']]['name'].split('_')[0] for v in subv['actions'].values()}   # takeoff / somersault / twist / entry
fam_of_id = {v['raw_id']: subv['phrases'][v['phrase']]['name'].split('_')[0] for v in subv['actions'].values()}
held = {verb['actions'][c]['code'] for c in json.load(open(f'{A6}/configs/finediving/fd_dive_attribute_table_s1.json'))['held_out_classes']}
members = {v['code']: list(v['member_names']) for v in verb['actions'].values()}   # the dive's sub-action names, in order
by_code = {v['code']: f"({v['activity']}) {v['name']}" for v in verb['actions'].values()}
def tiou(x, y):
    i = max(0.0, min(x[1], y[1]) - max(x[0], y[0])); u = max(x[1], y[1]) - min(x[0], y[0]); return i / u if u > 0 else 0.0
dives = {}
for l in open(f'{FD}/finediving_test_label.txt'):
    if not l.strip(): continue
    r = json.loads(l)
    for x in r['raw_value']:
        s, e = map(sec, x['span']); subs = [(sub.get(q['action_id'], q['action_id']), *map(sec, q['span'])) for q in x['actions']]
        K = len(subs); L = e - s
        if K < 2 or L <= 0: continue
        sc = statistics.mean(tiou((s + k * L / K, s + (k + 1) * L / K), (a0, b0)) for k, (_, a0, b0) in enumerate(subs))
        dives.setdefault(x['activity_id'], []).append(dict(code=x['activity_id'], key=by_code[x['activity_id']], video=r['video'], s=s, e=e, dur=L, K=K, tiou=sc,
                                                           subs=subs,
                                                           order_ok=[n for n, _, _ in subs] == members.get(x['activity_id']) and fam_of_name.get(subs[-1][0]) == 'entry'))   # order_ok: annotated names equal the definition's sub-actions and the dive ends with an entry
sel = []
for code, lst in dives.items():
    med = statistics.median(d['dur'] for d in lst)
    pool = [d for d in lst if d['order_ok'] and abs(d['dur'] - med) <= 0.3 * med and d['tiou'] >= a.min_tiou]
    rng = random.Random(f'{a.seed}-{code}')
    if pool: d = rng.choice(pool); d['flag'] = ''
    else: d = max(lst, key=lambda d: (d['order_ok'], d['tiou'])); d['flag'] = 'DROPPED: no dive of this code passes the filter'
    d['pool'] = len(pool); d['n'] = len(lst); d['held'] = code in held; sel.append(d)
sel.sort(key=lambda d: (not d['held'], -d['tiou']))
json.dump({'selected': [d for d in sel if not d['flag']], 'dropped': [d for d in sel if d['flag']]}, open(a.out, 'w'), indent=1)
print(f"{'#':3} {'code':6} {'held':5} {'pool/n':7} {'tIoU':5} {'dur':5} dive")
for i, d in enumerate(sel, 1):
    print(f"{i:3} {d['code']:6} {'yes' if d['held'] else '':5} {d['pool']:3}/{d['n']:<3} {d['tiou']:5.2f} {d['dur']:5.1f} {d['video']} {d['s']:.1f}-{d['e']:.1f}  {d['flag']}")
