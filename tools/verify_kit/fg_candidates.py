"""FineGym candidates: per class, --n RANDOM test instances (seed) with duration
within [0.5, 2] x the class median and from different videos; montage per class for an annotation
check (does the span cover the element?). Writes <out>/<cid>.png and <out>/candidates.json."""
import argparse, json, os, random, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_verify_clips_fg import find_video, sec, A6
from montage import frames_of, montage
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=f'{A6}/tools/verify_kit/fg_candidates'); ap.add_argument('--n', type=int, default=3)
ap.add_argument('--classes', nargs='*', default=None); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--fw', type=int, default=200)
a = ap.parse_args()
verb = json.load(open(f'{A6}/configs/finegym_zsl_verbalizer.json')); acts = verb['actions']
held = json.load(open(f'{A6}/configs/finegym_attribute_table_v2.json'))['held_out_classes']
trees = json.load(open(f'{A6}/configs/finegym_qwen_ensemble.json'))
inst = {}
for l in open('/data3/xiaodan8/FineGym/annotation/Dec16/gym99_val_label.txt'):
    if not l.strip(): continue
    r = json.loads(l)
    for x in r['raw_value']:
        for q in x['actions']:
            s, e = map(sec, q['span'])
            if e - s >= 0.5: inst.setdefault(q['action_id'], []).append(dict(clip=r['video'], s=s, e=e, dur=e - s))
classes = a.classes or held
os.makedirs(a.out, exist_ok=True); tmp = os.path.join(a.out, '_tmp'); os.makedirs(tmp, exist_ok=True)
chosen = {}
for cid in classes:
    lst = inst.get(cid, []); key = f"({acts[cid]['activity']}) {acts[cid]['name']}"
    if not lst: print('no instances', cid); continue
    med = statistics.median(d['dur'] for d in lst)
    pool = [d for d in lst if 0.5 * med <= d['dur'] <= 2 * med and find_video('/data3/xiaodan8/FineGym', d['clip'].split('_E_')[0])]
    rng = random.Random(f'{a.seed}-{cid}'); rng.shuffle(pool)
    picks, vids = [], set()
    for d in pool:
        yid = d['clip'].split('_E_')[0]
        if yid in vids: continue
        picks.append(d); vids.add(yid)
        if len(picks) == a.n: break
    chosen[cid] = picks
    rows = [(f"[{j}] {d['clip']}  {d['s']:.1f}-{d['e']:.1f}s ({d['dur']:.1f}s, class median {med:.1f}s)",
             frames_of(find_video('/data3/xiaodan8/FineGym', d['clip'].split('_E_')[0]), d['s'], d['e'], 8, a.fw, tmp, f'{cid}_{j}')) for j, d in enumerate(picks)]
    montage(rows, f"{cid} {'HELD-OUT' if cid in held else 'seen'}  {key}",
            'sentences 1-5: ' + ' | '.join(f"{k+1}. {s[:55]}" for k, s in enumerate(trees.get(key, [''] * 11)[1:6])), os.path.join(a.out, f'{cid}.png'), a.fw)
    print(cid, len(picks), flush=True)
json.dump(chosen, open(os.path.join(a.out, 'candidates.json'), 'w'), indent=1)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
os.rmdir(tmp); print('done')
