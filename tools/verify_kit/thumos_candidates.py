"""THUMOS14 candidates: per class, --n RANDOM test instances (seed) with duration
in [--min_dur, --max_dur] and from different videos; montage per class for an annotation check."""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from montage import frames_of, montage
ap = argparse.ArgumentParser()
ap.add_argument('--ann', default='/data3/xiaodan8/thumos/annotations/thumos14.json'); ap.add_argument('--videos', default='/data3/xiaodan8/thumos/video')
ap.add_argument('--trees', default='/data3/xiaodan8/tifad/configs/thumos14_qwen_ensemble.json')
ap.add_argument('--out', default='/data3/xiaodan8/actionformer6/tools/verify_kit/thumos_candidates')
ap.add_argument('--n', type=int, default=3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--fw', type=int, default=200)
ap.add_argument('--min_dur', type=float, default=3.0); ap.add_argument('--max_dur', type=float, default=25.0)
a = ap.parse_args()
j = json.load(open(a.ann)); db = j.get('database', j); trees = json.load(open(a.trees))
inst = {}
for vid, v in db.items():
    if not vid.startswith('video_test'): continue
    for x in v['annotations']:
        s, e = x['segment']
        if a.min_dur <= e - s <= a.max_dur: inst.setdefault(x['label'], []).append(dict(clip=vid, s=s, e=e, dur=e - s))
os.makedirs(a.out, exist_ok=True); tmp = os.path.join(a.out, '_tmp'); os.makedirs(tmp, exist_ok=True)
chosen = {}
for cls in sorted(trees):
    pool = inst.get(cls, []); rng = random.Random(f'{a.seed}-{cls}'); rng.shuffle(pool)
    picks, vids = [], set()
    for d in pool:
        if d['clip'] in vids: continue
        picks.append(d); vids.add(d['clip'])
        if len(picks) == a.n: break
    chosen[cls] = picks
    rows = [(f"[{k}] {d['clip']}  {d['s']:.1f}-{d['e']:.1f}s ({d['dur']:.1f}s)",
             frames_of(os.path.join(a.videos, d['clip'] + '.mp4'), d['s'], d['e'], 8, a.fw, tmp, f'{cls}_{k}')) for k, d in enumerate(picks)]
    montage(rows, cls, 'sentences 1-5: ' + ' | '.join(f"{k+1}. {s[:55]}" for k, s in enumerate(trees[cls][1:6])), os.path.join(a.out, f'{cls}.png'), a.fw)
    print(cls, len(picks), flush=True)
json.dump(chosen, open(os.path.join(a.out, 'candidates.json'), 'w'), indent=1)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
os.rmdir(tmp); print('done')
