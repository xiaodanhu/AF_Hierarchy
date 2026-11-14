#!/usr/bin/env python
"""Where does the tuned VLM (C3) lose on THUMOS: localisation or recognition?  Greedy predictions, split n.
  recall of unseen GT instances by ANY listed-class prediction at tIoU >= 0.5 (class-agnostic), correct-class share
  among those, top confusions, precision of predictions against unseen GT, per-class counts.
  python scripts/vlm_hier/diag_thumos.py --n 0 --pred logs/vlm_hier/thumos_s0_pred.jsonl
"""
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_localize_baseline import readable, SPLIT, ANN, nms


def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=0); ap.add_argument('--pred', required=True); ap.add_argument('--src', default='greedy')
    a = ap.parse_args()
    unseen = [l.strip() for l in open(SPLIT.format(n=a.n)) if l.strip()]
    name2cls = {readable(c): c for c in unseen}; name2cls.update({c.lower(): c for c in unseen})
    db = json.load(open(ANN)); db = db.get('database', db)
    gt = collections.defaultdict(list)                        # video -> [(cls, s, e)] unseen only, test subset
    for vid, v in db.items():
        if v.get('subset', 'test').lower() not in ('test', 'validation') or not vid.startswith('video_test'):
            continue
        for x in v.get('annotations', []):
            if x['label'] in unseen:
                gt[vid].append((x['label'], float(x['segment'][0]), float(x['segment'][1])))
    rows = [json.loads(l) for l in open(a.pred)]
    per_video = collections.defaultdict(list); n_raw = 0
    for r in rows:
        for p in r['preds']:
            if a.src == 'greedy' and p.get('src') != 'greedy':
                continue
            cls = name2cls.get(p['action']) or name2cls.get(p['action'].replace(' ', ''))
            if cls is None:
                continue
            n_raw += 1; per_video[r['video']].append((cls, r['win_start'] + p['start'], r['win_start'] + p['end'], float(p['conf'])))
    preds = {vid: nms(d, 0.5) for vid, d in per_video.items()}
    n_gt = sum(len(v) for v in gt.values()); n_pred = sum(len(v) for v in preds.values())
    loc_hit = cls_hit = 0; conf = collections.Counter(); per_cls = collections.defaultdict(lambda: [0, 0, 0])   # gt, localised, correct
    for vid, g in gt.items():
        P = preds.get(vid, [])
        for c, s, e in g:
            per_cls[c][0] += 1
            best = max(P, key=lambda p: tiou((s, e), (p[1], p[2])), default=None)
            if best is not None and tiou((s, e), (best[1], best[2])) >= 0.5:
                loc_hit += 1; per_cls[c][1] += 1
                if best[0] == c:
                    cls_hit += 1; per_cls[c][2] += 1
                else:
                    conf[(c, best[0])] += 1
    fp = 0
    for vid, P in preds.items():
        for p in P:
            if not any(tiou((s, e), (p[1], p[2])) >= 0.5 for _, s, e in gt.get(vid, [])):
                fp += 1
    print(f'DIAG C3 THUMOS split {a.n} [{a.src}]: unseen GT instances {n_gt}; predictions (listed classes, after NMS) {n_pred} (raw {n_raw})')
    print(f'  class-agnostic recall @0.5: {100*loc_hit/n_gt:.1f} %   correct class among localised: {100*cls_hit/max(1,loc_hit):.1f} %   => recall with correct class {100*cls_hit/n_gt:.1f} %')
    print(f'  predictions not overlapping any unseen GT @0.5: {100*fp/max(1,n_pred):.1f} % of predictions')
    print('  per class (gt / localised / correct): ' + ', '.join(f'{c} {v[0]}/{v[1]}/{v[2]}' for c, v in sorted(per_cls.items())))
    print('  top confusions (gt -> pred): ' + ', '.join(f'{g}->{p} {n}' for (g, p), n in conf.most_common(6)))


if __name__ == '__main__':
    main()
