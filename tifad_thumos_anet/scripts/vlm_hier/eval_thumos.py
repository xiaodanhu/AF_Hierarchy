#!/usr/bin/env python
"""Score C3 THUMOS predictions with the detector's evaluator (run in the `parsing2` env).

Reads predictions.jsonl from infer.py (window-relative instances with token-log-prob
confidence), maps names back to the split's unseen classes, converts to absolute
seconds, per-class NMS per video (same greedy NMS as C1), and evaluates unseen mAP
@[0.3:0.7] with the split file, exactly like vlm_localize_baseline.evaluate.

  python scripts/vlm_hier/eval_thumos.py --n 0 --pred logs/vlm_hier/thumos_s0_pred.jsonl [--src greedy|all]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vlm_localize_baseline import evaluate, nms, readable, SPLIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0); ap.add_argument('--pred', required=True)
    ap.add_argument('--src', default='all', help="'greedy' = greedy decode only; 'all' = greedy + samples")
    ap.add_argument('--nms', type=float, default=0.5)
    a = ap.parse_args()
    unseen = [l.strip() for l in open(SPLIT.format(n=a.n)) if l.strip()]
    label_id = {c: i for i, c in enumerate(unseen)}
    name2cls = {readable(c): c for c in unseen}; name2cls.update({c.lower(): c for c in unseen})
    rows = [json.loads(l) for l in open(a.pred)]
    per_video, n_inst, n_unk = {}, 0, 0
    for r in rows:
        for p in r['preds']:
            if a.src == 'greedy' and p.get('src') != 'greedy':
                continue
            cls = name2cls.get(p['action']) or name2cls.get(p['action'].replace(' ', ''))
            n_inst += 1
            if cls is None:
                n_unk += 1; continue
            per_video.setdefault(r['video'], []).append((cls, r['win_start'] + p['start'], r['win_start'] + p['end'], float(p['conf'])))
    preds = []
    for vid, dets in per_video.items():
        preds += [(vid, s, e, label_id[c], conf) for c, s, e, conf in nms(dets, a.nms)]
    print(f'{len(rows)} windows, {n_inst} instances ({n_unk} with names not in the list), {len(preds)} after NMS over {len(per_video)} videos')
    if not preds:
        print('no predictions'); return
    # ---- children (cuts) vs the equal split, greedy predictions of listed classes only
    import numpy as np
    from common import valid_cuts, equal_cuts, cuts_to_spans
    K = 10; n_c = n_valid = n_exact = 0; pos_t = {k: [] for k in range(K)}; drift = []
    for r in rows:
        for p in r['preds']:
            if p.get('src') != 'greedy' or not (name2cls.get(p['action']) or name2cls.get(p['action'].replace(' ', ''))):
                continue
            n_c += 1
            c = p.get('cuts')
            if not valid_cuts(c, p['start'], p['end'], K, tol=0.5):
                continue
            n_valid += 1
            eq = equal_cuts(p['start'], p['end'], K)
            if all(abs(x - y) <= 0.05 for x, y in zip(c, eq)):
                n_exact += 1
            for k, ((a0, b0), (a1, b1)) in enumerate(zip(cuts_to_spans(c), cuts_to_spans(eq))):
                inter = max(0.0, min(b0, b1) - max(a0, a1)); union = max(b0, b1) - min(a0, a1)
                pos_t[k].append(inter / union if union > 0 else 0.0)
            drift.append(np.mean([abs(x - y) for x, y in zip(c, eq)]) / max(1e-6, p['end'] - p['start']))
    if n_c:
        allt = [t for v in pos_t.values() for t in v]
        print(f'CHILDREN (greedy, {n_c} instances): valid K+1 cuts {100*n_valid/n_c:.1f} %, exactly the equal split {100*n_exact/max(1,n_valid):.1f} %, '
              f'step-vs-equal-chunk tIoU mean {np.mean(allt) if allt else 0:.3f}, >=0.5 {100*np.mean([t>=0.5 for t in allt]) if allt else 0:.1f} %, '
              f'mean cut drift {100*np.mean(drift) if drift else 0:.1f} % of the instance length')
        print('  per position tIoU: ' + ' '.join(f'{k+1}:{np.mean(v):.2f}' for k, v in pos_t.items() if v))
    mAP, per = evaluate(preds, a.n)
    out = a.pred.replace('.jsonl', f'_eval_{a.src}.json')
    json.dump({'mAP': mAP, 'per_tiou': per, 'n_pred': len(preds), 'src': a.src, 'nms': a.nms, 'children': {'n': n_c, 'valid': n_valid, 'exact_equal': n_exact, 'tiou_mean': float(np.mean(allt)) if n_c and allt else None, 'drift': float(np.mean(drift)) if drift else None}}, open(out, 'w'), indent=1)
    print(f'RESULT C3 THUMOS split {a.n} [{a.src}]: unseen mAP@[0.3:0.7] = {100*mAP:.2f}  per tIoU {[round(100*x,1) for x in per]}  -> {out}')


if __name__ == '__main__':
    main()
