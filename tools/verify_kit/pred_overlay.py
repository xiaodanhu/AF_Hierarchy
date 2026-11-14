"""Model predictions for the verification kits: the video-level detections
dumped by an eval-only run (output/<dataset>/pred_<ts>_video_level.json, {clip: [{t-start, t-end,
label, score}]}) matched to a ground-truth instance, so a rater sees ground truth and prediction side by side."""
import json


def load_pred(path):
    return json.load(open(path)) if path else None


def top_preds(pred, key, s, e, n=3):
    """Predictions of clip `key` that overlap [s, e], best score first."""
    out = []
    for x in (pred or {}).get(key, []):
        ps, pe, sc, lab = float(x['t-start']), float(x['t-end']), float(x['score']), int(x['label'])
        inter = max(0.0, min(e, pe) - max(s, ps))
        if inter <= 0:
            continue
        out.append(dict(label=lab, s=ps, e=pe, score=sc, iou=inter / (max(e, pe) - min(s, ps))))
    out.sort(key=lambda d: -d['score'])
    return out[:n]


def fmt_pred(p, name_of, rank=1):
    return f"PRED{rank}: {name_of(p['label'])}  [{p['s']:.1f}-{p['e']:.1f}s]  score {p['score']:.2f}  IoU {p['iou']:.2f}"
