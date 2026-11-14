"""PHASE PROBE on the RAW frozen I3D clip features (THUMOS14 75/25).

Question: do the raw two-stream I3D features the detector consumes carry
MORE information about the temporal PHASE inside an action than the
detector's cls-trunk features (scripts/phase_probe.py)?  The classification
loss pushes every location inside an action toward one class embedding, so
the trunk may erase within-action variation.

Same rows, same probe, same seeds as scripts/phase_probe.py, restricted to
LEVEL-0 locations under the 'in' rule (every level-0 location inside a GT
segment, shortest on ties):
  * rows come from the cached dumps data/phase_pseudo/probe_dump/split{n}_
    {train,test}.npz -> identical train / test videos, locations, labels.
  * the TRAIN dump was harvested on the loader's random crop (truncate_feats,
    crop_ratio [0.9, 1.0]); the crop offset `st` is not stored, so it is
    recovered per video from the rows themselves: a segment's cropped span
    (s', e') follows from two of its rows (t, t_norm), and st is the mode of
    raw_s - s' over the raw GT segments of equal length; every dump segment
    must then be explained (exact or window-clipped) by a raw segment.
    Level-0 point t is the integer feature index (no offset), so the raw
    feature of a train row is feats[t + st], of a test row feats[t].
  * raw feature = /data3/xiaodan8/thumos/i3d_features/<vid>.npy (T, 2048),
    feature stride 4 frames, 16-frame clips; dims [:1024] = RGB stream,
    [1024:] = flow stream (verified by the per-stream statistics in the
    output meta: the two halves have different activation scales).
  * feature variants: trunk level-0 (the dump's 256-d cls-trunk feature, the
    reference recomputed on exactly these rows), raw RGB, raw flow, raw both,
    both whitened (train-row mean / per-dim std), both minus the top-5
    principal components of the (centred) training features.
  * two targets: (a) the text-aligned pseudo phase (0..9) of
    data/phase_pseudo/thumos14_75_25_split{n}.json (seen) / *_unseen.json
    (unseen classes, evaluation only); (b) the position decile
    k = floor(10 * (t - s) / (e - s)) which needs no pseudo labels.
  * head: Linear(D,D)-ReLU-Linear(D,10), Adam 1e-3, 30 epochs, batch 4096,
    seeds 0 1 2 (phase_probe.train_head).  Metrics: acc10 / acc3 (seen and,
    where labels exist, unseen), per-segment Spearman of the argmax and the
    expected phase vs normalized time (phase_probe.spearman_metrics), and
    the segment-level boundary error of the left-to-right Viterbi decode of
    scripts/phase_timespans.py on segments with >= 10 level-0 locations,
    measured against the pseudo timespans AND against equal-width bins.

Usage (CPU or any GPU):
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=3 \
      python scripts/phase_probe_raw.py --splits 0 1 2
Output: data/phase_pseudo/raw_probe_results.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import phase_probe as pp                       # sets the reference envs
import phase_timespans as pt
from libs.core import load_config
from libs.datasets import make_dataset

N_PHASE = pp.N_PHASE
VARIANTS = ('trunk_l0', 'rgb', 'flow', 'both', 'both_whiten', 'both_abtt5')
TARGETS = ('pseudo_phase', 'decile')


# ------------------------------------------------------------- raw features
def load_raw(feat_folder, vid, cache):
    if vid not in cache:
        cache[vid] = np.load(os.path.join(feat_folder, vid + '.npy')).astype(np.float32)
    return cache[vid]


def explain(spans, raw_segs, st, tol):
    """How many dump spans (cropped coordinates) a crop offset st explains:
    exact = raw segment - st; clipped = the span sits inside a raw segment
    - st with its start at 0 (left clip) and/or its end at the (unknown)
    window end (right clip; all right-clipped ends must coincide)."""
    n_exact = n_clip = 0
    right_ends = []
    for s, e in spans:
        ds_ = np.abs(raw_segs[:, 0] - st - s)
        de_ = np.abs(raw_segs[:, 1] - st - e)
        if ((ds_ < tol) & (de_ < tol)).any():
            n_exact += 1
            continue
        left_ok = (ds_ < tol) | ((raw_segs[:, 0] - st < tol) & (abs(s) < tol))
        right_ok = (de_ < tol) | (raw_segs[:, 1] - st > e - tol)
        if (left_ok & right_ok).any():
            n_clip += 1
            if not (de_ < tol).any():
                right_ends.append(e)
    if right_ends and np.ptp(right_ends) > tol:
        return n_exact, 0
    return n_exact, n_clip


def recover_offset(d, vi, raw_segs, T_raw, tol=0.1):
    """Crop offset `st` of train video vi from its dump rows (see module
    doc).  Candidates: raw start - span start and raw end - span end over
    all pairs (fallback: every feasible st when a span stays unexplained,
    e.g. one action clipped on both sides); the winner explains the most
    spans (exact first; smallest st on ties -- a both-sides-clipped whole-
    video action leaves st ambiguous by a frame or two).  Returns
    (st, n_exact, n_clipped, n_unexplained, n_spans)."""
    m = (d['video'] == vi) & (d['level'] == 0) & (d['in_seg'] >= 0)
    spans = []
    for j in np.unique(d['in_seg'][m]):
        mm = m & (d['in_seg'] == j)
        t = d['t'][mm].astype(np.float64)
        tn = d['in_tnorm'][mm].astype(np.float64)
        i1, i2 = np.argmin(t), np.argmax(t)
        if tn[i2] - tn[i1] < 1e-6:
            continue
        L = (t[i2] - t[i1]) / (tn[i2] - tn[i1])
        spans.append((t[i1] - tn[i1] * L, t[i1] - tn[i1] * L + L))
    cands = set()
    for s, e in spans:
        for k in range(len(raw_segs)):
            cands.add(int(round(raw_segs[k, 0] - s)))
            cands.add(int(round(raw_segs[k, 1] - e)))
    max_t = float(d['t'][m].max()) if m.any() else 0.0
    max_e = max([e for _, e in spans] + [max_t])
    feasible = lambda st: 0 <= st and st + max_e <= T_raw + tol   # window inside video

    def search(sts):
        best = None
        for st in sts:
            if not feasible(st):
                continue
            ne, nc = explain(spans, raw_segs, st, tol)
            score = (ne + nc, ne)
            if best is None or score > best[0]:
                best = (score, st, ne, nc)
        return best
    best = search(sorted(cands))
    if best is None or best[0][0] < len(spans):
        b2 = search(range(0, int(T_raw - max_e) + 2))
        if b2 is not None and (best is None or b2[0] > best[0]):
            best = b2
    if best is None:
        return None, 0, 0, len(spans), len(spans)
    (_, st, ne, nc) = best
    return st, ne, nc, len(spans) - ne - nc, len(spans)


def gather_train(cfg, n, d, feat_folder, cache):
    """Level-0 'in' rows of the train dump with a pseudo phase -> raw feats."""
    ds = make_dataset(cfg['dataset_name'], n, True, cfg['train_split'], **cfg['dataset'])
    rows = np.flatnonzero((d['level'] == 0) & (d['in_phase'] >= 0))
    X = np.zeros((len(rows), cfg['dataset']['input_dim']), np.float32)
    stats = {'videos': 0, 'segs': 0, 'exact': 0, 'clipped': 0, 'unexplained': 0,
             'no_offset_videos': []}
    keep = np.ones(len(rows), bool)
    for vi in np.unique(d['video'][rows]):
        item = ds.data_list[vi]
        raw_segs = (item['segments'] * item['fps'] - 0.5 * cfg['dataset']['num_frames']) \
            / cfg['dataset']['feat_stride']
        feats = load_raw(feat_folder, item['id'], cache)
        st, ne, nc, nb, ns = recover_offset(d, vi, raw_segs, feats.shape[0])
        sel = np.flatnonzero(d['video'][rows] == vi)
        if st is None:
            stats['no_offset_videos'].append(item['id'])
            keep[sel] = False
            continue
        stats['videos'] += 1
        stats['segs'] += ns
        stats['exact'] += ne
        stats['clipped'] += nc
        stats['unexplained'] += nb
        idx = d['t'][rows[sel]].astype(np.int64) + st
        assert idx.min() >= 0 and idx.max() < feats.shape[0], (item['id'], st, idx.max(), feats.shape)
        X[sel] = feats[idx]
    rows = rows[keep]
    X = X[keep]
    assert stats['unexplained'] == 0, stats
    if stats['no_offset_videos']:
        print(f"  [gather] WARNING no crop offset for {stats['no_offset_videos']}", flush=True)
    stats['no_offset_videos'] = len(stats['no_offset_videos'])
    return rows, X, stats


def gather_test(cfg, n, d, feat_folder, cache, gt):
    """Level-0 'in' rows of the test dump inside any GT segment -> raw feats,
    plus per-row phase labels from the gt cuts (seen: pseudo, unseen:
    pseudo_unseen; -1 where a segment has no cuts)."""
    vid_of = {vi: g['vid'] for (vi, _), g in gt.items()}   # test-dump video order
    rows = np.flatnonzero((d['level'] == 0) & (d['in_seg'] >= 0))
    X = np.zeros((len(rows), cfg['dataset']['input_dim']), np.float32)
    phase = np.full(len(rows), -1, np.int64)
    for vi in np.unique(d['video'][rows]):
        vid = vid_of[int(vi)]
        feats = load_raw(feat_folder, vid, cache)
        sel = np.flatnonzero(d['video'][rows] == vi)
        idx = d['t'][rows[sel]].astype(np.int64)
        assert idx.min() >= 0 and idx.max() < feats.shape[0], (vid, idx.max(), feats.shape)
        X[sel] = feats[idx]
        for j in np.unique(d['in_seg'][rows[sel]]):
            g = gt[(int(vi), int(j))]
            assert g['vid'] == vid
            if np.isnan(g['cuts']).any():
                continue
            ss = sel[d['in_seg'][rows[sel]] == j]
            phase[ss] = (d['t'][rows[ss]][:, None] >= g['cuts'][None]).sum(1)
    # seen rows: the dump's own pseudo phase must agree with the recomputed one
    seen = ~d['in_unseen'][rows]
    agree = (d['in_phase'][rows][seen] == phase[seen]) | (d['in_phase'][rows][seen] < 0)
    assert agree.all(), f"seen phase mismatch on {int((~agree).sum())} rows"
    return rows, X, phase


# -------------------------------------------------------------- transforms
def make_transforms(Xtr_both):
    mu = Xtr_both.mean(0, keepdims=True)
    sd = Xtr_both.std(0, keepdims=True) + 1e-6
    Xc = torch.from_numpy(Xtr_both - mu)
    # top-5 principal directions of the centred training features
    _, S, V = torch.linalg.svd(Xc, full_matrices=False)
    P = V[:5].numpy()                                        # (5, D)
    var_frac = float((S[:5] ** 2).sum() / (S ** 2).sum())
    tf = {
        'rgb': lambda X: X[:, :1024],
        'flow': lambda X: X[:, 1024:],
        'both': lambda X: X,
        'both_whiten': lambda X: (X - mu) / sd,
        'both_abtt5': lambda X: (X - mu) - ((X - mu) @ P.T) @ P,
    }
    return tf, {'top5_var_frac': var_frac,
                'top5_singular_values': S[:5].numpy().round(2).tolist()}


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def log_post(head, x, device):
    out = []
    for i in range(0, len(x), 8192):
        lg = head(torch.as_tensor(x[i:i + 8192], dtype=torch.float32, device=device))
        out.append(torch.log_softmax(lg, -1).cpu().numpy())
    return np.concatenate(out, 0)


def decode_segments(lp_rows, t_rows, seg_keys, gt, ref):
    """Viterbi-decode every decodable segment (dict key -> row positions
    sorted by t) and score the 9 boundaries against `ref` ('pseudo' = the
    segment's cuts, 'uniform' = equal-width bins)."""
    ms = []
    for key, pos in seg_keys.items():
        g = gt[key]
        L = g['e'] - g['s']
        target = g['cuts'] if ref == 'pseudo' else g['s'] + L * np.arange(1, 10) / 10.0
        st = pt.viterbi_lr(lp_rows[pos])
        b = pt.boundaries_from_states(t_rows[pos].astype(np.float64), st, g['s'], g['e'])
        ms.append(pt.boundary_metrics(b, target, g['s'], g['e'], g['sec_per_grid']))
    return pt.agg(ms)


def evaluate(head, Xte, te, gt, device):
    """Per-row + per-segment metrics of one head on the test rows."""
    lp = log_post(head, Xte, device)
    pa = lp.argmax(1)
    pe = (np.exp(lp) * np.arange(N_PHASE)[None]).sum(1)
    out = {}
    for sub in ('seen', 'unseen'):
        m = te['unseen'] == (sub == 'unseen')
        r = {'n': int(m.sum())}
        for tg in TARGETS:
            y = te[tg]
            ok = m & (y >= 0)
            if ok.sum():
                a = pp.acc_metrics(pa[ok], y[ok])
                r[f'acc10_{tg}'] = a['top1_10way']
                r[f'acc3_{tg}'] = a['top1_3way']
                r[f'majority10_{tg}'] = a['majority_10way']
                r[f'n_{tg}'] = int(ok.sum())
        r['spearman'] = pp.spearman_metrics(pa[m], pe[m], te['tnorm'][m], te['key'][m])
        dec = te['decodable'][sub]
        if dec:
            r['decode_vs_pseudo'] = decode_segments(lp, te['t'], dec, gt, 'pseudo')
            r['decode_vs_uniform'] = decode_segments(lp, te['t'], dec, gt, 'uniform')
        out[sub] = r
    return out


def decodable_segments(te, gt):
    """Segments with cuts and >= 10 level-0 locations, per subset; value =
    row positions sorted by time."""
    out = {'seen': {}, 'unseen': {}}
    key = te['key']
    flat = key[:, 0].astype(np.int64) * 1000003 + key[:, 1].astype(np.int64)
    order = np.argsort(flat, kind='stable')
    _, starts = np.unique(flat[order], return_index=True)
    for grp in np.split(order, starts[1:]):
        k = (int(key[grp[0], 0]), int(key[grp[0], 1]))
        g = gt[k]
        if np.isnan(g['cuts']).any() or len(grp) < N_PHASE:
            continue
        sub = 'seen' if g['seen'] else 'unseen'
        out[sub][k] = grp[np.argsort(te['t'][grp], kind='stable')]
    return out


# ------------------------------------------------------------------- split
def run_split(n, args, device):
    cfg = load_config(args.config)
    feat_folder = cfg['dataset']['feat_folder']
    seen = pp.load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'train', n)
    unseen = pp.load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'test', n)
    with open(args.pseudo.format(n=n), encoding='utf-8') as f:
        pseudo = json.load(f)['videos']
    with open(args.pseudo_unseen.format(n=n), encoding='utf-8') as f:
        pseudo_unseen = json.load(f)['videos']
    gt = pt.test_segments(cfg, n, pseudo, seen, unseen, pseudo_unseen)
    dtr = dict(np.load(os.path.join(args.dump_dir, f'split{n}_train.npz')))
    dte = dict(np.load(os.path.join(args.dump_dir, f'split{n}_test.npz')))
    cache = {}
    t0 = time.time()
    tr_rows, Xtr_raw, tr_stats = gather_train(cfg, n, dtr, feat_folder, cache)
    te_rows, Xte_raw, te_phase = gather_test(cfg, n, dte, feat_folder, cache, gt)
    print(f"  [gather] train rows {len(tr_rows)} ({tr_stats}), test rows {len(te_rows)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    # per-stream statistics (RGB = [:1024], flow = [1024:])
    stream_stats = {s: {'mean_abs': float(np.abs(Xtr_raw[:, sl]).mean()),
                        'mean_l2': float(np.linalg.norm(Xtr_raw[:, sl], axis=1).mean()),
                        'frac_zero': float((Xtr_raw[:, sl] == 0).mean())}
                    for s, sl in (('rgb', slice(0, 1024)), ('flow', slice(1024, 2048)))}
    # targets
    ytr = {'pseudo_phase': dtr['in_phase'][tr_rows].astype(np.int64),
           'decile': np.minimum((dtr['in_tnorm'][tr_rows] * N_PHASE).astype(np.int64), 9)}
    te = {'t': dte['t'][te_rows], 'tnorm': dte['in_tnorm'][te_rows],
          'unseen': dte['in_unseen'][te_rows],
          'key': np.stack([dte['video'][te_rows], dte['in_seg'][te_rows]], 1),
          'pseudo_phase': te_phase,
          'decile': np.minimum((dte['in_tnorm'][te_rows] * N_PHASE).astype(np.int64), 9)}
    te['decodable'] = decodable_segments(te, gt)
    # features
    tf, tf_meta = make_transforms(Xtr_raw)
    Xtr = {'trunk_l0': dtr['feat'][tr_rows].astype(np.float32)}
    Xte = {'trunk_l0': dte['feat'][te_rows].astype(np.float32)}
    for v in VARIANTS[1:]:
        Xtr[v] = tf[v](Xtr_raw).astype(np.float32)
        Xte[v] = tf[v](Xte_raw).astype(np.float32)
    res = {'meta': {'n_train_rows': int(len(tr_rows)), 'n_test_rows': int(len(te_rows)),
                    'train_offset_recovery': tr_stats, 'stream_stats': stream_stats,
                    'transforms': tf_meta,
                    'train_hist_pseudo': np.bincount(ytr['pseudo_phase'], minlength=10).tolist(),
                    'train_hist_decile': np.bincount(ytr['decile'], minlength=10).tolist(),
                    'n_decodable': {s: len(te['decodable'][s]) for s in ('seen', 'unseen')},
                    'n_test_rows_unseen_with_pseudo': int(((te_phase >= 0) & te['unseen']).sum())}}
    # context rows for the decode: equal-width bins vs pseudo (chance) and
    # the pseudo-label oracle, per subset
    for sub in ('seen', 'unseen'):
        dec = te['decodable'][sub]
        if not dec:
            continue
        chance, oracle = [], []
        for key, pos in dec.items():
            g = gt[key]
            L = g['e'] - g['s']
            b = g['s'] + L * np.arange(1, 10) / 10.0
            chance.append(pt.boundary_metrics(b, g['cuts'], g['s'], g['e'], g['sec_per_grid']))
            ph = (te['t'][pos][:, None] >= g['cuts'][None]).sum(1)
            oh = np.full((len(pos), N_PHASE), -20.0)
            oh[np.arange(len(pos)), ph] = 0.0
            b = pt.boundaries_from_states(te['t'][pos].astype(np.float64), pt.viterbi_lr(oh),
                                          g['s'], g['e'])
            oracle.append(pt.boundary_metrics(b, g['cuts'], g['s'], g['e'], g['sec_per_grid']))
        res['meta'][f'decode_context_{sub}'] = {
            'equal_bins_vs_pseudo': pt.agg(chance), 'oracle_labels_vs_pseudo': pt.agg(oracle)}
    for tg in TARGETS:
        res[tg] = {}
        for v in VARIANTS + ('both_shuffled',):
            vv = 'both' if v == 'both_shuffled' else v
            per_seed = []
            t1 = time.time()
            for seed in args.seeds:
                y = ytr[tg]
                if v == 'both_shuffled':
                    y = np.random.RandomState(seed).permutation(y)
                head = pp.train_head(Xtr[vv], y, seed, Xtr[vv].shape[1], device=device)
                r = evaluate(head, Xte[vv], te, gt, device)
                tr_arg = log_post(head, Xtr[vv], device).argmax(1)
                r['train_acc10'] = float((tr_arg == y).mean())
                per_seed.append(r)
                del head
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
            res[tg][v] = pp.average_dicts(per_seed)
            res[tg][v]['d_in'] = int(Xtr[vv].shape[1])
            print(f"  [{tg}/{v}] " + fmt_row(res[tg][v]) + f"  ({time.time() - t1:.0f}s)",
                  flush=True)
    return res


# ------------------------------------------------------------------ report
def fmt_row(r):
    s, u = r['seen'], r['unseen']
    parts = []
    for tg in TARGETS:
        if f'acc10_{tg}' in s:
            parts.append(f"{tg[:6]}: acc10 {s[f'acc10_{tg}']:.3f}/"
                         f"{u.get(f'acc10_{tg}', float('nan')):.3f} acc3 {s[f'acc3_{tg}']:.3f}")
    parts.append(f"rhoE {s['spearman']['expect']['per_seg_mean']:.3f}/"
                 f"{u['spearman']['expect']['per_seg_mean']:.3f}")
    if 'decode_vs_pseudo' in s:
        parts.append(f"MAE%L vs pseudo {100 * s['decode_vs_pseudo']['mae_frac']:.1f}/"
                     f"{100 * u['decode_vs_pseudo']['mae_frac']:.1f} "
                     f"vs uniform {100 * s['decode_vs_uniform']['mae_frac']:.1f}/"
                     f"{100 * u['decode_vs_uniform']['mae_frac']:.1f}")
    parts.append(f"train {r['train_acc10']:.3f}")
    return ' | '.join(parts)


def fmt_table(res, title):
    lines = [f"===== {title} ====="]
    for tg in TARGETS:
        lines.append(f"--- target: {tg}")
        lines.append(f"  {'features':16s} {'acc10 S':>8s} {'acc10 U':>8s} {'acc3 S':>7s} "
                     f"{'rhoE S':>7s} {'rhoE U':>7s} {'rhoA S':>7s} {'rhoA U':>7s} | "
                     f"{'MAE%L~ps S':>10s} {'MAE%L~ps U':>10s} {'<10% S':>7s} {'tIoU S':>7s} "
                     f"{'MAE%L~uni S':>11s} {'MAE%L~uni U':>11s}")
        for v in VARIANTS + ('both_shuffled',):
            r = res[tg][v]
            s, u = r['seen'], r['unseen']
            a10u = u.get(f'acc10_{tg}', float('nan'))
            dp, du = s.get('decode_vs_pseudo'), s.get('decode_vs_uniform')
            dpu, duu = u.get('decode_vs_pseudo'), u.get('decode_vs_uniform')
            lines.append(
                f"  {v:16s} {s[f'acc10_{tg}']:8.3f} {a10u:8.3f} {s[f'acc3_{tg}']:7.3f} "
                f"{s['spearman']['expect']['per_seg_mean']:7.3f} "
                f"{u['spearman']['expect']['per_seg_mean']:7.3f} "
                f"{s['spearman']['argmax']['per_seg_mean']:7.3f} "
                f"{u['spearman']['argmax']['per_seg_mean']:7.3f} | "
                f"{100 * dp['mae_frac']:10.1f} {100 * dpu['mae_frac']:10.1f} "
                f"{100 * dp['within10']:7.1f} {dp['phase_tiou']:7.3f} "
                f"{100 * du['mae_frac']:11.1f} {100 * duu['mae_frac']:11.1f}")
    m = res['meta']
    for sub in ('seen', 'unseen'):
        c = m.get(f'decode_context_{sub}')
        if c:
            lines.append(f"  context {sub}: equal bins vs pseudo MAE%L "
                         f"{100 * c['equal_bins_vs_pseudo']['mae_frac']:.1f} (tIoU "
                         f"{c['equal_bins_vs_pseudo']['phase_tiou']:.3f}); oracle labels "
                         f"MAE%L {100 * c['oracle_labels_vs_pseudo']['mae_frac']:.1f}; "
                         f"decodable segs {m['n_decodable'][sub]:.0f}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--pseudo', default='data/phase_pseudo/thumos14_75_25_split{n}.json')
    ap.add_argument('--pseudo-unseen', default='data/phase_pseudo/thumos14_75_25_split{n}_unseen.json')
    ap.add_argument('--dump-dir', default='data/phase_pseudo/probe_dump')
    ap.add_argument('--out', default='data/phase_pseudo/raw_probe_results.json')
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--merge', nargs='*', default=None,
                    help='no training: merge the per-split results of these jsons into --out, '
                         'recompute the mean and print the tables')
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.merge is not None:
        merged = None
        for p in args.merge:
            with open(p) as f:
                r = json.load(f)
            merged = merged or {'meta': r['meta'], 'splits': {}}
            merged['splits'].update(r['splits'])
        merged['splits'] = dict(sorted(merged['splits'].items()))
        keys = list(merged['splits'])
        for k in keys:
            print(fmt_table(merged['splits'][k], f'split {k}'), flush=True)
        merged['mean'] = pp.average_dicts([merged['splits'][k] for k in keys])
        print(fmt_table(merged['mean'], f'mean over splits {keys}'), flush=True)
        with open(args.out, 'w') as f:
            json.dump(merged, f, indent=1)
        print(f"[save] {args.out}")
        return
    all_res = {'meta': {
        'head': 'Linear(D,D)-ReLU-Linear(D,10), Adam 1e-3, 30 ep, bs 4096 (phase_probe.train_head)',
        'seeds': args.seeds, 'rows': "level-0 locations, 'in' rule, from probe_dump (train: "
        "rows with a pseudo phase; test: rows inside any GT segment)",
        'variants': {'trunk_l0': "detector cls-trunk feature (256-d) on the same rows = "
                                 "phase_probe.py reference restricted to level 0",
                     'rgb': 'raw I3D dims [:1024]', 'flow': 'raw I3D dims [1024:]',
                     'both': 'raw I3D 2048-d',
                     'both_whiten': '(x - mean) / std, statistics over the training rows',
                     'both_abtt5': 'centred minus the projection on the top-5 PCs of the '
                                   'centred training rows',
                     'both_shuffled': 'raw 2048-d with permuted training labels (chance)'},
        'targets': {'pseudo_phase': 'text-aligned pseudo phase 0..9 (seen: split json; '
                                    'unseen: *_unseen.json, evaluation only)',
                    'decile': 'floor(10 * (t - s) / (e - s)), no pseudo labels'},
        'spearman': 'per segment (>= 5 locations; constant prediction -> 0) of the argmax (A) '
                    'and expected (E) phase vs normalized time',
        'decode': 'phase_timespans.viterbi_lr on segments with pseudo cuts and >= 10 level-0 '
                  'locations; boundaries vs the pseudo cuts and vs equal-width bins'},
        'splits': {}}
    for n in args.splits:
        t0 = time.time()
        print(f"===== split {n} =====", flush=True)
        r = run_split(n, args, args.device)
        all_res['splits'][str(n)] = r
        print(fmt_table(r, f'split {n}') + f"\n  ({time.time() - t0:.0f}s)", flush=True)
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
    keys = list(all_res['splits'])
    if len(keys) > 1:
        all_res['mean'] = pp.average_dicts([all_res['splits'][k] for k in keys])
        print(fmt_table(all_res['mean'], f'mean over splits {keys}'), flush=True)
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
    print(f"[save] {args.out}")


if __name__ == '__main__':
    main()
