"""REPRESENTATION-QUALITY METRICS of the learned representation (THUMOS14 75/25 splits 0-2, frozen final champion
ckpt/thumos14_i3d_seqchain_{n}; eval-only, nothing is trained except tiny
linear probes on the frozen features).

Three standard metrics, each with a chance / baseline column:

(1) MISTAKE SEVERITY (Bertinetto et al., CVPR'20 "Making better mistakes").
    Over the FINAL detections (the fork's postprocess_results on the raw
    post-NMS dump of the full-vocabulary eval, exactly as in
    scripts/tree_consistency.py; unfused and fused) that overlap a GT
    segment at tIoU >= 0.5 but carry the WRONG class, the tree distance
    between the predicted and the true class: 2 if both hang off the same
    root action (configs/thumos14_fam_map.json), 4 otherwise.  Reported as
    the fraction of same-root mistakes and the mean distance, over the
    top-200 detections per video and over the above-median-score subset,
    split by the SEEN / UNSEEN status of the ground-truth class.
    Baselines: (a) a random wrong class drawn uniformly from the other 19
    (analytic expectation per mistake), (b) the name-only model of the
    ablation (-HP, ckpt/thumos14_i3d_vanAS_{n}; same eval, its own raw dump
    under data/rep_metrics/vanAS_s{n}/).

(2) TEMPORAL ALIGNMENT, TCC family (Dwibedi et al., CVPR'19): on the
    level-0 cls-trunk features of every location inside a GT segment
    ('inside' rule of scripts/phase_probe.py), a LINEAR regressor from the
    frozen feature to the normalized time t_norm in [0,1] is trained on
    the champion's own TRAIN-video rows (data/phase_pseudo/probe_dump/
    seqchain_split{n}_train.npz) and evaluated on the held-out test
    videos: phase-progression R^2 and pooled Spearman on seen-class and
    on unseen-class segments, and Kendall's tau between the predicted
    progress and the true time order within each segment (>= 5 level-0
    locations), mean over segments.  Control: the same regressor trained
    on row-shuffled features.  3 seeds for every trained regressor.

(3) NEAREST-NEIGHBOUR RETRIEVAL of UNSEEN actions: the level-0 features
    inside every test GT segment are mean-pooled; each unseen-class
    segment queries the other segments (cosine, same video excluded) and
    recall@1 / @5 of the same class is reported against (a) all test
    segments and (b) unseen-only segments, versus the analytic chance of a
    random ranking, and versus the same retrieval on the mean-pooled RAW
    I3D features (2048-d, RGB+flow, stride 4; the backbone input) over
    the identical locations.

Test-video features of the champion are harvested once with the probe
stash (TIFAD_PROBE_STASH=1, scripts/phase_probe.py machinery, eval mode
= the test-vocabulary conditioning of the real eval) into
data/rep_metrics/champ_test_s{n}.npz (level 0 only, with video names).

Usage:
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=2 \
      python scripts/rep_metrics.py --splits 0 1 2
  --skip-dump reuses the feature dumps (CPU only from there on).
Outputs: data/rep_metrics.json and one table on stdout.
"""
import argparse
import json
import os
import pickle
import sys
import time

ROOT = '/data3/xiaodan8/tifad'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
# champion text design (must precede the libs / phase_probe imports; the
# probe module sets the remaining reference envs at import)
os.environ.setdefault('TIFAD_ATTNDESIGN', 'phasechain')
os.environ.pop('TIFAD_PH_SRC', None)

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

SPLIT_DIR = 'splits/train_75_test_25/THUMOS14/{mode}/split_{n}.list'
FAM_MAP = 'configs/thumos14_fam_map.json'
ANNO = '/data3/xiaodan8/thumos/annotations/thumos14.json'
RAW_FEAT = '/data3/xiaodan8/thumos/i3d_features'
TREE_DIR = 'data/tree_consistency'
CHAMP_TRAIN = 'data/phase_pseudo/probe_dump/seqchain_split{n}_train.npz'
OUT_DIR = 'data/rep_metrics'
CHAMP_CKPT = 'ckpt/thumos14_i3d_seqchain_{n}/mAP_best.pth.tar'
NAMEONLY_RAW = 'data/rep_metrics/vanAS_s{n}/raw.pkl'
K_TOP = 200
MIN_PTS = 5


def read_list(p):
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


# ------------------------------------------------------------ feature dump
def dump_champion_test(n, cfg, out_path):
    """Level-0 cls-trunk features of every test-video location that lies
    inside a GT segment (all 20 classes), eval mode, with video names.
    Mirrors scripts/phase_probe.dump_test (same model build, same
    'inside' assignment), minus the phase labels."""
    import phase_probe as pp          # sets the reference envs
    from libs.datasets import make_data_loader, make_dataset
    from libs.utils import fix_random_seed
    fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    seen = read_list(SPLIT_DIR.format(mode='train', n=n))
    unseen = read_list(SPLIT_DIR.format(mode='test', n=n))
    all_cls = seen + unseen
    model, notes = pp.build_model(cfg, n, CHAMP_CKPT.format(n=n))
    print(f"[dump] split {n}: {CHAMP_CKPT.format(n=n)} loaded; notes={notes}", flush=True)
    ds = make_dataset(cfg['dataset_name'], n, False, cfg['val_split'], **cfg['dataset'])
    ds.label_dict = None
    ds.subset_labels = all_cls
    ds.data_list, ds.label_dict = ds._load_json_db(ds.json_file)
    loader = make_data_loader(ds, False, None, 1, cfg['loader']['num_workers'])
    with open(ANNO, encoding='utf-8') as f:
        db = json.load(f)['database']
    cls_id = {c: i for i, c in enumerate(all_cls)}
    unseen_set = set(unseen)
    model.eval()
    model.test_pre_nms_thresh = 2.0
    model.test_nms_method = 'none'
    rows, names = [], []
    for vi, vl in enumerate(loader):
        item = vl[0]
        vid = item['video_id']
        fps, fstr, nfr = item['fps'], item['feat_stride'], item['feat_num_frames']
        segs, labs, uns = [], [], []
        for a in db[vid].get('annotations', []):
            if a['label'] not in cls_id:
                continue
            s, e = float(a['segment'][0]), float(a['segment'][1])
            segs.append([(s * fps - 0.5 * nfr) / fstr, (e * fps - 0.5 * nfr) / fstr])
            labs.append(cls_id[a['label']])
            uns.append(a['label'] in unseen_set)
        if not segs:
            continue
        segs_np = np.asarray(segs, np.float32)
        with torch.no_grad():
            model(vl)
            st = model._probe_stash
        feats = st['feats'][0][0].transpose(0, 1)          # (T0, D)
        mask = st['masks'][0][0].bool().cpu().numpy()
        t = st['points'][0][:, 0].cpu().numpy()
        cuts = np.full((len(segs), 9), np.nan, np.float32)
        seg_in, _, tn_in = pp.assign_inside(t, segs_np, cuts)
        keep = mask & (seg_in >= 0)
        idx = np.flatnonzero(keep)
        vix = len(names)
        names.append(vid)
        rows.append({
            'feat': feats[idx].half().cpu().numpy(),
            't': t[idx].astype(np.float32),
            'video': np.full(len(idx), vix, np.int32),
            'seg': seg_in[idx].astype(np.int32),
            'cls': np.asarray(labs, np.int16)[seg_in[idx]],
            'unseen': np.asarray(uns, bool)[seg_in[idx]],
            'tnorm': tn_in[idx].astype(np.float32),
            'n_segs': np.int32(len(segs)),
        })
    data = {k: np.concatenate([r[k] for r in rows], 0) for k in rows[0] if k != 'n_segs'}
    data['n_segs_per_video'] = np.asarray([r['n_segs'] for r in rows], np.int32)
    data['video_names'] = np.asarray(names)
    data['classes'] = np.asarray(all_cls)
    np.savez_compressed(out_path, **data)
    print(f"[dump] {out_path}: {len(names)} videos, {len(data['t'])} level-0 rows, "
          f"{int(data['n_segs_per_video'].sum())} GT segments", flush=True)
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------ (1) mistake severity
def gt_table(video_names, all_cls):
    with open(ANNO, encoding='utf-8') as f:
        db = json.load(f)['database']
    cid = {c: i for i, c in enumerate(all_cls)}
    gt = {}
    for v in video_names:
        segs, labs = [], []
        for a in db[v].get('annotations', []):
            if a['label'] in cid:
                segs.append([float(a['segment'][0]), float(a['segment'][1])])
                labs.append(cid[a['label']])
        gt[v] = (np.asarray(segs, np.float64).reshape(-1, 2), np.asarray(labs, int))
    return gt


def tiou(seg, segs):
    inter = np.maximum(0.0, np.minimum(seg[1], segs[:, 1]) - np.maximum(seg[0], segs[:, 0]))
    union = (seg[1] - seg[0]) + (segs[:, 1] - segs[:, 0]) - inter
    return inter / np.maximum(union, 1e-8)


def mistake_severity(raw, cfg, seen, fam, tag, n):
    """Final detections (unfused / fused) -> mistakes at tIoU>=0.5 ->
    tree distance to the GT class; random-class expectation alongside."""
    from tree_consistency import final_detections
    all_cls = list(raw['class_names'][0])
    C = len(all_cls)
    root = np.asarray([fam[c] for c in all_cls])
    root_size = np.asarray([(root == root[i]).sum() for i in range(C)])
    is_seen_cls = np.asarray([c in set(seen) for c in all_cls])
    vids_all = sorted(set(raw['video-id']))
    gt = gt_table(vids_all, all_cls)
    out = {}
    for mode in ('unfused', 'fused'):
        res = final_detections(raw, cfg, mode == 'fused', seen, all_cls)
        vids = np.asarray(res['video-id'])
        ts, te = np.asarray(res['t-start'], np.float64), np.asarray(res['t-end'], np.float64)
        lab, sc = np.asarray(res['label']).astype(int), np.asarray(res['score'], np.float64)
        N = len(vids)
        rank = np.zeros(N, int)
        for v in np.unique(vids):
            idx = np.flatnonzero(vids == v)
            order = idx[np.argsort(-sc[idx], kind='stable')]
            rank[order] = np.arange(len(idx))
        hit = np.zeros(N, bool)          # tIoU>=0.5 with a GT of the predicted class
        gt_cls = np.full(N, -1, int)     # class of the best-overlapping GT (any class)
        best_iou = np.zeros(N)
        for i in range(N):
            segs, labs = gt[vids[i]]
            if len(labs) == 0:
                continue
            io = tiou((ts[i], te[i]), segs)
            j = int(io.argmax())
            best_iou[i] = io[j]
            gt_cls[i] = labs[j]
            same = labs == lab[i]
            if same.any() and io[same].max() >= 0.5:
                hit[i] = True
        localized = best_iou >= 0.5
        mistake = localized & ~hit
        dist = np.where(root[lab] == root[np.maximum(gt_cls, 0)], 2, 4)
        p_rand = (root_size[np.maximum(gt_cls, 0)] - 1) / (C - 1)      # P(random wrong class shares the root)
        r = {'n_final': int(N), 'median_score': float(np.median(sc))}
        for sel_name, sel in (('topk', rank < K_TOP), ('above_median', sc > np.median(sc))):
            rr = {}
            for grp, gm in (('all', np.ones(N, bool)),
                            ('seen', is_seen_cls[np.maximum(gt_cls, 0)]),
                            ('unseen', ~is_seen_cls[np.maximum(gt_cls, 0)])):
                m = sel & mistake & gm
                nloc = int((sel & localized & gm).sum())
                rr[grp] = {
                    'n_localized': nloc, 'n_mistakes': int(m.sum()),
                    'mistake_rate': float(m.sum() / nloc) if nloc else float('nan'),
                    'same_root_frac': float((dist[m] == 2).mean()) if m.sum() else float('nan'),
                    'mean_dist': float(dist[m].mean()) if m.sum() else float('nan'),
                    'random_same_root_frac': float(p_rand[m].mean()) if m.sum() else float('nan'),
                    'random_mean_dist': float((2 * p_rand[m] + 4 * (1 - p_rand[m])).mean()) if m.sum() else float('nan'),
                }
            # by the PREDICTED class' seen/unseen status (for the record)
            for grp, gm in (('pred_seen', is_seen_cls[lab]), ('pred_unseen', ~is_seen_cls[lab])):
                m = sel & mistake & gm
                rr[grp] = {'n_mistakes': int(m.sum()),
                           'same_root_frac': float((dist[m] == 2).mean()) if m.sum() else float('nan'),
                           'mean_dist': float(dist[m].mean()) if m.sum() else float('nan')}
            r[sel_name] = rr
        # confusion of the mistakes (top-k): gt -> pred counts
        m = (rank < K_TOP) & mistake
        conf = {}
        for g, p in zip(gt_cls[m], lab[m]):
            conf.setdefault(all_cls[g], {})
            conf[all_cls[g]][all_cls[p]] = conf[all_cls[g]].get(all_cls[p], 0) + 1
        r['topk_confusions'] = conf
        out[mode] = r
    out['model'] = tag
    return out


# ------------------------------------------------- (2) TCC-style alignment
def train_linear(X, y, seed, epochs=100, bs=4096, lr=3e-3):
    """Linear regressor feature -> t_norm, Adam (bias initialised at the
    target mean; 100 epochs reach the closed-form least-squares fit to
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)[:, None]
    lin = torch.nn.Linear(X.shape[1], 1)
    lin.bias.data.fill_(float(y.mean()))
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(lin(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    W = lin.weight.detach().numpy()[0]
    b = float(lin.bias.detach().numpy()[0])
    return lambda Z: Z @ W + b


def ridge_closed(X, y, lam=1.0):
    Xa = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1).astype(np.float64)
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    w = np.linalg.solve(A, Xa.T @ y.astype(np.float64))
    return lambda Z: Z.astype(np.float64) @ w[:-1] + w[-1]


def seg_groups(video, seg):
    key = video.astype(np.int64) * 1000003 + seg.astype(np.int64)
    order = np.argsort(key, kind='stable')
    _, starts = np.unique(key[order], return_index=True)
    return np.split(order, starts[1:])


def align_metrics(pred, tn, t, groups):
    ss_res = float(((pred - tn) ** 2).sum())
    ss_tot = float(((tn - tn.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rho = float(spearmanr(pred, tn).correlation) if np.ptp(pred) > 0 else 0.0
    taus = []
    for g in groups:
        if len(g) < MIN_PTS:
            continue
        if np.ptp(pred[g]) == 0:
            taus.append(0.0)
            continue
        taus.append(float(kendalltau(pred[g], t[g]).correlation))
    return {'r2': r2, 'spearman': rho, 'kendall_tau': float(np.mean(taus)) if taus else float('nan'),
            'kendall_frac_pos': float(np.mean(np.asarray(taus) > 0)) if taus else float('nan'),
            'n_rows': int(len(pred)), 'n_segs_tau': len(taus)}


def tcc_metrics(train, test, seeds):
    m = (train['level'] == 0) & (train['in_seg'] >= 0)
    Xtr = train['feat'][m].astype(np.float32)
    ytr = train['in_tnorm'][m].astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (test['feat'].astype(np.float32) - mu) / sd
    tn, t = test['tnorm'].astype(np.float32), test['t'].astype(np.float32)
    sets = {'seen': ~test['unseen'], 'unseen': test['unseen']}
    groups = {k: seg_groups(test['video'][v], test['seg'][v]) for k, v in sets.items()}
    out = {'n_train_rows': int(len(ytr)), 'n_train_videos': int(len(np.unique(train['video'][m]))),
           'train_fit': {}, 'feature': {}, 'shuffled': {}, 'ridge': {}}
    for variant in ('feature', 'shuffled'):
        per_seed = {k: [] for k in sets}
        fit = []
        for seed in seeds:
            X_in = Xtr if variant == 'feature' else Xtr[np.random.RandomState(seed).permutation(len(Xtr))]
            f = train_linear(X_in, ytr, seed)
            p_tr = f(X_in)
            fit.append(1.0 - float(((p_tr - ytr) ** 2).sum()) / float(((ytr - ytr.mean()) ** 2).sum()))
            for k, v in sets.items():
                per_seed[k].append(align_metrics(f(Xte[v]), tn[v], t[v], groups[k]))
        out[variant] = {k: {kk: float(np.mean([r[kk] for r in rs])) for kk in rs[0]}
                        for k, rs in per_seed.items()}
        out[variant]['per_seed'] = per_seed
        out['train_fit'][variant] = float(np.mean(fit))
    f = ridge_closed(Xtr, ytr)
    for k, v in sets.items():
        out['ridge'][k] = align_metrics(f(Xte[v]), tn[v], t[v], groups[k])
    return out


# -------------------------------------------------------- (3) retrieval
def pool_segments(test):
    groups = seg_groups(test['video'], test['seg'])
    feats = test['feat'].astype(np.float32)
    P, cls, uns, vid, tidx = [], [], [], [], []
    for g in groups:
        P.append(feats[g].mean(0))
        cls.append(int(test['cls'][g[0]]))
        uns.append(bool(test['unseen'][g[0]]))
        vid.append(int(test['video'][g[0]]))
        tidx.append(test['t'][g].astype(int))
    return np.stack(P), np.asarray(cls), np.asarray(uns), np.asarray(vid), tidx


def pool_raw(test, vid, tidx):
    names = test['video_names']
    cache = {}
    P = []
    for v, ti in zip(vid, tidx):
        if v not in cache:
            cache[v] = np.load(os.path.join(RAW_FEAT, str(names[v]) + '.npy')).astype(np.float32)
        x = cache[v]
        ti = ti[ti < len(x)]
        P.append(x[ti].mean(0))
    return np.stack(P)


def retrieval(P, cls, uns, vid, query_mask, gallery_mask, ks=(1, 5)):
    Z = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)
    S = Z @ Z.T
    q_idx = np.flatnonzero(query_mask)
    rec = {k: [] for k in ks}
    chance = {k: [] for k in ks}
    from math import comb
    for i in q_idx:
        gal = gallery_mask & (vid != vid[i])
        gal[i] = False
        gi = np.flatnonzero(gal)
        if len(gi) == 0:
            continue
        order = gi[np.argsort(-S[i, gi], kind='stable')]
        same = cls[order] == cls[i]
        n_same, N = int(same.sum()), len(gi)
        for k in ks:
            rec[k].append(float(same[:k].any()))
            kk = min(k, N)
            chance[k].append(1.0 - (comb(N - n_same, kk) / comb(N, kk) if N - n_same >= kk else 0.0))
    out = {f'recall@{k}': float(np.mean(rec[k])) for k in ks}
    out.update({f'chance@{k}': float(np.mean(chance[k])) for k in ks})
    out['n_queries'] = len(rec[ks[0]])
    return out


def retrieval_block(P, cls, uns, vid):
    out = {}
    for qname, qm in (('unseen', uns), ('seen', ~uns)):
        out[qname] = {'gallery_all': retrieval(P, cls, uns, vid, qm, np.ones(len(P), bool)),
                      'gallery_same_group': retrieval(P, cls, uns, vid, qm, uns if qname == 'unseen' else ~uns)}
    return out


# ------------------------------------------------------------- reporting
def nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float('nan')


def table(all_res):
    keys = sorted(all_res['splits'])
    rows = []

    def row(metric, get_seen, get_unseen, get_ctrl, ctrl_name):
        vals = []
        for k in keys:
            s = all_res['splits'][k]
            vals.append((get_seen(s), get_unseen(s), get_ctrl(s)))
        mean = tuple(nanmean([v[i] for v in vals]) for i in range(3))
        rows.append((metric, ctrl_name, vals, mean))

    ms = lambda s, m, sel, g, f: s['mistake']['champion'][m][sel][g][f]
    for mode in ('fused', 'unfused'):
        for sel in ('topk', 'above_median'):
            row(f'(1) same-root mistake frac [{mode},{sel}]',
                lambda s, m=mode, sel=sel: ms(s, m, sel, 'seen', 'same_root_frac'),
                lambda s, m=mode, sel=sel: ms(s, m, sel, 'unseen', 'same_root_frac'),
                lambda s, m=mode, sel=sel: ms(s, m, sel, 'all', 'random_same_root_frac'),
                'random wrong class')
            row(f'(1) mean tree dist of mistakes [{mode},{sel}]',
                lambda s, m=mode, sel=sel: ms(s, m, sel, 'seen', 'mean_dist'),
                lambda s, m=mode, sel=sel: ms(s, m, sel, 'unseen', 'mean_dist'),
                lambda s, m=mode, sel=sel: ms(s, m, sel, 'all', 'random_mean_dist'),
                'random wrong class')
            if 'nameonly' in all_res['splits'][keys[0]]['mistake']:
                nm = lambda s, m, sel, g, f: s['mistake']['nameonly'][m][sel][g][f]
                row(f'(1) same-root frac, name-only model [{mode},{sel}]',
                    lambda s, m=mode, sel=sel: nm(s, m, sel, 'seen', 'same_root_frac'),
                    lambda s, m=mode, sel=sel: nm(s, m, sel, 'unseen', 'same_root_frac'),
                    lambda s, m=mode, sel=sel: nm(s, m, sel, 'all', 'random_same_root_frac'),
                    'random wrong class')
    for met in ('r2', 'spearman', 'kendall_tau'):
        row(f'(2) progression {met}',
            lambda s, met=met: s['tcc']['feature']['seen'][met],
            lambda s, met=met: s['tcc']['feature']['unseen'][met],
            lambda s, met=met: s['tcc']['shuffled']['unseen'][met],
            'shuffled-feature (unseen)')
    for gal in ('gallery_all', 'gallery_same_group'):
        for k in (1, 5):
            row(f'(3) unseen retrieval recall@{k} [{gal}]',
                lambda s, gal=gal, k=k: s['retrieval']['champion']['unseen'][gal][f'recall@{k}'],
                lambda s, gal=gal, k=k: s['retrieval']['raw_i3d']['unseen'][gal][f'recall@{k}'],
                lambda s, gal=gal, k=k: s['retrieval']['champion']['unseen'][gal][f'chance@{k}'],
                'chance')
    lines = []
    hdr = f"{'metric':52s} | " + ' | '.join(f"split {k}: seen/unseen/ctrl" for k in keys) + \
        " | mean: seen/unseen/ctrl | ctrl"
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for metric, ctrl, vals, mean in rows:
        cells = [f"{a:.3f}/{b:.3f}/{c:.3f}" for a, b, c in vals]
        lines.append(f"{metric:52s} | " + ' | '.join(f"{c:24s}" for c in cells) +
                     f" | {mean[0]:.3f}/{mean[1]:.3f}/{mean[2]:.3f} | {ctrl}")
    lines.append("(3) columns: seen = champion features, unseen = RAW I3D features (both for unseen "
                 "queries), ctrl = chance")
    return '\n'.join(lines), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--skip-dump', action='store_true')
    ap.add_argument('--out', default='data/rep_metrics.json')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    from libs.core import load_config
    cfg = load_config(args.config)
    with open(FAM_MAP, encoding='utf-8') as f:
        fam = json.load(f)
    all_res = {'meta': {
        'ckpt': CHAMP_CKPT, 'envs': 'phase_probe.REF_ENVS + TIFAD_ATTNDESIGN=phasechain (test-vocabulary conditioning)',
        'nameonly_ckpt': 'ckpt/thumos14_i3d_vanAS_{n}/mAP_best.pth.tar (-HP row; envs TIFAD_FAMCE=0.3 '
                         'TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1)',
        'mistake': f'final detections (tree_consistency.final_detections; unfused and fused), tIoU>=0.5 with a GT '
                   f'but wrong class; tree distance 2 = same root (fam_map), 4 = other root; top-{K_TOP} per video '
                   f'and above-median score; seen/unseen = status of the GT class; random baseline = uniform wrong '
                   f'class among the other 19 (analytic)',
        'tcc': f'linear regressor feature->t_norm (Adam 3e-3, 100 ep, bs 4096, bias init at the target mean, standardized level-0 features, '
               f'inside rule) trained on the champion train-video rows; R^2, pooled Spearman, per-segment Kendall '
               f'tau (>= {MIN_PTS} level-0 locations) vs true time order; shuffled = row-permuted train features; '
               f'seeds {args.seeds}; ridge(lam=1) closed form for reference',
        'retrieval': 'mean-pooled level-0 features per test GT segment; cosine; same video excluded; recall@1/5 of '
                     'the same class; gallery_all = every other test segment, gallery_same_group = unseen-only '
                     '(seen-only for seen queries); chance = analytic random ranking; raw_i3d = mean of the raw '
                     '2048-d I3D rows at the identical locations',
        'seeds': args.seeds, 'k_top': K_TOP},
        'splits': {}}
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            all_res['splits'] = json.load(f).get('splits', {})
    for n in args.splits:
        t0 = time.time()
        seen = read_list(SPLIT_DIR.format(mode='train', n=n))
        dump = os.path.join(OUT_DIR, f'champ_test_s{n}.npz')
        if not args.skip_dump or not os.path.exists(dump):
            dump_champion_test(n, cfg, dump)
        test = dict(np.load(dump))
        res = {}
        # (1) mistake severity: champion + name-only model
        res['mistake'] = {}
        with open(os.path.join(TREE_DIR, f's{n}', 'raw.pkl'), 'rb') as f:
            raw = pickle.load(f)
        res['mistake']['champion'] = mistake_severity(raw, cfg, seen, fam, 'seqchain', n)
        nm_pkl = NAMEONLY_RAW.format(n=n)
        if os.path.exists(nm_pkl):
            with open(nm_pkl, 'rb') as f:
                raw_nm = pickle.load(f)
            res['mistake']['nameonly'] = mistake_severity(raw_nm, cfg, seen, fam, 'vanAS', n)
        else:
            print(f"[warn] no name-only dump {nm_pkl}; random baseline only", flush=True)
        # (2) TCC-style alignment
        train = dict(np.load(CHAMP_TRAIN.format(n=n)))
        res['tcc'] = tcc_metrics(train, test, args.seeds)
        # (3) retrieval
        P, cls, uns, vid, tidx = pool_segments(test)
        R = pool_raw(test, vid, tidx)
        res['retrieval'] = {'n_segments': int(len(P)), 'n_unseen_segments': int(uns.sum()),
                            'n_segments_dropped_no_level0_row': int(test['n_segs_per_video'].sum() - len(P)),
                            'champion': retrieval_block(P, cls, uns, vid),
                            'raw_i3d': retrieval_block(R, cls, uns, vid)}
        res['seen'] = seen
        res['unseen'] = [c for c in test['classes'].tolist() if c not in set(seen)]
        all_res['splits'][str(n)] = res
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
        print(f"--- split {n} done ({time.time() - t0:.0f}s)", flush=True)
    txt, rows = table(all_res)
    all_res['table_rows'] = [{'metric': m, 'control': c, 'per_split': v, 'mean': mean}
                             for m, c, v, mean in rows]
    with open(args.out, 'w') as f:
        json.dump(all_res, f, indent=1)
    print(txt)
    print(f"[save] {args.out}")


if __name__ == '__main__':
    main()
