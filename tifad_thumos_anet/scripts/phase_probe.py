"""POST-HOC PHASE PROBE on a FROZEN Ti-FAD detector (THUMOS14 75/25).

Question: do the detector's per-location cls-trunk features (the exact
features the class cosine is computed on) linearly-ish encode the temporal
PHASE (0..9) of the action they sit in?  Nothing about the detector changes.

Pipeline (per split n):
  1. Build the model with the reference (champion stage-1) envs plus
     TIFAD_LEAFLOSS=1 (target machinery) and TIFAD_PROBE_STASH=1 (feature
     stash, meta_archs.py), prompt dropout / semantic degradation forced to
     0 so train-mode features are deterministic.  Load the reference
     checkpoint (the 2-scalar Kendall logvar is zero-padded to 3; every
     other tensor loads strictly).
  2. TRAIN videos: model in train() mode under torch.no_grad() (needed for
     the LEAFLOSS point->GT assignment + phase targets), with every
     Dropout / DropPath / MultiheadAttention module switched to eval so no
     randomness touches the features.  Harvest, for every location inside
     a GT segment, the cls-trunk feature, the (class, phase) target and the
     location's normalized time within its segment -- under TWO location
     rules:  'leaf' = the detector's own positive rule (center sampling +
     regression range, exactly the LEAFLOSS positives) and 'inside' = every
     location of every pyramid level whose time falls inside the segment
     (shortest containing segment on ties).
  3. TEST videos: model in eval() mode (test vocabulary conditioning, as at
     evaluation time).  Same two rules; phase targets for SEEN-class test
     segments come from the Stage-A pseudo json, UNSEEN-class segments (no
     pseudo entry) only carry the normalized time.
  4. Train a tiny MLP head Linear(D,D)-ReLU-Linear(D,10) (Adam 1e-3, 30
     epochs, batch 4096, softmax CE) on the train-video features and report
     held-out seen accuracy (10-way / 3-way merged), Spearman(pred phase,
     normalized time) on seen AND unseen test segments (per segment with
     >= 5 locations, and pooled), a position-only control (MLP on the
     normalized time alone = the ceiling the pseudo labels allow) and a
     shuffled-label chance control.  Three head seeds averaged.

Usage (GPU 2; never GPU 3; the fork's json assets are UTF-8, so force the
locale when launching from a non-UTF-8 shell such as nohup):
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=2 \
      python scripts/phase_probe.py --splits 0 1 2
Outputs: data/phase_pseudo/probe_dump/split{n}_{train,test}.npz (features)
         data/phase_pseudo/probe_results.json (numbers)
"""
import argparse
import json
import math
import os
import sys
import time

# ---- reference (champion stage-1) envs + probe envs; MUST precede the
# libs imports (text.py / meta_archs.py read them at construction).
REF_ENVS = {
    'TIFAD_PROMPT_ENSEMBLE': 'configs/thumos14_qwen_ensemble.json',
    'TIFAD_ATTR_ATTN': '1',
    'TIFAD_FAMCE': '0.3',
    'TIFAD_FAMCE_MAP': 'configs/thumos14_fam_map.json',
    'TIFAD_FAMCE_UW': '1',
    # train-only randomness OFF so the probe features are deterministic
    'TIFAD_PROMPT_DROPOUT': '0',
    'TIFAD_SEM_DEGRADE': '0',
    # probe machinery
    'TIFAD_LEAFLOSS': '1',
    'TIFAD_PROBE_STASH': '1',
}
for _k, _v in REF_ENVS.items():
    os.environ[_k] = _v

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.modeling.text import data_split_dir
from libs.modeling.blocks import AffineDropPath, DropPath
from libs.modeling import tv_attn as _tva
from libs.modeling.clip import CLIPAttention as _ClipAttn
from libs.utils import fix_random_seed

N_PHASE = 10
MERGE3 = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])   # phases 0-2 | 3-6 | 7-9


# --------------------------------------------------------------------- utils
def load_classes(subset_file, data_split, mode, n):
    with open(data_split_dir(subset_file, data_split, mode, n), encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def freeze_randomness(model):
    """train() mode for the target machinery, but every stochastic module
    (Dropout, DropPath, AffineDropPath, MultiheadAttention's dropout, the
    TiCA cross-attention's functional dropout, CLIP attention dropout) in
    eval so features are deterministic.  Returns the number switched."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, DropPath,
                          AffineDropPath, nn.MultiheadAttention,
                          _tva.TVMultiHeadAttention, _tva.DropPath, _ClipAttn)):
            m.eval()
            n += 1
    return n


def build_model(cfg, n, ckpt_file):
    model = make_meta_arch(n, cfg['model_name'], cfg['dataset']['subset_file'],
                           cfg['dataset']['data_split'], **cfg['model'])
    ck = torch.load(ckpt_file, map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    own = model.state_dict()
    notes = []
    for k in list(sd.keys()):
        if k in own and own[k].shape != sd[k].shape:
            if k == 'famce_uw_logvar' and sd[k].numel() == 2 and own[k].numel() == 3:
                pad = torch.zeros(3)
                pad[:2] = sd[k]
                sd[k] = pad
                notes.append(f"{k}: ckpt {tuple(ck['state_dict']['module.' + k].shape)} "
                             f"-> zero-padded to {tuple(own[k].shape)} (leaf slot unused)")
            else:
                raise RuntimeError(f"shape mismatch {k}: {own[k].shape} vs {sd[k].shape}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # only tolerate keys that the probe itself introduces (none expected)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"missing keys: {missing}"
    model = model.cuda()
    return model, notes


def phase_cuts_grid(entry, fps, feat_stride, num_frames):
    """9 internal cut points (end of phases 0..8) on the feature grid, or
    None.  Mirrors libs/datasets/thumos14.py [LEAFLOSS]."""
    pb = entry.get('phase_boundaries')
    s, e = float(entry['segment'][0]), float(entry['segment'][1])
    if pb is None or len(pb) != 10 or any(b is None for b in pb):
        cuts = [s + (e - s) * k / 10.0 for k in range(1, 10)]
    else:
        cuts = [float(pb[k][1]) for k in range(9)]
    return (np.asarray(cuts, dtype=np.float32) * fps - 0.5 * num_frames) / feat_stride


def assign_inside(t, segs, cuts):
    """'inside' rule: every location strictly inside a GT segment; ties ->
    shortest segment.  t (FT,), segs (N,2) grid, cuts (N,9) grid (NaN = no
    phase).  Returns seg index (-1 = none), phase (-1 = none), t_norm."""
    FT, N = t.shape[0], segs.shape[0]
    seg_idx = np.full(FT, -1, dtype=np.int64)
    phase = np.full(FT, -1, dtype=np.int64)
    tnorm = np.full(FT, np.nan, dtype=np.float32)
    if N == 0:
        return seg_idx, phase, tnorm
    left = t[:, None] - segs[None, :, 0]
    right = segs[None, :, 1] - t[:, None]
    inside = (left > 0) & (right > 0)
    lens = np.where(inside, (segs[:, 1] - segs[:, 0])[None, :], np.inf)
    j = lens.argmin(1)
    ok = np.isfinite(lens[np.arange(FT), j])
    seg_idx[ok] = j[ok]
    s, e = segs[j[ok], 0], segs[j[ok], 1]
    tnorm[ok] = (t[ok] - s) / np.maximum(e - s, 1e-6)
    cj = cuts[j[ok]]
    has = ~np.isnan(cj).any(1)
    k = (t[ok][:, None] >= cj).sum(1)
    ph = np.where(has, k, -1)
    phase[ok] = ph
    return seg_idx, phase, tnorm


def leaf_rule(model, points_cat, segs_t, labs_t, cuts_t, n_cls):
    """'leaf' rule = the detector's own positive-location rule via the
    fork's label_points (+ _leaf_targets).  Returns seg index, phase."""
    model._leaf_assign = []
    model.label_points([points_cat], [segs_t], [labs_t], n_cls)
    tg = model._leaf_targets(points_cat, [labs_t], [{'leaf_cuts': cuts_t}])[0]
    inds, pos = model._leaf_assign[0]
    seg_idx = np.where(pos.cpu().numpy(), inds.cpu().numpy(), -1)
    phase = tg[:, 1].cpu().numpy()
    return seg_idx, phase


def unpack_stash(stash):
    feats = torch.cat([f[0].transpose(0, 1) for f in stash['feats']], 0)  # (FT, D)
    mask = torch.cat([m[0] for m in stash['masks']], 0).bool()             # (FT,)
    pts = torch.cat(stash['points'], 0)                                     # (FT, 4)
    level = torch.cat([torch.full((p.shape[0],), l, dtype=torch.long)
                       for l, p in enumerate(stash['points'])], 0)
    return feats, mask, pts, level


def collect_rows(feats, mask, pts, level, video_ix, seg_leaf, ph_leaf, seg_in,
                 ph_in, tn_in, segs_np, labels_np, unseen_np, cuts_np):
    """Rows for every valid location touched by either rule."""
    keep = mask.cpu().numpy() & ((seg_leaf >= 0) | (seg_in >= 0))
    idx = np.nonzero(keep)[0]
    t = pts[:, 0].cpu().numpy()
    out = {}
    out['feat'] = feats[idx].half().cpu().numpy()
    out['t'] = t[idx].astype(np.float32)
    out['level'] = level.numpy()[idx].astype(np.int8)
    out['video'] = np.full(len(idx), video_ix, dtype=np.int32)
    for name, seg, ph in (('leaf', seg_leaf, ph_leaf), ('in', seg_in, ph_in)):
        sj = seg[idx]
        ok = sj >= 0
        cls = np.where(ok, labels_np[np.maximum(sj, 0)], -1)
        uns = np.where(ok, unseen_np[np.maximum(sj, 0)], False)
        s = segs_np[np.maximum(sj, 0), 0]
        e = segs_np[np.maximum(sj, 0), 1]
        tn = np.where(ok, (t[idx] - s) / np.maximum(e - s, 1e-6), np.nan)
        out[f'{name}_seg'] = sj.astype(np.int32)
        out[f'{name}_phase'] = np.where(ok, ph[idx], -1).astype(np.int8)
        out[f'{name}_cls'] = cls.astype(np.int16)
        out[f'{name}_unseen'] = uns.astype(bool)
        out[f'{name}_tnorm'] = tn.astype(np.float32)
    return out


def cat_rows(rows):
    keys = rows[0].keys()
    return {k: np.concatenate([r[k] for r in rows], 0) for k in keys}


# ------------------------------------------------------------------- dumps
def dump_train(model, cfg, n, out_path, max_videos=-1):
    ds = make_dataset(cfg['dataset_name'], n, True, cfg['train_split'], **cfg['dataset'])
    loader = make_data_loader(ds, False, None, 1, cfg['loader']['num_workers'])
    n_cls = len(ds.label_dict)
    model.train()
    n_frozen = freeze_randomness(model)
    print(f"[train-dump] split {n}: {len(ds)} videos, {n_cls} seen classes, "
          f"{n_frozen} stochastic modules forced to eval", flush=True)
    rows, stats = [], {'videos': 0, 'segs': 0, 'segs_with_cuts': 0}
    det_check = None
    for vi, vl in enumerate(loader):
        item = vl[0]
        if item['segments'] is None:
            continue
        with torch.no_grad():
            model(vl)
            st = model._probe_stash
            if vi == 0:            # determinism check: same crop, twice
                f1 = torch.cat([f[0] for f in st['feats']], 1).clone()
                model(vl)
                st = model._probe_stash
                f2 = torch.cat([f[0] for f in st['feats']], 1)
                det_check = float((f1 - f2).abs().max().item())
        feats, mask, pts, level = unpack_stash(st)
        tg = st['leaf_targets'][0]
        inds, pos = st['leaf_assign'][0]
        seg_leaf = np.where(pos.cpu().numpy(), inds.cpu().numpy(), -1)
        ph_leaf = tg[:, 1].cpu().numpy()
        # TIFAD_PROBE_FEAT=ph (v5): the stash holds level 0 only; the leaf
        # targets are computed on the full pyramid, level 0 first
        seg_leaf, ph_leaf = seg_leaf[:len(mask)], ph_leaf[:len(mask)]
        segs_np = item['segments'].numpy().astype(np.float32)
        labs_np = item['labels'].numpy().astype(np.int64)
        cuts_np = item['leaf_cuts'].numpy().astype(np.float32)
        seg_in, ph_in, tn_in = assign_inside(pts[:, 0].cpu().numpy(), segs_np, cuts_np)
        rows.append(collect_rows(feats, mask, pts, level, vi, seg_leaf, ph_leaf,
                                 seg_in, ph_in, tn_in, segs_np, labs_np,
                                 np.zeros(len(labs_np), bool), cuts_np))
        stats['videos'] += 1
        if 0 < max_videos <= stats['videos']:
            break
        stats['segs'] += len(labs_np)
        stats['segs_with_cuts'] += int((~np.isnan(cuts_np).any(1)).sum())
    assert det_check is not None and det_check < 1e-3, \
        f"train-mode features are NOT deterministic (max |diff| {det_check})"
    data = cat_rows(rows)
    data['det_check_maxabs'] = np.float32(det_check)
    np.savez_compressed(out_path, **data)
    stats['rows'] = int(len(data['t']))
    stats['leaf_rows'] = int((data['leaf_phase'] >= 0).sum())
    stats['inside_rows'] = int((data['in_phase'] >= 0).sum())
    stats['det_check_maxabs'] = det_check
    print(f"[train-dump] {stats}", flush=True)
    return stats


def dump_test(model, cfg, n, pseudo, seen, unseen, out_path, max_videos=-1):
    ds = make_dataset(cfg['dataset_name'], n, False, cfg['val_split'], **cfg['dataset'])
    all_cls = list(seen) + list(unseen)
    # the fork's test dataset keeps only videos with >= 1 UNSEEN-class
    # segment; the probe needs every annotated test video (seen-class test
    # segments live mostly in the others), so rebuild the video list with
    # the full 20-class vocabulary.  Only the item feats / timing meta are
    # used from the loader; GT comes from the annotation json below.
    ds.label_dict = None
    ds.subset_labels = all_cls
    ds.data_list, ds.label_dict = ds._load_json_db(ds.json_file)
    loader = make_data_loader(ds, False, None, 1, cfg['loader']['num_workers'])
    with open(cfg['dataset']['json_file'], encoding='utf-8') as f:
        db = json.load(f)['database']
    cls_id = {c: i for i, c in enumerate(all_cls)}
    unseen_set = set(unseen)
    model.eval()
    # speed only: decoded detections are never used
    model.test_pre_nms_thresh = 2.0
    model.test_nms_method = 'none'
    print(f"[test-dump] split {n}: {len(ds)} test videos", flush=True)
    rows = []
    stats = {'videos': 0, 'seen_segs': 0, 'seen_with_cuts': 0, 'unseen_segs': 0,
             'dropped_labels': 0}
    for vi, vl in enumerate(loader):
        item = vl[0]
        vid = item['video_id']
        fps, fstr, nfr = item['fps'], item['feat_stride'], item['feat_num_frames']
        anns = db[vid].get('annotations', [])
        pe = pseudo.get(vid, [])
        segs, labs, uns, cuts = [], [], [], []
        for a in anns:
            lab = a['label']
            if lab not in cls_id:
                stats['dropped_labels'] += 1
                continue
            s, e = float(a['segment'][0]), float(a['segment'][1])
            c = np.full(9, np.nan, np.float32)
            if lab in unseen_set:
                stats['unseen_segs'] += 1
            else:
                stats['seen_segs'] += 1
                for p in pe:
                    if (p['label'] == lab and abs(float(p['segment'][0]) - s) <= 1e-3
                            and abs(float(p['segment'][1]) - e) <= 1e-3):
                        c = phase_cuts_grid(p, fps, fstr, nfr)
                        stats['seen_with_cuts'] += 1
                        break
            segs.append([(s * fps - 0.5 * nfr) / fstr, (e * fps - 0.5 * nfr) / fstr])
            labs.append(cls_id[lab])
            uns.append(lab in unseen_set)
            cuts.append(c)
        if not segs:
            continue
        segs_np = np.asarray(segs, np.float32)
        labs_np = np.asarray(labs, np.int64)
        uns_np = np.asarray(uns, bool)
        cuts_np = np.stack(cuts, 0).astype(np.float32)
        with torch.no_grad():
            model(vl)
            st = model._probe_stash
            feats, mask, pts, level = unpack_stash(st)
            seg_leaf, ph_leaf = leaf_rule(
                model, pts.to(feats.device),
                torch.from_numpy(segs_np).to(feats.device),
                torch.from_numpy(labs_np).to(feats.device),
                torch.from_numpy(cuts_np).to(feats.device), len(all_cls))
        seg_in, ph_in, tn_in = assign_inside(pts[:, 0].cpu().numpy(), segs_np, cuts_np)
        rows.append(collect_rows(feats, mask, pts, level, vi, seg_leaf, ph_leaf,
                                 seg_in, ph_in, tn_in, segs_np, labs_np, uns_np, cuts_np))
        stats['videos'] += 1
        if 0 < max_videos <= stats['videos']:
            break
    data = cat_rows(rows)
    np.savez_compressed(out_path, **data)
    stats['rows'] = int(len(data['t']))
    print(f"[test-dump] {stats}", flush=True)
    return stats


# ------------------------------------------------------------------- probe
class Head(nn.Module):
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(inplace=True),
                                 nn.Linear(d_hid, N_PHASE))

    def forward(self, x):
        return self.net(x)


def train_head(x, y, seed, d_hid, epochs=30, bs=4096, lr=1e-3, device='cuda'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    head = Head(x.shape[1], d_hid).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = x.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(head(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head, x, device='cuda'):
    out = []
    for i in range(0, len(x), 65536):
        lg = head(torch.as_tensor(x[i:i + 65536], dtype=torch.float32, device=device))
        out.append(torch.softmax(lg, -1).cpu().numpy())
    p = np.concatenate(out, 0)
    return p.argmax(1), (p * np.arange(N_PHASE)[None]).sum(1)


def spearman_metrics(pred_arg, pred_exp, tnorm, seg_key, min_pts=5):
    """Per-segment Spearman (segments with >= min_pts locations; a constant
    prediction gives an undefined rho, counted as 0) + pooled."""
    res = {}
    flat = seg_key[:, 0].astype(np.int64) * 1000003 + seg_key[:, 1].astype(np.int64)
    order = np.argsort(flat, kind='stable')
    _, starts = np.unique(flat[order], return_index=True)
    groups = np.split(order, starts[1:])
    for name, pred in (('argmax', pred_arg), ('expect', pred_exp)):
        rhos, n_const = [], 0
        for m in groups:
            if len(m) < min_pts:
                continue
            if np.ptp(pred[m]) == 0 or np.ptp(tnorm[m]) == 0:
                n_const += 1
                rhos.append(0.0)
                continue
            rhos.append(float(spearmanr(pred[m], tnorm[m]).correlation))
        pooled = float(spearmanr(pred, tnorm).correlation) if np.ptp(pred) > 0 else 0.0
        res[name] = {'per_seg_mean': float(np.mean(rhos)) if rhos else float('nan'),
                     'per_seg_median': float(np.median(rhos)) if rhos else float('nan'),
                     'n_segs': len(rhos), 'n_const_segs': n_const, 'pooled': pooled}
    return res


def acc_metrics(pred_arg, y):
    a10 = float((pred_arg == y).mean())
    a3 = float((MERGE3[pred_arg] == MERGE3[y]).mean())
    maj10 = float(np.bincount(y, minlength=N_PHASE).max() / len(y))
    maj3 = float(np.bincount(MERGE3[y], minlength=3).max() / len(y))
    return {'top1_10way': a10, 'top1_3way': a3, 'majority_10way': maj10,
            'majority_3way': maj3, 'n': int(len(y))}


def eval_sets(test, rule):
    """Slices of the test dump under a rule: seen (phase-labeled) and unseen."""
    ph, tn, uns, seg, vid = (test[f'{rule}_phase'], test[f'{rule}_tnorm'],
                             test[f'{rule}_unseen'], test[f'{rule}_seg'], test['video'])
    seen = (ph >= 0) & ~uns
    unseen = (seg >= 0) & uns
    key = np.stack([vid, seg], 1)
    return {'seen': (seen, ph, tn, key), 'unseen': (unseen, ph, tn, key)}


def run_probe(train, tests, seeds, d_hid, device):
    """train: train dump; tests: {vocab_name: test dump}.  Returns nested
    results dict keyed train_<rule> / variant / <test_rule>@<vocab>."""
    X = train['feat'].astype(np.float32)
    results = {}
    # training sets: 'leaf' = LEAFLOSS positives (task spec), 'in' = inside
    for tr_rule in ('leaf', 'in'):
        m = train[f'{tr_rule}_phase'] >= 0
        Xtr, ytr, ttr = X[m], train[f'{tr_rule}_phase'][m].astype(np.int64), \
            train[f'{tr_rule}_tnorm'][m].astype(np.float32)[:, None]
        hist = np.bincount(ytr, minlength=N_PHASE).tolist()
        lvl = np.bincount(train['level'][m].astype(np.int64), minlength=6).tolist()
        print(f"  [probe] train rule={tr_rule}: {len(ytr)} rows, phase hist {hist}, "
              f"level hist {lvl}", flush=True)
        variants = {
            'feature': (Xtr, ytr, lambda te, msk: te['feat'][msk].astype(np.float32)),
            'position_only': (ttr, ytr, None),
            'chance_shuffled': (Xtr, None, lambda te, msk: te['feat'][msk].astype(np.float32)),
        }
        out = {'n_train': int(len(ytr)), 'train_phase_hist': hist, 'train_level_hist': lvl}
        for vname, (xin, yin, xfn) in variants.items():
            per_seed = []
            for seed in seeds:
                yy = yin
                if yy is None:                      # shuffled labels
                    yy = np.random.RandomState(seed).permutation(ytr)
                head = train_head(xin, yy, seed, d_hid, device=device)
                r = {}
                tr_arg, _ = predict(head, xin, device)
                r['train'] = {'top1_10way': float((tr_arg == yy).mean())}
                for vocab, test in tests.items():
                    for te_rule in ('leaf', 'in'):
                        sets = eval_sets(test, te_rule)
                        key_r = f'{te_rule}@{vocab}'
                        r[key_r] = {}
                        for sname, (msk, ph, tn, key) in sets.items():
                            if msk.sum() == 0:
                                continue
                            xte = (tn[msk].astype(np.float32)[:, None] if xfn is None
                                   else xfn(test, msk))
                            pa, pe = predict(head, xte, device)
                            rr = {}
                            if sname == 'seen':
                                rr.update(acc_metrics(pa, ph[msk].astype(np.int64)))
                            rr['spearman'] = spearman_metrics(pa, pe, tn[msk], key[msk])
                            rr['n'] = int(msk.sum())
                            r[key_r][sname] = rr
                per_seed.append(r)
            out[vname] = average_dicts(per_seed)
        results[f'train_{tr_rule}'] = out
    return results


def average_dicts(ds):
    """Mean over a list of nested dicts of numbers (ints kept if equal)."""
    if isinstance(ds[0], dict):
        return {k: average_dicts([d[k] for d in ds if k in d]) for k in ds[0]}
    if isinstance(ds[0], (list, tuple)):
        return np.mean(np.asarray(ds, dtype=np.float64), axis=0).round(1).tolist()
    if isinstance(ds[0], str):
        return ds[0]
    vals = [d for d in ds if d is not None]
    if all(isinstance(v, (int, np.integer)) for v in vals) and len(set(vals)) == 1:
        return int(vals[0])
    return float(np.mean([float(v) for v in vals]))


# ------------------------------------------------------------------ report
def fmt_split(r):
    lines = []
    for trk in ('train_leaf', 'train_in'):
        rr = r[trk]
        lines.append(f"  {trk}: n_train={rr['n_train']} hist={rr['train_phase_hist']}")
        for v in ('feature', 'position_only', 'chance_shuffled'):
            for tek in [k for k in rr[v] if k != 'train']:
                s = rr[v].get(tek, {})
                if 'seen' in s:
                    a = s['seen']
                    sp = a['spearman']
                    lines.append(
                        f"    {v:16s} test={tek:14s} SEEN   n={a['n']:8.0f} "
                        f"acc10={a['top1_10way']:.3f} (maj {a['majority_10way']:.3f}) "
                        f"acc3={a['top1_3way']:.3f} (maj {a['majority_3way']:.3f}) "
                        f"rho_seg[arg/exp]={sp['argmax']['per_seg_mean']:.3f}/"
                        f"{sp['expect']['per_seg_mean']:.3f} "
                        f"pooled={sp['argmax']['pooled']:.3f}/{sp['expect']['pooled']:.3f} "
                        f"(segs {sp['argmax']['n_segs']:.0f})")
                if 'unseen' in s:
                    a = s['unseen']
                    sp = a['spearman']
                    lines.append(
                        f"    {v:16s} test={tek:14s} UNSEEN n={a['n']:8.0f} "
                        f"rho_seg[arg/exp]={sp['argmax']['per_seg_mean']:.3f}/"
                        f"{sp['expect']['per_seg_mean']:.3f} "
                        f"pooled={sp['argmax']['pooled']:.3f}/{sp['expect']['pooled']:.3f} "
                        f"(segs {sp['argmax']['n_segs']:.0f})")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--ckpt', default='ckpt/thumos14_i3d_leafref_{n}/mAP_best.pth.tar')
    ap.add_argument('--pseudo', default='data/phase_pseudo/thumos14_75_25_split{n}.json')
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--dump-dir', default='data/phase_pseudo/probe_dump')
    ap.add_argument('--out', default='data/phase_pseudo/probe_results.json')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--max-videos', type=int, default=-1, help='debug: cap videos per dump')
    ap.add_argument('--report-only', action='store_true',
                    help='recompute the 3-split mean + report from the existing results json')
    ap.add_argument('--skip-dump', action='store_true',
                    help='reuse existing dumps, only (re)train the heads')
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)
    cfg = load_config(args.config)
    device = 'cuda'
    all_res = {'meta': {'envs': REF_ENVS, 'config': args.config, 'ckpt': args.ckpt,
                        'head': 'Linear(D,D)-ReLU-Linear(D,10), Adam 1e-3, 30 ep, bs 4096',
                        'seeds': args.seeds, 'rules': {
                            'leaf': 'detector positive rule (center sampling r=1.5*stride '
                                    '+ regression range) = LEAFLOSS positives',
                            'in': 'every location of every level inside the GT segment '
                                  '(shortest on ties)'},
                        'spearman': 'per segment (>=5 locations; constant prediction -> 0) '
                                    'and pooled; argmax and E[k]',
                        'test_vocab': {'testvocab': 'eval-time condition: features conditioned '
                                                    'on the unseen (test) class text',
                                       'seenvocab': 'control: same test videos, features '
                                                    'conditioned on the seen class text'}},
               'splits': {}}
    if args.report_only:
        with open(args.out, encoding='utf-8') as f:
            all_res = json.load(f)
        for n in all_res['splits']:
            print(f"--- split {n}\n" + fmt_split(all_res['splits'][n]), flush=True)
    for n in ([] if args.report_only else args.splits):
        t0 = time.time()
        print(f"===== split {n} =====", flush=True)
        tr_path = os.path.join(args.dump_dir, f'split{n}_train.npz')
        te_path = os.path.join(args.dump_dir, f'split{n}_test.npz')
        te2_path = os.path.join(args.dump_dir, f'split{n}_test_seenvocab.npz')
        meta = {}
        if not args.skip_dump or not all(os.path.exists(p) for p in (tr_path, te_path, te2_path)):
            fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
            seen = load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'train', n)
            unseen = load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'test', n)
            model, notes = build_model(cfg, n, args.ckpt.format(n=n))
            print(f"[ckpt] {args.ckpt.format(n=n)} loaded; notes={notes}", flush=True)
            meta['ckpt_notes'] = notes
            meta['seen'] = seen
            meta['unseen'] = unseen
            with open(args.pseudo.format(n=n), encoding='utf-8') as f:
                pseudo = json.load(f)['videos']
            meta['train_dump'] = dump_train(model, cfg, n, tr_path, args.max_videos)
            meta['test_dump'] = dump_test(model, cfg, n, pseudo, seen, unseen, te_path, args.max_videos)
            # control: the SAME test videos with the features conditioned on
            # the SEEN vocabulary (TIFAD_PROTO_COLLECT=1 forces mode='train'
            # text in eval), isolating the vocabulary shift from generalization
            os.environ['TIFAD_PROTO_COLLECT'] = '1'
            meta['test_dump_seenvocab'] = dump_test(model, cfg, n, pseudo, seen, unseen,
                                                    te2_path, args.max_videos)
            os.environ.pop('TIFAD_PROTO_COLLECT', None)
            del model
            torch.cuda.empty_cache()
        train = dict(np.load(tr_path))
        tests = {'testvocab': dict(np.load(te_path)), 'seenvocab': dict(np.load(te2_path))}
        d_hid = train['feat'].shape[1]
        res = run_probe(train, tests, args.seeds, d_hid, device)
        res['meta'] = meta
        all_res['splits'][str(n)] = res
        print(f"--- split {n} ({time.time() - t0:.0f}s)\n" + fmt_split(res), flush=True)
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
    # 3-split mean
    keys = [k for k in all_res['splits']]
    if len(keys) > 1:
        all_res['mean'] = average_dicts([
            {k: v for k, v in all_res['splits'][s].items() if k != 'meta'} for s in keys])
        print(f"===== mean over splits {keys} =====\n" + fmt_split(all_res['mean']), flush=True)
    with open(args.out, 'w') as f:
        json.dump(all_res, f, indent=1)
    print(f"[save] {args.out}")


if __name__ == '__main__':
    main()
