"""REPRESENTATION VALIDATION for an arbitrary checkpoint: the representation
metrics of rep_metrics.py on other checkpoints for a fair comparison (Ti-FAD
baseline, PDA reproduction, and a full-supervision ceiling).

WHY NOT scripts/phase_probe.py's dump_train/dump_test directly: those need
TIFAD_LEAFLOSS=1 (+ TIFAD_PROMPT_ENSEMBLE, phase-sentence json) at MODEL
construction time to expose per-location phase targets. TIFAD_PROMPT_ENSEMBLE
is explicitly incompatible with TIFAD_PDA (text.py asserts against it), so
the champion's own dump machinery cannot build a PDA-architecture probe dump.
This script computes phase labels a different way that needs no model-side
LEAFLOSS at all: labels come straight from the dataset's raw GT segments +
the text-aligned pseudo-phase json (data/phase_pseudo/thumos14_75_25_split*.
json), matched by (video id, class name, segment seconds) -- exactly the
matching libs/datasets/thumos14.py's LEAFLOSS path performs, but done here in
numpy, independent of the model. This lets the SAME code probe every
architecture (Ti-FAD baseline, PDA, the full HP+ASA+AS champion recipe).

Reuses, as templates:
  - scripts/phase_probe.py: build_model, unpack_stash, assign_inside,
    phase_cuts_grid, freeze_randomness, train_head, predict (unmodified).
  - scripts/phase_probe_raw.py: recover_offset (unmodified) -- TRAIN videos
    are randomly cropped (truncate_feats); recover_offset finds the crop
    offset `st` per video from the dump's OWN rows, exactly as it does for
    the champion's raw-I3D probe, so item['segments'] need not be trusted
    post-crop. tnorm is crop-invariant by construction ((t-s)/(e-s) is
    unchanged by a common shift), but the PHASE label needs one: once `st`
    is known, t_raw = t + st puts every row back in the same (uncropped)
    grid as the dataset's raw GT segments, and assign_inside on the raw
    segments/cuts gives the correct phase directly (no per-segment
    crop<->raw index bookkeeping needed).
  - scripts/rep_metrics.py: train_linear / seg_groups / align_metrics
    (Kendall's-tau TCC metric) -- copied verbatim rather than imported:
    rep_metrics.py sets TIFAD_ATTNDESIGN=phasechain via os.environ.setdefault
    at import time, which would leak into a non-champion architecture if
    imported after this script's own envs are set.

For each split n:
  1. set envs (clear all TIFAD_*, then TIFAD_PROBE_STASH=1 [trunk-feature
     stash; unconditional, no LEAFLOSS/ensemble dependency],
     TIFAD_PROMPT_DROPOUT=0, TIFAD_SEM_DEGRADE=0 [deterministic features],
     then the caller's --envs architecture string, e.g. "" for the Ti-FAD
     baseline or "TIFAD_PDA=1 TIFAD_PDA_TIF=off" for the PDA reproduction).
  2. build the model from --ckpt (build_model's strict state-dict load
     fails loudly if envs/ckpt architecture disagree).
  3. dump_train_safe / dump_test_safe -> data/phase_pseudo/
     probe_dump_variants/<tag>/split{n}_{train,test}.npz (same row
     structure as the champion's own dumps: level, video, t, in_seg,
     in_phase, in_tnorm, in_unseen, feat).
  4. probe accuracy: Linear(D,D)-ReLU-Linear(D,10), Adam 1e-3, 30 ep, bs
     4096, 3 seeds, trunk feature, pseudo-phase target, 'in' rule --
     identical protocol to phase_probe_raw.py's trunk_l0/pseudo_phase row.
  5. Kendall's tau: linear regressor feature->t_norm (Adam 3e-3, 100 ep, bs
     4096, standardized features), 3 seeds, per-segment (>=5 locations)
     Kendall tau vs true time order -- identical protocol to
     rep_metrics.py's TCC metric (2).
  6. shuffled-label / shuffled-feature controls for both metrics.

--full-vocab (the full-supervision ceiling, no held-out unseen classes):
  unseen = [], every row falls in a single 'all' group instead of
  seen/unseen.

Usage:
  cd /data3/xiaodan8/tifad
  CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 LC_ALL=C.UTF-8 python \
      scripts/repvalid_variant.py --tag vanilla \
      --ckpt 'ckpt/thumos14_i3d_vanilla_{n}/mAP_best.pth.tar' \
      --config configs/thumos14_i3d.yaml --envs "" --splits 0 1 2
Output: data/rep_metrics/<tag>.json

Post-video-ASA probe (vsa3 checkpoints, TIFAD_VSA=1): the stash reads
meta_archs' cls_feat, which with VSA on is the pyramid AFTER the video
subtree attention; pass TIFAD_PROBE_STASH_POST=1 in --envs to make that
explicit, or TIFAD_PROBE_STASH_POST=0 to probe the PRE-VSA trunk of the
same checkpoint (see the PROBE STASH block in libs/modeling/meta_archs.py):
  CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 LC_ALL=C.UTF-8 python \
      scripts/repvalid_variant.py --tag vsa3_post \
      --ckpt 'ckpt/thumos14_i3d_vsa3_{n}/mAP_best.pth.tar' \
      --envs "TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json \
              TIFAD_ATTR_ATTN=1 TIFAD_ATTNDESIGN=phasechain TIFAD_FAMCE=0.3 \
              TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 \
              TIFAD_VSA=1 TIFAD_PROBE_STASH_POST=1" --splits 0 1 2
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

import phase_probe as pp                # import first; its REF_ENVS import
import phase_probe_raw as pr            # side effect is overridden below
from libs.datasets import make_dataset, make_data_loader

N_PHASE = pp.N_PHASE
MIN_PTS = 5
OUT_DIR = 'data/rep_metrics'
DUMP_ROOT = 'data/phase_pseudo/probe_dump_variants'


# --------------------------------------------------------------- env setup
def set_envs(env_str):
    for k in list(os.environ):
        if k.startswith('TIFAD_'):
            del os.environ[k]
    os.environ['TIFAD_PROMPT_DROPOUT'] = '0'   # deterministic probe features
    os.environ['TIFAD_SEM_DEGRADE'] = '0'
    os.environ['TIFAD_PROBE_STASH'] = '1'      # unconditional trunk stash
    if env_str:
        for kv in env_str.split():
            k, v = kv.split('=', 1)
            os.environ[k] = v
    return {k: v for k, v in os.environ.items() if k.startswith('TIFAD_')}


# --------------------------------------------------------- pseudo matching
def match_pseudo_entry(pseudo, vid, label_name, s_sec, e_sec, tol=1e-3):
    for e in pseudo.get(vid, []):
        if (e['label'] == label_name and abs(float(e['segment'][0]) - s_sec) <= tol
                and abs(float(e['segment'][1]) - e_sec) <= tol):
            return e
    return None


# ------------------------------------------------------------- dump: train
def dump_train_safe(model, cfg, n, pseudo, out_path, feat_folder):
    ds = make_dataset(cfg['dataset_name'], n, True, cfg['train_split'], **cfg['dataset'])
    loader = make_data_loader(ds, False, None, 1, cfg['loader']['num_workers'])
    id2name = {v: k for k, v in ds.label_dict.items()}
    fstr = cfg['dataset']['feat_stride']
    nfr = cfg['dataset']['num_frames']
    model.train()
    pp.freeze_randomness(model)
    rows = []
    stats = {'videos': 0, 'skipped_no_offset': 0}
    for vi, vl in enumerate(loader):
        item = vl[0]
        if item['segments'] is None:
            continue
        with torch.no_grad():
            model(vl)
        st_dict = model._probe_stash
        feats, mask, pts, level = pp.unpack_stash(st_dict)
        t_crop = pts[:, 0].cpu().numpy()

        raw_item = ds.data_list[vi]
        fps = raw_item['fps']
        raw_segs = (raw_item['segments'] * fps - 0.5 * nfr) / fstr    # (N,2) grid
        T_raw = int(np.load(os.path.join(feat_folder, raw_item['id'] + '.npy'),
                            mmap_mode='r').shape[0])

        segs_np = item['segments'].numpy().astype(np.float32)
        dummy_cuts = np.full((len(segs_np), 9), np.nan, np.float32)
        seg_in_c, _, tn_in_c = pp.assign_inside(t_crop, segs_np, dummy_cuts)
        d_local = {'video': np.zeros(len(t_crop), np.int32),
                   'level': level.numpy().astype(np.int8),
                   'in_seg': seg_in_c.astype(np.int32),
                   't': t_crop.astype(np.float64),
                   'in_tnorm': tn_in_c.astype(np.float64)}
        st, ne, nc, nb, ns = pr.recover_offset(d_local, 0, raw_segs, T_raw)
        if st is None:
            stats['skipped_no_offset'] += 1
            continue

        t_raw = t_crop + st
        raw_cuts = np.full((len(raw_segs), 9), np.nan, np.float32)
        for i, lab in enumerate(raw_item['labels']):
            lab_name = id2name.get(int(lab))
            s_sec, e_sec = raw_item['segments'][i]
            entry = match_pseudo_entry(pseudo, raw_item['id'], lab_name,
                                       float(s_sec), float(e_sec))
            if entry is not None:
                raw_cuts[i] = pp.phase_cuts_grid(entry, fps, fstr, nfr)

        seg_idx, phase, tnorm = pp.assign_inside(t_raw, raw_segs, raw_cuts)
        keep = mask.cpu().numpy() & (seg_idx >= 0)
        idx = np.nonzero(keep)[0]
        rows.append({'feat': feats[idx].half().cpu().numpy(),
                     't': t_crop[idx].astype(np.float32),
                     'level': level.numpy()[idx].astype(np.int8),
                     'video': np.full(len(idx), vi, np.int32),
                     'in_seg': seg_idx[idx].astype(np.int32),
                     'in_phase': phase[idx].astype(np.int8),
                     'in_tnorm': tnorm[idx].astype(np.float32)})
        stats['videos'] += 1
    data = {k: np.concatenate([r[k] for r in rows], 0) for k in rows[0]}
    np.savez_compressed(out_path, **data)
    stats['n_rows'] = int(len(data['t']))
    print(f"  [train-dump] {stats}", flush=True)
    return stats


# -------------------------------------------------------------- dump: test
def dump_test_safe(model, cfg, n, pseudo, pseudo_unseen, seen, unseen, out_path):
    ds = make_dataset(cfg['dataset_name'], n, False, cfg['val_split'], **cfg['dataset'])
    all_cls = list(seen) + list(unseen)
    ds.label_dict = None
    ds.subset_labels = all_cls
    ds.data_list, ds.label_dict = ds._load_json_db(ds.json_file)
    loader = make_data_loader(ds, False, None, 1, cfg['loader']['num_workers'])
    with open(cfg['dataset']['json_file'], encoding='utf-8') as f:
        db = json.load(f)['database']
    cls_id = {c: i for i, c in enumerate(all_cls)}
    unseen_set = set(unseen)
    model.eval()
    model.test_pre_nms_thresh = 2.0
    model.test_nms_method = 'none'
    rows = []
    stats = {'videos': 0, 'seen_segs': 0, 'unseen_segs': 0}
    for vi, vl in enumerate(loader):
        item = vl[0]
        vid = item['video_id']
        fps, fstr, nfr = item['fps'], item['feat_stride'], item['feat_num_frames']
        segs, labs, uns, cuts = [], [], [], []
        for a in db[vid].get('annotations', []):
            lab = a['label']
            if lab not in cls_id:
                continue
            s, e = float(a['segment'][0]), float(a['segment'][1])
            c = np.full(9, np.nan, np.float32)
            src = pseudo_unseen if lab in unseen_set else pseudo
            entry = match_pseudo_entry(src, vid, lab, s, e)
            if entry is not None:
                c = pp.phase_cuts_grid(entry, fps, fstr, nfr)
            segs.append([(s * fps - 0.5 * nfr) / fstr, (e * fps - 0.5 * nfr) / fstr])
            labs.append(cls_id[lab])
            uns.append(lab in unseen_set)
            cuts.append(c)
            if lab in unseen_set:
                stats['unseen_segs'] += 1
            else:
                stats['seen_segs'] += 1
        if not segs:
            continue
        segs_np = np.asarray(segs, np.float32)
        cuts_np = np.stack(cuts, 0).astype(np.float32)
        with torch.no_grad():
            model(vl)
        st_dict = model._probe_stash
        feats, mask, pts, level = pp.unpack_stash(st_dict)
        t = pts[:, 0].cpu().numpy()
        seg_idx, phase, tnorm = pp.assign_inside(t, segs_np, cuts_np)
        keep = mask.cpu().numpy() & (seg_idx >= 0)
        idx = np.nonzero(keep)[0]
        uns_np = np.asarray(uns, bool)
        rows.append({'feat': feats[idx].half().cpu().numpy(),
                     't': t[idx].astype(np.float32),
                     'level': level.numpy()[idx].astype(np.int8),
                     'video': np.full(len(idx), vi, np.int32),
                     'in_seg': seg_idx[idx].astype(np.int32),
                     'in_phase': phase[idx].astype(np.int8),
                     'in_tnorm': tnorm[idx].astype(np.float32),
                     'in_unseen': uns_np[seg_idx[idx]]})
        stats['videos'] += 1
    data = {k: np.concatenate([r[k] for r in rows], 0) for k in rows[0]}
    np.savez_compressed(out_path, **data)
    stats['n_rows'] = int(len(data['t']))
    print(f"  [test-dump] {stats}", flush=True)
    return stats


# --------------------------------------------------------- (2) Kendall tau
def train_linear(X, y, seed, epochs=100, bs=4096, lr=3e-3):
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
            loss = F.mse_loss(lin(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    W = lin.weight.detach().numpy()[0]
    b = float(lin.bias.detach().numpy()[0])
    return lambda Z: Z @ W + b


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
    return {'r2': r2, 'spearman': rho,
            'kendall_tau': float(np.mean(taus)) if taus else float('nan'),
            'n_rows': int(len(pred)), 'n_segs_tau': len(taus)}


def tcc_metrics(Xtr, ttr, Xte, tnte, tte, groups_by_group, seeds):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    out = {}
    for variant in ('feature', 'shuffled'):
        per_seed = {k: [] for k in groups_by_group}
        for seed in seeds:
            X_in = Xtr_n if variant == 'feature' else \
                Xtr_n[np.random.RandomState(seed).permutation(len(Xtr_n))]
            f = train_linear(X_in, ttr, seed)
            pred = f(Xte_n)
            for k, (m, groups) in groups_by_group.items():
                per_seed[k].append(align_metrics(pred[m], tnte[m], tte[m], groups))
        out[variant] = {k: {kk: float(np.mean([r[kk] for r in rs])) for kk in rs[0]}
                        for k, rs in per_seed.items()}
    return out


# --------------------------------------------------------- (1) probe acc10
def probe_accuracy(Xtr, ytr, Xte, phase_te, groups_masks, seeds, device):
    d_hid = Xtr.shape[1]
    out = {'feature': {k: [] for k in groups_masks}, 'shuffled': {k: [] for k in groups_masks}}
    for variant in ('feature', 'shuffled'):
        for seed in seeds:
            y = ytr if variant == 'feature' else np.random.RandomState(seed).permutation(ytr)
            head = pp.train_head(Xtr, y, seed, d_hid, device=device)
            pa, _ = pp.predict(head, Xte, device)
            for k, m in groups_masks.items():
                ok = m & (phase_te >= 0)
                out[variant][k].append(float((pa[ok] == phase_te[ok]).mean()) if ok.sum() else float('nan'))
            del head
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
    return {v: {k: float(np.nanmean(vs)) for k, vs in gg.items()} for v, gg in out.items()}


# ------------------------------------------------------------------- split
def gather(dump_dir, n):
    dtr = dict(np.load(os.path.join(dump_dir, f'split{n}_train.npz')))
    dte = dict(np.load(os.path.join(dump_dir, f'split{n}_test.npz')))
    tr_m = (dtr['level'] == 0) & (dtr['in_phase'] >= 0)
    Xtr = dtr['feat'][tr_m].astype(np.float32)
    ytr = dtr['in_phase'][tr_m].astype(np.int64)
    ttr = dtr['in_tnorm'][tr_m].astype(np.float32)

    te_m = (dte['level'] == 0) & (dte['in_seg'] >= 0)
    Xte = dte['feat'][te_m].astype(np.float32)
    te_t = dte['t'][te_m].astype(np.float32)
    te_tnorm = dte['in_tnorm'][te_m].astype(np.float32)
    te_unseen = dte['in_unseen'][te_m]
    te_video = dte['video'][te_m]
    te_seg = dte['in_seg'][te_m]
    phase = dte['in_phase'][te_m].astype(np.int64)
    return {'Xtr': Xtr, 'ytr': ytr, 'ttr': ttr, 'Xte': Xte, 'te_t': te_t,
            'te_tnorm': te_tnorm, 'te_unseen': te_unseen, 'te_video': te_video,
            'te_seg': te_seg, 'phase': phase,
            'n_train_rows': int(tr_m.sum()), 'n_test_rows': int(te_m.sum())}


def run_split(n, args, device):
    from libs.core import load_config
    cfg = load_config(args.config)
    dump_dir = args.dump_dir or os.path.join(DUMP_ROOT, args.tag)
    os.makedirs(dump_dir, exist_ok=True)
    tr_path = os.path.join(dump_dir, f'split{n}_train.npz')
    te_path = os.path.join(dump_dir, f'split{n}_test.npz')

    seen = pp.load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'train', n)
    unseen = [] if args.full_vocab else \
        pp.load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'test', n)

    with open(args.pseudo.format(n=n), encoding='utf-8') as f:
        pseudo = json.load(f)['videos']
    pseudo_unseen = {}
    if not args.full_vocab:
        with open(args.pseudo_unseen.format(n=n), encoding='utf-8') as f:
            pseudo_unseen = json.load(f)['videos']

    if not args.skip_dump or not (os.path.exists(tr_path) and os.path.exists(te_path)):
        from libs.utils import fix_random_seed
        fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
        model, notes = pp.build_model(cfg, n, args.ckpt.format(n=n))
        print(f"[ckpt] {args.ckpt.format(n=n)} loaded; notes={notes}", flush=True)
        dump_train_safe(model, cfg, n, pseudo, tr_path, cfg['dataset']['feat_folder'])
        dump_test_safe(model, cfg, n, pseudo, pseudo_unseen, seen, unseen, te_path)
        del model
        torch.cuda.empty_cache()

    g = gather(dump_dir, n)

    if args.full_vocab:
        all_m = np.ones(len(g['Xte']), bool)
        groups_masks = {'all': all_m}
        groups_by_group = {'all': (all_m, seg_groups(g['te_video'], g['te_seg']))}
    else:
        seen_m = ~g['te_unseen']
        uns_m = g['te_unseen']
        groups_masks = {'seen': seen_m, 'unseen': uns_m}
        groups_by_group = {
            'seen': (seen_m, seg_groups(g['te_video'][seen_m], g['te_seg'][seen_m])),
            'unseen': (uns_m, seg_groups(g['te_video'][uns_m], g['te_seg'][uns_m])),
        }

    acc = probe_accuracy(g['Xtr'], g['ytr'], g['Xte'], g['phase'], groups_masks,
                         args.seeds, device)
    tau = tcc_metrics(g['Xtr'], g['ttr'], g['Xte'], g['te_tnorm'], g['te_t'],
                      groups_by_group, args.seeds)
    return {'acc10': acc, 'tcc': tau, 'seen': list(seen), 'unseen': list(unseen),
            'n_train_rows': g['n_train_rows'], 'n_test_rows': g['n_test_rows']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--envs', default='')
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--pseudo', default='data/phase_pseudo/thumos14_75_25_split{n}.json')
    ap.add_argument('--pseudo-unseen', default='data/phase_pseudo/thumos14_75_25_split{n}_unseen.json')
    ap.add_argument('--full-vocab', action='store_true')
    ap.add_argument('--dump-dir', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--skip-dump', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    envs = set_envs(args.envs)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, f'{args.tag}.json')
    all_res = {'meta': {'tag': args.tag, 'ckpt': args.ckpt, 'config': args.config,
                        'envs': envs, 'full_vocab': args.full_vocab,
                        'seeds': args.seeds}, 'splits': {}}
    for n in args.splits:
        t0 = time.time()
        print(f"===== {args.tag} split {n} =====", flush=True)
        res = run_split(n, args, device)
        all_res['splits'][str(n)] = res
        print(f"  acc10 feature={res['acc10']['feature']} shuffled={res['acc10']['shuffled']}",
              flush=True)
        print(f"  tau feature={ {k: v['kendall_tau'] for k, v in res['tcc']['feature'].items()} } "
              f"shuffled={ {k: v['kendall_tau'] for k, v in res['tcc']['shuffled'].items()} }",
              flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
        with open(out_path, 'w') as f:
            json.dump(all_res, f, indent=1)
    keys = list(all_res['splits'])
    if len(keys) > 1:
        groups = list(all_res['splits'][keys[0]]['acc10']['feature'].keys())
        mean = {'acc10': {v: {k: float(np.mean([all_res['splits'][s]['acc10'][v][k] for s in keys]))
                              for k in groups} for v in ('feature', 'shuffled')},
                'tcc': {v: {k: {'kendall_tau': float(np.mean(
                    [all_res['splits'][s]['tcc'][v][k]['kendall_tau'] for s in keys]))}
                    for k in groups} for v in ('feature', 'shuffled')}}
        all_res['mean'] = mean
        print(f"===== {args.tag} mean over splits {keys} =====")
        print(f"  acc10 feature={mean['acc10']['feature']} shuffled={mean['acc10']['shuffled']}")
        print(f"  tau feature={ {k: v['kendall_tau'] for k, v in mean['tcc']['feature'].items()} } "
              f"shuffled={ {k: v['kendall_tau'] for k, v in mean['tcc']['shuffled'].items()} }")
    with open(out_path, 'w') as f:
        json.dump(all_res, f, indent=1)
    print(f"[save] {out_path}")


if __name__ == '__main__':
    main()
