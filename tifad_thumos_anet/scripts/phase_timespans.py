"""PHASE TIMESPAN ACCURACY of the post-hoc phase probe (eval-only, THUMOS14 75/25).

scripts/phase_probe.py showed that the frozen detector's per-location
cls-trunk features encode the temporal PHASE (0..9) of the action they sit
in (per-location label accuracy / Spearman).  This script asks the segment-
level question: decoded into phase SEGMENTS, how close are the predicted
phase boundaries to the text-aligned pseudo-timespans?

Per split n (cached dumps data/phase_pseudo/probe_dump/split{n}_*.npz):
  1. Train the probe head of phase_probe.py (Linear(D,D)-ReLU-Linear(D,10),
     3 seeds) on the TRAIN-video features ('leaf' = detector positives and
     'in' = every inside location, as in the probe), plus the position-only
     head (input = normalized time) = the ceiling the pseudo labels allow.
  2. For every held-out SEEN-class test segment that has pseudo phase cuts,
     take its level-0 locations (feature stride 4 frames; 'in' rule), sort
     them by time, and decode the head's per-location log-posteriors with a
     left-to-right Viterbi over the 10 phases (transitions k->k or k->k+1,
     start in phase 0, end in phase 9; segments with < 10 level-0 locations
     cannot host all phases and are excluded, counted).  The boundary between
     phases k and k+1 is the midpoint between the last location of phase k
     and the first of phase k+1.
  3. Compare the 9 predicted boundaries with the 9 pseudo cuts of the
     segment: mean |error| in seconds and as a fraction of the segment
     length, and the fraction of boundaries within 10 % of the segment
     length.  Controls: equal-width bins (chance; boundary k at s+(e-s)k/10)
     and the position-only head decoded the same way (ceiling).

Usage (GPU 2 or CPU; never GPU 3):
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=2 \
      python scripts/phase_timespans.py --splits 0 1 2
Output: data/phase_pseudo/phase_timespans.json
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
from libs.core import load_config
from libs.datasets import make_dataset

N_PHASE = pp.N_PHASE


# ------------------------------------------------------------ GT timespans
def test_segments(cfg, n, pseudo, seen, unseen, pseudo_unseen=None):
    """Rebuild the test-dump video order of phase_probe.dump_test and, per
    (video index, segment index), the segment span (grid) and its 9 pseudo
    cuts (grid; NaN = none).  Mirrors dump_test exactly; unseen-class
    segments take their cuts from pseudo_unseen (gen_phase_pseudo_unseen.py)
    when given."""
    pseudo_unseen = pseudo_unseen or {}
    ds = make_dataset(cfg['dataset_name'], n, False, cfg['val_split'], **cfg['dataset'])
    all_cls = list(seen) + list(unseen)
    ds.label_dict = None
    ds.subset_labels = all_cls
    ds.data_list, ds.label_dict = ds._load_json_db(ds.json_file)
    with open(cfg['dataset']['json_file'], encoding='utf-8') as f:
        db = json.load(f)['database']
    cls_id = {c: i for i, c in enumerate(all_cls)}
    unseen_set = set(unseen)
    fps, fstr, nfr = (ds.default_fps, cfg['dataset']['feat_stride'],
                      cfg['dataset']['num_frames'])
    segs = {}
    for vi, item in enumerate(ds.data_list):
        vid = item['id']
        fps_v = item['fps'] if fps is None else fps
        j = 0
        for a in db[vid].get('annotations', []):
            lab = a['label']
            if lab not in cls_id:
                continue
            s, e = float(a['segment'][0]), float(a['segment'][1])
            cuts = np.full(9, np.nan, np.float32)
            pe = (pseudo_unseen if lab in unseen_set else pseudo).get(vid, [])
            if True:
                for p in pe:
                    if (p['label'] == lab and abs(float(p['segment'][0]) - s) <= 1e-3
                            and abs(float(p['segment'][1]) - e) <= 1e-3):
                        cuts = pp.phase_cuts_grid(p, fps_v, fstr, nfr)
                        break
            segs[(vi, j)] = dict(
                vid=vid, label=lab, seen=lab not in unseen_set,
                s=(s * fps_v - 0.5 * nfr) / fstr, e=(e * fps_v - 0.5 * nfr) / fstr,
                cuts=cuts, sec_per_grid=fstr / fps_v)
            j += 1
    return segs


# ------------------------------------------------------------------ decode
def viterbi_lr(logp):
    """Left-to-right Viterbi.  logp (T, K) log-posteriors; transitions k->k
    or k->k+1, start 0, end K-1.  Requires T >= K.  Returns states (T,)."""
    T, K = logp.shape
    assert T >= K
    NEG = -1e18
    score = np.full((T, K), NEG)
    back = np.zeros((T, K), np.int8)
    score[0, 0] = logp[0, 0]
    for t in range(1, T):
        stay = score[t - 1]
        move = np.concatenate([[NEG], score[t - 1, :-1]])
        take_move = move > stay
        score[t] = np.where(take_move, move, stay) + logp[t]
        back[t] = take_move.astype(np.int8)
    states = np.zeros(T, np.int64)
    states[-1] = K - 1
    for t in range(T - 1, 0, -1):
        states[t - 1] = states[t] - back[t, states[t]]
    return states


def boundaries_from_states(t, states, s, e):
    """9 cut points: midpoint between the last location of phase k and the
    first of phase k+1 (every phase present under the no-skip decode)."""
    b = np.zeros(9)
    for k in range(9):
        last_k = t[states <= k].max()
        first_k1 = t[states > k].min()
        b[k] = 0.5 * (last_k + first_k1)
    return np.clip(b, s, e)


def boundary_metrics(pred, gt, s, e, sec_per_grid):
    L = max(e - s, 1e-6)
    err = np.abs(pred - gt)
    # phase-segment tIoU: the 10 intervals [s,c1],[c1,c2],...,[c9,e] of the
    # prediction vs the pseudo timespans, mean over the 10 phases
    pb = np.concatenate([[s], pred, [e]])
    gb = np.concatenate([[s], gt, [e]])
    inter = np.maximum(0.0, np.minimum(pb[1:], gb[1:]) - np.maximum(pb[:-1], gb[:-1]))
    union = np.maximum(pb[1:], gb[1:]) - np.minimum(pb[:-1], gb[:-1])
    tiou = np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 1.0)
    return {'mae_sec': float(err.mean() * sec_per_grid),
            'mae_frac': float((err / L).mean()),
            'within10': float((err / L <= 0.10).mean()),
            'within_0.5s': float((err * sec_per_grid <= 0.5).mean()),
            'phase_tiou': float(tiou.mean())}


def agg(ms):
    keys = ('mae_sec', 'mae_frac', 'within10', 'within_0.5s', 'phase_tiou')
    out = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    out['n_segs'] = len(ms)
    return out


# -------------------------------------------------------------------- main
def run_split(n, args, device):
    cfg = load_config(args.config)
    seen = pp.load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'train', n)
    unseen = pp.load_classes(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], 'test', n)
    with open(args.pseudo.format(n=n), encoding='utf-8') as f:
        pseudo = json.load(f)['videos']
    pseudo_unseen = None
    pu_path = args.pseudo_unseen.format(n=n)
    if os.path.exists(pu_path):
        with open(pu_path, encoding='utf-8') as f:
            pseudo_unseen = json.load(f)['videos']
    gt = test_segments(cfg, n, pseudo, seen, unseen, pseudo_unseen)
    train = dict(np.load(os.path.join(args.dump_dir, f'split{n}_train.npz')))
    tests = {'testvocab': dict(np.load(os.path.join(args.dump_dir, f'split{n}_test.npz')))}
    if args.seenvocab:
        tests['seenvocab'] = dict(np.load(os.path.join(args.dump_dir, f'split{n}_test_seenvocab.npz')))
    X = train['feat'].astype(np.float32)
    d_hid = X.shape[1]
    res = {'n_gt_segments': len(gt)}
    for vocab, test in tests.items():
        out = {}
        # candidate segments per subset: with cuts, >= 10 level-0 inside
        # locations (seen: pseudo cuts of Stage A; unseen: the same
        # procedure run on the unseen classes, gen_phase_pseudo_unseen.py)
        lvl0 = test['level'] == 0
        for subset in ('seen', 'unseen'):
            seg_key = {}
            for (vi, j), g in gt.items():
                if g['seen'] != (subset == 'seen') or np.isnan(g['cuts']).any():
                    continue
                idx = np.flatnonzero(lvl0 & (test['video'] == vi) & (test['in_seg'] == j))
                if len(idx) < N_PHASE:
                    continue
                seg_key[(vi, j)] = idx[np.argsort(test['t'][idx], kind='stable')]
            n_with_cuts = sum(1 for g in gt.values()
                              if g['seen'] == (subset == 'seen') and not np.isnan(g['cuts']).any())
            print(f"  [{vocab}/{subset}] segments with pseudo cuts {n_with_cuts}, decodable "
                  f"(>=10 level-0 locations) {len(seg_key)}", flush=True)
            if not seg_key:
                continue
            # per-location phase labels from the cuts (identical to the
            # dump's in_phase for seen segments; needed for unseen)
            all_idx = np.concatenate(list(seg_key.values()))
            y_loc = np.concatenate([
                (test['t'][idx][:, None] >= gt[key]['cuts'][None]).sum(1)
                for key, idx in seg_key.items()]).astype(np.int64)
            chance, oracle, dev = [], [], []
            for key, idx in seg_key.items():
                g = gt[key]
                L = g['e'] - g['s']
                b = g['s'] + L * np.arange(1, 10) / 10.0
                chance.append(boundary_metrics(b, g['cuts'], g['s'], g['e'], g['sec_per_grid']))
                dev.append(np.abs(g['cuts'] - b) / max(L, 1e-6))
                ph = (test['t'][idx][:, None] >= g['cuts'][None]).sum(1)
                oh = np.full((len(idx), N_PHASE), -20.0)
                oh[np.arange(len(idx)), ph] = 0.0
                st = viterbi_lr(oh)
                b = boundaries_from_states(test['t'][idx].astype(np.float64), st, g['s'], g['e'])
                oracle.append(boundary_metrics(b, g['cuts'], g['s'], g['e'], g['sec_per_grid']))
            dev = np.concatenate(dev)
            so = {'chance_equal_bins': agg(chance), 'oracle_gt_phase_labels': agg(oracle),
                  'n_decodable': len(seg_key), 'n_with_cuts': n_with_cuts,
                  'n_locations': int(len(all_idx)),
                  'pseudo_cut_dev_from_uniform': {
                      'mean_frac': float(dev.mean()), 'median_frac': float(np.median(dev)),
                      'p90_frac': float(np.percentile(dev, 90)),
                      'frac_beyond_10pct': float((dev > 0.10).mean())}}
            so['chance_equal_bins']['acc10'] = float(
                (np.concatenate([np.minimum((test['in_tnorm'][idx] * N_PHASE).astype(int), 9)
                                 for idx in seg_key.values()]) == y_loc).mean())
            lens = [(gt[k]['e'] - gt[k]['s']) * gt[k]['sec_per_grid'] for k in seg_key]
            so['seg_len_sec_mean'] = float(np.mean(lens))
            so['seg_len_sec_median'] = float(np.median(lens))
            for tr_rule in args.train_rules:
                m = train[f'{tr_rule}_phase'] >= 0
                Xtr = X[m]
                ytr = train[f'{tr_rule}_phase'][m].astype(np.int64)
                ttr = train[f'{tr_rule}_tnorm'][m].astype(np.float32)[:, None]
                for variant in ('feature', 'position_only', 'chance_shuffled'):
                    per_seed = []
                    for seed in args.seeds:
                        xin = Xtr if variant != 'position_only' else ttr
                        yy = ytr if variant != 'chance_shuffled' else \
                            np.random.RandomState(seed).permutation(ytr)
                        head = pp.train_head(xin, yy, seed, d_hid, device=device)
                        xte = (test['feat'][all_idx].astype(np.float32)
                               if variant != 'position_only'
                               else test['in_tnorm'][all_idx].astype(np.float32)[:, None])
                        with torch.no_grad():
                            lp = []
                            for i in range(0, len(xte), 65536):
                                lg = head(torch.as_tensor(xte[i:i + 65536], dtype=torch.float32,
                                                          device=device))
                                lp.append(torch.log_softmax(lg, -1).cpu().numpy())
                        lp = np.concatenate(lp, 0)
                        pa = lp.argmax(1)
                        ms, off = [], 0
                        for key, idx in seg_key.items():
                            g = gt[key]
                            st = viterbi_lr(lp[off:off + len(idx)])
                            off += len(idx)
                            b = boundaries_from_states(test['t'][idx].astype(np.float64), st,
                                                       g['s'], g['e'])
                            ms.append(boundary_metrics(b, g['cuts'], g['s'], g['e'],
                                                       g['sec_per_grid']))
                        r = agg(ms)
                        r['acc10'] = float((pa == y_loc).mean())
                        r['acc3'] = float((pp.MERGE3[pa] == pp.MERGE3[y_loc]).mean())
                        per_seed.append(r)
                        del head
                    so[f'{variant}@train_{tr_rule}'] = pp.average_dicts(per_seed)
            out[subset] = so
        res[vocab] = out
    return res


def fmt(n, r):
    lines = [f"--- split {n} ({r['n_gt_segments']:.0f} test GT segments)"]
    for vocab, out in r.items():
        if vocab == 'n_gt_segments':
            continue
        for subset, so in out.items():
            lines.append(f"  [{vocab}/{subset}] decodable segments {so['n_decodable']:.0f} / "
                         f"{so['n_with_cuts']:.0f} with cuts ({so['n_locations']:.0f} level-0 "
                         f"locations); mean len {so['seg_len_sec_mean']:.2f}s "
                         f"(median {so['seg_len_sec_median']:.2f}s)")
            d = so['pseudo_cut_dev_from_uniform']
            lines.append(f"    pseudo cuts vs equal-width bins: mean |dev| "
                         f"{100 * d['mean_frac']:.1f}% of segment (median "
                         f"{100 * d['median_frac']:.1f}%, p90 {100 * d['p90_frac']:.1f}%), "
                         f"{100 * d['frac_beyond_10pct']:.1f}% of cuts beyond 10%")
            lines.append(f"    {'readout':28s} {'acc10':>6s} {'acc3':>6s} {'MAE(s)':>7s} "
                         f"{'MAE(%L)':>8s} {'<10%L':>6s} {'<0.5s':>6s} {'tIoU':>6s}")
            for k, v in so.items():
                if not isinstance(v, dict) or 'mae_sec' not in v:
                    continue
                a10 = f"{v['acc10']:6.3f}" if 'acc10' in v else '     -'
                a3 = f"{v['acc3']:6.3f}" if 'acc3' in v else '     -'
                lines.append(f"    {k:28s} {a10} {a3} {v['mae_sec']:7.3f} "
                             f"{100 * v['mae_frac']:8.1f} {100 * v['within10']:6.1f} "
                             f"{100 * v['within_0.5s']:6.1f} {v['phase_tiou']:6.3f}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--pseudo', default='data/phase_pseudo/thumos14_75_25_split{n}.json')
    ap.add_argument('--pseudo-unseen', default='data/phase_pseudo/thumos14_75_25_split{n}_unseen.json')
    ap.add_argument('--dump-dir', default='data/phase_pseudo/probe_dump')
    ap.add_argument('--out', default='data/phase_pseudo/phase_timespans.json')
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--train-rules', nargs='+', default=['in'])
    ap.add_argument('--seenvocab', action='store_true', help='also the seen-vocab control dump')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    all_res = {'meta': {'head': 'phase_probe.py head, 3 seeds', 'decode':
                        'left-to-right Viterbi over 10 phases (k->k / k->k+1, start 0, '
                        'end 9) on level-0 inside locations; boundary = midpoint between '
                        'consecutive phases', 'controls': 'equal-width bins (chance); '
                        'position-only head (ceiling of the pseudo labels)',
                        'seeds': args.seeds}, 'splits': {}}
    for n in args.splits:
        t0 = time.time()
        print(f"===== split {n} =====", flush=True)
        r = run_split(n, args, args.device)
        all_res['splits'][str(n)] = r
        print(fmt(n, r) + f"\n  ({time.time() - t0:.0f}s)", flush=True)
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
    keys = list(all_res['splits'])
    if len(keys) > 1:
        mean = pp.average_dicts([all_res['splits'][k] for k in keys])
        all_res['mean'] = mean
        print(f"===== mean over splits {keys} =====\n" + fmt('mean', mean), flush=True)
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
    print(f"[save] {args.out}")


if __name__ == '__main__':
    main()
