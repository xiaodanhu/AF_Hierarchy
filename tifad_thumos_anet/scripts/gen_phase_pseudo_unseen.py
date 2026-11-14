"""Stage A phase pseudo-timespans for the UNSEEN classes of each THUMOS14
75/25 split (companion of scripts/gen_phase_pseudo.py, which only labels the
SEEN classes).  Same procedure, byte-for-byte: per class, monotone DP
alignment of the 10 ordered phase slots to VMAE-s2 snippets with cost
w_pos*|t_rel - slot_center| + (1 - cos(f, mu_k)) and 5 EM iterations of the
per-class slot prototypes (init: position deciles); segments with < 10
snippets get proportional fallback labels.  Run over every annotated video
of the class (both subsets, as Stage A does for the seen classes).  No
training signal is involved: the timespans are targets for the eval-only
phase readouts (scripts/phase_timespans.py) on unseen actions.

Output: data/phase_pseudo/thumos14_75_25_split{n}_unseen.json  (CPU only)
Usage:  python scripts/gen_phase_pseudo_unseen.py [--splits 0 1 2]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

ANN = '/data3/xiaodan8/thumos/annotations/thumos14.json'
VMAE = '/data3/xiaodan8/thumos/vmae_s2_features/%s.npy'
PHASES = '/data3/xiaodan8/tifad/configs/thumos14_hier_additive_ensemble.json'
SPLITDIR = '/data3/xiaodan8/tifad/splits/train_75_test_25/THUMOS14/test'
OUTDIR = '/data3/xiaodan8/tifad/data/phase_pseudo'
K = 10
W_POS = 0.7
STRIDE = 2.0
CENTER_OFF = 8.0
EM_ITERS = 5
SLOT_CENTERS = (np.arange(K) + 0.5) / K


def collect_split(db, classes):
    by_class = defaultdict(list)
    for vid, v in db.items():
        anns = [a for a in v.get('annotations', []) if a['label'] in classes]
        if not anns:
            continue
        try:
            feat = np.load(VMAE % vid, mmap_mode='r')
        except FileNotFoundError:
            continue
        fps = v['fps']
        T = feat.shape[0]
        tt = (np.arange(T) * STRIDE + CENTER_OFF) / fps
        for a in anns:
            s, e = a['segment']
            ii = np.where((tt >= s) & (tt <= e))[0]
            if len(ii) == 0:
                continue
            f = np.asarray(feat[ii[0]:ii[-1] + 1], dtype=np.float32)
            f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            rel = (tt[ii] - s) / max(e - s, 1e-6)
            by_class[a['label']].append(dict(
                vid=vid, subset=v['subset'], segment=[s, e],
                idx=ii, feat=f, rel=rel, times=tt[ii]))
    return by_class


def align(cost):
    T, Kk = cost.shape
    D = np.full((T, Kk), np.inf, dtype=np.float64)
    B = np.zeros((T, Kk), dtype=np.int8)
    D[0, 0] = cost[0, 0]
    for t in range(1, T):
        D[t, 0] = D[t - 1, 0] + cost[t, 0]
        kmax = min(t + 1, Kk)
        prev_same = D[t - 1, 1:kmax]
        prev_diag = D[t - 1, 0:kmax - 1]
        take_diag = prev_diag < prev_same
        D[t, 1:kmax] = np.where(take_diag, prev_diag, prev_same) + cost[t, 1:kmax]
        B[t, 1:kmax] = take_diag
    lab = np.empty(T, dtype=np.int64)
    k = Kk - 1
    for t in range(T - 1, -1, -1):
        lab[t] = k
        if t > 0 and B[t, k]:
            k -= 1
    return lab


def decile_labels(rel):
    return np.minimum((rel * K).astype(np.int64), K - 1)


def prop_labels(T):
    return np.minimum((np.arange(T) / T * K).astype(np.int64), K - 1)


def run_em(segs):
    big = [s for s in segs if len(s['idx']) >= K]
    small = [s for s in segs if len(s['idx']) < K]
    if not big:
        for s in segs:
            s['phase'] = prop_labels(len(s['idx']))
            s['fallback'] = True
        return None
    for s in big:
        s['phase'] = decile_labels(s['rel'])
        s['fallback'] = False
    protos = None
    for it in range(EM_ITERS):
        Fs = np.concatenate([s['feat'] for s in big])
        Ls = np.concatenate([s['phase'] for s in big])
        protos = np.stack([Fs[Ls == k].mean(0) for k in range(K)])
        protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        changed = tot = 0
        for s in big:
            featdist = 1.0 - s['feat'] @ protos.T
            posdist = np.abs(s['rel'][:, None] - SLOT_CENTERS[None, :])
            lab = align(W_POS * posdist + featdist)
            changed += int((lab != s['phase']).sum())
            tot += len(lab)
            s['phase'] = lab
        if changed / max(tot, 1) < 0.005:
            break
    for s in small:
        s['phase'] = prop_labels(len(s['idx']))
        s['fallback'] = True
    return protos


def boundaries(s):
    out = []
    step = STRIDE / 30.0
    for k in range(K):
        m = np.where(s['phase'] == k)[0]
        out.append(None if len(m) == 0 else
                   [round(float(s['times'][m[0]] - step / 2), 3),
                    round(float(s['times'][m[-1]] + step / 2), 3)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()
    db = json.load(open(ANN))['database']
    for sp in args.splits:
        unseen = set(open(f'{SPLITDIR}/split_{sp}.list').read().split())
        print(f'######## SPLIT {sp}: {len(unseen)} unseen classes {sorted(unseen)}', flush=True)
        by_class = collect_split(db, unseen)
        out = dict(meta=dict(
            split=sp, protocol='train_75_test_25', dataset='THUMOS14', K=K,
            classes='UNSEEN classes of the split (scripts/gen_phase_pseudo_unseen.py)',
            feature='vmae_s2 (stride 2 frames, 16f window center)', w_pos=W_POS,
            em_iters=EM_ITERS, phase_sentence_source=PHASES,
            note='same Stage-A procedure as thumos14_75_25_split{n}.json; '
                 'fallback=proportional labels for segments with <10 snippets'),
            videos=defaultdict(list))
        for c in sorted(by_class):
            segs = by_class[c]
            run_em(segs)
            nf = sum(s['fallback'] for s in segs)
            print(f'  {c:<20} {len(segs):4d} segs ({nf} fallback)', flush=True)
            for s in segs:
                out['videos'][s['vid']].append(dict(
                    label=c, subset=s['subset'],
                    segment=[round(float(x), 3) for x in s['segment']],
                    snippet_idx=[int(s['idx'][0]), int(s['idx'][-1])],
                    phase=s['phase'].tolist(), fallback=bool(s['fallback']),
                    phase_boundaries=boundaries(s)))
        out['videos'] = dict(out['videos'])
        path = f'{OUTDIR}/thumos14_75_25_split{sp}_unseen.json'
        json.dump(out, open(path, 'w'))
        print(f'  wrote {path}', flush=True)


if __name__ == '__main__':
    main()
