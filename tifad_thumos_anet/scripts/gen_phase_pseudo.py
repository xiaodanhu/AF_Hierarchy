"""Stage A: phase pseudo-annotation for SEEN-class GT segments (THUMOS14, splits 0-2, 75/25).

For every seen-class GT segment: align 10 ordered phase slots (= the 10 ordered
phase sentences of configs/thumos14_hier_additive_ensemble.json, indices 3..12)
to VMAE-s2 snippets via monotone DP (DTW-style, every slot visited >=1 snippet,
order enforced), cost = w_pos*|t_rel - slot_center| + (1 - cos(f, mu_k)),
with CTE-style EM refinement of per-class slot prototypes mu_k
(init: position deciles). Segments with < 10 snippets get proportional
fallback labels (excluded from EM stats).

Outputs: /data3/xiaodan8/tifad/data/phase_pseudo/thumos14_75_25_split{s}.json
Then kill criteria:
  K1 transfer probe (GroupKFold by video, per class, weighted pool):
  K2 split-half prototype consistency: same-slot cos > cross-slot cos
Control: same probe on pure position-decile labels (diagnostic).
CPU only.
"""
import json, os, sys
import numpy as np
from collections import defaultdict

RNG = np.random.RandomState(0)
ANN = '/data3/xiaodan8/thumos/annotations/thumos14.json'
VMAE = '/data3/xiaodan8/thumos/vmae_s2_features/%s.npy'
PHASES = '/data3/xiaodan8/tifad/configs/thumos14_hier_additive_ensemble.json'
SPLITDIR = '/data3/xiaodan8/tifad/splits/train_75_test_25/THUMOS14/train'
OUTDIR = '/data3/xiaodan8/tifad/data/phase_pseudo'
K = 10
W_POS = 0.7          # position-prior weight (posdist in [0,0.95], featdist in [0,2])
STRIDE = 2.0         # frames per VMAE-s2 snippet step
CENTER_OFF = 8.0     # 16-frame window center offset, frames
EM_ITERS = 5
MIN_DUR = 0.0        # keep all segments; <K snippets -> fallback
PROBE_MAX_PER_CLASS = 3000

db = json.load(open(ANN))['database']
phase_bank = json.load(open(PHASES))
SLOT_CENTERS = (np.arange(K) + 0.5) / K

os.makedirs(OUTDIR, exist_ok=True)

# ---------------- collection ----------------
def collect_split(seen):
    """Return per-class list of segments with features."""
    by_class = defaultdict(list)
    for vid, v in db.items():
        anns = [a for a in v.get('annotations', []) if a['label'] in seen]
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
            if e - s < MIN_DUR:
                continue
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

# ---------------- monotone DP alignment ----------------
def align(cost):
    """cost (T,K); monotone path (0,0)->(T-1,K-1), steps right/diag.
    Every slot 0..K-1 visited. Returns labels (T,)."""
    T, Kk = cost.shape
    D = np.full((T, Kk), np.inf, dtype=np.float64)
    B = np.zeros((T, Kk), dtype=np.int8)  # 0 = same k, 1 = from k-1
    D[0, 0] = cost[0, 0]
    for t in range(1, T):
        # k = 0
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

# ---------------- EM per class ----------------
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
        # M: prototypes
        Fs = np.concatenate([s['feat'] for s in big])
        Ls = np.concatenate([s['phase'] for s in big])
        protos = np.stack([Fs[Ls == k].mean(0) for k in range(K)])
        protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        # E: monotone realignment
        changed = tot = 0
        for s in big:
            featdist = 1.0 - s['feat'] @ protos.T                      # (T,K)
            posdist = np.abs(s['rel'][:, None] - SLOT_CENTERS[None, :])
            lab = align(W_POS * posdist + featdist)
            changed += int((lab != s['phase']).sum()); tot += len(lab)
            s['phase'] = lab
        if changed / max(tot, 1) < 0.005:
            break
    for s in small:
        s['phase'] = prop_labels(len(s['idx']))
        s['fallback'] = True
    return protos

def boundaries(s):
    """per-slot [t0,t1] in seconds (None if slot absent)."""
    out = []
    step = STRIDE / 30.0
    for k in range(K):
        m = np.where(s['phase'] == k)[0]
        out.append(None if len(m) == 0 else
                   [round(float(s['times'][m[0]] - step / 2), 3),
                    round(float(s['times'][m[-1]] + step / 2), 3)])
    return out

# ---------------- kill criteria ----------------
def probe(by_class, label_key='phase', n_max=PROBE_MAX_PER_CLASS):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    res = {}
    for c, segs in sorted(by_class.items()):
        X = np.concatenate([s['feat'] for s in segs])
        y = np.concatenate([s[label_key] for s in segs])
        g = np.concatenate([[s['vid']] * len(s['idx']) for s in segs])
        if len(y) > n_max:
            sel = RNG.choice(len(y), n_max, replace=False)
            X, y, g = X[sel], y[sel], g[sel]
        ug = np.unique(g)
        ns = min(5, len(ug))
        if ns < 2:
            continue
        acc10 = acc3 = n = 0
        for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=300, C=1.0))
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            acc10 += (p == y[te]).sum()
            acc3 += ((p // 3).clip(0, 2) == (y[te] // 3).clip(0, 2)).sum()
            n += len(te)
        res[c] = (acc10 / n, acc3 / n, n)
    return res

def split_half_consistency(by_class):
    out = {}
    for c, segs in sorted(by_class.items()):
        vids = sorted({s['vid'] for s in segs})
        if len(vids) < 4:
            continue
        h = {v: i % 2 for i, v in enumerate(vids)}   # interleave halves
        P = []
        ok = True
        for half in (0, 1):
            F = [s['feat'] for s in segs if h[s['vid']] == half]
            L = [s['phase'] for s in segs if h[s['vid']] == half]
            if not F: ok = False; break
            F = np.concatenate(F); L = np.concatenate(L)
            pr = []
            for k in range(K):
                m = L == k
                if m.sum() == 0: ok = False; break
                v = F[m].mean(0); pr.append(v / (np.linalg.norm(v) + 1e-8))
            if not ok: break
            P.append(np.stack(pr))
        if not ok: continue
        S = P[0] @ P[1].T                                  # (K,K)
        same = float(np.diag(S).mean())
        cross = float((S.sum() - np.trace(S)) / (K * K - K))
        out[c] = (same, cross)
    return out

# ---------------- main ----------------
summary = {}
for sp in (0, 1, 2):
    seen = set(open(f'{SPLITDIR}/split_{sp}.list').read().split())
    print(f'\n######## SPLIT {sp}: {len(seen)} seen classes ########', flush=True)
    by_class = collect_split(seen)
    out = dict(meta=dict(
        split=sp, protocol='train_75_test_25', dataset='THUMOS14', K=K,
        feature='vmae_s2 (stride 2 frames, 16f window center)', w_pos=W_POS,
        em_iters=EM_ITERS, phase_sentence_source=PHASES,
        note='phase k corresponds to ordered phase sentence index 3+k in the '
             'hier_additive_ensemble entry for the class; labels are per '
             'VMAE-s2 snippet index (field idx); fallback=proportional '
             'labels for segments with <10 snippets'), videos=defaultdict(list))
    protos_all = {}
    for c in sorted(by_class):
        segs = by_class[c]
        # store decile control before EM overwrites nothing (separate key)
        for s in segs:
            s['decile'] = decile_labels(s['rel'])
        pr = run_em(segs)
        if pr is not None:
            protos_all[c] = pr
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
    path = f'{OUTDIR}/thumos14_75_25_split{sp}.json'
    json.dump(out, open(path, 'w'))
    print(f'  wrote {path}', flush=True)

    print('  -- K1 transfer probe (EM labels) --', flush=True)
    r = probe(by_class, 'phase')
    n_tot = sum(v[2] for v in r.values())
    a10 = sum(v[0] * v[2] for v in r.values()) / n_tot
    a3 = sum(v[1] * v[2] for v in r.values()) / n_tot
    print('  -- control probe (position deciles) --', flush=True)
    rc = probe(by_class, 'decile')
    c10 = sum(v[0] * v[2] for v in rc.values()) / n_tot if rc else 0
    c3 = sum(v[1] * v[2] for v in rc.values()) / n_tot if rc else 0
    print('  -- K2 split-half prototype consistency --', flush=True)
    cons = split_half_consistency(by_class)
    same = np.mean([v[0] for v in cons.values()])
    cross = np.mean([v[1] for v in cons.values()])
    summary[sp] = dict(probe=r, ctrl=(c10, c3), pooled=(a10, a3),
                       cons=cons, cons_pooled=(same, cross))
    print(f'  SPLIT {sp} POOLED: EM 10-way {a10:.3f} (>=0.20?) '
          f'3-way {a3:.3f} (>=0.45?) | decile ctrl 10-way {c10:.3f} '
          f'3-way {c3:.3f} | proto same {same:.3f} vs cross {cross:.3f}',
          flush=True)

print('\n================ FINAL TABLES ================', flush=True)
for sp, S in summary.items():
    print(f'\nSPLIT {sp} per class: 10way 3way n | proto same/cross')
    for c in sorted(S['probe']):
        a10, a3, n = S['probe'][c]
        sc = S['cons'].get(c)
        cs = f'{sc[0]:.3f}/{sc[1]:.3f}' if sc else '  -  '
        print(f'  {c:<20} {a10:.3f} {a3:.3f} {n:6d} | {cs}')
    a10, a3 = S['pooled']; c10, c3 = S['ctrl']; sm, cr = S['cons_pooled']
    k1 = 'PASS' if (a10 >= 0.20 and a3 >= 0.45) else 'FAIL'
    k2 = 'PASS' if sm > cr else 'FAIL'
    print(f'  POOLED EM {a10:.3f}/{a3:.3f} ctrl {c10:.3f}/{c3:.3f} '
          f'proto {sm:.3f}>{cr:.3f} | K1 {k1} K2 {k2}')
