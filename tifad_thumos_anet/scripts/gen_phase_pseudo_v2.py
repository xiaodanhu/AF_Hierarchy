"""Stage A v2: VISUALLY-DRIVEN phase pseudo timespans for THUMOS14 (75/25, splits 0-2).

Why: scripts/gen_phase_pseudo.py aligned 10 ordered phase slots to VMAE-s2
snippets with cost 0.7*|t_rel - slot_center| + (1 - cos(f, mu_k)) and
EM-refined prototypes initialised from position deciles.  Its cuts sit
within ~2.1 % of the segment length of equal-width tenths: the visual term
barely varies inside an action, so the position prior decides everything.
This script produces alternative label sets that are driven by visual
evidence and evaluates them against each other and against the current
labels / equal-width tenths / random cuts.

Label sets (every one is a set of 9 cut times in seconds per GT segment;
the json files keep the format of the current files -- per-VMAE-snippet
phase labels + phase_boundaries -- so every consumer keeps working):

  A  text alignment in the detector's joint space.  Phase sentence
     (10 per class, configs/thumos14_hier_additive_ensemble.json) -> frozen
     CLIP text tower (openai/clip-vit-base-patch16, pooled 512-d) -> the
     champion's trained text projection W_text (256x512, from
     ckpt/thumos14_i3d_leafref_{n}/mAP_best.pth.tar) -> de-anisotropised:
     the mean over the 200 phase embeddings (all phases of all 20 classes)
     is subtracted ('mean'; variant 'pca3' also removes the projection on
     the top-3 principal components of the centred bank), L2-renormalised.
     Video side = the champion's level-0 cls-trunk features of the same
     split (probe_dump/split{n}_{train,test,test_seenvocab}.npz, the exact
     features the class cosine is computed on; stride 4 frames).  Visual
     term = cosine(trunk, phase), z-scored per segment; cost =
     w_pos*|t_rel - slot_center| - 0.5*z, w_pos in {0.2, 0}; monotone DP
     (every slot visited, order enforced).  Seen classes: train-subset
     videos from the train dump (crop offset recovered as in
     scripts/phase_probe_raw.py) + test-subset videos from the seen-vocab
     test dump; unseen classes: test-subset videos only (test-vocab dump).
  B  CLIP frame-level step grounding (DistantSup-style).  CLIP ViT-B/16
     image embeddings of every 4th frame inside every GT segment (cached
     under data/phase_pseudo/clip_frames/<vid>.npz), scored against the 10
     phase sentences of the segment's class with CLIP text embeddings
     mean-centred over the 200 phase sentences of the vocabulary, smoothed
     over 5 frames, z-scored per segment, monotone DP with w_pos in
     {0.2, 0}.  'B_em': the text alignment (w_pos 0.2) initialises per-
     class frame prototypes refined by 5 EM iterations (M: mean frame
     embedding per phase over the class; E: realign with the z-scored
     prototype cosine), as the original script does.  '*_cc' (A and B):
     column-centred -- each phase's mean similarity over the segment is
     removed before the z-score, so a phase sentence that is globally
     closer to every frame cannot swallow the segment; only the temporal
     variation of each phase's similarity drives the alignment.
  C  visual change points, no text.  The snippet sequence of a segment
     (VMAE-s2 L2-normalised; raw I3D at stride 4 both streams, each stream
     L2-normalised, and flow only) is cut into exactly 10 ordered pieces
     minimising the within-piece sum of squared deviations (exact dynamic
     programme = ruptures' Dynp with the l2 cost, K-1 change points),
     pieces -> phases 1..10 in order.  'C_*_pen': K is chosen per class
     in 3..10 with a permutation penalty (the largest K whose mean relative
     cost drop cost_{K-1}-cost_K still exceeds 1.5x the drop obtained on
     time-shuffled sequences), then the K pieces are split proportionally
     into 10 phases (piecewise-linear map of the 9 tenths onto the pieces).

Metrics (per split over the SEEN classes of the split; 'eval population' =
seen-class segments >= 1.5 s with every feature available, so every label
set is scored on the same segments; the probe uses all rows):
  1  deviation of the 9 cuts from equal-width tenths, mean and p90, as a
     fraction of the segment length (current labels: ~2.1 %);
  2  split-half consistency (the original K2): per class, the videos are
     interleaved into two halves, per-half phase prototypes (mean feature
     per phase) are compared across halves: mean same-phase cosine minus
     mean cross-phase cosine, on VMAE-s2 and on raw I3D (both streams);
  3  cross-feature boundary agreement: mean |cut difference| / length
     against the change-point cuts of the OTHER feature (VMAE vs I3D);
     a label set whose cuts agree with change points found in a feature
     it never saw reflects visual events rather than the method;
  4  transfer probe: the phase_probe.py head (Linear-ReLU-Linear, Adam,
     30 epochs, 3 seeds) trained on the champion's level-0 trunk features
     of the train videos ('in' rule, probe_dump) relabelled with the label
     set; 10-way accuracy on held-out seen and unseen test segments
     (relabelled by the same method) and the per-segment Spearman of the
     expected phase vs normalised time.
     Because the deviation from uniform dominates |cut difference|, the
     table also reports the Pearson correlation between two sets'
     deviations from the equal-width tenths (corr_c_vmae / corr_c_i3d):
     0 = unrelated cuts, > 0 = the sets move the same cuts the same way.
  5  (E) separability: LDA over the 10 phases on the training rows after
     centring every row on its GT segment's mean trunk feature (within-
     action scatter only); held-out 10-way accuracy seen / unseen and the
     between-phase / within-phase variance ratio along the first LDA
     direction (train rows and held-out seen rows).
  D  (pilot, scripts/gen_phase_pseudo_qwenvl.py) Qwen2.5-VL-7B-Instruct
     asked for the 10 phase start times of 200 seen segments of split 0;
     its cut set 'qwenvl' is scored in a separate PILOT table on exactly
     those segments, with the probe / LDA retrained on the pilot rows only
     for every label set.
Controls: 'current' (the Stage-A labels), 'uniform' (equal-width tenths),
'random' (9 sorted uniform random cuts, seed 0).
CAVEAT on the probe for A and B (non-cc and cc alike): the A labels are a
deterministic function of the very trunk features the probe reads, and the
B labels of CLIP appearance that those features largely encode, so their
seen-class probe accuracy is partly circular; the unseen columns (labels
of other classes, made the same way) are the honest transfer numbers.

RESULT (3-split mean, data/phase_pseudo/v2/summary.json, 2026-09-18): no
visually driven set is far from uniform AND consistent AND learnable.
K2 consistency is at the uniform level (~0.01) for every set except the
current one on the feature it was fitted on; the only sets whose cuts
agree across features are the change points themselves (VMAE vs I3D
r = 0.32) and the permutation penalty supports 3-5 visual pieces per
0.187 unseen vs 0.248 / 0.176 current) but its cuts do not correlate with
change points (r <= 0.08) and K2 = uniform; A is circular (0.53 seen /
0.09 unseen); qwenvl (pilot) is unlearnable (0.145 < uniform 0.181).

Usage (cd /data3/xiaodan8/tifad):
  python scripts/gen_phase_pseudo_v2.py --stage clip   # GPU, ~15 min, once
  python scripts/gen_phase_pseudo_v2.py --stage labels # CPU
  python scripts/gen_phase_pseudo_v2.py --stage eval --skip-pilot   # CPU, ~5 min/split
  (after scripts/gen_phase_pseudo_qwenvl.py:)
  python scripts/gen_phase_pseudo_v2.py --stage eval --pilot-only --splits 0
Outputs: data/phase_pseudo/v2/<method>_split{n}{,_unseen}.json (+ cuts/*.npz),
         data/phase_pseudo/v2/summary.json (+ printed table; logs/phase_v2_eval*.log)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = '/data3/xiaodan8/tifad'
ANN = '/data3/xiaodan8/thumos/annotations/thumos14.json'
VMAE = '/data3/xiaodan8/thumos/vmae_s2_features/%s.npy'
I3D = '/data3/xiaodan8/thumos/i3d_features/%s.npy'
VIDEO = '/data3/xiaodan8/thumos/video/%s.mp4'
PHASES = ROOT + '/configs/thumos14_hier_additive_ensemble.json'
SPLITLIST = ROOT + '/splits/train_75_test_25/THUMOS14/%s/split_%d.list'
PSEUDO_DIR = ROOT + '/data/phase_pseudo'
OUTDIR = PSEUDO_DIR + '/v2'
CUTDIR = OUTDIR + '/cuts'
CLIPDIR = PSEUDO_DIR + '/clip_frames'
CUR = PSEUDO_DIR + '/thumos14_75_25_split%d%s.json'
DUMP = PSEUDO_DIR + '/probe_dump'
CKPT = ROOT + '/ckpt/thumos14_i3d_leafref_%d/mAP_best.pth.tar'
CLIP_NAME = 'openai/clip-vit-base-patch16'
K = 10
SLOT_CENTERS = (np.arange(K) + 0.5) / K
VMAE_STRIDE, VMAE_OFF = 2.0, 8.0
I3D_STRIDE, I3D_OFF = 4.0, 8.0
FRAME_STRIDE = 4
MIN_EVAL_SEC = 1.5
EM_ITERS = 5
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

DB = json.load(open(ANN, encoding='utf-8'))['database']
PHASE_BANK = json.load(open(PHASES, encoding='utf-8'))
CLASSES = sorted(PHASE_BANK)


# ------------------------------------------------------------ registry
def phase_sentences(c):
    """The 10 ordered phase sentences of a class: the run of 10 consecutive
    capitalised sentences ending in '.' (index 3..12 or 4..13 in the bank)."""
    v = PHASE_BANK[c]
    idx = [i for i, x in enumerate(v) if x[0].isupper() and x.rstrip().endswith('.')]
    assert len(idx) == K and idx == list(range(idx[0], idx[0] + K)), (c, idx)
    return [v[i] for i in idx]


def build_segments():
    """Every GT segment of every annotated video, key = (vid, annotation
    index) -- the same (video, j) indexing phase_timespans.test_segments
    uses on the 20-class vocabulary."""
    segs = []
    for vid in sorted(DB):
        v = DB[vid]
        for j, a in enumerate(v.get('annotations', [])):
            s, e = float(a['segment'][0]), float(a['segment'][1])
            segs.append(dict(key=(vid, j), vid=vid, j=j, label=a['label'],
                             subset=v['subset'], s=s, e=e, L=max(e - s, 1e-6),
                             fps=float(v['fps'])))
    return segs


SEGS = build_segments()
SEG_OF = {s['key']: s for s in SEGS}
BY_CLASS = defaultdict(list)
for _s in SEGS:
    BY_CLASS[_s['label']].append(_s)


def split_classes(sp):
    seen = open(SPLITLIST % ('train', sp)).read().split()
    unseen = open(SPLITLIST % ('test', sp)).read().split()
    return seen, unseen


# ------------------------------------------------------------ features
_CACHE = {}


def _load(path, key):
    if key not in _CACHE:
        _CACHE[key] = np.load(path, mmap_mode='r')
    return _CACHE[key]


def seg_feat(seg, kind):
    """(times (T,), features (T,D) L2-normalised) of the snippets whose
    centre lies inside [s, e]; kind in vmae | i3d_both | i3d_flow | i3d_rgb
    | clip.  Returns (None, None) if no snippet falls inside."""
    s, e, fps = seg['s'], seg['e'], seg['fps']
    if kind == 'vmae':
        f = _load(VMAE % seg['vid'], ('vmae', seg['vid']))
        tt = (np.arange(f.shape[0]) * VMAE_STRIDE + VMAE_OFF) / fps
    elif kind.startswith('i3d'):
        f = _load(I3D % seg['vid'], ('i3d', seg['vid']))
        tt = (np.arange(f.shape[0]) * I3D_STRIDE + I3D_OFF) / fps
    elif kind == 'clip':
        z = _load(CLIPDIR + '/%s.npz' % seg['vid'], ('clip', seg['vid']))
        idx, f = z['frame_idx'], z['emb']
        tt = idx / fps
    else:
        raise ValueError(kind)
    ii = np.where((tt >= s) & (tt <= e))[0]
    if len(ii) == 0:
        return None, None
    x = np.asarray(f[ii[0]:ii[-1] + 1], dtype=np.float32)
    if kind == 'i3d_both':
        a = x[:, :1024] / (np.linalg.norm(x[:, :1024], axis=1, keepdims=True) + 1e-8)
        b = x[:, 1024:] / (np.linalg.norm(x[:, 1024:], axis=1, keepdims=True) + 1e-8)
        x = np.concatenate([a, b], 1) / np.sqrt(2.0)
    else:
        if kind == 'i3d_flow':
            x = x[:, 1024:]
        elif kind == 'i3d_rgb':
            x = x[:, :1024]
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return tt[ii], x


def vmae_index(seg):
    f = _load(VMAE % seg['vid'], ('vmae', seg['vid']))
    tt = (np.arange(f.shape[0]) * VMAE_STRIDE + VMAE_OFF) / seg['fps']
    ii = np.where((tt >= seg['s']) & (tt <= seg['e']))[0]
    return ii, tt[ii]


# ------------------------------------------------------------ alignment
def align(cost):
    """Monotone DP (from gen_phase_pseudo.py): cost (T,K), path from (0,0)
    to (T-1,K-1) with right/diag steps, every slot visited.  Labels (T,)."""
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


def cuts_from_labels(tt, lab, s, e):
    """9 cut times: midpoint between the last snippet of phase k and the
    first of phase k+1 (every phase present)."""
    c = np.zeros(K - 1)
    for k in range(K - 1):
        c[k] = 0.5 * (tt[lab <= k].max() + tt[lab > k].min())
    return np.clip(c, s, e)


def uniform_cuts(seg):
    return seg['s'] + seg['L'] * np.arange(1, K) / K


def rel_time(tt, seg):
    return (tt - seg['s']) / seg['L']


def zscore(S, colcenter=False):
    """z-score over the whole (T,K) matrix; colcenter first removes each
    phase's mean similarity over the segment (per-phase bias), so only the
    temporal variation of each phase's similarity drives the alignment."""
    if colcenter:
        S = S - S.mean(0, keepdims=True)
    return (S - S.mean()) / (S.std() + 1e-8)


def smooth_time(S, w=5):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(S, size=w, axis=0, mode='nearest')


def text_align(segs, proto_of, kind, w_pos, smooth=0, em_iters=0, colcenter=False):
    """Monotone alignment of the segments' `kind` features to per-class
    text prototypes (dict class -> (K,D) unit rows).  Cost = w_pos*|t_rel -
    centre| - 0.5*z(cos).  Optional EM refinement of visual prototypes.
    Returns {key: cuts} (segments with < K snippets are skipped)."""
    out, feats = {}, {}
    for seg in segs:
        tt, x = seg_feat(seg, kind)
        if tt is None or len(tt) < K:
            continue
        feats[seg['key']] = (tt, x)
        P = proto_of[seg['label']]
        S = x @ P.T
        if smooth > 1:
            S = smooth_time(S, smooth)
        cost = w_pos * np.abs(rel_time(tt, seg)[:, None] - SLOT_CENTERS[None]) - 0.5 * zscore(S, colcenter)
        out[seg['key']] = align(cost)
    for it in range(em_iters):
        by_cls = defaultdict(list)
        for seg in segs:
            if seg['key'] in out:
                by_cls[seg['label']].append(seg)
        changed = tot = 0
        for c, cs in by_cls.items():
            F = np.concatenate([feats[s['key']][1] for s in cs])
            Lb = np.concatenate([out[s['key']] for s in cs])
            pr = np.stack([F[Lb == k].mean(0) if (Lb == k).any() else np.zeros(F.shape[1])
                           for k in range(K)])
            pr /= (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-8)
            for s in cs:
                tt, x = feats[s['key']]
                S = x @ pr.T
                if smooth > 1:
                    S = smooth_time(S, smooth)
                cost = w_pos * np.abs(rel_time(tt, s)[:, None] - SLOT_CENTERS[None]) - 0.5 * zscore(S, colcenter)
                lab = align(cost)
                changed += int((lab != out[s['key']]).sum())
                tot += len(lab)
                out[s['key']] = lab
        if changed / max(tot, 1) < 0.005:
            break
    return {k: cuts_from_labels(feats[k][0], lab, SEG_OF[k]['s'], SEG_OF[k]['e'])
            for k, lab in out.items()}


# ------------------------------------------------------------ change points
def kseg_costs(X, Kmax):
    """Optimal K-segmentation of X (T,D) for every K <= Kmax under the l2
    cost (within-piece sum of squared deviations from the piece mean).
    Returns (costs[1..Kmax] (Kmax+1,), labels for K=Kmax, labels for all K
    as dict)."""
    T = X.shape[0]
    cs = np.zeros((T + 1, X.shape[1]))
    cs[1:] = np.cumsum(X, 0)
    cs2 = np.zeros(T + 1)
    cs2[1:] = np.cumsum((X * X).sum(1))
    G = cs @ cs.T                                    # (T+1,T+1)
    n = np.arange(T + 1)
    length = n[None, :] - n[:, None]                 # b - a
    with np.errstate(divide='ignore', invalid='ignore'):
        sq = np.diag(G)[None, :] + np.diag(G)[:, None] - 2 * G
        C = (cs2[None, :] - cs2[:, None]) - sq / np.where(length > 0, length, 1)
    C = np.where(length > 0, np.maximum(C, 0.0), np.inf)  # C[a,b] piece [a,b)
    Kmax = min(Kmax, T)
    D = np.full((Kmax + 1, T + 1), np.inf)
    A = np.zeros((Kmax + 1, T + 1), np.int64)
    D[1] = C[0]
    for k in range(2, Kmax + 1):
        cand = D[k - 1][:, None] + C                  # (a, b)
        A[k] = cand.argmin(0)
        D[k] = cand[A[k], n]
    costs = D[:, T]
    labels = {}
    for k in range(1, Kmax + 1):
        bounds = [T]
        b = T
        for kk in range(k, 1, -1):
            b = A[kk, b]
            bounds.append(b)
        bounds = bounds[::-1]                        # starts of pieces 2..k, then T
        lab = np.zeros(T, np.int64)
        prev = 0
        for i, b in enumerate(bounds[:-1]):
            lab[b:] = i + 1
        labels[k] = lab
    return costs, labels


def piece_cuts_to_10(tt, lab, kk, s, e):
    """Map kk ordered pieces to 10 phases by proportional splitting: the
    piecewise-linear map u in [0,kk] -> time through the piece boundaries,
    evaluated at u = m*kk/10, m = 1..9."""
    b = [s]
    for k in range(kk - 1):
        b.append(0.5 * (tt[lab <= k].max() + tt[lab > k].min()))
    b.append(e)
    b = np.asarray(b)
    u = np.arange(1, K) * kk / K
    return np.clip(np.interp(u, np.arange(kk + 1), b), s, e)


def changepoint_cuts(segs, kind, per_class_k=False, rng_seed=0):
    """{key: cuts} for exactly K pieces (per_class_k=False) or with K chosen
    per class by the permutation penalty then split proportionally."""
    rng = np.random.RandomState(rng_seed)
    out, chosen = {}, {}
    by_cls = defaultdict(list)
    for seg in segs:
        by_cls[seg['label']].append(seg)
    for c, cs in sorted(by_cls.items()):
        res = {}
        gains_real, gains_null = [], []
        for seg in cs:
            tt, x = seg_feat(seg, kind)
            if tt is None or len(tt) < K:
                continue
            costs, labels = kseg_costs(x, K)
            res[seg['key']] = (tt, labels)
            if per_class_k and costs[1] > 1e-9:
                xn = x[rng.permutation(len(x))]
                cn, _ = kseg_costs(xn, K)
                gains_real.append((costs[1:K] - costs[2:K + 1]) / costs[1])
                gains_null.append((cn[1:K] - cn[2:K + 1]) / cn[1])
        if per_class_k:
            gr = np.mean(gains_real, 0)              # index k-2 -> drop K-1 -> K
            gn = np.mean(gains_null, 0)
            ok = [k for k in range(3, K + 1) if gr[k - 2] > 1.5 * gn[k - 2]]
            kk = max(ok) if ok else 3
            chosen[c] = dict(K=kk, gain_real=gr.round(4).tolist(), gain_null=gn.round(4).tolist())
        else:
            kk = K
        for key, (tt, labels) in res.items():
            seg = SEG_OF[key]
            out[key] = (cuts_from_labels(tt, labels[K], seg['s'], seg['e']) if kk == K
                        else piece_cuts_to_10(tt, labels[kk], kk, seg['s'], seg['e']))
    return out, chosen


# ------------------------------------------------------------ CLIP stage
def clip_text_embeddings(device):
    """Pooled (pre-projection, 512) and joint-space (post-projection, 512)
    CLIP text embeddings of the 200 phase sentences, sorted classes x 10."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    pooled, joint = [], []
    with torch.no_grad():
        for c in CLASSES:
            t = tok(phase_sentences(c), padding=True, return_tensors='pt').to(device)
            po = model.text_model(**t).pooler_output
            pooled.append(po.cpu().numpy())
            joint.append(model.text_projection(po).cpu().numpy())
    return np.stack(pooled), np.stack(joint)             # (20,10,512)


class _FrameDS:
    """Frames at stride 4 inside the GT segments of one video (decord)."""
    def __init__(self, vids):
        self.vids = vids

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, i):
        import decord
        vid = self.vids[i]
        v = DB[vid]
        fps = float(v['fps'])
        vr = decord.VideoReader(VIDEO % vid, num_threads=2)
        n = len(vr)
        want = set()
        for a in v['annotations']:
            s, e = a['segment']
            lo = int(np.ceil(s * fps)) - FRAME_STRIDE
            hi = int(np.floor(e * fps)) + FRAME_STRIDE
            lo = max(0, (lo // FRAME_STRIDE) * FRAME_STRIDE)
            for j in range(lo, min(hi, n - 1) + 1, FRAME_STRIDE):
                want.add(j)
        idx = np.array(sorted(want), np.int64)
        frames = []
        for i0 in range(0, len(idx), 256):
            frames.append(vr.get_batch(idx[i0:i0 + 256].tolist()).asnumpy())
        frames = np.concatenate(frames, 0) if frames else np.zeros((0, 16, 16, 3), np.uint8)
        return vid, idx.astype(np.int32), frames


def stage_clip(args):
    """Extract CLIP ViT-B/16 image embeddings (fp16, 512-d) of every 4th
    frame inside every GT segment -> clip_frames/<vid>.npz; text bank ->
    clip_frames/phase_text.npz."""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel
    os.makedirs(CLIPDIR, exist_ok=True)
    device = args.device
    if not os.path.exists(CLIPDIR + '/phase_text.npz'):
        pooled, joint = clip_text_embeddings(device)
        np.savez(CLIPDIR + '/phase_text.npz', classes=np.array(CLASSES), pooled=pooled, joint=joint,
                 sentences=np.array([phase_sentences(c) for c in CLASSES]))
        print('[clip] text bank written', flush=True)
    todo = [v for v in sorted(DB) if not os.path.exists(CLIPDIR + '/%s.npz' % v)]
    if args.max_videos > 0:
        todo = todo[:args.max_videos]
    print(f'[clip] {len(todo)} videos to extract on {device}', flush=True)
    if not todo:
        return
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval().half()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    loader = torch.utils.data.DataLoader(_FrameDS(todo), batch_size=None, num_workers=args.workers,
                                         prefetch_factor=2)
    t0 = time.time()
    n_fr = 0
    with torch.no_grad():
        for i, (vid, idx, frames) in enumerate(loader):
            embs = []
            for i0 in range(0, len(idx), args.batch):
                x = torch.from_numpy(np.asarray(frames[i0:i0 + args.batch])).to(device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                h, w = x.shape[-2:]
                sc = 224.0 / min(h, w)
                x = F.interpolate(x, size=(max(224, int(round(h * sc))), max(224, int(round(w * sc)))),
                                  mode='bicubic', align_corners=False, antialias=True)
                h, w = x.shape[-2:]
                y0, x0 = (h - 224) // 2, (w - 224) // 2
                x = x[:, :, y0:y0 + 224, x0:x0 + 224]
                x = ((x - mean) / std).half()
                e = model.get_image_features(pixel_values=x)
                embs.append(e.float().cpu().numpy().astype(np.float16))
            emb = np.concatenate(embs, 0) if embs else np.zeros((0, 512), np.float16)
            np.savez(CLIPDIR + '/%s.npz' % vid, frame_idx=np.asarray(idx, np.int32), emb=emb)
            n_fr += len(idx)
            if i % 20 == 0:
                print(f'  [clip] {i + 1}/{len(todo)} {vid} {len(idx)} frames '
                      f'({n_fr / max(time.time() - t0, 1e-6):.0f} fr/s)', flush=True)
    print(f'[clip] done {n_fr} frames in {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------ trunk rows
def _probe_modules():
    sys.path.insert(0, ROOT + '/scripts')
    sys.path.insert(0, ROOT)
    import phase_probe as pp                        # sets the reference envs
    import phase_timespans as pt
    import phase_probe_raw as ppr
    from libs.core import load_config
    from libs.datasets import make_dataset
    return pp, pt, ppr, load_config, make_dataset


_TRUNK = {}


def load_trunk(sp):
    """Level-0 'in' rows of the champion's cls-trunk features of split sp
    (probe_dump), each mapped to a registry segment key.  Returns
    {'train': rows, 'test': rows, 'test_seenvocab': rows} with rows =
    dict(feat (N,256) f32 unit, t_sec (N,), tnorm (N,), seg (N,) index into
    SEGS or -1, video (N,), in_seg (N,))."""
    if sp in _TRUNK:
        return _TRUNK[sp]
    pp, pt, ppr, load_config, make_dataset = _probe_modules()
    cfg = load_config('configs/thumos14_i3d.yaml')
    fstr, nfr = cfg['dataset']['feat_stride'], cfg['dataset']['num_frames']
    seen, unseen = split_classes(sp)
    seen_set = set(seen)
    seg_index = {s['key']: i for i, s in enumerate(SEGS)}
    out = {}
    # ---- train dump: crop offset recovered per video (phase_probe_raw)
    ds = make_dataset(cfg['dataset_name'], sp, True, cfg['train_split'], **cfg['dataset'])
    d = dict(np.load(f'{DUMP}/split{sp}_train.npz'))
    rows = np.flatnonzero((d['level'] == 0) & (d['in_seg'] >= 0))
    seg = np.full(len(rows), -1, np.int64)
    t_sec = np.zeros(len(rows))
    stats = dict(videos=0, no_offset=0, unassigned=0, tnorm_mismatch=0)
    for vi in np.unique(d['video'][rows]):
        item = ds.data_list[vi]
        vid = item['id']
        fps = float(item['fps'])
        raw_segs = (np.asarray(item['segments']) * fps - 0.5 * nfr) / fstr
        T_raw = _load(I3D % vid, ('i3d', vid)).shape[0]
        st, ne, nc, nb, ns = ppr.recover_offset(d, vi, raw_segs, T_raw)
        sel = np.flatnonzero(d['video'][rows] == vi)
        if st is None:
            stats['no_offset'] += 1
            continue
        stats['videos'] += 1
        ts = ((d['t'][rows[sel]].astype(np.float64) + st) * fstr + 0.5 * nfr) / fps
        t_sec[sel] = ts
        cand = [s for s in BY_CLASS_VID[vid] if s['label'] in seen_set]
        if not cand:
            continue
        S = np.array([[c['s'], c['e']] for c in cand])
        inside = (ts[:, None] > S[None, :, 0]) & (ts[:, None] < S[None, :, 1])
        lens = np.where(inside, (S[:, 1] - S[:, 0])[None], np.inf)
        j = lens.argmin(1)
        ok = np.isfinite(lens[np.arange(len(ts)), j])
        seg[sel[ok]] = [seg_index[cand[jj]['key']] for jj in j[ok]]
        stats['unassigned'] += int((~ok).sum())
        # sanity: the dump's own normalised time agrees on non-clipped rows
        tn = (ts[ok] - S[j[ok], 0]) / (S[j[ok], 1] - S[j[ok], 0])
        stats['tnorm_mismatch'] += int((np.abs(tn - d['in_tnorm'][rows[sel[ok]]]) > 0.05).sum())
    feat = d['feat'][rows].astype(np.float32)
    out['train'] = dict(feat=feat, t_sec=t_sec, tnorm=d['in_tnorm'][rows].astype(np.float32),
                        seg=seg, video=d['video'][rows], in_seg=d['in_seg'][rows])
    print(f'  [trunk s{sp}] train rows {len(rows)} {stats}', flush=True)
    # ---- test dumps: (video index, annotation index) via phase_timespans.test_segments
    gt = pt.test_segments(cfg, sp, {}, seen, unseen, {})
    for name, fn in (('test', f'split{sp}_test.npz'), ('test_seenvocab', f'split{sp}_test_seenvocab.npz')):
        d = dict(np.load(f'{DUMP}/{fn}'))
        rows = np.flatnonzero((d['level'] == 0) & (d['in_seg'] >= 0))
        seg = np.full(len(rows), -1, np.int64)
        t_sec = np.zeros(len(rows))
        for (vi, j), g in gt.items():
            m = np.flatnonzero((d['video'][rows] == vi) & (d['in_seg'][rows] == j))
            if len(m) == 0:
                continue
            key = (g['vid'], j)
            assert SEG_OF[key]['label'] == g['label'], (key, g['label'])
            seg[m] = seg_index[key]
            t_sec[m] = d['t'][rows[m]].astype(np.float64) * g['sec_per_grid'] + 0.5 * nfr / DB[g['vid']]['fps']
        out[name] = dict(feat=d['feat'][rows].astype(np.float32), t_sec=t_sec,
                         tnorm=d['in_tnorm'][rows].astype(np.float32), seg=seg,
                         video=d['video'][rows], in_seg=d['in_seg'][rows])
        print(f'  [trunk s{sp}] {name} rows {len(rows)} (unassigned {(seg < 0).sum()})', flush=True)
    _TRUNK[sp] = out
    return out


BY_CLASS_VID = defaultdict(list)
for _s in SEGS:
    BY_CLASS_VID[_s['vid']].append(_s)


def trunk_segments(sp, which):
    """{key: (t_sec sorted, unit features)} of the trunk rows; which =
    'seen' (train dump + seen-vocab test dump) or 'unseen' (test dump)."""
    tr = load_trunk(sp)
    seen, unseen = split_classes(sp)
    parts = [tr['train'], tr['test_seenvocab']] if which == 'seen' else [tr['test']]
    want = set(seen if which == 'seen' else unseen)
    out = {}
    for rows in parts:
        for si in np.unique(rows['seg']):
            if si < 0:
                continue
            seg = SEGS[si]
            if seg['label'] not in want:
                continue
            m = np.flatnonzero(rows['seg'] == si)
            o = np.argsort(rows['t_sec'][m], kind='stable')
            f = rows['feat'][m[o]]
            f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            out[seg['key']] = (rows['t_sec'][m[o]], f)
    return out


def text_protos_joint(sp, mode):
    """Per-class (K,256) unit phase embeddings in the champion's joint
    space, de-anisotropised ('mean' or 'pca3')."""
    import torch
    z = np.load(CLIPDIR + '/phase_text.npz')
    ck = torch.load(CKPT % sp, map_location='cpu')['state_dict']
    W = ck['module.text_features.txt_model.text_projection.weight'].numpy()      # (256,512)
    E = z['pooled'].reshape(-1, 512) @ W.T                                       # (200,256)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    raw_cos = float((E @ E.T)[~np.eye(len(E), dtype=bool)].mean())
    E = E - E.mean(0, keepdims=True)
    if mode == 'pca3':
        _, _, Vt = np.linalg.svd(E, full_matrices=False)
        E = E - (E @ Vt[:3].T) @ Vt[:3]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    de_cos = float((E @ E.T)[~np.eye(len(E), dtype=bool)].mean())
    E = E.reshape(len(CLASSES), K, -1)
    return {c: E[i] for i, c in enumerate(CLASSES)}, dict(mean_pairwise_cos_raw=raw_cos,
                                                          mean_pairwise_cos_deaniso=de_cos)


def text_protos_clip():
    z = np.load(CLIPDIR + '/phase_text.npz')
    E = z['joint'].reshape(-1, 512).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    raw_cos = float((E @ E.T)[~np.eye(len(E), dtype=bool)].mean())
    E = E - E.mean(0, keepdims=True)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    de_cos = float((E @ E.T)[~np.eye(len(E), dtype=bool)].mean())
    E = E.reshape(len(CLASSES), K, -1).astype(np.float32)
    return {c: E[i] for i, c in enumerate(CLASSES)}, dict(mean_pairwise_cos_raw=raw_cos,
                                                          mean_pairwise_cos_centred=de_cos)


def trunk_align(sp, which, protos, w_pos, colcenter=False):
    """Method A alignment on the trunk rows of one split."""
    feats = trunk_segments(sp, which)
    out = {}
    for key, (tt, x) in feats.items():
        if len(tt) < K:
            continue
        seg = SEG_OF[key]
        S = x @ protos[seg['label']].T
        cost = w_pos * np.abs(rel_time(tt, seg)[:, None] - SLOT_CENTERS[None]) - 0.5 * zscore(S, colcenter)
        out[key] = cuts_from_labels(tt, align(cost), seg['s'], seg['e'])
    return out


# ------------------------------------------------------------ label sets
def current_cuts(sp, unseen=False):
    js = json.load(open(CUR % (sp, '_unseen' if unseen else ''), encoding='utf-8'))['videos']
    out = {}
    for vid, entries in js.items():
        for p in entries:
            pb = p.get('phase_boundaries')
            if pb is None or any(b is None for b in pb) or p.get('fallback'):
                continue
            s, e = float(p['segment'][0]), float(p['segment'][1])
            for cand in BY_CLASS_VID[vid]:
                if cand['label'] == p['label'] and abs(cand['s'] - s) <= 1e-3 and abs(cand['e'] - e) <= 1e-3:
                    out[cand['key']] = np.clip(np.array([float(pb[k][1]) for k in range(K - 1)]), s, e)
                    break
    return out


def random_cuts(segs, seed=0):
    rng = np.random.RandomState(seed)
    return {s['key']: s['s'] + s['L'] * np.sort(rng.uniform(size=K - 1)) for s in segs}


def save_cuts(name, cuts, extra=None):
    os.makedirs(CUTDIR, exist_ok=True)
    keys = sorted(cuts)
    np.savez(f'{CUTDIR}/{name}.npz', vid=np.array([k[0] for k in keys]),
             j=np.array([k[1] for k in keys], np.int64),
             cuts=np.stack([cuts[k] for k in keys]) if keys else np.zeros((0, K - 1)),
             extra=json.dumps(extra or {}))


def load_cuts(name):
    p = f'{CUTDIR}/{name}.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {(str(v), int(j)): c for v, j, c in zip(z['vid'], z['j'], z['cuts'])}


def write_json(path, cuts, classes, meta):
    """Same format as data/phase_pseudo/thumos14_75_25_split{n}.json: per
    video a list of {label, subset, segment, snippet_idx, phase (per VMAE-s2
    snippet), fallback, phase_boundaries}.  Segments the method could not
    cut (fewer than 10 snippets, or no feature rows) get equal-width
    fallback cuts, flagged."""
    videos = defaultdict(list)
    n_fb = 0
    for seg in SEGS:
        if seg['label'] not in classes:
            continue
        c = cuts.get(seg['key'])
        fb = c is None
        if fb:
            c = uniform_cuts(seg)
            n_fb += 1
        ii, tt = vmae_index(seg)
        phase = (tt[:, None] >= c[None]).sum(1) if len(ii) else np.zeros(0, np.int64)
        b = np.concatenate([[seg['s']], c, [seg['e']]])
        videos[seg['vid']].append(dict(
            label=seg['label'], subset=seg['subset'],
            segment=[round(seg['s'], 3), round(seg['e'], 3)],
            snippet_idx=[int(ii[0]), int(ii[-1])] if len(ii) else [-1, -1],
            phase=phase.astype(int).tolist(), fallback=bool(fb),
            phase_boundaries=[[round(float(b[k]), 3), round(float(b[k + 1]), 3)] for k in range(K)]))
    meta = dict(meta, K=K, protocol='train_75_test_25', dataset='THUMOS14',
                phase_sentence_source=PHASES, n_segments=sum(len(v) for v in videos.values()),
                n_fallback=n_fb,
                note='phase k corresponds to ordered phase sentence k of the class in the '
                     'hier_additive_ensemble entry; labels are per VMAE-s2 snippet index (field '
                     'idx), derived from the 9 cut times phase_boundaries[k][1]; fallback = '
                     'equal-width cuts where the method had no visual rows')
    json.dump(dict(meta=meta, videos=dict(videos)), open(path, 'w'))


METHODS_CLASS = ['B_p0.2', 'B_p0', 'B_em_p0.2', 'B_p0.2_cc', 'B_p0_cc', 'C_vmae', 'C_i3d_both', 'C_i3d_flow',
                 'C_vmae_pen', 'C_i3d_both_pen', 'uniform', 'random']
METHODS_SPLIT = ['A_mean_p0.2', 'A_mean_p0', 'A_pca3_p0.2', 'A_pca3_p0', 'A_mean_p0.2_cc', 'A_mean_p0_cc']
METHODS = ['current', 'uniform', 'random'] + METHODS_SPLIT + METHODS_CLASS[:-2]


def compute_class_method(name):
    """Split-independent label sets over ALL segments of all 20 classes."""
    t0 = time.time()
    extra = {}
    if name == 'uniform':
        cuts = {s['key']: uniform_cuts(s) for s in SEGS}
    elif name == 'random':
        cuts = random_cuts(SEGS)
    elif name.startswith('B'):
        protos, extra = text_protos_clip()
        w = float(name.split('_p')[-1].replace('_cc', ''))
        cuts = text_align(SEGS, protos, 'clip', w, smooth=5, em_iters=EM_ITERS if '_em' in name else 0,
                          colcenter=name.endswith('_cc'))
    elif name.startswith('C'):
        kind = name[2:].replace('_pen', '')
        cuts, chosen = changepoint_cuts(SEGS, kind, per_class_k=name.endswith('_pen'))
        extra = dict(chosen_K=chosen)
    else:
        raise ValueError(name)
    save_cuts(name, cuts, extra)
    print(f'  [labels] {name}: {len(cuts)} segments cut ({time.time() - t0:.0f}s) {extra if extra and len(str(extra)) < 400 else ""}',
          flush=True)
    return cuts


def stage_labels(args):
    os.makedirs(OUTDIR, exist_ok=True)
    for name in METHODS_CLASS:
        if args.only is not None and name not in args.only:
            continue
        if load_cuts(name) is None or args.force:
            compute_class_method(name)
    if args.only is not None:
        return
    for sp in args.splits:
        seen, unseen = split_classes(sp)
        for name in METHODS_SPLIT:
            mode, w = name.split('_')[1], float(name.split('_p')[-1].replace('_cc', ''))
            protos, info = text_protos_joint(sp, mode)
            for which in ('seen', 'unseen'):
                tag = f'{name}_split{sp}' + ('' if which == 'seen' else '_unseen')
                if load_cuts(tag) is not None and not args.force:
                    continue
                t0 = time.time()
                cuts = trunk_align(sp, which, protos, w, colcenter=name.endswith('_cc'))
                save_cuts(tag, cuts, info)
                print(f'  [labels] {tag}: {len(cuts)} segments cut ({time.time() - t0:.0f}s) {info}', flush=True)
        # json files
        for name in METHODS_SPLIT + METHODS_CLASS[:-2]:
            for which, classes in (('seen', seen), ('unseen', unseen)):
                suffix = '' if which == 'seen' else '_unseen'
                cuts = load_cuts(f'{name}_split{sp}{suffix}' if name in METHODS_SPLIT else name)
                write_json(f'{OUTDIR}/{name}_split{sp}{suffix}.json', cuts, set(classes),
                           dict(split=sp, method=name, classes=which,
                                generator='scripts/gen_phase_pseudo_v2.py'))
        print(f'  [labels] split {sp}: json files written to {OUTDIR}', flush=True)


# ------------------------------------------------------------ evaluation
def eval_population(sp, sets):
    seen, _ = split_classes(sp)
    keys = [s['key'] for s in SEGS if s['label'] in set(seen) and s['L'] >= MIN_EVAL_SEC]
    keys = [k for k in keys if all(k in c for c in sets.values())]
    return keys


def dev_from_uniform(cuts, keys):
    d = np.concatenate([np.abs(cuts[k] - uniform_cuts(SEG_OF[k])) / SEG_OF[k]['L'] for k in keys])
    return float(d.mean()), float(np.percentile(d, 90))


def agreement(a, b, keys):
    return float(np.mean([np.abs(a[k] - b[k]).mean() / SEG_OF[k]['L'] for k in keys]))


def dev_corr(a, b, keys):
    """Pearson correlation, over all 9 cuts of all segments, between the
    two sets' deviations from the equal-width tenth (cut - tenth)/L.  Zero
    for unrelated cuts (and for equal-width cuts), positive when the two
    sets move the same cuts the same way."""
    da = np.concatenate([(a[k] - uniform_cuts(SEG_OF[k])) / SEG_OF[k]['L'] for k in keys])
    db = np.concatenate([(b[k] - uniform_cuts(SEG_OF[k])) / SEG_OF[k]['L'] for k in keys])
    if da.std() < 1e-9 or db.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


_FEATS = {}


def feat_cached(key, kind):
    if (key, kind) not in _FEATS:
        _FEATS[(key, kind)] = seg_feat(SEG_OF[key], kind)
    return _FEATS[(key, kind)]


def split_half(cuts, keys, kind):
    """Original K2: per class, videos interleaved into halves, per-half
    phase prototypes, mean same-phase minus mean cross-phase cosine."""
    by_cls = defaultdict(list)
    for k in keys:
        by_cls[SEG_OF[k]['label']].append(k)
    vals = []
    for c, ks in sorted(by_cls.items()):
        vids = sorted({k[0] for k in ks})
        if len(vids) < 4:
            continue
        h = {v: i % 2 for i, v in enumerate(vids)}
        P = []
        ok = True
        for half in (0, 1):
            F, L = [], []
            for k in ks:
                if h[k[0]] != half:
                    continue
                tt, x = feat_cached(k, kind)
                if tt is None:
                    continue
                F.append(x)
                L.append((tt[:, None] >= cuts[k][None]).sum(1))
            if not F:
                ok = False
                break
            F = np.concatenate(F)
            L = np.concatenate(L)
            pr = []
            for kk in range(K):
                m = L == kk
                if m.sum() == 0:
                    ok = False
                    break
                v = F[m].mean(0)
                pr.append(v / (np.linalg.norm(v) + 1e-8))
            if not ok:
                break
            P.append(np.stack(pr))
        if not ok:
            continue
        S = P[0] @ P[1].T
        vals.append(float(np.diag(S).mean() - (S.sum() - np.trace(S)) / (K * K - K)))
    return float(np.mean(vals)), len(vals)


def probe(sp, cuts_seen, cuts_unseen, seeds, device='cpu', restrict=None):
    """(4) transfer probe; restrict = optional array of SEGS indices the
    training rows are limited to (pilot table)."""
    pp, pt, ppr, load_config, make_dataset = _probe_modules()
    tr = load_trunk(sp)
    seen, unseen = split_classes(sp)
    ytr = row_labels(tr['train'], cuts_seen, set(seen))
    if restrict is not None:
        ytr[~np.isin(tr['train']['seg'], restrict)] = -1
    mtr = ytr >= 0
    te = tr['test']
    y_seen = row_labels(te, cuts_seen, set(seen))
    y_uns = row_labels(te, cuts_unseen, set(unseen))
    key = np.stack([te['video'], te['in_seg']], 1)
    res = []
    for seed in seeds:
        head = pp.train_head(tr['train']['feat'][mtr], ytr[mtr], seed, 256, device=device)
        pa, pe = pp.predict(head, te['feat'], device)
        r = {}
        for name, y in (('seen', y_seen), ('unseen', y_uns)):
            m = y >= 0
            sp_ = pp.spearman_metrics(pa[m], pe[m], te['tnorm'][m], key[m])
            r[name] = dict(acc10=float((pa[m] == y[m]).mean()),
                           acc3=float((pp.MERGE3[pa[m]] == pp.MERGE3[y[m]]).mean()),
                           rho_expect=sp_['expect']['per_seg_mean'],
                           rho_argmax=sp_['argmax']['per_seg_mean'], n=int(m.sum()))
        tr_arg, _ = pp.predict(head, tr['train']['feat'][mtr], device)
        r['train_acc10'] = float((tr_arg == ytr[mtr]).mean())
        r['n_train'] = int(mtr.sum())
        res.append(r)
    return pp.average_dicts(res)


def row_labels(rows, cuts, want):
    """Per-row phase from a cut set (uniform fallback where the set has no
    cuts for the segment, as the json files do); -1 outside `want`."""
    y = np.full(len(rows['seg']), -1, np.int64)
    for si in np.unique(rows['seg']):
        if si < 0:
            continue
        seg = SEGS[si]
        if seg['label'] not in want:
            continue
        c = cuts.get(seg['key'])
        if c is None:
            c = uniform_cuts(seg)
        m = rows['seg'] == si
        y[m] = (rows['t_sec'][m][:, None] >= c[None]).sum(1)
    return y


def center_per_segment(rows):
    X = rows['feat'].copy()
    for si in np.unique(rows['seg']):
        if si < 0:
            continue
        m = rows['seg'] == si
        X[m] -= X[m].mean(0, keepdims=True)
    return X


_CENTERED = {}


def lda_sep(sp, cuts_seen, cuts_unseen, restrict=None):
    """(E) separability of the phase labels in the trunk level-0 features
    after centring every row on its GT segment's mean feature (= within-
    action scatter only).  LDA (10 phases, eigen solver, automatic
    shrinkage) on the training rows; held-out 10-way accuracy on the seen /
    unseen test rows (centred the same way); ratio of between-phase to
    within-phase variance along the first discriminant direction on the
    training rows and on the held-out seen / unseen rows."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    tr = load_trunk(sp)
    seen, unseen = split_classes(sp)
    if sp not in _CENTERED:
        _CENTERED[sp] = dict(train=center_per_segment(tr['train']), test=center_per_segment(tr['test']))
    Xtr, Xte = _CENTERED[sp]['train'], _CENTERED[sp]['test']
    ytr = row_labels(tr['train'], cuts_seen, set(seen))
    if restrict is not None:
        ytr[~np.isin(tr['train']['seg'], restrict)] = -1
    mtr = ytr >= 0
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto').fit(Xtr[mtr], ytr[mtr])
    w = lda.scalings_[:, 0]

    def ratio(X, y):
        p = X @ w
        mu = p.mean()
        ks = np.unique(y)
        between = sum((y == k).sum() * (p[y == k].mean() - mu) ** 2 for k in ks) / len(p)
        within = sum(((p[y == k] - p[y == k].mean()) ** 2).sum() for k in ks) / len(p)
        return float(between / max(within, 1e-12))

    out = dict(lda_ratio_train=ratio(Xtr[mtr], ytr[mtr]),
               lda_train_acc=float(lda.score(Xtr[mtr], ytr[mtr])), n_train=int(mtr.sum()))
    for name, cuts, want in (('seen', cuts_seen, set(seen)), ('unseen', cuts_unseen, set(unseen))):
        y = row_labels(tr['test'], cuts, want)
        m = y >= 0
        out[f'lda_acc_{name}'] = float(lda.score(Xte[m], y[m]))
        out[f'lda_ratio_{name}'] = ratio(Xte[m], y[m])
    return out


COLS = [('dev_mean', 'dev%'), ('dev_p90', 'p90%'), ('k2_vmae', 'K2vmae'), ('k2_i3d', 'K2i3d'),
        ('agree_c_vmae', 'agCvm%'), ('agree_c_i3d', 'agCi3d%'), ('corr_c_vmae', 'rCvm'), ('corr_c_i3d', 'rCi3d'),
        ('acc10_seen', 'accS'),
        ('acc10_unseen', 'accU'), ('rho_seen', 'rhoS'), ('rho_unseen', 'rhoU'),
        ('lda_acc_seen', 'ldaS'), ('lda_acc_unseen', 'ldaU'), ('lda_ratio_train', 'ldaR'),
        ('lda_ratio_seen', 'ldaRS')]


def fmt_table(rows, title):
    lines = [f'==== {title} ====',
             f"{'label set':16s} " + ' '.join(f'{h:>7s}' for _, h in COLS) + '   n_eval  n_train']
    for name, r in rows.items():
        cells = []
        for k, _ in COLS:
            v = r.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append(f"{'-':>7s}")
            elif k.startswith('dev') or k.startswith('agree'):
                cells.append(f'{100 * v:7.1f}')
            else:
                cells.append(f'{v:7.3f}')
        lines.append(f'{name:16s} ' + ' '.join(cells) + f"  {r.get('n_eval', 0):7.0f} {r.get('n_train', 0):8.0f}")
    return '\n'.join(lines)


def eval_set(sp, n, sets, sets_u, keys, args, restrict=None):
    seen, unseen = split_classes(sp)
    c = sets[n]
    r = {}
    r['dev_mean'], r['dev_p90'] = dev_from_uniform(c, keys)
    r['k2_vmae'], r['k2_n'] = split_half(c, keys, 'vmae')
    r['k2_i3d'], _ = split_half(c, keys, 'i3d_both')
    r['agree_c_vmae'] = agreement(c, sets['C_vmae'], keys)
    r['agree_c_i3d'] = agreement(c, sets['C_i3d_both'], keys)
    r['agree_uniform'] = agreement(c, sets['uniform'], keys)
    r['corr_c_vmae'] = dev_corr(c, sets['C_vmae'], keys)
    r['corr_c_i3d'] = dev_corr(c, sets['C_i3d_both'], keys)
    r['n_eval'] = len(keys)
    r['n_cut_seen'] = sum(1 for s in SEGS if s['label'] in set(seen) and s['key'] in c)
    r['n_cut_unseen'] = sum(1 for s in SEGS if s['label'] in set(unseen) and s['key'] in sets_u[n])
    pr = probe(sp, c, sets_u[n], args.seeds, args.device, restrict=restrict)
    r.update(acc10_seen=pr['seen']['acc10'], acc3_seen=pr['seen']['acc3'],
             acc10_unseen=pr['unseen']['acc10'], acc3_unseen=pr['unseen']['acc3'],
             rho_seen=pr['seen']['rho_expect'], rho_unseen=pr['unseen']['rho_expect'],
             rho_argmax_seen=pr['seen']['rho_argmax'], rho_argmax_unseen=pr['unseen']['rho_argmax'],
             train_acc10=pr['train_acc10'], n_train=pr['n_train'],
             n_test_seen=pr['seen']['n'], n_test_unseen=pr['unseen']['n'])
    r.update(lda_sep(sp, c, sets_u[n], restrict=restrict))
    print(f'  {n:16s} dev {100 * r["dev_mean"]:5.1f}/{100 * r["dev_p90"]:5.1f}  K2 {r["k2_vmae"]:.3f}/{r["k2_i3d"]:.3f}  '
          f'agree {100 * r["agree_c_vmae"]:.1f}/{100 * r["agree_c_i3d"]:.1f}  probe {r["acc10_seen"]:.3f}/{r["acc10_unseen"]:.3f} '
          f'rho {r["rho_seen"]:.3f}/{r["rho_unseen"]:.3f}  lda {r["lda_acc_seen"]:.3f}/{r["lda_acc_unseen"]:.3f} '
          f'ratio {r["lda_ratio_train"]:.3f}/{r["lda_ratio_seen"]:.3f}', flush=True)
    return r


def stage_eval(args):
    pp, *_ = _probe_modules()
    summary = dict(meta=dict(
        generator='scripts/gen_phase_pseudo_v2.py', seeds=args.seeds, min_eval_sec=MIN_EVAL_SEC,
        columns={
            'dev_mean/dev_p90': 'mean / p90 |cut - equal-width tenth| as a fraction of the segment length, seen classes, eval population',
            'k2_vmae/k2_i3d': 'split-half prototype consistency (same-phase cos minus cross-phase cos across video halves), mean over seen classes, on VMAE-s2 / raw I3D both streams',
            'agree_c_vmae/agree_c_i3d': 'mean |cut difference| / length vs the change-point cuts C_vmae / C_i3d_both (the cross-FEATURE column is the one whose feature the label set did not use)',
            'acc10_seen/unseen': 'transfer probe (phase_probe.train_head on champion level-0 trunk features, train videos, relabelled) 10-way accuracy on held-out seen / unseen test segments relabelled by the same method',
            'rho_seen/unseen': 'per-segment Spearman of the expected phase vs normalised time',
            'lda_acc_seen/unseen': '(E) LDA (10 phases, eigen solver, auto shrinkage) on the training rows centred per GT segment (within-action scatter only); held-out 10-way accuracy on the seen / unseen test rows centred the same way',
            'lda_ratio_train/seen/unseen': 'between-phase / within-phase variance along the first LDA direction, on the training rows / held-out seen / unseen rows',
            'agree_uniform': 'mean |cut - equal-width tenth| / length = dev_mean (kept for symmetry)',
            'corr_c_vmae/corr_c_i3d': 'Pearson correlation over all cuts between the set\'s deviation from the equal-width tenth and that of C_vmae / C_i3d_both: 0 = unrelated cuts, >0 = the two sets move the same cuts the same way (the cross-FEATURE column is the evidence for visual events)',
            'n_eval': 'segments of the eval population (seen classes, >= 1.5 s, cut by every method)',
            'n_train': 'probe training rows',
            'pilot': '(D) qwenvl pilot: every set re-scored on the pilot segments only, probe and LDA trained on the pilot rows only'}),
        splits={}, mean={})
    cls_sets = {n: load_cuts(n) for n in METHODS_CLASS}
    if args.pilot_only and os.path.exists(f'{OUTDIR}/summary.json'):
        summary = json.load(open(f'{OUTDIR}/summary.json'))
    for sp in args.splits:
        t0 = time.time()
        seen, unseen = split_classes(sp)
        sets, sets_u = {}, {}
        sets['current'] = current_cuts(sp)
        sets_u['current'] = current_cuts(sp, unseen=True)
        for n in METHODS_CLASS:
            sets[n] = cls_sets[n]
            sets_u[n] = cls_sets[n]
        for n in METHODS_SPLIT:
            sets[n] = load_cuts(f'{n}_split{sp}')
            sets_u[n] = load_cuts(f'{n}_split{sp}_unseen')
        keys = eval_population(sp, sets)
        print(f'===== split {sp}: eval population {len(keys)} seen segments =====', flush=True)
        if not args.pilot_only:
            rows = {}
            for n in METHODS:
                rows[n] = eval_set(sp, n, sets, sets_u, keys, args)
            summary['splits'][str(sp)] = rows
            print(fmt_table(rows, f'split {sp}') + f'\n  ({time.time() - t0:.0f}s)', flush=True)
            json.dump(summary, open(f'{OUTDIR}/summary.json', 'w'), indent=1)
        if args.skip_pilot:
            continue
        # ---- (D) pilot table: the VLM label set exists only for a pilot
        # subset; every set is re-scored on exactly those segments and the
        # probe / LDA are retrained on the pilot rows only.
        qv = load_cuts(f'qwenvl_split{sp}')
        if qv is not None:
            sets_p = dict(sets, qwenvl=qv)
            sets_up = dict(sets_u, qwenvl={})
            keys_p = [k for k in keys if k in qv]
            seg_index = {s['key']: i for i, s in enumerate(SEGS)}
            restrict = np.array([seg_index[k] for k in keys_p])
            print(f'===== split {sp}: PILOT population {len(keys_p)} segments (qwenvl) =====', flush=True)
            prow = {}
            for n in ['qwenvl'] + METHODS:
                prow[n] = eval_set(sp, n, sets_p, sets_up, keys_p, args, restrict=restrict)
            summary.setdefault('pilot', {})[str(sp)] = prow
            print(fmt_table(prow, f'split {sp} PILOT subset (probe/LDA trained on the pilot rows only; '
                                  f'unseen columns use the full unseen sets, none for qwenvl)'), flush=True)
            json.dump(summary, open(f'{OUTDIR}/summary.json', 'w'), indent=1)
    keys = list(summary['splits'])
    if len(keys) > 1 and not args.pilot_only:
        summary['mean'] = pp.average_dicts([summary['splits'][k] for k in keys])
        print(fmt_table(summary['mean'], f'MEAN over splits {keys}'), flush=True)
    json.dump(summary, open(f'{OUTDIR}/summary.json', 'w'), indent=1)
    print(f'[save] {OUTDIR}/summary.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['clip', 'labels', 'eval', 'all'])
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--max-videos', type=int, default=-1, help='debug: cap the CLIP extraction')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--force', action='store_true', help='recompute cached label sets')
    ap.add_argument('--only', nargs='*', default=None, help='debug: restrict the label stage to these sets')
    ap.add_argument('--skip-pilot', action='store_true', help='eval: no qwenvl pilot table')
    ap.add_argument('--pilot-only', action='store_true', help='eval: only the qwenvl pilot table (merged into summary.json)')
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.stage in ('clip', 'all'):
        stage_clip(args)
    if args.stage in ('labels', 'all'):
        if args.stage == 'labels':
            args.device = 'cpu'
        stage_labels(args)
    if args.stage in ('eval', 'all'):
        args.device = 'cpu'
        stage_eval(args)
