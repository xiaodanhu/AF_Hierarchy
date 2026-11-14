"""VSA DIAGNOSIS (analysis only, no training): why did the video-side
subtree attention (libs/modeling/vsa.py, TIFAD_VSA=1) not improve detection?

Compares the stage-3 vsa3 checkpoints (ckpt/thumos14_i3d_vsa3_{n}) with the
champion stage-2 seqchain checkpoints (ckpt/thumos14_i3d_seqchain_{n}) that
they were trained on top of (everything frozen except vsa.*).

Per split (deep set, default 0 1 2):
  Q1 near-identity : ||v_n - f_n|| / ||f_n|| and cos(v_n, f_n) at unseen-GT /
                     seen-GT / background locations (test videos, test vocab);
                     + the change the residual makes to the cls logits.
  Q2 attention     : clip_attn self-vs-chunk weights, per-chunk profile,
                     entropy; chain_attn self-vs-neighbour; split by the
                     predicted segment length (<2s, 2-6s, >6s).
  Q3 seen/unseen   : unfused (and fused) mAP on the unseen classes (standard
                     eval) and on the SEEN-class test segments with the seen
                     vocabulary (TIFAD_PROTO_COLLECT=1 control), both models.
  Q4 fragments     : per unseen GT, top-scoring vs highest-IoU overlapping
                     detection (fragment-outranks-full rate, duration ratio,
                     recall@1 at tIoU 0.5), both models.
Per split (all splits, default 0..9), cheap:
  Q5 per-class AP  : fused + unfused per-class AP of both models; aggregated
                     by class name with mean GT duration / phases-per-second.
  Q6 training loss : cls_loss at the start vs end of stage 3 (logs/vsa3_s{n})
                     and stage 2 (logs/tifad_seqchain_75_s{n}); mAP curves.
  weights          : out_proj / in_proj norms of vsa.clip_attn / chain_attn.

Usage (GPU 2 or 0/1; NEVER GPU 3):
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=2 \
      python scripts/vsa_diag.py --deep-splits 0 1 2 --splits 0 1 2 3 4 5 6 7 8 9
Output: data/vsa_diag.json (+ a printed summary; --report-only re-prints).
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

# ---- champion (CH) envs; train-only randomness OFF (both are gated on the
# TEXT mode == 'train', which the seen-vocab control forces in eval, so they
# MUST be 0 here; for the standard unseen eval they are inert either way).
# TIFAD_VSA is toggled per model build (read at construction).
CH_ENVS = {
    'TIFAD_PROMPT_ENSEMBLE': 'configs/thumos14_qwen_ensemble.json',
    'TIFAD_ATTR_ATTN': '1',
    'TIFAD_PROMPT_DROPOUT': '0',
    'TIFAD_SEM_DEGRADE': '0',
    'TIFAD_FAMCE': '0.3',
    'TIFAD_FAMCE_MAP': 'configs/thumos14_fam_map.json',
    'TIFAD_FAMCE_UW': '1',
    'TIFAD_ATTNDESIGN': 'phasechain',
}
for _k, _v in CH_ENVS.items():
    os.environ[_k] = _v
for _k in ('TIFAD_VSA', 'TIFAD_FUSE_ALPHA', 'TIFAD_DURPRIOR', 'TIFAD_PROTO_COLLECT',
           'TIFAD_PERCLASS_AP', 'TIFAD_DUMP_PREDS', 'TIFAD_DUMP_RAW'):
    os.environ.pop(_k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.modeling.text import data_split_dir
from libs.utils import ANETdetection, fix_random_seed
from libs.utils.postprocessing import postprocess_results

FUSED = {'TIFAD_FUSE_ALPHA': '0.35', 'TIFAD_DURPRIOR': '0.3'}
LEN_BINS = [('lt2s', 0.0, 2.0), ('2to6s', 2.0, 6.0), ('gt6s', 6.0, 1e9)]
CKPT = {'champ': 'ckpt/thumos14_i3d_seqchain_{n}/mAP_best.pth.tar',
        'vsa3': 'ckpt/thumos14_i3d_vsa3_{n}/mAP_best.pth.tar'}
LOGS = {'vsa3': 'logs/vsa3_s{n}.log', 'seqchain': 'logs/tifad_seqchain_75_s{n}.log'}
FUSED_LOGS = {'vsa3': 'logs/vsa3e_s{n}.log', 'champ': 'logs/seqchfuse2_s{n}.log'}


# ------------------------------------------------------------------ helpers
def load_classes(cfg, mode, n):
    p = data_split_dir(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], mode, n)
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()], p


def build_model(cfg, n, ckpt_file, vsa):
    if vsa:
        os.environ['TIFAD_VSA'] = '1'
    else:
        os.environ.pop('TIFAD_VSA', None)
    model = make_meta_arch(n, cfg['model_name'], cfg['dataset']['subset_file'],
                           cfg['dataset']['data_split'], **cfg['model'])
    ck = torch.load(ckpt_file, map_location='cpu')
    sd = {k[len('module.'):] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model = model.cuda().eval()
    return model, int(ck.get('epoch', -1))


def make_val_dataset(cfg, n, labels=None):
    ds = make_dataset(cfg['dataset_name'], n, False, cfg['val_split'], **cfg['dataset'])
    if labels is not None:
        ds.label_dict = None
        ds.subset_labels = list(labels)
        ds.data_list, ds.label_dict = ds._load_json_db(ds.json_file)
    return ds


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p10': pct(x, 10), 'p90': pct(x, 90)}


def seg_iou(gt, dets):
    """gt (2,), dets (N, 2) -> IoU (N,)"""
    inter = np.maximum(0.0, np.minimum(gt[1], dets[:, 1]) - np.maximum(gt[0], dets[:, 0]))
    union = (gt[1] - gt[0]) + (dets[:, 1] - dets[:, 0]) - inter
    return inter / np.maximum(union, 1e-8)


# --------------------------------------------------------- per-video VSA stats
class VSACollector:
    """Wraps model.vsa.forward to stash inputs / outputs; recomputes the
    attention weights (need_weights=True) from the stashed inputs."""

    def __init__(self, model):
        self.model = model
        self.vsa = model.vsa
        self.last = None
        self._orig = self.vsa.forward
        self.vsa.forward = self._wrapped
        self.text = None
        model.backbone.register_forward_hook(self._bb_hook)

    def _bb_hook(self, m, inp, out):
        self.text = out[2].detach()             # (B, C_all, D) updated text

    def _wrapped(self, feats, offsets, fpn_masks):
        out = self._orig(feats, offsets, fpn_masks)
        self.last = {'feats': [f.detach() for f in feats],
                     'offsets': [o.detach() for o in offsets],
                     'masks': [m.detach() for m in fpn_masks],
                     'out': [o.detach() for o in out]}
        return out

    @torch.no_grad()
    def attn_weights(self, x, off, mask):
        """x (1,D,T), off (1,T,2), mask (1,1,T) -> head-averaged clip weights
        (T, K+1), per-head clip self weight (T, H), chain weights (T, K, K)."""
        vsa = self.vsa
        B, D, T = x.shape
        K = vsa.K
        Z = vsa._pool_chunks(x, off, mask).reshape(T, K, D)
        _u, wc = vsa.chain_attn(Z, Z, Z, attn_mask=vsa.band_mask, need_weights=True,
                                average_attn_weights=True)
        Z = Z + _u
        f = x.transpose(1, 2).reshape(T, 1, D)
        kv = torch.cat([f, Z], dim=1)
        _, wa_h = vsa.clip_attn(f, kv, kv, need_weights=True, average_attn_weights=False)
        wa_h = wa_h[:, :, 0, :]                          # (T, H, K+1)
        return wa_h.mean(1), wa_h[:, :, 0], wc


def logits_from(feat, text_n, bias, scale):
    """feat (D,T), text_n (C,D) normalized, bias (C,) -> (T,C)"""
    return feat.t() @ text_n.t() / scale + bias.unsqueeze(0)


@torch.no_grad()
def video_vsa_stats(model, col, item, gt_grid, gt_lab, gt_unseen, gt_cls_idx, fps, fstr,
                    points, acc):
    """Accumulate Q1/Q2 rows for one video from col.last."""
    st = col.last
    n_cls = len(model.text_features.test_classes)
    text = col.text[0, :n_cls]
    text_n = text / text.norm(dim=-1, keepdim=True)
    bias = text_n @ model.bias_lang + model.bias0
    scale = model.logit_scale.exp()
    for l, (f, v, off, m, p) in enumerate(zip(st['feats'], st['out'], st['offsets'],
                                              st['masks'], points)):
        valid = m[0, 0].bool()
        if valid.sum() == 0:
            continue
        f0, v0 = f[0], v[0]                              # (D, T)
        d = v0 - f0
        rel = d.norm(dim=0) / f0.norm(dim=0).clamp(min=1e-8)
        cos = F.cosine_similarity(v0, f0, dim=0)
        t = p[:, 0]
        stride = p[:, 3]
        seglen = (off[0, :, 0] + off[0, :, 1]) * stride * fstr / fps      # seconds
        # location category from GT (grid coords): inside -> shortest seg
        cat = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)  # 0 bg
        gcls = torch.full((t.shape[0],), -1, dtype=torch.long, device=t.device)
        if gt_grid.shape[0] > 0:
            g = torch.as_tensor(gt_grid, device=t.device)
            inside = (t[:, None] > g[None, :, 0]) & (t[:, None] < g[None, :, 1])
            lens = torch.where(inside, (g[:, 1] - g[:, 0])[None, :],
                               torch.full_like(inside, float('inf'), dtype=torch.float))
            j = lens.argmin(1)
            ok = torch.isfinite(lens[torch.arange(t.shape[0]), j])
            uns = torch.as_tensor(gt_unseen, device=t.device)[j]
            cat = torch.where(ok & uns, torch.full_like(cat, 1),
                              torch.where(ok, torch.full_like(cat, 2), cat))
            gcls = torch.where(ok & uns, torch.as_tensor(gt_cls_idx, device=t.device)[j], gcls)
        # logits before / after
        lf = logits_from(f0, text_n, bias, scale)        # (T, C)
        lv = logits_from(v0, text_n, bias, scale)
        wa, wa_self_h, wc = col.attn_weights(f, off, m)
        ent = -(wa * (wa + 1e-12).log()).sum(-1)
        wmax = wa.max(-1).values
        wc_self = torch.diagonal(wc, dim1=-2, dim2=-1).mean(-1)
        idx = valid.nonzero(as_tuple=True)[0]
        rows = {
            'level': torch.full_like(idx, l), 'cat': cat[idx], 'rel': rel[idx], 'cos': cos[idx],
            'seglen': seglen[idx], 'wa_self': wa[idx, 0], 'wa_chunks': wa[idx, 1:],
            'wa_self_h': wa_self_h[idx], 'ent': ent[idx], 'wc_self': wc_self[idx],
            'dnorm': d.norm(dim=0)[idx], 'fnorm': f0.norm(dim=0)[idx], 'wmax': wmax[idx],
        }
        # linearity probe rows (subsample): residual r vs [f; mean chunk; 1]
        sub = idx[torch.rand(idx.shape[0], device=idx.device) < 0.05]
        if sub.numel() > 0:
            zmean = col.vsa._pool_chunks(f, off, m)[0].mean(1)         # (T, D)
            acc['lin_f'].append(f0.t()[sub].float().cpu().numpy())
            acc['lin_z'].append(zmean[sub].float().cpu().numpy())
            acc['lin_r'].append(d.t()[sub].float().cpu().numpy())
        # GT-class logit change at unseen positives; max-logit change elsewhere
        gi = gcls[idx]
        has = gi >= 0
        ar = torch.arange(idx.shape[0], device=t.device)
        lf_i, lv_i = lf[idx], lv[idx]
        gt_f = torch.where(has, lf_i[ar, gi.clamp(min=0)], torch.zeros_like(rel[idx]))
        gt_v = torch.where(has, lv_i[ar, gi.clamp(min=0)], torch.zeros_like(rel[idx]))
        oth_f = lf_i.clone()
        oth_v = lv_i.clone()
        oth_f[ar[has], gi[has]] = -1e4
        oth_v[ar[has], gi[has]] = -1e4
        rows['gt_logit_f'] = gt_f
        rows['gt_logit_v'] = gt_v
        rows['margin_f'] = gt_f - oth_f.max(1).values
        rows['margin_v'] = gt_v - oth_v.max(1).values
        rows['max_logit_f'] = lf_i.max(1).values
        rows['max_logit_v'] = lv_i.max(1).values
        rows['max_prob_f'] = lf_i.max(1).values.sigmoid()
        rows['max_prob_v'] = lv_i.max(1).values.sigmoid()
        for k, val in rows.items():
            acc[k].append(val.float().cpu().numpy() if val.is_floating_point()
                          else val.cpu().numpy())


def linear_fit(acc):
    """How much of the residual r = v - f is a LINEAR function of the
    location feature and the mean-pooled candidate segment (which is what the
    module reduces to when the attention is uniform)?  Held-out R^2."""
    if 'lin_r' not in acc:
        return None
    Fm = np.concatenate(acc['lin_f'], 0).astype(np.float64)
    Zm = np.concatenate(acc['lin_z'], 0).astype(np.float64)
    Rm = np.concatenate(acc['lin_r'], 0).astype(np.float64)
    n = len(Rm)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    out = {'n': int(n)}
    for name, X in (('f_only', Fm), ('zmean_only', Zm), ('f_and_zmean', np.concatenate([Fm, Zm], 1))):
        X1 = np.concatenate([X, np.ones((n, 1))], 1)
        lam = 1e-3 * n
        A = X1[tr].T @ X1[tr] + lam * np.eye(X1.shape[1])
        W = np.linalg.solve(A, X1[tr].T @ Rm[tr])
        pred = X1[te] @ W
        ss_res = ((Rm[te] - pred) ** 2).sum()
        ss_tot = ((Rm[te] - Rm[tr].mean(0)) ** 2).sum()
        out[name + '_R2'] = float(1 - ss_res / ss_tot)
    # how much of r is a CONSTANT direction (mean residual)?
    mu = Rm[tr].mean(0)
    out['mean_residual_norm_over_mean_norm_of_residual'] = float(
        np.linalg.norm(Rm.mean(0)) / np.linalg.norm(Rm, axis=1).mean())
    out['cos_residual_to_mean_residual'] = float(np.mean(
        (Rm @ mu) / (np.linalg.norm(Rm, axis=1) * np.linalg.norm(mu) + 1e-8)))
    return out


def summarize_vsa(acc):
    lin = linear_fit(acc)
    A = {k: np.concatenate(v, 0) for k, v in acc.items() if not k.startswith('lin_')}
    cat = A['cat']
    out = {'n_locations': int(len(cat)),
           'n_by_cat': {'unseen_gt': int((cat == 1).sum()), 'seen_gt': int((cat == 2).sum()),
                        'background': int((cat == 0).sum())}}
    q1 = {}
    for name, c in (('unseen_gt', 1), ('seen_gt', 2), ('background', 0), ('all', None)):
        m = np.ones_like(cat, bool) if c is None else (cat == c)
        q1[name] = {'rel_residual': stats(A['rel'][m]), 'cosine': stats(A['cos'][m]),
                    'residual_norm': stats(A['dnorm'][m]), 'feat_norm': stats(A['fnorm'][m])}
    # logit effect
    pos = cat == 1
    q1['logit_change'] = {
        'unseen_gt_class_logit_delta': stats(A['gt_logit_v'][pos] - A['gt_logit_f'][pos]),
        'unseen_gt_margin_f': stats(A['margin_f'][pos]),
        'unseen_gt_margin_v': stats(A['margin_v'][pos]),
        'unseen_gt_margin_delta': stats(A['margin_v'][pos] - A['margin_f'][pos]),
        'background_max_logit_delta': stats(A['max_logit_v'][cat == 0] - A['max_logit_f'][cat == 0]),
        'background_max_prob_f': stats(A['max_prob_f'][cat == 0]),
        'background_max_prob_v': stats(A['max_prob_v'][cat == 0]),
        'seen_gt_max_logit_delta': stats(A['max_logit_v'][cat == 2] - A['max_logit_f'][cat == 2]),
    }
    q1['linearity'] = lin
    out['q1'] = q1
    # per level
    out['q1_rel_by_level'] = {int(l): stats(A['rel'][A['level'] == l])
                              for l in np.unique(A['level'])}
    # Q2
    K = A['wa_chunks'].shape[1]
    uniform_ent = math.log(K + 1)

    def q2_block(m):
        if m.sum() == 0:
            return {'n': 0}
        return {'n': int(m.sum()),
                'clip_self_weight': stats(A['wa_self'][m]),
                'clip_chunks_total_weight': stats(1.0 - A['wa_self'][m]),
                'clip_per_chunk_mean': A['wa_chunks'][m].mean(0).tolist(),
                'clip_per_head_self_weight_mean': A['wa_self_h'][m].mean(0).tolist(),
                'clip_entropy': stats(A['ent'][m]),
                'clip_max_weight': stats(A['wmax'][m]),
                'clip_entropy_uniform': uniform_ent,
                'chain_self_weight': stats(A['wc_self'][m]),
                'chain_neighbour_weight': stats(1.0 - A['wc_self'][m]),
                'seglen_sec': stats(A['seglen'][m])}
    q2 = {'all': q2_block(np.ones_like(cat, bool)), 'unseen_gt': q2_block(cat == 1),
          'background': q2_block(cat == 0), 'by_seglen': {}, 'by_seglen_unseen_gt': {}}
    for name, lo, hi in LEN_BINS:
        m = (A['seglen'] >= lo) & (A['seglen'] < hi)
        q2['by_seglen'][name] = q2_block(m)
        q2['by_seglen_unseen_gt'][name] = q2_block(m & (cat == 1))
    q2['by_level'] = {int(l): q2_block(A['level'] == l) for l in np.unique(A['level'])}
    out['q2'] = q2
    return out


# ------------------------------------------------------------- eval passes
@torch.no_grad()
def run_pass(model, loader, class_names, col=None, stats_fn=None):
    results = {k: [] for k in ('video-id', 't-start', 't-end', 'label', 'score', 'class_names')}
    model.eval()
    for vl in loader:
        out = model(vl)
        for i, o in enumerate(out):
            if o['segments'].shape[0] > 0:
                results['video-id'].extend([o['video_id']] * o['segments'].shape[0])
                results['t-start'].append(o['segments'][:, 0])
                results['t-end'].append(o['segments'][:, 1])
                results['label'].append(o['labels'])
                results['score'].append(o['scores'])
                results['class_names'].append(list(class_names))
        if stats_fn is not None:
            stats_fn(vl[0])
    for k in ('t-start', 't-end', 'label', 'score'):
        results[k] = torch.cat(results[k]).numpy()
    return results


def evaluate(results, evaluator, cfg, fused):
    for k, v in FUSED.items():
        if fused:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    res = postprocess_results(dict(results), cfg['test_cfg']['ext_score_file'],
                              cfg['opt']['prior_prob'], cfg['opt']['num_pred'])
    for k in FUSED:
        os.environ.pop(k, None)
    # verbose=True is required (the evaluator only builds the tIoU list there)
    _, mAP, tiou_list = evaluator.evaluate(res, verbose=True)
    id2name = {v: k for k, v in evaluator.label_dict.items()}
    per_class = {id2name[lid]: float(evaluator.ap[:, c].mean() * 100)
                 for lid, c in evaluator.activity_index.items()}
    return float(mAP * 100), tiou_list, per_class, res


def fragment_analysis(det, gt_df, label_dict):
    """det: post-processed dict (video-id, t-start, t-end, label, score);
    gt_df: evaluator ground truth with ORIGINAL label ids (subset index)."""
    vids = np.asarray(det['video-id'])
    segs = np.stack([det['t-start'], det['t-end']], 1)
    labs = np.asarray(det['label']).astype(int)
    sc = np.asarray(det['score'])
    by_vc = defaultdict(list)
    for i, (v, l) in enumerate(zip(vids, labs)):
        by_vc[(v, l)].append(i)
    n = 0
    n_overlap = 0
    n_frag_outranks = 0
    n_top_ge05 = 0
    n_any_ge05 = 0
    n_top1_vc = 0      # AR@1-style: the single top-scoring det of (video, class)
    ratios, top_ious, best_ious = [], [], []
    per_class = defaultdict(lambda: [0, 0, 0, 0.0])   # n, frag, top>=.5, ratio sum
    top1_cache = {}
    for _, r in gt_df.iterrows():
        n += 1
        key = (r['video-id'], int(r['label']))
        g = np.array([r['t-start'], r['t-end']])
        ii = by_vc.get(key, [])
        pc = per_class[int(r['label'])]
        pc[0] += 1
        if key not in top1_cache:
            top1_cache[key] = (ii[int(np.argmax(sc[ii]))] if ii else None)
        t1 = top1_cache[key]
        if t1 is not None and seg_iou(g, segs[[t1]])[0] >= 0.5:
            n_top1_vc += 1
        if not ii:
            continue
        ii = np.asarray(ii)
        ious = seg_iou(g, segs[ii])
        ov = ious > 0
        if not ov.any():
            continue
        n_overlap += 1
        ii, ious = ii[ov], ious[ov]
        top = ii[np.argmax(sc[ii])]
        top_iou = ious[np.argmax(sc[ii])]
        best_iou = ious.max()
        top_ious.append(top_iou)
        best_ious.append(best_iou)
        ratio = (segs[top, 1] - segs[top, 0]) / max(g[1] - g[0], 1e-6)
        ratios.append(ratio)
        pc[3] += ratio
        if top_iou >= 0.5:
            n_top_ge05 += 1
            pc[2] += 1
        if best_iou >= 0.5:
            n_any_ge05 += 1
        if top_iou < 0.5 and best_iou >= 0.5:
            n_frag_outranks += 1
            pc[1] += 1
    id2name = {v: k for k, v in label_dict.items()}
    ratios = np.asarray(ratios)
    return {
        'n_gt': n, 'n_gt_with_overlapping_det': n_overlap,
        'frag_outranks_full_frac_of_gt': n_frag_outranks / max(n, 1),
        'frag_outranks_full_frac_of_overlapped': n_frag_outranks / max(n_overlap, 1),
        'n_frag_outranks_full': n_frag_outranks,
        'recall_at1_tiou05_top_overlapping': n_top_ge05 / max(n, 1),
        'recall_at1_tiou05_top_of_video_class': n_top1_vc / max(n, 1),
        'recall_any_det_tiou05': n_any_ge05 / max(n, 1),
        'top_det_iou': stats(top_ious), 'best_det_iou': stats(best_ious),
        'duration_ratio_top_over_gt': stats(ratios),
        'duration_ratio_frac_lt_0.5': float((ratios < 0.5).mean()) if len(ratios) else float('nan'),
        'duration_ratio_frac_gt_1.5': float((ratios > 1.5).mean()) if len(ratios) else float('nan'),
        'per_class': {id2name[c]: {'n': v[0], 'frag_outranks': v[1],
                                   'recall_at1': v[2] / max(v[0], 1),
                                   'dur_ratio_mean': v[3] / max(v[0], 1)}
                      for c, v in per_class.items()},
    }


# --------------------------------------------------------------- offline
def weight_norms(n):
    ck = torch.load(CKPT['vsa3'].format(n=n), map_location='cpu')
    sd = {k[len('module.'):]: v for k, v in ck['state_dict'].items() if k.startswith('module.')}
    out = {'epoch': int(ck.get('epoch', -1))}
    for a in ('clip_attn', 'chain_attn'):
        w = sd[f'vsa.{a}.out_proj.weight']
        b = sd[f'vsa.{a}.out_proj.bias']
        ip = sd[f'vsa.{a}.in_proj_weight']
        out[a] = {'out_proj_weight_fro': float(w.norm()), 'out_proj_bias_l2': float(b.norm()),
                  'out_proj_weight_spectral': float(torch.linalg.matrix_norm(w, ord=2)),
                  'in_proj_weight_fro': float(ip.norm()),
                  'in_proj_q_fro': float(ip[:256].norm()), 'in_proj_k_fro': float(ip[256:512].norm()),
                  'in_proj_v_fro': float(ip[512:].norm())}
    # text-side chain_attn (stage 2) for scale reference
    w = sd['text_features.chain_attn.out_proj.weight']
    out['text_chain_attn_out_proj_weight_fro'] = float(w.norm())
    return out


def parse_log(path):
    if not os.path.exists(path):
        return None
    ep_inst = defaultdict(list)
    ep_run = {}
    ep_loss_inst = defaultdict(list)
    ep_loss_run = {}
    cur = None
    mAP_list, best = None, None
    with open(path, errors='replace') as f:
        for line in f:
            m = re.match(r'Epoch: \[(\d+)\]\[(\d+)/(\d+)\]\s+Time.*Loss ([\d.]+) \(([\d.]+)\)', line)
            if m:
                cur = int(m.group(1))
                ep_loss_inst[cur].append(float(m.group(4)))
                ep_loss_run[cur] = float(m.group(5))
                continue
            m = re.search(r'cls_loss ([\d.]+) \(([\d.]+)\)', line)
            if m and cur is not None:
                ep_inst[cur].append(float(m.group(1)))
                ep_run[cur] = float(m.group(2))
                continue
            if line.startswith('mAP_list='):
                mAP_list = json.loads(line.split('=', 1)[1].strip())
            m = re.match(r'max\(mAP_list\) ([\d.]+)', line)
            if m:
                best = float(m.group(1))
    eps = sorted(ep_inst)
    if not eps:
        return None
    first, last = eps[0], eps[-1]
    early = [x for e in eps[1:6] for x in ep_inst[e]]
    late = [x for e in eps[-5:] for x in ep_inst[e]]
    early_t = [x for e in eps[1:6] for x in ep_loss_inst[e]]
    late_t = [x for e in eps[-5:] for x in ep_loss_inst[e]]
    return {'epochs': len(eps),
            'cls_loss_early_ep1to5': float(np.mean(early)),
            'cls_loss_late_last5': float(np.mean(late)),
            'total_loss_early_ep1to5': float(np.mean(early_t)),
            'total_loss_late_last5': float(np.mean(late_t)),
            'cls_loss_epoch0': float(np.mean(ep_inst[first])),
            'cls_loss_epoch_mean_first': float(np.mean(ep_inst[first])),
            'cls_loss_epoch_mean_last': float(np.mean(ep_inst[last])),
            'cls_loss_running_first_epoch_end': ep_run[first],
            'cls_loss_running_last_epoch_end': ep_run[last],
            'cls_loss_per_epoch_mean': [float(np.mean(ep_inst[e])) for e in eps],
            'total_loss_epoch_mean_first': float(np.mean(ep_loss_inst[first])),
            'total_loss_epoch_mean_last': float(np.mean(ep_loss_inst[last])),
            'mAP_list': mAP_list, 'mAP_best': best,
            'mAP_best_index': (int(np.argmax(mAP_list)) if mAP_list else None),
            'mAP_first_eval': (mAP_list[0] if mAP_list else None),
            'mAP_last_eval': (mAP_list[-1] if mAP_list else None)}


def parse_fused_log(path):
    if not os.path.exists(path):
        return None
    v = None
    with open(path, errors='replace') as f:
        for line in f:
            m = re.match(r'Avearge mAP: ([\d.]+)', line)
            if m:
                v = float(m.group(1))
    return v


def class_durations(json_file, split='test'):
    with open(json_file, encoding='utf-8') as f:
        db = json.load(f)['database']
    d = defaultdict(list)
    for v in db.values():
        if v['subset'].lower() != split:
            continue
        for a in v['annotations']:
            d[a['label']].append(float(a['segment'][1]) - float(a['segment'][0]))
    return {k: {'mean_dur': float(np.mean(v)), 'median_dur': float(np.median(v)), 'n': len(v)}
            for k, v in d.items()}


# ------------------------------------------------------------------- main
def run_split(cfg, n, deep, res_all):
    t0 = time.time()
    R = {'deep': deep}
    seen, train_file = load_classes(cfg, 'train', n)
    unseen, test_file = load_classes(cfg, 'test', n)
    R['seen'] = seen
    R['unseen'] = unseen
    fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    ds_unseen = make_val_dataset(cfg, n)
    assert list(ds_unseen.label_dict.keys()) == unseen
    loader_unseen = make_data_loader(ds_unseen, False, None, 1, cfg['loader']['num_workers'])
    tiou = ds_unseen.get_attributes()['tiou_thresholds']
    ev_unseen = ANETdetection(ds_unseen.json_file, ds_unseen.split[0], tiou_thresholds=tiou,
                              subset_file=test_file)
    gt_unseen_df = ev_unseen.ground_truth.copy()
    # evaluator remapped labels via activity_index (sorted unique ids)
    inv = {v: k for k, v in ev_unseen.activity_index.items()}
    gt_unseen_df['label'] = gt_unseen_df['label'].map(inv)
    with open(cfg['dataset']['json_file'], encoding='utf-8') as f:
        db = json.load(f)['database']
    unseen_set = set(unseen)
    cls_idx = {c: i for i, c in enumerate(unseen)}
    if deep:
        ds_seen = make_val_dataset(cfg, n, labels=seen)
        loader_seen = make_data_loader(ds_seen, False, None, 1, cfg['loader']['num_workers'])
        ev_seen = ANETdetection(ds_seen.json_file, ds_seen.split[0], tiou_thresholds=tiou,
                                subset_file=train_file)
        R['n_test_videos'] = {'unseen_eval': len(ds_unseen), 'seen_vocab_eval': len(ds_seen)}
        R['n_gt'] = {'unseen': int(len(ev_unseen.ground_truth)),
                     'seen': int(len(ev_seen.ground_truth))}
    for mname in ('champ', 'vsa3'):
        model, ep = build_model(cfg, n, CKPT[mname].format(n=n), vsa=(mname == 'vsa3'))
        assert model.text_features.test_classes == unseen
        assert model.text_features.train_classes == seen
        M = {'ckpt_epoch': ep}
        col = VSACollector(model) if mname == 'vsa3' else None
        acc = defaultdict(list)

        def stats_fn(item, _col=col, _acc=acc):
            if _col is None or not deep:
                return
            vid = item['video_id']
            fps, fstr, nfr = item['fps'], item['feat_stride'], item['feat_num_frames']
            gg, ll, uu, ci = [], [], [], []
            for a in db[vid].get('annotations', []):
                s, e = float(a['segment'][0]), float(a['segment'][1])
                gg.append([(s * fps - 0.5 * nfr) / fstr, (e * fps - 0.5 * nfr) / fstr])
                ll.append(a['label'])
                uu.append(a['label'] in unseen_set)
                ci.append(cls_idx.get(a['label'], -1))
            points = model.point_generator(_col.last['feats'])
            video_vsa_stats(model, _col, item, np.asarray(gg, np.float32).reshape(-1, 2), ll,
                            np.asarray(uu, bool), np.asarray(ci, np.int64), fps, fstr, points, _acc)

        # ---- unseen classes, test vocabulary (standard eval)
        res = run_pass(model, loader_unseen, unseen, col, stats_fn)
        for fused in (False, True):
            mAP, tl, pc, post = evaluate(res, ev_unseen, cfg, fused)
            tag = 'fused' if fused else 'unfused'
            M[f'unseen_{tag}_mAP'] = mAP
            M[f'unseen_{tag}_tiou'] = tl
            M[f'unseen_{tag}_per_class'] = pc
            if deep:
                M[f'frag_{tag}'] = fragment_analysis(post, gt_unseen_df, ev_unseen.label_dict)
        if col is not None and deep:
            M['vsa_stats'] = summarize_vsa(acc)
        # ---- seen classes, seen vocabulary (control)
        if deep:
            os.environ['TIFAD_PROTO_COLLECT'] = '1'
            res_s = run_pass(model, loader_seen, seen, None, None)
            os.environ.pop('TIFAD_PROTO_COLLECT', None)
            for fused in (False, True):
                mAP, tl, pc, _ = evaluate(res_s, ev_seen, cfg, fused)
                tag = 'fused' if fused else 'unfused'
                M[f'seen_{tag}_mAP'] = mAP
                M[f'seen_{tag}_tiou'] = tl
                M[f'seen_{tag}_per_class'] = pc
        R[mname] = M
        print(f"  [split {n}] {mname}: unseen unfused {M['unseen_unfused_mAP']:.2f} "
              f"fused {M['unseen_fused_mAP']:.2f}"
              + (f" | seen unfused {M['seen_unfused_mAP']:.2f} fused {M['seen_fused_mAP']:.2f}"
                 if deep else '') + f" (ckpt epoch {ep})", flush=True)
        del model, col
        torch.cuda.empty_cache()
    R['weights'] = weight_norms(n)
    R['logs'] = {k: parse_log(v.format(n=n)) for k, v in LOGS.items()}
    R['fused_logs'] = {k: parse_fused_log(v.format(n=n)) for k, v in FUSED_LOGS.items()}
    R['time_sec'] = time.time() - t0
    res_all['splits'][str(n)] = R
    return R


def aggregate(res, cfg):
    S = res['splits']
    keys = sorted(S, key=int)
    deep = [k for k in keys if S[k].get('deep')]
    A = {'splits': keys, 'deep_splits': deep}
    # Q3 means
    for tag in ('unseen_unfused', 'unseen_fused'):
        for m in ('champ', 'vsa3'):
            A[f'{tag}_mAP_{m}'] = float(np.mean([S[k][m][f'{tag}_mAP'] for k in keys]))
        A[f'{tag}_mAP_delta'] = A[f'{tag}_mAP_vsa3'] - A[f'{tag}_mAP_champ']
        A[f'{tag}_mAP_delta_per_split'] = [round(S[k]['vsa3'][f'{tag}_mAP'] - S[k]['champ'][f'{tag}_mAP'], 2)
                                           for k in keys]
    if deep:
        for tag in ('seen_unfused', 'seen_fused'):
            for m in ('champ', 'vsa3'):
                A[f'{tag}_mAP_{m}'] = float(np.mean([S[k][m][f'{tag}_mAP'] for k in deep]))
            A[f'{tag}_mAP_delta'] = A[f'{tag}_mAP_vsa3'] - A[f'{tag}_mAP_champ']
            A[f'{tag}_mAP_delta_per_split'] = [round(S[k]['vsa3'][f'{tag}_mAP'] - S[k]['champ'][f'{tag}_mAP'], 2)
                                               for k in deep]
        for tag in ('unseen_unfused', 'unseen_fused'):
            A[f'{tag}_mAP_delta_deep_only'] = float(np.mean(
                [S[k]['vsa3'][f'{tag}_mAP'] - S[k]['champ'][f'{tag}_mAP'] for k in deep]))
        # Q1 / Q2 means over deep splits
        def mean_over(path):
            vals = []
            for k in deep:
                d = S[k]['vsa3']['vsa_stats']
                for p in path:
                    d = d[p]
                vals.append(d)
            if isinstance(vals[0], list):
                return np.mean(np.asarray(vals), 0).round(4).tolist()
            return float(np.mean(vals))
        q1 = {}
        for c in ('unseen_gt', 'seen_gt', 'background', 'all'):
            q1[c] = {'rel_median': mean_over(['q1', c, 'rel_residual', 'median']),
                     'rel_p90': mean_over(['q1', c, 'rel_residual', 'p90']),
                     'rel_mean': mean_over(['q1', c, 'rel_residual', 'mean']),
                     'cos_median': mean_over(['q1', c, 'cosine', 'median']),
                     'cos_p10': mean_over(['q1', c, 'cosine', 'p10'])}
        q1['logit'] = {k: mean_over(['q1', 'logit_change', k, 'mean'])
                       for k in S[deep[0]]['vsa3']['vsa_stats']['q1']['logit_change']}
        q1['linearity'] = {k: mean_over(['q1', 'linearity', k])
                           for k in S[deep[0]]['vsa3']['vsa_stats']['q1']['linearity']}
        A['q1'] = q1
        q2 = {}
        for c in ('all', 'unseen_gt', 'background'):
            q2[c] = {'clip_self_weight_mean': mean_over(['q2', c, 'clip_self_weight', 'mean']),
                     'clip_per_chunk_mean': mean_over(['q2', c, 'clip_per_chunk_mean']),
                     'clip_per_head_self_weight_mean': mean_over(['q2', c, 'clip_per_head_self_weight_mean']),
                     'clip_entropy_mean': mean_over(['q2', c, 'clip_entropy', 'mean']),
                     'clip_max_weight_mean': mean_over(['q2', c, 'clip_max_weight', 'mean']),
                     'chain_self_weight_mean': mean_over(['q2', c, 'chain_self_weight', 'mean'])}
        for name, _, _ in LEN_BINS:
            q2[f'seglen_{name}'] = {
                'n': int(sum(S[k]['vsa3']['vsa_stats']['q2']['by_seglen'][name]['n'] for k in deep)),
                'clip_self_weight_mean': mean_over(['q2', 'by_seglen', name, 'clip_self_weight', 'mean']),
                'clip_per_chunk_mean': mean_over(['q2', 'by_seglen', name, 'clip_per_chunk_mean']),
                'clip_entropy_mean': mean_over(['q2', 'by_seglen', name, 'clip_entropy', 'mean']),
                'chain_self_weight_mean': mean_over(['q2', 'by_seglen', name, 'chain_self_weight', 'mean'])}
        q2['clip_entropy_uniform'] = S[deep[0]]['vsa3']['vsa_stats']['q2']['all']['clip_entropy_uniform']
        A['q2'] = q2
        # Q4 means
        q4 = {}
        for tag in ('unfused', 'fused'):
            for m in ('champ', 'vsa3'):
                q4[f'{tag}_{m}'] = {
                    kk: float(np.mean([S[k][m][f'frag_{tag}'][kk] for k in deep]))
                    for kk in ('frag_outranks_full_frac_of_gt', 'frag_outranks_full_frac_of_overlapped',
                               'recall_at1_tiou05_top_overlapping', 'recall_at1_tiou05_top_of_video_class',
                               'recall_any_det_tiou05')}
                q4[f'{tag}_{m}']['duration_ratio_top_over_gt_mean'] = float(np.mean(
                    [S[k][m][f'frag_{tag}']['duration_ratio_top_over_gt']['mean'] for k in deep]))
                q4[f'{tag}_{m}']['duration_ratio_top_over_gt_median'] = float(np.mean(
                    [S[k][m][f'frag_{tag}']['duration_ratio_top_over_gt']['median'] for k in deep]))
                q4[f'{tag}_{m}']['top_det_iou_mean'] = float(np.mean(
                    [S[k][m][f'frag_{tag}']['top_det_iou']['mean'] for k in deep]))
                q4[f'{tag}_{m}']['n_gt'] = int(sum(S[k][m][f'frag_{tag}']['n_gt'] for k in deep))
        A['q4'] = q4
    # Q5 per-class deltas aggregated by class name
    durs = class_durations(cfg['dataset']['json_file'])
    per_cls = defaultdict(list)
    per_cls_unf = defaultdict(list)
    for k in keys:
        for c, ap in S[k]['vsa3']['unseen_fused_per_class'].items():
            per_cls[c].append((int(k), ap - S[k]['champ']['unseen_fused_per_class'][c], ap,
                               S[k]['champ']['unseen_fused_per_class'][c]))
        for c, ap in S[k]['vsa3']['unseen_unfused_per_class'].items():
            per_cls_unf[c].append(ap - S[k]['champ']['unseen_unfused_per_class'][c])
    table = []
    for c, rows in per_cls.items():
        table.append({'class': c, 'n_splits_unseen': len(rows),
                      'splits': [r[0] for r in rows],
                      'delta_fused_mean': float(np.mean([r[1] for r in rows])),
                      'delta_fused_per_split': [round(r[1], 2) for r in rows],
                      'ap_fused_vsa3_mean': float(np.mean([r[2] for r in rows])),
                      'ap_fused_champ_mean': float(np.mean([r[3] for r in rows])),
                      'delta_unfused_mean': float(np.mean(per_cls_unf[c])),
                      'mean_gt_dur_test': durs[c]['mean_dur'], 'n_gt_test': durs[c]['n'],
                      'phases_per_sec': 10.0 / durs[c]['mean_dur']})
    table.sort(key=lambda r: -r['delta_fused_mean'])
    d = np.array([r['delta_fused_mean'] for r in table])
    du = np.array([r['delta_unfused_mean'] for r in table])
    dur = np.array([r['mean_gt_dur_test'] for r in table])
    pps = np.array([r['phases_per_sec'] for r in table])
    A['q5'] = {'table': table,
               'n_classes': len(table),
               'delta_fused_abs_mean': float(np.abs(d).mean()),
               'delta_fused_std': float(d.std()),
               'corr_delta_fused_vs_dur': {'spearman': float(spearmanr(d, dur).correlation),
                                           'pearson': float(pearsonr(d, dur)[0])},
               'corr_delta_fused_vs_phases_per_sec': {'spearman': float(spearmanr(d, pps).correlation),
                                                      'pearson': float(pearsonr(d, pps)[0])},
               'corr_delta_unfused_vs_dur': {'spearman': float(spearmanr(du, dur).correlation),
                                             'pearson': float(pearsonr(du, dur)[0])},
               'corr_delta_unfused_vs_phases_per_sec': {'spearman': float(spearmanr(du, pps).correlation),
                                                        'pearson': float(pearsonr(du, pps)[0])},
               'note': 'every class has exactly 10 phase sentences, so phases_per_sec = 10 / '
                       'mean_gt_dur is a monotone transform of duration (Spearman is exactly '
                       'the negative)'}
    # split-level: per-split abs-delta vs mean delta (redistribution check)
    A['q5_split_level'] = {
        'mean_abs_per_class_delta_fused': float(np.mean(
            [abs(S[k]['vsa3']['unseen_fused_per_class'][c] - S[k]['champ']['unseen_fused_per_class'][c])
             for k in keys for c in S[k]['vsa3']['unseen_fused_per_class']])),
        'per_split_max_abs_class_delta_fused': [round(max(
            abs(S[k]['vsa3']['unseen_fused_per_class'][c] - S[k]['champ']['unseen_fused_per_class'][c])
            for c in S[k]['vsa3']['unseen_fused_per_class']), 2) for k in keys]}
    # Q6 logs
    q6 = {}
    for stage in ('vsa3', 'seqchain'):
        L = [S[k]['logs'][stage] for k in keys if S[k]['logs'].get(stage)]
        if not L:
            continue
        q6[stage] = {
            'n_logs': len(L),
            'cls_loss_early_ep1to5': float(np.mean([l['cls_loss_early_ep1to5'] for l in L])),
            'cls_loss_late_last5': float(np.mean([l['cls_loss_late_last5'] for l in L])),
            'cls_loss_late_minus_early_per_split': [round(l['cls_loss_late_last5'] - l['cls_loss_early_ep1to5'], 4) for l in L],
            'total_loss_early_ep1to5': float(np.mean([l['total_loss_early_ep1to5'] for l in L])),
            'total_loss_late_last5': float(np.mean([l['total_loss_late_last5'] for l in L])),
            'cls_loss_first_epoch_mean': float(np.mean([l['cls_loss_epoch_mean_first'] for l in L])),
            'cls_loss_last_epoch_mean': float(np.mean([l['cls_loss_epoch_mean_last'] for l in L])),
            'cls_loss_running_first_epoch_end': float(np.mean([l['cls_loss_running_first_epoch_end'] for l in L])),
            'cls_loss_running_last_epoch_end': float(np.mean([l['cls_loss_running_last_epoch_end'] for l in L])),
            'total_loss_first_epoch_mean': float(np.mean([l['total_loss_epoch_mean_first'] for l in L])),
            'total_loss_last_epoch_mean': float(np.mean([l['total_loss_epoch_mean_last'] for l in L])),
            'cls_loss_per_epoch_mean_avg_splits': np.mean(
                [l['cls_loss_per_epoch_mean'] for l in L], 0).round(4).tolist(),
            'mAP_first_eval_mean': float(np.mean([l['mAP_first_eval'] for l in L])),
            'mAP_last_eval_mean': float(np.mean([l['mAP_last_eval'] for l in L])),
            'mAP_best_mean': float(np.mean([l['mAP_best'] for l in L])),
            'mAP_best_index_per_split': [l['mAP_best_index'] for l in L],
            'mAP_curve_range_mean': float(np.mean([max(l['mAP_list']) - min(l['mAP_list']) for l in L])),
        }
    A['q6'] = q6
    # weights
    W = [S[k]['weights'] for k in keys]
    A['weights'] = {
        'ckpt_epoch_per_split': [w['epoch'] for w in W],
        'clip_out_proj_weight_fro_mean': float(np.mean([w['clip_attn']['out_proj_weight_fro'] for w in W])),
        'clip_out_proj_weight_fro_per_split': [round(w['clip_attn']['out_proj_weight_fro'], 4) for w in W],
        'clip_out_proj_bias_l2_mean': float(np.mean([w['clip_attn']['out_proj_bias_l2'] for w in W])),
        'clip_out_proj_spectral_mean': float(np.mean([w['clip_attn']['out_proj_weight_spectral'] for w in W])),
        'chain_out_proj_weight_fro_mean': float(np.mean([w['chain_attn']['out_proj_weight_fro'] for w in W])),
        'chain_out_proj_weight_fro_per_split': [round(w['chain_attn']['out_proj_weight_fro'], 4) for w in W],
        'chain_out_proj_bias_l2_mean': float(np.mean([w['chain_attn']['out_proj_bias_l2'] for w in W])),
        'clip_in_proj_weight_fro_mean': float(np.mean([w['clip_attn']['in_proj_weight_fro'] for w in W])),
        'chain_in_proj_weight_fro_mean': float(np.mean([w['chain_attn']['in_proj_weight_fro'] for w in W])),
        'text_chain_attn_out_proj_weight_fro_mean': float(np.mean([w['text_chain_attn_out_proj_weight_fro'] for w in W])),
        'reference_default_in_proj_fro': 'nn.MultiheadAttention xavier_uniform in_proj (768x256): sqrt(768*256*(6/1024)/3) = 19.6 at init',
    }
    # logged fused numbers for cross-check
    A['logged_fused_mAP'] = {m: float(np.mean([S[k]['fused_logs'][m] for k in keys
                                                if S[k]['fused_logs'].get(m) is not None]))
                             for m in ('champ', 'vsa3')}
    return A


def report(res):
    A = res['aggregate']
    L = []
    L.append(f"splits {A['splits']} (deep {A['deep_splits']})")
    L.append("Q3  unseen unfused mAP  champ %.2f  vsa3 %.2f  delta %+.2f  per-split %s" % (
        A['unseen_unfused_mAP_champ'], A['unseen_unfused_mAP_vsa3'], A['unseen_unfused_mAP_delta'],
        A['unseen_unfused_mAP_delta_per_split']))
    L.append("    unseen fused   mAP  champ %.2f  vsa3 %.2f  delta %+.2f  per-split %s" % (
        A['unseen_fused_mAP_champ'], A['unseen_fused_mAP_vsa3'], A['unseen_fused_mAP_delta'],
        A['unseen_fused_mAP_delta_per_split']))
    if A['deep_splits']:
        L.append("    seen   unfused mAP  champ %.2f  vsa3 %.2f  delta %+.2f  per-split %s" % (
            A['seen_unfused_mAP_champ'], A['seen_unfused_mAP_vsa3'], A['seen_unfused_mAP_delta'],
            A['seen_unfused_mAP_delta_per_split']))
        L.append("    seen   fused   mAP  champ %.2f  vsa3 %.2f  delta %+.2f  per-split %s" % (
            A['seen_fused_mAP_champ'], A['seen_fused_mAP_vsa3'], A['seen_fused_mAP_delta'],
            A['seen_fused_mAP_delta_per_split']))
        q1 = A['q1']
        for c in ('unseen_gt', 'seen_gt', 'background', 'all'):
            L.append("Q1  %-10s rel-residual median %.4f p90 %.4f mean %.4f | cos median %.5f p10 %.5f" % (
                c, q1[c]['rel_median'], q1[c]['rel_p90'], q1[c]['rel_mean'], q1[c]['cos_median'], q1[c]['cos_p10']))
        L.append("    logit: " + ', '.join(f"{k} {v:+.4f}" for k, v in q1['logit'].items()))
        L.append("    linearity of residual (held-out R2): " + str({k: (round(v, 3) if isinstance(v, float) else v)
                                                                    for k, v in q1['linearity'].items()}))
        q2 = A['q2']
        for c in ('all', 'unseen_gt', 'background'):
            L.append("Q2  %-10s clip self %.3f | chunks %s | H %.3f (unif %.3f) max-w %.3f | chain self %.3f | heads self %s" % (
                c, q2[c]['clip_self_weight_mean'], [round(x, 3) for x in q2[c]['clip_per_chunk_mean']],
                q2[c]['clip_entropy_mean'], q2['clip_entropy_uniform'], q2[c]['clip_max_weight_mean'], q2[c]['chain_self_weight_mean'],
                [round(x, 3) for x in q2[c]['clip_per_head_self_weight_mean']]))
        for name, _, _ in LEN_BINS:
            b = q2[f'seglen_{name}']
            L.append("    seglen %-6s n=%d clip self %.3f | chunks %s | H %.3f | chain self %.3f" % (
                name, b['n'], b['clip_self_weight_mean'], [round(x, 3) for x in b['clip_per_chunk_mean']],
                b['clip_entropy_mean'], b['chain_self_weight_mean']))
        for tag in ('unfused', 'fused'):
            for m in ('champ', 'vsa3'):
                b = A['q4'][f'{tag}_{m}']
                L.append("Q4  %-7s %-5s n_gt %d frag-outranks-full %.3f (of overlapped %.3f) | R@1@0.5 top-overlap %.3f "
                         "top-of-(vid,cls) %.3f any %.3f | dur ratio mean %.3f median %.3f | top IoU %.3f" % (
                             tag, m, b['n_gt'], b['frag_outranks_full_frac_of_gt'],
                             b['frag_outranks_full_frac_of_overlapped'], b['recall_at1_tiou05_top_overlapping'],
                             b['recall_at1_tiou05_top_of_video_class'], b['recall_any_det_tiou05'],
                             b['duration_ratio_top_over_gt_mean'], b['duration_ratio_top_over_gt_median'],
                             b['top_det_iou_mean']))
    q5 = A['q5']
    L.append("Q5  per-class fused delta (vsa3-champ), %d classes: |delta| mean %.2f std %.2f; "
             "corr(delta, dur) spearman %.3f pearson %.3f; corr(delta, phases/s) spearman %.3f pearson %.3f" % (
                 q5['n_classes'], q5['delta_fused_abs_mean'], q5['delta_fused_std'],
                 q5['corr_delta_fused_vs_dur']['spearman'], q5['corr_delta_fused_vs_dur']['pearson'],
                 q5['corr_delta_fused_vs_phases_per_sec']['spearman'], q5['corr_delta_fused_vs_phases_per_sec']['pearson']))
    for r in q5['table']:
        L.append("    %-18s delta %+6.2f (unf %+6.2f) splits %s  AP %.1f<-%.1f  dur %.1fs  ph/s %.2f" % (
            r['class'], r['delta_fused_mean'], r['delta_unfused_mean'], r['splits'],
            r['ap_fused_vsa3_mean'], r['ap_fused_champ_mean'], r['mean_gt_dur_test'], r['phases_per_sec']))
    L.append("    per-class |delta| mean %.2f; per-split max |class delta| %s" % (
        A['q5_split_level']['mean_abs_per_class_delta_fused'], A['q5_split_level']['per_split_max_abs_class_delta_fused']))
    for st, b in A['q6'].items():
        L.append("Q6  %-8s (%d logs) cls_loss epochs1-5 %.4f -> last5 %.4f (per-split late-early %s); epoch0 %.4f, last %.4f; "
                 "total epochs1-5 %.4f -> last5 %.4f; mAP first-eval %.2f last %.2f best %.2f (best idx %s; curve range %.2f)" % (
                     st, b['n_logs'], b['cls_loss_early_ep1to5'], b['cls_loss_late_last5'],
                     b['cls_loss_late_minus_early_per_split'], b['cls_loss_first_epoch_mean'], b['cls_loss_last_epoch_mean'],
                     b['total_loss_early_ep1to5'], b['total_loss_late_last5'],
                     b['mAP_first_eval_mean'], b['mAP_last_eval_mean'], b['mAP_best_mean'],
                     b['mAP_best_index_per_split'], b['mAP_curve_range_mean']))
    W = A['weights']
    L.append("W   clip out_proj ||W||_F %.4f (per split %s) bias %.4f spectral %.4f | chain out_proj ||W||_F %.4f "
             "(per split %s) bias %.4f | in_proj clip %.2f chain %.2f | text chain out_proj %.4f | ckpt epochs %s" % (
                 W['clip_out_proj_weight_fro_mean'], W['clip_out_proj_weight_fro_per_split'],
                 W['clip_out_proj_bias_l2_mean'], W['clip_out_proj_spectral_mean'],
                 W['chain_out_proj_weight_fro_mean'], W['chain_out_proj_weight_fro_per_split'],
                 W['chain_out_proj_bias_l2_mean'], W['clip_in_proj_weight_fro_mean'],
                 W['chain_in_proj_weight_fro_mean'], W['text_chain_attn_out_proj_weight_fro_mean'],
                 W['ckpt_epoch_per_split']))
    L.append("    logged fused mAP (cross-check): " + str(A['logged_fused_mAP']))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--splits', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--deep-splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out', default='data/vsa_diag.json')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--resume', action='store_true', help='skip splits already in --out')
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = {'meta': {'envs': CH_ENVS, 'fused': FUSED, 'ckpt': CKPT, 'logs': LOGS,
                    'config': args.config, 'len_bins': LEN_BINS}, 'splits': {}}
    if (args.report_only or args.resume) and os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            res = json.load(f)
    if not args.report_only:
        for n in args.splits:
            if args.resume and str(n) in res['splits'] and (
                    res['splits'][str(n)].get('deep') or n not in args.deep_splits):
                print(f"[skip] split {n} present", flush=True)
                continue
            print(f"===== split {n} (deep={n in args.deep_splits}) =====", flush=True)
            R = run_split(cfg, n, n in args.deep_splits, res)
            print(f"  [split {n}] done in {R['time_sec']:.0f}s", flush=True)
            res['aggregate'] = aggregate(res, cfg)
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=1)
    res['aggregate'] = aggregate(res, cfg)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(report(res), flush=True)
    print(f"[save] {args.out}")


if __name__ == '__main__':
    main()
