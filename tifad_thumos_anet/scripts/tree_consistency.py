"""TREE CONSISTENCY of the final detections (eval-only).

Question: when the detector fires an action c at location n, does the
ANCESTOR-level readout at the same location agree with c's ancestor a(c)?
The ancestor readout is the root-action argmax over the aggregated
ancestor scores  A_a(n) = logsumexp_{c' in children(a)} logit(n, c')  computed
from the location's raw class logits over the FULL evaluation vocabulary of
the split (seen + unseen classes; configs/thumos14_fam_map.json maps each of
the 20 classes to one of the 9 root actions).  Nothing about the detector
changes; the metric is a readout of what the representation already encodes
(analogue of the draft's bracketing precision).

Pipeline (per split n, final champion ckpt/thumos14_i3d_seqchain_{n}):
  1. Run the fork's eval.py with the champion envs on a FULL-VOCABULARY split
     (temporary split lists: train = the split's seen list, test = seen +
     unseen, so the test videos, the text conditioning and the class logits
     cover all 20 classes) with TIFAD_DUMP_CAND=<dir> TIFAD_DUMP_CAND_LOGITS=1
     (pre-NMS candidates + every level's per-location logits, per video) and
     TIFAD_DUMP_RAW=<pkl> (post-NMS detections BEFORE the ext-score fusion).
  2. Offline: turn the raw post-NMS detections into the FINAL detections with
     the fork's own postprocess_results (video-level ext-score top-k filter,
     num_pred cap): 'unfused' (no envs) and 'fused' (TIFAD_FUSE_ALPHA=0.35
     TIFAD_DURPRIOR=0.3, the champion's fused eval; the duration prior of a
     SEEN class is its own training log-normal, an UNSEEN class keeps the
     attribute-transferred prior).  Soft-NMS keeps segment
     coordinates, so every final detection is matched back to its pre-NMS
     candidate (same class, same decoded span) and thereby to its anchor
     location (level, index) whose logits row gives the ancestor readout.
  3. Report the consistency rate (ancestor argmax == a(c)) over the top-k
     (k=200) detections per video and over the detections scoring above the
     split-wide median, separately for SEEN- and UNSEEN-class detections, and
     a shuffled-ancestor control (class->ancestor map randomly permuted,
     ancestor scores and a(c) both under the permuted map; mean of 20 perms).

Usage (GPU 2; never GPU 3):
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=2 \
      python scripts/tree_consistency.py --splits 0 1 2
  --skip-eval reuses the dumps under data/tree_consistency/.
"""
import argparse
import glob
import json
import os
import pickle
import subprocess
import sys

import numpy as np

ROOT = '/data3/xiaodan8/tifad'
sys.path.insert(0, ROOT)

CH = ("TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json "
      "TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 "
      "TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json "
      "TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain")
FUSED = {'TIFAD_FUSE_ALPHA': '0.35', 'TIFAD_DURPRIOR': '0.3'}
SPLIT_DIR = 'splits/train_75_test_25/THUMOS14/{mode}/split_{n}.list'
FAM_MAP = 'configs/thumos14_fam_map.json'
ANNO = '/data3/xiaodan8/thumos/annotations/thumos14.json'


def read_list(path):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def write_list(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for c in names:
            f.write(c + '\n')          # metrics.py reads l[:-1]


def make_fullvocab_config(cfg_path, out_dir, n, seen, unseen):
    """Config copy whose subset_file points at full-vocabulary lists."""
    pat = os.path.join(out_dir, 'splits', 'train_{r1}_test_{r2}', 'THUMOS14',
                       '{mode}', 'split_{r3}.list')
    write_list(pat.format(r1=75, r2=25, mode='train', r3=n), seen)
    write_list(pat.format(r1=75, r2=25, mode='test', r3=n), seen + unseen)
    with open(cfg_path, encoding='utf-8') as f:
        txt = f.read()
    assert "subset_file: 'splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list'" in txt
    txt = txt.replace(
        "subset_file: 'splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list'",
        f"subset_file: '{pat}'")
    out_cfg = os.path.join(out_dir, f'thumos14_i3d_fullvocab_{n}.yaml')
    with open(out_cfg, 'w', encoding='utf-8') as f:
        f.write(txt)
    return out_cfg


def run_eval(n, ckpt, cfg_full, dump_dir, raw_pkl, log):
    env = (f"{CH} TIFAD_DUMP_CAND={dump_dir} TIFAD_DUMP_CAND_LOGITS=1 "
           f"TIFAD_DUMP_RAW={raw_pkl}")
    cmd = (f"env PYTHONUTF8=1 LC_ALL=C.UTF-8 {env} python eval.py {cfg_full} "
           f"{ckpt} --n {n} > {log} 2>&1")
    print('[tree] running:', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT)


def final_detections(raw, cfg, fused, seen_names, test_names):
    """The fork's postprocess_results on the raw post-NMS dump -> final
    detections (video, t-start, t-end, label, score)."""
    from libs.utils import postprocessing as pp
    saved = {k: os.environ.get(k) for k in FUSED}
    orig_dp = pp._durprior_params
    try:
        for k in FUSED:
            os.environ.pop(k, None)
        if fused:
            os.environ.update(FUSED)
            unseen_names = [c for c in test_names if c not in set(seen_names)]

            def _dp_full(names):
                # unseen classes: the attribute-transferred prior
                # (computed exactly as in the unseen-only eval); seen
                # classes: own training log-normal.
                mu_u, sig_u = orig_dp(unseen_names)
                with open(ANNO, encoding='utf-8') as f:
                    db = json.load(f)['database']
                logdur = {}
                for v in db.values():
                    if v['subset'] != 'Validation':
                        continue
                    for a in v['annotations']:
                        d = max(float(a['segment'][1]) - float(a['segment'][0]), 1e-3)
                        logdur.setdefault(a['label'], []).append(np.log(d))
                mu = np.zeros(len(names))
                sig = np.zeros(len(names))
                for i, c in enumerate(names):
                    if c in unseen_names:
                        j = unseen_names.index(c)
                        mu[i], sig[i] = mu_u[j], sig_u[j]
                    else:
                        mu[i] = np.mean(logdur[c])
                        sig[i] = max(np.std(logdur[c]), pp._DURPRIOR_SIG_FLOOR)
                return mu, sig
            pp._durprior_params = _dp_full
        res = pp.postprocess_results(
            {k: (v.copy() if isinstance(v, np.ndarray) else list(v))
             for k, v in raw.items()},
            cfg['test_cfg']['ext_score_file'], cfg['opt']['prior_prob'],
            cfg['opt']['num_pred'])
    finally:
        pp._durprior_params = orig_dp
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return res


def match_to_candidates(res, dump_dir):
    """For every final detection find its pre-NMS candidate (same class,
    same decoded span) -> anchor (level, idx) and the logits row there."""
    vids = np.asarray(res['video-id'])
    ts, te = res['t-start'], res['t-end']
    lab, sc = res['label'].astype(int), res['score']
    out_logits = np.full((len(vids), 0), np.nan)
    rows, matched = [], np.zeros(len(vids), bool)
    cache = {}
    for i in range(len(vids)):
        v = vids[i]
        if v not in cache:
            d = np.load(os.path.join(dump_dir, v + '.npz'))
            fps, stride, nfr = float(d['fps']), float(d['feat_stride']), float(d['nframes'])
            vlen = float(d['duration'])
            segs = d['segs'].astype(np.float32)
            segs = (segs * np.float32(stride) + np.float32(0.5 * nfr)) / np.float32(fps)
            segs = np.where(segs <= 0.0, 0.0, segs)
            segs = np.where(segs >= vlen, vlen, segs)
            n_lvl = len([k for k in d.files if k.startswith('logits_l')])
            logits = [d[f'logits_l{l}'] for l in range(n_lvl)]
            cache[v] = dict(segs=segs, label=d['label'].astype(int), score=d['score'],
                            lvl=d['lvl'].astype(int), pti=d['pti'].astype(int),
                            logits=logits)
        c = cache[v]
        m = (c['label'] == lab[i]) & (np.abs(c['segs'][:, 0] - ts[i]) < 1e-3) \
            & (np.abs(c['segs'][:, 1] - te[i]) < 1e-3)
        if not m.any():
            rows.append(None)
            continue
        j = np.flatnonzero(m)
        j = j[np.argmax(c['score'][j])]
        rows.append(c['logits'][c['lvl'][j]][c['pti'][j]])
        matched[i] = True
    C = next(r for r in rows if r is not None).shape[0]
    L = np.full((len(vids), C), np.nan, np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            L[i] = r
    return L, matched


def ancestor_scores(L, cls2anc, n_anc):
    """A_a = logsumexp_{c in a} logit_c  -> (N, n_anc)."""
    A = np.full((L.shape[0], n_anc), -np.inf, np.float64)
    for a in range(n_anc):
        cols = np.flatnonzero(cls2anc == a)
        if len(cols) == 0:
            continue
        x = L[:, cols].astype(np.float64)
        mx = x.max(1, keepdims=True)
        A[:, a] = (mx + np.log(np.exp(x - mx).sum(1, keepdims=True)))[:, 0]
    return A


def rates(consistent, sel, is_seen):
    r = {}
    for name, msk in (('all', sel), ('seen', sel & is_seen), ('unseen', sel & ~is_seen)):
        r[name] = {'rate': float(consistent[msk].mean()) if msk.sum() else float('nan'),
                   'n': int(msk.sum())}
    return r


def evaluate_split(n, dump_dir, raw_pkl, cfg, seen, unseen, fam, k_top, n_perm):
    from libs.core import load_config
    with open(raw_pkl, 'rb') as f:
        raw = pickle.load(f)
    test_names = list(raw['class_names'][0])
    assert test_names == seen + unseen, (test_names, seen + unseen)
    anc_names = sorted(set(fam.values()))
    anc_id = {a: i for i, a in enumerate(anc_names)}
    cls2anc = np.asarray([anc_id[fam[c]] for c in test_names])
    is_seen_cls = np.asarray([c in set(seen) for c in test_names])
    out = {'n_test_videos': int(len(set(raw['video-id']))), 'ancestors': anc_names,
           'classes': test_names}
    for mode in ('unfused', 'fused'):
        res = final_detections(raw, cfg, mode == 'fused', seen, test_names)
        L, matched = match_to_candidates(res, dump_dir)
        vids = np.asarray(res['video-id'])
        lab = res['label'].astype(int)
        sc = res['score']
        # top-k per video (final detections are already score-sorted per
        # video; use an explicit rank for safety)
        rank = np.zeros(len(vids), int)
        for v in np.unique(vids):
            idx = np.flatnonzero(vids == v)
            order = idx[np.argsort(-sc[idx], kind='stable')]
            rank[order] = np.arange(len(idx))
        sel_topk = (rank < k_top) & matched
        sel_med = (sc > np.median(sc)) & matched
        is_seen = is_seen_cls[lab]
        A = ancestor_scores(L, cls2anc, len(anc_names))
        pred_anc = A.argmax(1)
        cons = pred_anc == cls2anc[lab]
        r = {'n_final': int(len(vids)), 'n_matched': int(matched.sum()),
             'match_rate': float(matched.mean()),
             'topk': rates(cons, sel_topk, is_seen),
             'above_median': rates(cons, sel_med, is_seen),
             'median_score': float(np.median(sc))}
        # per-class consistency (top-k)
        r['per_class_topk'] = {
            c: {'rate': float(cons[sel_topk & (lab == ci)].mean())
                if (sel_topk & (lab == ci)).sum() else float('nan'),
                'n': int((sel_topk & (lab == ci)).sum()),
                'seen': bool(is_seen_cls[ci])}
            for ci, c in enumerate(test_names)}
        # shuffled-ancestor control: permute the class->ancestor assignment
        perm_topk, perm_med = [], []
        for p in range(n_perm):
            rs = np.random.RandomState(1000 * n + p)
            c2a_p = cls2anc[rs.permutation(len(cls2anc))]
            A_p = ancestor_scores(L, c2a_p, len(anc_names))
            cons_p = A_p.argmax(1) == c2a_p[lab]
            perm_topk.append(rates(cons_p, sel_topk, is_seen))
            perm_med.append(rates(cons_p, sel_med, is_seen))
        r['shuffled_topk'] = {s: {'rate': float(np.nanmean([x[s]['rate'] for x in perm_topk])),
                                  'n': perm_topk[0][s]['n']} for s in ('all', 'seen', 'unseen')}
        r['shuffled_above_median'] = {s: {'rate': float(np.nanmean([x[s]['rate'] for x in perm_med])),
                                          'n': perm_med[0][s]['n']} for s in ('all', 'seen', 'unseen')}
        # ---- LEAVE-ONE-OUT variant: the detected class's OWN logit is
        # removed before aggregation, so the ancestor argmax must come from
        # the SIBLINGS / other classes (a stricter test of tree structure:
        # a detection whose class is the top logit is trivially consistent
        # under any map above). Only ancestors with >= 2 children can win
        # without their own child -> restricted to those detections.
        anc_size = np.bincount(cls2anc, minlength=len(anc_names))
        L_loo = L.copy()
        L_loo[np.arange(len(lab)), lab] = -np.inf
        A_loo = ancestor_scores(L_loo, cls2anc, len(anc_names))
        cons_loo = A_loo.argmax(1) == cls2anc[lab]
        multi = anc_size[cls2anc[lab]] >= 2
        r['loo_topk'] = rates(cons_loo, sel_topk & multi, is_seen)
        r['loo_above_median'] = rates(cons_loo, sel_med & multi, is_seen)
        r['loo_excluded_singleton_ancestor_topk'] = int((sel_topk & ~multi).sum())
        perm_loo_t, perm_loo_m = [], []
        for p in range(n_perm):
            rs = np.random.RandomState(1000 * n + p)
            c2a_p = cls2anc[rs.permutation(len(cls2anc))]
            size_p = np.bincount(c2a_p, minlength=len(anc_names))
            multi_p = size_p[c2a_p[lab]] >= 2
            cons_p = ancestor_scores(L_loo, c2a_p, len(anc_names)).argmax(1) == c2a_p[lab]
            perm_loo_t.append(rates(cons_p, sel_topk & multi_p, is_seen))
            perm_loo_m.append(rates(cons_p, sel_med & multi_p, is_seen))
        r['shuffled_loo_topk'] = {s: {'rate': float(np.nanmean([x[s]['rate'] for x in perm_loo_t])),
                                      'n': int(np.mean([x[s]['n'] for x in perm_loo_t]))}
                                  for s in ('all', 'seen', 'unseen')}
        r['shuffled_loo_above_median'] = {
            s: {'rate': float(np.nanmean([x[s]['rate'] for x in perm_loo_m])),
                'n': int(np.mean([x[s]['n'] for x in perm_loo_m]))} for s in ('all', 'seen', 'unseen')}
        out[mode] = r
    return out


def fmt(n, r):
    lines = [f"--- split {n}: {r['n_test_videos']} test videos, "
             f"{len(r['classes'])} classes -> {len(r['ancestors'])} root actions"]
    for mode in ('unfused', 'fused'):
        x = r[mode]
        lines.append(f"  [{mode}] final dets {x['n_final']}, matched to a candidate "
                     f"{x['n_matched']} ({100 * x['match_rate']:.1f}%)")
        for sel, shuf in (('topk', 'shuffled_topk'), ('above_median', 'shuffled_above_median'),
                          ('loo_topk', 'shuffled_loo_topk'),
                          ('loo_above_median', 'shuffled_loo_above_median')):
            s = x[sel]
            sh = x[shuf]
            lines.append(
                f"    {sel:17s} all {100 * s['all']['rate']:5.1f}% (n={s['all']['n']:5d}) | "
                f"seen {100 * s['seen']['rate']:5.1f}% (n={s['seen']['n']:5d}) | "
                f"unseen {100 * s['unseen']['rate']:5.1f}% (n={s['unseen']['n']:5d}) | "
                f"shuffled all/seen/unseen {100 * sh['all']['rate']:.1f}/"
                f"{100 * sh['seen']['rate']:.1f}/{100 * sh['unseen']['rate']:.1f}%")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--ckpt', default='ckpt/thumos14_i3d_seqchain_{n}/mAP_best.pth.tar')
    ap.add_argument('--out-dir', default='data/tree_consistency')
    ap.add_argument('--k', type=int, default=200)
    ap.add_argument('--n-perm', type=int, default=20)
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args()
    os.chdir(ROOT)
    from libs.core import load_config
    cfg = load_config(args.config)
    with open(FAM_MAP, encoding='utf-8') as f:
        fam = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    all_res = {'meta': {'ckpt': args.ckpt, 'envs': CH, 'fused_envs': FUSED, 'k': args.k,
                        'n_perm': args.n_perm,
                        'ancestor_score': 'logsumexp of the raw class logits over the '
                                          'children of each root action (fam_map), full '
                                          '20-class vocabulary at the detection anchor',
                        'consistent': 'argmax_a A_a(n) == a(detected class)'},
               'splits': {}}
    res_path = os.path.join(args.out_dir, 'tree_consistency.json')
    if os.path.exists(res_path):
        with open(res_path, encoding='utf-8') as f:
            all_res['splits'] = json.load(f).get('splits', {})
    for n in args.splits:
        seen = read_list(SPLIT_DIR.format(mode='train', n=n))
        unseen = read_list(SPLIT_DIR.format(mode='test', n=n))
        sdir = os.path.join(args.out_dir, f's{n}')
        os.makedirs(sdir, exist_ok=True)
        dump_dir = os.path.join(sdir, 'cand')
        raw_pkl = os.path.join(sdir, 'raw.pkl')
        if not args.skip_eval or not os.path.exists(raw_pkl):
            cfg_full = make_fullvocab_config(args.config, sdir, n, seen, unseen)
            run_eval(n, args.ckpt.format(n=n), cfg_full, dump_dir, raw_pkl,
                     os.path.join(sdir, 'eval.log'))
        r = evaluate_split(n, dump_dir, raw_pkl, cfg, seen, unseen, fam, args.k, args.n_perm)
        # the 20-class (GZSL-style) mAP of that eval, for the record
        for l in open(os.path.join(sdir, 'eval.log'), encoding='utf-8', errors='ignore'):
            if l.startswith('Avearge mAP') or l.startswith('Average mAP'):
                r['fullvocab_avg_mAP_line'] = l.strip()
        all_res['splits'][str(n)] = r
        print(fmt(n, r), flush=True)
        with open(res_path, 'w') as f:
            json.dump(all_res, f, indent=1)
    keys = sorted(all_res['splits'])
    if len(keys) > 1:
        print(f"===== mean over splits {keys} =====")
        for mode in ('unfused', 'fused'):
            for sel, shuf in (('topk', 'shuffled_topk'), ('above_median', 'shuffled_above_median'),
                              ('loo_topk', 'shuffled_loo_topk'),
                              ('loo_above_median', 'shuffled_loo_above_median')):
                vals = {s: np.mean([all_res['splits'][k][mode][sel][s]['rate'] for k in keys])
                        for s in ('all', 'seen', 'unseen')}
                sh = {s: np.mean([all_res['splits'][k][mode][shuf][s]['rate'] for k in keys])
                      for s in ('all', 'seen', 'unseen')}
                print(f"  [{mode}] {sel:17s} all {100 * vals['all']:.1f}% | seen "
                      f"{100 * vals['seen']:.1f}% | unseen {100 * vals['unseen']:.1f}% | "
                      f"shuffled {100 * sh['all']:.1f}/{100 * sh['seen']:.1f}/"
                      f"{100 * sh['unseen']:.1f}%")
    print(f"[save] {res_path}")


if __name__ == '__main__':
    main()
