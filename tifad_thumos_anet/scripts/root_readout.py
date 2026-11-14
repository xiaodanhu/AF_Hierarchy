"""ROOT-LEVEL DETECTION READOUT (eval-only, no training): does the same representation detect at a HIGHER level of the
tree (l_d = 1, the 9 root actions of configs/thumos14_fam_map.json) without
retraining?  THUMOS14 75/25 splits 0-2, frozen final champion.

Two scorings of the 9 roots at every location, same decode / soft-NMS /
evaluator path as the class-level eval:
  lse  : the aggregated ancestor score the model already produces,
         u_a = logsumexp_{c in children(a)} logit_c over the FULL 20-class
         vocabulary (seen + unseen; meta_archs.py [ROOT READOUT] env
         TIFAD_ROOT_READOUT).
  flat : the root NAME as a text class, no aggregation -- the model is
         queried with the 9 root names as its test vocabulary (prompt
         ensemble of the root: 'a video of a sports action: <root>' repeated,
         so the description attention sees the name only).
Ground truth: every test GT segment mapped to its root (label_id = root
index, so THUMOS' CliffDiving-inside-Diving duplicates collapse). Video-level
external classification scores (the fork's UntrimmedNet fusion / top-k
filter) are aggregated to roots by max over children. Unfused eval (no
TIFAD_FUSE_ALPHA / DURPRIOR) -- exactly the class-level 'max(mAP_list)'
protocol at the root level.

Reported per split: root-level mAP@[0.3:0.7] over all 9 roots, over the
roots whose children are all SEEN, over the roots with >= 1 UNSEEN child,
and per-root AP (JSON).

Usage (through the dispatcher queue, one split per job):
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 python scripts/root_readout.py --n 0
  python scripts/root_readout.py --merge      # -> data/tree_consistency/root_readout.json
"""
import argparse
import json
import os
import sys
import time

ROOT = '/data3/xiaodan8/tifad'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CH = {
    'TIFAD_PROMPT_ENSEMBLE': 'configs/thumos14_qwen_ensemble.json',
    'TIFAD_ATTR_ATTN': '1', 'TIFAD_PROMPT_DROPOUT': '0.3', 'TIFAD_SEM_DEGRADE': '0.3',
    'TIFAD_FAMCE': '0.3', 'TIFAD_FAMCE_MAP': 'configs/thumos14_fam_map.json',
    'TIFAD_FAMCE_UW': '1', 'TIFAD_ATTNDESIGN': 'phasechain',
}
SPLIT_DIR = 'splits/train_75_test_25/THUMOS14/{mode}/split_{n}.list'
FAM_MAP = 'configs/thumos14_fam_map.json'
ANNO = '/data3/xiaodan8/thumos/annotations/thumos14.json'
CLS_SCORES = '/data3/xiaodan8/thumos/annotations/thumos14_cls_scores.json'
OUT_DIR = 'data/tree_consistency/root'
CKPT = 'ckpt/thumos14_i3d_seqchain_{n}/mAP_best.pth.tar'


def read_list(p):
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def write_list(p, names):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        for c in names:
            f.write(c + '\n')


def prepare_assets(fam):
    """Root GT json, root ext-score json, root prompt ensemble (once)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    roots = sorted(set(fam.values()))
    rid = {r: i for i, r in enumerate(roots)}
    gt_path = os.path.join(OUT_DIR, 'thumos14_root.json')
    if not os.path.exists(gt_path):
        with open(ANNO, encoding='utf-8') as f:
            db = json.load(f)
        for v in db['database'].values():
            for a in v.get('annotations', []):
                a['label'] = fam[a['label']]
                a['label_id'] = rid[a['label']]
        with open(gt_path, 'w') as f:
            json.dump(db, f)
    sc_path = os.path.join(OUT_DIR, 'thumos14_root_cls_scores.json')
    if not os.path.exists(sc_path):
        with open(CLS_SCORES, encoding='utf-8') as f:
            d = json.load(f)
        cols = [[i for i, c in enumerate(d['class']) if fam[c] == r] for r in roots]
        out = {'class': roots, 'results': {}}
        for vid, sc in d['results'].items():
            out['results'][vid] = [max(sc[i] for i in m) for m in cols]
        with open(sc_path, 'w') as f:
            json.dump(out, f)
    ens_path = os.path.join(OUT_DIR, 'ensemble_root.json')
    if not os.path.exists(ens_path):
        with open(CH['TIFAD_PROMPT_ENSEMBLE'], encoding='utf-8') as f:
            ens = json.load(f)
        K = len(next(iter(ens.values())))
        for r in roots:
            ens[r] = [f'a video of a sports action: {r}'] * K
        with open(ens_path, 'w') as f:
            json.dump(ens, f)
    return roots, gt_path, sc_path, ens_path


def make_cfg(base_cfg_path, n, train_names, test_names, tag):
    """Config copy whose subset_file points at custom class lists."""
    pat = os.path.join(OUT_DIR, f'splits_{tag}', 'train_{r1}_test_{r2}', 'THUMOS14',
                       '{mode}', 'split_{r3}.list')
    write_list(pat.format(r1=75, r2=25, mode='train', r3=n), train_names)
    write_list(pat.format(r1=75, r2=25, mode='test', r3=n), test_names)
    with open(base_cfg_path, encoding='utf-8') as f:
        txt = f.read()
    key = "subset_file: 'splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list'"
    assert key in txt
    txt = txt.replace(key, f"subset_file: '{pat}'")
    p = os.path.join(OUT_DIR, f'cfg_{tag}_{n}.yaml')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(txt)
    return p


def run_variant(variant, n, seen, unseen, roots, gt_path, sc_path, ens_path, base_cfg):
    import torch
    import torch.nn as nn
    from libs.core import load_config
    from libs.datasets import make_data_loader, make_dataset
    from libs.modeling import make_meta_arch
    from libs.utils import ANETdetection, fix_random_seed, valid_one_epoch
    envs = dict(CH)
    if variant == 'lse':
        test_names = seen + unseen
        envs['TIFAD_ROOT_READOUT'] = FAM_MAP
    else:
        test_names = list(roots)
        envs['TIFAD_PROMPT_ENSEMBLE'] = ens_path
        envs.pop('TIFAD_ROOT_READOUT', None)
    for k in ('TIFAD_ROOT_READOUT', 'TIFAD_FUSE_ALPHA', 'TIFAD_DURPRIOR'):
        os.environ.pop(k, None)
    os.environ.update(envs)
    cfg_path = make_cfg(base_cfg, n, seen, test_names, variant)
    cfg = load_config(cfg_path)
    fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    # the VIDEO list must be the same for both variants: every annotated
    # test video (20-class vocabulary); only the feats / timing are used
    vid_cfg = load_config(make_cfg(base_cfg, n, seen, seen + unseen, 'videos'))
    val_dataset = make_dataset(vid_cfg['dataset_name'], n, False, vid_cfg['val_split'],
                               **vid_cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])
    model = make_meta_arch(n, cfg['model_name'], cfg['dataset']['subset_file'],
                           cfg['dataset']['data_split'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    ck = torch.load(CKPT.format(n=n), map_location='cuda:0')
    model.load_state_dict(ck['state_dict'], strict=True)
    del ck
    root_list = os.path.join(OUT_DIR, f'roots_{n}.list')
    write_list(root_list, roots)
    det_eval = ANETdetection(gt_path, val_dataset.split[0],
                             tiou_thresholds=val_dataset.get_attributes()['tiou_thresholds'],
                             subset_file=root_list)
    t0 = time.time()
    mAP, tiou_list = valid_one_epoch(val_loader, model, -1, evaluator=det_eval,
                                     output_file=None, ext_score_file=sc_path,
                                     tb_writer=None, print_freq=50,
                                     prior_prob=cfg['opt']['prior_prob'],
                                     num_pred=cfg['opt']['num_pred'])
    ap = det_eval.ap.mean(0)                                   # (R,) mean over tIoU
    # label ids follow the root list order (= sorted root order), and the
    # evaluator's activity_index is the identity over the ids present
    assert len(ap) == len(roots) and all(
        det_eval.activity_index[i] == i for i in range(len(roots)))
    per_root = {r: float(100 * ap[i]) for i, r in enumerate(roots)}
    children = {r: [c for c in seen + unseen if fam_of(c) == r] for r in roots}
    unseen_set = set(unseen)
    seen_roots = [r for r in roots if not any(c in unseen_set for c in children[r])]
    uns_roots = [r for r in roots if any(c in unseen_set for c in children[r])]
    res = {'avg_mAP_all_roots': float(100 * mAP), 'tiou_mAP_list': tiou_list,
           'mAP_seen_only_roots': float(sum(per_root[r] for r in seen_roots) / len(seen_roots)),
           'mAP_roots_with_unseen_child': float(sum(per_root[r] for r in uns_roots) / len(uns_roots)),
           'seen_only_roots': seen_roots, 'roots_with_unseen_child': uns_roots,
           'per_root_AP': per_root, 'n_test_videos': len(val_dataset),
           'eval_seconds': time.time() - t0}
    del model
    torch.cuda.empty_cache()
    return res


_FAM = None


def fam_of(c):
    return _FAM[c]


def merge(fam):
    roots = sorted(set(fam.values()))
    out = {'meta': {'ckpt': CKPT, 'envs': CH, 'metric': 'root-level avg mAP@[0.3:0.7], unfused',
                    'roots': {r: [c for c in fam if fam[c] == r] for r in roots}},
           'splits': {}}
    for n in (0, 1, 2):
        p = os.path.join(OUT_DIR, f'root_readout_s{n}.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                out['splits'][str(n)] = json.load(f)
    keys = sorted(out['splits'])
    if keys:
        out['mean'] = {}
        for v in ('lse', 'flat'):
            if all(v in out['splits'][k] for k in keys):
                out['mean'][v] = {m: float(sum(out['splits'][k][v][m] for k in keys) / len(keys))
                                  for m in ('avg_mAP_all_roots', 'mAP_seen_only_roots',
                                            'mAP_roots_with_unseen_child')}
    with open('data/tree_consistency/root_readout.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f"root-level detection (l_d=1), unfused avg mAP@[0.3:0.7]; splits {keys}")
    print(f"{'variant':6s} {'split':5s} {'all roots':>10s} {'seen-only roots':>16s} "
          f"{'roots w/ unseen':>16s}")
    for v in ('lse', 'flat'):
        for k in keys:
            r = out['splits'][k].get(v)
            if r:
                print(f"{v:6s} {k:5s} {r['avg_mAP_all_roots']:10.2f} "
                      f"{r['mAP_seen_only_roots']:16.2f} {r['mAP_roots_with_unseen_child']:16.2f}")
        if v in out.get('mean', {}):
            m = out['mean'][v]
            print(f"{v:6s} {'mean':5s} {m['avg_mAP_all_roots']:10.2f} "
                  f"{m['mAP_seen_only_roots']:16.2f} {m['mAP_roots_with_unseen_child']:16.2f}")
    return out


def main():
    global _FAM
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--variants', nargs='+', default=['lse', 'flat'])
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    with open(FAM_MAP, encoding='utf-8') as f:
        fam = json.load(f)
    _FAM = fam
    if args.merge:
        merge(fam)
        return
    roots, gt_path, sc_path, ens_path = prepare_assets(fam)
    n = args.n
    seen = read_list(SPLIT_DIR.format(mode='train', n=n))
    unseen = read_list(SPLIT_DIR.format(mode='test', n=n))
    out_p = os.path.join(OUT_DIR, f'root_readout_s{n}.json')
    res = {}
    if os.path.exists(out_p):
        with open(out_p, encoding='utf-8') as f:
            res = json.load(f)
    for v in args.variants:
        print(f"===== split {n} variant {v} =====", flush=True)
        res[v] = run_variant(v, n, seen, unseen, roots, gt_path, sc_path, ens_path, args.config)
        print(f"[root-readout] split {n} {v}: all {res[v]['avg_mAP_all_roots']:.2f} | "
              f"seen-only roots {res[v]['mAP_seen_only_roots']:.2f} | roots w/ unseen child "
              f"{res[v]['mAP_roots_with_unseen_child']:.2f} | per-root "
              f"{ {k: round(x, 1) for k, x in res[v]['per_root_AP'].items()} }", flush=True)
        with open(out_p, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
