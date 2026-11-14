"""SEEN-VOCAB CONTROL + UNSEEN EVAL of ONE checkpoint on ONE split (generic).

For a checkpoint, measures on the test videos of split n:
  * the standard zero-shot eval: UNSEEN-class segments, unseen vocabulary
    (fused = TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3, and unfused);
  * the "seen-vocab control": SEEN-class test segments scored with the seen
    vocabulary (TIFAD_PROTO_COLLECT=1 forces the text mode to 'train' in
    eval), fused and unfused -- exactly as scripts/vsa_diag.py run_split does
    it (make_val_dataset(cfg, n, labels=seen), ANETdetection with
    subset_file=train_file);
  * the 20-class pooled mean = mean of the 15 seen + 5 unseen per-class APs.

Usage:
  cd /data3/xiaodan8/tifad
  PYTHONUTF8=1 LC_ALL=C.UTF-8 python scripts/seen_eval.py \
      --ckpt ckpt/thumos14_i3d_seqchain_0/mAP_best.pth.tar --split 0 \
      --envs "TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json ..." \
      --out data/seen_eval/champ_s0.json
--envs is the FULL env spec of the arm (applied before the model is built;
the CLI env is inherited too). TIFAD_PROMPT_DROPOUT / TIFAD_SEM_DEGRADE are
forced to 0: both are gated on the text mode == 'train', which the seen pass
forces, and they are inert in the standard unseen eval either way (see the
vsa_diag.py note). TIFAD_FUSE_ALPHA / TIFAD_DURPRIOR / TIFAD_PROTO_COLLECT
are managed by this script and must not be passed.
"""
import argparse
import json
import os
import shlex
import sys
import time

MANAGED = ('TIFAD_FUSE_ALPHA', 'TIFAD_DURPRIOR', 'TIFAD_PROTO_COLLECT')
FORCED_ZERO = ('TIFAD_PROMPT_DROPOUT', 'TIFAD_SEM_DEGRADE')
FUSED = {'TIFAD_FUSE_ALPHA': '0.35', 'TIFAD_DURPRIOR': '0.3'}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--envs', default='', help='"K=V K=V ..." applied before the model build')
    ap.add_argument('--out', required=True, help='output json')
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--tag', default='', help='free-form label stored in the json')
    return ap.parse_args()


def apply_envs(spec):
    applied = {}
    for tok in shlex.split(spec):
        assert '=' in tok, f"bad env token {tok!r}"
        k, v = tok.split('=', 1)
        assert k not in MANAGED, f"{k} is managed by seen_eval.py; do not pass it"
        os.environ[k] = v
        applied[k] = v
    forced = {}
    for k in FORCED_ZERO:
        if os.environ.get(k, '0') not in ('0', '0.0', ''):
            forced[k] = os.environ[k]
        os.environ[k] = '0'
        applied[k] = '0'
    for k in MANAGED:
        os.environ.pop(k, None)
    return applied, forced


ARGS = parse_args()
APPLIED, FORCED = apply_envs(ARGS.envs)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from libs.core import load_config                        # noqa: E402
from libs.datasets import make_data_loader, make_dataset  # noqa: E402
from libs.modeling import make_meta_arch                 # noqa: E402
from libs.modeling.text import data_split_dir            # noqa: E402
from libs.utils import ANETdetection, fix_random_seed    # noqa: E402
from libs.utils.postprocessing import postprocess_results  # noqa: E402


# ---- helpers (copied from scripts/vsa_diag.py so the seen control is identical)
def load_classes(cfg, mode, n):
    p = data_split_dir(cfg['dataset']['subset_file'], cfg['dataset']['data_split'], mode, n)
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()], p


def build_model(cfg, n, ckpt_file):
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


@torch.no_grad()
def run_pass(model, loader, class_names):
    results = {k: [] for k in ('video-id', 't-start', 't-end', 'label', 'score', 'class_names')}
    model.eval()
    for vl in loader:
        out = model(vl)
        for o in out:
            if o['segments'].shape[0] > 0:
                results['video-id'].extend([o['video_id']] * o['segments'].shape[0])
                results['t-start'].append(o['segments'][:, 0])
                results['t-end'].append(o['segments'][:, 1])
                results['label'].append(o['labels'])
                results['score'].append(o['scores'])
                results['class_names'].append(list(class_names))
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
    return float(mAP * 100), [float(x) for x in tiou_list], per_class


def main():
    args = ARGS
    t0 = time.time()
    n = args.split
    cfg = load_config(args.config)
    seen, train_file = load_classes(cfg, 'train', n)
    unseen, test_file = load_classes(cfg, 'test', n)
    if FORCED:
        print(f"[seen_eval] forced to 0 (seen pass must not use them): {FORCED}", flush=True)
    print(f"[seen_eval] ckpt {args.ckpt} split {n} envs {APPLIED}", flush=True)

    fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    ds_unseen = make_val_dataset(cfg, n)
    assert list(ds_unseen.label_dict.keys()) == unseen
    loader_unseen = make_data_loader(ds_unseen, False, None, 1, cfg['loader']['num_workers'])
    tiou = ds_unseen.get_attributes()['tiou_thresholds']
    ev_unseen = ANETdetection(ds_unseen.json_file, ds_unseen.split[0], tiou_thresholds=tiou,
                              subset_file=test_file)
    ds_seen = make_val_dataset(cfg, n, labels=seen)
    loader_seen = make_data_loader(ds_seen, False, None, 1, cfg['loader']['num_workers'])
    ev_seen = ANETdetection(ds_seen.json_file, ds_seen.split[0], tiou_thresholds=tiou,
                            subset_file=train_file)

    model, ep = build_model(cfg, n, args.ckpt)
    assert model.text_features.test_classes == unseen
    assert model.text_features.train_classes == seen

    R = {'meta': {'ckpt': args.ckpt, 'split': n, 'tag': args.tag, 'config': args.config,
                  'envs': APPLIED, 'forced_zero': FORCED, 'fused': FUSED, 'ckpt_epoch': ep,
                  'seen': seen, 'unseen': unseen,
                  'n_test_videos': {'unseen_eval': len(ds_unseen), 'seen_vocab_eval': len(ds_seen)},
                  'n_gt': {'unseen': int(len(ev_unseen.ground_truth)),
                           'seen': int(len(ev_seen.ground_truth))},
                  'tiou_thresholds': [float(x) for x in tiou]}}
    # ---- unseen classes, test vocabulary (standard eval)
    res_u = run_pass(model, loader_unseen, unseen)
    for fused in (False, True):
        tag = 'fused' if fused else 'unfused'
        mAP, tl, pc = evaluate(res_u, ev_unseen, cfg, fused)
        R[f'unseen_{tag}_mAP'] = mAP
        R[f'unseen_{tag}_tiou'] = tl
        R[f'unseen_{tag}_per_class'] = pc
    # ---- seen classes, seen vocabulary (control); PROTO_COLLECT forces mode='train'
    os.environ['TIFAD_PROTO_COLLECT'] = '1'
    res_s = run_pass(model, loader_seen, seen)
    os.environ.pop('TIFAD_PROTO_COLLECT', None)
    for fused in (False, True):
        tag = 'fused' if fused else 'unfused'
        mAP, tl, pc = evaluate(res_s, ev_seen, cfg, fused)
        R[f'seen_{tag}_mAP'] = mAP
        R[f'seen_{tag}_tiou'] = tl
        R[f'seen_{tag}_per_class'] = pc
        pooled = dict(R[f'seen_{tag}_per_class'])
        pooled.update(R[f'unseen_{tag}_per_class'])
        assert len(pooled) == len(seen) + len(unseen)
        R[f'pooled20_{tag}_per_class'] = pooled
        R[f'pooled20_{tag}_mean'] = float(np.mean(list(pooled.values())))
    R['meta']['time_sec'] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(R, f, indent=1)
    print(f"[seen_eval] split {n} {args.tag or os.path.basename(args.out)}: "
          f"unseen unfused {R['unseen_unfused_mAP']:.2f} fused {R['unseen_fused_mAP']:.2f} | "
          f"seen unfused {R['seen_unfused_mAP']:.2f} fused {R['seen_fused_mAP']:.2f} | "
          f"pooled20 unfused {R['pooled20_unfused_mean']:.2f} fused {R['pooled20_fused_mean']:.2f} "
          f"(ckpt epoch {ep}, {R['meta']['time_sec']:.0f}s)", flush=True)
    print(f"[save] {args.out}", flush=True)


if __name__ == '__main__':
    main()
