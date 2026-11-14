"""ASPEC ORACLE: per-class composition-alpha ceiling for a trained ASPEC ckpt.

For each candidate global alpha in --alphas (default 0, 0.3, 1.0) it runs the
standard detection eval ONCE with TIFAD_ASPEC_ALPHA=<alpha>, collects the
PER-CLASS AP (averaged over the tIoU grid) from the ANETdetection evaluator,
then reports:
    oracle mAP = mean over classes c of  max over alpha of AP(c, alpha)
vs the best-global-alpha mAP and the alpha=0 (text-only) mAP.

Validity: the ASPEC eval composition in meta_archs.py is per-class ADDITIVE --
    logit[:, c] += alpha * mean_{a in attrs(c)} logsigmoid(attr_logit_a)
so per-class alpha_c only moves class-c scores; a per-class alpha assignment
therefore realizes exactly the per-class AP columns measured under the
matching global alpha. (TIFAD_ASPEC_GATE is force-cleared here because the
entropy gate couples classes; the only residual coupling is the shared
pre-NMS top-k / max_seg_num caps, a second-order effect.)

Usage (envs must match training so the ckpt strict-loads, e.g.):
  cd /data3/xiaodan8/tifad
  TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json \
  TIFAD_ATTR_ATTN=1 \
  TIFAD_ASPEC_ATTRS=configs/thumos14_ttb_attrs.json \
  python scripts/aspec_oracle.py configs/thumos14_i3d.yaml \
      ckpt/thumos14_i3d_aspec_0 --n 0
(TIFAD_ASPEC_ATTRS falls back to --attrs when unset; TIFAD_ASPEC_ALPHA is
managed by the script itself.)
"""

# python imports
import argparse
import glob
import os
import sys
import time

# make repo root importable when run as scripts/aspec_oracle.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch imports
import numpy as np
import torch
import torch.nn as nn

# our code
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.modeling.text import data_split_dir
from libs.utils import ANETdetection, fix_random_seed, valid_one_epoch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # resolve checkpoint (prefer mAP_best.pth.tar inside a dir, as diag does)
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT folder does not exist!"
        best = os.path.join(args.ckpt, 'mAP_best.pth.tar')
        if os.path.isfile(best):
            ckpt_file = best
        else:
            ckpt_file_list = sorted(
                glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            assert len(ckpt_file_list) > 0, "No .pth.tar found in ckpt folder"
            ckpt_file = ckpt_file_list[-1]

    # the attrs json gates the specialist head construction (needed for
    # strict ckpt loading); alpha is swept below; the entropy gate would
    # break per-class additivity, so force it off
    os.environ.setdefault('TIFAD_ASPEC_ATTRS', args.attrs)
    os.environ.pop('TIFAD_ASPEC_GATE', None)
    os.environ.pop('TIFAD_ASPEC_COLLECT', None)

    alphas = [float(a) for a in args.alphas.split(',')]
    assert len(alphas) >= 2, "need at least two candidate alphas"

    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # TEST (unseen-class) dataset / loader, eval style (bs=1, no shuffle)
    val_dataset = make_dataset(
        cfg['dataset_name'], args.n, False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    val_db_vars = val_dataset.get_attributes()
    subset_file = data_split_dir(
        cfg['dataset']['subset_file'], cfg['dataset']['data_split'],
        mode='test', split_num=args.n)

    # model, built exactly as eval.py does
    model = make_meta_arch(
        args.n,
        cfg['model_name'],
        cfg['dataset']['subset_file'],
        cfg['dataset']['data_split'],
        **cfg['model']
    )
    assert model.aspec_head is not None, \
        "aspec head missing -- TIFAD_ASPEC_ATTRS not picked up?"
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    print("=> loading checkpoint '{}'".format(ckpt_file))
    checkpoint = torch.load(ckpt_file, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint

    # subset-file order == label id order == prediction label ids
    with open(subset_file, 'rt') as f:
        subset_names = [l.rstrip('\n') for l in f.readlines() if l.strip()]

    tious = np.asarray(val_db_vars['tiou_thresholds'], dtype=np.float64)

    # one full detection eval per candidate global alpha
    per_alpha_ap = {}       # alpha -> {class_name: avg-tIoU AP}
    per_alpha_map = {}      # alpha -> avg-tIoU mAP (evaluator's average_mAP)
    for alpha in alphas:
        os.environ['TIFAD_ASPEC_ALPHA'] = repr(alpha)
        print(f"\n===== eval with global TIFAD_ASPEC_ALPHA={alpha} =====")
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'],
            subset_file=subset_file,
        )
        start = time.time()
        mAP, tiou_mAP_list = valid_one_epoch(
            val_loader,
            model,
            -1,
            evaluator=det_eval,
            output_file=None,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=args.print_freq,
            prior_prob=cfg['opt']['prior_prob'],
            num_pred=cfg['opt']['num_pred'],
        )
        print(f"[alpha={alpha}] avg mAP {100 * mAP:.2f} "
              f"({time.time() - start:.1f}s)  per-tIoU {tiou_mAP_list}")
        # per-class AP averaged over the tIoU grid; evaluator columns map
        # back to subset-file label ids through activity_index
        ap = det_eval.ap.mean(axis=0)                     # (C_gt,)
        col2name = {}
        for orig_lab, col in det_eval.activity_index.items():
            col2name[col] = subset_names[int(orig_lab)]
        per_alpha_ap[alpha] = {
            col2name[c]: float(ap[c]) for c in range(len(ap))
        }
        per_alpha_map[alpha] = float(mAP)
    os.environ.pop('TIFAD_ASPEC_ALPHA', None)

    # ---- per-class AP x alpha table + oracle assignment ----
    classes = sorted(per_alpha_ap[alphas[0]].keys())
    for a in alphas:
        assert sorted(per_alpha_ap[a].keys()) == classes, \
            "class sets differ across alphas?!"

    hdr = f"{'class':<24s}" + "".join(f"  a={a:<6g}" for a in alphas) \
        + "  best_a"
    print(f"\n== per-class AP (avg tIoU {tious[0]:.1f}:{tious[-1]:.1f}) "
          f"x global alpha ==")
    print(hdr)
    oracle_aps, oracle_pick = [], {}
    for c in classes:
        row = [per_alpha_ap[a][c] for a in alphas]
        best_i = int(np.argmax(row))
        oracle_aps.append(row[best_i])
        oracle_pick[c] = alphas[best_i]
        cells = "".join(f"  {100 * v:6.2f}" + ("*" if i == best_i else " ")
                        for i, v in enumerate(row))
        print(f"{c:<24s}{cells}  {alphas[best_i]:g}")

    oracle_map = float(np.mean(oracle_aps))
    best_global = max(alphas, key=lambda a: per_alpha_map[a])
    print("\n== summary ==")
    for a in alphas:
        tag = "  <- best global" if a == best_global else ""
        print(f"  global alpha={a:<6g} mAP = {100 * per_alpha_map[a]:6.2f}{tag}")
    print(f"  ORACLE per-class alpha mAP = {100 * oracle_map:6.2f}")
    print(f"  gain over alpha=0          = "
          f"{100 * (oracle_map - per_alpha_map[alphas[0]]):+6.2f}")
    print(f"  gain over best global      = "
          f"{100 * (oracle_map - per_alpha_map[best_global]):+6.2f}")
    n_nonzero = sum(1 for c in classes if oracle_pick[c] != 0)
    print(f"  oracle picks alpha!=0 for {n_nonzero}/{len(classes)} classes: "
          f"{ {c: a for c, a in oracle_pick.items() if a != 0} }")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ASPEC oracle: per-class composition-alpha mAP ceiling')
    parser.add_argument('config', type=str, help='path to a config file')
    parser.add_argument('ckpt', type=str,
                        help='ckpt dir (uses mAP_best.pth.tar) or .pth.tar')
    parser.add_argument('--n', default=0, type=int, help='split number')
    parser.add_argument('--attrs', default='configs/thumos14_ttb_attrs.json',
                        type=str, help='class->attrs json (fallback when '
                        'TIFAD_ASPEC_ATTRS is not already set)')
    parser.add_argument('--alphas', default='0,0.3,1.0', type=str,
                        help='comma-separated candidate global alphas '
                        '(first one should be 0 = text-only reference)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency')
    args = parser.parse_args()
    main(args)
