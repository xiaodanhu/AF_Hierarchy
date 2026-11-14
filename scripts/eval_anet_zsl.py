"""Single-GPU eval of an ActivityNet ZSL checkpoint, with optional oracle
diagnostics (perfect phrase / specialist predictions on GT-covered tokens).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_anet_zsl.py \
      --config configs/activitynet_i3d_zsl_split0_75.yaml \
      --ckpt ckpt/activitynet_i3d_zsl_split0_75_anet_smoke/epoch_020 \
      --oracle none|phrase|all|<specialist>[,<specialist>...]

Prints the standard [ZSL window/video] seen/held-out breakdown; the held-out
video mAP line is the number of interest.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch_slide_dual_eval, ANETdetection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True, help='deepspeed epoch dir with mp_rank_00_model_states.pt')
    ap.add_argument('--oracle', default='none',
                    help="none | all | phrase | comma list of specialist slots")
    ap.add_argument('--alpha', default=None, type=float,
                    help="override ALL held_out_*_alpha composition weights")
    ap.add_argument('--max-seg-num', default=None, type=int,
                    help="override test_cfg max_seg_num (per-video top-k cap)")
    ap.add_argument('--extend-levels', action='store_true',
                    help="propagate held-out composed scores to non-cascade FPN levels")
    ap.add_argument('--tag', default='oracle_eval')
    args = ap.parse_args()

    cfg = load_config(args.config)
    ad = json.load(open(cfg['model']['aux_attribute_table_path']))
    cfg['dataset']['held_out_class_ids'] = [int(a[1:]) for a in ad['held_out_classes']]
    cfg['model']['train_cfg']['head_empty_cls'] = []
    cfg['model']['active_learning_method'] = cfg.get('active_learning_method', None)
    oracle = [] if args.oracle == 'none' else [s.strip() for s in args.oracle.split(',')]
    cfg['model']['train_cfg']['oracle_specialists'] = oracle
    print(f"[oracle-eval] oracle_specialists = {oracle or 'NONE (baseline)'}")
    if args.alpha is not None:
        for k in list(cfg['model']['train_cfg']):
            if k.startswith('held_out_') and k.endswith('_alpha'):
                cfg['model']['train_cfg'][k] = args.alpha
        print(f"[oracle-eval] all held_out_*_alpha -> {args.alpha}")
    if args.max_seg_num is not None:
        cfg['model']['test_cfg']['max_seg_num'] = args.max_seg_num
        print(f"[oracle-eval] test_cfg max_seg_num -> {args.max_seg_num}")
    if args.extend_levels:
        cfg['model']['train_cfg']['extend_held_out_to_all_levels'] = True
        print("[oracle-eval] extend_held_out_to_all_levels -> True")

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'],
        cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, 2)

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    sd_path = os.path.join(args.ckpt, 'mp_rank_00_model_states.pt')
    sd = torch.load(sd_path, map_location='cpu')['module']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    if unexpected:
        print(f"[oracle-eval] ignoring unexpected keys: {unexpected[:5]}")
    model = model.float().cuda().eval()
    print(f"[oracle-eval] loaded {sd_path}")

    val_db_vars = val_dataset.get_attributes()
    with torch.no_grad():
        results = valid_one_epoch_slide_dual_eval(
            val_loader, model, val_dataset, ANETdetection,
            val_db_vars['tiou_thresholds'], 0,
            output_file=args.tag, print_freq=500,
            if_save_data=False, best_map=1e9, local_rank=0,
        )
    if isinstance(results, dict):
        print(f"[oracle-eval] RESULT oracle={args.oracle} "
              f"video_mAP={results.get('video_mAP', float('nan')):.4f}")


if __name__ == '__main__':
    main()
