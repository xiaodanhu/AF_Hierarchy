# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import DistributedSampler

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, make_data_loader_distributed
from libs.modeling import make_meta_arch
from libs.utils import (valid_one_epoch, valid_one_epoch_distributed,
                        valid_one_epoch_slide_dual_eval,
                        ANETdetection, fix_random_seed,
                        make_optimizer, make_scheduler)
import json
import math
import deepspeed
import warnings
warnings.filterwarnings("ignore")


def get_val_dataset_config(args, cfg):
    """
    Get validation dataset configuration.
    If --val_dataset is specified, use that dataset for evaluation instead of the training config.
    This allows evaluating a model trained on finegym_slide using finegym_original's test set.

    If --eval_train is specified, evaluate on the training set instead of validation set.
    This swaps train/val json files so that is_training=False loads from train annotations,
    while still using test-time behavior (overlapped sliding windows, no skipping).
    """
    if args.val_dataset == 'finegym_original':
        # Use finegym_original test set
        val_dataset_name = 'finegym_original'
        val_dataset_cfg = {
            'json_file': '/home/jupyter/xhu3/video/dataset/finegym/original_data/finegym_merged_win32_int16.json',
            'num_classes': cfg['dataset'].get('num_classes', 99),
            'num_frames': cfg['dataset'].get('num_frames', 16),
            'trunc_thresh': cfg['dataset'].get('trunc_thresh', 0.5),
            'crop_ratio': cfg['dataset'].get('crop_ratio', [0.9, 1.0]),
            'max_seq_len': cfg['dataset'].get('max_seq_len', 144),
            'sample_stride': cfg['dataset'].get('sample_stride', 16),
        }
        return val_dataset_name, val_dataset_cfg
    elif args.val_dataset == 'finegym_slide':
        # Use finegym_slide test set
        val_dataset_name = 'finegym_slide'
        val_dataset_cfg = {
            'train_json_file': cfg['dataset'].get('train_json_file',
                '/home/jupyter/xhu3/video/dataset/finegym/annotation/Dec14/gym99_train_label.txt'),
            'val_json_file': cfg['dataset'].get('val_json_file',
                '/home/jupyter/xhu3/video/dataset/finegym/annotation/Dec14/gym99_val_label.txt'),
            'num_classes': cfg['dataset'].get('num_classes', 99),
            'num_frames': cfg['dataset'].get('num_frames', 16),
            'trunc_thresh': cfg['dataset'].get('trunc_thresh', 0.5),
            'crop_ratio': cfg['dataset'].get('crop_ratio', [0.9, 1.0]),
            'max_seq_len': cfg['dataset'].get('max_seq_len', 144),
            'window_length': cfg['dataset'].get('window_length', 32),
            'window_stride': cfg['dataset'].get('window_stride', 16),
            'sample_stride': cfg['dataset'].get('sample_stride', 16),
            'test_overlap': True,  # Use 50% overlap for test
        }
        return val_dataset_name, val_dataset_cfg
    else:
        # Use config's dataset (default behavior)
        dataset_cfg = cfg['dataset'].copy()

        # If --eval_train is specified, swap train/val json files
        # This makes is_training=False load from train annotations while keeping test-time behavior
        if args.eval_train:
            train_json = dataset_cfg.get('train_json_file')
            val_json = dataset_cfg.get('val_json_file')
            if train_json and val_json:
                # Swap: val_json_file points to train data, train_json_file points to val data
                # When is_training=False, dataset uses val_json_file, which now points to train data
                dataset_cfg['val_json_file'] = train_json
                dataset_cfg['train_json_file'] = val_json

        return cfg['dataset_name'], dataset_cfg


################################################################################
def main(args):

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()

    with open("configs/deepspeed_config.json", 'r') as f:
        ds_config = json.load(f)

    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
    ckpt_file = os.path.join(args.ckpt, args.filename)
    assert os.path.isdir(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    # Get validation dataset config (may differ from training config if --val_dataset is specified)
    val_dataset_name, val_dataset_cfg = get_val_dataset_config(args, cfg)
    eval_split = 'train' if args.eval_train else 'val'
    if args.local_rank == 0:
        print(f"\n=== Evaluating on {eval_split.upper()} set using dataset: {val_dataset_name} ===\n")

    val_dataset = make_dataset(
        val_dataset_name, False, cfg['val_split'], cfg['model']['backbone_type'], cfg['round'], **val_dataset_cfg
    )
    # Distributed validation: split test data across GPUs for faster inference
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = make_data_loader_distributed(val_dataset, val_sampler, False, None, 1, cfg['loader']['num_workers'])

    """3. create model and evaluator"""
    # model
    cfg['model']['active_learning_method'] = cfg['active_learning_method']
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = math.ceil(len(val_loader) / ds_config["gradient_accumulation_steps"])
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config="configs/deepspeed_config.json"
    )

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    _, client_sd = model_engine.load_checkpoint(
        args.ckpt,
        args.filename,
        load_optimizer_states=True,
        load_lr_scheduler_states=True
    )


    """5. Test the model"""
    print("\nStart testing model {:s} on {:s} set ...".format(cfg['model_name'], eval_split.upper()))
    start = time.time()

    val_db_vars = val_dataset.get_attributes()
    # Include eval_split in output filename to distinguish train vs val results
    output_file = f"{args.filename}_{eval_split}"

    # Use dual evaluation (window + video level) for finegym_slide
    if val_dataset_name in ('finegym_slide', 'finediving_slide') and not args.saveonly:
        eval_results = valid_one_epoch_slide_dual_eval(
            val_loader,
            model_engine,
            val_dataset,
            ANETdetection,
            val_db_vars['tiou_thresholds'],
            -1,  # epoch
            output_file=output_file,
            print_freq=args.print_freq,
            if_save_data=True,
            best_map=0.0,
            local_rank=args.local_rank
        )
        # Use video-level mAP (clip-level aggregation)
        mAP = eval_results.get('video_mAP', 0.0) if isinstance(eval_results, dict) else eval_results
    else:
        # Standard evaluation for other datasets
        if not args.saveonly:
            det_eval = ANETdetection(
                ant_file=None,
                split=None,
                tiou_thresholds=val_db_vars['tiou_thresholds'],
                ground_truth_df=val_dataset.get_ground_truth_df(),
                dataset_name=f'{val_dataset_name}_{eval_split}'
            )
        else:
            det_eval = None
            output_file = 'eval_results.pkl'

        mAP = valid_one_epoch_distributed(
            val_loader,
            model_engine,
            -1,
            evaluator=det_eval,
            output_file=output_file,
            print_freq=args.print_freq,
            if_save_data=True,
            best_map=0.0,
            local_rank=args.local_rank
        )

    end = time.time()

    # Only rank 0 prints results
    if args.local_rank == 0:
        print("[{:s} set] mAP: {:0.2f}".format(eval_split.upper(), mAP))
        print("All done! Total time: {:0.2f} sec, {:s} set mAP: {:0.2f}".format(end - start, eval_split.upper(), mAP))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR', default='./configs/finegym_i3d.yaml', help='path to a config file')
    parser.add_argument('--ckpt', type=str, metavar='DIR', default='./ckpt/finegym_i3d_deepspeed', help='path to a checkpoint')
    parser.add_argument('--filename', type=str, default='epoch_001')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true', help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument("--local_rank", default=-1, type=int, help='local_rank for distributed training on gpus')
    parser.add_argument('--val_dataset', type=str, default=None, choices=['finegym_original', 'finegym_slide', 'finediving_slide'])
    parser.add_argument('--eval_train', action='store_true',
                        help='Evaluate on training set instead of validation set. '
                             'Uses test-time behavior (overlapped sliding windows, no skipping).')
    args = parser.parse_args()
    print(args.local_rank)

    main(args)

'''

# Evaluate on validation set (default)
deepspeed --include=localhost:0,1 --master_port=29502 eval_shard.py \
    --config ./configs/finegym_i3d.yaml \
    --ckpt ./ckpt/finegym_i3d_deepspeed \
    --filename epoch_001 \
    --val_dataset finegym_original

deepspeed --include=localhost:0,1,2,3 --master_port=29502 eval_shard.py \
    --config configs/finegym_i3d.yaml \
    --ckpt ./ckpt/finegym_i3d_deepspeed_raw_video \
    --filename epoch_030

# Evaluate on TRAINING set (with test-time behavior: overlapped sliding windows, no skipping)
deepspeed --include=localhost:0,1,2,3 --master_port=29502 eval_shard.py \
    --config configs/finegym_i3d.yaml \
    --ckpt ./ckpt/finegym_i3d_deepspeed_raw_video \
    --filename epoch_055 \
    --eval_train
'''
