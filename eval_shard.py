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

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (valid_one_epoch, ANETdetection, fix_random_seed,
                        make_optimizer, make_scheduler)
import json
import math
import deepspeed
import warnings
warnings.filterwarnings("ignore")


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
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

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


    # set up evaluator
    det_eval, output_file = None, args.filename
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
    else:
        output_file = 'eval_results.pkl'

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model_engine,
        -1,
        evaluator=det_eval,
        output_file = output_file,
        print_freq=args.print_freq,
        if_save_data=True
    )
    end = time.time()
    print("mAP: {:0.2f}".format(mAP))
    print("All done! Total time: {:0.2f} sec, mAP: {:0.2f}".format(end - start, mAP))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR', default='./configs/anet_i3d_v2.yaml', help='path to a config file')
    parser.add_argument('--ckpt', type=str, metavar='DIR', default='./ckpt/anet_i3d_v2_deepspeed', help='path to a checkpoint')
    parser.add_argument('--filename', type=str, default='epoch_002')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true', help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    print(args.local_rank)

    main(args)
