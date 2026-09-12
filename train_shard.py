import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

# Suppress DeepSpeed logging spam from data loader workers
os.environ["DS_ACCELERATOR"] = "cuda"  # Skip auto-detection
# Increase NCCL timeout to prevent training hangs (default 10min -> 30min)
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes in seconds
import logging
logging.getLogger("deepspeed").setLevel(logging.WARNING)

# python imports
import argparse
import time
from datetime import datetime
import pprint

import math

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
import json
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, make_data_loader_distributed
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, valid_one_epoch_distributed,
                        valid_one_epoch_slide_dual_eval,
                        ANETdetection, save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, TrainingLogger)
import warnings
warnings.filterwarnings("ignore")
import gc


################################################################################
def main(args):
    """main function that handles training / inference"""

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg_for_dscfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    ds_config_path = cfg_for_dscfg.get('deepspeed_config_path', 'configs/deepspeed_config.json')
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)
    print(f"[deepspeed] using config: {ds_config_path}")

    cfg = cfg_for_dscfg

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']) and args.local_rank == 0:
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    ts = datetime.fromtimestamp(int(time.time())).strftime("%Y-%m-%d_%H-%M-%S")
    if len(args.output) == 0:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder) and args.local_rank == 0:
        os.mkdir(ckpt_folder)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    # Phase 2D ZSL: load held-out class ids from the attribute table when the model
    # uses the auxiliary attribute head. The training dataset filters them out;
    # validation keeps all classes (we'll report seen vs held-out mAP separately).
    # Load held-out class ids whenever the attribute table is provided (used by
    # any Phase 2D ZSL variant: aux-attribute, CLIP-text, or parent-conditional v4).
    if cfg['model'].get('aux_attribute_table_path'):
        import json as _json
        _attr_path = cfg['model'].get('aux_attribute_table_path', '')
        if _attr_path:
            with open(_attr_path) as _f:
                _ad = _json.load(_f)
            _held = [int(aid[1:]) for aid in _ad.get('held_out_classes', [])]
            cfg['dataset']['held_out_class_ids'] = _held
            print(f"[ZSL] held-out class ids ({len(_held)}): {_held}")

    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_sampler_for_test = DistributedSampler(train_dataset, shuffle=False)
    train_loader = make_data_loader_distributed(train_dataset, train_sampler, True, rng_generator, **cfg['loader'])
    train_loader_for_test = make_data_loader_distributed(train_dataset, train_sampler_for_test, False, None, 1, cfg['loader']['num_workers'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset']
    )
    # Distributed validation: split test data across GPUs for faster inference
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = make_data_loader_distributed(val_dataset, val_sampler, False, None, 1, cfg['loader']['num_workers'] // 2)

    """3. create model, optimizer, and scheduler"""
    # model
    cfg['model']['active_learning_method'] = cfg['active_learning_method']
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    # ---- second-stage training (APA phase-level attention) ----
    # --init_from: a DeepSpeed checkpoint dir (e.g. .../vit_best_model) or a
    # .pt file; its module weights initialise this model (strict=False so
    # newly added modules such as text_pathway.phase_attn stay at their
    # zero-init). --freeze_except: comma-separated substrings; only parameters
    # whose name contains one of them are trained. Together they reproduce
    # the THUMOS/ActivityNet recipe: train the detector + action-level
    # attention first, then learn only the phase-level attention on top.
    if args.init_from:
        src = args.init_from
        if os.path.isdir(src):
            src = os.path.join(src, 'mp_rank_00_model_states.pt')
        sd = torch.load(src, map_location='cpu')
        sd = sd.get('module', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init_from] loaded {src}: {len(sd)} tensors; "
              f"missing={sorted(missing)}; unexpected={sorted(unexpected)}")
    if args.freeze_except:
        keep = [k.strip() for k in args.freeze_except.split(',') if k.strip()]
        n_train = 0
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n for k in keep)
            n_train += int(p.requires_grad)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert n_train > 0, f"--freeze_except {keep} matched no parameters"
        print(f"[freeze_except] {n_train} trainable tensors: {trainable}")

    # not ideal for multi GPU training, ok for now
    # model = nn.DataParallel(model, device_ids=cfg['devices'])
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = math.ceil(len(train_loader) / ds_config["gradient_accumulation_steps"]) #len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)


    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config_path
    )

    # enable model EMA
    print("Using model EMA ...")
    model_ema = None # ModelEma(model_engine, device='cuda')

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.exists(os.path.join(ckpt_folder, args.resume)):
            # load ckpt, reset epoch / best rmse
            print("Resuming from checkpoint: {:s}".format(args.resume))
            _, client_sd = model_engine.load_checkpoint(
                ckpt_folder,
                args.resume,
                load_optimizer_states=True, 
                load_lr_scheduler_states=True
            )
            args.start_epoch = client_sd['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # Initialize the logger
    start_time = datetime.now().strftime("%m-%d_%H-%M")
    logger = TrainingLogger(os.path.join(ckpt_folder, f"training_logs_{start_time}.txt"))

    if args.local_rank == 0:
        # save the current config
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint.pprint(cfg, stream=fid)
            fid.flush()

        logger.log(pprint.pformat(cfg))

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    best_mAP = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model_engine,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq,
            save_log = args.local_rank == 0
        )

        # save ckpt once in a while
        if (
            epoch == args.start_epoch or
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            print("Saving model at epoch: ", epoch + 1)
            # Release PyTorch's cached-but-unused GPU memory so the NCCL barrier
            # inside save_checkpoint (which gathers sharded ZeRO-2 optimizer
            # state across ranks) has room for its working buffers. Without
            # this, training peak (27+ GB on 40 GB GPUs) leaves too little free
            # for the gather → CUDA OOM in NCCL.
            gc.collect()
            torch.cuda.empty_cache()
            client_sd = {}
            client_sd['epoch'] = epoch
            model_engine.save_checkpoint(ckpt_folder, tag='epoch_{:03d}'.format(epoch + 1), client_state=client_sd)
            print("Model saved at epoch: ", epoch + 1)

            # set up evaluator
            # train_db_vars = train_dataset.get_attributes()
            # train_eval = ANETdetection(
            #     train_dataset.json_file,
            #     train_dataset.split[0],
            #     tiou_thresholds = train_db_vars['tiou_thresholds']
            # )

            # mAP = valid_one_epoch(
            #     train_loader_for_test,
            #     model_engine,
            #     epoch,
            #     evaluator=train_eval,
            #     print_freq=args.print_freq,
            #     if_save_data=False
            # )
            # if args.local_rank == 0:
            #     print("Epoch: ", epoch, ", Train mAP: ", mAP)
            #     logger.log(f"[Train] Epoch {epoch}: Trainset mAP = {mAP:.4f}")

            ##############################

            val_db_vars = val_dataset.get_attributes()

            # Use dual evaluation (window + video level) for finegym_slide
            if cfg['dataset_name'] in ('finegym_slide', 'finediving_slide', 'thumos14_slide', 'thumos14_raw', 'activitynet_raw'):
                eval_results = valid_one_epoch_slide_dual_eval(
                    val_loader,
                    model_engine,
                    val_dataset,
                    ANETdetection,
                    val_db_vars['tiou_thresholds'],
                    epoch,
                    output_file=str(ts),
                    print_freq=args.print_freq,
                    if_save_data=True,
                    best_map=best_mAP,
                    local_rank=args.local_rank
                )
                # Use video-level mAP for model selection (clip-level aggregation)
                mAP = eval_results.get('video_mAP', 0.0) if isinstance(eval_results, dict) else eval_results
            else:
                # Standard evaluation for other datasets
                det_eval = ANETdetection(
                    ant_file=None,
                    split=None,
                    tiou_thresholds=val_db_vars['tiou_thresholds'],
                    ground_truth_df=val_dataset.get_ground_truth_df(),
                    dataset_name=cfg['dataset_name'] + '_val'
                )
                mAP = valid_one_epoch_distributed(
                    val_loader,
                    model_engine,
                    epoch,
                    evaluator=det_eval,
                    output_file=str(ts),
                    print_freq=args.print_freq,
                    if_save_data=True,
                    best_map=best_mAP,
                    local_rank=args.local_rank
                )
                del det_eval

            # Only rank 0 has the final mAP, determine if best
            is_best = False
            if args.local_rank == 0:
                print("Epoch: ", epoch, ", Test mAP: ", mAP)
                logger.log(f"[Test] Epoch {epoch}: Testset mAP = {mAP:.4f}")

                if mAP > best_mAP:
                    best_mAP = mAP
                    best_epoch = epoch
                    is_best = True

            # Memory cleanup after testing
            gc.collect()
            torch.cuda.empty_cache()

            # Synchonize before checkpoint save (all ranks must participate)
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

            # Broadcase is_best decision from rank 0 to all ranks
            if dist.is_initialized():
                is_best_tensor = torch.tensor(1 if is_best else 0, device='cuda')
                dist.broadcast(is_best_tensor, src=0)
                is_best = is_best_tensor.item() == 1
            
            # All ranks must call save_checkpoint together (DeepSpeed requirement)
            if is_best:
                client_sd = {'epoch': epoch}
                model_engine.save_checkpoint(ckpt_folder, tag='vit_best_model', client_state=client_sd)
                if args.local_rank == 0:
                    print("Best model saved at epoch: ", epoch + 1)

            # Final synchronization before next epoch
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

    # wrap up
    logger.log(f"Best model saved at epoch {best_epoch}: best testset mAP = {best_mAP:.4f}")
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR', default='./configs/finediving_i3d.yaml', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=5, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int, help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='deepspeed', type=str, help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a DeepSpeed checkpoint tag (default: none)')
    parser.add_argument('--init_from', default='', type=str, metavar='PATH',
                        help='initialise weights from a DeepSpeed checkpoint dir or .pt (strict=False); for second-stage training')
    parser.add_argument('--freeze_except', default='', type=str,
                        help='comma-separated substrings; only parameters whose names contain one are trained')
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    print(args.local_rank)

    main(args)
