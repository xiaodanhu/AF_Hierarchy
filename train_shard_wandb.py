import os
from typing import Any
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
import wandb
import subprocess

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
def main(args, cfg):
    """main function that handles training / inference"""

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()
    
    with open("configs/deepspeed_config.json", 'r') as f:
        ds_config = json.load(f)

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0

    # Load wandb config on rank 0 and broadcase sweep parameters
    if args.local_rank == 0:
        wandb.init(config=cfg,
                   reinit=True
                   )
        run_config = wandb.config
        # Apply sweep parameters (sliding window parameters for finegym_slide dataset)
        cfg["dataset"]["max_seq_len"] = run_config["max_seq_len"]
        cfg["opt"]["learning_rate"] = run_config["learning_rate"]
        cfg["loader"]["batch_size"] = run_config["batch_size"]
        ds_config["gradient_accumulation_steps"] = run_config["gradient_accumulation_steps"]

    # Broadcast sweep parameters from rank 0 to all ranks
    if dist.is_initialized():
        # Create tensors to broadcast
        params_tensor = torch.tensor([
            cfg["dataset"].get("max_seq_len", 144),
            cfg["opt"].get("learning_rate", 0.0001),
            cfg["loader"].get("batch_size", 4),
            ds_config.get("gradient_accumulation_steps", 6)
        ], dtype=torch.float32, device='cuda')

        dist.broadcast(params_tensor, src=0)

        # Apply broadcasted parameters on non-rank-0 processes
        if args.local_rank != 0:
            cfg["dataset"]["max_seq_len"] = int(params_tensor[0].item())
            cfg["opt"]["learning_rate"] = params_tensor[1].item()
            cfg["loader"]["batch_size"] = int(params_tensor[2].item())
            ds_config["gradient_accumulation_steps"] = int(params_tensor[3].item())

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

    try:
        """2. create dataset / dataloader"""
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
        # not ideal for multi GPU training, ok for now
        # model = nn.DataParallel(model, device_ids=cfg['devices'])
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer
        optimizer = make_optimizer(model, cfg['opt'])
        # schedule
        num_iters_per_epoch = math.ceil(len(train_loader) / ds_config["gradient_accumulation_steps"]) #len(train_loader)
        scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)


        # IMPORTANT: Pass the modified ds_config dict directly, NOT the file path!
        # The file doesn't include sweep parameter overrides (gradient_accumulation_steps, etc.)
        if args.local_rank == 0:
            print(f"[DeepSpeed Config] gradient_accumulation_steps={ds_config.get('gradient_accumulation_steps')}")
            print(f"[DeepSpeed Config] train_micro_batch_size_per_gpu={ds_config.get('train_micro_batch_size_per_gpu')}")

        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=parameters,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config  # Use the dict with sweep parameters applied
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

            logger.log("***************** Sweep Config (selected parameters) *****************")
            logger.log("max_seq_len: {}".format(cfg["dataset"]["max_seq_len"]))
            logger.log("learning_rate: {}".format(cfg["opt"]["learning_rate"]))
            logger.log("batch_size: {}".format(cfg["loader"]["batch_size"]))
            logger.log("gradient_accumulation_steps: {}".format(ds_config["gradient_accumulation_steps"]))
            logger.log("***********************************************************************")

            # Also print to console for visibility
            print("\n" + "=" * 60)
            print("SWEEP CONFIG (selected parameters):")
            print("=" * 60)
            print(f"  max_seq_len:                {cfg['dataset']['max_seq_len']}")
            print(f"  learning_rate:              {cfg['opt']['learning_rate']}")
            print(f"  batch_size:                {cfg['loader']['batch_size']}")
            print(f"  gradient_accumulation_steps: {ds_config['gradient_accumulation_steps']}")
            print("*" * 60 + "\n")

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
            train_loss = train_one_epoch(
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

            if args.local_rank == 0:
                wandb.log({"train/loss": train_loss})

            # save ckpt once in a while
            if (
                epoch == args.start_epoch or
                ((epoch + 1) == max_epochs) or
                ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
            ):
                print("Saving model at epoch: ", epoch + 1)
                client_sd = {}
                client_sd['epoch'] = epoch
                model_engine.save_checkpoint(ckpt_folder, tag='epoch_{:03d}'.format(epoch + 1), client_state=client_sd)
                print("Model saved at epoch: ", epoch + 1)

                ##############################
                # Distributed validation: all GPUs process their portion, gather results on rank 0
                val_db_vars = val_dataset.get_attributes()

                # Use dual evaluation (window + video level) for finegym_slide
                if cfg['dataset_name'] in ('finegym_slide', 'finediving_slide'):
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
                    window_mAP = eval_results.get('window_mAP', 0.0) if isinstance(eval_results, dict) else 0.0
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
                    window_mAP = None

                # Only rank 0 has the final mAP, determine if best
                is_best = False
                if args.local_rank == 0:
                    print("Epoch: ", epoch, ", Test mAP: ", mAP)
                    logger.log(f"[Test] Epoch {epoch}: Testset mAP = {mAP:.4f}")
                    wandb.log({"val/mAP": mAP})
                    if window_mAP is not None:
                        wandb.log({"val/window_mAP": window_mAP})

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
                    is_best = bool(is_best_tensor.item())
                
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

    except Exception as e:
        print("Exception occurred during training: ", str(e))
        import traceback
        traceback.print_exc()
        os._exit(1)
    
    finally:
        # Delete model and optimizer references
        try:
            del model_engine
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del optimizer
        except Exception:
            pass
        try:
            del scheduler
        except Exception:
            pass
        # Empty CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        # If using deepspeed, also call deepspeed's cleanup if available
        if hasattr(deepspeed, 'zero'):
            try:
                deepspeed.zero.Init.deallocate()
            except Exception:
                pass

    return

def run_sweep(args, cfg):
    """Run W&B sweep with subprocess isolation for fault tolerance."""
    wandb.login()

    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val/mAP', 'goal': 'maximize'},
        'parameters': {
            # Sliding window parameters (relevant for finegym_slide dataset)
            'max_seq_len': {'values': [144, 288, 576]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-7, 'max': 1e-3},
            'batch_size': {'values': [1, 2, 4, 8, 16]},
            'gradient_accumulation_steps': {'values': [2, 4, 6]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='finegym_sweep')

    def _agent_fn():
        """Agent function that runs each sweep trial in a subprocess for isolation."""
        # Build the deepspeed command
        cmd = [
            "deepspeed",
            f"--include=localhost:{args.gpus}",  # Format: localhost: 0,1
            "--master_port=12345",
            "train_shard_wandb.py",
            "--config", args.config,
            "--output", args.output,
        ]

        # Run in subprocess - OS reclaims all GPU memory when process exits
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Trail job failed (exist code {result.returncode}). GPUs freed.")
        else:
            print("Trial job finished successfully. GPUs freed.")

    wandb.agent(sweep_id, function=_agent_fn, count=args.sweep_count)

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR', default='./configs/finegym_i3d.yaml', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=1, type=int, help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='deepspeed_raw_video', type=str, help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a checkpoint (default: none)')
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus")
    parser.add_argument("--sweep", action='store_true', help="Run a W&B hyperparameter sweep")
    parser.add_argument("--sweep_count", default=50, type=int, help="number of sweep trials to run")
    parser.add_argument("--gpus", default="0,1", type=str, help='GPUs indices for sweep (e.g., "0,1" or "2,3")')
    args = parser.parse_args()
    print(args.local_rank)

    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError(f"Config file doees not exist.")

    if args.sweep:
        # Run W&B sweep (this is the sweep controller, not the training process)
        run_sweep(args, cfg)
    else:
        # Run training directly (this is called by deepspeed from subprocess)
        main(args, cfg)