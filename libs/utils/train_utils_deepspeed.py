import os
import shutil
import time
import pickle
import json
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from collections import defaultdict

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm

import logging
from tqdm import tqdm 
import copy
import deepspeed


def get_pacific_time():
    """Get current time in Pacific timezone as formatted string."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S")

################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure the logger
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(message)s',
            datefmt='%Y-%m-%d',
            level=logging.INFO
        )

    def log(self, message):
        logging.info(message)


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)
            elif 'encoder' in pn:
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        model_to_copy = model.module if hasattr(model, 'module') else model
        self.module = deepcopy(model_to_copy)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        underlying_model = model.module if hasattr(model, 'module') else model

        with deepspeed.zero.GatheredParameters(underlying_model.parameters(), modifier_rank=0):
            self._update(underlying_model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

        # self._update(underlying_model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema = None,
    clip_grad_l2norm = -1,
    print_freq = 20,
    save_log = True
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    if save_log:
        print("\n[Train]: Epoch {:d} started".format(curr_epoch))
        logger.log("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    init_time = copy.copy(start)
    for iter_idx, video_list in enumerate(train_loader, 0):
        # forward / backward the model
        losses = model(video_list)
        model.backward(losses['final_loss'])
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        model.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if save_log and (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # print to terminal
            block1 = '[{}] Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                get_pacific_time(), curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))
            logger.log('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    lr = model.lr_scheduler.optimizer.param_groups[0]['lr']
    if save_log:
        print("[Train]: Epoch {:d} finished with lr={:.8f} Total Time: {:.4f} mins \n".format(curr_epoch, lr, (time.time() - init_time) / 60))
        logger.log("[Train]: Epoch {:d} finished with lr={:.8f} Total Time: {:.4f} mins \n".format(curr_epoch, lr, (time.time() - init_time) / 60))
    
    # Return average final loss for logging (e.g., wandb)
    avg_loss = losses_tracker['final_loss'].avg if 'final_loss' in losses_tracker else 0.0
    return avg_loss


def save_logits(
    train_loader,
    model,
    round,
    al_method = 'adaptive'  # 'uniform', 'adaptive'
):
    """save logits for each video in the training set"""
    # switch to train mode
    model.eval()
    video_logits_dict = {}

    for iter_idx, video_list in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        with torch.no_grad():
            # forward / backward the model
            model(video_list)
            core_model = model.module if hasattr(model, 'module') else model
            video_logits_dict.update(core_model.last_video_label_logits)
        
    # finish up and print
    dataset_name = train_loader.dataset.db_attributes['dataset_name'].split('-')[0]
    save_path = f'/data3/xiaodan8/actionformer4_1/output/{dataset_name}/logits_{al_method}_{round}.pt'
    torch.save(video_logits_dict, save_path)
    print(f'[Train] Saved per-frame logits to {save_path}')
    return


def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    print_freq = 20,
    best_map = 0.0,
    if_save_data = False
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    if len(results['video-id']) == 0:
        print("No results found!")
        return 0.0
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP, _, ap = evaluator.evaluate(results, verbose=if_save_data)
        if mAP > best_map:
            if_save_data = True
        if if_save_data:
            dataset_name = evaluator.dataset_name.split('_')[0]
            np.save(f'/data3/xiaodan8/actionformer4_1/output/{dataset_name}/pred_vit_{output_file}_ap.npy', np.mean(ap, axis=0))
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(f'/data3/xiaodan8/actionformer4_1/output/{output_file}', "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    if if_save_data:
        from collections import defaultdict
        import json
        new_res = defaultdict(list)
        for k in range(len(results['video-id'])):
            new_res[results['video-id'][k]].append({'t-start': str(results['t-start'][k]), 't-end': str(results['t-end'][k]), 'label': str(results['label'][k]), 'score': str(results['score'][k])})
        new_res = {key:sorted(values, key=lambda x:float(x['t-start'])) for key,values in new_res.items()}
        dataset_name = evaluator.dataset_name.split('_')[0]
        with open(f'/data3/xiaodan8/actionformer4_1/output/pred_vit_{output_file}.json', 'w') as json_file:
            json.dump(new_res, json_file)

    return mAP


def valid_one_epoch_distributed(
    val_loader,
    model,
    current_epoch,
    evaluator,
    output_file,
    print_freq=20,
    if_save_data=True,
    best_map=0.0,
    local_rank=0
):
    """
    Distributed validation: each GPU processes its portion, then gather results to rank 0.
    this avoids redundant computation where all GPUs run the full test set.
    """
    batch_time = AverageMeter()
    model.eval()
    
    # Each GPU collects its partial results
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)

            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0 and local_rank == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            print('[{0}] GPU{1} Test: [{2:05d}/{3:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  get_pacific_time(), local_rank, iter_idx, len(val_loader), batch_time=batch_time))

    # Convert partial results to numpy
    if len(results['video-id']) > 0:
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
    else:
        results['t-start'] = np.array([])
        results['t-end'] = np.array([])
        results['label'] = np.array([])
        results['score'] = np.array([])

    # Gather results from all GPUs to rank 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size > 1:
        # Serialize results for gathering
        results_bytes = pickle.dumps(results)
        results_tensor = torch.ByteTensor(list(results_bytes)).cuda()

        # Gather sizes first (resultsmay have different sizes on each GPU)
        local_size = torch.tensor([results_tensor.numel()], dtype=torch.long, device='cuda')
        size_list = [torch.tensor([0], dtype=torch.long, device='cuda') for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # Pad tensors to max size
        if results_tensor.numel() < max_size:
            padding = torch.zeros(max_size - results_tensor.numel(), dtype=torch.uint8, device='cuda')
            results_tensor = torch.cat((results_tensor, padding))

        # Gather all tensors to rank 0
        if local_rank == 0:
            gather_list = [torch.empty((max_size,), dtype=torch.uint8, device='cuda') for _ in range(world_size)]
        else:
            gather_list = None

        dist.gather(results_tensor, gather_list, dst=0)

        # Only rank 0 processes gathered results
        if local_rank == 0:
            all_results = {
                'video-id': [],
                't-start' : [],
                't-end': [],
                'label': [],
                'score': []
            }
            
            for i, tensor in enumerate(gather_list):
                # Deserialize
                result_bytes = bytes(tensor[:size_list[i]].cpu().numpy().tolist())
                result_part = pickle.loads(result_bytes)

                all_results['video-id'].extend(result_part['video-id'])
                if len(result_part['t-start']) > 0:
                    all_results['t-start'].append(result_part['t-start'])
                    all_results['t-end'].append(result_part['t-end'])
                    all_results['label'].append(result_part['label'])
                    all_results['score'].append(result_part['score'])

            # Concatenate all results
            if len(all_results['t-start']) > 0:
                all_results['t-start'] = np.concatenate(all_results['t-start'])
                all_results['t-end'] = np.concatenate(all_results['t-end'])
                all_results['label'] = np.concatenate(all_results['label'])
                all_results['score'] = np.concatenate(all_results['score'])
            else:
                print("[{}] No results found!".format(get_pacific_time()))
                return 0.0

            results = all_results

    # Only rank 0 evaluates
    mAP = 0.0
    if local_rank == 0:
        if len(results['video-id']) == 0:
            print("No results found!")
            return 0.0

        _, mAP, _, ap = evaluator.evaluate(results, verbose=True)

        # Save results if better than best and if_save_data is True
        if if_save_data and mAP > best_map:
            dataset_name = evaluator.dataset_name.split('_')[0]
            np.save(f'/data3/xiaodan8/actionformer4_1/output/pred_vit_{output_file}_ap.npy', np.mean(ap, axis=0))
            
            new_res = defaultdict(list)
            for k in range(len(results['video-id'])):
                new_res[results['video-id'][k]].append({
                    't-start': str(results['t-start'][k]), 
                    't-end': str(results['t-end'][k]), 
                    'label': str(results['label'][k]), 
                    'score': str(results['score'][k])
                    })
            new_res = {key:sorted(values, key=lambda x:float(x['t-start'])) for key,values in new_res.items()}
            with open(f'/data3/xiaodan8/actionformer4_1/output/pred_vit_{output_file}.json', 'w') as json_file:
                json.dump(new_res, json_file)
    
    return mAP

def valid_one_epoch_video_level(
    val_loader,
    model,
    curr_epoch,
    evaluator,
    output_file,
    print_freq=20,
    if_save_data=True,
    best_map=0.0,
    local_rank=0,
    iou_threshold=0.5,
    min_score=0.001,
    max_seg_num=200
):
    """
    Validation with video-level aggregation for overlapping windows.

    This function:
    1. Runs inference on all windows
    2. Converts window-relative predictions to video-relative coordinates
    3. Aggregates predictions by video and applies NMS
    4. Evaluate s at video level

    Use this with finegym_slide when test_overlap=True.
    """
    from ..datasets.finegym_slide import aggregate_window_predictions
    from .nms import batched_nms

    batch_time = AverageMeter()
    model.eval()

    # Collect predictions with window offset info
    results = {
        'video-id': [],           # Window ID (e.g., "video_w0")
        'video-name' : [],        # Original video name (e.g., "video")
        'window-start-time' : [], # Window start time in video
        't-start' : [],           # Prediction start (window-relative)
        't-end': [],              # Prediction end (window-relative)
        'label': [],
        'score': []
    }

    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)

            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    n_preds = output[vid_idx]['segments'].shape[0]

                    # Window ID
                    results['video-id'].extend([output[vid_idx]['video_id']] * n_preds)

                    # Original video name (for aggregation)
                    video_name = video_list[vid_idx].get('video_name', output[vid_idx]['video_id'].rsplit('_w', 1)[0])
                    results['video-name'].extend([video_name] * n_preds)

                    # Window start time
                    window_start = video_list[vid_idx].get('window_start_time', 0.0)
                    results['window-start-time'].extend([window_start] * n_preds)

                    # Prediction (window-relative)
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        if (iter_idx != 0) and iter_idx % (print_freq) == 0 and local_rank == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            print('[{0}] Test: [{1:05d}/{2:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  get_pacific_time(), iter_idx, len(val_loader), batch_time=batch_time))
            
    # Convert to numpy
    if len(results['video-id']) > 0:
        results['t-start'] = torch.cat(results['t-start']).cpu().numpy()
        results['t-end'] = torch.cat(results['t-end']).cpu().numpy()
        results['label'] = torch.cat(results['label']).cpu().numpy()
        results['score'] = torch.cat(results['score']).cpu().numpy()
        results['window-start-time'] = np.array(results['window-start-time'])
    else:
        results['t-start'] = np.array([])
        results['t-end'] = np.array([])
        results['label'] = np.array([])
        results['score'] = np.array([])
        results['window-start-time'] = np.array([])

    # Gather results from all GPUs
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size > 1:
        results_bytes = pickle.dumps(results)
        results_tensor = torch.ByteTensor(list(results_bytes)).cud()

        local_size = torch.tensor([results_tensor.numel()], dtype=torch.long, device='cuda')
        size_list = [torch.tensor([0], dtype=torch.long, device='cuda') for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        if results_tensor.numel() < max_size:
            padding = torch.zeros(max_size - results_tensor.numel(), dtype=torch.uint8, device='cuda')
            results_tensor = torch.cat((results_tensor, padding))

        if local_rank == 0:
            gather_list = [torch.empty((max_size,), dtype=torch.uint8, device='cuda') for _ in range(world_size)]
        else:
            gather_list = None

        dist.all_gather(gather_list, results_tensor, dst=0)

        # Only rank 0 processes gathered results
        if local_rank == 0:
            all_results = {
            'video-id': [],
            'video-name' : [],
            'window-start-time' : [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }
            
        for i, tensor in enumerate(gather_list):
            result_bytes = bytes(tensor[:size_list[i]].cpu().numpy().tolist())
            result_part = pickle.loads(result_bytes)

            all_results['video-id'].extend(result_part['video-id'])
            all_results['video-name'].extend(result_part['video-name'])
            
            if len(result_part['t-start']) > 0:
                all_results['window-start-time'].append(result_part['window-start-time'])
                all_results['t-start'].append(result_part['t-start'])
                all_results['t-end'].append(result_part['t-end'])
                all_results['label'].append(result_part['label'])
                all_results['score'].append(result_part['score'])
        
        if len(all_results['t-start']) > 0:
            all_results['window-start-time'] = np.concatenate(all_results['window-start-time'])
            all_results['t-start'] = np.concatenate(all_results['t-start'])
            all_results['t-end'] = np.concatenate(all_results['t-end'])
            all_results['label'] = np.concatenate(all_results['label'])
            all_results['score'] = np.concatenate(all_results['score'])
        else:
            print("[{}] No results found!".format(get_pacific_time()))
            return 0.0

        results = all_results

    # Only rank 0 does aggregation and evaluation
    mAP = 0.0
    if local_rank == 0:
        if len(results['video-id']) == 0:
            print("No results found!")
            return 0.0

        # Convert window-relative predictions to video-relative
        video_results = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }

        # Group by video and convert coordinates
        video_predictions = defaultdict(lambda: {'segs': [], 'scores': [], 'labels': []})

        for i in range(len(results['video-id'])):
            video_name = results['video-name'][i]
            window_start = results['window-start-time'][i]

            # Convert to video-relative coordinates
            t_start_video = results['t-start'][i] + window_start
            t_end_video = results['t-end'][i] + window_start

            video_predictions[video_name]['segs'].append([t_start_video, t_end_video])
            video_predictions[video_name]['scores'].append(results['score'][i])
            video_predictions[video_name]['labels'].append(results['label'][i])

        # Apply NMS per video
        print(f"[{get_pacific_time()}] Aggregating predictions from {len(video_predictions)} videos...")

        for vid, preds in video_predictions.items():
            if len(preds['segs']) == 0:
                continue

            segs = torch.tensor(preds['segs'], dtype=torch.float32)
            scores = torch.tensor(preds['scores'], dtype=torch.float32)
            labels = torch.tensor(preds['labels'], dtype=torch.float32)

            # Apply NMS
            nms_segs, nms_scores, nms_labels = batched_nms(
                segs, scores, labels, 
                iou_threshold=iou_threshold,
                min_score=min_score,
                max_seg_num=max_seg_num,
                use_soft_nms=True,
                multiclass=True
            )

            # Add to results
            for j in range(len(nms_segs)):
                video_results['video-id'].append(vid)
                video_results['t-start'].append(nms_segs[j, 0].item())
                video_results['t-end'].append(nms_segs[j, 1].item())
                video_results['label'].append(nms_labels[j].item())
                video_results['score'].append(nms_scores[j].item())
        
        # Convert to arrays for evaluator
        video_results['t-start'] = np.array(video_results['t-start'])
        video_results['t-end'] = np.array(video_results['t-end'])
        video_results['label'] = np.array(video_results['label'])
        video_results['score'] = np.array(video_results['score'])

        print(f"[{get_pacific_time()}] After NMS: {len(video_results['video-id'])} predictions")

        # Evaluate
        _, mAP, _, ap = evaluator.evaluate(video_results, verbose=True)

        # Save results if better
        if if_save_data and mAP > best_map:
            dataset_name = evaluator.dataset_name.split('_')[0]
            np.save(f'/data3/xiaodan8/actionformer4_1/output/pred_{output_file}_video_level_ap.npy', np.mean(ap, axis=0))
            
            new_res = defaultdict(list)
            for k in range(len(video_results['video-id'])):
                new_res[video_results['video-id'][k]].append({
                    't-start': str(video_results['t-start'][k]), 
                    't-end': str(video_results['t-end'][k]), 
                    'label': str(video_results['label'][k]), 
                    'score': str(video_results['score'][k])
                })
            new_res = {key:sorted(values, key=lambda x:float(x['t-start'])) for key,values in new_res.items()}
            with open(f'/data3/xiaodan8/actionformer4_1/output/pred_{output_file}_video_level.json', 'w') as json_file:
                json.dump(new_res, json_file)

    return mAP


def valid_one_epoch_slide_dual_eval(
    val_loader,
    model,
    val_dataset,
    evaluator_class,
    tiou_thresholds,
    curr_epoch,
    output_file,
    print_freq=20,
    if_save_data=True,
    best_map=0.0,
    local_rank=0,
    iou_threshold=0.5,
    min_score=0.001,
    max_seg_num=200
):
    """
    Validation for FineGym sliding window with BOTH window-level and video-level evaluation.

    This function:
    1. Runs inference on all windows
    2. Evaluates at window level (each window as independent instance)
    3. Converts window-relative predictions to video-relative coordinates
    4. Aggregates predictions by video and applies NMS
    5. Evaluates at video level

    Args:
        val_loader: DataLoader for validation
        model: Model to evaluate
        val_dataset: FineGymSlideDataset instance (needed for ground truth)
        evaluator_class: ANETdetection class
        tiou_thresholds: tIoU thresholds for evaluation
        curr_epoch: Current epoch number
        output_file: Output file prefix
        print_freq: Print frequency
        if_save_data: Whether to save results
        best_map: Best mAP so far (for comparison)
        local_rank: GPU rank for distributed training
        iou_threshold: IoU threshold for NMS
        min_score: Minimum score threshold
        max_seg_num: Maximum number of segments per video

    Returns:
        dict with 'window_mAP' and 'video_mAP'
    """
    from .nms import batched_nms

    batch_time = AverageMeter()
    model.eval()

    # Collect predictions with window offset info
    results = {
        'video-id': [],           # Window ID (e.g., "video_w0")
        'video-name': [],         # Original video name (e.g., "video")
        'window-start-time': [],  # Window start time in video
        't-start': [],            # Prediction start (window-relative)
        't-end': [],              # Prediction end (window-relative)
        'label': [],
        'score': []
    }

    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)

            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    n_preds = output[vid_idx]['segments'].shape[0]

                    # Window ID
                    results['video-id'].extend([output[vid_idx]['video_id']] * n_preds)

                    # Original video name (for aggregation)
                    video_name = video_list[vid_idx].get('video_name', output[vid_idx]['video_id'].rsplit('_w', 1)[0])
                    results['video-name'].extend([video_name] * n_preds)

                    # Window start time
                    window_start = video_list[vid_idx].get('window_start_time', 0.0)
                    results['window-start-time'].extend([window_start] * n_preds)

                    # Prediction (window-relative)
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        if (iter_idx != 0) and iter_idx % (print_freq) == 0 and local_rank == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()
            print('[{0}] Test: [{1:05d}/{2:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  get_pacific_time(), iter_idx, len(val_loader), batch_time=batch_time))

    # Convert to numpy
    if len(results['video-id']) > 0:
        results['t-start'] = torch.cat(results['t-start']).cpu().numpy()
        results['t-end'] = torch.cat(results['t-end']).cpu().numpy()
        results['label'] = torch.cat(results['label']).cpu().numpy()
        results['score'] = torch.cat(results['score']).cpu().numpy()
        results['window-start-time'] = np.array(results['window-start-time'])
    else:
        results['t-start'] = np.array([])
        results['t-end'] = np.array([])
        results['label'] = np.array([])
        results['score'] = np.array([])
        results['window-start-time'] = np.array([])

    # Gather results from all GPUs
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size > 1:
        results_bytes = pickle.dumps(results)
        results_tensor = torch.ByteTensor(list(results_bytes)).cuda()

        local_size = torch.tensor([results_tensor.numel()], dtype=torch.long, device='cuda')
        size_list = [torch.tensor([0], dtype=torch.long, device='cuda') for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        if results_tensor.numel() < max_size:
            padding = torch.zeros(max_size - results_tensor.numel(), dtype=torch.uint8, device='cuda')
            results_tensor = torch.cat((results_tensor, padding))

        if local_rank == 0:
            gather_list = [torch.empty((max_size,), dtype=torch.uint8, device='cuda') for _ in range(world_size)]
        else:
            gather_list = None
        dist.gather(results_tensor, gather_list, dst=0)

        # Only rank 0 processes gathered results
        if local_rank == 0:
            all_results = {
                'video-id': [],
                'video-name': [],
                'window-start-time': [],
                't-start': [],
                't-end': [],
                'label': [],
                'score': []
            }

            for i, tensor in enumerate(gather_list):
                result_bytes = bytes(tensor[:size_list[i]].cpu().numpy().tolist())
                result_part = pickle.loads(result_bytes)

                all_results['video-id'].extend(result_part['video-id'])
                all_results['video-name'].extend(result_part['video-name'])

                if len(result_part['t-start']) > 0:
                    all_results['window-start-time'].append(result_part['window-start-time'])
                    all_results['t-start'].append(result_part['t-start'])
                    all_results['t-end'].append(result_part['t-end'])
                    all_results['label'].append(result_part['label'])
                    all_results['score'].append(result_part['score'])

            if len(all_results['t-start']) > 0:
                all_results['window-start-time'] = np.concatenate(all_results['window-start-time'])
                all_results['t-start'] = np.concatenate(all_results['t-start'])
                all_results['t-end'] = np.concatenate(all_results['t-end'])
                all_results['label'] = np.concatenate(all_results['label'])
                all_results['score'] = np.concatenate(all_results['score'])
            else:
                print("[{}] No results found!".format(get_pacific_time()))
                return {'window_mAP': 0.0, 'video_mAP': 0.0}

            results = all_results

    # Only rank 0 does evaluation
    eval_results = {'window_mAP': 0.0, 'video_mAP': 0.0}

    if local_rank == 0:
        if len(results['video-id']) == 0:
            print("No results found!")
            return eval_results

        # ===================== WINDOW-LEVEL EVALUATION ====================
        print("\n" + "=" * 60)
        print("WINDOW-LEVEL EVALUATION")
        print("(Each sliding window treated as independent instance)")
        print("=" * 60)

        window_gt_df = val_dataset.get_ground_truth_df()
        window_evaluator = evaluator_class(
            ant_file=None,
            split=None,
            tiou_thresholds=tiou_thresholds,
            ground_truth_df=window_gt_df,
            dataset_name='finegym_slide_window'
        )

        # Window results (already in window-relative coordinates)
        window_eval_results = {
            'video-id': results['video-id'],
            't-start': results['t-start'],
            't-end': results['t-end'],
            'label': results['label'],
            'score': results['score']
        }

        _, window_mAP, _, window_ap = window_evaluator.evaluate(window_eval_results, verbose=True)
        eval_results['window_mAP'] = window_mAP

        # ==================== VIDEO-LEVEL EVALUATION ====================
        print("\n" + "=" * 60)
        print("VIDEO-LEVEL EVALUATION")
        print("(Predictions aggregated across overlapping windows)")
        print("=" * 60)

        video_gt_df = val_dataset.get_video_level_ground_truth_df()
        video_evaluator = evaluator_class(
            ant_file=None,
            split=None,
            tiou_thresholds=tiou_thresholds,
            ground_truth_df=video_gt_df,
            dataset_name=val_dataset.db_attributes['dataset_name'] + '_video'
        )

        # Convert window-relative to video-relative and aggregate
        video_results = {
            'video-id': [],
            't-start': [],
            't-end': [],
            'label': [],
            'score': []
        }

        video_predictions = defaultdict(lambda: {'segs': [], 'scores': [], 'labels': []})

        for i in range(len(results['video-id'])):
            video_name = results['video-name'][i]
            window_start = results['window-start-time'][i]

            # Convert to video-relative coordinates
            t_start_video = results['t-start'][i] + window_start
            t_end_video = results['t-end'][i] + window_start

            video_predictions[video_name]['segs'].append([t_start_video, t_end_video])
            video_predictions[video_name]['scores'].append(results['score'][i])
            video_predictions[video_name]['labels'].append(results['label'][i])

        print(f"[{get_pacific_time()}] Aggregating predictions from {len(video_predictions)} videos...")

        # Apply NMS per video
        for vid, preds in video_predictions.items():
            if len(preds['segs']) == 0:
                continue

            segs = torch.tensor(preds['segs'], dtype=torch.float32)
            scores = torch.tensor(preds['scores'], dtype=torch.float32)
            labels = torch.tensor(preds['labels'], dtype=torch.int64)

            nms_segs, nms_scores, nms_labels = batched_nms(
                segs, scores, labels,
                iou_threshold=iou_threshold,
                min_score=min_score,
                max_seg_num=max_seg_num,
                use_soft_nms=True,
                multiclass=True
            )

            for j in range(len(nms_segs)):
                video_results['video-id'].append(vid)
                video_results['t-start'].append(nms_segs[j, 0].item())
                video_results['t-end'].append(nms_segs[j, 1].item())
                video_results['label'].append(nms_labels[j].item())
                video_results['score'].append(nms_scores[j].item())

        video_results['t-start'] = np.array(video_results['t-start'])
        video_results['t-end'] = np.array(video_results['t-end'])
        video_results['label'] = np.array(video_results['label'])
        video_results['score'] = np.array(video_results['score'])

        print(f"[{get_pacific_time()}] After NMS: {len(video_results['video-id'])} predictions")

        # ==================== REMOVE EXACT DUPLICATES ====================
        # Remove duplicate predictions with same (video-id, t-start, t-end, label),
        # keeping only the highest-scoring one
        if len(video_results['video-id']) > 0:
            # Create unique key for each prediction
            unique_preds = {}  # key: (vid, t_start, t_end, label) -> index of highest score
            for i in range(len(video_results['video-id'])):
                key = (
                    video_results['video-id'][i],
                    round(video_results['t-start'][i], 6),  # Round to avoid floating point issues
                    round(video_results['t-end'][i], 6),
                    int(video_results['label'][i])
                )
                if key not in unique_preds:
                    unique_preds[key] = i
                elif video_results['score'][i] > video_results['score'][unique_preds[key]]:
                    unique_preds[key] = i  # Keep the higher scoring one

            # Get indices to keep
            keep_indices = sorted(unique_preds.values())
            num_duplicates = len(video_results['video-id']) - len(keep_indices)

            # Filter results
            video_results['video-id'] = [video_results['video-id'][i] for i in keep_indices]
            video_results['t-start'] = video_results['t-start'][keep_indices]
            video_results['t-end'] = video_results['t-end'][keep_indices]
            video_results['label'] = video_results['label'][keep_indices]
            video_results['score'] = video_results['score'][keep_indices]
            
            print(f"[{get_pacific_time()}] Removed {num_duplicates} exact duplicates, {len(video_results['video-id'])} predictions remaining")

        _, video_mAP, _, video_ap = video_evaluator.evaluate(video_results, verbose=True)
        eval_results['video_mAP'] = video_mAP

        # ==================== SUMMARY ====================
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Window-level mAP: {window_mAP:.4f}")
        print(f"Video-level mAP:  {video_mAP:.4f}")
        print("=" * 60 + "\n")

        # Save results if better
        if if_save_data and video_mAP > best_map:
            dataset_name = 'finegym'
            output_dir = f'/data3/xiaodan8/actionformer4_1/output/{dataset_name}'
            os.makedirs(output_dir, exist_ok=True)
            np.save(f'{output_dir}/pred_{output_file}_window_ap.npy', np.mean(window_ap, axis=0))
            np.save(f'{output_dir}/pred_{output_file}_video_ap.npy', np.mean(video_ap, axis=0))

            # Save video-level predictions
            new_res = defaultdict(list)
            for k in range(len(video_results['video-id'])):
                new_res[video_results['video-id'][k]].append({
                    't-start': str(video_results['t-start'][k]),
                    't-end': str(video_results['t-end'][k]),
                    'label': str(video_results['label'][k]),
                    'score': str(video_results['score'][k])
                })
            new_res = {key: sorted(values, key=lambda x: float(x['t-start'])) for key, values in new_res.items()}
            with open(f'{output_dir}/pred_{output_file}_video_level.json', 'w') as json_file:
                json.dump(new_res, json_file)

    return eval_results
