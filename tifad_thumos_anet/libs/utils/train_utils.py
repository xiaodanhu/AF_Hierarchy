import json
import os
import pickle
import random
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from .lr_schedulers import LinearWarmupCosineAnnealingLR, LinearWarmupMultiStepLR
from .postprocessing import postprocess_results

import time
from thop import profile
# from torch.cuda.amp import autocast, GradScaler
import json



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


def load_init_from(model, state_dict):
    """Warm-start `model` from a donor state_dict (TIFAD_INIT_FROM /
    TIFAD_EVAL_NONSTRICT): strict=False so NEW (zero-init) arm parameters
    absent from the donor keep their init. Shape-mismatched entries would
    make load_state_dict raise even with strict=False; the one legitimate
    case is `famce_uw_logvar`, which grows from 2 to 3 scalars under
    TIFAD_LEAFLOSS=1 -- the donor's values are copied into the FIRST slots
    and the new slot keeps its init (0). Any other mismatch still raises.
    Returns (missing, unexpected) key lists."""
    own = model.state_dict()
    sd = dict(state_dict)
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape:
            if (k.endswith('famce_uw_logvar') and v.dim() == 1
                    and own[k].dim() == 1 and own[k].numel() > v.numel()):
                new = own[k].detach().clone().cpu()
                new[:v.numel()] = v.to(new.dtype)
                sd[k] = new
                print(f"[INITFROM] {k}: donor {tuple(v.shape)} copied into "
                      f"the first slots of {tuple(own[k].shape)} "
                      f"-> {new.tolist()}", flush=True)
            else:
                raise RuntimeError(
                    f"[INITFROM] shape mismatch for {k}: donor "
                    f"{tuple(v.shape)} vs model {tuple(own[k].shape)}")
    return model.load_state_dict(sd, strict=False)


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
    light_decay = set()    
    no_decay = set()
    

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue    # frozen (TIFAD_FREEZE_EXCEPT) params stay out
        if pn.endswith(".bias"):
            no_decay.add(pn)    # for bias
        elif pn.endswith(".weight") and "norm" in pn:
            no_decay.add(pn)    # for LayerNorm
        elif "cls_head" in pn or "reg_head" in pn:
            light_decay.add(pn)
        elif pn.endswith(".rel_pe"):
            no_decay.add(pn)    # for positional encoding
        else:
            decay.add(pn)    # others


    # validate that we considered every parameter (trainable only)
    param_dict = {pn: p for pn, p in model.named_parameters()
                  if p.requires_grad}
    inter_params = decay & no_decay & light_decay
    union_params = decay | no_decay | light_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(light_decay))], "weight_decay": 0.0001},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # ---- TIFAD_LR_MULT_REST=<float> (2026-09-17, VSA v2 arm (C)): a
    # 'whole model at low lr' stage. Parameters whose name contains one of
    # the TIFAD_FREEZE_EXCEPT substrings keep the full learning rate;
    # every OTHER trainable parameter goes into a separate param group
    # (same weight-decay class) with lr = learning_rate * multiplier.
    # train_eval.py does NOT freeze anything when this env is set (the
    # FREEZE_EXCEPT names then only select the full-lr set). Per-group
    # 'lr' is honoured by the warmup/cosine schedulers (base_lrs are read
    # per group). Byte-inert when unset (the three groups above unchanged).
    _rest_mult = os.environ.get('TIFAD_LR_MULT_REST', '')
    if _rest_mult:
        _rest_mult = float(_rest_mult)
        _full = [k.strip() for k in
                 os.environ.get('TIFAD_FREEZE_EXCEPT', '').split(',')
                 if k.strip()]
        assert _full, "TIFAD_LR_MULT_REST needs TIFAD_FREEZE_EXCEPT names"
        _base_lr = optimizer_config["learning_rate"]
        _new_groups, _n_full, _n_rest = [], 0, 0
        for _names, _wd in ((decay, optimizer_config['weight_decay']),
                            (light_decay, 0.0001), (no_decay, 0.0)):
            _f = [pn for pn in sorted(list(_names))
                  if any(k in pn for k in _full)]
            _r = [pn for pn in sorted(list(_names))
                  if not any(k in pn for k in _full)]
            _n_full += len(_f)
            _n_rest += len(_r)
            if _f:
                _new_groups.append({"params": [param_dict[pn] for pn in _f],
                                    "weight_decay": _wd, "lr": _base_lr})
            if _r:
                _new_groups.append({"params": [param_dict[pn] for pn in _r],
                                    "weight_decay": _wd,
                                    "lr": _base_lr * _rest_mult})
        optim_groups = _new_groups
        print(f"[LR_MULT_REST] full lr ({_base_lr:g}) for {_n_full} params "
              f"matching {_full}; {_n_rest} other params at "
              f"{_base_lr * _rest_mult:g} (x{_rest_mult:g})", flush=True)

    # ---- TIFAD_LR_MULT_HEAD=<float> (2026-09-18, VSA v5 phase head): the
    # trainable parameters whose name contains one of the
    # TIFAD_FREEZE_EXCEPT substrings get lr = learning_rate * multiplier
    # (own param groups, same weight-decay class); everything else keeps
    # the base lr. For sequential stages that train a randomly initialised
    # head from scratch on the frozen champion, where the recipe's 1e-4 is
    # too slow (the post-hoc probe of the same head used 1e-3). Per-group
    # 'lr' is honoured by the schedulers. Byte-inert when unset.
    _head_mult = os.environ.get('TIFAD_LR_MULT_HEAD', '')
    if _head_mult and not _rest_mult:
        _head_mult = float(_head_mult)
        _hn = [k.strip() for k in
               os.environ.get('TIFAD_FREEZE_EXCEPT', '').split(',')
               if k.strip()]
        assert _hn, "TIFAD_LR_MULT_HEAD needs TIFAD_FREEZE_EXCEPT names"
        _base_lr = optimizer_config["learning_rate"]
        _new_groups, _n_head = [], 0
        for _names, _wd in ((decay, optimizer_config['weight_decay']),
                            (light_decay, 0.0001), (no_decay, 0.0)):
            _h = [pn for pn in sorted(list(_names))
                  if any(k in pn for k in _hn)]
            _o = [pn for pn in sorted(list(_names))
                  if not any(k in pn for k in _hn)]
            _n_head += len(_h)
            if _h:
                _new_groups.append({"params": [param_dict[pn] for pn in _h],
                                    "weight_decay": _wd,
                                    "lr": _base_lr * _head_mult})
            if _o:
                _new_groups.append({"params": [param_dict[pn] for pn in _o],
                                    "weight_decay": _wd, "lr": _base_lr})
        optim_groups = _new_groups
        print(f"[LR_MULT_HEAD] {_n_head} params matching {_hn} at "
              f"{_base_lr * _head_mult:g} (x{_head_mult:g}); the rest at "
              f"{_base_lr:g}", flush=True)

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
    elif optimizer_config["type"] == "Adam":
        # PDA reproduction: plain Adam (weight_decay taken from the groups)
        optimizer = optim.Adam(
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
                gamma=optimizer_config["schedule_gamma"],
                # gamma=schedule_config["gamma"],
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
        self.module = deepcopy(model)
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
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    # scaler,                     
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
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
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)

        # with autocast():
        # forward / backward the model
        losses = model(video_list)
        # scaler.scale(losses['final_loss']).backward()
        losses['final_loss'].backward()

        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm,
            )
        
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        # if (iter_idx != 0) and (iter_idx % print_freq) == 0:
        if iter_idx % print_freq == 0:
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

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
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

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    
    return

@torch.no_grad()
def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    # label_to_name = {},
    prior_prob = 0.15,
    num_pred = 300
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
        'score': [],
        'class_names': [],
    }
    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # upack the results into ANet format
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
                    results['class_names'].append(output[vid_idx]['class_names'])

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
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    
    # ---- TIFAD_DUMP_RAW=<path> (EVAL ONLY, env-gated, byte-inert unset):
    # pickle the decoded per-video detections BEFORE ext-score fusion so
    # post-hoc passes (e.g. stitch tuning) can be swept offline on CPU
    # without GPU re-runs. The dump is exactly the `results` dict the
    # fusion + evaluator consume.
    import os as _osdr
    _draw = _osdr.environ.get('TIFAD_DUMP_RAW', '')
    if _draw:
        with open(_draw, 'wb') as f:
            pickle.dump(results, f)
        print(f"[DUMPRAW] wrote {_draw}", flush=True)

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file, prior_prob, num_pred)

        # call the evaluator
        _, mAP, tiou_mAP_list = evaluator.evaluate(results, verbose=True)

    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP, tiou_mAP_list