#!/usr/bin/env python3
"""Cache the FROZEN CLIP-encoder output of a FineGym champion checkpoint for
every training and test window (frozen-encoder stages, e.g. stage 3 = video
branch of the action--sub-action attention).

What is cached, and why it reproduces the frame path bit for bit
------------------------------------------------------------------
The model pads every window to max_seq_len (288) with ZERO IMAGES before the
encoder (meta_archs.preprocessing), so the encoder also sees 256 padding
frames per window and the first embedding conv (kernel 3, unmasked input)
reads the padding-frame feature next to the last real frame. Under the bf16
DeepSpeed engine the eval path runs ONE window at a time: 288 frames through
CLIP in chunks of clip_forward_chunk = 64 frames ([32 real + 32 pad], 3 x
[64 pad], [32 pad]), mean-pooled, then the 768->2048 projection over the 288
rows. Rows of every op involved are independent of the other rows in the
batch given the same GEMM shape, so we reproduce EXACTLY that computation:
  * chunk 0 = the window's 32 frames + 32 zero frames -> CLIP (64 frames)
  * the pure-padding chunks (64 and 32 zero frames) are computed once
  * the projection is applied to the assembled (1, 288, 768) tensor
and keep rows 0..31 as the window's features (bf16 bit pattern, uint16) and
row 32 (first padding row) as `pad_feat.pt`, which the backbone substitutes
at padded timesteps (backbones.py, feature input). --verify N re-runs the
untouched frame path (288-frame padded input through model.backbone.encoder)
on N windows per split and asserts bitwise equality.

Training windows are cached with the DETERMINISTIC eval transform (resize
256 + centre crop 224, no flip) -- one fixed view per window, like the I3D
features of the THUMOS stage-3 recipe (no augmentation in that stage).

Usage (from the repo root, one GPU):
  CUDA_VISIBLE_DEVICES=G python scripts/cache_fg3_clip_feats.py \
      --config configs/finegym_t3_champ_stage3_vsa.yaml \
      --ckpt ckpt/finegym_t3_champ_stage2_fg3_champ2/vit_best_model \
      --out /data3/xiaodan8/FineGym/cache_fg3_champ2 --verify 8
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
# FG3_RESERVE_GB: same early GPU-memory claim as train_shard.py (shared GPUs)
if float(os.environ.get('FG3_RESERVE_GB', '0') or 0) > 0:
    _blk = torch.empty(int(float(os.environ['FG3_RESERVE_GB']) * (1 << 30)),
                       dtype=torch.uint8, device='cuda')
    del _blk
    print(f"[reserve] {os.environ['FG3_RESERVE_GB']} GB held in the CUDA caching allocator", flush=True)
from torch.utils.data import DataLoader, Dataset

from libs.core import load_config
from libs.datasets import make_dataset
from libs.datasets.data_utils import get_transforms
from libs.modeling import make_meta_arch


class _Frames(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return i, self.base.load_window_frames(i)     # C x T x H x W float32


def _collate(b):
    return [x[0] for x in b], torch.stack([x[1] for x in b])


def _hs(enc, x):
    """CLIP hidden states mean-pooled over patches for x (n, 3, H, W) bf16."""
    return enc.clip_image_encoder(pixel_values=x).last_hidden_state.mean(dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True, help='DeepSpeed ckpt dir or .pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--batch', type=int, default=8, help='windows per loader batch')
    ap.add_argument('--verify', type=int, default=8,
                    help='windows per split re-run through the frame path')
    ap.add_argument('--splits', default='val,train')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N windows per split')
    args = ap.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    # same held-out injection as train_shard.py (window list bookkeeping)
    with open(cfg['model']['aux_attribute_table_path']) as f:
        cfg['dataset']['held_out_class_ids'] = [
            int(a[1:]) for a in json.load(f).get('held_out_classes', [])]
    cfg['model']['active_learning_method'] = cfg['active_learning_method']
    max_len = int(cfg['dataset']['max_seq_len'])
    win = int(cfg['dataset']['window_length'])
    chunk = int(cfg['model'].get('clip_forward_chunk', 0) or 0)
    assert chunk == 64 and win == 32 and max_len == 288, \
        "the exact-chunk reconstruction below assumes chunk 64 / window 32 / max_seq_len 288"

    # ---- model: only the encoder is used ----
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    src = args.ckpt
    if os.path.isdir(src):
        src = os.path.join(src, 'mp_rank_00_model_states.pt')
    sd = torch.load(src, map_location='cpu')
    sd = sd.get('module', sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {src}: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    assert all(k.startswith('video_subtree_attn.') for k in missing), missing
    enc = model.backbone.encoder.to(dev).to(torch.bfloat16).eval()
    del model
    D = enc.projection.out_features if hasattr(enc.projection, 'out_features') \
        else enc.clip_image_encoder.config.hidden_size

    # ---- constant padding chunks (zero images = padding_val 0.0) ----
    zeros64 = torch.zeros(64, 3, 224, 224, dtype=torch.bfloat16, device=dev)
    hs_pad64 = _hs(enc, zeros64)                 # (64, 768)
    hs_pad32 = _hs(enc, zeros64[:32])            # (32, 768)
    # 288 = [64 (chunk0)] + 3 x 64 + 32
    hs_tail = torch.cat([hs_pad64, hs_pad64, hs_pad64, hs_pad32], dim=0)   # (224, 768)
    assert hs_tail.shape[0] == max_len - 64

    def encode_windows(frames):
        """frames: (b, 3, 32, H, W) float32 -> (b, 32, D) bf16 features and
        (b, 288 - 32, D) padding rows, both EXACTLY as the frame path."""
        b = frames.shape[0]
        x = frames.permute(0, 2, 1, 3, 4).contiguous().to(dev).to(torch.bfloat16)  # (b, 32, 3, H, W)
        feats, pads = [], []
        for i in range(b):
            c0 = torch.cat([x[i], zeros64[:32]], dim=0)          # (64, 3, H, W)
            hs0 = _hs(enc, c0)                                   # (64, 768)
            hs = torch.cat([hs0, hs_tail], dim=0).view(1, max_len, -1)
            out = enc.projection(hs)[0]                          # (288, D) bf16
            feats.append(out[:win])
            pads.append(out[win:])
        return torch.stack(feats), torch.stack(pads)

    pad_ref = None
    meta = {'ckpt': src, 'config': args.config, 'transform': 'eval (resize 256, centre crop 224)',
            'layout': 'uint16 bit pattern of bf16, (N_windows, window_length, D)', 'D': int(D),
            'splits': {}}
    for split in args.splits.split(','):
        is_train = split == 'train'
        ds = make_dataset(cfg['dataset_name'], is_train,
                          cfg['train_split'] if is_train else cfg['val_split'],
                          cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
        ds.transform = get_transforms(False, 224)     # deterministic view
        n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
        man = [(w['id'], int(w['window_start_frame'])) for w in ds.windows[:n]]
        with open(os.path.join(args.out, f'{split}_manifest.json'), 'w') as f:
            json.dump(man, f)
        arr = np.lib.format.open_memmap(
            os.path.join(args.out, f'{split}_feats.npy'), mode='w+',
            dtype=np.uint16, shape=(n, win, D))
        sub = _Frames(ds)
        if args.limit > 0:
            sub = torch.utils.data.Subset(sub, list(range(n)))
        loader = DataLoader(sub, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=_collate,
                            pin_memory=True)
        t0 = time.time()
        pad_dev = 0.0
        for it, (idx, frames) in enumerate(loader):
            feats, pads = encode_windows(frames)
            if pad_ref is None:
                pad_ref = pads[0, 0].clone()
            pad_dev = max(pad_dev, (pads.float() - pad_ref.float()).abs().max().item())
            arr[idx[0]:idx[-1] + 1] = feats.view(torch.int16).cpu().numpy().view(np.uint16)
            if it % 50 == 0:
                el = time.time() - t0
                print(f"[{split}] {idx[-1] + 1}/{n}  {el / 60:.1f} min  "
                      f"eta {el / (idx[-1] + 1) * (n - idx[-1] - 1) / 60:.1f} min", flush=True)
        arr.flush()
        del arr
        el = time.time() - t0
        print(f"[{split}] done: {n} windows in {el / 60:.1f} min; "
              f"max |pad row - pad_ref| = {pad_dev:.3e}", flush=True)
        meta['splits'][split] = {'n': n, 'minutes': el / 60, 'pad_row_max_dev': pad_dev}

        # ---- verification against the untouched frame path ----
        if args.verify > 0:
            arr = np.load(os.path.join(args.out, f'{split}_feats.npy'), mmap_mode='r')
            picks = np.linspace(0, n - 1, args.verify).astype(int).tolist()
            worst = 0.0
            for i in picks:
                fr = ds.load_window_frames(i)                     # (3, 32, H, W)
                # exactly meta_archs.preprocessing (eval): pad to 288 with 0.0,
                # -> (1, 288, 3, H, W) bf16, then backbone.encoder.forward
                fr = fr.permute(0, 2, 3, 1)                        # (3, H, W, 32)
                fr = torch.nn.functional.pad(fr, [0, max_len - win], value=0.0).unsqueeze(0)
                fr = fr.permute(0, 4, 1, 2, 3).to(dev).to(torch.bfloat16)
                ref = enc(fr)[0]                                   # (D, 288)
                got = torch.from_numpy(np.array(arr[i]).view(np.int16)).view(torch.bfloat16).to(dev)
                d_real = (ref[:, :win].t().float() - got.float()).abs().max().item()
                d_pad = (ref[:, win:].float() - pad_ref.float().view(-1, 1)).abs().max().item()
                worst = max(worst, d_real, d_pad)
                print(f"[verify {split}] window {i}: real rows max|diff|={d_real:.3e} "
                      f"pad rows max|diff|={d_pad:.3e}", flush=True)
            meta['splits'][split]['verify_max_abs_diff'] = worst
            print(f"[verify {split}] worst = {worst:.3e} ({'BITWISE EQUAL' if worst == 0 else 'NOT bitwise'})",
                  flush=True)
            del arr
        del loader, ds

    torch.save(pad_ref.cpu(), os.path.join(args.out, 'pad_feat.pt'))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print("[done]", json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
