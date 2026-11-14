"""Check that the PNG store yields bit-identical model inputs to the original JPEG store.

Builds the val and train datasets twice (original RGB root vs FINEGYM_RGB_ROOT/.png) and compares
__getitem__ outputs for a sample of windows using the deterministic eval transform (train transform
has random crop/flip, so both datasets are built with is_training=False for the comparison; the
train-split windows/frame resolution are still exercised). Also times the two loaders.
Usage: PYTHONPATH=<worktree> python scripts/verify_rgb256_store.py [--n 200]
"""
import os, sys, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from libs.core import load_config

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=200)
ap.add_argument('--store', default='/home/jupyter/xhu3/video/dataset/finegym/RGB256png'); a = ap.parse_args()
cfg = load_config('configs/finegym_t3_tifad.yaml')

import libs.datasets.finegym_slide as fs

def build(split, store):
    # NOTE: never reload the module here -- a reload re-executes it in the SAME namespace, so the
    # already-built reference dataset would silently start looking for .png files too.
    d = fs.FineGymSlideDataset(False, split, cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
    if store:
        d.rgb_root = store
    return d

def get(d, i, ext):
    fs._FRAME_EXT = ext
    try:
        return d[i]
    finally:
        fs._FRAME_EXT = '.jpg'

random.seed(0); worst = 0.0; n_cmp = 0; t_ref = t_new = 0.0
for split in (cfg['val_split'], cfg['train_split']):
    d_ref = build(split, None); d_new = build(split, a.store)
    assert len(d_ref) == len(d_new)
    idxs = random.sample(range(len(d_ref)), a.n // 2)
    for i in idxs:
        t0 = time.time(); r = get(d_ref, i, '.jpg'); t_ref += time.time() - t0
        t0 = time.time(); n = get(d_new, i, '.png'); t_new += time.time() - t0
        assert r['feats'].shape == n['feats'].shape, (r['feats'].shape, n['feats'].shape)
        worst = max(worst, (r['feats'] - n['feats']).abs().max().item()); n_cmp += 1
    print(f"split {split}: {len(idxs)} windows compared, running max |diff| = {worst:.3e}", flush=True)
print(f"windows compared: {n_cmp}; max abs diff of model input tensors: {worst:.3e}")
print(f"mean __getitem__ time: original {t_ref/n_cmp:.2f}s  png-store {t_new/n_cmp:.2f}s")
assert worst == 0.0, "PNG store is not bit-identical"
print("RGB256 STORE CHECKS PASSED")
