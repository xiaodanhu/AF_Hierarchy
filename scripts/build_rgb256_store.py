"""Build a LOSSLESS 256-short-side PNG frame store for the FineGym sliding-window loader.

Why: the loader decodes 1280x720 JPEGs (~170-340 KB, ~40-240 ms each incl. Resize(256)) for every
frame; with 1280 frames per rank per step that is the data bottleneck. get_transforms() starts
with Resize(256, antialias=True); storing that resized image losslessly (PNG) makes the pipeline's
Resize a verified exact no-op (torchvision returns the image unchanged when the short side already
equals the target), so the tensors fed to the model are bit-identical to the original pipeline.

Only frames the train/val windows actually resolve to are converted (index clamp + nearest-frame
snapping mirrored from load_sliding_window_jpg). Files keep their names under
<out_root>/<youtube_id>/<frame>.png. Use with:
    FINEGYM_RGB_ROOT=<out_root> FINEGYM_FRAME_EXT=.png
Usage: PYTHONPATH=<worktree> nice python scripts/build_rgb256_store.py [--workers 32] [--out ...]
"""
import os, sys, json, time, bisect, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torchvision as tv
from libs.core import load_config
from libs.datasets import make_dataset
from libs.datasets.finegym_slide import get_frame_path, _get_cached_frame_indices

RESIZE = tv.transforms.Resize(256, antialias=True)   # identical to libs/datasets/data_utils.get_transforms

def resolve_paths(rgb_root, w, window_length, stride):
    yid, total = w['youtube_id'], w['total_frames']
    idxs = [min(max(0, w['window_start_frame'] + i * stride), max(0, total - 1)) for i in range(window_length)]
    paths = [get_frame_path(rgb_root, yid, i, total) for i in idxs]
    missing = [i for i, p in enumerate(paths) if not os.path.exists(p)]
    if missing:
        cached = _get_cached_frame_indices(rgb_root, yid)
        if cached:
            tol = max(2, stride // 2)
            for i in missing:
                fi = idxs[i]; j = bisect.bisect_left(cached, fi); best = None
                for k in (j - 1, j):
                    if 0 <= k < len(cached) and (best is None or abs(cached[k] - fi) < abs(best - fi)):
                        best = cached[k]
                if best is not None and abs(best - fi) <= tol:
                    paths[i] = get_frame_path(rgb_root, yid, best, total)
    return [p for p in paths if os.path.exists(p)]

def convert(job):
    src, dst = job
    if os.path.exists(dst):
        return 0, os.path.getsize(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = RESIZE(Image.open(src).convert('RGB'))
    tmp = dst + '.tmp'
    img.save(tmp, format='PNG', compress_level=1)
    os.replace(tmp, dst)
    return 1, os.path.getsize(dst)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/finegym_t3_tifad.yaml')
    ap.add_argument('--out', default='/home/jupyter/xhu3/video/dataset/finegym/RGB256png')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    cfg = load_config(a.config)
    # held-out ids as train_shard injects them (needed for held_out_mode=cutout, whose sub-clip windows differ)
    if cfg['model'].get('aux_attribute_table_path'):
        _t = json.load(open(cfg['model']['aux_attribute_table_path']))
        cfg['dataset']['held_out_class_ids'] = [int(x[1:]) for x in _t.get('held_out_classes', [])]
    srcs = set(); per_split = {}
    for is_train, split in ((True, cfg['train_split']), (False, cfg['val_split'])):
        ds = make_dataset(cfg['dataset_name'], is_train, split, cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
        n0 = len(srcs)
        for w in ds.windows:
            srcs.update(resolve_paths(ds.rgb_root, w, ds.window_length, ds.sample_stride))
        per_split[str(split)] = {'windows': len(ds.windows), 'new_unique_frames': len(srcs) - n0}
        rgb_root = ds.rgb_root
    print(f"[build] {len(srcs)} unique source frames to convert; per split {per_split}", flush=True)
    jobs = sorted((s, os.path.join(a.out, os.path.relpath(s, rgb_root))[:-4] + '.png') for s in srcs)
    t0 = time.time(); done = 0; new = 0; nbytes = 0
    with Pool(a.workers) as pool:
        for k, (isnew, sz) in enumerate(pool.imap_unordered(convert, jobs, chunksize=64), 1):
            new += isnew; nbytes += sz
            if k % 20000 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"[build] {k}/{len(jobs)} ({new} new) {nbytes/1e9:.1f} GB, {el/60:.1f} min, eta {(len(jobs)-k)*el/k/60:.1f} min", flush=True)
    json.dump({'source_root': rgb_root, 'out_root': a.out, 'frames': len(jobs), 'bytes': nbytes,
               'per_split': per_split, 'png_compress_level': 1, 'resize': 'torchvision Resize(256, antialias=True)'},
              open(os.path.join(a.out, 'MANIFEST.json'), 'w'), indent=1)
    print("[build] DONE", flush=True)
