#!/usr/bin/env python
"""Pre-build the FineDiving JPEG frame store.

The dive loader caches decoded frames as JPEG under $FD_ROOT/RGB and falls back
to the .mp4 when one is missing. Filling that cache during training is very slow
because every frame becomes its own seek into a large H.264 file: measured here
at about 11 s per 64-frame window, which is roughly nine hours for one pass over
the 3033 windows, and it trips the loader's five-minute batch timeout.

Decoding each video once, sequentially, and writing only the frames the windows
actually ask for is far cheaper, because sequential decode costs nothing per
frame beyond the decode itself. Videos are processed in parallel, one worker per
video.

    python scripts/prebuild_finediving_jpg_store.py [--workers 12] [--config ...]

Safe to re-run: frames already present are skipped, so an interrupted build
simply resumes.
"""
import argparse, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def needed_frames(cfg_path):
    """video_name -> {video_path, total_frames, set of raw frame indices}"""
    from libs.core import load_config
    from libs.datasets import make_dataset
    cfg = load_config(cfg_path)
    want = defaultdict(lambda: {'path': None, 'total': 0, 'idx': set()})
    for is_train, split in ((True, cfg['train_split']), (False, cfg['val_split'])):
        ds = make_dataset(cfg['dataset_name'], is_train, split,
                          cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
        W, S = ds.window_length, ds.sample_stride
        for w in ds.windows:
            name = w.get('video_name')
            if name is None:
                continue
            e = want[name]
            e['path'], e['total'] = w['video_path'], w['total_frames']
            for i in range(W):
                fi = min(max(0, w['window_start_frame'] + i * S), max(0, w['total_frames'] - 1))
                e['idx'].add(fi)
    return {k: dict(v) for k, v in want.items()}, ds.data_root


def build_one(args):
    """Decode one video sequentially and write the frames that are missing."""
    name, path, total, idx, rgb_root = args
    import av
    from libs.datasets.finegym_slide import get_frame_path
    targets = {}
    for fi in idx:
        p = get_frame_path(rgb_root, name, fi, total)
        if not os.path.exists(p):
            targets[fi] = p
    if not targets:
        return name, 0, 0.0
    os.makedirs(os.path.dirname(next(iter(targets.values()))), exist_ok=True)
    t0, written, last = time.time(), 0, max(targets)
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    for n, frame in enumerate(container.decode(stream)):
        if n in targets:
            frame.to_image().save(targets[n], quality=95)
            written += 1
        if n >= last:
            break
    container.close()
    return name, written, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/finediving/v3/fd_champ_s1_a4000.yaml')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()

    want, data_root = needed_frames(a.config)
    rgb_root = os.path.join(data_root, 'RGB')
    os.makedirs(rgb_root, exist_ok=True)
    total_frames = sum(len(v['idx']) for v in want.values())
    print(f'{len(want)} videos, {total_frames} distinct frames needed', flush=True)

    jobs = [(n, v['path'], v['total'], v['idx'], rgb_root) for n, v in want.items()]
    jobs.sort(key=lambda j: -len(j[3]))
    done = written = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, j): j[0] for j in jobs}
        for f in as_completed(futs):
            name, w, dt = f.result()
            done += 1; written += w
            print(f'[{done}/{len(jobs)}] {name}: wrote {w} frames in {dt:.0f}s '
                  f'(total {written}, elapsed {(time.time()-t0)/60:.1f} min)', flush=True)
    print(f'done: {written} frames written in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
