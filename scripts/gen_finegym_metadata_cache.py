#!/usr/bin/env python3
"""Reconstruct the FineGym raw-video metadata caches WITHOUT the raw videos.

This machine has the RGB JPG frame store (/data3/xiaodan8/FineGym/RGB) but
video_raw/ is empty, so finegym_slide's get_video_metadata() would skip every
video on a fresh build. This script reconstructs
    video_metadata_cache_{train,val}_raw.json   ({youtube_id: {fps,
    total_frames, duration}})
from two verified sources:
  * fps: the legacy precomputed-window database
    finegym_merged_win32_int16.json carries the true per-video fps
    (validated EXACTLY against the three raw videos still present in
    FineGym/archive: mI7pLIDnTiQ/zNL3kn3UBmg 29.97003, 5X85zLeLmks 30.0).
  * total_frames: max cached JPG frame index + 16 (frames were cached by
    real runs over all windows, so this covers every window the dataset
    will request); duration = total_frames / fps.

Usage: python scripts/gen_finegym_metadata_cache.py
"""
import json
import os
from collections import Counter, defaultdict

ROOT = '/data3/xiaodan8/FineGym'
WIN_JSON = os.path.join(ROOT, 'finegym_merged_win32_int16.json')
RGB = os.path.join(ROOT, 'RGB')
LABELS = [os.path.join(ROOT, 'annotation/Dec16/gym99_train_label.txt'),
          os.path.join(ROOT, 'annotation/Dec16/gym99_val_label.txt')]


def main():
    # per-video fps from the legacy windows database
    with open(WIN_JSON) as f:
        db = json.load(f)['database']
    fps_votes = defaultdict(Counter)
    for k, v in db.items():
        yt = k.rsplit('_', 2)[0]
        fps_votes[yt][float(v['fps'])] += 1
    fps_of = {yt: c.most_common(1)[0][0] for yt, c in fps_votes.items()}

    # youtube ids required by the Dec16 labels
    need = set()
    for lp in LABELS:
        with open(lp) as f:
            for line in f:
                if line.strip():
                    need.add(json.loads(line)['video'].split('_E_')[0].rstrip('_'))

    cache = {}
    missing_fps, missing_rgb = [], []
    for yt in sorted(need):
        if yt not in fps_of:
            missing_fps.append(yt)
            continue
        d = os.path.join(RGB, yt)
        if not os.path.isdir(d):
            missing_rgb.append(yt)
            continue
        max_idx = max(int(fn[:-4]) for fn in os.listdir(d)
                      if fn.endswith('.jpg'))
        fps = fps_of[yt]
        total_frames = max_idx + 16
        cache[yt] = {'fps': fps, 'total_frames': total_frames,
                     'duration': total_frames / fps}

    for suffix in ('train', 'val'):
        out = os.path.join(ROOT, f'video_metadata_cache_{suffix}_raw.json')
        with open(out, 'w') as f:
            json.dump(cache, f)
        print(f"wrote {out}: {len(cache)} videos")
    if missing_fps:
        print(f"WARNING: no fps source for {len(missing_fps)}: {missing_fps}")
    if missing_rgb:
        print(f"WARNING: no RGB dir for {len(missing_rgb)}: {missing_rgb}")


if __name__ == '__main__':
    main()
