#!/usr/bin/env python3
"""
Generate video metadata cache (fps, total_frames, duration) for FineDiving.
Similar to FineGym's video_metadata_cache_train_raw.json.

Produces one cache per split (train/test) keyed by video name,
plus a combined cache with all videos.
"""

import json
import os
import csv
import av


VIDEO_DIR = "raw/FineDiving/Released_FineDiving_Dataset/Untrimmed_Videos"
SPLIT_DIR = "raw/FineDiving/Released_FineDiving_Dataset/train_test_split"


def get_video_metadata(video_path):
    """Get fps, total_frames, duration from a video file."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    total_frames = stream.frames

    if total_frames == 0:
        duration = 0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000.0
        total_frames = int(duration * fps) if duration > 0 else 0

    duration = total_frames / fps if fps > 0 else 0
    container.close()
    return {"fps": fps, "total_frames": total_frames, "duration": duration}


def load_split_videos(split_file):
    """Get unique video names from a split file."""
    videos = set()
    with open(split_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.add(row['video'].strip())
    return videos


def main():
    # Discover all video files
    video_files = {}
    for fname in sorted(os.listdir(VIDEO_DIR)):
        if fname.endswith('.mp4'):
            name = fname[:-4]   # strip .mp4
            video_files[name] = os.path.join(VIDEO_DIR, fname)

    print(f"Found {len(video_files)} video files")

    # Load all metadata
    all_metadata = {}
    for i, (name, path) in enumerate(video_files.items()):
        try:
            meta = get_video_metadata(path)
            all_metadata[name] = meta
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(video_files)}: {name} "
                      f"(fps={meta['fps']}, frames={meta['total_frames']})")
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

    # Save combined cache
    with open("video_metadata_cache_all.json", 'w') as f:
        json.dump(all_metadata, f)
    print(f"\nSaved video_metadata_cache_all.json ({len(all_metadata)} videos)")

    # Load splits and create per-split caches
    for split_name, split_file in [
        ("train", f"{SPLIT_DIR}/train_split.csv"),
        ("test", f"{SPLIT_DIR}/test_split.csv"),
    ]:
        split_videos = load_split_videos(split_file)
        split_metadata = {v: all_metadata[v] for v in split_videos if v in all_metadata}
        out_file = f"video_metadata_cache_{split_name}.json"
        with open(out_file, 'w') as f:
            json.dump(split_metadata, f)
        print(f"Saved {out_file} ({len(split_metadata)} videos)")

        # Report any missing videos
        missing = split_videos - set(all_metadata.keys())
        if missing:
            print(f"  WARNING: {len(missing)} videos in {split_name} split not found: {missing}")

    # Print summary stats
    fps_values = set(m['fps'] for m in all_metadata.values())
    print(f"\nFPS values found: {sorted(fps_values)}")
    total_duration = sum(m['duration'] for m in all_metadata.values())
    print(f"Total duration: {total_duration / 3600:.1f} hours")


if __name__ == '__main__':
    main()
