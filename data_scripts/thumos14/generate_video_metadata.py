#!/usr/bin/env python3
"""
Generate video metadata cache (fps, total_frames, duration) for THUMOS14.
Produces per-split caches (train/test) keyed by video name.

THUMOS14 convention:
  - "Validation" subset -> train split
  - "Test" subset -> test split
"""

import json
import os
import av


VIDEO_DIR = '/data3/xiaodan8/thumos/video'
ANNOTATION_FILE = '/data3/xiaodan8/thumos/annotations/thumos14.json'
OUTPUT_DIR = '/data3/xiaodan8/thumos'


def get_video_metadata(video_path):
    """Get fps, total_frames, duration from a video file."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
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


def main():
    # Load annotations to get subset assignments
    with open(ANNOTATION_FILE, 'r') as f:
        data = json.load(f)
    database = data['database']

    # Build subset mapping
    subset_videos = {'Validation': set(), 'Test': set()}
    for video_name, video_data in database.items():
        subset = video_data['subset']
        if subset in subset_videos:
            subset_videos[subset].add(video_name)

    # Find all video files
    video_files = {}
    for fname in sorted(os.listdir(VIDEO_DIR)):
        if fname.endswith('.mp4'):
            name = fname[:-4]
            video_files[name] = os.path.join(VIDEO_DIR, fname)

    print(f"Found {len(video_files)} video files")
    print(f"Annotation: {len(subset_videos['Validation'])} Validation, "
          f"{len(subset_videos['Test'])} Test")

    # Load all metadata
    all_metadata = {}
    for i, (name, path) in enumerate(video_files.items()):
        try:
            meta = get_video_metadata(path)
            all_metadata[name] = meta
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(video_files)}: {name} "
                      f"(fps={meta['fps']}, frames={meta['total_frames']})")
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

    # Save combined cache
    combined_path = os.path.join(OUTPUT_DIR, 'video_metadata_cache_all.json')
    with open(combined_path, 'w') as f:
        json.dump(all_metadata, f)
    print(f"\nSaved {combined_path} ({len(all_metadata)} videos)")

    # Save per-split caches (THUMOS convention: Validation=train, Test=test)
    for subset_name, split_name in [('Validation', 'train'), ('Test', 'test')]:
        split_videos = subset_videos[subset_name]
        split_metadata = {v: all_metadata[v] for v in split_videos if v in all_metadata}
        out_file = os.path.join(OUTPUT_DIR, f'video_metadata_cache_{split_name}.json')
        with open(out_file, 'w') as f:
            json.dump(split_metadata, f)
        print(f"Saved {out_file} ({len(split_metadata)} videos)")

        missing = split_videos - set(all_metadata.keys())
        if missing:
            print(f"  WARNING: {len(missing)} videos missing: {missing}")

    # Summary
    fps_values = set(m['fps'] for m in all_metadata.values())
    print(f"\nFPS values: {sorted(fps_values)}")
    total_duration = sum(m['duration'] for m in all_metadata.values())
    print(f"Total duration: {total_duration / 3600:.1f} hours")


if __name__ == '__main__':
    main()
