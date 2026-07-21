#!/usr/bin/env python3
"""
Generate video metadata cache (fps, total_frames, duration) for ActivityNet v1.3.
Produces per-split caches (train/test) keyed by video name.

Primary source: frame_count_raw_video_v1.json (pre-computed from raw videos).
Fallback: open video with PyAV to extract metadata.

ActivityNet convention:
  - "training" subset -> train split
  - "validation" subset -> test split
"""

import json
import os


DATA_ROOT = '/data3/xiaodan8/Activitynet'
VIDEO_DIR = os.path.join(DATA_ROOT, 'video')
ANNOTATION_FILE = os.path.join(DATA_ROOT, 'activity_net.v1-3.min.json')
FRAME_COUNT_FILE = os.path.join(DATA_ROOT, 'frame_count_raw_video_v1.json')
OUTPUT_DIR = DATA_ROOT


def get_video_metadata_av(video_path):
    """Get fps, total_frames, duration from a video file using PyAV."""
    import av
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


def find_video_file(video_dir, video_id):
    """Find video file with v_ prefix and any extension."""
    for ext in ['.mp4', '.mkv', '.avi', '.webm']:
        path = os.path.join(video_dir, f'v_{video_id}{ext}')
        if os.path.exists(path):
            return path
    return None


def main():
    # Load annotations to get subset assignments
    print("Loading annotations...")
    with open(ANNOTATION_FILE, 'r') as f:
        data = json.load(f)
    database = data['database']

    # Build subset mapping
    subset_videos = {'training': set(), 'validation': set()}
    for video_name, video_data in database.items():
        subset = video_data['subset']
        if subset in subset_videos:
            subset_videos[subset].add(video_name)

    print(f"Annotation: {len(subset_videos['training'])} training, "
          f"{len(subset_videos['validation'])} validation")

    # Load pre-computed frame counts (primary metadata source)
    frame_counts = {}
    if os.path.exists(FRAME_COUNT_FILE):
        print(f"Loading pre-computed frame counts from {FRAME_COUNT_FILE}...")
        with open(FRAME_COUNT_FILE, 'r') as f:
            frame_counts = json.load(f)
        print(f"  Loaded metadata for {len(frame_counts)} videos")

    # Build metadata for all annotated videos
    all_metadata = {}
    missing_metadata = []
    fallback_count = 0

    all_videos = subset_videos['training'] | subset_videos['validation']
    print(f"\nProcessing {len(all_videos)} annotated videos...")

    for i, video_id in enumerate(sorted(all_videos)):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(all_videos)}...")

        # Try pre-computed frame counts first
        if video_id in frame_counts:
            fc = frame_counts[video_id]
            all_metadata[video_id] = {
                'fps': fc['video_fps'],
                'total_frames': fc['total_frames'],
                'duration': fc['video_seconds'],
            }
            continue

        # Fallback: use duration from annotation + default fps
        if video_id in database:
            duration = database[video_id].get('duration', 0)
            if duration > 0:
                # Try to get real metadata from video file
                video_path = find_video_file(VIDEO_DIR, video_id)
                if video_path and os.path.exists(video_path):
                    try:
                        meta = get_video_metadata_av(video_path)
                        all_metadata[video_id] = meta
                        fallback_count += 1
                        continue
                    except Exception as e:
                        pass

                # Last resort: estimate from annotation duration
                est_fps = 30.0
                all_metadata[video_id] = {
                    'fps': est_fps,
                    'total_frames': int(duration * est_fps),
                    'duration': duration,
                }
                fallback_count += 1
                continue

        missing_metadata.append(video_id)

    print(f"\nMetadata summary:")
    print(f"  From frame_count file: {len(all_metadata) - fallback_count}")
    print(f"  From video/annotation fallback: {fallback_count}")
    print(f"  Missing: {len(missing_metadata)}")

    # Save per-split caches
    for subset_name, split_name in [('training', 'train'), ('validation', 'test')]:
        split_videos = subset_videos[subset_name]
        split_metadata = {v: all_metadata[v] for v in split_videos if v in all_metadata}
        out_file = os.path.join(OUTPUT_DIR, f'video_metadata_cache_{split_name}.json')
        with open(out_file, 'w') as f:
            json.dump(split_metadata, f)
        print(f"\nSaved {out_file} ({len(split_metadata)} videos)")

        missing = split_videos - set(all_metadata.keys())
        if missing:
            print(f"  WARNING: {len(missing)} videos missing metadata")

    # Summary stats
    if all_metadata:
        fps_values = sorted(set(round(m['fps'], 1) for m in all_metadata.values()))
        total_duration = sum(m['duration'] for m in all_metadata.values())
        print(f"\nFPS values: {fps_values[:20]}{'...' if len(fps_values) > 20 else ''}")
        print(f"Total duration: {total_duration / 3600:.1f} hours")

    if missing_metadata:
        print(f"\nMissing metadata for {len(missing_metadata)} videos:")
        for vid in missing_metadata[:10]:
            print(f"  {vid}")
        if len(missing_metadata) > 10:
            print(f"  ... and {len(missing_metadata) - 10} more")


if __name__ == '__main__':
    main()
