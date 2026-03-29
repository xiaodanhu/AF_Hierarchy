#!/usr/bin/env python3
"""
Generate FineGym-format annotation files from FineDiving dataset.

FineDiving has two levels:
  - action_type (e.g., "207c") → used directly as activity_id
  - sub_action (per-frame labels) → mapped to action_id (c00-c28) via Sub_action_Types_Table_filtered.csv

Requires video_metadata_cache_all.json (run generate_video_metadata.py first).

Output format (JSONL, one JSON per line):
{
  "video": "{video_name}_{clip_id}",
  "span": ["<boundary_start seconds>", "<boundary_end seconds>"],
  "raw_value": [{"activity_id": "207c", "span": [...], "actions": [...]}],
  "new_value": [{"activity_id": "207c", "span": [...], "actions": [...]}]
}
"""

import csv
import json
import random
import ast
from collections import defaultdict

ANNO_DIR = "/data3/xiaodan8/FineDiving/annotation/raw/Annotations"
SPLIT_DIR = "/data3/xiaodan8/FineDiving/annotation/raw/train_test_split"
METADATA_FILE = "/data3/xiaodan8/FineDiving/annotation/raw/video_metadata_cache_all.json"
SUB_ACTION_TABLE = f"{ANNO_DIR}/Sub_action_Types_Table_filtered.csv"


def load_sub_action_mapping(filepath):
    """Load original_label_id -> new_id mapping from filtered sub-action table."""
    mapping = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row['id'])] = int(row['new_id'])
    return mapping


def load_video_metadata(filepath):
    """Load video metadata cache: video_name -> {fps, total_frames, duration}"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_coarse_annotations(filepath):
    """Load coarse annotation: (video, clip_id) -> {action_type, start_frame, end_frame, ...}"""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['video'].strip(), row['clip_id'].strip())
            data[key] = {
                'action_type': row['action_type'].strip(),
                'start_frame': int(row['start_frame']),
                'end_frame': int(row['end_frame']),
            }
    return data


def load_fine_grained_annotations(filepath):
    """Load fine-grained annotation: (video, clip_id) -> {frames_labels, steps_transit_frames}"""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['video'].strip(), row['clip_id'].strip())
            frames_labels = ast.literal_eval(row['frames_labels'].strip())
            steps_transit = ast.literal_eval(row['steps_transit_frames'].strip())
            data[key] = {
                'frames_labels': frames_labels,
                'steps_transit_frames': steps_transit,
                'action_type': row['action_type'].strip()
            }
    return data


def load_split(filepath):
    """Load train/test split: list of (video, clip_id)"""
    keys = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.append((row['video'].strip(), row['clip_id'].strip()))
    return keys


def build_action_type_mapping(coarse_data):
    """Build action_type -> activity_id mapping (sorted, 0-indexed)."""
    action_types = sorted(set(v['action_type'] for v in coarse_data.values()))
    return {at: i for i, at in enumerate(action_types)}


def frames_to_action_spans(frames_labels, start_frame, fps):
    """
    Convert per-frame labels to action spans.

    Groups consecutive frames with the same label into spans.
    Returns list of (label, start_sec, end_sec) in raw video seconds.
    """
    if not frames_labels:
        return []

    spans = []
    current_label = frames_labels[0]
    span_start = 0

    for i in range(1, len(frames_labels)):
        if frames_labels[i] != current_label:
            raw_start_sec = (start_frame + span_start) / fps
            raw_end_sec = (start_frame + i) / fps
            spans.append((current_label, round(raw_start_sec, 1), round(raw_end_sec, 1)))
            current_label = frames_labels[i]
            span_start = i

    # Last run
    raw_start_sec = (start_frame + span_start) / fps
    raw_end_sec = (start_frame + len(frames_labels)) / fps
    spans.append((current_label, round(raw_start_sec, 1), round(raw_end_sec, 1)))

    return spans


def generate_annotation(key, coarse, fine_grained, fps, label_remap, add_padding=False):
    """Generate a single annotation entry using the video's real FPS."""
    video_name, clip_id = key
    video_id = f"{video_name}_{clip_id}"

    start_frame = coarse['start_frame']
    end_frame = coarse['end_frame']
    action_type = coarse['action_type']
    frames_labels = fine_grained['frames_labels']

    # Activity span in raw video seconds
    activity_start_sec = round(start_frame / fps, 1)
    activity_end_sec = round(end_frame / fps, 1)

    # Boundary: optionally add random padding (2-5 seconds each side)
    if add_padding:
        start_padding = random.uniform(2.0, 5.0)
        end_padding = random.uniform(2.0, 5.0)
    else:
        start_padding = 0.0
        end_padding = 0.0
    boundary_start = round(max(0, activity_start_sec - start_padding), 1)
    boundary_end = round(activity_end_sec + end_padding, 1)

    # Activity ID: use raw dive code directly
    activity_id = action_type

    # Convert frame labels to action spans using real FPS
    action_spans = frames_to_action_spans(frames_labels, start_frame, fps)

    # Build raw_value actions (sub-action label → c{new_id:02d} via filtered table)
    raw_actions = []
    for label, span_start, span_end in action_spans:
        action_id = f"c{label_remap[label]:02d}"
        raw_actions.append({
            'action_id': action_id,
            'span': [f"<{span_start} seconds>", f"<{span_end} seconds>"]
        })

    raw_value = [{
        'activity_id': activity_id,
        'span': [f"<{activity_start_sec} seconds>", f"<{activity_end_sec} seconds>"],
        'actions': raw_actions
    }]

    # Build new_value (relative to boundary_start)
    new_activity_start = round(activity_start_sec - boundary_start, 1)
    new_activity_end = round(activity_end_sec - boundary_start, 1)

    new_actions = []
    for label, span_start, span_end in action_spans:
        action_id = f"c{label_remap[label]:02d}"
        new_start = round(span_start - boundary_start, 1)
        new_end = round(span_end - boundary_start, 1)
        new_actions.append({
            'action_id': action_id,
            'span': [f"<{new_start} seconds>", f"<{new_end} seconds>"]
        })

    new_value = [{
        'activity_id': activity_id,
        'span': [f"<{new_activity_start} seconds>", f"<{new_activity_end} seconds>"],
        'actions': new_actions
    }]

    return {
        'video': video_id,
        'span': [f"<{boundary_start} seconds>", f"<{boundary_end} seconds>"],
        'raw_value': raw_value,
        'new_value': new_value,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--add-padding', action='store_true',
                        help='Add random 2-5s padding around activity span')
    args = parser.parse_args()

    random.seed(42)

    print(f"Padding: {'enabled (2-5s random)' if args.add_padding else 'disabled'}")
    print("Loading video metadata...")
    metadata = load_video_metadata(METADATA_FILE)
    print(f"  Loaded metadata for {len(metadata)} videos")
    fps_values = sorted(set(m['fps'] for m in metadata.values()))
    print(f"  FPS values: {fps_values}")

    print("Loading sub-action label mapping...")
    label_remap = load_sub_action_mapping(SUB_ACTION_TABLE)
    print(f"  {len(label_remap)} sub-action types (original IDs → contiguous c00-c{len(label_remap)-1:02d})")

    print("Loading annotations...")
    coarse = load_coarse_annotations(f"{ANNO_DIR}/FineDiving_coarse_annotation.csv")
    fine_grained = load_fine_grained_annotations(f"{ANNO_DIR}/FineDiving_fine-grained_annotation_corrected.csv")

    print(f"  Coarse entries: {len(coarse)}")
    print(f"  Fine-grained entries: {len(fine_grained)}")

    # Match by (video, clip_id)
    common_keys = set(coarse.keys()) & set(fine_grained.keys())
    print(f"  Matched entries: {len(common_keys)}")

    # Verify action_type consistency
    mismatches = 0
    for key in common_keys:
        if coarse[key]['action_type'] != fine_grained[key]['action_type']:
            mismatches += 1
    if mismatches:
        print(f"  WARNING: {mismatches} action_type mismatches between coarse and fine-grained")

    # Check metadata coverage
    video_names_in_annotations = set(k[0] for k in common_keys)
    missing_metadata = video_names_in_annotations - set(metadata.keys())
    if missing_metadata:
        print(f"  WARNING: {len(missing_metadata)} videos missing from metadata: {missing_metadata}")

    # List unique action types
    action_types = sorted(set(v['action_type'] for v in coarse.values()))
    print(f"\n{len(action_types)} action types (using raw dive codes as activity_id)")

    # Load splits
    train_keys = load_split(f"{SPLIT_DIR}/train_split.csv")
    test_keys = load_split(f"{SPLIT_DIR}/test_split.csv")
    print(f"\nSplit: {len(train_keys)} train, {len(test_keys)} test")

    # Generate annotations
    for split_name, split_keys in [("train", train_keys), ("test", test_keys)]:
        output_file = f"/data3/xiaodan8/FineDiving/annotation/finediving_{split_name}_label.txt"
        annotations = []
        skipped = 0
        no_metadata = 0

        for key in split_keys:
            if key not in common_keys:
                skipped += 1
                continue
            video_name = key[0]
            if video_name not in metadata:
                no_metadata += 1
                continue
            fps = metadata[video_name]['fps']
            ann = generate_annotation(key, coarse[key], fine_grained[key], fps, label_remap,
                                      add_padding=args.add_padding)
            annotations.append(ann)

        with open(output_file, 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')

        print(f"\n=== {split_name} ===")
        print(f"  Generated: {len(annotations)} annotations → {output_file}")
        if skipped:
            print(f"  Skipped: {skipped} (not in both coarse & fine-grained)")
        if no_metadata:
            print(f"  No metadata: {no_metadata}")

        # Stats
        if annotations:
            action_counts = [len(ann['raw_value'][0]['actions']) for ann in annotations]
            print(f"  Actions per clip: min={min(action_counts)}, max={max(action_counts)}, "
                  f"avg={sum(action_counts)/len(action_counts):.1f}")

    # Summary
    print(f"\nSub-action types: 29 of 42 (see Sub_action_Types_Table_filtered.csv)")
    print(f"Activity types: {len(action_types)} dive codes (raw codes as activity_id)")
    print("\nDone!")


if __name__ == '__main__':
    main()
