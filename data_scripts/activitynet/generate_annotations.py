#!/usr/bin/env python3
"""
Generate annotation files for ActivityNet v1.3 in the JSONL format used by
the sliding-window dataloaders (same format as THUMOS14).

ActivityNet is a single-level action dataset (200 classes, no hierarchy).

Input:
  - activity_net.v1-3.min.json: official ActivityNet v1.3 annotation
  - label_dict.json: label name -> class_id mapping (0-199)

Output format (one JSON per line):
{
  "video": "<video_id>",
  "span": ["<boundary_start seconds>", "<boundary_end seconds>"],
  "raw_value": [{"action_id": "cNNN", "span": ["<start seconds>", "<end seconds>"]}],
  "new_value": [{"action_id": "cNNN", "span": ["<start seconds>", "<end seconds>"]}]
}

raw_value  = absolute timestamps (within the full untrimmed video)
new_value  = relative timestamps (same as raw_value since span starts at 0)

ActivityNet convention:
  - "training" subset -> train split
  - "validation" subset -> test split (has annotations; "testing" has none)
"""

import json
import os


def main():
    data_root = '/data3/xiaodan8/Activitynet'
    annotation_file = os.path.join(data_root, 'activity_net.v1-3.min.json')
    label_dict_file = os.path.join(data_root, 'label_dict.json')
    output_dir = os.path.join(data_root, 'annotation')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading ActivityNet v1.3 annotations...")
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    print("Loading label dictionary...")
    with open(label_dict_file, 'r') as f:
        label_dict = json.load(f)  # {"label_name": class_id, ...}

    database = data['database']
    print(f"Loaded {len(database)} videos, {len(label_dict)} action classes")

    # Invert label_dict for lookup: class_id -> label_name (for summary)
    id_to_label = {v: k for k, v in label_dict.items()}

    # Separate by subset
    subsets = {'training': [], 'validation': []}
    unknown_labels = set()
    for video_name, video_data in database.items():
        subset = video_data['subset']
        if subset in subsets:
            subsets[subset].append((video_name, video_data))

    print(f"training (train): {len(subsets['training'])} videos")
    print(f"validation (test): {len(subsets['validation'])} videos")

    # Generate annotations for each split
    split_mapping = {
        'training': 'train',
        'validation': 'test',
    }

    for subset_name, split_name in split_mapping.items():
        output_file = os.path.join(output_dir, f'activitynet_{split_name}_label.txt')
        annotations = []
        skipped = 0
        label_missing = 0

        for video_name, video_data in sorted(subsets[subset_name]):
            duration = video_data['duration']
            anns = video_data.get('annotations', [])

            if not anns:
                skipped += 1
                continue

            boundary_start = 0.0
            boundary_end = round(duration, 3)

            # Build raw_value: flat list of actions
            raw_value = []
            for ann in anns:
                label_name = ann['label']
                if label_name not in label_dict:
                    unknown_labels.add(label_name)
                    label_missing += 1
                    continue

                label_id = label_dict[label_name]
                seg_start = round(ann['segment'][0], 3)
                seg_end = round(ann['segment'][1], 3)

                # Clamp to video bounds
                seg_start = max(0.0, seg_start)
                seg_end = min(boundary_end, seg_end)

                if seg_end <= seg_start:
                    continue

                raw_value.append({
                    'action_id': f'c{label_id:03d}',
                    'span': [f'<{seg_start} seconds>', f'<{seg_end} seconds>']
                })

            if not raw_value:
                skipped += 1
                continue

            # new_value: relative to boundary_start (which is 0, so same as raw_value)
            new_value = []
            for action in raw_value:
                raw_start = float(action['span'][0].strip('<>').replace(' seconds', ''))
                raw_end = float(action['span'][1].strip('<>').replace(' seconds', ''))

                new_start = round(raw_start - boundary_start, 3)
                new_end = round(raw_end - boundary_start, 3)

                new_value.append({
                    'action_id': action['action_id'],
                    'span': [f'<{new_start} seconds>', f'<{new_end} seconds>']
                })

            annotation = {
                'video': video_name,
                'span': [f'<{boundary_start} seconds>', f'<{boundary_end} seconds>'],
                'raw_value': raw_value,
                'new_value': new_value,
            }
            annotations.append(annotation)

        # Write JSONL
        with open(output_file, 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')

        print(f"\n=== {split_name} ({subset_name}) ===")
        print(f"  Generated: {len(annotations)} annotations -> {output_file}")
        if skipped:
            print(f"  Skipped: {skipped} videos (no valid annotations)")
        if label_missing:
            print(f"  Label misses: {label_missing} annotations with unknown labels")

        # Stats
        if annotations:
            action_counts = [len(ann['raw_value']) for ann in annotations]
            durations = [
                float(ann['span'][1].strip('<>').replace(' seconds', ''))
                for ann in annotations
            ]
            print(f"  Actions per video: min={min(action_counts)}, "
                  f"max={max(action_counts)}, avg={sum(action_counts)/len(action_counts):.1f}")
            print(f"  Duration: min={min(durations):.1f}s, "
                  f"max={max(durations):.1f}s, avg={sum(durations)/len(durations):.1f}s")

    if unknown_labels:
        print(f"\nWARNING: {len(unknown_labels)} unknown labels not in label_dict.json:")
        for label in sorted(unknown_labels):
            print(f"  {label}")

    print(f"\nAction classes: {len(label_dict)}")
    for label_name, label_id in sorted(label_dict.items(), key=lambda x: x[1]):
        print(f"  c{label_id:03d}: {label_name}")
    print("\nDone!")


if __name__ == '__main__':
    main()
