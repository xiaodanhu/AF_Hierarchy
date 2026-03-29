#!/usr/bin/env python3
"""
Generate annotation files for THUMOS14 in the same JSONL format used by
FineGym and FineDiving dataloaders.

THUMOS14 is a single-level action dataset (no hierarchy), so the format
is simpler: no activity/phrase grouping, just flat action spans.

Input:
  - thumos14.json: standard THUMOS14 annotation with video metadata + action segments

Output format (one JSON per line):
{
  "video": "<video_name>",
  "span": ["<boundary_start seconds>", "<boundary_end seconds>"],
  "raw_value": [{"action_id": "cNN", "span": ["<start seconds>", "<end seconds>"]}],
  "new_value": [{"action_id": "cNN", "span": ["<start seconds>", "<end seconds>"]}]
}

raw_value  = absolute timestamps (within the full untrimmed video)
new_value  = relative timestamps (with respect to clip boundary start)

For THUMOS14, each *video* becomes one entry (no splitting into clips).
The span is the full video duration (no random padding needed since THUMOS
videos are already untrimmed and densely annotated).
"""

import json
import os
import random


def main():
    random.seed(42)

    annotation_file = '/data3/xiaodan8/thumos/annotations/thumos14.json'
    output_dir = '/data3/xiaodan8/thumos/annotations'

    print("Loading THUMOS14 annotations...")
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    database = data['database']
    print(f"Loaded {len(database)} videos")

    # Collect all labels for summary
    all_labels = set()
    for video_data in database.values():
        for ann in video_data.get('annotations', []):
            all_labels.add((ann['label'], ann['label_id']))
    print(f"Found {len(all_labels)} action classes")

    # Separate by subset: Validation = train, Test = test (standard THUMOS14 protocol)
    subsets = {'Validation': [], 'Test': []}
    for video_name, video_data in database.items():
        subset = video_data['subset']
        if subset in subsets:
            subsets[subset].append((video_name, video_data))

    print(f"Validation (train): {len(subsets['Validation'])} videos")
    print(f"Test (test): {len(subsets['Test'])} videos")

    # Generate annotations for each split
    split_mapping = {
        'Validation': 'train',  # THUMOS14 convention: validation set used for training
        'Test': 'test',
    }

    for subset_name, split_name in split_mapping.items():
        output_file = os.path.join(output_dir, f'thumos14_{split_name}_label.txt')
        annotations = []
        skipped = 0

        for video_name, video_data in sorted(subsets[subset_name]):
            duration = video_data['duration']
            fps = video_data['fps']
            anns = video_data.get('annotations', [])

            if not anns:
                skipped += 1
                continue

            # Boundary: full video duration (THUMOS videos are untrimmed)
            # Add small padding to avoid boundary effects
            boundary_start = 0.0
            boundary_end = round(duration, 1)

            # Build raw_value: flat list of actions (no activity grouping)
            raw_value = []
            for ann in anns:
                label_id = ann['label_id']
                seg_start = round(ann['segment'][0], 1)
                seg_end = round(ann['segment'][1], 1)

                # Clamp to video bounds
                seg_start = max(0.0, seg_start)
                seg_end = min(boundary_end, seg_end)

                if seg_end <= seg_start:
                    continue

                raw_value.append({
                    'action_id': f'c{label_id:02d}',
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

                new_start = round(raw_start - boundary_start, 1)
                new_end = round(raw_end - boundary_start, 1)

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

    print(f"\nAction classes: {len(all_labels)}")
    for label, label_id in sorted(all_labels, key=lambda x: x[1]):
        print(f"  c{label_id:02d}: {label}")
    print("\nDone!")


if __name__ == '__main__':
    main()
