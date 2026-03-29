#!/usr/bin/env python3
"""
Generate detailed annotation files in JSON format from merged segments.
Each merged segment will be converted to a detailed JSON annotation with:
- Video boundary (with random padding)
- Raw values (absolute timestamps)
- New values (relative timestamps)
"""

import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def load_fine_to_coarse_mappings(filepath: str) -> Dict[int, int]:
    """Load action-to-phrase mapping."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}


def load_finegym_annotation(filepath: str) -> Dict:
    """Load the main annotation file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_action_labels(train_file: str, val_file: str) -> Dict[str, int]:
    """
    Load action labels from train and val files.
    Returns mapping: "video_E_start_end_A_action" -> label
    """
    labels = {}

    for filepath in [train_file, val_file]:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    segment_name = parts[0]
                    label = int(parts[1])
                    labels[segment_name] = label

    return labels


def parse_merged_segment(segment_name: str) -> Tuple[str, int, int]:
    """
    Parse merged segment name.
    Example: rrrgsw--AE8_E_000510_000856
    Returns: (video_name, start_frame, end_frame)
    """
    pattern = r'^(.+)_E_(\d+)_(\d+)$'
    match = re.match(pattern, segment_name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    raise ValueError(f"Cannot parse segment: {segment_name}")


def get_video_boundary(start_frame: int, end_frame: int,
                      all_segments: List[Tuple[int, int]],
                      min_padding: float = 2.0,
                      max_padding: float = 5.0) -> Tuple[float, float]:
    """
    Get video boundary with random padding before start and after end.
    Ensure it doesn't overlap with other segments.
    """
    # Random padding
    start_padding = random.uniform(min_padding, max_padding)
    end_padding = random.uniform(min_padding, max_padding)

    boundary_start = start_frame - start_padding
    boundary_end = end_frame + end_padding

    # Check for conflicts with other segments
    for seg_start, seg_end in all_segments:
        if seg_start == start_frame and seg_end == end_frame:
            continue  # Skip self

        # Adjust if too close to other segments
        if boundary_start < seg_start < start_frame:
            boundary_start = max(boundary_start, seg_start + 1.0)
        if end_frame < seg_end < boundary_end:
            boundary_end = min(boundary_end, seg_end - 1.0)

    # Ensure boundaries are valid
    boundary_start = max(0, boundary_start)
    boundary_end = max(boundary_end, end_frame + 1.0)

    return round(boundary_start, 1), round(boundary_end, 1)


def generate_annotation_for_segment(
    segment_name: str,
    finegym_data: Dict,
    action_labels: Dict[str, int],
    action_to_phrase: Dict[int, int],
    all_video_segments: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """Generate detailed annotation for a single merged segment."""

    video_name, start_frame, end_frame = parse_merged_segment(segment_name)

    # Get video boundary
    boundary_start, boundary_end = get_video_boundary(
        start_frame, end_frame, all_video_segments
    )

    # Get all activities within this merged segment
    if video_name not in finegym_data:
        print(f"Warning: Video {video_name} not found in annotation data")
        return None

    video_data = finegym_data[video_name]

    # Find all activities (E_xxx_yyy) that fall within [start_frame, end_frame]
    raw_values = []

    for activity_key, activity_data in video_data.items():
        if not activity_key.startswith('E_'):
            continue

        # Parse activity frames
        pattern = r'E_(\d+)_(\d+)'
        match = re.match(pattern, activity_key)
        if not match:
            continue

        act_start = int(match.group(1))
        act_end = int(match.group(2))

        # Check if this activity is within our merged segment
        if act_start >= start_frame and act_end <= end_frame:
            # Get activity event (class)
            event = activity_data.get('event', 0)

            # Get activity timestamps
            timestamps = activity_data.get('timestamps', [[act_start, act_end]])
            if timestamps:
                act_timestamp_start = timestamps[0][0]
                act_timestamp_end = timestamps[0][1]
            else:
                act_timestamp_start = float(act_start)
                act_timestamp_end = float(act_end)

            # Get all actions within this activity
            actions = []
            segments = activity_data.get('segments', {})
            if segments is None:
                segments = {}

            for action_key, action_data in segments.items():

                # Build full action name
                full_action_name = f"{video_name}_{activity_key}_{action_key}"

                # Get action label
                action_label = action_labels.get(full_action_name)
                if action_label is None:
                    continue

                # Get phrase from action label
                phrase = action_to_phrase.get(action_label, 0)

                # Get action timestamps
                action_timestamps = action_data.get('timestamps', [])
                if not action_timestamps:
                    continue

                for ts in action_timestamps:
                    action_start = round(act_timestamp_start + ts[0], 1)
                    action_end = round(act_timestamp_start + ts[1], 1)

                    actions.append({
                        'phrase_id': f'p{phrase:02d}',
                        'action_id': f'c{action_label:02d}',
                        'span': [f'<{action_start} seconds>', f'<{action_end} seconds>']
                    })

            # Add activity to raw_values only if it has actions
            if actions:
                raw_values.append({
                    'activity_id': f'a{event-1}',
                    'span': [f'<{round(act_timestamp_start, 1)} seconds>', f'<{round(act_timestamp_end, 1)} seconds>'],
                    'actions': actions
                })

    # Generate new_values (relative timestamps)
    new_values = []
    for raw_activity in raw_values:

        # Parse raw timestamps
        raw_start_str = raw_activity['span'][0]
        raw_end_str = raw_activity['span'][1]

        raw_start = float(raw_start_str.replace('<', '').replace(' seconds>', ''))
        raw_end = float(raw_end_str.replace('<', '').replace(' seconds>', ''))

        # Calculate relative timestamps
        new_start = round(raw_start - boundary_start, 1)
        new_end = round(raw_end - boundary_start, 1)

        # Process actions
        new_actions = []
        for action in raw_activity['actions']:
            action_raw_start_str = action['span'][0]
            action_raw_end_str = action['span'][1]

            action_raw_start = float(action_raw_start_str.replace('<', '').replace(' seconds>', ''))
            action_raw_end = float(action_raw_end_str.replace('<', '').replace(' seconds>', ''))

            action_new_start = round(action_raw_start - boundary_start, 1)
            action_new_end = round(action_raw_end - boundary_start, 1)

            new_actions.append({
                'phrase_id': action['phrase_id'],
                'action_id': action['action_id'],
                'span': [f'<{action_new_start} seconds>', f'<{action_new_end} seconds>']
            })

        new_values.append({
            'activity_id': raw_activity['activity_id'],
            'span': [f'<{new_start} seconds>', f'<{new_end} seconds>'],
            'actions': new_actions
        })

    # Build final annotation
    annotation = {
        'video': segment_name,
        'span': [f'<{boundary_start} seconds>', f'<{boundary_end} seconds>'],
        'raw_value': raw_values,
        'new_value': new_values
    }

    return annotation


def process_merged_segments(
    merged_file: str,
    output_file: str,
    finegym_data: Dict,
    action_labels: Dict[str, int],
    action_to_phrase: Dict[int, int]
):
    """Process all merged segments and generate annotations."""

    # Load merged segments
    segments = []
    with open(merged_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(line)

    print(f"Processing {len(segments)} segments from {merged_file}...")

    # Group segments by video
    video_segments = defaultdict(list)
    for segment in segments:
        video_name, start, end = parse_merged_segment(segment)
        video_segments[video_name].append((start, end))

    # Generate annotations
    annotations = []
    activity_counts = []  # Track number of activities per segment

    for segment in segments:
        video_name, _, _ = parse_merged_segment(segment)
        all_segs = video_segments[video_name]

        annotation = generate_annotation_for_segment(
            segment,
            finegym_data,
            action_labels,
            action_to_phrase,
            all_segs
        )

        if annotation:
            annotations.append(annotation)
            # Count activities in this segment
            num_activities = len(annotation.get('raw_value', []))
            activity_counts.append(num_activities)

    # Write to file
    with open(output_file, 'w') as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + '\n')

    print(f"Generated {len(annotations)} annotations in {output_file}")

    # Print statistics
    if activity_counts:
        from collections import Counter
        activity_distribution = Counter(activity_counts)

        print("\nActivity Statistics:")
        print(f"  Total segments: {len(activity_counts)}")
        print(f"  Min activities per segment: {min(activity_counts)}")
        print(f"  Max activities per segment: {max(activity_counts)}")
        print(f"  Average activities per segment: {sum(activity_counts) / len(activity_counts):.2f}")

        print(f"\n  Distribution:")
        for num_activities in sorted(activity_distribution.keys()):
            count = activity_distribution[num_activities]
            percentage = (count / len(activity_counts)) * 100
            print(f"    {num_activities} activities: {count} segments ({percentage:.1f}%)")


def main():
    # Set random seed for reproducibility
    random.seed(42)

    print("Loading data files...")

    # Load fine-to-coarse mappings
    action_to_phrase = load_fine_to_coarse_mappings('raw/fine_to_coarse_mappings.json')

    # Load FineGym annotation
    finegym_data = load_finegym_annotation('raw/finegym_annotation_info_v1.1.json')

    # Load action labels
    action_labels = load_action_labels(
        'raw/gym99_train_element_new.txt',
        'raw/gym99_val_element_new.txt'
    )

    print(f"Loaded {len(action_to_phrase)} action-to-phrase mappings")
    print(f"Loaded {len(finegym_data)} videos from FineGym")
    print(f"Loaded {len(action_labels)} action labels")

    # Process train set
    print("\n=== Processing Train Set ===")
    process_merged_segments(
        'Dec16/gym99_train_element.txt',
        'Dec16/gym99_train_label.txt',
        finegym_data,
        action_labels,
        action_to_phrase
    )

    # Process val set
    print("\n=== Processing Val Set ===")
    process_merged_segments(
        'Dec16/gym99_val_element.txt',
        'Dec16/gym99_val_label.txt',
        finegym_data,
        action_labels,
        action_to_phrase
    )

    print("\n=== Done! ===")
    print("Generated files:")
    print(" - Dec16/gym99_train_label.txt")
    print(" - Dec16/gym99_val_label.txt")


if __name__ == '__main__':
    main()
