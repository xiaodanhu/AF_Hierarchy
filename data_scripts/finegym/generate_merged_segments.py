#!/usr/bin/env python3
"""
Generate merged video segments for train and val sets.
Each merged segment should be less than 45 seconds and contain multiple activities (>=1).
Merged segments cannot overlap with segments from the other set.
"""

import re
from collections import defaultdict
from typing import List, Tuple, Dict


def parse_segment_name(segment_name: str) -> Tuple[str, int, int, str]:
    """
    Parse segment name to extract video, start, end, and action.
    Format: "video_E_000510_000574_A_0016_0017"
    Returns: (video_name, start_frame, end_frame, action_part)
    """
    # Match Pattern: "video_E_start_end_A_action"
    pattern = r'^(.+)_E_(\d+)_(\d+)_A_(.+)$'
    match = re.match(pattern, segment_name)
    if match:
        video_name = match.group(1)
        start_frame = int(match.group(2))
        end_frame = int(match.group(3))
        action_part = match.group(4)
        return video_name, start_frame, end_frame, action_part
    else:
        raise ValueError(f"Cannot parse segment name: {segment_name}")


def load_segments(filepath: str) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Load segments from file and group by video name.
    Returns dict mapping video_name -> list of (start, end, label) tuples
    """
    video_segments = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            segment_name = parts[0]
            label = parts[1] if len(parts) > 1 else ""

            try:
                video_name, start, end, action = parse_segment_name(segment_name)
                video_segments[video_name].append((start, end, label))
            except ValueError as e:
                print(f"Warning: {e}")
                continue

    # Sort segments by start time for each video
    for video_name in video_segments:
        video_segments[video_name].sort(key=lambda x: x[0])

    return video_segments


def get_activity_segments(segments: List[Tuple[int, int, str]]) -> List[Tuple[int, int]]:
    """
    Extract unique activity segments (E_start_end) from action segments.
    Returns: sorted list of (start, end) tuples for activities
    """
    activities = set()
    for start, end, _ in segments:
        activities.add((start, end))
    return sorted(list(activities))


def check_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two time ranges overlap."""
    return not (end1 < start2 or end2 < start1)


def merge_segments(
        train_segments: Dict[str, List[Tuple[int, int, str]]],
        val_segments: Dict[str, List[Tuple[int, int, str]]],
        max_duration_seconds: int = 45  # 45 seconds
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
    """
    Merge segments for train and val sets.
    Returns: (merged_train, merged_val) where each is a dict mapping video_name -> list of (start, end)
    """
    merged_train = {}
    merged_val = {}

    # Process train set
    for video_name, segments in train_segments.items():
        activities = get_activity_segments(segments)

        # Get val activities for this video (if any) to check for conflicts
        val_activities = []
        if video_name in val_segments:
            val_activities = get_activity_segments(val_segments[video_name])

        merged = merge_video_activities(activities, val_activities, max_duration_seconds)
        if merged:
            merged_train[video_name] = merged

    # Process val set
    for video_name, segments in val_segments.items():
        activities = get_activity_segments(segments)

        # Get train activities for this video (if any) to check for conflicts
        train_activities = []
        if video_name in train_segments:
            train_activities = get_activity_segments(train_segments[video_name])

        merged = merge_video_activities(activities, train_activities, max_duration_seconds)
        if merged:
            merged_val[video_name] = merged

    return merged_train, merged_val


def merge_video_activities(
        activities: List[Tuple[int, int]],
        other_set_activities: List[Tuple[int, int]],
        max_duration_seconds: int
) -> List[Tuple[int, int]]:
    """
    Merge activities for a single video, ensuring no overlap with other_set_activities.
    """
    if not activities:
        return []

    merged = []
    current_start = activities[0][0]
    current_end = activities[0][1]

    for i in range(1, len(activities)):
        next_start, next_end = activities[i]

        # Calculate duration if we merge
        potential_end = next_end
        duration = potential_end - current_start

        # Check if merging would exceed max duration
        if duration > max_duration_seconds:
            # Save current merged segment
            merged.append((current_start, current_end))
            # Start new segment
            current_start = next_start
            current_end = next_end
        else:
            # Check if the range [current_start, next_end] would overlap with other set
            # We need to check if there's any segment from other_set between current_end and next_start
            would_overlap = False
            for other_start, other_end in other_set_activities:
                # Skip if this segment is in our activities list (same segment in both sets)
                is_our_activity = (other_start, other_end) in activities
                if is_our_activity:
                    continue

                # Check if the merged range [current_start, next_end] would overlap with other set segment
                # that is NOT in our activities
                if check_overlap(current_start, next_end, other_start, other_end):
                    would_overlap = True
                    break

            if would_overlap:
                merged.append((current_start, current_end))
                current_start = next_start
                current_end = next_end
            else:
                # Can merge, extend current segment
                current_end = next_end

    # Don't forget the last segment
    merged.append((current_start, current_end))

    return merged


def write_merged_segments(
        merged_segments: Dict[str, List[Tuple[int, int]]],
        output_filepath: str
):
    """Write merged segments to file."""
    with open(output_filepath, 'w') as f:
        for video_name in sorted(merged_segments.keys()):
            for start, end in merged_segments[video_name]:
                f.write(f"{video_name}_E_{start:06d}_{end:06d}\n")


def main():
    # Load original segments
    print("Loading train segments...")
    train_segments = load_segments('/data3/xiaodan8/FineGym/annotation/raw/gym99_train_element_new.txt')
    print(f"Loaded {len(train_segments)} videos from train set")

    print("Loading val segments...")
    val_segments = load_segments('/data3/xiaodan8/FineGym/annotation/raw/gym99_val_element_new.txt')
    print(f"Loaded {len(val_segments)} videos from val set")

    # Merge segments
    print("\nMerging segments...")
    merged_train, merged_val = merge_segments(train_segments, val_segments)

    # Write output
    print("\nWriting merged train segments...")
    write_merged_segments(merged_train, '/data3/xiaodan8/FineGym/annotation/Dec16/gym99_train_element.txt')

    print("Writing merged val segments...")
    write_merged_segments(merged_val, '/data3/xiaodan8/FineGym/annotation/Dec16/gym99_val_element.txt')

    # Print statistics
    total_train_merged = sum(len(segs) for segs in merged_train.values())
    total_val_merged = sum(len(segs) for segs in merged_val.values())

    print(f"\n=== Statistics ===")
    print(f"Train: {len(merged_train)} videos, {total_train_merged} merged segments")
    print(f"Val: {len(merged_val)} videos, {total_val_merged} merged segments")

    # Show examples
    print("\n=== Example merged segments ===")
    print("Train (first 10):")
    count = 0
    for video_name in sorted(merged_train.keys()):
        for start, end in merged_train[video_name]:
            print(f"  {video_name}_E_{start:06d}_{end:06d}")
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break

    print("\nVal (first 10):")
    count = 0
    for video_name in sorted(merged_val.keys()):
        for start, end in merged_val[video_name]:
            print(f"  {video_name}_E_{start:06d}_{end:06d}")
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break


if __name__ == '__main__':
    main()
