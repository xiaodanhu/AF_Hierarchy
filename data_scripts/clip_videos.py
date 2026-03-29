#!/usr/bin/env python3
"""
Script to clip videos based on segment annotations.
Reads annotation files and creates clipped videos using ffmpeg.
"""

import json
import os
import subprocess
import re
from pathlib import Path

# Paths
VIDEO_RAW_DIR = Path("/taiga/illinois/eng/ece/n-ahuja/xiaodan/dataset/FineGym/video")
OUTPUT_DIR = Path("/taiga/illinois/eng/ece/n-ahuja/xiaodan/dataset/FineGym/finegym_qwen/videos_Dec16")
TRAIN_ANNOTATION = Path("/taiga/illinois/eng/ece/n-ahuja/xiaodan/dataset/FineGym/finegym_qwen/annotation/Dec16/gym99_train_label.txt")
VAL_ANNOTATION = Path("/taiga/illinois/eng/ece/n-ahuja/xiaodan/dataset/FineGym/finegym_qwen/annotation/Dec16/gym99_val_label.txt")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_time_in_seconds(time_str):
    """Extract time from '<X.XX seconds>' format to float."""
    match = re.search(r'<([\d.]+)\s+seconds>', time_str)
    if match:
        return float(match.group(1))
    return None

def find_video_file(video_id):
    """Find the video file in video_raw_1 directory by ID."""
    # Check for various video extensions
    for ext in ['.mp4', '.mkv', '.webm', '.avi']:
        video_path = VIDEO_RAW_DIR / f"{video_id}{ext}"
        if video_path.exists():
            return video_path
    return None

def clip_video(input_path, output_path, start_time, end_time):
    """
    Clip a video using ffmpeg with accurate seeking.

    Args:
        input_path: Path to input video
        output_path: Path to output clipped video
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    # ffmpeg command: Use -ss and -to after -i for frame-accurate seeking
    # This is slower but ensures precise timing
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output file if exists
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error clipping video: {e}")
        print(f"stderr: {e.stderr}")
        return False

def process_annotation_file(annotation_file):
    """Process a single annotation file and clip videos."""
    print(f"\nProcessing {annotation_file.name}...")

    stats = {
        'total': 0,
        'clipped': 0,
        'skipped_no_video': 0,
        'failed': 0
    }

    with open(annotation_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                stats['total'] += 1

                # Extract video name and segment information
                segment_name = data['video']  # e.g., "0LtLS9wROrk_E_000147_000152"
                span = data['span']

                # Extract video ID (part before first underscore sequence)
                # Format: {video_id}_E_{start}_{end}
                parts = segment_name.split('_E_')
                if len(parts) != 2:
                    print(f"Warning: Unexpected segment name format: {segment_name}")
                    continue

                video_id = parts[0]  # e.g., "0LtLS9wROrk"

                # Find the source video file
                video_path = find_video_file("gym_" + video_id)
                if video_path is None:
                    stats['skipped_no_video'] += 1
                    continue

                # Extract start and end times
                # Handle both old format (dict) and new format (list)
                if isinstance(span, dict):
                    start_time = extract_time_in_seconds(span['start'])
                    end_time = extract_time_in_seconds(span['end'])
                elif isinstance(span, list):
                    start_time = extract_time_in_seconds(span[0])
                    end_time = extract_time_in_seconds(span[1])
                else:
                    print(f"Warning: Unexpected span format: {span}")
                    stats['failed'] += 1
                    continue

                if start_time is None or end_time is None:
                    print(f"Warning: Could not extract times from {span}")
                    stats['failed'] += 1
                    continue

                # Output file path
                output_path = OUTPUT_DIR / f"{segment_name}.mp4"

                # Skip if already exists
                if output_path.exists():
                    # print(f"Skipping {segment_name}.mp4 (already exists)")
                    stats['clipped'] += 1
                    continue

                # Clip the video
                print(f"Clipping {segment_name}.mp4 from {video_id} ({start_time:.2f}s - {end_time:.2f}s)")
                if clip_video(video_path, output_path, start_time, end_time):
                    stats['clipped'] += 1
                else:
                    stats['failed'] += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                stats['failed'] += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                stats['failed'] += 1

    return stats

def main():
    """Main function to process all annotation files."""
    print("Starting video clipping process...")
    print(f"Source videos: {VIDEO_RAW_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        return

    # Process training annotations
    if TRAIN_ANNOTATION.exists():
        train_stats = process_annotation_file(TRAIN_ANNOTATION)
        print(f"\nTraining set statistics:")
        print(f"  Total segments: {train_stats['total']}")
        print(f"  Clipped: {train_stats['clipped']}")
        print(f"  Skipped (no source video): {train_stats['skipped_no_video']}")
        print(f"  Failed: {train_stats['failed']}")
    else:
        print(f"Warning: Training annotation file not found: {TRAIN_ANNOTATION}")

    # Process validation annotations
    if VAL_ANNOTATION.exists():
        val_stats = process_annotation_file(VAL_ANNOTATION)
        print(f"\nValidation set statistics:")
        print(f"  Total segments: {val_stats['total']}")
        print(f"  Clipped: {val_stats['clipped']}")
        print(f"  Skipped (no source video): {val_stats['skipped_no_video']}")
        print(f"  Failed: {val_stats['failed']}")
    else:
        print(f"Warning: Validation annotation file not found: {VAL_ANNOTATION}")

    print("\nClipping complete!")

if __name__ == "__main__":
    main()
