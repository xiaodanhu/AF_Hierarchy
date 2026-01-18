import os
import sys
import json
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image
import warnings
from contextlib import contextmanager
import signal
import gc

# Suppress FFmpeg/libav filter warnings (pix_fmt errors)
os.environ["FFREPORT"] = "file=/dev/null:level=quiet"
os.environ["LIBAV_LOG_LEVEL"] = "quiet"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"
os.environ["AV_LOG_FORCE_QUIET"] = "1"
warnings.filterwarnings("ignore", message=".*mmco.*")
warnings.filterwarnings("ignore", message=".*h264.*")
warnings.filterwarnings("ignore", message=".*pix_fmt.*")
warnings.filterwarnings("ignore", message=".*buffer.*")

import av
av.logging.set_level(av.logging.PANIC)

# Try to use decord (faster), fallback to av
try:
    from decord import VideoReader, cpu
    import decord
    decord.bridge.set_bridge('native')  # Use native bridge for stability
    # Suppress decord's FFmpeg warnings
    decord.logging.set_level(decord.logging.ERROR)
    USE_DECORD = True
except ImportError:
    USE_DECORD = False


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (FFmpeg warnings)."""
    # Save original stderr
    old_stderr = sys.stderr
    # Redirect stderr to devnull
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


class VideoLoadTimeout(Exception):
    """Exception raised when video loading times out."""
    pass


def _timeout_handler(signum, frame):
    raise VideoLoadTimeout("Video loading timed out")


@contextmanager
def timeout_context(seconds):
    """Context manager that raises VideoLoadTimeout after specified seconds.
    Note: Only works in main thread on Unix systems.
    """
    # Check if we can use signals (main thread only)
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except ValueError:
        # Not in main thread, skip timeout (workers)
        yield


from .datasets import register_dataset
from .data_utils import truncate_feats, get_transforms, truncate_video


def load_frames_by_locations_decord(video_path, locations):
    """
    Load specific frames from video using decord.

    Args:
        video_path: Path to video file
        locations: List of frame indices to load (absolute frame numbers)

    Returns:
        frames (list of PIL), video_fps
    """
    with suppress_stderr():
        # Use num_threads=1 to avoid decord's multi-threaded decoder bugs
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())

        # Clamp frame indices to valid range
        valid_locations = [min(max(0, loc), total_frames - 1) for loc in locations]

        # Batch decode
        frames_array = vr.get_batch(valid_locations).asnumpy()
        frames = [Image.fromarray(frames_array[i]) for i in range(len(frames_array))]

        del vr
    return frames, video_fps


def load_frames_by_locations_av(video_path, locations):
    """
    Load specific frames from video using PyAV.

    Args:
        video_path: Path to video file
        locations: List of frame indices to load (absolute frame numbers)

    Returns:
        frames (list of PIL), video_fps
    """
    with suppress_stderr():
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        video_fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        total_frames = video_stream.frames

        if total_frames == 0:
            duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0
            total_frames = int(duration * video_fps) if duration > 0 else 0

        # Clamp locations
        valid_locations = [min(max(0, loc), max(0, total_frames - 1)) for loc in locations]
        location_set = set[int](valid_locations)

        # Max frame to decode for safety
        max_frame_to_decode = max(location_set) + 100 if location_set else 0

        frames_dict = {}
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in location_set:
                frames_dict[frame_idx] = frame.to_image()
            if len(frames_dict) == len(location_set):
                break
            if frame_idx > max_frame_to_decode:
                print(f"Warning: PyAV safety limit reached at frame {frame_idx}")
                break

        container.close()
        del container

    # Return frames in order
    frames = [frames_dict.get(loc, frames_dict[min(frames_dict.keys())]) for loc in valid_locations]
    return frames, video_fps


def load_frames_by_locations(video_path, locations, max_retries=2):
    """Load frames by locations using decord, with PyAV fallback."""
    last_error = None

    # Try decord first (faster)
    if USE_DECORD:
        for attempt in range(max_retries):
            try:
                frames, fps = load_frames_by_locations_decord(video_path, locations)
                gc.collect()
                return frames, fps
            except Exception as e:
                last_error = e
                gc.collect()

    # Check if it's a pixel format error - PyAV can handle these
    error_str = str(last_error) if last_error else ""
    is_pixel_format_error = "Cannot create buffer source" in error_str or "filter_graph" in error_str

    if is_pixel_format_error:
        try:
            frames, fps = load_frames_by_locations_av(video_path, locations)
            gc.collect()
            return frames, fps
        except Exception as e:
            print(f"Warning: PyAV fallback also failed for {video_path}: {e}")

    # If all else fails, return dummy frames
    print(f"Warning: Failed to load video {video_path}: {last_error}")
    dummy_frame = Image.new('RGB', (224, 224), (0, 0, 0))
    return [dummy_frame] * len(locations), 30.0


def find_video_file(video_root, video_id):
    """Find video file with any extension (.mkv, .mp4, etc.)"""
    for ext in ['.mkv', '.mp4', '.avi', '.webm']:
        video_path = os.path.join(video_root, video_id + ext)
        if os.path.exists(video_path):
            return video_path
    # Return .mkv as default (most common in finegym)
    return os.path.join(video_root, video_id + '.mkv')


@register_dataset("finegym_original")
class FineGymOriginalDataset(Dataset):
    """
    FineGym dataset with PRE-COMPUTED sliding windows.

    This loads from a pre-processed JSON file (finegym_merged_win32_int16.json)
    that contains:
    - Pre-computed frame locations for each 32-frame window
    - Segment annotations with frame-level boundaries
    - Train/test split information

    This dataset was used to achieve ~50% mAP.
    """
    def __init__(
        self,
        is_training,    # if in training mode
        split,          # split, a tuple/list allowing concat of subsets
        backbone_type,
        round,
        use_full,       # use full dataset for training
        json_file,      # Path to finegym_merged_win32_int16.json
        max_seq_len,    # maximum sequence length during training
        trunc_thresh,
        crop_ratio,
        num_classes,    # number of action categories (99)
        num_frames=1,   # number of frames for each feat (for offset calculation)
        sample_stride=16,  # Stride used for sampling (16 in the original experiment)
        train_json_file=None,  # Ignored - for compatibility with finegym_slide config
        val_json_file=None,    # Ignored - for compatibility with finegym_slide config
        window_length=32,      # Ignored - for compatibility
        window_stride=16,      # Ignored - for compatibility
        **kwargs        # ignore extra kwargs for compatibility
    ):
        # file path
        assert isinstance(split, tuple) or isinstance(split, list)
        if crop_ratio is not None:
            assert crop_ratio == 'None' or len(crop_ratio) == 2

        self.json_file = json_file
        if self.json_file is None:
            raise ValueError("No json_file specified for finegym_original dataset. "
                           "Provide json_file path to finegym_merged_win32_int16.json")

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.num_frames = num_frames
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = None if (crop_ratio is None or crop_ratio == 'None') else crop_ratio
        self.backbone_type = backbone_type
        self.round = round

        # Video root for raw videos
        self.video_root = '/home/jupyter/xhu3/video/dataset/finegym/video_raw'

        # Fixed parameters for this dataset (matching original 50% mAP experiment)
        self.window_length = 32            # 32 sampled frames per window (fixed)
        self.sample_stride = sample_stride  # Original frames between samples (16)
        self.feat_stride = float(sample_stride)

        # load database
        self.clips = self._load_json_db(self.json_file)
        print(f"Loaded {len(self.clips)} clips for {'training' if is_training else 'validation'}")

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'finegym_original',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }
        self.transform = get_transforms(is_training, 224)

    def _load_json_db(self, json_file):
        """Load database from the merged JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        database = data['database']

        # Determine which subset to load
        subset_name = 'train' if self.is_training else 'test'

        clips = []
        for clip_id, clip_data in database.items():
            # Filter by subset
            if clip_data['subset'] != subset_name:
                continue

            # Extract video ID (first part before underscore)
            video_id = clip_id.split('_')[0]

            # Find video file path
            video_path = find_video_file(self.video_root, video_id)

            # Get frame locations
            locations = clip_data['locations']

            # Parse annotations
            annotations = clip_data['annotations']
            if not annotations:
                continue  # Skip clips without annotations

            segments = []
            labels = []
            for ann in annotations:
                # segment(frames) is relative to the 512-frame window
                # Convert to feature index by dividing by sample_stride (16)
                seg_frames = ann['segment(frames)']
                seg_start = seg_frames[0] / self.sample_stride
                seg_end = seg_frames[1] / self.sample_stride
                segments.append([seg_start, seg_end])
                labels.append(ann['label_id'])

            clips.append({
                'id': clip_id,
                'video_id': video_id,
                'video_path': video_path,
                'locations': locations,
                'segments': np.array(segments, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64),
                'fps': clip_data['fps'],
                'duration_frame': clip_data['duration_frame'],
            })

        return clips

    def get_attributes(self):
        return self.db_attributes

    def get_ground_truth_df(self):
        """
        Generate ground truth DataFrame for ANETdetection evaluator.
        Returns segments in seconds (converted from feature indices).
        """
        import pandas as pd
        vids, starts, stops, labels = [], [], [], []

        for clip in self.clips:
            for i in range(len(clip['segments'])):
                vids.append(clip['id'])
                # Convert feature indices back to seconds
                # segment_seconds = segment_features * feat_stride / fps
                seg_start = clip['segments'][i, 0] * self.feat_stride / clip['fps']
                seg_end = clip['segments'][i, 1] * self.feat_stride / clip['fps']
                starts.append(float(seg_start))
                stops.append(float(seg_end))
                labels.append(int(clip['labels'][i]))

        return pd.DataFrame({
            'video-id': vids,
            't-start': starts,
            't-end': stops,
            'label': labels
        })

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """
        Load a clip and return data dict.
        """
        clip = self.clips[idx]

        # Load frames using pre-computed locations
        try:
            frames, video_fps = load_frames_by_locations(
                clip['video_path'],
                clip['locations']
            )
        except Exception as e:
            print(f"[ERROR] Failed to load video {clip['video_path']}: {e}")
            # Return dummy frames on failure
            dummy_frame = Image.new('RGB', (224, 224), (0, 0, 0))
            frames = [dummy_frame] * self.window_length
            video_fps = clip['fps']

        # Handle case where we got fewer frames
        if len(frames) < self.window_length:
            while len(frames) < self.window_length:
                frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), (0, 0, 0)))

        # Apply transforms: list of PIL -> T x C x H x W tensor
        feats = torch.stack([self.transform(frm) for frm in frames])

        # Explicit cleanup to prevent memory leaks in workers
        del frames

        # T x C x H x W -> C x T x H x W
        feats = feats.permute(1, 0, 2, 3).contiguous()

        # feat_stride is fixed at sample_stride (16)
        feat_stride = self.feat_stride

        # feat_offset for center alignment
        feat_offset = 0.5 * self.num_frames / feat_stride

        # Segments are already in feature indices from _load_json_db
        # Apply offset for center alignment
        segments = torch.from_numpy(clip['segments'] - feat_offset)
        labels = torch.from_numpy(clip['labels'])

        # Duration in seconds for this window
        # 32 samples × 16 stride / fps
        duration = self.window_length * self.sample_stride / video_fps

        # Return data dict
        data_dict = {
            'video_id': clip['id'],
            'feats': feats,              # C x T x H x W (T = 32)
            'segments': segments,        # N x 2 (in feature indices)
            'labels': labels,            # N
            'fps': video_fps,
            'duration': duration,
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames
        }

        # For training, truncate/augment if needed
        if self.is_training and (segments is not None):
            # Only truncate if sequence is longer than max_seq_len
            if feats.shape[1] > self.max_seq_len:
                data_dict = truncate_feats(
                    data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
                )

        # Handle empty segments (shouldn't happen since we filter clips)
        if data_dict['segments'] is None or len(data_dict['segments']) == 0:
            # Skip to next item
            return self.__getitem__((idx + 1) % len(self))

        return data_dict
