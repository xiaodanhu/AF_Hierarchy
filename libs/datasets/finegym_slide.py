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


def find_video_file(video_root, video_id):
    """Find video file with any extension (.mkv, .mp4, etc.)"""
    for ext in ['.mkv', '.mp4', '.avi', '.webm']:
        video_path = os.path.join(video_root, video_id + ext)
        if os.path.exists(video_path):
            return video_path
    # Return .mkv as default (most common in finegym raw videos)
    return os.path.join(video_root, video_id + '.mkv')


def extract_youtube_id(clip_name):
    """Extract YouTube video ID from clip name.

    Clip names are formatted as: {youtube_id}_E_{start_frame}_{end_frame}
    e.g., '0LtLS9wROrk_E_000217_000300' -> '0LtLS9wROrk'
    """
    if '_E_' in clip_name:
        return clip_name.split('_E_')[0]
    return clip_name


def get_video_metadata_decord(video_path):
    """Get video metadata (fps, total_frames, duration) using decord."""
    try:
        # For large files (>500MB), skip decord – it's slow for parsing
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        if file_size > 500 * 1024 * 1024:  # >500MB
            return None, None, None  # Skip to PyAV

        with suppress_stderr():
            # Use num_threads=1 to avoid decord's multi-threaded decoder bugs
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            if total_frames == 0:
                del vr
                return 30.0, 0, 0.0
            video_fps = float(vr.get_avg_fps())
            duration = total_frames / video_fps
            del vr
        return video_fps, total_frames, duration
    except Exception:
        return None, None, None


def get_video_metadata_av(video_path):
    """Get video metadata (fps, total_frames, duration) using PyAV."""
    try:
        with suppress_stderr():
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            video_fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
            total_frames = video_stream.frames

            if total_frames == 0:
                # Try to get duration from stream or container
                duration = 0
                if video_stream.duration and video_stream.time_base:
                    duration = float(video_stream.duration * video_stream.time_base)
                elif container.duration:
                    duration = container.duration / 1000000.0  # Convert from microseconds

                if duration > 0:
                    total_frames = int(duration * video_fps)
                else:
                    # Last resort: estimate from file size (very rough)
                    # DON'T count frames manually – way too slow for large videos!
                    print(f"Warning: Could not determine frame count for {video_path}")
                    total_frames = 0

            duration = total_frames / video_fps if video_fps > 0 else 0
            container.close()
        return video_fps, total_frames, duration
    except Exception:
        return 30.0, 0, 0.0


def get_video_metadata(video_path):
    """Get video metadata, trying decord first then PyAV as fallback."""
    import time
    start = time.time()

    if USE_DECORD:
        result = get_video_metadata_decord(video_path)
        if result[0] is not None:
            elapsed = time.time() - start
            if elapsed > 2:
                print(f"  [decord] {os.path.basename(video_path)}: {result[1]} frames, {elapsed:.1f}s")
            return result

    # Fallback to PyAV for metadata
    result = get_video_metadata_av(video_path)
    elapsed = time.time() - start
    if result[1] > 0:  # total_frames > 0
        if elapsed > 2:
            print(f"  [PyAV] {os.path.basename(video_path)}: {result[1]} frames, {elapsed:.1f}s")
        return result

    # Return default values if both fail
    print(f"Warning: Could not get metadata for {video_path}, using defaults")
    return 30.0, 0, 0.0


def load_sliding_window_decord(video_path, window_start_frame, window_length, stride):
    """
    Load a specific sliding window from video using decord.

    Args:
        video_path: Path to video file
        window_start_frame: Starting frame in original video coordinates
        window_length: Number of frames to sample (e.g., 32)
        stride: Stride in original frames (e.g., 16)

    Returns:
        frames (list of PIL), video_fps
    """
    with suppress_stderr():
        # Use num_threads=1 to avoid decord's multi-threaded decoder bugs
        # that cause random access failures with "Error sending packet"
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())

        # Calculate frame indices for this window
        # window_start_frame is in original video frames
        # We sample every `stride` frames, getting `window_length` frames total
        frame_indices = []
        for i in range(window_length):
            frame_idx = window_start_frame + i * stride
            # Clamp to valid range
            frame_idx = min(max(0, frame_idx), total_frames - 1)
            frame_indices.append(frame_idx)

        # Batch decode
        frames_array = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(frames_array[i]) for i in range(len(frames_array))]

        del vr
    return frames, video_fps


def load_sliding_window_av(video_path, window_start_frame, window_length, stride):
    """
    Load a specific sliding window from video using PyAV with seeking.

    Args:
        video_path: Path to video file
        window_start_frame: Starting frame in original video coordinates
        window_length: Number of frames to sample (e.g., 32)
        stride: Stride in original frames (e.g., 16)

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
            duration = 0
            if video_stream.duration and video_stream.time_base:
                duration = float(video_stream.duration * video_stream.time_base)
            elif container.duration:
                duration = container.duration / 1000000.0
            total_frames = int(duration * video_fps) if duration > 0 else 0

        # Calculate frame indices for this window
        frame_idx_list = []
        for i in range(window_length):
            frame_idx = window_start_frame + i * stride
            frame_idx = min(max(0, frame_idx), max(0, total_frames - 1))
            frame_idx_list.append(frame_idx)

        min_frame = min(frame_idx_list)
        max_frame = max(frame_idx_list)
        frame_indices = set(frame_idx_list)

        # Seek to near the first needed frame (seek is keyframe-based, so we might be earlier)
        if min_frame > 0 and video_fps > 0:
            seek_time = max(0, (min_frame - 100) / video_fps)  # Seek a bit before
            container.seek(int(seek_time * 1000000), backward=True, any_frame=False)

        # Decode frames starting from seek position
        frames_dict = {}
        frame_idx = max(0, min_frame - 200)  # Estimate starting frame after seek

        for frame in container.decode(video=0):
            # PyAV doesn't give frame index directly after seek, so we track it
            # Use pts to calculate frame index
            if frame.pts is not None and video_stream.time_base:
                frame_time = float(frame.pts * video_stream.time_base)
                frame_idx = int(frame_time * video_fps)

            if frame_idx in frame_indices and frame_idx not in frames_dict:
                frames_dict[frame_idx] = frame.to_image()

            if len(frames_dict) == len(frame_indices):
                break

            # Safety: stop if we've gone way past
            if frame_idx > max_frame + 100:
                break

            frame_idx += 1

        container.close()
        del container

    # Return frames in order, with fallback for missing frames
    if frames_dict:
        fallback_frame = frames_dict[min(frames_dict.keys())]
    else:
        fallback_frame = Image.new('RGB', (224, 224), (0, 0, 0))

    frames = [frames_dict.get(idx, fallback_frame) for idx in frame_idx_list]
    return frames, video_fps


def load_sliding_window(video_path, window_start_frame, window_length, stride, max_retries=2):
    """Load sliding window frames using decord, with PyAV fallback for pixel format issues."""
    last_error = None

    # Try decord first (faster, but has issues with some videos)
    if USE_DECORD:
        for attempt in range(max_retries):
            try:
                frames, fps = load_sliding_window_decord(video_path, window_start_frame, window_length, stride)
                gc.collect()
                return frames, fps
            except Exception as e:
                last_error = e
                gc.collect()

    # Fall back to PyAV (handles pixel format issues and works when decord is unavailable)
    try:
        frames, fps = load_sliding_window_av(video_path, window_start_frame, window_length, stride)
        gc.collect()
        return frames, fps
    except Exception as e:
        print(f"Warning: PyAV fallback also failed for {video_path}: {e}")

    # If all else fails, return dummy frames
    print(f"Warning: Failed to load video {video_path} (frame {window_start_frame}): {last_error}")
    dummy_frame = Image.new('RGB', (224, 224), (0, 0, 0))
    return [dummy_frame] * window_length, 30.0


def get_frame_path(rgb_root, youtube_id, frame_idx, total_frames):
    """
    Get the path to a cached JPG frame.

    Args:
        rgb_root: Root directory for RGB frames (e.g., /data3/xiaodan8/FineGym/RGB)
        youtube_id: YouTube video ID
        frame_idx: Frame index in the video
        total_frames: Total number of frames in the video (for zero-padding width)

    Returns:
        Path to the JPG frame file
    """
    # Determine number of digits for zero-padding based on total frames
    num_digits = len(str(total_frames))
    # Minimum 7 digits: the ENTIRE existing RGB store on this machine
    # (/data3/xiaodan8/FineGym/RGB, verified over all 180 video dirs) uses
    # 7-digit names ('0011264.jpg') written by an earlier code version;
    # with the old minimum of 6 every lookup would miss and fall back to
    # the (absent) raw videos, silently yielding dummy black frames.
    num_digits = max(num_digits, 7)

    frame_dir = os.path.join(rgb_root, youtube_id)
    frame_filename = f"{frame_idx:0{num_digits}d}.jpg"
    return os.path.join(frame_dir, frame_filename)


_CACHED_FRAME_INDICES = {}


def _get_cached_frame_indices(rgb_root, youtube_id):
    """Sorted list of cached JPG frame indices for a video (memoized).

    Used by the nearest-frame snapping in load_sliding_window_jpg. One
    os.listdir per video per worker process (~0.1 s, then cached).
    """
    key = (rgb_root, youtube_id)
    if key not in _CACHED_FRAME_INDICES:
        frame_dir = os.path.join(rgb_root, youtube_id)
        try:
            idxs = sorted(int(f[:-4]) for f in os.listdir(frame_dir)
                          if f.endswith('.jpg'))
        except OSError:
            idxs = []
        _CACHED_FRAME_INDICES[key] = idxs
    return _CACHED_FRAME_INDICES[key]


def load_sliding_window_jpg(rgb_root, youtube_id, video_path, window_start_frame,
                            window_length, stride, total_frames, video_fps):
    """
    Load sliding window frames from cached JPG files.
    If any frame is missing, load from video and save missing frames.

    Args:
        rgb_root: Root directory for RGB frames
        youtube_id: YouTube video ID
        video_path: Path to the video file (fallback for missing frames)
        window_start_frame: Starting frame in original video coordinates
        window_length: Number of frames to sample (e.g., 32)
        stride: Stride in original frames (e.g., 16)
        total_frames: Total number of frames in the video
        video_fps: Video FPS (returned for consistency)

    Returns:
        frames (list of PIL), video_fps
    """
    # Calculate frame indices for this window
    frame_indices = []
    for i in range(window_length):
        frame_idx = window_start_frame + i * stride
        # Clamp to valid range
        frame_idx = min(max(0, frame_idx), max(0, total_frames - 1))
        frame_indices.append(frame_idx)

    # Check which frames exist
    frame_paths = [get_frame_path(rgb_root, youtube_id, idx, total_frames) for idx in frame_indices]
    missing_indices = [i for i, path in enumerate(frame_paths) if not os.path.exists(path)]

    # NEAREST-FRAME SNAPPING (this machine has the JPG store but NOT the raw
    # videos): the store was cached by earlier runs with different window
    # alignments, so ~47% of exact indices miss by parity — measured 98.9%
    # of misses have a cached neighbor within 1 raw frame (99.6% within 2).
    # Snap a missing index to the nearest cached frame within stride//2
    # (max 33-266 ms error, negligible vs the 16-frame sampling stride);
    # only truly uncovered frames (~0.2%) fall through to the video path.
    if missing_indices:
        cached = _get_cached_frame_indices(rgb_root, youtube_id)
        if cached:
            import bisect
            tol = max(2, stride // 2)
            still_missing = []
            for i in missing_indices:
                fi = frame_indices[i]
                j = bisect.bisect_left(cached, fi)
                best = None
                for k in (j - 1, j):
                    if 0 <= k < len(cached):
                        if best is None or abs(cached[k] - fi) < abs(best - fi):
                            best = cached[k]
                if best is not None and abs(best - fi) <= tol:
                    frame_paths[i] = get_frame_path(
                        rgb_root, youtube_id, best, total_frames)
                else:
                    still_missing.append(i)
            missing_indices = still_missing

    frames = [None] * window_length

    if len(missing_indices) == 0:
        # All frames exist – load from JPG
        for i, path in enumerate(frame_paths):
            try:
                frames[i] = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load JPG {path}: {e}")
                missing_indices.append(i)

    if len(missing_indices) > 0:
        # Load whatever JPGs DO exist first (the original all-or-nothing
        # logic left every frame as dummy when the video is absent)
        _missing_set = set(missing_indices)
        for i, path in enumerate(frame_paths):
            if i not in _missing_set and os.path.exists(path):
                try:
                    frames[i] = Image.open(path).convert('RGB')
                except Exception:
                    pass
        # Some frames are missing – load from video
        try:
            video_frames, _ = load_sliding_window(video_path, window_start_frame, window_length, stride)

            # Ensure frame directory exists
            frame_dir = os.path.join(rgb_root, youtube_id)
            os.makedirs(frame_dir, exist_ok=True)

            # Save missing frames and fill in the frames list
            for i in missing_indices:
                if i < len(video_frames):
                    frame = video_frames[i]
                    frame_path = frame_paths[i]

                    # Save the frame
                    try:
                        frame.save(frame_path, 'JPEG', quality=95)
                    except Exception as e:
                        print(f"Warning: Failed to save JPG {frame_path}: {e}")

                    frames[i] = frame

            # Fill in frames that were loaded from video
            for i, frame in enumerate(frames):
                if frame is None and i < len(video_frames):
                    frames[i] = video_frames[i]

        except Exception as e:
            print(f"Warning: Failed to load video for missing frames {video_path}: {e}")

    # Fill any remaining None frames with dummy
    dummy_frame = Image.new('RGB', (224, 224), (0, 0, 0))
    frames = [f if f is not None else dummy_frame for f in frames]

    return frames, video_fps


@register_dataset("finegym_slide")
class FineGymSlideDataset(Dataset):
    """
    FineGym dataset with SLIDING WINDOW approach.

    Key features:
    - Training: Only include FULLY CONTAINED actions (no truncation)
    - Training: Skip windows with no complete actions
    - Testing: Use overlapping windows with clip-level aggregation
    - Sliding windows only cover the instance "span" (activity timespan), not entire video/clip

    Parameters:
    - window_length: Number of sampled frames per window (default: 32)
    - window_stride: Stride for sliding windows in sampled frames (default: 16 = 50% overlap)
    - sample_stride: Stride for sampling frames from video (default: 16)
    - test_overlap: Whether to use 50% overlap for test (default: True)
    - use_raw_video: If True, load FRAMES from raw YouTube videos (using raw_value annotations)
                     If False, load FRAMES from cropped clips (using new_value annotations)
    - load_jpg: If True (with use_raw_video=True), cache/load frames as JPG files for speed
                Frames are stored in RGB/{youtube_id}/{frame_number:06d}.jpg
                On first access, frames are extracted from video and saved as JPG.
                On subsequent accesses, JPG files are loaded directly (much faster).

    Sliding behavior (same for both modes):
    - use_raw_video=False: Slide from frame 0 to clip end (clip-relative coordinates)
    - use_raw_video=True: Slide from instance_span_start to instance_span_end (raw video coordinates)
    - NO padding around clips – exactly matches the cropped clip content

    Frame loading source:
    - use_raw_video=False: Load frames from cropped clip files (videos_Dec14/)
    - use_raw_video=True: Load frames from raw YouTube videos (video_raw/)
    - load_jpg=True: Load from cached JPG files in RGB/ folder (with fallback to video)

    Window boundary: Uses (window_length - 1) * sample_stride = 496 frames (not 512)
    This matches generate_json_short_clip.py line 85: segment <= locations[-1]

    For use_raw_video=True, metadata is cached by YouTube ID to avoid redundant I/O.
    Each clip/instance is treated separately for training and evaluation.
    """
    def __init__(
        self,
        is_training,          # if in training mode
        split,                # split, a tuple/list allowing concat of subsets
        backbone_type,
        round,
        train_json_file,
        val_json_file,
        max_seq_len,          # maximum sequence length during training
        trunc_thresh,
        crop_ratio,
        num_classes,          # number of action categories
        num_frames=1,         # number of frames for each feat (for offset calculation)
        video_min_frames=32,  # Minimum frames (not used in sliding window mode)
        video_max_frames=128, # Maximum frames (not used in sliding window mode)
        window_length=32,     # Number of sampled frames per window (default: 32)
        window_stride=16,     # Stride for sliding windows in sampled frames (default: 16 = 50% overlap)
        sample_stride=16,     # Stride for sampling frames from video (default: 16)
        test_overlap=True,    # Use overlap for test (for better aggregation)
        use_raw_video=False,  # If True, use raw videos from video_raw/ with raw_value annotations
        load_jpg=False,       # If True (with use_raw_video=True), cache/load frames as JPG for speed
        complete_only=True,   # If True, training windows only keep fully-contained actions; if False, partial-overlap actions are clipped to window
        data_fraction=1.0,                # float in (0,1]; deterministic subsample of training videos
        data_fraction_seed=42,            # seed for the subsample
        label_noise_type='none',          # 'none' | 'sibling' | 'cross_phrase'
        label_noise_rate=0.0,             # probability of corrupting any given action's label
        label_noise_seed=42,              # seed for the noise injection
        held_out_class_ids=None,          # Phase 2D ZSL: list of int c-action ids removed from training
        **kwargs              # ignore extra kwargs for compatibility
    ):
        # file path
        assert isinstance(split, tuple) or isinstance(split, list)
        if crop_ratio is not None:
            assert crop_ratio == 'None' or len(crop_ratio) == 2

        # Determine which json file to use based on is_training
        if is_training:
            self.json_file = train_json_file
        else:
            self.json_file = val_json_file

        if self.json_file is None:
            raise ValueError("No json_file specified for dataset. "
                             "Provide train_json_file/val_json_file or json_file.")

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.num_frames = num_frames
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = None if (crop_ratio is None or crop_ratio == 'None') else crop_ratio
        self.backbone_type = backbone_type
        self.round = round
        self.data_root = '/data3/xiaodan8/FineGym'

        # Raw video vs cropped clip option
        self.use_raw_video = use_raw_video
        if use_raw_video:
            self.video_dir = os.path.join(self.data_root, 'video_raw')
            print(f"[FineGymSlide] Using RAW videos from {self.video_dir}")
            print(f"[FineGymSlide] Annotations: raw_value (raw video coordinates)")
        else:
            self.video_dir = os.path.join(self.data_root, 'videos_Dec14')
            print(f"[FineGymSlide] Using CROPPED clips from {self.video_dir}")
            print(f"[FineGymSlide] Annotations: new_value (clip-relative coordinates)")

        # JPG frame caching option (only for use_raw_video=True)
        self.load_jpg = load_jpg and use_raw_video
        self.rgb_root = os.path.join(self.data_root, 'RGB')
        if self.load_jpg:
            print(f"[FineGymSlide] JPG caching ENABLED: {self.rgb_root}")
            os.makedirs(self.rgb_root, exist_ok=True)

        # Sliding window parameters (matching original 50% mAP experiment)
        self.window_length = window_length      # 32 sampled frames per window
        self.window_stride = window_stride      # Stride between windows (in sampled frames)
        self.sample_stride = sample_stride      # Sample every 16 original frames
        self.test_overlap = test_overlap        # Use 50% overlap for test
        # feat_stride = sample_stride (16 original frames per sampled frame)
        self.feat_stride = float(sample_stride)
        self.complete_only = complete_only
        if not complete_only:
            print(f"[FineGymSlide] complete_only=False — partial-overlap actions clipped to window will also be kept")
        self.data_fraction = float(data_fraction)
        self.data_fraction_seed = int(data_fraction_seed)
        self.label_noise_type = str(label_noise_type)
        self.label_noise_rate = float(label_noise_rate)
        self.label_noise_seed = int(label_noise_seed)

        # Window duration: window_length * sample_stride / fps
        # For 30fps: 32 * 16 / 30 = 17.07 seconds

        # Load verbalizer for hierarchy labels
        verbalizer_path = kwargs.get('verbalizer_path', '')
        self.verbalizer_path = verbalizer_path if verbalizer_path else ''
        if self.verbalizer_path:
            from libs.modeling.hierarchy_utils import (
                load_verbalizer, get_action_to_phrase_map, get_action_to_activity_map
            )
            self.verbalizer = load_verbalizer(self.verbalizer_path)
            self.action_to_phrase = get_action_to_phrase_map(self.verbalizer)
            self.action_to_activity = get_action_to_activity_map(self.verbalizer)
        else:
            self.verbalizer = None
            self.action_to_phrase = None
            self.action_to_activity = None

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # Optional deterministic data-fraction subsampling for low-data experiments.
        # Only subsamples training; val/test pass data_fraction=1.0 by default and
        # are never reduced. The seed must be deterministic for reproducibility.
        if self.is_training and 0.0 < self.data_fraction < 1.0:
            import random as _r
            import builtins as _b
            rng = _r.Random(self.data_fraction_seed)
            n_total = len(dict_db)
            keep_n = max(1, int(_b.round(n_total * self.data_fraction)))
            dict_db_list = list(dict_db)
            dict_db = tuple(rng.sample(dict_db_list, keep_n))
            print(f"[FineGymSlide] data_fraction={self.data_fraction} → kept {keep_n}/{n_total} videos (seed={self.data_fraction_seed})")
        # Optional deterministic label-noise injection (no-op when self.label_noise_rate == 0).
        dict_db = self._maybe_corrupt_labels(dict_db)
        assert len(label_dict) == num_classes, f"Expected {num_classes} classes, got {len(label_dict)}"
        self.label_dict = label_dict

        # Phase 2D v4 ZSL: KEEP held-out segments in the database. They contribute
        # to phrase/activity head training (parent labels are valid) and reg head
        # training. The action cls head's loss masks them out — see meta_archs.py.
        self.held_out_class_ids = set(int(x) for x in (held_out_class_ids or []))
        if self.is_training and self.held_out_class_ids:
            n_total = 0
            n_held = 0
            for v in dict_db:
                if v.get('labels') is None:
                    continue
                lbls = v['labels']
                n_total += len(lbls)
                n_held += sum(1 for l in lbls if int(l) in self.held_out_class_ids)
            print(f"[FineGymSlide] held_out: KEEPING {n_held}/{n_total} held-out action "
                  f"segments in training (action cls loss masks them, phrase/reg trains on all)")

        # Cache file for video metadata (speeds up subsequent runs)
        # Different cache for raw vs cropped videos
        cache_suffix = 'train' if is_training else 'val'
        video_type = 'raw' if use_raw_video else 'cropped'
        self.cache_file = os.path.join(self.data_root, f'video_metadata_cache_{cache_suffix}_{video_type}.json')

        # Create sliding windows for each video (with caching)
        # Returns (windows, video_items) where video_items is used for ground truth
        self.windows, video_items = self._create_sliding_windows_cached(dict_db)
        print(f"Created {len(self.windows)} sliding windows from {len(video_items)} videos")

        # Store video list for ground truth (used in validation)
        # Each clip/instance is kept separate (not merged by YouTube ID)
        self.video_list = video_items

        # Build video_id to windows mapping for test-time aggregation
        if not is_training:
            self._build_video_to_windows_map()

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'finegym_slide',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }
        self.transform = get_transforms(is_training, 224)

    def _build_video_to_windows_map(self):
        """Build mapping from video_id to list of window indices for aggregation."""
        self.video_to_windows = {}
        for idx, window in enumerate(self.windows):
            video_id = window['video']    # Original video ID (e.g., "MVLZz2J6tcE_E_003701_003784")
            if video_id not in self.video_to_windows:
                self.video_to_windows[video_id] = []
            self.video_to_windows[video_id].append(idx)

    def _load_video_metadata_cache(self):
        """Load cached video metadata if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"Loaded video metadata cache from {self.cache_file}")
                return cache
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return {}

    def _save_video_metadata_cache(self, cache):
        """Save video metadata cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f)
            print(f"Saved video metadata cache to {self.cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _create_sliding_windows_cached(self, dict_db):
        """
        Create sliding windows with caching for video metadata.
        This makes subsequent runs much faster.

        Both use_raw_video modes:
        - Process each clip/instance separately
        - Slide windows only over the instance's "span" (activity timespan)
        - Use clip-level IDs for training and evaluation

        For use_raw_video=True:
        - Metadata is cached by YouTube ID (shared across clips from same video)
        - Frame loading happens from raw video at __getitem__ time
        - Evaluation is still at clip/instance level

        Returns:
            tuple: (windows list, video_items list for ground truth)
        """
        # Load existing cache
        metadata_cache = self._load_video_metadata_cache()
        cache_updated = False

        windows = []
        skipped_videos = 0

        # Keep each clip/instance separate (no merging by YouTube ID)
        video_items = list(dict_db)

        total_videos = len(video_items)

        for idx, video_item in enumerate(video_items):
            if idx % 100 == 0:
                print(f"Processing video {idx}/{total_videos}... (skipped: {skipped_videos})")

            video_name = video_item['video']  # Clip name (e.g., 0LtLS9wROrk_E_000147_000152)

            # Construct video path based on use_raw_video setting
            if self.use_raw_video:
                # For raw videos: extract YouTube ID from clip name
                youtube_id = extract_youtube_id(video_name)
                video_path = find_video_file(self.video_dir, youtube_id)
                cache_key = youtube_id  # Cache by YouTube ID (shared across clips)

                # Check cache for raw video metadata (FPS and total_frames of RAW video)
                if cache_key in metadata_cache:
                    video_fps = metadata_cache[cache_key]['fps']
                    raw_total_frames = metadata_cache[cache_key]['total_frames']
                    raw_duration = metadata_cache[cache_key]['duration']
                else:
                    # Get raw video metadata (slow – opens video file)
                    video_fps, raw_total_frames, raw_duration = get_video_metadata(video_path)
                    # Cache raw video metadata (shared across clips from same YouTube video)
                    metadata_cache[cache_key] = {
                        'fps': video_fps,
                        'total_frames': raw_total_frames,
                        'duration': raw_duration
                    }
                    cache_updated = True

                if video_fps is None or video_fps == 0 or raw_total_frames == 0:
                    print(f"Warning: Could not get metadata for {video_path}")
                    skipped_videos += 1
                    continue

                # For use_raw_video=True: use raw video's total_frames/duration for clamping
                # but instance_span will define the actual sliding range
                total_frames = raw_total_frames
                duration = raw_duration

                # Validate instance span exists
                instance_span_start = video_item.get('instance_span_start')
                instance_span_end = video_item.get('instance_span_end')

                if instance_span_start is None or instance_span_end is None:
                    # No span info – try to get from segments
                    if video_item['segments'] is not None and len(video_item['segments']) > 0:
                        # Use segment bounds as fallback span
                        video_item['instance_span_start'] = float(video_item['segments'][:, 0].min())
                        video_item['instance_span_end'] = float(video_item['segments'][:, 1].max())
                    else:
                        print(f"Warning: No span info for {video_name}, skipping")
                        skipped_videos += 1
                        continue

            else:
                # For cropped clips: use video name directly
                video_path = video_name
                if not os.path.isabs(video_path):
                    video_path = os.path.join(self.video_dir, video_name + '.mp4')
                cache_key = video_name

                # Check cache (clip-level metadata)
                if cache_key in metadata_cache:
                    video_fps = metadata_cache[cache_key]['fps']
                    total_frames = metadata_cache[cache_key]['total_frames']
                    duration = metadata_cache[cache_key]['duration']
                else:
                    # Get video metadata (slow – opens video file)
                    video_fps, total_frames, duration = get_video_metadata(video_path)
                    # Update cache with clip-level metadata
                    metadata_cache[cache_key] = {
                        'fps': video_fps,
                        'total_frames': total_frames,
                        'duration': duration
                    }
                    cache_updated = True

                if total_frames == 0:
                    print(f"Warning: Could not get metadata for {video_path}")
                    skipped_videos += 1
                    continue

            # Create windows for this video
            # For use_raw_video=True, pass youtube_id for JPG caching
            yt_id = youtube_id if self.use_raw_video else None
            video_windows = self._create_windows_for_video(
                video_item, video_path, video_fps, total_frames, duration, yt_id
            )

            windows.extend(video_windows)

        # Save updated cache
        if cache_updated:
            self._save_video_metadata_cache(metadata_cache)

        if skipped_videos > 0:
            print(f"Warning: Skipped {skipped_videos}/{total_videos} videos due to metadata issues")

        # Return both windows and video_items (for ground truth)
        return windows, video_items

    def _create_windows_for_video(self, video_item, video_path, video_fps, total_frames, duration, youtube_id=None):
        """Create sliding windows for a single video.

        Args:
            youtube_id: YouTube video ID (only for use_raw_video=True, used for JPG caching)
        """
        windows = []

        # Calculate number of sampled frames possible
        total_sampled_frames = total_frames // self.sample_stride

        # Determine overlap based on training/test mode
        if self.is_training:
            # Training: use configured window_stride (default: 16 = 50% overlap)
            effective_stride = self.window_stride
        else:
            # Test: use 50% overlap if test_overlap is True, else no overlap
            if self.test_overlap:
                effective_stride = self.window_length // 2  # 50% overlap
            else:
                effective_stride = self.window_length  # No overlap

        # CRITICAL: Window boundary should match finegym_original
        # We sample 32 frames at positions 0, 16, 32, ..., 496 (relative to window start)
        # So the LAST sampled frame is at (window_length - 1) * sample_stride = 31 * 16 = 496
        # Actions must fit within [0, 496], not [0, 512]!
        # This matches how generate_json_short_clip.py checks: segment <= locations[-1]
        window_span_frames = (self.window_length - 1) * self.sample_stride  # 496, not 512
        window_duration = self.window_length * self.sample_stride / video_fps
        window_stride_frames = effective_stride * self.sample_stride

        # Determine sliding range based on use_raw_video mode:
        # Both modes slide over the SAME clip duration (no padding), simulating finegym_original
        # - use_raw_video=True: slide over instance_span in raw video coordinates
        # - use_raw_video=False: slide over clip (0 to clip_duration, same as instance_span duration)
        instance_span_start = video_item.get('instance_span_start')
        instance_span_end = video_item.get('instance_span_end')

        if self.use_raw_video and instance_span_start is not None and instance_span_end is not None:
            # For raw video: slide exactly over the instance span (NO padding)
            # This simulates finegym_original which processes exactly the cropped clip content
            slide_start_frame = int(instance_span_start * video_fps)
            slide_end_frame = int(instance_span_end * video_fps)

            # Ensure we don't exceed video bounds
            slide_start_frame = max(0, slide_start_frame)
            slide_end_frame = min(total_frames, slide_end_frame)
        elif not self.use_raw_video:
            # For cropped clips: slide over entire clip (already cropped to activity span)
            # Clip frames are 0 to total_frames, clip time is 0 to duration
            slide_start_frame = 0
            slide_end_frame = total_frames
        elif video_item['segments'] is not None and len(video_item['segments']) > 0:
            # Fallback for use_raw_video=True without instance_span:
            # Use action segment bounds (no padding)
            min_seg_time = float(video_item['segments'][:, 0].min())
            max_seg_time = float(video_item['segments'][:, 1].max())

            slide_start_frame = int(min_seg_time * video_fps)
            slide_end_frame = int(max_seg_time * video_fps)

            slide_start_frame = max(0, slide_start_frame)
            slide_end_frame = min(total_frames, slide_end_frame)
        else:
            # No annotations – skip this video for training, minimal window for test
            if self.is_training:
                return windows  # Return empty list – no windows to create
            else:
                slide_start_frame = 0
                slide_end_frame = min(window_span_frames, total_frames)

        # If video/clip is shorter than window_length, create one window
        # Note: For use_raw_video=True, this uses instance span; for False, uses clip start (0)
        if total_sampled_frames <= self.window_length:
            # Use slide_start_frame which was set based on use_raw_video mode
            window_start_frame = slide_start_frame
            window_start_time = window_start_frame / video_fps
            # Use the actual window span, not full video duration
            window_end_time = min(window_start_time + window_span_frames / video_fps, duration)

            segments, labels = self._get_window_segments_complete_only(
                video_item, window_start_time, window_end_time, video_fps,
                complete_only=self.complete_only,
            )

            # Training: only keep windows with at least one complete action
            # Testing: keep ALL windows (for aggregation coverage)
            if self.is_training:
                if segments is not None and len(segments) > 0:
                    # Collect unique activity IDs from actions in this window
                    window_activity_set = set()
                    if self.action_to_activity is not None:
                        for label in labels:
                            window_activity_set.add(self.action_to_activity[int(label)])
                    phrase_segs_win, phrase_lbls_win = self._filter_segments_to_window(
                        video_item.get('phrase_segments'), video_item.get('phrase_labels'),
                        window_start_time, window_end_time,
                    )
                    activity_segs_win, activity_lbls_win = self._filter_segments_to_window(
                        video_item.get('activity_segments'), video_item.get('activity_labels'),
                        window_start_time, window_end_time,
                    )
                    windows.append({
                        'id': video_item['id'],
                        'video': video_item['video'],
                        'video_path': video_path,
                        'window_start_frame': window_start_frame,
                        'window_start_time': window_start_time,
                        'window_end_time': window_end_time,
                        'segments': segments,
                        'labels': labels,
                        'phrase_segments': phrase_segs_win,
                        'phrase_labels': phrase_lbls_win,
                        'activity_segments': activity_segs_win,
                        'activity_labels': activity_lbls_win,
                        'activity_id_set': sorted(window_activity_set),
                        'fps': video_fps,
                        'duration': window_duration,
                        'total_frames': total_frames,
                        'youtube_id': youtube_id,
                    })
            else:
                # Collect unique activity IDs from actions in this window
                window_activity_set = set()
                if self.action_to_activity is not None and labels is not None:
                    for label in labels:
                        window_activity_set.add(self.action_to_activity[int(label)])
                phrase_segs_win, phrase_lbls_win = self._filter_segments_to_window(
                    video_item.get('phrase_segments'), video_item.get('phrase_labels'),
                    window_start_time, window_end_time,
                )
                activity_segs_win, activity_lbls_win = self._filter_segments_to_window(
                    video_item.get('activity_segments'), video_item.get('activity_labels'),
                    window_start_time, window_end_time,
                )
                # Test: include all windows for complete coverage during aggregation
                windows.append({
                    'id': f"{video_item['id']}_w0",
                    'video': video_item['video'],
                    'video_path': video_path,
                    'window_start_frame': window_start_frame,
                    'window_start_time': window_start_time,
                    'window_end_time': window_end_time,
                    'segments': segments if segments is not None else np.array([], dtype=np.float32).reshape(0, 2),
                    'labels': labels if labels is not None else np.array([], dtype=np.int64),
                    'phrase_segments': phrase_segs_win,
                    'phrase_labels': phrase_lbls_win,
                    'activity_segments': activity_segs_win,
                    'activity_labels': activity_lbls_win,
                    'activity_id_set': sorted(window_activity_set),
                    'fps': video_fps,
                    'duration': window_duration,
                    'total_frames': total_frames,
                    'youtube_id': youtube_id,
                })
        else:
            # Create sliding windows within the determined range
            window_idx = 0
            window_start_frame = slide_start_frame

            while window_start_frame < slide_end_frame:
                window_start_time = window_start_frame / video_fps
                # The window boundary for segment filtering should be the LAST SAMPLED FRAME
                # which is at window_start_frame + (window_length - 1) * sample_stride
                window_end_frame_for_filter = min(window_start_frame + window_span_frames, total_frames)
                window_end_time = window_end_frame_for_filter / video_fps

                segments, labels = self._get_window_segments_complete_only(
                    video_item, window_start_time, window_end_time, video_fps,
                    complete_only=self.complete_only,
                )

                # Training: only include windows with at least one COMPLETE action
                # Testing: include ALL windows for complete coverage during aggregation
                if self.is_training:
                    if segments is not None and len(segments) > 0:
                        # Collect unique activity IDs from actions in this window
                        window_activity_set = set()
                        if self.action_to_activity is not None:
                            for label in labels:
                                window_activity_set.add(self.action_to_activity[int(label)])
                        phrase_segs_win, phrase_lbls_win = self._filter_segments_to_window(
                            video_item.get('phrase_segments'), video_item.get('phrase_labels'),
                            window_start_time, window_end_time,
                        )
                        activity_segs_win, activity_lbls_win = self._filter_segments_to_window(
                            video_item.get('activity_segments'), video_item.get('activity_labels'),
                            window_start_time, window_end_time,
                        )
                        windows.append({
                            'id': f"{video_item['id']}_w{window_idx}",
                            'video': video_item['video'],
                            'video_path': video_path,
                            'window_start_frame': window_start_frame,
                            'window_start_time': window_start_time,
                            'window_end_time': window_end_time,
                            'segments': segments,
                            'labels': labels,
                            'phrase_segments': phrase_segs_win,
                            'phrase_labels': phrase_lbls_win,
                            'activity_segments': activity_segs_win,
                            'activity_labels': activity_lbls_win,
                            'activity_id_set': sorted(window_activity_set),
                            'fps': video_fps,
                            'duration': window_duration,
                            'total_frames': total_frames,
                            'youtube_id': youtube_id,
                        })
                else:
                    # Collect unique activity IDs from actions in this window
                    window_activity_set = set()
                    if self.action_to_activity is not None and labels is not None:
                        for label in labels:
                            window_activity_set.add(self.action_to_activity[int(label)])
                    phrase_segs_win, phrase_lbls_win = self._filter_segments_to_window(
                        video_item.get('phrase_segments'), video_item.get('phrase_labels'),
                        window_start_time, window_end_time,
                    )
                    activity_segs_win, activity_lbls_win = self._filter_segments_to_window(
                        video_item.get('activity_segments'), video_item.get('activity_labels'),
                        window_start_time, window_end_time,
                    )
                    # Test: include all windows for complete coverage
                    windows.append({
                        'id': f"{video_item['id']}_w{window_idx}",
                        'video': video_item['video'],
                        'video_path': video_path,
                        'window_start_frame': window_start_frame,
                        'window_start_time': window_start_time,
                        'window_end_time': window_end_time,
                        'segments': segments if segments is not None else np.array([], dtype=np.float32).reshape(0, 2),
                        'labels': labels if labels is not None else np.array([], dtype=np.int64),
                        'phrase_segments': phrase_segs_win,
                        'phrase_labels': phrase_lbls_win,
                        'activity_segments': activity_segs_win,
                        'activity_labels': activity_lbls_win,
                        'activity_id_set': sorted(window_activity_set),
                        'fps': video_fps,
                        'duration': window_duration,
                        'total_frames': total_frames,
                        'youtube_id': youtube_id,
                    })

                window_idx += 1
                window_start_frame += window_stride_frames

                # Stop if we've processed the last possible window
                if window_start_frame >= slide_end_frame - self.sample_stride:
                    break

        return windows

    def _filter_segments_to_window(self, segments, labels, window_start_time, window_end_time):
        """Return per-window segments + labels in seconds (window-relative).

        A segment is kept if it has any temporal overlap with the window. The
        kept segment is clipped to the window edge and made window-relative.
        Used for phrase + activity GT — both can be longer than a window and
        partial overlaps still produce useful supervision.
        """
        if segments is None or len(segments) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        kept_segs, kept_lbls = [], []
        for (s, e), lbl in zip(segments, labels):
            if e <= window_start_time or s >= window_end_time:
                continue
            cs = max(s, window_start_time)
            ce = min(e, window_end_time)
            kept_segs.append([cs - window_start_time, ce - window_start_time])
            kept_lbls.append(int(lbl))
        if not kept_segs:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.asarray(kept_segs, dtype=np.float32),
            np.asarray(kept_lbls, dtype=np.int64),
        )

    def _get_window_segments_complete_only(self, video_item, window_start_time, window_end_time, video_fps, complete_only=True):
        """
        Get segments inside a window.

        complete_only=True (default): only keep actions FULLY CONTAINED within
        the window — matches finegym_original's clean training behavior.
        complete_only=False: also keep actions that partially overlap the
        window, clipped to [window_start, window_end].

        Returns:
            segments: np.array of shape (N, 2) with times relative to window start (in seconds)
            labels: np.array of shape (N,)
        """
        if video_item['segments'] is None:
            return None, None

        segments = []
        labels = []

        for i, (seg, label) in enumerate(zip(video_item['segments'], video_item['labels'])):
            seg_start, seg_end = seg[0], seg[1]

            if complete_only:
                # Action must be fully inside the window.
                if seg_start >= window_start_time and seg_end <= window_end_time:
                    rel_start = seg_start - window_start_time
                    rel_end = seg_end - window_start_time
                    segments.append([rel_start, rel_end])
                    labels.append(label)
            else:
                # Any non-empty overlap is kept, clipped to the window.
                ov_start = max(seg_start, window_start_time)
                ov_end = min(seg_end, window_end_time)
                if ov_start < ov_end:
                    rel_start = ov_start - window_start_time
                    rel_end = ov_end - window_start_time
                    segments.append([rel_start, rel_end])
                    labels.append(label)

        if len(segments) > 0:
            return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)
        else:
            return None, None

    def get_attributes(self):
        return self.db_attributes

    def get_ground_truth_df(self):
        """
        Generate ground truth DataFrame for ANETdetection evaluator.
        For sliding window, we use WINDOW IDs (with _w suffix) and window-relative times.

        Note: For video-level evaluation with aggregation, use get_video_level_ground_truth_df()
        """
        import pandas as pd
        vids, starts, stops, labels = [], [], [], []

        for window in self.windows:
            if window['segments'] is not None and len(window['segments']) > 0:
                for i in range(len(window['segments'])):
                    vids.append(window['id'])
                    starts.append(float(window['segments'][i, 0]))
                    stops.append(float(window['segments'][i, 1]))
                    labels.append(int(window['labels'][i]))

        return pd.DataFrame({
            'video-id': vids,
            't-start': starts,
            't-end': stops,
            'label': labels
        })

    def get_video_level_ground_truth_df(self):
        """
        Generate ground truth DataFrame at CLIP/INSTANCE level (not window level).
        Used for evaluation when aggregating predictions across overlapping windows.

        Returns segments in clip-relative coordinates (not window-relative).

        Both use_raw_video modes use CLIP/INSTANCE level evaluation:
        - use_raw_video=False: coordinates relative to cropped clip (new_value annotations)
        - use_raw_video=True: coordinates relative to raw video (raw_value annotations)

        Each clip/instance in the annotation file is treated as a separate "video" for evaluation.
        Clips from the same YouTube video are NOT merged – they are evaluated independently.
        """
        import pandas as pd
        vids, starts, stops, labels = [], [], [], []

        # Use original video annotations
        for video_item in self.video_list:
            if video_item['segments'] is not None:
                for i in range(len(video_item['segments'])):
                    vids.append(video_item['video'])  # Use video name, not window ID
                    starts.append(float(video_item['segments'][i, 0]))
                    stops.append(float(video_item['segments'][i, 1]))
                    labels.append(int(video_item['labels'][i]))

        return pd.DataFrame({
            'video-id': vids,
            't-start': starts,
            't-end': stops,
            'label': labels
        })

    def _parse_span_time(self, span_str):
        """Parse span string like "<4.0 seconds>" to float value."""
        return float(span_str.strip('<>').replace(' seconds', ''))

    def _derive_hierarchy_gt(self, value, annotation_key, label_dict):
        """Derive phrase + activity GT segments from one annotation entry.

        Returns:
            phrase_segments: np.float32 (Pp, 2)  — merged-span phrase segments
            phrase_labels:    np.int64   (Pp,)
            activity_segments: np.float32 (Aa, 2)  — directly from activity 'span'
            activity_labels:   np.int64   (Aa,)
            consistency_skipped: bool — True if any phrase overlap was detected
                                        (in that case, phrase GT is empty for that activity)

        FineGym hierarchy semantics:
        — Each activity has a 'span' field that is the activity's temporal extent.
        — A phrase corresponds to a contiguous time region within an activity that
          spans all child actions sharing a phrase_id. We compute the merged span
          per (activity_instance, phrase_id) pair.
        """
        from libs.modeling.hierarchy_utils import (
            load_verbalizer, get_action_to_phrase_map, get_action_to_activity_map,
        )
        # Cache verbalizer maps once per dataset instance.
        if not hasattr(self, '_a2p'):
            verb_path = 'configs/finegym_zsl_verbalizer.json'
            verb = load_verbalizer(verb_path)
            self._a2p = get_action_to_phrase_map(verb)
            self._a2a = get_action_to_activity_map(verb)

        phrase_segs, phrase_lbls = [], []
        activity_segs, activity_lbls = [], []
        consistency_skipped = False

        for activity in value.get(annotation_key, []):
            # Activity segment from 'span'.
            if 'span' in activity and activity['span']:
                a_start = self._parse_span_time(activity['span'][0])
                a_end = self._parse_span_time(activity['span'][1])
                activity_id_str = activity.get('activity_id', None)
                if activity_id_str is None:
                    continue
                # 'a0' → 0, 'a1' → 1, etc.
                activity_label = int(activity_id_str[1:])
                activity_segs.append([a_start, a_end])
                activity_lbls.append(activity_label)

            # Phrase segments: merged span per CONTIGUOUS RUN of same-phrase actions.
            # FineGym actions of different phrases interleave within an activity, so a
            # naive "merge all same-phrase actions" produces overlapping phrase intervals.
            # Walking actions in temporal order and starting a new phrase instance when
            # the phrase_id changes yields temporally-disjoint phrase instances by
            # construction; the same phrase class may appear multiple times.
            actions_in_order = []
            for action in activity.get('actions', []):
                aid_str = action['action_id']
                aid = int(aid_str[1:])
                pid = self._a2p.get(aid, None)
                if pid is None:
                    continue
                s = self._parse_span_time(action['span'][0])
                e = self._parse_span_time(action['span'][1])
                actions_in_order.append((s, e, pid))
            actions_in_order.sort(key=lambda x: x[0])

            this_activity_phrases = []  # list of [start, end, phrase_id]
            cur = None
            for (s, e, pid) in actions_in_order:
                if cur is not None and cur[2] == pid:
                    cur[1] = max(cur[1], e)
                else:
                    if cur is not None:
                        this_activity_phrases.append(cur)
                    cur = [s, e, pid]
            if cur is not None:
                this_activity_phrases.append(cur)

            # Sanity check: phrase instances should now be temporally disjoint by
            # construction. This is just a guard — if it ever fires there's an annotation
            # pathology (e.g., overlapping action timestamps within one activity).
            this_activity_skipped = False
            for i in range(len(this_activity_phrases) - 1):
                if this_activity_phrases[i][1] > this_activity_phrases[i + 1][0] + 1e-6:
                    consistency_skipped = True
                    this_activity_skipped = True
                    break

            if this_activity_skipped:
                continue

            for s, e, pid in this_activity_phrases:
                phrase_segs.append([s, e])
                phrase_lbls.append(int(pid))

        if phrase_segs:
            phrase_segs = np.asarray(phrase_segs, dtype=np.float32)
            phrase_lbls = np.asarray(phrase_lbls, dtype=np.int64)
        else:
            phrase_segs = np.zeros((0, 2), dtype=np.float32)
            phrase_lbls = np.zeros((0,), dtype=np.int64)

        if activity_segs:
            activity_segs = np.asarray(activity_segs, dtype=np.float32)
            activity_lbls = np.asarray(activity_lbls, dtype=np.int64)
        else:
            activity_segs = np.zeros((0, 2), dtype=np.float32)
            activity_lbls = np.zeros((0,), dtype=np.int64)

        return phrase_segs, phrase_lbls, activity_segs, activity_lbls, consistency_skipped

    def _load_json_db(self, json_file):
        """Load database from JSON Lines format.

        When use_raw_video=True, uses 'raw_value' annotations (raw video coordinates).
        When use_raw_video=False, uses 'new_value' annotations (clip-relative coordinates).
        """
        json_db = {}
        with open(json_file, 'r') as fid:
            for line in fid:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    video_name = entry['video']
                    json_db[video_name] = entry

        # Choose annotation source based on use_raw_video setting
        annotation_key = 'raw_value' if self.use_raw_video else 'new_value'

        # Build label_dict
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                annotations = value.get(annotation_key, [])
                for activity in annotations:
                    for action in activity.get('actions', []):
                        action_id = action['action_id']
                        label_id = int(action_id[1:])
                        label_dict[action_id] = label_id
        else:
            label_dict = self.label_dict

        # Fill in the db
        dict_db = tuple()
        for key, value in json_db.items():
            video_path = value.get('video', key)
            annotations = value.get(annotation_key, [])

            # Parse the instance-level span (activity timespan in raw video)
            # This is different from action-level segments
            instance_span = value.get('span', None)
            if instance_span is not None:
                instance_span_start = self._parse_span_time(instance_span[0])
                instance_span_end = self._parse_span_time(instance_span[1])
            else:
                instance_span_start = None
                instance_span_end = None

            segments, labels = [], []
            for activity in annotations:
                for action in activity.get('actions', []):
                    span = action['span']
                    start_time = self._parse_span_time(span[0])
                    end_time = self._parse_span_time(span[1])
                    segments.append([start_time, end_time])

                    action_id = action['action_id']
                    labels.append(label_dict[action_id])

            if segments:
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.asarray(labels, dtype=np.int64)
            else:
                segments = None
                labels = None

            phrase_segs, phrase_lbls, activity_segs, activity_lbls, consistency_skipped = (
                self._derive_hierarchy_gt(value, annotation_key, label_dict)
            )
            if consistency_skipped:
                self._consistency_skipped_count = getattr(self, '_consistency_skipped_count', 0) + 1

            dict_db += ({
                'id': key,
                'video': video_path,
                'segments': segments,
                'labels': labels,
                # Store instance span for sliding window range
                'instance_span_start': instance_span_start,
                'instance_span_end': instance_span_end,
                'phrase_segments': phrase_segs,
                'phrase_labels': phrase_lbls,
                'activity_segments': activity_segs,
                'activity_labels': activity_lbls,
            }, )

        if hasattr(self, '_consistency_skipped_count') and self._consistency_skipped_count > 0:
            print(f"[FineGymSlide] phrase-overlap consistency check skipped {self._consistency_skipped_count} activity instances")

        return dict_db, label_dict

    def _maybe_corrupt_labels(self, dict_db):
        """If label noise is configured, swap action_ids per the chosen scheme.

        sibling: pick a random action within the same phrase (different action id).
        cross_phrase: pick a random action with the same activity but a different phrase.
                      Falls back to a sibling if the cross-phrase pool is empty.
        Validation/test data is never corrupted (gated by self.is_training).
        Deterministic given self.label_noise_seed.
        """
        if not self.is_training:
            return dict_db
        if self.label_noise_type == 'none' or self.label_noise_rate <= 0.0:
            return dict_db
        assert self.label_noise_type in ('sibling', 'cross_phrase'), \
            f"unknown label_noise_type={self.label_noise_type}"

        import random as _r
        from libs.modeling.hierarchy_utils import (
            load_verbalizer, get_action_to_phrase_map, get_action_to_activity_map,
        )
        verb_path = 'configs/finegym_zsl_verbalizer.json'
        verb = load_verbalizer(verb_path)
        a2p = get_action_to_phrase_map(verb)
        a2a = get_action_to_activity_map(verb)

        # Build candidate pools per source action.
        sibling_pool = {a: [b for b, p in a2p.items() if p == a2p[a] and b != a] for a in a2p}
        cross_phrase_pool = {
            a: [b for b in a2p
                if a2a.get(b) == a2a.get(a) and a2p.get(b) != a2p.get(a)]
            for a in a2p
        }

        rng = _r.Random(self.label_noise_seed)
        new_db = []
        n_corrupted = 0
        n_total = 0
        for item in dict_db:
            if item.get('labels') is None or len(item['labels']) == 0:
                new_db.append(item); continue
            new_labels = item['labels'].copy()
            for i in range(len(new_labels)):
                n_total += 1
                if rng.random() < self.label_noise_rate:
                    src = int(new_labels[i])
                    if self.label_noise_type == 'sibling':
                        pool = sibling_pool.get(src, [])
                    else:  # cross_phrase
                        pool = cross_phrase_pool.get(src, [])
                        if not pool:
                            pool = sibling_pool.get(src, [])
                    if pool:
                        new_labels[i] = rng.choice(pool)
                        n_corrupted += 1
            new_item = dict(item)
            new_item['labels'] = new_labels
            new_db.append(new_item)
        if isinstance(dict_db, tuple):
            new_db = tuple(new_db)
        print(f"[FineGymSlide] label_noise_type={self.label_noise_type} rate={self.label_noise_rate} → corrupted {n_corrupted}/{n_total} action labels (seed={self.label_noise_seed})")
        return new_db

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Load a sliding window and return data dict.
        """
        window = self.windows[idx]

        # Load frames for this window
        try:
            # Use JPG caching if enabled (only for use_raw_video=True)
            if self.load_jpg and window.get('youtube_id') is not None:
                frames, video_fps = load_sliding_window_jpg(
                    self.rgb_root,
                    window['youtube_id'],
                    window['video_path'],
                    window['window_start_frame'],
                    self.window_length,
                    self.sample_stride,
                    window['total_frames'],
                    window['fps']
                )
            else:
                frames, video_fps = load_sliding_window(
                    window['video_path'],
                    window['window_start_frame'],
                    self.window_length,
                    self.sample_stride
                )
        except Exception as e:
            print(f"[ERROR] Failed to load video {window['video_path']}: {e}")
            # Return dummy frames on failure
            dummy_frame = Image.new('RGB', (224, 224), (0, 0, 0))
            frames = [dummy_frame] * self.window_length
            video_fps = 30.0

        # Handle case where we got fewer frames (end of video)
        if len(frames) < self.window_length:
            # Pad with last frame
            while len(frames) < self.window_length:
                frames.append(frames[-1])

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

        # IMPORTANT: Use window['fps'] (from metadata during window creation)
        # NOT video_fps (from decoder) to ensure consistency
        # frame_idx = time_seconds * fps / feat_stride
        window_fps = window['fps']  # Use the fps from window creation
        # Convert segments (in seconds, relative to window) to frame indices
        if window['segments'] is not None and len(window['segments']) > 0:
            # segments are in seconds relative to window start
            # Convert to sampled frame indices
            segments = torch.from_numpy(
                window['segments'] * window_fps / feat_stride - feat_offset
            )
            labels = torch.from_numpy(window['labels'])
        else:
            segments = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Phrase + activity GT in feature-grid coordinates (analogous to action segments).
        if window.get('phrase_segments') is not None and len(window['phrase_segments']) > 0:
            phrase_segments = torch.from_numpy(
                window['phrase_segments'] * window_fps / feat_stride - feat_offset
            )
            phrase_labels_t = torch.from_numpy(window['phrase_labels'])
        else:
            phrase_segments = torch.zeros((0, 2), dtype=torch.float32)
            phrase_labels_t = torch.zeros((0,), dtype=torch.int64)

        if window.get('activity_segments') is not None and len(window['activity_segments']) > 0:
            activity_segments = torch.from_numpy(
                window['activity_segments'] * window_fps / feat_stride - feat_offset
            )
            activity_labels_t = torch.from_numpy(window['activity_labels'])
        else:
            activity_segments = torch.zeros((0, 2), dtype=torch.float32)
            activity_labels_t = torch.zeros((0,), dtype=torch.int64)

        # Derive phrase labels from action labels
        if self.action_to_phrase is not None and len(labels) > 0:
            phrase_labels = torch.tensor(
                [self.action_to_phrase[l.item()] for l in labels],
                dtype=torch.long
            )
        else:
            phrase_labels = torch.zeros_like(labels)

        # Window-level activity: multi-hot vector (4 classes)
        num_activities = 4
        activity_label = torch.zeros(num_activities, dtype=torch.float32)
        if self.action_to_activity is not None:
            for act_idx in window.get('activity_id_set', []):
                activity_label[act_idx] = 1.0

        # Return data dict
        # Use window['fps'] for consistency (same fps used for segment conversion and duration)
        data_dict = {
            'video_id': window['id'],
            'feats': feats,                # C x T x H x W (T = window_length = 32)
            'segments': segments,          # N x 2 (in sampled frame indices)
            'labels': labels,              # N
            'phrase_segments': phrase_segments,
            'phrase_labels': phrase_labels_t,
            'activity_segments': activity_segments,
            'activity_labels': activity_labels_t,
            'activity_label': activity_label, # (4,) multi-hot window-level activity
            'fps': window['fps'],          # Use stored fps for consistency
            'duration': window['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames,
            # Additional info for test-time aggregation
            'window_start_time': window['window_start_time'],
            'video_name': window['video'],  # Original video ID for aggregation
            # HGDD: raw video info for dense re-loading
            'video_path': window.get('video_path'),
            'window_start_frame': window.get('window_start_frame'),
            'total_frames': window.get('total_frames'),
            'youtube_id': window.get('youtube_id'),
            'use_jpg': self.load_jpg and window.get('youtube_id') is not None,
            'rgb_root': getattr(self, 'rgb_root', None),
        }

        # For training, truncate/augment if needed
        if self.is_training and (segments is not None) and len(segments) > 0:
            # Only truncate if sequence is longer than max_seq_len
            if feats.shape[1] > self.max_seq_len:
                data_dict = truncate_feats(
                    data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
                )

        # For training: skip windows with no segments
        if self.is_training:
            if data_dict['segments'] is None or len(data_dict['segments']) == 0:
                # Skip to next item
                return self.__getitem__((idx + 1) % len(self))

        return data_dict


def aggregate_window_predictions(all_results, nms_func, iou_threshold=0.5, min_score=0.001, max_seg_num=200):
    """
    Aggregate predictions from overlapping windows to video-level results.

    This function:
    1. Groups predictions by original video ID
    2. Converts window-relative predictions to video-relative coordinates using window_start_time
    3. Applies NMS at the video level

    Args:
        all_results: Dict with keys 'video-id', 't-start', 't-end', 'label', 'score', 'window_start_time'
            where video-id includes window suffix (e.g., "video_w0", "video_w1")
            and window_start_time contains the offset for each prediction
        nms_func: NMS function (e.g., batched_nms from libs.utils.nms)
        iou_threshold: IoU threshold for NMS
        min_score: Minimum score threshold
        max_seg_num: Maximum number of segments to keep per video

    Returns:
        Aggregated results dict with video-level predictions (times in video coordinates)
    """
    import torch
    from collections import defaultdict

    # Parse window info from video_id
    # Format: "original_video_id_wN" where N is window index
    video_predictions = defaultdict(lambda: {'segs': [], 'scores': [], 'labels': []})

    # Check if window_start_time is available (support both key names)
    if 'window_start_time' in all_results and len(all_results['window_start_time']) > 0:
        window_offset_key = 'window_start_time'
        has_window_offset = True
    elif 'window-start-time' in all_results and len(all_results['window-start-time']) > 0:
        window_offset_key = 'window-start-time'
        has_window_offset = True
    else:
        window_offset_key = None
        has_window_offset = False

    # Group predictions by original video
    for i in range(len(all_results['video-id'])):
        vid = all_results['video-id'][i]
        t_start = all_results['t-start'][i] if not isinstance(all_results['t-start'][i], torch.Tensor) else all_results['t-start'][i].item()
        t_end = all_results['t-end'][i] if not isinstance(all_results['t-end'][i], torch.Tensor) else all_results['t-end'][i].item()
        label = all_results['label'][i] if not isinstance(all_results['label'][i], torch.Tensor) else all_results['label'][i].item()
        score = all_results['score'][i] if not isinstance(all_results['score'][i], torch.Tensor) else all_results['score'][i].item()

        # Get window offset to convert to video coordinates
        if has_window_offset:
            window_offset = all_results[window_offset_key][i]
            if isinstance(window_offset, torch.Tensor):
                window_offset = window_offset.item()
        else:
            window_offset = 0.0

        # Convert window-relative times to video-relative times
        video_t_start = t_start + window_offset
        video_t_end = t_end + window_offset

        # Extract original video ID (remove window suffix)
        # Format: "video_name_wN"
        if '_w' in vid:
            parts = vid.rsplit('_w', 1)
            original_vid = parts[0]
        else:
            original_vid = vid

        video_predictions[original_vid]['segs'].append((video_t_start, video_t_end))
        video_predictions[original_vid]['scores'].append(score)
        video_predictions[original_vid]['labels'].append(label)

    # Apply NMS per video and collect results
    aggregated_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    for vid, preds in video_predictions.items():
        if len(preds['segs']) == 0:
            continue

        segs = torch.tensor(preds['segs'], dtype=torch.float32)
        scores = torch.tensor(preds['scores'], dtype=torch.float32)
        labels = torch.tensor(preds['labels'], dtype=torch.int64)

        # Apply NMS
        nms_segs, nms_scores, nms_labels = nms_func(
            segs, scores, labels,
            iou_threshold=iou_threshold,
            min_score=min_score,
            max_seg_num=max_seg_num,
            use_soft_nms=True,
            multiclass=True
        )

        # Add to results
        for i in range(len(nms_segs)):
            aggregated_results['video-id'].append(vid)
            aggregated_results['t-start'].append(nms_segs[i, 0].item())
            aggregated_results['t-end'].append(nms_segs[i, 1].item())
            aggregated_results['label'].append(nms_labels[i].item())
            aggregated_results['score'].append(nms_scores[i].item())

    return aggregated_results
