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


def get_video_metadata_decord(video_path):
    """Get video metadata (fps, total_frames, duration) using decord."""
    try:
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
                duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0
                if duration > 0:
                    total_frames = int(duration * video_fps)
                else:
                    # Count frames manually
                    total_frames = sum(1 for _ in container.decode(video=0))
                    container.seek(0)

            duration = total_frames / video_fps if video_fps > 0 else 0
            container.close()
            return video_fps, total_frames, duration
    except Exception:
        return 30.0, 0, 0.0


def get_video_metadata(video_path):
    """Get video metadata using decord only (PyAV can hang on corrupted videos)."""
    if USE_DECORD:
        result = get_video_metadata_decord(video_path)
        if result[0] is not None:
            return result
    # Return default values instead of trying PyAV (which can hang)
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
    Load a specific sliding window from video using PyAV.

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
            duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0
            total_frames = int(duration * video_fps) if duration > 0 else 0

        # Calculate frame indices for this window
        frame_indices = set(Any)()
        frame_idx_list = []
        for i in range(window_length):
            frame_idx = window_start_frame + i * stride
            frame_idx = min(max(0, frame_idx), max(0, total_frames - 1))
            frame_indices.add(frame_idx)
            frame_idx_list.append(frame_idx)

        # Decode needed frames with a safety limit to prevent hangs
        # Max frames to decode = highest needed frame + some buffer
        max_frame_to_decode = max(frame_indices) + 100 if frame_indices else 0

        frames_dict = {}
        for frame_idx, frame in enumerate(VideoFrame)(container.decode(video=0)):
            if frame_idx in frame_indices:
                frames_dict[frame_idx] = frame.to_image()
                if len(frames_dict) == len(frame_indices):
                    break
            # Safety limit: stop if we've gone way past the frames we need
            if frame_idx > max_frame_to_decode:
                print(f"Warning: PyAV safety limit reached at frame {frame_idx}, got {len(frames_dict)}/{len(frame_indices)} frames")
                break

        container.close()
        del container

        # Return frames in order (with possible duplicates for clamped indices)
        frames = [frames_dict.get(idx, frames_dict[min(frames_dict.keys())]) for idx in frame_idx_list]
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

    # Check if it's a pixel format error (-22 EINVAL) - PyAV can handle these
    # Note: We tested that PyAV successfully decodes videos with broken pixel format metadata
    # (e.g., pix_fmt=-1 in container but valid yuv420p H.264 stream)
    error_str = str(last_error) if last_error else ""
    is_pixel_format_error = "Cannot create buffer source" in error_str or "filter_graph" in error_str

    if is_pixel_format_error:
        # Fall back to PyAV for pixel format issues (it's more lenient with metadata)
        # PyAV handles these well - tested on IP9GHTdCvWs_E_003942_004037.mp4
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


@register_dataset("finegym_slide")
class FineGymSlideDataset(Dataset):
    """
    FineGym dataset with SLIDING WINDOW approach.

    Key features matching finegym_original (50% mAP):
    - Training: Only include FULLY CONTAINED actions (no truncation)
    - Training: Skip windows with no complete actions
    - Testing: Use overlapping windows with video-level aggregation

    Parameters:
    - window_length: Number of sampled frames per window (default: 32)
    - window_stride: Stride for sliding windows in sampled frames (default: 16 = 50% overlap)
    - sample_stride: Stride for sampling frames from video (default: 16)
    - test_overlap: Whether to use 50% overlap for test (default: True)
    """
    def __init__(
        self,
        is_training,          # if in training mode
        split,                # split, a tuple/list allowing concat of subsets
        backbone_type,
        round,
        use_full,             # use full dataset for training
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
        test_overlap=True,    # Use 50% overlap for test set (for aggregation)
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
        self.data_root = '/home/jupyter/xhu3/video/dataset/finegym'

        # Sliding window parameters (matching original 50% mAP experiment)
        self.window_length = window_length      # 32 sampled frames per window
        self.window_stride = window_stride      # Stride between windows (in sampled frames)
        self.sample_stride = sample_stride      # Sample every 16 original frames
        self.test_overlap = test_overlap        # Use 50% overlap for test
        # feat_stride = sample_stride (16 original frames per sampled frame)
        self.feat_stride = float(sample_stride)

        # Window duration: window_length * sample_stride / fps
        # For 30fps: 32 * 16 / 30 = 17.07 seconds

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes, f"Expected {num_classes} classes, got {len(label_dict)}"
        self.label_dict = label_dict

        # Cache file for video metadata (speeds up subsequent runs)
        cache_suffix = 'train' if is_training else 'val'
        self.cache_file = os.path.join(self.data_root, f'video_metadata_cache_{cache_suffix}.json')

        # Create sliding windows for each video (with caching)
        self.windows = self._create_sliding_windows_cached(dict_db)
        print(f"Created {len(self.windows)} sliding windows from {len(dict_db)} videos")

        # Store original video list for ground truth (used in validation)
        self.video_list = dict_db

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
        for idx, window in enumerate[Any](self.windows):
            video_id = window['video']   # Original video ID (e.g., "MVLZz2J6tcE_E_003701_003784")
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
        """
        # Load existing cache
        metadata_cache = self._load_video_metadata_cache()
        cache_updated = False

        windows = []
        total_videos = len(dict_db)

        for idx, video_item in enumerate[Any](dict_db):
            if idx % 100 == 0:
                print(f"Processing video {idx}/{total_videos}...")

            video_name = video_item['video']
            video_path = video_name
            if not os.path.isabs(video_path):
                video_path = os.path.join(self.data_root, 'videos_Dec14', video_name + '.mp4')

            # Check cache first
            if video_name in metadata_cache:
                video_fps = metadata_cache[video_name]['fps']
                total_frames = metadata_cache[video_name]['total_frames']
                duration = metadata_cache[video_name]['duration']
            else:
                # Get video metadata (slow - opens video file)
                video_fps, total_frames, duration = get_video_metadata(video_path)
                # Update cache
                metadata_cache[video_name] = {
                    'fps': video_fps,
                    'total_frames': total_frames,
                    'duration': duration
                }
                cache_updated = True

            if total_frames == 0:
                print(f"Warning: Could not get metadata for {video_path}")
                continue

            # Create windows for this video
            video_windows = self._create_windows_for_video(
                video_item, video_path, video_fps, total_frames, duration
            )
            windows.extend(video_windows)
            
        # Save updated cache
        if cache_updated:
            self._save_video_metadata_cache(metadata_cache)

        return windows

    def _create_windows_for_video(self, video_item, video_path, video_fps, total_frames, duration):
        """Create sliding windows for a single video."""
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

        # If video is shorter than window_length, create one window covering the whole video
        if total_sampled_frames <= self.window_length:
            window_start_frame = 0
            window_end_time = duration
            window_start_time = 0.0

            segments, labels = self._get_window_segments_complete_only(
                video_item, window_start_time, window_end_time, video_fps
            )

            # For training: only keep if it has complete actions
            # For testing: keep all windows (even empty) for coverage
            if self.is_training:
                if segments is not None and len(segments) > 0:
                    windows.append({
                        'id': video_item['id'],
                        'video': video_item['video'],
                        'video_path': video_path,
                        'window_start_frame': window_start_frame,
                        'window_start_time': window_start_time,
                        'window_end_time': window_end_time,
                        'segments': segments,
                        'labels': labels,
                        'fps': video_fps,
                        'duration': duration,
                        'total_frames': total_frames,
                    })
            else:
                # For test: include all windows, use empty arrays if no segments
                windows.append({
                    'id': f"{video_item['id']}_w0",
                    'video': video_item['video'],
                    'video_path': video_path,
                    'window_start_frame': window_start_frame,
                    'window_start_time': window_start_time,
                    'window_end_time': window_end_time,
                    'segments': segments if segments is not None else np.array([], dtype=np.float32).reshape(0, 2),
                    'labels': labels if labels is not None else np.array([], dtype=np.int64),
                    'fps': video_fps,
                    'duration': duration,
                    'total_frames': total_frames,
                })
        else:
            # Create sliding windows
            window_stride_frames = effective_stride * self.sample_stride
            window_length_frames = self.window_length * self.sample_stride

            window_idx = 0
            window_start_frame = 0

            while window_start_frame < total_frames:
                window_start_time = window_start_frame / video_fps
                window_end_frame = min(window_start_frame + window_length_frames, total_frames)
                window_end_time = window_end_frame / video_fps

                segments, labels = self._get_window_segments_complete_only(
                    video_item, window_start_time, window_end_time, video_fps
                )

                if self.is_training:
                    # Training: only include windows with at least one COMPLETE action
                    if segments is not None and len(segments) > 0:
                        windows.append({
                            'id': f"{video_item['id']}_w{window_idx}",
                            'video': video_item['video'],
                            'video_path': video_path,
                            'window_start_frame': window_start_frame,
                            'window_start_time': window_start_time,
                            'window_end_time': window_end_time,
                            'segments': segments,
                            'labels': labels,
                            'fps': video_fps,
                            'duration': window_end_time - window_start_time,
                            'total_frames': total_frames,
                        })
                else:
                    # Test: include ALL windows for full coverage
                    # Store window offset for later aggregation
                    windows.append({
                        'id': f"{video_item['id']}_w{window_idx}",
                        'video': video_item['video'],
                        'video_path': video_path,
                        'window_start_frame': window_start_frame,
                        'window_start_time': window_start_time,
                        'window_end_time': window_end_time,
                        'segments': segments if segments is not None else np.array([], dtype=np.float32).reshape(0, 2),
                        'labels': labels if labels is not None else np.array([], dtype=np.int64),
                        'fps': video_fps,
                        'duration': window_end_time - window_start_time,
                        'total_frames': total_frames,
                    })

                window_idx += 1
                window_start_frame += window_stride_frames

                # Stop if we've processed the last possible window
                if window_start_frame >= total_frames - self.sample_stride:
                    break

        return windows

    def _get_window_segments_complete_only(self, video_item, window_start_time, window_end_time, video_fps):
        """
        Get ONLY segments that are FULLY CONTAINED within the window.
        This matches finegym_original's approach for clean training.

        Unlike the old _get_window_segments, this does NOT truncate actions.
        Actions that extend beyond window boundaries are EXCLUDED entirely.

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

            # KEY CHANGE: Only include actions that are FULLY CONTAINED
            # Action must start at or after window start AND end at or before window end
            if seg_start >= window_start_time and seg_end <= window_end_time:
                # Make segment relative to window start
                rel_start = seg_start - window_start_time
                rel_end = seg_end - window_start_time

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
        Generate ground truth DataFrame at VIDEO level (not window level).
        Used for evaluation when aggregating predictions across overlapping windows.

        Returns segments in video-relative coordinates (not window-relative).
        Only includes COMPLETE actions (matching training).
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

    def _load_json_db(self, json_file):
        """Load database from JSON Lines format."""
        json_db = {}
        with open(json_file, 'r') as fid:
            for line in fid:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    video_name = entry['video']
                    json_db[video_name] = entry

        # Build label_dict
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                new_value = value.get('new_value', [])
                for activity in new_value:
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
            new_value = value.get('new_value', [])

            segments, labels = [], []
            for activity in new_value:
                for action in activity.get('actions', []):
                    span = action['span']
                    start_time = self._parse_span_time(span[0])
                    end_time = self._parse_span_time(span[1])
                    segments.append([start_time, end_time])

                    action_id = action['action_id']
                    labels.append(label_dict[action_id])

            if segments:
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None

            dict_db += ({
                'id': key,
                'video': video_path,
                'segments': segments,
                'labels': labels,
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Load a sliding window and return data dict.
        """
        window = self.windows[idx]

        # Debug: print which video we're loading (helps identify stuck videos)
        frame_end = window['window_start_frame'] + self.window_length * self.sample_stride

        # Load frames for this window
        try:
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

        # Convert segments (in seconds, relative to window) to frame indices
        if window['segments'] is not None and len(window['segments']) > 0:
            # segments are in seconds relative to window start
            # Convert to sampled frame indices
            # frame_idx = time_seconds * fps / feat_stride
            segments = torch.from_numpy(
                window['segments'] * video_fps / feat_stride - feat_offset
            )
            labels = torch.from_numpy(window['labels'])
        else:
            segments = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Return data dict
        data_dict = {
            'video_id': window['id'],
            'feats': feats,                # C x T x H x W (T = window_length = 32)
            'segments': segments,          # N x 2 (in sampled frame indices)
            'labels': labels,              # N
            'fps': video_fps,
            'duration': window['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames,
            # Additional info for test-time aggregation
            'window_start_time': window['window_start_time'],
            'video_name': window['video'],  # Original video ID for aggregation
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
    2. Converts window-relative predictions to video-relative coordinates
    3. Applies NMS at the video level

    Args:
        all_results: Dict with keys 'video-id', 't-start', 't-end', 'label', 'score'
                    where video-id includes window suffix (e.g., "video_w0", "video_w1")
        nms_func: NMS function (e.g., batched_nms from libs.utils.nms)
        iou_threshold: IoU threshold for NMS
        min_score: Minimum score threshold
        max_seg_num: Maximum number of segments to keep per video

    Returns:
        Aggregated results dict with video-level predictions
    """
    import torch
    from collections import defaultdict

    # Parse window info from video_id
    # Format: "original_video_id_wN" where N is window index
    video_predictions = defaultdict[Any, dict[str, list[Any]]](lambda: {'segs': [], 'scores': [], 'labels': [], 'window_offsets': []})

    # We need window_start_time for each prediction to convert to video coordinates
    # This info should be passed through the results

    # Group predictions by original video
    for i in range(len(all_results['video-id'])):
        vid = all_results['video-id'][i]
        t_start = all_results['t-start'][i] if not isinstance(all_results['t-start'][i], torch.Tensor) else all_results['t-start'][i].item()
        t_end = all_results['t-end'][i] if not isinstance(all_results['t-end'][i], torch.Tensor) else all_results['t-end'][i].item()
        label = all_results['label'][i] if not isinstance(all_results['label'][i], torch.Tensor) else all_results['label'][i].item()
        score = all_results['score'][i] if not isinstance(all_results['score'][i], torch.Tensor) else all_results['score'][i].item()

        # Extract original video ID and window info
        # Format: "video_name_wN"
        if '_w' in vid:
            parts = vid.rsplit('_w', 1)
            original_vid = parts[0]
            # Window offset should be stored somewhere - for now, we'll need to get it from dataset
            # This is a limitation - we need window_start_time to be passed through
        else:
            original_vid = vid

        video_predictions[original_vid]['segs'].append((t_start, t_end))
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