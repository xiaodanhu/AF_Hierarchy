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

# Suppress FFmpeg/libav filter warnings
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

try:
    from decord import VideoReader, cpu
    import decord
    decord.bridge.set_bridge('native')
    decord.logging.set_level(decord.logging.ERROR)
    USE_DECORD = True
except ImportError:
    USE_DECORD = False


from .datasets import register_dataset
from .data_utils import truncate_feats, get_transforms, truncate_video

# Reuse video loading utilities from finegym_slide
from .finegym_slide import (
    suppress_stderr,
    VideoLoadTimeout,
    timeout_context,
    get_video_metadata,
    load_sliding_window,
    load_sliding_window_jpg,
    get_frame_path,
    find_video_file,
    aggregate_window_predictions,
)


def extract_video_name(clip_name):
    """Extract untrimmed video name from clip name.

    Clip names are formatted as: {video_name}_{clip_id}
    e.g., 'FINAWorldChampionships2019_Men10mSynchronised_final_r4_1' -> 'FINAWorldChampionships2019_Men10mSynchronised_final_r4'
    or '01_5' -> '01'

    The clip_id is always the last underscore-separated token (a number).
    """
    parts = clip_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return clip_name


@register_dataset("finediving_slide")
class FineDivingSlideDataset(Dataset):
    """
    FineDiving dataset with SLIDING WINDOW approach.
    Mirrors FineGymSlideDataset but adapted for FineDiving's annotation format.

    FineDiving has 2 levels (vs FineGym's 3):
        - action_type (dive code like "207c") → activity_id (a00, a01, ...)
        - sub_action (per-frame label 1-42)  → action_id  (c00, c01, ...)

    Videos are untrimmed competition videos. Each clip is a single dive
    identified by (video_name, clip_id) in the coarse annotation.

    Parameters match FineGymSlideDataset for compatibility.
    """
    def __init__(
        self,
        is_training,
        split,
        backbone_type,
        round,
        train_json_file,
        val_json_file,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        num_classes,
        num_frames=1,
        video_min_frames=32,
        video_max_frames=128,
        window_length=32,
        window_stride=16,
        sample_stride=16,
        test_overlap=True,
        use_raw_video=True,        # FineDiving always uses raw (untrimmed) videos
        load_jpg=False,
        **kwargs
    ):

        assert isinstance(split, tuple) or isinstance(split, list)
        if crop_ratio is not None:
            assert crop_ratio == 'None' or len(crop_ratio) == 2

        if is_training:
            self.json_file = train_json_file
        else:
            self.json_file = val_json_file

        if self.json_file is None:
            raise ValueError("No json_file specified for dataset. "
                             "Provide train_json_file/val_json_file.")

        self.split = split
        self.is_training = is_training
        self.num_frames = num_frames
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = None if (crop_ratio is None or crop_ratio == 'None') else crop_ratio
        self.backbone_type = backbone_type
        self.round = round

        # FineDiving data root
        self.data_root = '/data3/xiaodan8/FineDiving'

        self.use_raw_video = use_raw_video
        self.video_dir = os.path.join(
            self.data_root, 'raw/FineDiving/Released_FineDiving_Dataset/Untrimmed_Videos'
        )
        print(f"[FineDivingSlide] Using untrimmed videos from {self.video_dir}")
        print(f"[FineDivingSlide] Annotations: {'raw_value' if use_raw_video else 'new_value'}")

        # JPG frame caching
        self.load_jpg = load_jpg
        self.rgb_root = os.path.join(self.data_root, 'RGB')
        if self.load_jpg:
            print(f"[FineDivingSlide] JPG caching ENABLED: {self.rgb_root}")
            os.makedirs(self.rgb_root, exist_ok=True)

        # Sliding window parameters
        self.window_length = window_length
        self.window_stride = window_stride
        self.sample_stride = sample_stride
        self.test_overlap = test_overlap
        self.feat_stride = float(sample_stride)

        # Load database
        dict_db, label_dict = self._load_json_db(self.json_file)
        # Not all sub-action types may appear in annotations (29 of 42 used)
        # but num_classes should cover the max label index + 1
        if len(label_dict) > num_classes:
            raise ValueError(f"Found {len(label_dict)} classes but num_classes={num_classes}")
        print(f"[FineDivingSlide] {len(label_dict)} action classes in annotations, "
              f"num_classes={num_classes}")
        self.label_dict = label_dict

        # Metadata cache
        cache_suffix = 'train' if is_training else 'test'
        self.cache_file = os.path.join(
            self.data_root, f'video_metadata_cache_{cache_suffix}.json'
        )

        # Create sliding windows
        self.windows, video_items = self._create_sliding_windows_cached(dict_db)
        print(f"Created {len(self.windows)} sliding windows from {len(video_items)} clips")

        self.video_list = video_items

        if not is_training:
            self._build_video_to_windows_map()

        self.db_attributes = {
            'dataset_name': 'finediving_slide',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }
        self.transform = get_transforms(is_training, 224)

    def _build_video_to_windows_map(self):
        self.video_to_windows = {}
        for idx, window in enumerate(self.windows):
            video_id = window['video']
            if video_id not in self.video_to_windows:
                self.video_to_windows[video_id] = []
            self.video_to_windows[video_id].append(idx)

    def _load_video_metadata_cache(self):
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
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f)
            print(f"Saved video metadata cache to {self.cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _create_sliding_windows_cached(self, dict_db):
        metadata_cache = self._load_video_metadata_cache()
        cache_updated = False

        windows = []
        skipped_videos = 0
        video_items = list(dict_db)
        total_videos = len(video_items)

        for idx, video_item in enumerate(video_items):
            if idx % 100 == 0:
                print(f"Processing clip {idx}/{total_videos}... (skipped: {skipped_videos})")

            clip_name = video_item['video']
            video_name = extract_video_name(clip_name)

            # Find the untrimmed video file
            video_path = find_video_file(self.video_dir, video_name)
            cache_key = video_name

            if cache_key in metadata_cache:
                video_fps = metadata_cache[cache_key]['fps']
                raw_total_frames = metadata_cache[cache_key]['total_frames']
                raw_duration = metadata_cache[cache_key]['duration']
            else:
                video_fps, raw_total_frames, raw_duration = get_video_metadata(video_path)
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

            total_frames = raw_total_frames
            duration = raw_duration

            # Validate instance span
            instance_span_start = video_item.get('instance_span_start')
            instance_span_end = video_item.get('instance_span_end')

            if instance_span_start is None or instance_span_end is None:
                if video_item['segments'] is not None and len(video_item['segments']) > 0:
                    video_item['instance_span_start'] = float(video_item['segments'][:, 0].min())
                    video_item['instance_span_end'] = float(video_item['segments'][:, 1].max())
                else:
                    print(f"Warning: No span info for {clip_name}, skipping")
                    skipped_videos += 1
                    continue

            video_windows = self._create_windows_for_video(
                video_item, video_path, video_fps, total_frames, duration, video_name
            )
            windows.extend(video_windows)

        if cache_updated:
            self._save_video_metadata_cache(metadata_cache)

        if skipped_videos > 0:
            print(f"Warning: Skipped {skipped_videos}/{total_videos} clips due to metadata issues")

        return windows, video_items

    def _create_windows_for_video(self, video_item, video_path, video_fps,
                                  total_frames, duration, video_name=None):

        windows = []

        total_sampled_frames = total_frames // self.sample_stride

        if self.is_training:
            effective_stride = self.window_stride
        else:
            effective_stride = self.window_length // 2 if self.test_overlap else self.window_length

        window_span_frames = (self.window_length - 1) * self.sample_stride
        window_duration = self.window_length * self.sample_stride / video_fps
        window_stride_frames = effective_stride * self.sample_stride

        # Slide over instance span (raw video coordinates)
        instance_span_start = video_item.get('instance_span_start')
        instance_span_end = video_item.get('instance_span_end')

        if instance_span_start is not None and instance_span_end is not None:
            slide_start_frame = max(0, int(instance_span_start * video_fps))
            slide_end_frame = min(total_frames, int(instance_span_end * video_fps))
        elif video_item['segments'] is not None and len(video_item['segments']) > 0:
            min_seg_time = float(video_item['segments'][:, 0].min())
            max_seg_time = float(video_item['segments'][:, 1].max())
            slide_start_frame = max(0, int(min_seg_time * video_fps))
            slide_end_frame = min(total_frames, int(max_seg_time * video_fps))
        else:
            if self.is_training:
                return windows
            else:
                slide_start_frame = 0
                slide_end_frame = min(window_span_frames, total_frames)

        if total_sampled_frames <= self.window_length:
            window_start_frame = slide_start_frame
            window_start_time = window_start_frame / video_fps
            window_end_time = min(window_start_time + window_span_frames / video_fps, duration)

            segments, labels = self._get_window_segments_complete_only(
                video_item, window_start_time, window_end_time, video_fps
            )

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
                        'duration': window_duration,
                        'total_frames': total_frames,
                        'video_name': video_name,
                    })
            else:
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
                    'duration': window_duration,
                    'total_frames': total_frames,
                    'video_name': video_name,
                })
        else:
            window_idx = 0
            window_start_frame = slide_start_frame

            while window_start_frame < slide_end_frame:
                window_start_time = window_start_frame / video_fps
                window_end_frame_for_filter = min(window_start_frame + window_span_frames, total_frames)
                window_end_time = window_end_frame_for_filter / video_fps

                segments, labels = self._get_window_segments_complete_only(
                    video_item, window_start_time, window_end_time, video_fps
                )

                if self.is_training:
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
                            'duration': window_duration,
                            'total_frames': total_frames,
                            'video_name': video_name,
                        })

                else:
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
                        'duration': window_duration,
                        'total_frames': total_frames,
                        'video_name': video_name,
                    })

                window_idx += 1
                window_start_frame += window_stride_frames

                if window_start_frame >= slide_end_frame - self.sample_stride:
                    break

        return windows

    def _get_window_segments_complete_only(self, video_item, window_start_time, window_end_time, video_fps):
        if video_item['segments'] is None:
            return None, None

        segments = []
        labels = []

        for seg, label in zip(video_item['segments'], video_item['labels']):
            seg_start, seg_end = seg[0], seg[1]
            if seg_start >= window_start_time and seg_end <= window_end_time:
                rel_start = seg_start - window_start_time
                rel_end = seg_end - window_start_time
                segments.append([rel_start, rel_end])
                labels.append(label)

        if len(segments) > 0:
            return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)
        return None, None

    def get_attributes(self):
        return self.db_attributes

    def get_ground_truth_df(self):
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
        import pandas as pd
        vids, starts, stops, labels = [], [], [], []

        for video_item in self.video_list:
            if video_item['segments'] is not None:
                for i in range(len(video_item['segments'])):
                    vids.append(video_item['video'])
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
        return float(span_str.strip('<>').replace(' seconds', ''))

    def _load_json_db(self, json_file):
        """Load database from JSONL annotation file.

        Uses 'raw_value' when use_raw_video=True, else 'new_value'.
        FineDiving format: no phrase_id, just activity_id and action_id.
        """
        json_db = {}
        with open(json_file, 'r') as fid:
            for line in fid:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    video_name = entry['video']
                    json_db[video_name] = entry

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

            dict_db += ({
                'id': key,
                'video': video_path,
                'segments': segments,
                'labels': labels,
                'instance_span_start': instance_span_start,
                'instance_span_end': instance_span_end,
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        try:
            if self.load_jpg and window.get('video_name') is not None:
                frames, video_fps = load_sliding_window_jpg(
                    self.rgb_root,
                    window['video_name'],
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
            dummy_frame = Image.new('RGB', (224, 224), (0, 0, 0))
            frames = [dummy_frame] * self.window_length
            video_fps = 25.0

        if len(frames) < self.window_length:
            while len(frames) < self.window_length:
                frames.append(frames[-1])

        feats = torch.stack([self.transform(frm) for frm in frames])
        del frames

        # T x C x H x W -> C x T x H x W
        feats = feats.permute(1, 0, 2, 3).contiguous()

        feat_stride = self.feat_stride
        feat_offset = 0.5 * self.num_frames / feat_stride

        if window['segments'] is not None and len(window['segments']) > 0:
            window_fps = window['fps']
            segments = torch.from_numpy(
                window['segments'] * window_fps / feat_stride - feat_offset
            )
            labels = torch.from_numpy(window['labels'])
        else:
            segments = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        data_dict = {
            'video_id': window['id'],
            'feats': feats,
            'segments': segments,
            'labels': labels,
            'fps': window['fps'],
            'duration': window['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames,
            'window_start_time': window['window_start_time'],
            'video_name': window['video'],
        }

        if self.is_training and (segments is not None) and len(segments) > 0:
            if feats.shape[1] > self.max_seq_len:
                data_dict = truncate_feats(
                    data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
                )

        if self.is_training:
            if data_dict['segments'] is None or len(data_dict['segments']) == 0:
                return self.__getitem__((idx + 1) % len(self))

        return data_dict
