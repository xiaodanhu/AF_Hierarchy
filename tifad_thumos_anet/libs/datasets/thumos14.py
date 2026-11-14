import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from libs.modeling.text import data_split_dir


@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        split_num,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        subset_file,     # data split path
        data_split,      # 50 for 50%-50% setting and 75 for 75%-25% setting
        feature_type="", # feature_type of data
    ):  
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.feature_type = feature_type
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh

        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.subset_file = subset_file
        self.data_split = data_split
        self.mode = 'train' if is_training else 'test'
        self.subset_file = data_split_dir(self.subset_file, self.data_split, self.mode, split_num)

        with open(self.subset_file, 'rt') as f:
            self.subset_labels = [line[:-1] for line in f.readlines()]

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        self.data_list = dict_db
        self.label_dict = label_dict

        # ---- TRAIN_FRAC (TIFAD_TRAIN_FRAC=<f in (0,1]>, TRAIN ONLY; data-
        # efficiency curve): keep a deterministic random
        # fraction f of the TRAINING VIDEOS (video-level subsampling).
        # Stratified so every seen class keeps >= 1 video: the videos are
        # permuted with RandomState(TIFAD_TRAIN_FRAC_SEED, default 0); the
        # first pass walks the seen classes in label order and, for each
        # class not yet covered, takes the first permuted video containing
        # it; the second pass fills up to n_keep = max(round(f*N), 1)
        # videos in permutation order (coverage may exceed n_keep by a few
        # videos at very small f; reported). The kept list keeps the
        # original json order so the per-epoch shuffle / all downstream
        # code (LEAFLOSS cuts, POSEEV ...) is untouched. Byte-inert when
        # unset (identical data_list object) and at eval time.
        _tf_env = os.environ.get('TIFAD_TRAIN_FRAC', '')
        if _tf_env and self.is_training:
            _tf_f = float(_tf_env)
            assert 0.0 < _tf_f <= 1.0, f"TIFAD_TRAIN_FRAC={_tf_env} not in (0,1]"
            _tf_seed = int(os.environ.get('TIFAD_TRAIN_FRAC_SEED', '0') or 0)
            _tf_N = len(self.data_list)
            _tf_keep_n = max(int(round(_tf_f * _tf_N)), 1)
            _tf_perm = np.random.RandomState(_tf_seed).permutation(_tf_N)
            _tf_keep = set()
            _tf_covered = set()
            _tf_n_cls = len(self.label_dict)
            for _tf_c in range(_tf_n_cls):
                if _tf_c in _tf_covered:
                    continue
                for _tf_i in _tf_perm:
                    if _tf_c in set(self.data_list[_tf_i]['labels'].tolist()):
                        _tf_keep.add(int(_tf_i))
                        _tf_covered |= set(
                            self.data_list[_tf_i]['labels'].tolist())
                        break
            _tf_n_strat = len(_tf_keep)
            for _tf_i in _tf_perm:
                if len(_tf_keep) >= _tf_keep_n:
                    break
                _tf_keep.add(int(_tf_i))
            _tf_full = self.data_list
            self.data_list = tuple(
                _tf_full[_tf_i] for _tf_i in range(_tf_N) if _tf_i in _tf_keep)
            _tf_id2name = {v: k for k, v in self.label_dict.items()}
            _tf_vid_per_cls = {c: 0 for c in range(_tf_n_cls)}
            _tf_seg_per_cls = {c: 0 for c in range(_tf_n_cls)}
            _tf_vid_full = {c: 0 for c in range(_tf_n_cls)}
            _tf_seg_full = {c: 0 for c in range(_tf_n_cls)}
            for _tf_src, _tf_v, _tf_s in ((_tf_full, _tf_vid_full, _tf_seg_full),
                                          (self.data_list, _tf_vid_per_cls,
                                           _tf_seg_per_cls)):
                for _tf_item in _tf_src:
                    _tf_labs = _tf_item['labels'].tolist()
                    for _tf_c in set(_tf_labs):
                        _tf_v[_tf_c] += 1
                    for _tf_c in _tf_labs:
                        _tf_s[_tf_c] += 1
            _tf_segs_kept = sum(len(v['labels']) for v in self.data_list)
            _tf_segs_full = sum(len(v['labels']) for v in _tf_full)
            assert all(_tf_vid_per_cls[c] >= 1 for c in range(_tf_n_cls)), \
                "[TRAIN-FRAC] a seen class lost all its videos"
            print(f"[TRAIN-FRAC] f={_tf_f} seed={_tf_seed}: kept "
                  f"{len(self.data_list)}/{_tf_N} training videos "
                  f"(target {_tf_keep_n}, {_tf_n_strat} from the class-"
                  f"coverage pass), {_tf_segs_kept}/{_tf_segs_full} GT "
                  f"segments; per-class videos (segments) kept/full: "
                  + ', '.join(
                      f"{_tf_id2name[c]} {_tf_vid_per_cls[c]}/{_tf_vid_full[c]} "
                      f"({_tf_seg_per_cls[c]}/{_tf_seg_full[c]})"
                      for c in range(_tf_n_cls)), flush=True)

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

        # ---- TIFAD_CAT_SCALE=auto pre-compute (POSEEV counter lesson:
        # dataloader workers each fork a PRIVATE dataset copy, so a lazy
        # per-worker cache would yield a DIFFERENT scale per worker).
        # Compute the ratio mean_rowL2(base)/mean_rowL2(extra) HERE, in
        # the main process from the first video of the FIRST dataset
        # built in this process (train in train_eval.py), cache it
        # PROCESS-GLOBALLY keyed by (feat_folder, cat_dir), and reuse it
        # for every later dataset (val) -- so train and eval apply the
        # SAME scale and every worker inherits one value through the
        # fork (printed once per process). Byte-inert when
        # TIFAD_CAT_FEATS_DIR / TIFAD_CAT_SCALE unset.
        self._cat_scale_auto = None
        _cs_dir = os.environ.get('TIFAD_CAT_FEATS_DIR', '')
        if _cs_dir and os.environ.get('TIFAD_CAT_SCALE', '') == 'auto' \
                and len(self.data_list) > 0:
            import sys as _cs_sys
            _cs_mod = _cs_sys.modules[__name__]
            _cs_cache = getattr(_cs_mod, '_CAT_SCALE_CACHE', None)
            if _cs_cache is None:
                _cs_cache = {}
                setattr(_cs_mod, '_CAT_SCALE_CACHE', _cs_cache)
            _cs_key = (self.feat_folder, _cs_dir)
            if _cs_key not in _cs_cache:
                _cs_id = self.data_list[0]['id']
                _cs_base = np.load(os.path.join(
                    self.feat_folder, self.file_prefix + _cs_id
                    + self.feature_type + self.file_ext)).astype(np.float32)
                _cs_extra = np.load(os.path.join(
                    _cs_dir, _cs_id + '.npy')).astype(np.float32)
                _cs_cache[_cs_key] = (
                    float(np.linalg.norm(_cs_base, axis=1).mean())
                    / max(float(np.linalg.norm(
                        _cs_extra, axis=1).mean()), 1e-8))
                print(f"[CAT-SCALE] auto scale = "
                      f"{_cs_cache[_cs_key]:.4f} (mean rowL2 base/extra, "
                      f"video {_cs_id}, {self.mode})", flush=True)
            self._cat_scale_auto = _cs_cache[_cs_key]

        # ---- POSEEV (TIFAD_POSEEV=1): also load per-video pose-kinematics
        # arrays (2D + 3D, concat to (T_grid, 234)) aligned to the I3D grid
        # and return them as item['pose_feats'] (234 x T) -- the model's
        # pose evidence branch consumes them (meta_archs.py [POSEEV]).
        # Byte-inert when unset (no extra key, no I/O).
        # Env: TIFAD_POSEEV=1,
        #      TIFAD_POSEEV_DIR2D (default
        #        /data3/xiaodan8/thumos/pose_features_grid, (T_grid, 110)),
        #      TIFAD_POSEEV_DIR3D (default
        #        /data3/xiaodan8/thumos/pose3d_features_grid, (T_grid, 124)).
        # Missing / malformed files (extraction still running) fall back to
        # zeros with a warning counter (first 5 printed). Coverage is
        # counted HERE (deterministic, main process, once per split --
        # dataloader workers each hold a private copy, so an in-epoch
        # counter would not print reliably).
        self.poseev = os.environ.get('TIFAD_POSEEV', '') == '1'
        if self.poseev:
            self.poseev_dir2d = os.environ.get(
                'TIFAD_POSEEV_DIR2D',
                '/data3/xiaodan8/thumos/pose_features_grid')
            self.poseev_dir3d = os.environ.get(
                'TIFAD_POSEEV_DIR3D',
                '/data3/xiaodan8/thumos/pose3d_features_grid')
            self._poseev_dims = (110, 124)
            self._poseev_warn = 0
            _pe_found = sum(
                1 for v in self.data_list
                if os.path.exists(os.path.join(
                    self.poseev_dir2d, v['id'] + '.npy'))
                and os.path.exists(os.path.join(
                    self.poseev_dir3d, v['id'] + '.npy')))
            print(f"[POSEEV] pose features: {_pe_found}/"
                  f"{len(self.data_list)} videos ({self.mode})", flush=True)

        # ---- PHASEHEAD (TIFAD_PHASEHEAD=1): Stage-B detached phase
        # predictor DATA side. Two extra per-item tensors:
        #   item['phase_feats']   (768 x T)  I3D-grid-aligned VideoMAE
        #                                    features (vmae_grid), same
        #                                    slicing mirror as pose_feats;
        #   item['phase_targets'] (T,) long  per-grid-step phase label,
        #                                    -1 = outside any usable
        #                                    segment, else 0..9.
        # Targets come from the Stage-A pseudo-label JSON (per-snippet
        # phase labels on the VMAE-s2 grid, stride 2 frames = 2x the I3D
        # grid, centers exactly aligned: s2 snippet 2i and I3D step i both
        # center on frame 4i+8). Downsample s2 -> I3D per pair (2i, 2i+1):
        # equal -> that label; disagree -> the exactly-aligned s2[2i]
        # unless it is unlabeled (-1) and s2[2i+1] is labeled (half-step
        # tolerance keeps segment-edge coverage). EXCLUDED from training
        # targets (left at -1): fallback segments (proportional labels for
        # <10-snippet segments) and the K1-failing weak five classes (the
        # per-class 10-way transfer table is not stored in the JSON, so
        # the <0.20 set is hardcoded per the Stage-A report). Byte-inert
        # when TIFAD_PHASEHEAD unset (no extra keys, no I/O).
        # Env: TIFAD_PHASEHEAD=1,
        #      TIFAD_PHASEHEAD_FEATS_DIR (default
        #        /data3/xiaodan8/thumos/vmae_grid, (T_i3d, 768)),
        #      TIFAD_PHASEHEAD_LABELS (default
        #        data/phase_pseudo/thumos14_75_25_split{n}.json; {n} is
        #        filled with this dataset's split_num).
        self.phasehead = os.environ.get('TIFAD_PHASEHEAD', '') == '1'
        if self.phasehead:
            self.phasehead_dir = os.environ.get(
                'TIFAD_PHASEHEAD_FEATS_DIR',
                '/data3/xiaodan8/thumos/vmae_grid')
            _ph_path = os.environ.get(
                'TIFAD_PHASEHEAD_LABELS',
                'data/phase_pseudo/thumos14_75_25_split{n}.json'
            ).format(n=split_num)
            self._phasehead_dim = 768
            self._phasehead_warn = 0
            # K1-failing classes (per-class 10-way transfer < 0.20)
            self._phasehead_excluded = {
                'Billiards', 'TennisSwing', 'CricketShot', 'Shotput',
                'SoccerPenalty'}
            with open(_ph_path, 'r') as _ph_f:
                _ph_json = json.load(_ph_f)
            self._phasehead_segs = {}
            _ph_ok = _ph_fb = _ph_ex = 0
            for _ph_vid, _ph_ss in _ph_json['videos'].items():
                _ph_rows = []
                for _ph_s in _ph_ss:
                    if _ph_s['label'] in self._phasehead_excluded:
                        _ph_ex += 1
                        continue
                    if _ph_s.get('fallback', False):
                        _ph_fb += 1
                        continue
                    _ph_rows.append((int(_ph_s['snippet_idx'][0]),
                                     [int(_p) for _p in _ph_s['phase']]))
                    _ph_ok += 1
                if _ph_rows:
                    self._phasehead_segs[_ph_vid] = _ph_rows
            _ph_cov = sum(1 for v in self.data_list
                          if v['id'] in self._phasehead_segs)
            _ph_ff = sum(1 for v in self.data_list
                         if os.path.exists(os.path.join(
                             self.phasehead_dir, v['id'] + '.npy')))
            print(f"[PHASEHEAD] labels {_ph_path}: {_ph_ok} usable segs "
                  f"(excluded {_ph_ex} weak-class, {_ph_fb} fallback); "
                  f"coverage {_ph_cov}/{len(self.data_list)} videos, "
                  f"vmae_grid feats {_ph_ff}/{len(self.data_list)} "
                  f"({self.mode})", flush=True)

        # ---- LEAFLOSS (TIFAD_LEAFLOSS=1, TRAIN ONLY): leaf-level (temporal
        # phase) supervision of the detection trunk. For every GT segment of
        # a training video, look up its Stage-A phase pseudo-label entry
        # (matched on video id + segment seconds, tol 1e-3, + class name)
        # and keep the 9 INTERNAL phase cut points (end of phases 0..8, in
        # seconds; the 10 ordered pseudo-intervals partition the segment).
        # Fallback entries (proportional labels, <10 snippets; some of their
        # `phase_boundaries` are None) get proportional cuts s+(e-s)*i/10.
        # Segments without an entry get NaN cuts (-> ignored, target -1).
        # __getitem__ converts the cuts to the feature grid EXACTLY like
        # `segments` and adds item['leaf_cuts'] (N x 9); truncate_feats
        # filters / shifts them with the segments. Byte-inert when unset.
        # Env: TIFAD_LEAFLOSS=1,
        #      TIFAD_LEAFLOSS_JSON (default
        #        data/phase_pseudo/thumos14_75_25_split{n}.json; {n} is
        #        filled with this dataset's split_num).
        # (also loaded under TIFAD_PH_SRC=head -- the v5 phase head trains
        # on the same cuts, libs/modeling/phaseparse.py)
        self.leafloss = ((os.environ.get('TIFAD_LEAFLOSS', '') == '1'
                          or os.environ.get('TIFAD_PH_SRC', '') == 'head')
                         and self.is_training)
        if self.leafloss:
            _lf_path = os.environ.get(
                'TIFAD_LEAFLOSS_JSON',
                'data/phase_pseudo/thumos14_75_25_split{n}.json'
            ).format(n=split_num)
            with open(_lf_path, 'r') as _lf_f:
                _lf_json = json.load(_lf_f)
            _lf_by_vid = {}
            for _lf_vid, _lf_ss in _lf_json['videos'].items():
                _lf_by_vid[_lf_vid] = [
                    (_s['label'], float(_s['segment'][0]),
                     float(_s['segment'][1]), _s.get('phase_boundaries'),
                     bool(_s.get('fallback', False)))
                    for _s in _lf_ss]
            _lf_id2name = {v: k for k, v in self.label_dict.items()}
            self._leaf_cuts = {}          # video id -> (N, 9) float32 sec
            _lf_n = _lf_hit = _lf_fb = 0
            for _lf_v in self.data_list:
                _segs, _labs = _lf_v['segments'], _lf_v['labels']
                _cuts = np.full((len(_segs), 9), np.nan, dtype=np.float32)
                for _i, ((_s0, _e0), _l) in enumerate(zip(_segs, _labs)):
                    _lf_n += 1
                    for (_nm, _ps, _pe, _pb, _fb) in _lf_by_vid.get(
                            _lf_v['id'], []):
                        if (abs(_ps - float(_s0)) > 1e-3
                                or abs(_pe - float(_e0)) > 1e-3
                                or _nm != _lf_id2name[int(_l)]):
                            continue
                        if (_pb is None or len(_pb) != 10
                                or any(_b is None for _b in _pb)):
                            # proportional 10-way split (fallback entries)
                            _cuts[_i] = [_ps + (_pe - _ps) * _k / 10.0
                                         for _k in range(1, 10)]
                        else:
                            _cuts[_i] = [float(_pb[_k][1])
                                         for _k in range(9)]
                        _lf_hit += 1
                        _lf_fb += int(_fb)
                        break
                self._leaf_cuts[_lf_v['id']] = _cuts
            print(f"[LEAFLOSS] pseudo phases {_lf_path}: matched "
                  f"{_lf_hit}/{_lf_n} GT segments ({_lf_fb} fallback -> "
                  f"proportional cuts) over {len(self.data_list)} videos "
                  f"({self.mode})", flush=True)


    def get_attributes(self):
        return self.db_attributes

    def _poseev_load_one(self, folder, vid, dim, t_raw):
        """Load one pose-kinematics array (T, dim) and align it to the RAW
        (pre-downsample) I3D feature length t_raw: truncate if longer,
        zero-pad if shorter. Missing / malformed file -> zeros (warned,
        counted). NaN/Inf sanitized to 0 (0-gated scoring path must stay
        exact at init)."""
        path = os.path.join(folder, vid + '.npy')
        arr = None
        if os.path.exists(path):
            try:
                arr = np.load(path).astype(np.float32)
                if arr.ndim != 2 or arr.shape[1] != dim:
                    raise ValueError(f"bad shape {arr.shape}, want (*, {dim})")
            except Exception as e:
                arr = None
                self._poseev_warn += 1
                if self._poseev_warn <= 5:
                    print(f"[POSEEV] WARNING ({self._poseev_warn}): "
                          f"unreadable {path} ({e}) -> zeros", flush=True)
        else:
            self._poseev_warn += 1
            if self._poseev_warn <= 5:
                print(f"[POSEEV] WARNING ({self._poseev_warn}): missing "
                      f"{path} -> zeros", flush=True)
        if arr is None:
            return np.zeros((t_raw, dim), dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] >= t_raw:
            return arr[:t_raw]
        pad = np.zeros((t_raw - arr.shape[0], dim), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def _phasehead_load_feats(self, vid, t_raw):
        """PHASEHEAD: load the vmae_grid array (T, 768) for one video and
        align it to the RAW (pre-downsample) I3D feature length t_raw:
        truncate if longer, zero-pad if shorter (POSEEV loader pattern).
        Missing / malformed file -> zeros (warned, counted). NaN/Inf
        sanitized to 0."""
        path = os.path.join(self.phasehead_dir, vid + '.npy')
        arr = None
        if os.path.exists(path):
            try:
                arr = np.load(path).astype(np.float32)
                if arr.ndim != 2 or arr.shape[1] != self._phasehead_dim:
                    raise ValueError(
                        f"bad shape {arr.shape}, want "
                        f"(*, {self._phasehead_dim})")
            except Exception as e:
                arr = None
                self._phasehead_warn += 1
                if self._phasehead_warn <= 5:
                    print(f"[PHASEHEAD] WARNING ({self._phasehead_warn}): "
                          f"unreadable {path} ({e}) -> zeros", flush=True)
        else:
            self._phasehead_warn += 1
            if self._phasehead_warn <= 5:
                print(f"[PHASEHEAD] WARNING ({self._phasehead_warn}): "
                      f"missing {path} -> zeros", flush=True)
        if arr is None:
            return np.zeros((t_raw, self._phasehead_dim), dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] >= t_raw:
            return arr[:t_raw]
        pad = np.zeros((t_raw - arr.shape[0], self._phasehead_dim),
                       dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def _phasehead_targets(self, vid, t_raw):
        """PHASEHEAD: build the (t_raw,) int64 per-I3D-grid-step phase
        target vector from the s2-grid pseudo labels (see __init__ doc
        for the exact 2:1 downsampling rule). -1 everywhere no usable
        label exists."""
        s2 = np.full(2 * t_raw, -1, dtype=np.int64)
        for start, phases in self._phasehead_segs.get(vid, []):
            if start >= s2.size:
                continue
            end = min(start + len(phases), s2.size)
            s2[start:end] = phases[:end - start]
        a = s2[0::2]                       # exactly-aligned s2 snippet 2i
        b = s2[1::2]                       # half-step-later snippet 2i+1
        return np.where((a < 0) & (b >= 0), b, a)

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        if self.label_dict is None:
            label_dict = {}
            for idx, label in enumerate(self.subset_labels):
                label_dict[label] = idx

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.feature_type + self.file_ext)

            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case

                segments, labels = [], []
                for act in value['annotations']:
                    if act['label'] not in label_dict.keys():
                        continue
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                if len(segments) == 0:
                    segments = None
                    labels = None
                else:
                    segments = np.asarray(segments, dtype=np.float32)
                    labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            
            if segments is not None:   
                dict_db += ({'id': key,
                            'fps' : fps,
                            'duration' : duration,
                            'segments' : segments,
                            'labels' : labels
                }, )
        
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.feature_type + self.file_ext)
        if "npy" in self.file_ext:
            feats = np.load(filename).astype(np.float32)
        elif "pt" in self.file_ext:
            feats = torch.load(filename)
            feats = feats.numpy().astype(np.float32)
        else:
            raise NotImplementedError

        # ---- TIFAD_CAT_FEATS_DIR: concat a second feature set (same grid,
        # e.g. vmae_grid 768-d next to i3d 2048-d). input_dim in the config
        # must equal the summed dim. Length mismatch |dT|<=2 tolerated by
        # truncation to the shorter. Inert when env unset.
        import os as _oscf
        _catdir = _oscf.environ.get('TIFAD_CAT_FEATS_DIR', '')
        if _catdir:
            _cf = os.path.join(_catdir, video_item['id'] + '.npy')
            extra = np.load(_cf).astype(np.float32)
            # ---- TIFAD_CAT_SCALE (<float|'auto'>): rescale the concat
            # block to fix the feature-scale clash between the two
            # feature sets. A float multiplies `extra` by it directly;
            # 'auto' uses the single ratio mean_rowL2(base)/
            # mean_rowL2(extra) precomputed in __init__ from the split's
            # first video (one value for ALL videos and ALL dataloader
            # workers; lazy fallback kept for safety). Inert when unset.
            _csc = _oscf.environ.get('TIFAD_CAT_SCALE', '')
            if _csc:
                if _csc == 'auto':
                    if getattr(self, '_cat_scale_auto', None) is None:
                        self._cat_scale_auto = (
                            float(np.linalg.norm(feats, axis=1).mean())
                            / max(float(np.linalg.norm(
                                extra, axis=1).mean()), 1e-8))
                        print("[CAT-SCALE] auto scale (lazy fallback) = "
                              f"{self._cat_scale_auto:.4f} (mean rowL2 "
                              f"base/extra, video {video_item['id']})",
                              flush=True)
                    extra = extra * np.float32(self._cat_scale_auto)
                else:
                    extra = extra * np.float32(float(_csc))
            dT = feats.shape[0] - extra.shape[0]
            assert abs(dT) <= 2, (filename, feats.shape, extra.shape)
            T = min(feats.shape[0], extra.shape[0])
            feats = np.concatenate([feats[:T], extra[:T]], axis=1)

        # ---- POSEEV: load + concat the pose arrays at the RAW I3D length
        # (before downsampling), then apply EXACTLY the same slicing as
        # feats below so the grids stay aligned. Inert when unset.
        if self.poseev:
            pose = np.concatenate([
                self._poseev_load_one(self.poseev_dir2d, video_item['id'],
                                      self._poseev_dims[0], feats.shape[0]),
                self._poseev_load_one(self.poseev_dir3d, video_item['id'],
                                      self._poseev_dims[1], feats.shape[0]),
            ], axis=1)                                     # (T_raw, 234)

        # ---- PHASEHEAD: load vmae_grid feats + build phase targets at
        # the RAW I3D length, then apply EXACTLY the same slicing as feats
        # below so the grids stay aligned. Inert when unset.
        if self.phasehead:
            phf = self._phasehead_load_feats(
                video_item['id'], feats.shape[0])          # (T_raw, 768)
            pht = self._phasehead_targets(
                video_item['id'], feats.shape[0])          # (T_raw,)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        if self.poseev:
            # SAME downsampling as feats (grid alignment invariant)
            pose = pose[::self.downsample_rate, :]
            pose = torch.from_numpy(np.ascontiguousarray(pose.transpose()))
        if self.phasehead:
            # SAME downsampling as feats (grid alignment invariant)
            phf = phf[::self.downsample_rate, :]
            phf = torch.from_numpy(np.ascontiguousarray(phf.transpose()))
            pht = torch.from_numpy(
                np.ascontiguousarray(pht[::self.downsample_rate]))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}
        if self.poseev:
            # 234 x T, same grid as feats; truncate_feats slices it with
            # the SAME [st:ed] window (data_utils.py [POSEEV])
            data_dict['pose_feats'] = pose
        if self.phasehead:
            # 768 x T and (T,), same grid as feats; truncate_feats slices
            # both with the SAME [st:ed] window (data_utils.py [PHASEHEAD])
            data_dict['phase_feats'] = phf
            data_dict['phase_targets'] = pht
        if self.leafloss and segments is not None:
            # (N, 9) phase cut points on the SAME feature grid as segments
            # (identical seconds -> grid conversion); NaN rows = no pseudo
            # entry. truncate_feats filters / shifts them with the segments
            # (data_utils.py [LEAFLOSS]).
            data_dict['leaf_cuts'] = torch.from_numpy(
                (self._leaf_cuts[video_item['id']] * video_item['fps']
                 - 0.5 * self.num_frames) / feat_stride
            ).to(torch.float32)

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )
        
        return data_dict