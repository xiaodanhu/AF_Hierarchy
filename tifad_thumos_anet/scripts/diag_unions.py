"""Diagnostic: does same-class union stitching create detections that
cover GTs which no baseline detection covers at tIoU>=0.5/0.7?
Per class: count GTs whose best baseline det IoU < thr but whose best
union IoU >= thr (the recoverable pool), and the union's rank context."""
import os
import pickle
import sys

sys.path.insert(0, '/data3/xiaodan8/tifad')

import numpy as np
import torch

from libs.core import load_config
from libs.datasets import make_dataset
from libs.utils import ANETdetection
from libs.modeling.text import data_split_dir
from libs.modeling.meta_archs import TiFAD


def seg_iou(a, b):
    # a: (N,2) b: (M,2) -> (N,M)
    inter = (np.minimum(a[:, None, 1], b[None, :, 1])
             - np.maximum(a[:, None, 0], b[None, :, 0])).clip(min=0)
    union = ((a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :]
             - inter)
    return inter / np.maximum(union, 1e-8)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dump = sys.argv[2]
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    rel = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    cfg = load_config('/data3/xiaodan8/tifad/configs/thumos14_i3d.yaml')
    os.chdir('/data3/xiaodan8/tifad')
    val_dataset = make_dataset(
        cfg['dataset_name'], n, False, cfg['val_split'], **cfg['dataset'])
    subset_file = data_split_dir(
        cfg['dataset']['subset_file'], cfg['dataset']['data_split'],
        mode='test', split_num=n)
    det_eval = ANETdetection(
        val_dataset.json_file, val_dataset.split[0],
        tiou_thresholds=val_dataset.get_attributes()['tiou_thresholds'],
        subset_file=subset_file)
    gt = det_eval.ground_truth  # df: video-id, t-start, t-end, label
    id2name = {v: k for k, v in det_eval.label_dict.items()} \
        if hasattr(det_eval, 'label_dict') else {}

    with open(dump, 'rb') as f:
        raw = pickle.load(f)
    vid = np.asarray(raw['video-id'])
    ts = np.asarray(raw['t-start'], dtype=np.float64)
    te = np.asarray(raw['t-end'], dtype=np.float64)
    lb = np.asarray(raw['label'])
    sc = np.asarray(raw['score'], dtype=np.float64)

    # gt['label'] has been replaced by activity_index and, for THUMOS
    # subsets, label_dict maps test classes to 0..C-1 so the mapping is
    # identity with the dump's label indices.
    stats = {}
    for thr in (0.5, 0.7):
        tot = rec_base = rec_union = 0
        per_cls = {}
        for (v, gl), g in gt.groupby(['video-id', 'label']):
            cid = int(gl)
            gsegs = g[['t-start', 't-end']].values
            m = (vid == v) & (lb == cid)
            tot += len(gsegs)
            key = id2name.get(gl, str(gl))
            pc = per_cls.setdefault(key, [0, 0, 0])
            pc[0] += len(gsegs)
            if not m.any():
                continue
            dsegs = np.stack([ts[m], te[m]], -1)
            dsc = sc[m]
            best = seg_iou(gsegs, dsegs).max(1)
            nb = int((best >= thr).sum())
            rec_base += nb
            pc[1] += nb
            # unions
            segs_t = torch.from_numpy(dsegs.astype(np.float32))
            s, c, l = TiFAD._stitch_segments(
                segs_t, torch.from_numpy(dsc.astype(np.float32)),
                torch.zeros(len(dsc), dtype=torch.long),
                gap, 'max', 0.0, rel, True, 1.0)
            usegs = s.numpy().astype(np.float64)
            bestu = seg_iou(gsegs, usegs).max(1)
            nu = int((np.maximum(best, bestu) >= thr).sum())
            rec_union += nu
            pc[2] += nu
        stats[thr] = (tot, rec_base, rec_union, per_cls)
        print(f"== tIoU {thr}: GT={tot} covered_base={rec_base} "
              f"covered_with_unions={rec_union} "
              f"(recoverable +{rec_union - rec_base})")
        for k, (t, b, u) in sorted(per_cls.items()):
            if u > b:
                print(f"   {k}: GT={t} base={b} +union={u - b}")


if __name__ == '__main__':
    main()
