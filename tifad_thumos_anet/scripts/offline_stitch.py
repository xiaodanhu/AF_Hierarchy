"""Offline stitch sweep: consumes a TIFAD_DUMP_RAW pickle (post-NMS,
pre-fusion detections) and replays stitch -> ext-score fusion -> ANET
evaluation entirely on CPU. Reuses TiFAD._stitch_segments verbatim, so
the swept code path IS the deployed one."""
import argparse
import os
import pickle
import sys

sys.path.insert(0, '/data3/xiaodan8/tifad')

import numpy as np
import torch

from libs.core import load_config
from libs.datasets import make_dataset
from libs.utils import ANETdetection
from libs.utils.postprocessing import postprocess_results
from libs.modeling.text import data_split_dir
from libs.modeling.meta_archs import TiFAD


def apply_rerank(results, lam, theta, norm, mu, nu):
    vid = np.asarray(results['video-id'])
    ts = np.asarray(results['t-start'], dtype=np.float32)
    te = np.asarray(results['t-end'], dtype=np.float32)
    lb = np.asarray(results['label'])
    sc = np.asarray(results['score'], dtype=np.float32)
    out_v, out_s, out_e, out_l, out_c = [], [], [], [], []
    for v in np.unique(vid):
        m = vid == v
        segs = torch.from_numpy(np.stack([ts[m], te[m]], -1))
        s, c, l = TiFAD._rerank_detections(
            segs, torch.from_numpy(sc[m]), torch.from_numpy(lb[m]),
            lam, theta, norm, mu, nu)
        n = s.shape[0]
        out_v.extend([v] * n)
        out_s.append(s[:, 0].numpy())
        out_e.append(s[:, 1].numpy())
        out_l.append(l.numpy())
        out_c.append(c.numpy())
    return {
        'video-id': out_v,
        't-start': np.concatenate(out_s),
        't-end': np.concatenate(out_e),
        'label': np.concatenate(out_l),
        'score': np.concatenate(out_c),
        'class_names': results['class_names'],
    }


def apply_stitch(results, gap, score_mode='max', maxlen=0.0, rel=0.0,
                 add=False, disc=1.0):
    vid = np.asarray(results['video-id'])
    ts = np.asarray(results['t-start'], dtype=np.float32)
    te = np.asarray(results['t-end'], dtype=np.float32)
    lb = np.asarray(results['label'])
    sc = np.asarray(results['score'], dtype=np.float32)
    out_v, out_s, out_e, out_l, out_c = [], [], [], [], []
    for v in np.unique(vid):
        m = vid == v
        segs = torch.from_numpy(np.stack([ts[m], te[m]], -1))
        s, c, l = TiFAD._stitch_segments(
            segs, torch.from_numpy(sc[m]), torch.from_numpy(lb[m]),
            gap, score_mode, maxlen, rel, add, disc)
        n = s.shape[0]
        out_v.extend([v] * n)
        out_s.append(s[:, 0].numpy())
        out_e.append(s[:, 1].numpy())
        out_l.append(l.numpy())
        out_c.append(c.numpy())
    return {
        'video-id': out_v,
        't-start': np.concatenate(out_s),
        't-end': np.concatenate(out_e),
        'label': np.concatenate(out_l),
        'score': np.concatenate(out_c),
        'class_names': results['class_names'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--fuse', type=float, default=0.0)
    ap.add_argument('--perclass', action='store_true')
    # each config: gap,score,maxlen,rel,mode,disc  (or the word "base")
    ap.add_argument('configs', nargs='+')
    args = ap.parse_args()

    cfg = load_config('/data3/xiaodan8/tifad/configs/thumos14_i3d.yaml')
    os.chdir('/data3/xiaodan8/tifad')
    val_dataset = make_dataset(
        cfg['dataset_name'], args.n, False, cfg['val_split'],
        **cfg['dataset'])
    subset_file = data_split_dir(
        cfg['dataset']['subset_file'], cfg['dataset']['data_split'],
        mode='test', split_num=args.n)
    det_eval = ANETdetection(
        val_dataset.json_file, val_dataset.split[0],
        tiou_thresholds=val_dataset.get_attributes()['tiou_thresholds'],
        subset_file=subset_file)

    with open(args.dump, 'rb') as f:
        raw = pickle.load(f)

    if args.fuse > 0:
        os.environ['TIFAD_FUSE_ALPHA'] = str(args.fuse)
    if args.perclass:
        os.environ['TIFAD_PERCLASS_AP'] = '1'

    for spec in args.configs:
        if spec == 'base':
            stitched = raw
            tag = 'base'
        elif spec.startswith('rr:'):
            lam, theta, norm, mu, nu = spec[3:].split(',')
            stitched = apply_rerank(
                raw, float(lam), float(theta), norm, float(mu), float(nu))
            tag = spec
        else:
            g, sm, ml, rel, mode, d = spec.split(',')
            stitched = apply_stitch(
                raw, float(g), sm, float(ml), float(rel),
                mode == 'add', float(d))
            tag = spec
        res = postprocess_results(
            dict(stitched), cfg['test_cfg']['ext_score_file'],
            cfg['opt']['prior_prob'], cfg['opt']['num_pred'])
        _, mAP, tiou_list = det_eval.evaluate(res, verbose=True)
        print(f"[OFFLINE] {tag} avg={mAP*100:.2f} tious={tiou_list}",
              flush=True)


if __name__ == '__main__':
    main()
