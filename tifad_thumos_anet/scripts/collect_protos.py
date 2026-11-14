"""Collect seen-class VISUAL prototypes and fit a text->visual ridge map.

EXEM-style zero-shot transfer migrated from image ZSL to Ti-FAD:
  1. Run the trained model (eval mode) over the TRAIN (seen-class) loader
     with TIFAD_PROTO_COLLECT=1, which (a) forces the seen-class text set
     and (b) stashes the level-0 backbone features + projected text matrix
     used in the classification matmul (see libs/modeling/meta_archs.py).
  2. For every GT segment of class c, average the level-0 feature columns
     inside the segment (dataset 'segments' are already in level-0 feature
     grid units for anet: seg_grid = (seg_sec * fps - 0.5*num_frames)/feat_stride,
     with T0 == max_seq_len under force_upsampling). Prototype P_c is the
     mean over all segment means of class c -- same D-dim space as the
     matmul features.
  3. Fit ridge W: T_c (projected text) -> P_c over seen classes, closed
     form, lambda swept by leave-one-out MSE.
  4. Save {'W', 'lam', 'protos_seen', 'seen_classes', ...} for use at eval
     via TIFAD_PROTO_FILE / TIFAD_PROTO_GAMMA (see meta_archs.py).

Usage (GPU):
  cd /data3/xiaodan8/tifad
  TIFAD_PROTO_COLLECT=1 python scripts/collect_protos.py \
      configs/anet_i3d_75.yaml ckpt/anet_i3d_75_h19_0 --n 0 \
      --out ckpt/anet_i3d_75_h19_0/protos_split0.pt

(The script also sets TIFAD_PROTO_COLLECT=1 itself, so the env prefix is
optional.)
"""

# python imports
import argparse
import glob
import math
import os
import sys

# make repo root importable when run as scripts/collect_protos.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch imports
import torch
import torch.nn as nn

# our code
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


def fit_ridge_loo(T, P, lambdas=(0.1, 1.0, 10.0)):
    """Closed-form ridge T @ W ~= P with leave-one-out lambda selection.

    T: (C, D) inputs (projected text vectors, seen classes)
    P: (C, D) targets (visual prototypes)
    Returns (W (D, D), best_lambda, loo_mse_dict).
    """
    T = T.double()
    P = P.double()
    C, D = T.shape
    TtT = T.t() @ T                       # (D, D)
    TtP = T.t() @ P                       # (D, D)
    eye = torch.eye(D, dtype=T.dtype, device=T.device)

    best = None
    loo_mse = {}
    for lam in lambdas:
        inv = torch.linalg.inv(TtT + lam * eye)     # (D, D)
        W = inv @ TtP                               # (D, D)
        # hat matrix diag: H = T inv T^T ; h_c = t_c^T inv t_c
        h = ((T @ inv) * T).sum(dim=1)              # (C,)
        resid = P - T @ W                           # (C, D)
        denom = (1.0 - h).clamp(min=1e-6).unsqueeze(1)
        loo_resid = resid / denom
        mse = (loo_resid ** 2).mean().item()
        loo_mse[lam] = mse
        if best is None or mse < best[1]:
            best = (lam, mse, W)

    lam, mse, W = best
    return W.float(), lam, loo_mse


def main(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # resolve checkpoint (prefer mAP_best.pth.tar inside a dir)
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT folder does not exist!"
        best = os.path.join(args.ckpt, 'mAP_best.pth.tar')
        if os.path.isfile(best):
            ckpt_file = best
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            assert len(ckpt_file_list) > 0, "No .pth.tar found in ckpt folder"
            ckpt_file = ckpt_file_list[-1]

    # the model-side stash + seen-class text set are gated on this env var
    os.environ['TIFAD_PROTO_COLLECT'] = '1'
    # make sure blending is OFF while collecting
    os.environ.pop('TIFAD_PROTO_FILE', None)

    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # TRAIN (seen-class) dataset; is_training=True so the seen split file /
    # seen label_dict are used. Loader in eval style: no shuffle, bs=1
    # (model.eval() preprocessing only supports batch_size 1).
    train_dataset = make_dataset(
        cfg['dataset_name'], args.n, True, cfg['train_split'], **cfg['dataset']
    )
    train_loader = make_data_loader(
        train_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    # seen class names in label-index order (label_dict: name -> idx)
    label_dict = train_dataset.label_dict
    seen_classes = [None] * len(label_dict)
    for name, idx in label_dict.items():
        seen_classes[idx] = name
    num_seen = len(seen_classes)

    # model, built exactly as eval.py does (single device for the stash)
    model = make_meta_arch(
        args.n,
        cfg['model_name'],
        cfg['dataset']['subset_file'],
        cfg['dataset']['data_split'],
        **cfg['model']
    )
    model = nn.DataParallel(model, device_ids=cfg['devices'][:1])

    print("=> loading checkpoint '{}'".format(ckpt_file))
    checkpoint = torch.load(ckpt_file, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint

    model.eval()

    # speed only: we never use the decoded detections, so keep NMS trivial
    try:
        model.module.test_pre_nms_thresh = 2.0   # sigmoid scores <= 1 -> keep none
        model.module.test_nms_method = 'none'
    except AttributeError:
        pass

    # per-class running sums of segment-mean feature vectors
    proto_sum = None      # (C_seen, D)
    proto_cnt = torch.zeros(num_seen, dtype=torch.long)
    text0_cpu = None      # (C_seen, D) projected text (constant across videos)

    num_vids = 0
    for video_list in train_loader:
        with torch.no_grad():
            model(video_list)
        stash = getattr(model.module, '_proto_stash', None)
        assert stash is not None, (
            "no _proto_stash on model; is TIFAD_PROTO_COLLECT=1 set and the "
            "meta_archs.py stash block present?"
        )
        feat0 = stash['feat0'][0].float().cpu()   # (D, T0)
        if text0_cpu is None:
            text0_cpu = stash['text0'][0].float().cpu()   # (C_seen, D)
            assert text0_cpu.shape[0] == num_seen, (
                f"text rows {text0_cpu.shape[0]} != #seen classes {num_seen}"
            )
            proto_sum = torch.zeros(num_seen, feat0.shape[0])

        T0 = feat0.shape[1]
        # valid (unpadded) length of this video on the level-0 grid
        feat_len = min(int(video_list[0]['feats'].shape[-1]), T0)

        segments = video_list[0]['segments']      # (N, 2) level-0 grid units
        labels = video_list[0]['labels']          # (N,)
        for seg, lab in zip(segments, labels):
            c = int(lab)
            if not (0 <= c < num_seen):
                continue
            s = int(math.floor(float(seg[0])))
            e = int(math.ceil(float(seg[1])))
            s = max(0, min(s, feat_len - 1))
            e = max(s + 1, min(e, feat_len))
            vec = feat0[:, s:e].mean(dim=1)       # (D,)
            if torch.isnan(vec).any():
                continue
            proto_sum[c] += vec
            proto_cnt[c] += 1

        num_vids += 1
        if num_vids % 200 == 0:
            print(f"[collect] {num_vids} videos, "
                  f"{int((proto_cnt > 0).sum())}/{num_seen} classes covered")
        if args.max_videos > 0 and num_vids >= args.max_videos:
            break

    covered = proto_cnt > 0
    print(f"[collect] done: {num_vids} videos, "
          f"{int(covered.sum())}/{num_seen} classes with >=1 segment")

    protos_seen = torch.zeros_like(proto_sum)
    protos_seen[covered] = (
        proto_sum[covered] / proto_cnt[covered].unsqueeze(1).float()
    )

    # ridge fit on covered classes only
    T_fit = text0_cpu[covered]                    # (C_cov, D)
    P_fit = protos_seen[covered]                  # (C_cov, D)
    W, lam, loo_mse = fit_ridge_loo(T_fit, P_fit, lambdas=(0.1, 1.0, 10.0))
    print(f"[ridge] LOO MSE per lambda: {loo_mse} -> picked lambda={lam}")

    # sanity: report fit quality in-sample
    with torch.no_grad():
        pred = T_fit @ W
        fit_mse = ((pred - P_fit) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(pred, P_fit, dim=1)
        print(f"[ridge] in-sample MSE={fit_mse:.6f}, "
              f"mean cos(pred, proto)={cos.mean().item():.4f}")

    out = {
        'W': W,                                   # (D, D): P_pred = text @ W
        'lam': lam,
        'loo_mse': loo_mse,
        'protos_seen': protos_seen,               # (C_seen, D)
        'proto_counts': proto_cnt,                # (C_seen,)
        'seen_classes': seen_classes,
        'text_seen': text0_cpu,                   # projected text, seen classes
        'note': (
            'EXEM-style text->visual prototype ridge map. '
            'Prototypes are mean level-0 updated_feat columns inside GT '
            'segments (dataset segments already in level-0 grid units). '
            f'ckpt={ckpt_file}, config={args.config}, split={args.n}, '
            f'videos={num_vids}'
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"[save] wrote {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect seen-class visual prototypes + fit text->visual ridge map')
    parser.add_argument('config', type=str, help='path to a config file')
    parser.add_argument('ckpt', type=str,
                        help='ckpt dir (uses mAP_best.pth.tar) or .pth.tar file')
    parser.add_argument('--n', default=0, type=int, help='split number')
    parser.add_argument('--out', required=True, type=str,
                        help='output .pt path for the prototype/ridge blob')
    parser.add_argument('--max-videos', default=-1, type=int,
                        help='debug: stop after this many train videos')
    args = parser.parse_args()
    main(args)
