"""ASPEC diagnostic: probe the trained attribute specialists on the test set.

For every GT segment of an UNSEEN (test) class it pools the level-0
per-snippet logits inside the segment and reports:
  (i)  per-attribute AP -- segment is a positive for attribute a iff a is in
       the GT class's attribute set; score = pooled specialist logit.
  (ii) unseen-class identification top-1 accuracy from
         - attribute COMPOSITION alone:
             score(c) = mean_{a in attrs(c)} logsigmoid(pooled attr logit a)
         - TEXT logits alone (pooled snippet text-classifier logits).

Capture reuses the TIFAD_PROTO_COLLECT stash pattern: the model-side block
gated on TIFAD_ASPEC_COLLECT=1 (libs/modeling/meta_archs.py) stashes the
attribute logits + PRE-composition text logits per forward.

Usage (env must match training so the ckpt strict-loads, e.g.):
  cd /data3/xiaodan8/tifad
  TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json \
  TIFAD_ATTR_ATTN=1 \
  python scripts/aspec_diag.py configs/thumos14_i3d.yaml \
      ckpt/thumos14_i3d_aspec_0 --n 0
(TIFAD_ASPEC_COLLECT=1 and TIFAD_ASPEC_ATTRS are set by the script itself;
TIFAD_ASPEC_ALPHA is force-cleared so the text logits stay pure.)
"""

# python imports
import argparse
import glob
import json
import math
import os
import sys

# make repo root importable when run as scripts/aspec_diag.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# our code
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


def average_precision(scores, labels):
    """AP over a 1-D score/binary-label pair (list/1-D tensors)."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return float('nan')
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    tp = torch.cumsum(labels, dim=0)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    precision = tp / ranks
    return float((precision * labels).sum().item() / n_pos)


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
            ckpt_file_list = sorted(
                glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            assert len(ckpt_file_list) > 0, "No .pth.tar found in ckpt folder"
            ckpt_file = ckpt_file_list[-1]

    # model-side stash is gated on this env var; the attrs json gates the
    # specialist head construction (needed for strict ckpt loading)
    os.environ['TIFAD_ASPEC_COLLECT'] = '1'
    os.environ.setdefault('TIFAD_ASPEC_ATTRS', args.attrs)
    # keep the stashed text logits pure (no composition blended in)
    os.environ.pop('TIFAD_ASPEC_ALPHA', None)

    class2attrs = json.load(open(os.environ['TIFAD_ASPEC_ATTRS']))
    attr_names = sorted({a for v in class2attrs.values() for a in v})
    aid = {a: i for i, a in enumerate(attr_names)}
    A = len(attr_names)

    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # TEST (unseen-class) dataset, eval style loader (bs=1, no shuffle)
    val_dataset = make_dataset(
        cfg['dataset_name'], args.n, False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    # unseen class names in label-index order (label_dict: name -> idx);
    # subset files drive both the dataset label_dict and the text class
    # list, so label idx == text logit column
    label_dict = val_dataset.label_dict
    unseen_classes = [None] * len(label_dict)
    for name, idx in label_dict.items():
        unseen_classes[idx] = name
    C = len(unseen_classes)
    missing = [c for c in unseen_classes if c not in class2attrs]
    assert not missing, f"classes missing from attrs json: {missing}"

    # (C, A) binary incidence matrix for the unseen classes
    M = torch.zeros(C, A)
    for ci, c in enumerate(unseen_classes):
        for a in class2attrs[c]:
            M[ci, aid[a]] = 1.0
    cnt = M.sum(-1).clamp(min=1.0)

    # model, built exactly as eval.py does (single device for the stash)
    model = make_meta_arch(
        args.n,
        cfg['model_name'],
        cfg['dataset']['subset_file'],
        cfg['dataset']['data_split'],
        **cfg['model']
    )
    assert model.aspec_head is not None, \
        "aspec head missing -- TIFAD_ASPEC_ATTRS not picked up?"
    assert model.aspec_attr_names == attr_names
    model = nn.DataParallel(model, device_ids=cfg['devices'][:1])

    print("=> loading checkpoint '{}'".format(ckpt_file))
    checkpoint = torch.load(ckpt_file, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint

    model.eval()

    # speed only: decoded detections are unused, keep NMS trivial
    try:
        model.module.test_pre_nms_thresh = 2.0
        model.module.test_nms_method = 'none'
    except AttributeError:
        pass

    # per-GT-segment pooled scores
    seg_attr_scores = []   # (A,) pooled attribute logits
    seg_txt_scores = []    # (C,) pooled text logits
    seg_labels = []        # int class idx

    num_vids = 0
    for video_list in val_loader:
        segments = video_list[0]['segments']   # (N, 2), level-0 grid units
        labels = video_list[0]['labels']       # (N,)
        if segments is None or labels is None:
            continue
        with torch.no_grad():
            model(video_list)
        stash = getattr(model.module, '_aspec_stash', None)
        assert stash is not None, (
            "no _aspec_stash on model; is TIFAD_ASPEC_COLLECT=1 set and the "
            "meta_archs.py stash block present?"
        )
        aspec0 = stash['aspec'][0][0].float().cpu()   # (T0, A)
        cls0 = stash['cls'][0][0].float().cpu()       # (T0, C)
        assert aspec0.shape[1] == A, f"attr dims {aspec0.shape[1]} != {A}"
        assert cls0.shape[1] == C, f"text logit cols {cls0.shape[1]} != {C}"

        T0 = aspec0.shape[0]
        feat_len = min(int(video_list[0]['feats'].shape[-1]), T0)

        for seg, lab in zip(segments, labels):
            c = int(lab)
            if not (0 <= c < C):
                continue
            s = int(math.floor(float(seg[0])))
            e = int(math.ceil(float(seg[1])))
            s = max(0, min(s, feat_len - 1))
            e = max(s + 1, min(e, feat_len))
            av = aspec0[s:e].mean(dim=0)              # (A,)
            tv = cls0[s:e].mean(dim=0)                # (C,)
            if torch.isnan(av).any() or torch.isnan(tv).any():
                continue
            seg_attr_scores.append(av)
            seg_txt_scores.append(tv)
            seg_labels.append(c)

        num_vids += 1
        if num_vids % 50 == 0:
            print(f"[diag] {num_vids} videos, {len(seg_labels)} GT segments")
        if args.max_videos > 0 and num_vids >= args.max_videos:
            break

    assert len(seg_labels) > 0, "no GT segments collected"
    attr_s = torch.stack(seg_attr_scores)             # (N, A)
    txt_s = torch.stack(seg_txt_scores)               # (N, C)
    labs = torch.tensor(seg_labels)                   # (N,)
    N = labs.numel()
    print(f"[diag] done: {num_vids} videos, {N} unseen-class GT segments, "
          f"{C} unseen classes, {A} attributes")

    # (i) per-attribute AP on unseen-class GT segments
    attr_gt = M[labs]                                 # (N, A) binary
    print("\n== (i) per-attribute AP (unseen-class GT segments) ==")
    aps = []
    for a in range(A):
        n_pos = int(attr_gt[:, a].sum().item())
        ap = average_precision(attr_s[:, a], attr_gt[:, a])
        base = n_pos / N if N else float('nan')
        flag = '' if not math.isnan(ap) else '  (no positives)'
        print(f"  {attr_names[a]:<24s} AP={ap:6.4f}  "
              f"pos={n_pos:4d}/{N}  prior={base:.4f}{flag}")
        if not math.isnan(ap):
            aps.append(ap)
    print(f"  mean AP over {len(aps)} attrs with positives: "
          f"{sum(aps) / max(len(aps), 1):.4f}")

    # (ii) unseen-class identification top-1
    comp = (F.logsigmoid(attr_s) @ M.t()) / cnt       # (N, C)
    comp_pred = comp.argmax(dim=-1)
    txt_pred = txt_s.argmax(dim=-1)
    comp_acc = float((comp_pred == labs).float().mean().item())
    txt_acc = float((txt_pred == labs).float().mean().item())
    print("\n== (ii) unseen-class identification top-1 "
          "(per GT segment) ==")
    print(f"  attribute composition alone : {comp_acc:.4f}")
    print(f"  text logits alone           : {txt_acc:.4f}")
    print(f"  chance ({C} classes)         : {1.0 / C:.4f}")

    # per-class breakdown
    print("\n  per-class top-1 (composition / text):")
    for ci, cname in enumerate(unseen_classes):
        m = labs == ci
        n = int(m.sum().item())
        if n == 0:
            continue
        ca = float((comp_pred[m] == ci).float().mean().item())
        ta = float((txt_pred[m] == ci).float().mean().item())
        print(f"    {cname:<24s} n={n:4d}  comp={ca:.4f}  text={ta:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({
            'attr_scores': attr_s, 'txt_scores': txt_s, 'labels': labs,
            'attr_names': attr_names, 'unseen_classes': unseen_classes,
            'M': M, 'ckpt': ckpt_file, 'config': args.config, 'n': args.n,
        }, args.out)
        print(f"[save] wrote {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ASPEC diagnostics: attribute AP + unseen-class '
                    'identification from video-side attribute specialists')
    parser.add_argument('config', type=str, help='path to a config file')
    parser.add_argument('ckpt', type=str,
                        help='ckpt dir (uses mAP_best.pth.tar) or .pth.tar')
    parser.add_argument('--n', default=0, type=int, help='split number')
    parser.add_argument('--attrs', default='configs/thumos14_ttb_attrs.json',
                        type=str, help='class->attrs json (fallback when '
                        'TIFAD_ASPEC_ATTRS is not already set)')
    parser.add_argument('--out', default='', type=str,
                        help='optional .pt path to dump raw pooled scores')
    parser.add_argument('--max-videos', default=-1, type=int,
                        help='debug: stop after this many test videos')
    args = parser.parse_args()
    main(args)
