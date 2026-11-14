"""Isolate attribute RECOGNITION from localization on a trained ZSL checkpoint.

For every GT segment in val windows, read the phrase + specialist outputs at the
segment's level-0 tokens and measure:
  1. per-slot specialist top-1 accuracy at GT tokens, split seen vs held-out
     classes (attribute TRANSFER gap; also accuracy on 'untrained' targets);
  2. held-out class identification: rank the 50 held-out classes by the composed
     score (phrase logit at parent + sum alpha*logprob at targets) averaged over
     the segment's tokens -> top-1 / top-5 accuracy and mean rank of the true
     class. This is "mAP with perfect localization".

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/analyze_attr_transfer.py \
    --config configs/anet_zsl/eval_arch224_split0_75.yaml \
    --ckpt ckpt/activitynet_i3d_zsl_split0_75_anet_smoke/epoch_020 [--limit 4000]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch

SLOTS = ['body_shape', 'direction', 'multiplicity', 'named_element',
         'motion_category', 'leg_configuration', 'inverted_phase',
         'hands_on_apparatus']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--limit', type=int, default=4000)
    ap.add_argument('--alpha', type=float, default=0.3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ad = json.load(open(cfg['model']['aux_attribute_table_path']))
    held_cids = [int(a[1:]) for a in ad['held_out_classes']]
    cfg['dataset']['held_out_class_ids'] = held_cids
    cfg['model']['train_cfg']['head_empty_cls'] = []
    cfg['model']['active_learning_method'] = cfg.get('active_learning_method', None)

    val = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                       cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    sd = torch.load(os.path.join(args.ckpt, 'mp_rank_00_model_states.pt'),
                    map_location='cpu')['module']
    model.load_state_dict(sd, strict=False)
    model = model.float().cuda().eval()

    held_set = set(held_cids)
    held_t = torch.tensor(held_cids, dtype=torch.long)
    parent_of = model.action_to_phrase_buf.cpu()          # (200,)
    tgt = {s: getattr(model, f'{s}_target_per_class').cpu().long() for s in SLOTS}
    val_ok = {s: getattr(model, f'{s}_valid_per_class').cpu() for s in SLOTS}
    # which targets are trained (appear in seen classes)
    seen_cids = [c for c in range(200) if c not in held_set]
    trained_targets = {s: {int(tgt[s][c]) for c in seen_cids if val_ok[s][c] > 0}
                       for s in SLOTS}

    acc = {s: defaultdict(lambda: [0, 0]) for s in SLOTS}  # group -> [correct, total]
    ph_acc = defaultdict(lambda: [0, 0])                   # phrase-head accuracy by group
    ident = {'top1': 0, 'top5': 0, 'n': 0, 'ranks': []}
    per_cls_dir = {}
    MOTION_NAMES = ['throw_strike', 'locomotion', 'rotate_swing_flow',
                    'manipulate_fine', 'pose_balance']

    n_proc = 0
    with torch.no_grad():
        for wi in range(len(val)):
            if n_proc >= args.limit:
                break
            item = val[wi]
            segs, labs = item.get('segments'), item.get('labels')
            if segs is None or labs is None or len(segs) == 0:
                continue
            n_proc += 1
            model([{k: (v.cuda() if torch.is_tensor(v) else v)
                    for k, v in item.items()}])
            # level-0 buffers captured during forward
            ph = model._cascade_phrase_logits[0][0].float().cpu()      # (T0, P)
            sp_logp = {}
            for s in SLOTS:
                lg = getattr(model, f'_{s}_logits')
                if lg is None:
                    continue
                sp_logp[s] = F.log_softmax(lg[0].float(), dim=0).cpu()  # (C, T0)
            T0 = ph.shape[0]
            for i in range(len(labs)):
                c = int(labs[i])
                s0, e0 = float(segs[i][0]), float(segs[i][1])
                a, b = max(0, int(np.floor(s0))), min(T0 - 1, int(np.ceil(e0)))
                if b < a:
                    continue
                grp = 'held' if c in held_set else 'seen'
                # 0) phrase-head top-1 accuracy at GT tokens (coarse->fine decision)
                ph_pred = ph[a:b + 1, :].mean(dim=0).argmax().item()
                ph_acc[grp][0] += int(ph_pred == int(parent_of[c]))
                ph_acc[grp][1] += 1
                # 1) specialist accuracy at GT tokens
                for s, lp in sp_logp.items():
                    if val_ok[s][c] <= 0:
                        continue
                    t_c = int(tgt[s][c])
                    pred = lp[:, a:b + 1].mean(dim=1).argmax().item()
                    key = grp
                    if grp == 'held' and t_c not in trained_targets[s]:
                        key = 'held_untrained'
                    acc[s][key][0] += int(pred == t_c)
                    acc[s][key][1] += 1
                    if s == 'direction' and grp == 'held':
                        st = per_cls_dir.setdefault(c, [0, 0, t_c])
                        st[0] += int(pred == t_c)
                        st[1] += 1
                # 2) held-out identification among the 50 held-out classes
                if grp == 'held':
                    scores = ph[a:b + 1, :].mean(dim=0)[parent_of[held_t]]  # (H,)
                    for s, lp in sp_logp.items():
                        t_h = tgt[s][held_t].clone()
                        ok_h = val_ok[s][held_t] > 0
                        t_h[~ok_h] = 0
                        contrib = lp[:, a:b + 1].mean(dim=1)[t_h]
                        contrib[~ok_h] = 0.0
                        scores = scores + args.alpha * contrib
                    order = torch.argsort(scores, descending=True)
                    rank = (order == (held_t == c).nonzero()[0, 0]).nonzero()[0, 0].item()
                    ident['top1'] += int(rank == 0)
                    ident['top5'] += int(rank < 5)
                    ident['n'] += 1
                    ident['ranks'].append(rank)

    print(f"\nprocessed {n_proc} GT windows")
    if per_cls_dir:
        print("\n=== held-out per-class DIRECTION accuracy (worst first) ===")
        rows = sorted((ok/tot, c, tot, t) for c, (ok, tot, t) in per_cls_dir.items())
        for accv, c, tot, t in rows[:15]:
            print(f"  c{c:03d} acc={accv*100:5.1f}% n={tot:3d} assigned={MOTION_NAMES[t]}")
    print("\n=== PHRASE head top-1 accuracy at GT tokens (coarse->fine rule: enable iff held < ~0.5) ===")
    for key in ('seen', 'held'):
        c_, t_ = ph_acc[key]
        print(f"  phrase {key}: {c_/t_*100:5.1f}% (n={t_})" if t_ else f"  phrase {key}: n/a")
    print("\n=== specialist top-1 accuracy at GT tokens (transfer test) ===")
    for s in SLOTS:
        row = []
        for key in ('seen', 'held', 'held_untrained'):
            c_, t_ = acc[s][key]
            row.append(f"{key}: {c_/t_*100:5.1f}% (n={t_})" if t_ else f"{key}: n/a")
        print(f"  {s:20s} " + " | ".join(row))
    if ident['n']:
        print(f"\n=== held-out class identification among 50 (perfect localization) ===")
        print(f"  top-1: {ident['top1']/ident['n']*100:.1f}%   "
              f"top-5: {ident['top5']/ident['n']*100:.1f}%   "
              f"mean rank: {np.mean(ident['ranks']):.1f} / 50   (n={ident['n']} segments)"
              f"   [chance top-1 = 2.0%]")


if __name__ == '__main__':
    main()
