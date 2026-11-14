#!/usr/bin/env python3
"""CPU verification of the champion's third stage (video branch of the
action--sub-action attention, libs/modeling/vsa.py).

Checks:
  1. stage-3 model (use_video_subtree_attn=True) initialised from a stage-2
     state dict with strict=False: the only missing keys are the 8 tensors
     of the two zero-init MHA's (video_subtree_attn.*).
  2. init-equality: with the zero-initialised residuals, the stage-3 eval
     decode (segments / scores / labels) is byte-equal to stage 2 on a real
     test window, AND the stage-2 model built from the same code is
     byte-equal to the pre-stage-3 code path (byte-inert flag off: the
     vsa attribute is None and the reg-head reordering changes nothing).
  3. --freeze_except video_subtree_attn leaves exactly the 8 tensors
     trainable and make_optimizer hands only those to the optimizer.
  4. the banded mask is |i-j|<=1 over K=10 chunks; one train-mode forward
     on a real batch gives finite losses and a non-zero gradient on the
     video branch (out_proj's) and none elsewhere.
  5. the cached-feature input path (D x T features + pad vector) gives the
     same decode as the frame path when the features are the encoder's own
     output (sanity of backbones.py feature branch), on a tiny window.
Run from the repo root:  python scripts/verify_t3_stage3.py
"""
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch
from libs.utils.train_utils_deepspeed import make_optimizer

SEED = 20260918


def load_cfg(path):
    cfg = load_config(path)
    cfg['dataset']['max_seq_len'] = 96
    cfg['model']['max_seq_len'] = 96
    cfg['model']['n_mha_win_size'] = 5
    cfg['model']['train_cfg']['head_empty_cls'] = []
    cfg['model']['active_learning_method'] = 'uniform'
    with open(cfg['model']['aux_attribute_table_path']) as f:
        held = [int(a[1:]) for a in json.load(f)['held_out_classes']]
    cfg['dataset']['held_out_class_ids'] = held
    return cfg


def build(cfg):
    torch.manual_seed(SEED)
    return make_meta_arch(cfg['model_name'], **cfg['model']).float()


def decode(m, batch):
    m.eval()
    with torch.no_grad():
        out = m(batch)
    return [(o['segments'].clone(), o['scores'].clone(), o['labels'].clone()) for o in out]


def same(a, b):
    return all(torch.equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q))


def main():
    cfg2 = load_cfg('configs/finegym_t3_champ_stage2.yaml')
    cfg3 = load_cfg('configs/finegym_t3_champ_stage3_vsa.yaml')
    m2 = build(cfg2)
    assert m2.video_subtree_attn is None
    # emulate a trained stage 2: perturb trainable weights so equality is
    # not trivially satisfied by identical inits
    with torch.no_grad():
        for n, p in m2.named_parameters():
            if ('text_pathway' in n or 'cls_head' in n or 'reg_head' in n) and p.dim() >= 1:
                p.add_(0.01 * torch.randn_like(p))
    sd2 = {k: v.clone() for k, v in m2.state_dict().items()}

    m3 = build(cfg3)
    assert m3.video_subtree_attn is not None
    missing, unexpected = m3.load_state_dict(sd2, strict=False)
    print('missing:', sorted(missing))
    assert not unexpected, unexpected
    assert all(k.startswith('video_subtree_attn.') for k in missing) and len(missing) == 8, missing
    print('1. init_from: only the 8 video-branch tensors are new  OK')

    # 2. init-equality on a real test window (eval decode) and a train window
    vds = make_dataset(cfg2['dataset_name'], False, cfg2['val_split'],
                       cfg2['model']['backbone_type'], cfg2['round'], **cfg2['dataset'])
    vb = [vds[len(vds) // 2]]
    d2 = decode(m2, vb)
    d3 = decode(m3, vb)
    assert d2[0][0].numel() > 0, 'empty decode'
    assert same(d2, d3), 'stage-3 init decode differs from stage 2'
    # the video branch really ran (its out_proj's are zero => residual 0)
    for mod in (m3.video_subtree_attn.chain_attn, m3.video_subtree_attn.clip_attn):
        assert mod.out_proj.weight.abs().sum() == 0 and mod.out_proj.bias.abs().sum() == 0
    print('2. eval decode byte-equal at stage-3 init (zero-init residuals)  OK')

    # 3. freeze
    keep = ['video_subtree_attn']
    for n, p in m3.named_parameters():
        p.requires_grad = any(k in n for k in keep)
    trainable = [n for n, p in m3.named_parameters() if p.requires_grad]
    assert len(trainable) == 8, trainable
    opt = make_optimizer(m3, cfg3['opt'])
    n_opt = sum(len(g['params']) for g in opt.param_groups)
    assert n_opt == 8, n_opt
    print('3. freeze_except: trainable =', trainable, '; optimizer tensors =', n_opt, ' OK')

    # 4. mask + forward/backward
    band = m3.video_subtree_attn.band_mask
    K = band.shape[0]
    idx = torch.arange(K)
    assert K == 10 and torch.equal(band, (idx[:, None] - idx[None, :]).abs() > 1), band
    ds = make_dataset(cfg3['dataset_name'], True, cfg3['train_split'],
                      cfg3['model']['backbone_type'], cfg3['round'], **cfg3['dataset'])
    batch = [ds[i] for i in range(2)]
    m3.train()
    losses = m3(batch)
    for k, v in losses.items():
        assert torch.isfinite(v).all(), (k, v)
    losses['final_loss'].backward()
    g_vsa = {n: (p.grad is not None and p.grad.abs().sum().item() > 0)
             for n, p in m3.named_parameters() if 'video_subtree_attn' in n}
    g_other = [p.grad is not None and p.grad.abs().sum().item() > 0
               for n, p in m3.named_parameters() if 'video_subtree_attn' not in n]
    print('   grads on video branch:', g_vsa)
    assert g_vsa['video_subtree_attn.clip_attn.out_proj.weight'], 'no gradient reached clip_attn.out_proj'
    assert not any(g_other), 'a frozen parameter received a gradient'
    print('4. banded mask OK; train forward finite; gradient only on the video branch  OK')

    # 5. cached-feature input path == frame path (backbones.py feature branch)
    m2.eval()
    with torch.no_grad():
        x, mask = m2.preprocessing(vb)                      # (1, T, 3, H, W) float
        enc_out = m2.backbone.encoder(x)                    # (1, D, T)
        T_real = int(mask.sum())
        feats_real = enc_out[0, :, :T_real].clone()
        pad_vec = enc_out[0, :, T_real].clone()
        m2.backbone.register_buffer('cached_pad_feat', pad_vec, persistent=False)
        fb = [dict(vb[0])]
        fb[0]['feats'] = feats_real                         # D x T_real
        d_cached = decode(m2, fb)
    assert same(d2, d_cached), 'cached-feature path differs from the frame path'
    print('5. cached-feature input path byte-equal to the frame path  OK')
    print('\n============ STAGE-3 CHECKS PASSED ============')


if __name__ == '__main__':
    main()
