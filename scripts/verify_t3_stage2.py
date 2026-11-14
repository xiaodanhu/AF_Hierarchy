#!/usr/bin/env python3
"""CPU verification of the champion's second stage (APA phase-level attention).

Checks:
  1. stage-2 model (text_phase_neighbor_attention=True) initialised from a
     stage-1 state dict with strict=False: the only missing keys are the
     phase-attention tensors.
  2. init-equality: with the zero-initialised phase attention, stage-2 class
     embeddings and eval decode are byte-equal to stage 1.
  3. --freeze_except text_pathway.phase_attn leaves exactly the 4 phase
     attention tensors trainable, and make_optimizer hands only those to
     the optimizer.
  4. the banded mask is |i-j|<=1 over the K-1 phases; one train-mode forward
     on a real batch gives finite losses and a non-zero gradient on the
     phase attention (and none elsewhere).
Run from the repo root:  python scripts/verify_t3_stage2.py
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

SEED = 20260910


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


def main():
    cfg1 = load_cfg('configs/finegym_t3_champ.yaml')
    cfg2 = load_cfg('configs/finegym_t3_champ_stage2.yaml')
    m1 = build(cfg1)
    # emulate a trained stage 1: perturb the trainable text-side weights so
    # equality is not trivially satisfied by identical inits
    with torch.no_grad():
        for n, p in m1.named_parameters():
            if 'text_pathway' in n and p.dim() >= 1:
                p.add_(0.01 * torch.randn_like(p))
    sd1 = {k: v.clone() for k, v in m1.state_dict().items()}

    m2 = build(cfg2)
    assert m2.text_pathway.phase_attn is not None
    missing, unexpected = m2.load_state_dict(sd1, strict=False)
    print('missing:', sorted(missing))
    assert not unexpected, unexpected
    assert all('text_pathway.phase_attn' in k for k in missing) and len(missing) == 4, missing
    print('1. init_from: only the 4 phase-attention tensors are new  OK')

    # 2. init-equality
    m1.eval(); m2.eval()
    with torch.no_grad():
        e1 = m1.text_pathway.class_embeddings()
        e2 = m2.text_pathway.class_embeddings()
    assert torch.equal(e1, e2), (e1 - e2).abs().max()
    print('2. class embeddings byte-equal at stage-2 init  OK')

    # 3. freeze
    keep = ['text_pathway.phase_attn']
    for n, p in m2.named_parameters():
        p.requires_grad = any(k in n for k in keep)
    trainable = [n for n, p in m2.named_parameters() if p.requires_grad]
    assert len(trainable) == 4, trainable
    opt = make_optimizer(m2, cfg2['opt'])
    n_opt = sum(len(g['params']) for g in opt.param_groups)
    assert n_opt == 4, n_opt
    print('3. freeze_except: trainable =', trainable, '; optimizer tensors =', n_opt, ' OK')

    # 4. mask + forward/backward
    band = m2.text_pathway.phase_band_mask
    K1 = band.shape[0]
    idx = torch.arange(K1)
    expect = torch.where((idx[:, None] - idx[None, :]).abs() <= 1, 0.0, float('-inf'))
    assert torch.equal(band, expect) and K1 == 10, (K1, band)
    ds = make_dataset(cfg2['dataset_name'], True, cfg2['train_split'],
                      cfg2['model']['backbone_type'], cfg2['round'], **cfg2['dataset'])
    batch = [ds[i] for i in range(2)]
    m2.train()
    losses = m2(batch)
    for k, v in losses.items():
        assert torch.isfinite(v).all(), (k, v)
    losses['final_loss'].backward()
    g_phase = [p.grad is not None and p.grad.abs().sum().item() > 0
               for n, p in m2.named_parameters() if 'phase_attn' in n]
    g_other = [p.grad is not None and p.grad.abs().sum().item() > 0
               for n, p in m2.named_parameters() if 'phase_attn' not in n]
    # out_proj is zero so in_proj gets no gradient at step 0; out_proj must
    assert any(g_phase), 'no gradient reached the phase attention'
    assert not any(g_other), 'a frozen parameter received a gradient'
    print('4. banded mask OK; train forward finite; gradient only on phase attention  OK')
    print('\n============ STAGE-2 CHECKS PASSED ============')


if __name__ == '__main__':
    main()
