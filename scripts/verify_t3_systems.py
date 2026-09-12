#!/usr/bin/env python3
"""CPU verification of the FineGym Table-3 three-system comparison.

Checks (all CPU, no GPU touched):
  1. asset: qwen ensemble 99 x 11, key resolution vs gym99 names (incl. the
     duplicated "(BB) salto backward tucked" -> v1/v2 disambiguation)
  2. construction of the three systems + text cache shapes
  3. one train-mode forward (real batch, losses finite) and one eval-mode
     forward (results finite) per system
  4. init-equality: champion with a synthetic ensemble (all 11 sentences ==
     baseline prompt) is byte-equal to the baseline at init (zero-init PA,
     Kendall UW weights (1,1), DP gamma forced to 0)
  5. DP unit test on synthetic candidates

Smoke-only overrides (documented): max_seq_len 288 -> 96 and n_mha_win_size
19 -> 5 so the CPU CLIP forward stays tractable; neither changes any
parameter shape.

Run from the repo root:  python scripts/verify_t3_systems.py
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch
from libs.modeling.text_pathway import (
    load_verbalizer_actions, resolve_ensemble_keys, DurationPrior)

SEED = 20260910
ENSEMBLE = 'configs/finegym_qwen_ensemble.json'
SCRATCH = os.environ.get('T3_SCRATCH', '/tmp/t3_verify')
CFGS = {
    'base':  'configs/finegym_t3_base.yaml',
    'tifad': 'configs/finegym_t3_tifad.yaml',
    'champ': 'configs/finegym_t3_champ.yaml',
}


def hr(msg):
    print('\n' + '=' * 12 + ' ' + msg + ' ' + '=' * 12, flush=True)


def load_cfg(path):
    cfg = load_config(path)
    # smoke overrides (CPU): shorter padded sequence + smaller attn window
    cfg['dataset']['max_seq_len'] = 96
    cfg['model']['max_seq_len'] = 96
    cfg['model']['n_mha_win_size'] = 5
    cfg['model']['train_cfg']['head_empty_cls'] = []
    cfg['model']['active_learning_method'] = 'uniform'
    # held-out injection (mirrors train_shard.py)
    with open(cfg['model']['aux_attribute_table_path']) as f:
        held = [int(a[1:]) for a in json.load(f)['held_out_classes']]
    cfg['dataset']['held_out_class_ids'] = held
    return cfg


def build_model(cfg):
    torch.manual_seed(SEED)
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = model.float()
    return model


def main():
    torch.set_grad_enabled(True)

    # ---------- 1. asset verification ----------
    hr('1. ensemble asset verification')
    actions = load_verbalizer_actions('configs/finegym_zsl_verbalizer.json')
    with open(ENSEMBLE) as f:
        ens = json.load(f)
    assert len(ens) == 99, f"ensemble has {len(ens)} keys, expected 99"
    lens = {len(v) for v in ens.values()}
    assert lens == {11}, f"sentence counts {lens}, expected {{11}}"
    key_of = resolve_ensemble_keys(actions, ens)
    n_exact = sum(1 for a in actions
                  if key_of[a['id']] == f"({a['activity']}) {a['name']}")
    print(f"ensemble: 99 classes x 11 sentences OK; "
          f"{n_exact}/99 exact '(ACT) name' key matches")
    for a in actions:
        k = key_of[a['id']]
        if k != f"({a['activity']}) {a['name']}":
            print(f"  disambiguated: c{a['id']:02d} (phrase "
                  f"{a['phrase_name']}) -> '{k}'")
            print(f"    sentence0: {ens[k][0][:100]}")
    # the duplicate must map flight-salto->v1 (aerial) and dismounts->v2
    a58 = key_of[58]
    a70 = key_of[70]
    assert a58 != a70
    assert 'dismount' in ' '.join(ens[a70]).lower(), \
        f"c70 (BB_dismounts) mapped to '{a70}' whose text lacks 'dismount'"
    print(f"  c58 (BB_flight_salto) -> '{a58}'")
    print(f"  c70 (BB_dismounts)    -> '{a70}'  [dismount keyword verified]")

    # ---------- 2+3. construction + real-batch forwards ----------
    hr('2. dataset (real batch)')
    cfg0 = load_cfg(CFGS['base'])
    torch.manual_seed(SEED)
    t0 = time.time()
    train_ds = make_dataset(cfg0['dataset_name'], True, cfg0['train_split'],
                            cfg0['model']['backbone_type'], cfg0['round'],
                            **cfg0['dataset'])
    print(f"train dataset: {len(train_ds)} windows ({time.time()-t0:.1f}s)")
    torch.manual_seed(SEED)
    items = [train_ds[3], train_ds[4]]
    for it in items:
        print(f"  item {it['video_id']}: feats {tuple(it['feats'].shape)}, "
              f"{len(it['labels'])} segs, labels {it['labels'].tolist()}")
    # an eval item: same windows, eval-style single-item batch
    eval_item = items[0]

    results_by_system = {}
    for name, cfg_path in CFGS.items():
        hr(f'3. system "{name}"')
        cfg = load_cfg(cfg_path)
        model = build_model(cfg)
        n_param = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tp = model.text_pathway
        print(f"constructed: {n_param/1e6:.1f}M params ({n_train/1e6:.1f}M trainable); "
              f"text_feats {tuple(tp.text_feats.shape)}, "
              f"stem_feats {tuple(tp.stem_feats.shape)}")
        if name == 'base':
            assert tuple(tp.text_feats.shape) == (99, 1, 512)
        if name == 'champ':
            assert tuple(tp.text_feats.shape) == (99, 11, 512)
            assert tp.attr_attn is not None
            assert float(tp.attr_attn.out_proj.weight.abs().max()) == 0.0
            assert model.as_uw_logvar is not None
            assert float(model.as_uw_logvar.abs().max()) == 0.0
            assert model.duration_prior is not None
        if name == 'tifad':
            assert model.text_tv_attn is not None and model.fg_head is not None

        # train-mode forward (loss)
        model.train()
        torch.manual_seed(SEED)
        t0 = time.time()
        losses = model([copy.deepcopy(it) for it in items])
        print(f"train fwd {time.time()-t0:.1f}s; losses:")
        for k, v in losses.items():
            v = float(v)
            print(f"    {k:.<28}{v:.4f}")
            assert v == v and abs(v) < 1e6, f"{name}: loss {k} not finite"
        if name == 'tifad':
            assert 'fg_loss' in losses
        if name == 'champ':
            assert 'as_loss_phrase' in losses and 'as_uw_s_main' in losses

        # eval-mode forward (inference incl. NMS decode; DP active for champ)
        model.eval()
        with torch.no_grad():
            t0 = time.time()
            res = model([copy.deepcopy(eval_item)])
        r = res[0]
        assert torch.isfinite(r['scores']).all() and \
            torch.isfinite(r['segments']).all()
        if len(r['scores']) > 0:
            rng = (f"score range [{float(r['scores'].min()):.4f}, "
                   f"{float(r['scores'].max()):.4f}]")
        else:
            # expected for the tifad system at INIT: decode score is
            # sigmoid(cls)*sigmoid(fg) ~ 0.01*0.01 = 1e-4, below the
            # pre-NMS threshold 1e-3 until the heads calibrate in training
            rng = "(none pass pre-NMS threshold at init — fg product score)"
        print(f"eval fwd {time.time()-t0:.1f}s; {len(r['scores'])} "
              f"detections {rng}")
        results_by_system[name] = r
        del model

    # ---------- 4. init-equality ----------
    hr('4. init-equality: champ(synthetic ensemble) vs base')
    # synthetic ensemble: all 11 sentences per class == the baseline prompt,
    # keyed exactly like the real ensemble
    os.makedirs(SCRATCH, exist_ok=True)
    syn = {key_of[a['id']]: [f"a video of action {a['name']}"] * 11
           for a in actions}
    syn_path = os.path.join(SCRATCH, 'synthetic_ensemble.json')
    with open(syn_path, 'w') as f:
        json.dump(syn, f)

    cfg_b = load_cfg(CFGS['base'])
    model_b = build_model(cfg_b)
    os.environ['FG3_TEXT_ENSEMBLE'] = syn_path
    os.environ['FG3_DP_GAMMA'] = '0'          # DP multiplier exp(0) == 1
    cfg_c = load_cfg(CFGS['champ'])
    model_c = build_model(cfg_c)
    del os.environ['FG3_TEXT_ENSEMBLE'], os.environ['FG3_DP_GAMMA']

    # shared trainable weights must be identical draws
    for k in ['text_pathway.proj.weight', 'text_pathway.proj.bias']:
        wb = dict(model_b.named_parameters())[k]
        wc = dict(model_c.named_parameters())[k]
        assert torch.equal(wb, wc), f"RNG divergence in shared param {k}"
    print("shared proj weights: identical draws OK")
    # class embeddings byte-equal (zero-init PA => emb == proj(sentence0))
    model_b.eval(); model_c.eval()
    with torch.no_grad():
        eb = model_b.text_pathway.class_embeddings()
        ec = model_c.text_pathway.class_embeddings()
    assert torch.equal(eb, ec), "class embeddings differ at init"
    print("class embeddings: byte-equal OK")
    with torch.no_grad():
        rb = model_b([copy.deepcopy(eval_item)])[0]
        rc = model_c([copy.deepcopy(eval_item)])[0]
    for k in ['segments', 'scores', 'labels']:
        assert torch.equal(rb[k], rc[k]), f"eval output '{k}' differs at init"
    # per-GT-token logits stored by the eval branch: byte compare
    lb = model_b.last_video_label_logits
    lc = model_c.last_video_label_logits
    for vid in lb:
        for cls_id in lb[vid]:
            for x, y in zip(lb[vid][cls_id], lc[vid][cls_id]):
                assert torch.equal(x, y)
    print("eval decode (segments/scores/labels) + GT-token logits: "
          "byte-equal OK")
    print("NOTE (documented deviation): with the REAL ensemble, sentence 0 "
          "is 'a video of a gymnastics element: {name}, ...' which differs "
          "from the baseline prompt — the HP prompt text itself is the only "
          "init-time difference; PA/AS/DP are exact no-ops at init.")
    del model_b, model_c

    # ---------- 5. DP unit test ----------
    hr('5. duration-prior unit test')
    dp = DurationPrior('configs/finegym_t3_duration_prior.json', gamma=0.3)
    stats = json.load(open('configs/finegym_t3_duration_prior.json'))
    import math
    cid = 0
    mu, sig = stats['classes'][str(cid)]['mu'], stats['classes'][str(cid)]['sigma']
    labels = torch.tensor([cid, cid, cid])
    scores = torch.ones(3)
    durs = torch.tensor([math.exp(mu), math.exp(mu + sig), math.exp(mu + 2 * sig)])
    out = dp.rescore(scores, labels, durs)
    exp = torch.tensor([1.0, math.exp(-0.5 * 0.3), math.exp(-0.5 * 0.3 * 4)])
    assert torch.allclose(out, exp, atol=1e-5), (out, exp)
    print(f"z=0/1/2 sigma multipliers: {[round(float(x),4) for x in out]} "
          f"== exp(-gamma/2 z^2) with gamma=0.3 OK")
    # held-out class prior comes from the Jaccard mixture (no train data)
    h = stats['meta']['held_out'][0]
    assert stats['classes'][str(h)]['kind'] == 'held_out'
    assert stats['classes'][str(h)]['n'] == 0
    print(f"held-out class {h}: mixture prior "
          f"(exp(mu)={math.exp(stats['classes'][str(h)]['mu']):.2f}s, "
          f"top seen weights {stats['classes'][str(h)]['top_seen_weights']}) OK")

    hr('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
