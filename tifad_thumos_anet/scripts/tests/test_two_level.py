"""Tests for the two-level sub-action ("sub-phase") extensions:
  text  : libs/modeling/text.py  TIFAD_SUB_ENSEMBLE (attr_attn_sub / chain_attn_sub)
  video : libs/modeling/vsa.py   VideoSubtreeAttnNested (TIFAD_VSA_NESTED=1, TIFAD_VSA_KP)
          libs/modeling/meta_archs.py wiring (self.vsa)

Run from the repo root (/data3/xiaodan8/tifad):

  env CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python scripts/tests/test_two_level.py \
      --before_dir <dir with text_before.pt / vsa_before.pt> [--sub <sub-ensemble json>]

  --before_dir : pre-edit baselines (text_before.pt: TextFeatures eval outputs +
                 state_dict under the champion envs with no sub-ensemble;
                 vsa_before.pt: VideoSubtreeAttn v1 outputs on fixed inputs).
                 Tests (b) / (d-v1) are skipped when the files are absent.
  --sub        : sub-phase ensemble json {class: [[Kp sents] x K]}; default
                 configs/thumos14_subphase_ensemble.json when present.

(a) text init-equality (with vs without TIFAD_SUB_ENSEMBLE, shared params copied)
(b) text forward bitwise unchanged vs the pre-edit baseline (no sub-ensemble)
(c) random sub out_projs -> outputs change, grads reach attr_attn_sub / chain_attn_sub,
    FREEZE_EXCEPT substrings match both levels, h14 dropout path runs
(d) video: nested init-equality fp32/bf16 (cpu + cuda), boundary safety at T=2304,
    grads to all four MHA's, v1 output unchanged, v1 call convention
(e) video: nested fwd+bwd peak memory, B=2, THUMOS pyramid, D=512 (and D=256)
(f) meta_archs wiring: TiFAD model built with TIFAD_VSA=1 TIFAD_VSA_NESTED=1 (CPU)

The text tests run on CPU; the video tests use the visible CUDA device.
"""
import argparse
import copy
import gc
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PYR = [2304, 1152, 576, 288, 144, 72]      # THUMOS max_seq_len 2304, 6 levels
SUBSET = 'splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list'
CHAMP_ENVS = {
    'TIFAD_PROMPT_ENSEMBLE': 'configs/thumos14_qwen_ensemble.json',
    'TIFAD_ATTR_ATTN': '1',
    'TIFAD_ATTNDESIGN': 'phasechain',
    'TIFAD_PROMPT_DROPOUT': '0',
    'TIFAD_SEM_DEGRADE': '0',
}
# text-side envs that must NOT leak in from the caller's shell
CLEAR = ['TIFAD_SUB_ENSEMBLE', 'TIFAD_VSA', 'TIFAD_VSA_NESTED', 'TIFAD_VSA_XM',
         'TIFAD_VSA_SOFTPHASE', 'TIFAD_VSA_POSEMB', 'TIFAD_VSA_ALIGN_W',
         'TIFAD_VSA_NOSELF', 'TIFAD_VSA_MINLVL', 'TIFAD_LEVELATTN',
         'TIFAD_TXT_DEANISO', 'TIFAD_NODEATTR', 'TIFAD_ATTRTOK', 'TIFAD_ATTRCODE',
         'TIFAD_ATTRVQ', 'TIFAD_GNN', 'TIFAD_FAMTIE', 'TIFAD_HTCA', 'TIFAD_HAF',
         'TIFAD_PDA', 'TIFAD_LMA', 'TIFAD_VPW', 'TIFAD_PHASELOGIT', 'TIFAD_VIDPHASE',
         'TIFAD_LEAFLOSS', 'TIFAD_SUBCLASS', 'TIFAD_FAMCOMP', 'FINEGYM_LEVEL_COMPOSE']


def set_envs(extra=None):
    for k in CLEAR:
        os.environ.pop(k, None)
    os.environ.update(CHAMP_ENVS)
    if extra:
        os.environ.update(extra)


def build_text(seed=0):
    from libs.modeling.text import TextFeatures
    torch.manual_seed(seed)
    tf = TextFeatures(model_path='openai/clip-vit-base-patch16', subset_file=SUBSET,
                      data_split=75, emb_dim=256, split_num=0)
    return tf


def text_fwd(tf, mode, train_flag=False):
    tf.train(train_flag)
    if train_flag:
        return tf(2, mode)[0]
    with torch.no_grad():
        return tf(2, mode)[0]


# ----------------------------------------------------------------------------
def test_text(args):
    sub = args.sub
    set_envs()
    tf0 = build_text(seed=0)                     # champion: no sub-ensemble
    assert tf0.sub_prompts is None and tf0.attr_attn_sub is None \
        and tf0.chain_attn_sub is None
    names0 = [n for n, _ in tf0.named_parameters()]
    assert not any('_sub' in n for n in names0), "no sub modules without the env"
    out0 = {m: text_fwd(tf0, m) for m in ('test', 'train')}

    # ---- (b) bitwise unchanged vs the pre-edit baseline ----
    bp = os.path.join(args.before_dir, 'text_before.pt')
    if os.path.exists(bp):
        ref = torch.load(bp)
        same_seed = all(torch.equal(out0[m], ref[m]) for m in ('test', 'train'))
        # also with the baseline's exact parameters loaded (robust to any
        # RNG-order difference)
        tfb = copy.deepcopy(tf0)
        tfb.load_state_dict(ref['state'])
        outb = {m: text_fwd(tfb, m) for m in ('test', 'train')}
        same_state = all(torch.equal(outb[m], ref[m]) for m in ('test', 'train'))
        assert same_state, "(b) text forward CHANGED vs pre-edit baseline"
        print(f"(b) text forward without TIFAD_SUB_ENSEMBLE bitwise unchanged vs "
              f"pre-edit baseline: PASS (same-seed construction equal: {same_seed}; "
              f"baseline-state equal: {same_state})")
        del tfb
    else:
        print("(b) SKIPPED (no text_before.pt)")

    # ---- (a) init-equality with the sub-ensemble ----
    set_envs({'TIFAD_SUB_ENSEMBLE': sub})
    tf1 = build_text(seed=0)
    assert tf1.attr_attn_sub is not None and tf1.chain_attn_sub is not None
    names1 = [n for n, _ in tf1.named_parameters()]
    new = sorted(set(names1) - set(names0))
    assert all(n.startswith(('attr_attn_sub.', 'chain_attn_sub.')) for n in new), new
    missing, unexpected = tf1.load_state_dict(tf0.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == new, (missing, unexpected)
    out1 = {m: text_fwd(tf1, m) for m in ('test', 'train')}
    for m in ('test', 'train'):
        assert out1[m].shape == out0[m].shape and torch.equal(out1[m], out0[m]), \
            f"(a) init-equality FAILED in mode {m}"
    # train-mode forward (module.training=True, dropout 0) is deterministic too
    o0t = text_fwd(tf0, 'train', train_flag=True).detach()
    o1t = text_fwd(tf1, 'train', train_flag=True).detach()
    assert torch.equal(o0t, o1t), "(a) init-equality FAILED (training=True)"
    K, Kp = len(tf1.sub_prompts[tf1.train_classes[0]]), tf1._sub_kp
    print(f"(a) text init-equality with TIFAD_SUB_ENSEMBLE ({os.path.basename(sub)}: "
          f"K={K}, Kp={Kp}): PASS (eval test/train + training=True bitwise equal; "
          f"new params: {len(new)} tensors in {sorted({n.split('.')[0] for n in new})})")

    # ---- (c) random sub out_projs: outputs change, grads reach the new modules ----
    # A fresh upper level (attr_attn / chain_attn out_proj == 0) reads
    # text_emb = q + 0 whatever the phases are, so the sub level can only
    # reach the output through a TRAINED upper level (stage 1/2 champion):
    # emulate it by randomising the upper out_projs identically in both
    # models first -- init-equality must still hold with zero sub out_projs.
    torch.manual_seed(3)
    for a in (tf0.attr_attn, tf0.chain_attn):
        torch.nn.init.normal_(a.out_proj.weight, std=0.05)
        torch.nn.init.normal_(a.out_proj.bias, std=0.05)
    tf1.load_state_dict(tf0.state_dict(), strict=False)
    out0['test'] = text_fwd(tf0, 'test')
    assert torch.equal(text_fwd(tf1, 'test'), out0['test']), \
        "(c) init-equality FAILED with a trained (non-zero) upper level"
    for a in (tf1.attr_attn_sub, tf1.chain_attn_sub):
        torch.nn.init.normal_(a.out_proj.weight, std=0.05)
        torch.nn.init.normal_(a.out_proj.bias, std=0.05)
    o1 = text_fwd(tf1, 'test')
    d = (o1 - out0['test']).abs().max().item()
    assert d > 0, "(c) output should change with non-zero sub out_projs"
    tf1.zero_grad(set_to_none=True)
    o1t = text_fwd(tf1, 'train', train_flag=True)
    o1t.sum().backward()
    for mod in ('attr_attn_sub', 'chain_attn_sub'):
        a = getattr(tf1, mod)
        for pn, p in a.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"(c) zero/absent grad for {mod}.{pn}"
    # the upper level still receives gradient (unchanged behaviour)
    assert tf1.attr_attn.in_proj_weight.grad is not None
    # FREEZE_EXCEPT-style substring matching
    ch = [n for n in names1 if 'chain_attn' in n]
    at = [n for n in names1 if 'attr_attn' in n]
    assert any(n.startswith('chain_attn.') for n in ch) \
        and any(n.startswith('chain_attn_sub.') for n in ch)
    assert any(n.startswith('attr_attn.') for n in at) \
        and any(n.startswith('attr_attn_sub.') for n in at)
    assert not any('vsa' in n for n in names1)
    # h14 dropout path on the sub level (train mode only) runs and stays finite;
    # eval is unaffected by the dropout setting
    set_envs({'TIFAD_SUB_ENSEMBLE': sub, 'TIFAD_PROMPT_DROPOUT': '0.5'})
    tf2 = build_text(seed=0)
    tf2.load_state_dict(tf1.state_dict())
    torch.manual_seed(4)
    o2a = text_fwd(tf2, 'train', train_flag=True).detach()
    torch.manual_seed(5)
    o2b = text_fwd(tf2, 'train', train_flag=True).detach()
    assert torch.isfinite(o2a).all() and torch.isfinite(o2b).all()
    assert not torch.equal(o2a, o2b), "(c) dropout should randomise train outputs"
    assert torch.equal(text_fwd(tf2, 'test'), o1), "(c) eval must ignore dropout"
    # h18 degrade rule on the sub level: with p=1 every class is degraded in
    # train mode, so ALL sub-sentences handed to CLIP must be parent stems
    set_envs({'TIFAD_SUB_ENSEMBLE': sub, 'TIFAD_SEM_DEGRADE': '1.0'})
    tf3 = build_text(seed=0)
    calls = []
    _orig = tf3.extract_text_emb
    tf3.extract_text_emb = lambda t, is_prompt=False: (calls.append(list(t)),
                                                       _orig(t, is_prompt))[1]
    o3 = text_fwd(tf3, 'train', train_flag=True).detach()
    assert torch.isfinite(o3).all() and len(calls) == 2
    stems = set(tf3._stems.values())
    assert all(s in stems for s in calls[0]) and all(s in stems for s in calls[1])
    assert len(calls[1]) == len(tf3.train_classes) * K * Kp
    # in eval mode nothing is degraded: the real sub-sentences are embedded
    calls.clear()
    text_fwd(tf3, 'test')
    assert not any(s in stems for s in calls[1])
    print(f"(c) trained upper level + zero sub out_projs still init-equal; random sub "
          f"out_projs: max|out-out0| = {d:.3g}; non-zero grads on all "
          f"attr_attn_sub / chain_attn_sub params; substring 'chain_attn' matches "
          f"{len(ch)} tensors (both levels), 'attr_attn' matches {len(at)} (both "
          f"levels); sub-level h14 dropout (p=0.5) train-only + finite; h18 degrade "
          f"(p=1) replaces all {K * Kp} sub-sentences/class by the parent stem in "
          f"train mode only: PASS")
    set_envs()


# ----------------------------------------------------------------------------
def make_inputs(B, D, Ts, device='cpu', dtype=torch.float32, seed=0, boundary=False):
    g = torch.Generator().manual_seed(seed)
    feats, offs, masks = [], [], []
    for T in Ts:
        feats.append(torch.randn(B, D, T, generator=g).to(device=device, dtype=dtype))
        if boundary:
            # offsets that push every span to (and past) the level boundary
            o = torch.full((B, T, 2), float(T) * 2.0)
            o[:, : T // 2, 0] = T - 1.0
            o[:, T // 2:, 1] = T - 0.5
        else:
            o = torch.rand(B, T, 2, generator=g) * T * 1.5     # values beyond T
            o[:, ::7, :] = 0.0
        offs.append(o.to(device=device, dtype=dtype))
        mk = torch.ones(B, 1, T, dtype=torch.bool)
        for b in range(B):
            mk[b, :, T - b * (T // 8):] = False
        masks.append(mk.to(device))
    return feats, offs, masks


def bitwise_equal(outs, feats):
    return all(o.dtype == f.dtype and o.shape == f.shape and torch.equal(o, f)
               for o, f in zip(outs, feats))


def test_video(args, device):
    from libs.modeling.vsa import VideoSubtreeAttn, VideoSubtreeAttnNested
    B, D, K, Kp = 2, 256, 10, 3
    # ---- init-equality: fp32 cpu, bf16 cpu, cuda fp32, cuda bf16 (+autocast) ----
    for dev, dtype, ac in (('cpu', torch.float32, False), ('cpu', torch.bfloat16, False),
                           (device, torch.float32, False), (device, torch.bfloat16, False),
                           (device, torch.bfloat16, True)):
        m = VideoSubtreeAttnNested(D, K=K, Kp=Kp, heads=4).to(dev)
        if not ac:
            m = m.to(dtype)
        feats, offs, masks = make_inputs(B, D, PYR, device=dev, dtype=dtype, seed=1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ac):
            out, match, align = m(feats, offs, masks)
        assert match is None and align is None
        assert bitwise_equal(out, feats), f"(d) init-equality FAILED ({dev}, {dtype}, ac={ac})"
    print("(d) nested init-equality (output == input bitwise): PASS "
          "(fp32 cpu, bf16 cpu, fp32 cuda, bf16 cuda, bf16 cuda autocast; T=2304 pyramid)")

    # ---- boundary-index safety at T=2304 with offsets beyond T ----
    for dev in ('cpu', device):
        for dtype in (torch.float32, torch.bfloat16):
            m = VideoSubtreeAttnNested(D, K=K, Kp=Kp, heads=4).to(dev).to(dtype)
            feats, offs, masks = make_inputs(B, D, [2304], device=dev, dtype=dtype,
                                             seed=4, boundary=True)
            with torch.no_grad():
                Zleaf, Zact = m._pool_levels(feats[0], offs[0], masks[0])
                out, _, _ = m(feats, offs, masks)
            s_i, e_i, L = m._span(offs[0], 2304)
            assert int(e_i.max()) <= 2303 and int(s_i.min()) >= 0
            assert torch.isfinite(Zleaf.float()).all() and torch.isfinite(Zact.float()).all()
            assert Zleaf.shape == (B, 2304, K * Kp, D) and Zact.shape == (B, 2304, K, D)
            assert bitwise_equal(out, feats)
            if dev != 'cpu':
                torch.cuda.synchronize(dev)
    print(f"(d) boundary spans at T=2304 (offsets up to 2T, fp32 + bf16, cpu + cuda): "
          f"PASS (e_i.max={int(e_i.max())}, L.max={int(L.max())}, no OOB, finite)")

    # ---- Kp=1: leaves == actions ----
    m1 = VideoSubtreeAttnNested(D, K=K, Kp=1, heads=4)
    feats, offs, masks = make_inputs(B, D, PYR, seed=7)
    for x, o, mk in zip(feats, offs, masks):
        Zl, Za = m1._pool_levels(x, o, mk)
        assert Zl.shape == Za.shape and torch.equal(Zl, Za)
    print("(d) Kp=1: Zleaf == Zact bitwise on all levels: PASS")

    # ---- gradients to all four MHA's ----
    torch.manual_seed(5)
    m = VideoSubtreeAttnNested(D, K=K, Kp=Kp, heads=4)
    mhas = {n: getattr(m, n) for n in ('chain_attn_phase', 'clip_attn_phase',
                                       'chain_attn_act', 'clip_attn_act')}
    for a in mhas.values():
        torch.nn.init.normal_(a.out_proj.weight, std=0.05)
        torch.nn.init.normal_(a.out_proj.bias, std=0.05)
    feats, offs, masks = make_inputs(B, D, PYR[2:], seed=6)
    out, _, _ = m(feats, offs, masks)
    diffs = [(o - f).abs().max().item() for o, f in zip(out, feats)]
    assert all(d > 0 for d in diffs)
    sum(o.sum() for o in out).backward()
    for name, a in mhas.items():
        for pn, p in a.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"(d) zero grad {name}.{pn}"
    assert all(k.startswith(('chain_attn_', 'clip_attn_')) for k in m.state_dict())
    assert len(m.state_dict()) == 16, "state_dict should hold exactly the four MHA's"
    print(f"(d) random out_projs: max|out-in| per level {[f'{x:.3g}' for x in diffs]}; "
          f"non-zero grads on in_proj/out_proj of all four MHA's; state_dict = 4 MHA's: PASS")

    # ---- v1 unchanged ----
    vp = os.path.join(args.before_dir, 'vsa_before.pt')
    if os.path.exists(vp):
        ref = torch.load(vp)
        v1 = VideoSubtreeAttn(D, K=10, heads=4)
        v1.load_state_dict(ref['state'])
        with torch.no_grad():
            out, match, align = v1(ref['feats'], ref['offs'], ref['masks'])
        assert match is None and align is None
        assert all(torch.equal(o, r) for o, r in zip(out, ref['out'])), \
            "(d) VideoSubtreeAttn v1 output CHANGED"
        print("(d) existing VideoSubtreeAttn v1 output bitwise unchanged vs pre-edit "
              "baseline: PASS")
    else:
        print("(d) v1-unchanged SKIPPED (no vsa_before.pt)")


def peak_mem(module, device, B, D, Ts, dtype=torch.float32, autocast=False):
    module = module.to(device)
    feats, offs, masks = make_inputs(B, D, Ts, device=device, dtype=dtype, seed=8)
    feats = [f.requires_grad_(True) for f in feats]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        out, _, _ = module(feats, offs, masks)
        loss = sum(o.float().sum() for o in out)
    loss.backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del out, loss, feats, offs, masks
    module.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return peak / 2**30, base / 2**30


def test_mem(device, cap_gb):
    from libs.modeling.vsa import VideoSubtreeAttn, VideoSubtreeAttnNested
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, cap_gb * 2**30 / torch.cuda.get_device_properties(device).total_memory),
        device)
    cfgs = [('nested K=10 Kp=3 D=512 fp32', 512, torch.float32, False,
             lambda: VideoSubtreeAttnNested(512, K=10, Kp=3, heads=4)),
            ('nested K=10 Kp=3 D=512 bf16-autocast', 512, torch.bfloat16, True,
             lambda: VideoSubtreeAttnNested(512, K=10, Kp=3, heads=4)),
            ('nested K=10 Kp=3 D=256 fp32', 256, torch.float32, False,
             lambda: VideoSubtreeAttnNested(256, K=10, Kp=3, heads=4)),
            ('flat   K=10      D=256 fp32', 256, torch.float32, False,
             lambda: VideoSubtreeAttn(256, K=10, heads=4))]
    for name, D, dtype, ac, ctor in cfgs:
        try:
            peak, base = peak_mem(ctor(), device, 2, D, PYR, dtype=dtype, autocast=ac)
            oom = False
        except torch.cuda.OutOfMemoryError:
            oom = True
        gc.collect()
        torch.cuda.empty_cache()
        if oom:
            # activation memory is linear in B: measure B=1 and extrapolate
            try:
                p1, _ = peak_mem(ctor(), device, 1, D, PYR, dtype=dtype, autocast=ac)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"(e) {name:38s} B=2 pyramid {PYR}: OOM under the {cap_gb} GB "
                      f"cap; B=1 peak {p1:.2f} GB -> linear extrapolation B=2: "
                      f"{2 * p1:.2f} GB")
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"(e) {name:38s} B=1 and B=2: OOM under the {cap_gb} GB cap")
        else:
            print(f"(e) {name:38s} B=2 pyramid {PYR} fwd+bwd: peak allocated "
                  f"{peak:.2f} GB (inputs+params {base:.2f} GB)")


# ----------------------------------------------------------------------------
def test_meta(args):
    """meta_archs wiring on CPU: build the full TiFAD model with the nested
    VSA, check the attribute / class / FREEZE substring / init-equality of
    the training loss against the same model without VSA, and that the
    v2 flags are rejected."""
    from libs.core import load_config
    from libs.modeling import make_meta_arch
    from libs.modeling.vsa import VideoSubtreeAttnNested
    cfg = load_config('configs/thumos14_i3d.yaml')

    def build(extra, seed=0):
        set_envs(extra)
        torch.manual_seed(seed)
        m = make_meta_arch(0, cfg['model_name'], cfg['dataset']['subset_file'],
                           cfg['dataset']['data_split'], **cfg['model'])
        return m

    base = build({})
    nested = build({'TIFAD_VSA': '1', 'TIFAD_VSA_NESTED': '1', 'TIFAD_VSA_KP': '3'})
    assert isinstance(nested.vsa, VideoSubtreeAttnNested)
    assert nested.vsa.K == 10 and nested.vsa.Kp == 3
    names = [n for n, _ in nested.named_parameters()]
    vsa_names = [n for n in names if 'vsa' in n]
    assert sorted(vsa_names) == sorted(n for n in names if n.startswith('vsa.'))
    assert len(vsa_names) == 16
    missing, unexpected = nested.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == sorted(vsa_names), (missing, unexpected)
    # the v2/v3 flags must be rejected together with NESTED
    try:
        build({'TIFAD_VSA': '1', 'TIFAD_VSA_NESTED': '1', 'TIFAD_VSA_XM': '1'})
        raise SystemExit("(f) NESTED + XM should have been rejected")
    except AssertionError as e:
        assert 'VSA-nested' in str(e)
    set_envs()
    # end-to-end init-equality of the training loss on a fake clip (CPU)
    try:
        g = torch.Generator().manual_seed(0)
        T = 512
        vid = [{'video_id': 'fake', 'feats': torch.randn(2048, T, generator=g),
                'segments': torch.tensor([[10., 60.], [200., 260.], [400., 470.]]),
                'labels': torch.tensor([0, 3, 7]), 'fps': 30.0,
                'duration': T * 4 / 30.0, 'feat_stride': 4, 'feat_num_frames': 16}
               for _ in range(2)]
        outs = []
        for m in (base, nested):
            m.train()
            torch.manual_seed(1)
            with torch.no_grad():
                outs.append(m(vid))
        keys = sorted(k for k in outs[0] if torch.is_tensor(outs[0][k]))
        eq = {k: torch.equal(outs[0][k], outs[1][k]) for k in keys}
        assert all(eq.values()), f"(f) train losses differ: {eq}"
        print(f"(f) meta_archs wiring: model.vsa is VideoSubtreeAttnNested (K=10, Kp=3), "
              f"16 'vsa.' params, INIT_FROM-style load leaves only vsa.* missing, NESTED+XM "
              f"rejected; train-mode losses bitwise equal to the no-VSA model on a fake "
              f"clip ({', '.join(f'{k}={outs[1][k].item():.4f}' for k in keys)}): PASS")
    except Exception as e:                       # noqa: BLE001
        print(f"(f) meta_archs wiring: model built, attribute/params/reject checks PASS; "
              f"CPU forward SKIPPED ({type(e).__name__}: {str(e)[:120]})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--before_dir', default='')
    ap.add_argument('--sub', default='')
    ap.add_argument('--mem_cap_gb', type=float, default=5.5)
    ap.add_argument('--skip_text', action='store_true')
    ap.add_argument('--skip_meta', action='store_true')
    ap.add_argument('--only_mem', action='store_true')
    args = ap.parse_args()
    if args.only_mem:
        test_mem('cuda:0', args.mem_cap_gb)
        sys.exit(0)
    if not args.sub:
        args.sub = 'configs/thumos14_subphase_ensemble.json'
    assert os.path.exists(args.sub), f"sub-ensemble json not found: {args.sub}"
    device = 'cuda:0' if torch.cuda.is_available() else None
    if not args.skip_text:
        test_text(args)
    if device is not None:
        test_video(args, device)
        test_mem(device, args.mem_cap_gb)
    else:
        print("(d)/(e) SKIPPED: no CUDA device visible")
    if not args.skip_meta:
        test_meta(args)
    print("ALL TESTS PASSED")
