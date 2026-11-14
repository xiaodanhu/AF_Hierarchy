"""Tests for libs/modeling/vsa.py VideoSubtreeAttnNested (and the refactor of
VideoSubtreeAttn._pool_chunks). Run:

  python scripts/tests/test_vsa_nested.py [--before PATH] [--gpu N]

  --before : a .pt saved by the pre-edit VideoSubtreeAttn on a fixed seed
             (test e; skipped when absent)
  --gpu    : CUDA device for the bf16-autocast / memory tests (default 0)

(a) init-equality fp32 + bf16 (+ autocast on GPU)   (d) Kp=1: Zleaf == Zact
(b) no out-of-bounds at level boundary, bf16, K=3/Kp=10  (e) flat module unchanged
(c) random out_proj -> output != input, all 4 MHA's get grads  (f) peak GPU memory
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from libs.modeling.vsa import VideoSubtreeAttn, VideoSubtreeAttnNested  # noqa: E402

PYR = [288, 144, 72, 36]


def make_inputs(B, D, Ts, device='cpu', dtype=torch.float32, seed=0,
                boundary=False):
    g = torch.Generator().manual_seed(seed)
    feats, offs, masks = [], [], []
    for T in Ts:
        feats.append(torch.randn(B, D, T, generator=g).to(device=device, dtype=dtype))
        if boundary:
            # offsets that push every span to (and past) the level boundary
            o = torch.full((B, T, 2), float(T) * 2.0)
            o[:, : T // 2, 0] = T - 1.0          # exact-boundary values
            o[:, T // 2:, 1] = T - 0.5
        else:
            # non-negative offsets, grid units, with values > T
            o = torch.rand(B, T, 2, generator=g) * T * 1.5
            o[:, ::7, :] = 0.0
        offs.append(o.to(device=device, dtype=dtype))
        mk = torch.ones(B, 1, T, dtype=torch.bool)
        for b in range(B):
            mk[b, :, T - b * (T // 8):] = False   # padding at the end
        masks.append(mk.to(device))
    return feats, offs, masks


def bitwise_equal(outs, feats):
    return all(o.dtype == f.dtype and o.shape == f.shape and torch.equal(o, f)
               for o, f in zip(outs, feats))


def test_a(device):
    B, D = 4, 512
    m = VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4)
    # fp32, cpu
    feats, offs, masks = make_inputs(B, D, PYR, seed=1)
    with torch.no_grad():
        out = m(feats, offs, masks)
    assert bitwise_equal(out, feats), "(a) fp32 init-equality FAILED"
    # bf16 tensors + bf16 module, cpu
    mb = VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4).to(torch.bfloat16)
    feats, offs, masks = make_inputs(B, D, PYR, dtype=torch.bfloat16, seed=2)
    with torch.no_grad():
        out = mb(feats, offs, masks)
    assert bitwise_equal(out, feats), "(a) bf16 init-equality FAILED"
    # GPU autocast bf16 with bf16 inputs (fp32 params)
    mg = VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4).to(device)
    feats, offs, masks = make_inputs(B, D, PYR, device=device, dtype=torch.bfloat16, seed=3)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = mg(feats, offs, masks)
    assert bitwise_equal(out, feats), "(a) GPU autocast bf16 init-equality FAILED"
    print("(a) init-equality: PASS (fp32 cpu, bf16 cpu, bf16 tensors + cuda autocast)")


def test_b(device):
    B, D = 4, 512
    for dev in ('cpu', device):
        m = VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4).to(dev).to(torch.bfloat16)
        feats, offs, masks = make_inputs(B, D, [288], device=dev, dtype=torch.bfloat16,
                                         seed=4, boundary=True)
        with torch.no_grad():
            Zleaf, Zact = m._pool_levels(feats[0], offs[0], masks[0])
            out = m(feats, offs, masks)
        s_i, e_i, L = m._span(offs[0], 288)
        assert int(e_i.max()) <= 287 and int(s_i.min()) >= 0
        assert torch.isfinite(Zleaf.float()).all() and torch.isfinite(Zact.float()).all()
        assert bitwise_equal(out, feats)
        if dev != 'cpu':
            torch.cuda.synchronize(dev)
    print(f"(b) boundary spans, bf16, K=3 Kp=10, T=288: PASS (no OOB; e_i.max={int(e_i.max())}, "
          f"L.max={int(L.max())}, Zleaf {tuple(Zleaf.shape)} finite)")


def test_c(device):
    B, D = 4, 512
    torch.manual_seed(5)
    m = VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4)
    mhas = {'chain_attn_phase': m.chain_attn_phase, 'clip_attn_phase': m.clip_attn_phase,
            'chain_attn_act': m.chain_attn_act, 'clip_attn_act': m.clip_attn_act}
    for a in mhas.values():
        torch.nn.init.normal_(a.out_proj.weight, std=0.05)
        torch.nn.init.normal_(a.out_proj.bias, std=0.05)
    feats, offs, masks = make_inputs(B, D, PYR, seed=6)
    out = m(feats, offs, masks)
    diffs = [(o - f).abs().max().item() for o, f in zip(out, feats)]
    assert all(d > 0 for d in diffs), "(c) output should differ from input"
    sum(o.sum() for o in out).backward()
    for name, a in mhas.items():
        for pn in ('in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias'):
            p = dict(a.named_parameters())[pn]
            assert p.grad is not None and p.grad.abs().sum() > 0, f"(c) zero grad {name}.{pn}"
    print(f"(c) random out_proj: max|out-in| per level = {[f'{d:.3g}' for d in diffs]}; "
          f"non-zero grads for in_proj/out_proj of all 4 MHA's: PASS")


def test_d():
    B, D = 4, 512
    m = VideoSubtreeAttnNested(D, K=5, Kp=1, heads=4)
    feats, offs, masks = make_inputs(B, D, PYR, seed=7)
    for x, o, mk in zip(feats, offs, masks):
        Zleaf, Zact = m._pool_levels(x, o, mk)
        assert Zleaf.shape == Zact.shape and torch.equal(Zleaf, Zact)
    # and the K*Kp leaves, when Kp>1, average back to the action means only
    # where the widths are equal (sanity: shapes)
    m2 = VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4)
    Zleaf, Zact = m2._pool_levels(feats[0], offs[0], masks[0])
    assert Zleaf.shape == (B, 288, 30, D) and Zact.shape == (B, 288, 3, D)
    print("(d) Kp=1: Zleaf == Zact bitwise on all 4 levels: PASS")


def test_e(before_path):
    if not before_path or not os.path.exists(before_path):
        print("(e) SKIPPED (no --before file)")
        return
    ref = torch.load(before_path)
    m = VideoSubtreeAttn(512, K=10, heads=4)
    m.load_state_dict(ref['state'])
    with torch.no_grad():
        out = m(ref['feats'], ref['offs'], ref['masks'])
    ok = all(torch.equal(o, r) for o, r in zip(out, ref['out']))
    assert ok, "(e) existing VideoSubtreeAttn output CHANGED"
    # also: the generalised _pool_chunks with explicit K equals the default
    with torch.no_grad():
        z1 = m._pool_chunks(ref['feats'][0], ref['offs'][0], ref['masks'][0])
        z2 = m._pool_chunks(ref['feats'][0], ref['offs'][0], ref['masks'][0], K=10)
    assert torch.equal(z1, z2)
    print("(e) existing VideoSubtreeAttn output bitwise unchanged vs pre-edit baseline: PASS")


def peak_mem(module, device, B, D, Ts, cap_gb=None):
    module = module.to(device)
    feats, offs, masks = make_inputs(B, D, Ts, device=device, dtype=torch.bfloat16, seed=8)
    feats = [f.requires_grad_(True) for f in feats]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = module(feats, offs, masks)
        loss = sum(o.float().sum() for o in out)
    loss.backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del out, loss, feats, offs, masks
    module.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return peak / 2**30, base / 2**30


def test_f(device, cap_gb):
    D = 512
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, cap_gb * 2**30 / torch.cuda.get_device_properties(device).total_memory), device)
    configs = [('nested K=3 Kp=10', lambda: VideoSubtreeAttnNested(D, K=3, Kp=10, heads=4)),
               ('nested K=10 Kp=10', lambda: VideoSubtreeAttnNested(D, K=10, Kp=10, heads=4)),
               ('flat   K=10', lambda: VideoSubtreeAttn(D, K=10, heads=4))]
    for name, ctor in configs:
        oom = False
        try:
            peak, base = peak_mem(ctor(), device, 48, D, PYR)
        except torch.cuda.OutOfMemoryError:
            oom = True
        # leave the except block first: the live exception's traceback pins
        # every tensor of the failed forward until it is released
        gc.collect()
        torch.cuda.empty_cache()
        if not oom:
            print(f"(f) {name:18s} B=48 bf16-autocast fwd+bwd: peak allocated {peak:.2f} GB "
                  f"(inputs+params {base:.2f} GB)")
            continue
        # module memory is linear in B: measure at two small B and extrapolate
        # (halve the probe sizes while they still exceed the cap)
        probe = (6, 12)
        while True:
            pts, oom = [], False
            for b in probe:
                try:
                    peak, base = peak_mem(ctor(), device, b, D, PYR)
                    pts.append((b, peak))
                except torch.cuda.OutOfMemoryError:
                    oom = True
                gc.collect()
                torch.cuda.empty_cache()
                if oom:
                    break
            if not oom:
                break
            probe = (probe[0] // 2, probe[1] // 2)
            assert probe[0] >= 1, "(f) cannot fit even B=2 under the cap"
        slope = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
        est = pts[1][1] + slope * (48 - pts[1][0])
        print(f"(f) {name:18s} B=48 exceeds the {cap_gb} GB test cap; measured "
              f"B={pts[0][0]}: {pts[0][1]:.2f} GB, B={pts[1][0]}: {pts[1][1]:.2f} GB -> "
              f"linear extrapolation B=48: {est:.2f} GB")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', default='')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--mem_cap_gb', type=float, default=7.5)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'
    test_e(args.before)
    test_a(device)
    test_b(device)
    test_c(device)
    test_d()
    test_f(device, args.mem_cap_gb)
    print("ALL TESTS PASSED")
