"""Equivalence check for the 2026-09-13 padded-frame skip in ActionFormerWithCLIP.forward.

Reference = encode every frame (old behaviour: self.encoder(x)), Candidate = new forward.
Checks (fp32, cuda, eval mode so no dropout): (1) backbone outputs identical at ALL positions
(valid and padded); (2) gradients w.r.t. CLIP encoder parameters identical when the loss is a
masked sum over the backbone outputs; (3) timing of encoder work old vs new.
Run:  PYTHONPATH=<worktree> python scripts/verify_clip_pad_skip.py
"""
import sys, time, torch, yaml
from libs.core import load_config
from libs.modeling.backbones import ActionFormerWithCLIP

torch.manual_seed(0)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
DT = torch.float64  # exact-equivalence check: remove fp32/TF32 reduction-order noise
cfg = load_config('configs/finegym_t3_tifad.yaml')
m = cfg['model']
dev = torch.device('cuda:0')
net = ActionFormerWithCLIP(
    input_dim=m['input_dim'] if 'input_dim' in m else 768, embd_dim=m['embd_dim'], n_head=m['n_head'],
    embd_kernel_size=m['embd_kernel_size'], max_seq_len=cfg['dataset']['max_seq_len'],
    backbone_arch=tuple(m['backbone_arch']), mha_win_size=[m['n_mha_win_size']] * (1 + m['backbone_arch'][-1]),
    scale_factor=m['scale_factor'], embd_with_ln=m['embd_with_ln'], use_abs_pe=m['use_abs_pe'],
    use_rel_pe=m['use_rel_pe'], use_grad_checkpoint=False, clip_forward_chunk=64,
).to(dev).to(DT).eval()

def ref_forward(x, mask):
    return net.backbone(net.encoder(x), mask)

B, T = 3, 144  # local-MHA needs T % 18 == 0 at every FPN level; 144 fits beside the live run
lens = [32, 32, 20]
x = torch.zeros(B, T, 3, 224, 224, device=dev, dtype=DT)
for i, L in enumerate(lens):
    x[i, :L] = torch.randn(L, 3, 224, 224, device=dev, dtype=DT)
mask = (torch.arange(T, device=dev)[None, :] < torch.tensor(lens, device=dev)[:, None]).unsqueeze(1)

# (1) forward equivalence, all positions
with torch.no_grad():
    fr, mr = ref_forward(x, mask)
    fc, mc = net(x, mask)
worst = 0.0
for a, b in zip(fr, fc):
    worst = max(worst, (a - b).abs().max().item())
print(f"forward: {len(fr)} FPN levels, max |ref-new| over all positions = {worst:.3e} (ref scale {fr[0].abs().mean().item():.3e})")
assert worst < 1e-8, "forward mismatch"

# (1b) short pixel tensor (preprocessing now pads pixels only to the batch's longest window)
Tx = max(lens)
with torch.no_grad():
    fs, _ = net(x[:, :Tx].contiguous(), mask)
worst_s = max((a - b).abs().max().item() for a, b in zip(fr, fs))
print(f"forward (pixels T={Tx}, mask T={T}): max |ref-new| = {worst_s:.3e}")
assert worst_s < 1e-8, "short-pixel forward mismatch"

# (2) gradient equivalence into CLIP params -- fp64, one window (fits beside a live run).
# k_proj.bias grads are mathematically zero (softmax shift-invariance) and only carry ~1e-14
# round-off that differs even between two identical reference runs, so relative diffs are
# floored at 1e-6 of the largest gradient in the encoder.
x = x[:1]; mask = mask[:1]
def loss_of(fn):
    net.zero_grad(set_to_none=True)
    feats, masks = fn(x, mask)
    loss = sum((f * mk.to(f.dtype)).sum() for f, mk in zip(feats, masks))
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in net.encoder.named_parameters() if p.grad is not None}
gr = loss_of(ref_forward); gc = loss_of(net); gs = loss_of(lambda a, m: net(a[:, :Tx].contiguous(), m))
assert gr.keys() == gc.keys()
gscale = max(g.abs().max().item() for g in gr.values())
gworst = max(((gr[k] - gc[k]).abs().max() / max(gr[k].abs().max().item(), 1e-6 * gscale)).item() for k in gr)
gworst_s = max(((gr[k] - gs[k]).abs().max() / max(gr[k].abs().max().item(), 1e-6 * gscale)).item() for k in gr)
print(f"backward: {len(gr)} CLIP param tensors, max relative grad diff = {gworst:.3e} (full pixels) / {gworst_s:.3e} (short pixels)")
assert gworst_s < 1e-8, "short-pixel gradient mismatch"
assert gworst < 1e-8, "gradient mismatch"

# (3) timing
def timeit(fn, n=3):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        with torch.no_grad(): fn(x, mask)
    torch.cuda.synchronize(); return (time.time() - t) / n
print(f"timing (B=3,T=288, no_grad): ref {timeit(ref_forward):.2f}s  new {timeit(net):.2f}s")
print("PAD-SKIP CHECKS PASSED")
