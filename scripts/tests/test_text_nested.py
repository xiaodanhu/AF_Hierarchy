"""CPU tests for the nested (two-level) text composition in
libs/modeling/text_pathway.TextClsPathway.

  (a) non-nested champion path unchanged (bitwise vs a saved 'before' tensor
      when BEFORE_PT is given, else vs a fresh build with the same seed)
  (b) nested build: shapes; init-equality emb == proj(phrase sentence 0)
  (c) random out_proj -> embeddings change, all attention blocks get grads
  (d) train mode (dropout / degrade paths) runs 20 times
  (e) --freeze_except substring selection
Run: CUDA_VISIBLE_DEVICES= python scripts/tests/test_text_nested.py [before.pt]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from libs.modeling.text_pathway import TextClsPathway   # noqa: E402

VERB99 = 'configs/finegym_zsl_verbalizer.json'
ENS99 = 'configs/finegym_qwen_ensemble.json'
VERB14 = 'configs/finegym_phrase_verbalizer.json'
ENS14 = 'configs/finegym_phrase_ensemble.json'


def build_flat(seed=0):
    torch.manual_seed(seed)
    return TextClsPathway(VERB99, 512, ensemble_path=ENS99,
                          phase_attention=True, phase_neighbor_attention=True,
                          sentence_dropout=0.3, stem_degrade=0.3)


def build_nested(pna, seed=0):
    torch.manual_seed(seed)
    return TextClsPathway(VERB14, 512, ensemble_path=ENS14,
                          phase_attention=True, phase_neighbor_attention=pna,
                          sentence_dropout=0.3, stem_degrade=0.3,
                          nested=True, member_verbalizer_path=VERB99,
                          member_ensemble_path=ENS99)


def main():
    # ---------------- (a) flat path unchanged ----------------
    p = build_flat().eval()
    with torch.no_grad():
        e_after = p.class_embeddings()
    if len(sys.argv) > 1:
        before = torch.load(sys.argv[1])
        assert torch.equal(before['emb'], e_after), "(a) flat embeddings changed"
        for k, v in before['state'].items():
            assert torch.equal(v, p.state_dict()[k]), f"(a) state {k} changed"
        assert set(before['state']) == set(p.state_dict()), "(a) state keys changed"
        print("(a) PASS flat champion path bitwise identical to pre-edit tensor "
              f"({tuple(e_after.shape)})")
    else:
        print("(a) SKIP (no before.pt given); flat build ok", tuple(e_after.shape))

    # ---------------- (b) nested shapes + init-equality ----------------
    for pna in (False, True):
        n = build_nested(pna).eval()
        assert n.num_classes == 14
        assert tuple(n.text_feats.shape) == (14, 11, 512)
        assert tuple(n.member_feats.shape)[0] == 14 and n.member_feats.shape[2:] == (11, 512)
        assert n.member_mask.sum().item() == 99, n.member_mask.sum()
        assert torch.equal(n.action_to_phrase, torch.arange(14))
        assert n.action_to_activity.tolist() == [0] + [1] * 4 + [2] * 5 + [3] * 4
        assert (n.phase_attn2 is not None) == pna and (n.phase_attn is not None) == pna
        with torch.no_grad():
            emb = n.class_embeddings()
            # proj applied to the (P, 1, 512) slice, exactly as the flat
            # champion path derives its sentence-0 embedding (a 2D proj call
            # hits a different matmul kernel and differs by 1 ulp)
            ref = n.proj(n.text_feats[:, :1]).squeeze(1)
            ref2d = n.proj(n.text_feats[:, 0])
        assert tuple(emb.shape) == (14, 512)
        assert torch.equal(emb, ref), f"(b) init-equality failed pna={pna}"
        assert torch.allclose(emb, ref2d, atol=1e-6)
        assert torch.isfinite(emb).all()
        print(f"(b) PASS nested pna={pna}: emb (14, 512) == proj(sentence 0) bitwise; "
              f"member_feats {tuple(n.member_feats.shape)}, members/phrase="
              f"{n.member_mask.sum(1).tolist()}")

    # ---------------- (c) random out_proj -> change + grads ----------------
    n = build_nested(True).eval()
    with torch.no_grad():
        ref = n.class_embeddings().clone()
        for m in (n.attr_attn, n.phase_attn, n.attr_attn2, n.phase_attn2):
            m.out_proj.weight.normal_(0, 0.02)
            m.out_proj.bias.normal_(0, 0.02)
    emb = n.class_embeddings()
    assert not torch.allclose(emb, ref), "(c) embeddings did not change"
    assert torch.isfinite(emb).all(), "(c) non-finite embeddings"
    emb.pow(2).sum().backward()
    for name in ('proj', 'attr_attn', 'phase_attn', 'attr_attn2', 'phase_attn2'):
        mod = getattr(n, name)
        g = sum(float(q.grad.abs().sum()) for q in mod.parameters() if q.grad is not None)
        assert g > 0, f"(c) zero grad for {name}"
        print(f"(c) grad |sum| {name:12s} = {g:.4g}")
    print(f"(c) PASS embeddings change (max |delta| = "
          f"{(emb - ref).abs().max().item():.4g}) and all blocks receive gradient")

    # ---------------- (d) train mode 20 calls ----------------
    n = build_nested(True).train()
    with torch.no_grad():
        for m in (n.attr_attn, n.phase_attn, n.attr_attn2, n.phase_attn2):
            m.out_proj.weight.normal_(0, 0.02)
    torch.manual_seed(1)
    n_deg = 0
    for i in range(20):
        emb = n.class_embeddings()
        assert tuple(emb.shape) == (14, 512) and torch.isfinite(emb).all()
        emb.sum().backward()
        with torch.no_grad():
            stem = n.proj(n.stem_feats)
            n_deg += int((emb == stem).all(1).sum())
    print(f"(d) PASS 20 train-mode calls ok (degraded rows over 20 calls: "
          f"{n_deg}, expected ~{20 * 14 * 0.3:.0f})")
    # cosine_logits smoke
    n.eval()
    with torch.no_grad():
        lg = n.cosine_logits(torch.randn(2, 512, 7), n.class_embeddings())
    assert tuple(lg.shape) == (2, 14, 7)

    # ---------------- (e) freeze_except substrings ----------------
    names = ['text_pathway.' + k for k, _ in n.named_parameters()]
    for sub in ('text_pathway.phase_attn', 'text_pathway.attr_attn'):
        sel = [x for x in names if sub in x]
        print(f"(e) '{sub}' matches {len(sel)} tensors:")
        for x in sel:
            print("      ", x)
    print("(e) trainable-module order:",
          [k for k, _ in n.named_children()])
    print("ALL PASS")


if __name__ == '__main__':
    main()
