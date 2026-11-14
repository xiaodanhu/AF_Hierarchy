"""Tests for the ORDERED-COMPOSITION controls and the post-VSA probe stash:
  text  : libs/modeling/text.py   TIFAD_SHUFFLE_CHILDREN=eval|train|both
                                  (TIFAD_SHUFFLE_SEED), TIFAD_CHAIN_FULL=1
  video : libs/modeling/meta_archs.py  TIFAD_PROBE_STASH_POST=1|0
          (scripts/repvalid_variant.py consumes the stash unchanged)

Run from the repo root (/data3/xiaodan8/tifad):

  env CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONUTF8=1 LC_ALL=C.UTF-8 \
      python scripts/tests/test_order_controls.py --before_dir <dir> \
      [--ckpt ckpt/thumos14_i3d_vsa3_0/mAP_best.pth.tar] [--skip_text] [--skip_stash]

  --before_dir : pre-edit baselines written BEFORE the controls were added:
                 text_before.pt  (TextFeatures eval outputs for modes test /
                 train, training=True output, state_dict, and the same with
                 RANDOM chain/attr out_projs: rand_test / rand_train /
                 rand_state) and stash_before.pt (TIFAD_PROBE_STASH=1 stash
                 of the vsa3 split-0 checkpoint on the first THUMOS test
                 clip, eval mode). The bitwise-unchanged checks are skipped
                 when a file is absent.

(a) all three envs unset: text forward (zero AND random out_projs) and the
    probe stash bitwise identical to the pre-edit baselines
(b) TIFAD_SHUFFLE_CHILDREN=eval: unchanged with zero-init chain/attr
    out_projs (the root sentence alone determines the output); changes the
    eval-mode class embeddings with random out_projs; deterministic across
    two forwards and two constructions (same seed); a different
    TIFAD_SHUFFLE_SEED gives a different permutation; mode=='train' is
    untouched. 'train': fresh permutation per forward in mode=='train'
    (outputs differ across forwards), eval untouched. 'both': both. The
    shuffle composes with TIFAD_SUB_ENSEMBLE (same permutation applied to
    the refined phases, i.e. sub-children move with their parent phase).
(c) TIFAD_CHAIN_FULL=1: unchanged with zero-init out_projs; differs from
    the banded result for K=10 children with a random chain out_proj; for
    2 tokens the band covers everything so masked == full (allclose), for
    3 it already differs (the (0,2) pair is masked).
(d) TIFAD_PROBE_STASH_POST on the vsa3 checkpoint (TIFAD_VSA=1, GPU):
    unset == '1' (the legacy stash already read cls_feat, which IS the
    post-VSA pyramid when VSA is on) == pre-edit baseline bitwise; '0'
    (pre-VSA trunk `_vsa_in`) differs from '1' with the trained VSA
    weights and is bitwise equal to it once the two VSA out_projs are
    zeroed; masks / points identical in every case.

The text tests run on CPU; (d) uses the visible CUDA device (< 6 GB).
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

SUBSET = 'splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list'
CHAMP_ENVS = {
    'TIFAD_PROMPT_ENSEMBLE': 'configs/thumos14_qwen_ensemble.json',
    'TIFAD_ATTR_ATTN': '1',
    'TIFAD_ATTNDESIGN': 'phasechain',
    'TIFAD_PROMPT_DROPOUT': '0',
    'TIFAD_SEM_DEGRADE': '0',
}
# the vsa3 stage-3 recipe at eval (run_vsa_final.sh T_EV + TIFAD_VSA=1), plus
# the repvalid_variant.py probe envs
VSA3_ENVS = {
    'TIFAD_PROMPT_DROPOUT': '0', 'TIFAD_SEM_DEGRADE': '0',
    'TIFAD_PROBE_STASH': '1',
    'TIFAD_PROMPT_ENSEMBLE': 'configs/thumos14_qwen_ensemble.json',
    'TIFAD_ATTR_ATTN': '1', 'TIFAD_ATTNDESIGN': 'phasechain',
    'TIFAD_FAMCE': '0.3', 'TIFAD_FAMCE_MAP': 'configs/thumos14_fam_map.json',
    'TIFAD_FAMCE_UW': '1', 'TIFAD_VSA': '1',
}


def clear_envs():
    for k in list(os.environ):
        if k.startswith('TIFAD_') or k.startswith('FINEGYM_'):
            del os.environ[k]


def set_envs(extra=None, base=CHAMP_ENVS):
    clear_envs()
    os.environ.update(base)
    if extra:
        os.environ.update(extra)


def build_text(seed=0):
    from libs.modeling.text import TextFeatures
    torch.manual_seed(seed)
    return TextFeatures(model_path='openai/clip-vit-base-patch16', subset_file=SUBSET,
                        data_split=75, emb_dim=256, split_num=0)


def fwd(tf, mode, train_flag=False):
    tf.train(train_flag)
    with torch.no_grad():
        return tf(2, mode)[0].clone()


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


# ----------------------------------------------------------------------------
def test_text(args):
    bp = os.path.join(args.before_dir, 'text_before.pt')
    ref = torch.load(bp) if os.path.exists(bp) else None

    # ---- (a) envs unset: byte-inert ----
    set_envs()
    tf0 = build_text(0)
    assert not tf0.shuffle_children and tf0._shuffle_perm is None and not tf0.chain_full
    out0 = {'test': fwd(tf0, 'test'), 'train': fwd(tf0, 'train'),
            'train_flag': fwd(tf0, 'train', True)}
    if ref is not None:
        same_seed = all(eq(out0[m], ref[m]) for m in out0)
        tf0.load_state_dict(ref['state'])
        assert all(eq(fwd(tf0, m, f), ref[k]) for m, f, k in
                   (('test', False, 'test'), ('train', False, 'train'),
                    ('train', True, 'train_flag'))), \
            "(a) text forward CHANGED vs the pre-edit baseline (zero out_projs)"
        tf0.load_state_dict(ref['rand_state'])
        assert eq(fwd(tf0, 'test'), ref['rand_test']) and \
            eq(fwd(tf0, 'train'), ref['rand_train']), \
            "(a) text forward CHANGED vs the pre-edit baseline (random out_projs)"
        # training=True reference with random out_projs (nn.MultiheadAttention
        # takes its fast path only in eval, so eval / training outputs are not
        # bitwise equal once out_proj != 0; dropout is 0 so it is deterministic)
        ref['rand_train_flag'] = fwd(tf0, 'train', True)
        print(f"(a) text forward with the new envs unset bitwise unchanged vs the "
              f"pre-edit baseline (zero AND random chain/attr out_projs; same-seed "
              f"construction equal: {same_seed}): PASS")
    else:
        # self-consistent fallback: build the reference in-process
        ref = {'test': out0['test'], 'train': out0['train'],
               'train_flag': out0['train_flag'], 'state': tf0.state_dict()}
        torch.manual_seed(3)
        for a in (tf0.attr_attn, tf0.chain_attn):
            torch.nn.init.normal_(a.out_proj.weight, std=0.05)
            torch.nn.init.normal_(a.out_proj.bias, std=0.05)
        ref['rand_test'] = fwd(tf0, 'test')
        ref['rand_train'] = fwd(tf0, 'train')
        ref['rand_train_flag'] = fwd(tf0, 'train', True)
        ref['rand_state'] = tf0.state_dict()
        print("(a) SKIPPED bitwise-vs-baseline check (no text_before.pt); "
              "using in-process references")
    zero_state, rand_state = ref['state'], ref['rand_state']
    K = len(tf0.ensemble_prompts[tf0.train_classes[0]]) - 1
    del tf0

    # ---- (b) SHUFFLE_CHILDREN=eval ----
    set_envs({'TIFAD_SHUFFLE_CHILDREN': 'eval'})
    tf1 = build_text(0)
    assert tf1.shuffle_children == 'eval' and tf1._shuffle_perm is not None
    n_moved = sum(int((p != torch.arange(K)).sum()) for p in tf1._shuffle_perm.values())
    assert n_moved > 0
    assert sorted(tf1._shuffle_perm) == sorted(tf1.ensemble_prompts)
    tf1.load_state_dict(zero_state)
    assert eq(fwd(tf1, 'test'), ref['test']) and eq(fwd(tf1, 'train'), ref['train']), \
        "(b) zero-init out_projs: the shuffle must NOT change the output"
    tf1.load_state_dict(rand_state)
    o1 = fwd(tf1, 'test')
    assert not eq(o1, ref['rand_test']), "(b) eval shuffle should change the output"
    assert eq(o1, fwd(tf1, 'test')), "(b) eval shuffle not deterministic across forwards"
    # mode=='train' untouched by 'eval'
    assert eq(fwd(tf1, 'train'), ref['rand_train']) and \
        eq(fwd(tf1, 'train', True), ref['rand_train_flag']), \
        "(b) 'eval' must leave mode=='train' untouched"
    tf1b = build_text(0)
    tf1b.load_state_dict(rand_state)
    assert all(eq(tf1b._shuffle_perm[c], tf1._shuffle_perm[c]) for c in tf1._shuffle_perm)
    assert eq(fwd(tf1b, 'test'), o1), "(b) eval shuffle not deterministic across constructions"
    del tf1b
    set_envs({'TIFAD_SHUFFLE_CHILDREN': 'eval', 'TIFAD_SHUFFLE_SEED': '1'})
    tf1c = build_text(0)
    tf1c.load_state_dict(rand_state)
    assert any(not eq(tf1c._shuffle_perm[c], tf1._shuffle_perm[c]) for c in tf1._shuffle_perm)
    o1c = fwd(tf1c, 'test')
    assert not eq(o1c, o1) and not eq(o1c, ref['rand_test'])
    del tf1c
    d_eval = (o1 - ref['rand_test']).abs().max().item()
    print(f"(b) TIFAD_SHUFFLE_CHILDREN=eval (K={K} children, seed 0: {n_moved} "
          f"displaced children over {len(tf1._shuffle_perm)} classes): zero-init "
          f"out_projs -> unchanged; random out_projs -> eval output changes "
          f"(max|d|={d_eval:.3g}), deterministic across forwards + constructions, "
          f"seed 1 differs, mode=='train' untouched: PASS")

    # ---- (b) SHUFFLE_CHILDREN=train ----
    set_envs({'TIFAD_SHUFFLE_CHILDREN': 'train'})
    tf2 = build_text(0)
    assert tf2._shuffle_perm is None
    tf2.load_state_dict(rand_state)
    assert eq(fwd(tf2, 'test'), ref['rand_test']), "(b) 'train' must leave eval untouched"
    torch.manual_seed(10)
    a = fwd(tf2, 'train', True)
    torch.manual_seed(11)
    b = fwd(tf2, 'train', True)
    assert not eq(a, b) and not eq(a, ref['rand_train_flag']), \
        "(b) 'train' should give a fresh permutation per forward"
    torch.manual_seed(10)
    assert eq(fwd(tf2, 'train', True), a), "(b) 'train' should follow the torch RNG"
    tf2.load_state_dict(zero_state)
    assert eq(fwd(tf2, 'train', True), ref['train_flag'])
    del tf2
    # ---- (b) 'both' ----
    set_envs({'TIFAD_SHUFFLE_CHILDREN': 'both'})
    tf3 = build_text(0)
    tf3.load_state_dict(rand_state)
    assert eq(fwd(tf3, 'test'), o1), "(b) 'both' eval must equal the fixed 'eval' permutation"
    torch.manual_seed(10)
    a3 = fwd(tf3, 'train', True)
    torch.manual_seed(11)
    assert not eq(a3, fwd(tf3, 'train', True))
    del tf3
    print("(b) 'train': eval untouched, mode=='train' output differs across forwards "
          "(RNG-driven), zero-init unchanged; 'both': eval == fixed permutation, "
          "train random: PASS")

    # ---- (b) composes with TIFAD_SUB_ENSEMBLE ----
    sub = 'configs/thumos14_subphase_ensemble.json'
    if os.path.exists(sub):
        set_envs({'TIFAD_SHUFFLE_CHILDREN': 'eval', 'TIFAD_SUB_ENSEMBLE': sub})
        tfs = build_text(0)
        missing, unexpected = tfs.load_state_dict(rand_state, strict=False)
        assert not unexpected and all(m.startswith(('attr_attn_sub.', 'chain_attn_sub.'))
                                      for m in missing)
        assert eq(fwd(tfs, 'test'), o1), \
            "(b) shuffle + sub-ensemble (zero sub out_projs) must equal the shuffle alone"
        torch.manual_seed(5)
        for m in (tfs.attr_attn_sub, tfs.chain_attn_sub):
            torch.nn.init.normal_(m.out_proj.weight, std=0.05)
            torch.nn.init.normal_(m.out_proj.bias, std=0.05)
        os_ = fwd(tfs, 'test')
        assert not eq(os_, o1) and eq(os_, fwd(tfs, 'test'))
        del tfs
        print("(b) shuffle composes with TIFAD_SUB_ENSEMBLE (permutation applied to the "
              "sub-refined phases; zero sub out_projs == shuffle alone; random sub "
              "out_projs change it, deterministic): PASS")

    # ---- (c) CHAIN_FULL ----
    set_envs({'TIFAD_CHAIN_FULL': '1'})
    tf4 = build_text(0)
    assert tf4.chain_full
    tf4.load_state_dict(zero_state)
    assert eq(fwd(tf4, 'test'), ref['test']) and eq(fwd(tf4, 'train'), ref['train']), \
        "(c) zero-init out_projs: full attention must NOT change the output"
    tf4.load_state_dict(rand_state)
    o4 = fwd(tf4, 'test')
    o4t = fwd(tf4, 'train', True)
    assert not eq(o4, ref['rand_test']) and not eq(o4t, ref['rand_train_flag']), \
        "(c) full attention should differ from the banded result (K=10, both modes)"
    d_full = (o4 - ref['rand_test']).abs().max().item()
    # band == full for 2 tokens (|i-j| <= 1 covers everything; with 3 tokens
    # the pair (0, 2) is already masked, so 2 is the exact boundary)
    tf4.eval()
    x = torch.randn(4, 2, 256)
    with torch.no_grad():
        ub, _ = tf4.chain_attn(x, x, x, attn_mask=tf4._band_mask(2, x.device),
                               need_weights=False)
        uf, _ = tf4.chain_attn(x, x, x, need_weights=False)
    assert torch.allclose(ub, uf, atol=1e-6), "(c) band mask over 2 tokens must equal full"
    x3 = torch.randn(4, 3, 256)
    with torch.no_grad():
        ub3, _ = tf4.chain_attn(x3, x3, x3, attn_mask=tf4._band_mask(3, x3.device),
                                need_weights=False)
        uf3, _ = tf4.chain_attn(x3, x3, x3, need_weights=False)
    assert not torch.allclose(ub3, uf3, atol=1e-6), "(c) 3 tokens: band must mask (0,2)"
    # shuffle + full attention compose
    set_envs({'TIFAD_CHAIN_FULL': '1', 'TIFAD_SHUFFLE_CHILDREN': 'eval'})
    tf5 = build_text(0)
    tf5.load_state_dict(rand_state)
    o5 = fwd(tf5, 'test')
    assert not eq(o5, o4) and not eq(o5, o1)
    del tf4, tf5
    print(f"(c) TIFAD_CHAIN_FULL=1: zero-init unchanged; random chain out_proj -> "
          f"differs from the banded result for K={K} (max|d|={d_full:.3g}, eval and "
          f"training=True); 2-token band == full (allclose), 3-token band != full; "
          f"composes with the "
          f"shuffle: PASS")
    clear_envs()


# ----------------------------------------------------------------------------
def test_stash(args):
    import phase_probe as pp                    # sets REF_ENVS; overridden below
    from libs.core import load_config
    from libs.datasets import make_dataset, make_data_loader
    from libs.utils import fix_random_seed
    set_envs(base=VSA3_ENVS)
    cfg = load_config('configs/thumos14_i3d.yaml')
    fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    model, notes = pp.build_model(cfg, 0, args.ckpt)
    assert model.vsa is not None
    ds = make_dataset(cfg['dataset_name'], 0, False, cfg['val_split'], **cfg['dataset'])
    loader = make_data_loader(ds, False, None, 1, 1)
    vl = next(iter(loader))
    model.eval()
    model.test_pre_nms_thresh = 2.0
    model.test_nms_method = 'none'
    torch.cuda.reset_peak_memory_stats()

    def run(post):
        os.environ.pop('TIFAD_PROBE_STASH_POST', None)
        if post is not None:
            os.environ['TIFAD_PROBE_STASH_POST'] = post
        with torch.no_grad():
            model(vl)
        st = model._probe_stash
        return ([f.clone() for f in st['feats']], [m.clone() for m in st['masks']],
                [p.clone() for p in st['points']])

    def same(a, b):
        return all(len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y))
                   for x, y in zip(a, b))

    legacy = run(None)
    assert eq(legacy[0][0], run(None)[0][0]), "(d) eval forward not deterministic"
    sp = os.path.join(args.before_dir, 'stash_before.pt')
    if os.path.exists(sp):
        ref = torch.load(sp)
        assert ref['video_id'] == vl[0]['video_id'], (ref['video_id'], vl[0]['video_id'])
        dev = legacy[0][0].device
        refs = tuple([t.to(dev) for t in ref[k]] for k in ('feats', 'masks', 'points'))
        assert same(legacy, refs), "(d) legacy stash CHANGED vs the pre-edit baseline"
        print(f"(d) TIFAD_PROBE_STASH_POST unset: stash bitwise unchanged vs the pre-edit "
              f"baseline on clip {vl[0]['video_id']}: PASS")
    else:
        print("(d) SKIPPED bitwise-vs-baseline check (no stash_before.pt)")
    post = run('1')
    pre = run('0')
    assert same(post, legacy), "(d) POST=1 must equal the legacy stash (cls_feat)"
    assert same(post[1:], pre[1:]), "(d) masks / points must not depend on the option"
    diffs = [(a - b).abs().max().item() for a, b in zip(post[0], pre[0])]
    assert all(d > 0 for d in diffs), f"(d) trained VSA: post must differ from pre {diffs}"
    for m in (model.vsa.chain_attn, model.vsa.clip_attn):
        torch.nn.init.zeros_(m.out_proj.weight)
        torch.nn.init.zeros_(m.out_proj.bias)
    post0 = run('1')
    pre0 = run('0')
    assert same(post0, pre0), "(d) zero VSA out_projs: post must equal pre bitwise"
    assert same(pre0, pre), "(d) the pre-VSA trunk must not depend on the VSA weights"
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"(d) TIFAD_PROBE_STASH_POST on {args.ckpt} (TIFAD_VSA=1, clip "
          f"{vl[0]['video_id']}, {len(post[0])} levels): '1' == legacy; '0' (pre-VSA "
          f"trunk) differs with the trained weights (max|post-pre| per level "
          f"{[f'{d:.3g}' for d in diffs]}) and equals '1' bitwise with zeroed VSA "
          f"out_projs; masks/points identical; peak GPU {peak:.2f} GB: PASS")
    os.environ.pop('TIFAD_PROBE_STASH_POST', None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--before_dir', default='')
    ap.add_argument('--ckpt', default='ckpt/thumos14_i3d_vsa3_0/mAP_best.pth.tar')
    ap.add_argument('--skip_text', action='store_true')
    ap.add_argument('--skip_stash', action='store_true')
    args = ap.parse_args()
    if not args.skip_text:
        test_text(args)
    if not args.skip_stash:
        if torch.cuda.is_available():
            test_stash(args)
        else:
            print("(d) SKIPPED: no CUDA device visible")
    print("ALL TESTS PASSED")
