#!/usr/bin/env python
"""Why is mAP flat in K?  Re-run the trained video-tree block (VideoSubtreeAttn) of a THUMOS champion checkpoint
at several K on the SAME inputs (eval-only, CPU) and measure (per FPN level, valid positions only):
  res/x     : |v_K10 - x| / |x|            how much the video tree changes the feature at all (K=10, as trained)
  cos(v,x)  : cosine(v_K10, x)
  cos(vK,v10), |vK - v10| / |v10 - x| : how much of the tree's own contribution changes when K is changed
  cos(rK,r10): direction of the residual at K vs at K=10
Question: "79% of chunks are shorter than the I3D receptive field, so why is K=2 the same as K=10?"
  PYTHONUTF8=1 CUDA_VISIBLE_DEVICES= python scripts/k_invariance_probe.py --n 0 --videos 40 --Ks 1 2 3 5 10 20
"""
import argparse, glob, os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/data3/xiaodan8/tifad')
CH = ("TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0 TIFAD_SEM_DEGRADE=0 "
      "TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain TIFAD_VSA=1 "
      "TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3")
for kv in CH.split():
    k, v = kv.split('=', 1); os.environ[k] = v
import torch, torch.nn.functional as F, numpy as np
from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


class StopForward(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--ckpt', default='ckpt/thumos14_i3d_vsa3_{n}/mAP_best.pth.tar')
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--videos', type=int, default=40)
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 2, 3, 5, 10, 20])
    ap.add_argument('--out', default='data/k_invariance/thumos_vsa3_s{n}.json')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config(a.config)
    fix_random_seed(cfg['init_rand_seed'], include_cuda=False)
    ds = make_dataset(cfg['dataset_name'], a.n, False, cfg['val_split'], **cfg['dataset'])
    loader = make_data_loader(ds, False, None, 1, cfg["loader"]["num_workers"])
    model = make_meta_arch(a.n, cfg['model_name'], cfg['dataset']['subset_file'], cfg['dataset']['data_split'], **cfg['model'])
    ck = torch.load(a.ckpt.format(n=a.n), map_location='cpu')
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=True); model.eval(); del ck
    vsa = model.vsa
    K0 = vsa.K
    assert vsa.posemb is None and not vsa.xm and not vsa.softphase and not vsa.learn_bounds, 'probe assumes the plain champion block'
    orig = vsa.forward.__func__ if hasattr(vsa.forward, '__func__') else type(vsa).forward
    cap = {}

    def hooked(feats, offsets, fpn_masks, **kw):
        cap['args'] = ([f.detach() for f in feats], [o.detach() for o in offsets], [m.detach() for m in fpn_masks], kw)
        raise StopForward
    vsa.forward = hooked

    def set_K(K):
        vsa.K = K; i = torch.arange(K); vsa.band_mask = (i[:, None] - i[None, :]).abs() > 1

    def run(K):
        set_K(K); out, _, _ = orig(vsa, *cap['args'][:3], **cap['args'][3]); return out
    acc = {K: {'cos_v': [], 'rel_dv': [], 'cos_r': []} for K in a.Ks}
    base = {'res_over_x': [], 'cos_vx': [], 'res_over_x_lvl': {}, 'n_pos': 0}
    nv = 0
    with torch.no_grad():
        for vi, vl in enumerate(loader):
            if nv >= a.videos:
                break
            try:
                model(vl)
            except StopForward:
                pass
            if 'args' not in cap:
                continue
            feats, offs, masks, kw = cap['args']
            v10 = run(K0)
            outs = {K: run(K) for K in a.Ks}
            for l, (x, m) in enumerate(zip(feats, masks)):
                sel = m[:, 0, :]                                        # (B, T)
                xt = x.transpose(1, 2)[sel]; vt = v10[l].transpose(1, 2)[sel]       # (N, D)
                if xt.numel() == 0:
                    continue
                r10 = vt - xt
                rr = (r10.norm(dim=-1) / xt.norm(dim=-1).clamp_min(1e-6))
                base['res_over_x'] += rr.tolist(); base['cos_vx'] += F.cosine_similarity(vt, xt, dim=-1).tolist()
                base['res_over_x_lvl'].setdefault(l, []).append(float(rr.mean()))
                base['n_pos'] += int(xt.size(0))
                for K in a.Ks:
                    vk = outs[K][l].transpose(1, 2)[sel]; rk = vk - xt
                    acc[K]['cos_v'] += F.cosine_similarity(vk, vt, dim=-1).tolist()
                    acc[K]['rel_dv'] += ((vk - vt).norm(dim=-1) / r10.norm(dim=-1).clamp_min(1e-6)).tolist()
                    acc[K]['cos_r'] += F.cosine_similarity(rk, r10, dim=-1).tolist()
            cap.clear(); nv += 1
            if nv % 10 == 0:
                print(f'[probe] {nv} videos, {base["n_pos"]} positions', flush=True)
    set_K(K0)
    med = lambda z: float(np.median(z)); mean = lambda z: float(np.mean(z))
    res = {'split': a.n, 'videos': nv, 'positions': base['n_pos'], 'K_trained': K0,
           'tree_contribution': {'res_over_x_mean': mean(base['res_over_x']), 'res_over_x_median': med(base['res_over_x']),
                                 'cos_v_x_mean': mean(base['cos_vx']),
                                 'res_over_x_per_level': {str(l): mean(v) for l, v in base['res_over_x_lvl'].items()}},
           'vs_K': {str(K): {'cos_v_mean': mean(acc[K]['cos_v']), 'cos_v_min': float(np.min(acc[K]['cos_v'])),
                             'rel_change_of_tree_contribution_mean': mean(acc[K]['rel_dv']), 'rel_change_median': med(acc[K]['rel_dv']),
                             'cos_residual_mean': mean(acc[K]['cos_r'])} for K in a.Ks}}
    os.makedirs(os.path.dirname(a.out.format(n=a.n)), exist_ok=True)
    json.dump(res, open(a.out.format(n=a.n), 'w'), indent=1)
    print('RESULT', json.dumps(res), flush=True)
    t = res['tree_contribution']
    print(f"\nvideo tree at K={K0}: |v-x|/|x| mean {t['res_over_x_mean']:.3f} (median {t['res_over_x_median']:.3f}), cos(v,x) {t['cos_v_x_mean']:.4f}; per level " +
          ', '.join(f"L{l} {r:.3f}" for l, r in t['res_over_x_per_level'].items()))
    print(f"{'K':>3} {'cos(vK,v10)':>12} {'min':>7} {'|vK-v10|/|v10-x|':>18} {'cos(rK,r10)':>12}")
    for K in a.Ks:
        z = res['vs_K'][str(K)]
        print(f"{K:>3} {z['cos_v_mean']:>12.5f} {z['cos_v_min']:>7.4f} {z['rel_change_of_tree_contribution_mean']:>18.3f} {z['cos_residual_mean']:>12.4f}")


if __name__ == '__main__':
    main()
