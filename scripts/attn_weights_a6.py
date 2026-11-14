#!/usr/bin/env python
"""What does the trained video ASA attend to? 

Runs a frozen stage-3 checkpoint on val windows (cached features, eval mode) and records the attention weights
of every MultiheadAttention in the video ASA:
  nested (FineDiving): chain_attn_phase (leaves), clip_attn_phase (action query over [action; leaves]),
                       chain_attn_act (action chunks), clip_attn_act (location query over [f_n; action chunks])
  flat (FineGym d3):   chain_attn (chunks), clip_attn (location query over [f_n; chunks])
Reports, for the location query: mean weight on its own feature f_n vs on the chunks, and how uniform the
weights over the chunks are (entropy / log K; 1 = uniform). If the chunk weights are near-uniform, the branch
reads the span mean, which for equal-length chunks is independent of where the cuts fall.

  python scripts/attn_weights_a6.py --dataset fd --config configs/finediving/v3/fd_champ_stage3_nested_s1_1gpu.yaml \
      --ckpt ckpt/fd_champ_stage3_nested_s1_1gpu_fdK3ref/vit_best_model --cache /data3/xiaodan8/FineDiving/cache_fd3_s1 --n 200
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['fd', 'fg'], required=True); ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--attr', default='configs/finediving/fd_dive_attribute_table_s1.json')
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    import torch
    if a.device == 'cpu':
        torch.set_num_threads(max(8, os.cpu_count() // 2))
    from libs.core import load_config
    from libs.datasets import make_dataset
    from libs.modeling import make_meta_arch
    cfg = load_config(a.config)
    attr = a.attr if a.dataset == 'fd' else cfg['model'].get('aux_attribute_table_path')
    held = [int(x[1:]) if isinstance(x, str) else int(x) for x in (json.load(open(attr)).get('held_out_classes', []) if attr else [])]
    cfg['dataset']['held_out_class_ids'] = held; cfg['model']['active_learning_method'] = cfg['active_learning_method']
    ds = make_dataset(cfg['dataset_name'], False, cfg['val_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset']); ds.set_feature_cache(a.cache)
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model.backbone.register_buffer('cached_pad_feat', torch.load(os.path.join(a.cache, 'pad_feat.pt'), map_location='cpu'), persistent=False)
    sd = torch.load(os.path.join(a.ckpt, 'mp_rank_00_model_states.pt'), map_location='cpu'); sd = sd.get('module', sd)
    model.load_state_dict(sd, strict=False); model = model.to(a.device).eval()
    vsa = model.video_subtree_attn
    # ---- wrap every MHA of the video ASA to also return weights
    stats = {}
    def wrap(name, mha):
        orig = mha.forward
        def fwd(q, k, v, **kw):
            kw = dict(kw); kw['need_weights'] = True; kw['average_attn_weights'] = True
            out, w = orig(q, k, v, **kw)                       # w: (N, Lq, Lk)
            w = w.detach().float()
            if w.shape[1] == 1:                                # query = one token (location / action query)
                p = w[:, 0, :]                                 # (N, Lk) ; key 0 = the query's own feature
                self_w = p[:, 0]; ch = p[:, 1:]
                chn = ch / ch.sum(-1, keepdim=True).clamp(min=1e-9)
                ent = -(chn * (chn + 1e-12).log()).sum(-1) / math.log(ch.shape[1]) if ch.shape[1] > 1 else torch.ones_like(self_w)
                s = stats.setdefault(name, {'self': [], 'chunks_max': [], 'unif': [], 'per_chunk': []})
                s['self'].append(self_w.mean().item()); s['chunks_max'].append(chn.max(-1).values.mean().item()); s['unif'].append(ent.mean().item()); s['per_chunk'].append(chn.mean(0).numpy())
            else:                                              # self-attention among chunks: how much do tokens move off the diagonal
                p = w                                          # (N, L, L)
                diag = torch.diagonal(p, dim1=1, dim2=2).mean().item()
                s = stats.setdefault(name, {'diag': [], 'unif': []})
                L = p.shape[-1]; ent = -(p * (p + 1e-12).log()).sum(-1).mean().item() / math.log(L) if L > 1 else 1.0
                s['diag'].append(diag); s['unif'].append(ent)
            return out, None
        mha.forward = fwd
    for name in ('chain_attn_phase', 'clip_attn_phase', 'chain_attn_act', 'clip_attn_act', 'chain_attn', 'clip_attn'):
        if hasattr(vsa, name) and getattr(vsa, name) is not None:
            wrap(name, getattr(vsa, name))
    n = 0
    with torch.no_grad():
        for i in range(min(a.n, len(ds))):
            model([ds[i]]); n += 1
    print(f'{a.dataset}: {n} val windows, every pyramid level and location pooled')
    K = getattr(vsa, 'K', None); Kp = getattr(vsa, 'Kp', None)
    for name, s in stats.items():
        if 'self' in s:
            pc = np.mean(np.stack(s['per_chunk']), 0)
            print(f'  {name}: weight on own feature {np.mean(s["self"]):.3f} | over the chunks: max-weight {np.mean(s["chunks_max"]):.3f} '
                  f'(uniform would be {1/len(pc):.3f}), uniformity {np.mean(s["unif"]):.3f} (1 = uniform) | per-chunk share {np.round(pc, 3).tolist()}')
        else:
            print(f'  {name} (neighbour self-attention): weight kept on the token itself {np.mean(s["diag"]):.3f}, uniformity over the band {np.mean(s["unif"]):.3f}')
    json.dump({k: {kk: (float(np.mean(vv)) if kk != 'per_chunk' else np.mean(np.stack(vv), 0).tolist()) for kk, vv in v.items()} for k, v in stats.items()},
              open(f'data/order_probe/attn_{a.dataset}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
