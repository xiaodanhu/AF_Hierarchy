#!/usr/bin/env python
"""TEMPORAL-ORDER PROBE on frozen actionformer6 checkpoints (FineDiving / FineGym), the
counterpart of tifad's scripts/repvalid_variant.py (Sec. 4.3 "Representation Validation").

Question does the per-location cls-trunk feature -- the one the class
cosine reads -- encode WHERE in the action a location sits, and does the video ASA change that?
Nothing is trained except tiny probes on frozen features.

Per checkpoint:
  1. build the model from the config, load the DeepSpeed module weights (mp_rank_00_model_states.pt,
     exactly as train_shard --init_from), cached encoder features (--cache), eval mode,
     A6_PROBE_STASH=1 so meta_archs keeps the level-0 cls-trunk features BEFORE and AFTER the
     video ASA (post == pre when the checkpoint has no video branch).
  2. rows: every level-0 location inside a ground-truth segment (shortest containing segment on
     ties): feature, t_norm = (t - s)/(e - s), segment id, seen/unseen class. FineDiving adds the
     ANNOTATED sub-action position k/M and name at that time (from the label files).
  3. Kendall's tau (rep_metrics protocol): linear regressor feature -> t_norm on TRAIN windows
     (Adam 3e-3, 100 epochs, bs 4096, standardized features, 3 seeds), per-segment tau on VAL
     segments with >= --min_loc locations, mean over seen / unseen segments; shuffled-feature control.
  4. FineDiving only: sub-action POSITION probe, Linear(D,D)-ReLU-Linear(D,5), Adam 1e-3, 30 epochs,
     accuracy on val seen / unseen rows; shuffled-label control; same for the sub-action NAME (28-way).
Output: data/order_probe/<tag>.json and a printed table.

  python scripts/order_probe_a6.py --dataset fd --tag fdK3ref \
      --config configs/finediving/v3/fd_champ_stage3_nested_s1_1gpu.yaml \
      --ckpt ckpt/fd_champ_stage3_nested_s1_1gpu_fdK3ref/vit_best_model --cache /data3/xiaodan8/FineDiving/cache_fd3_s1 --device cpu
"""
import argparse, collections, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FD = '/data3/xiaodan8/FineDiving/OpenDataLab___FineDiving'
sec = lambda s: float(re.sub(r'[^0-9.]', '', s))


def load_fd_subs():
    """clip id -> [(name, abs start, abs end)] from both label files (raw ids -> verbalizer names)."""
    sub = json.load(open('configs/finediving/fd_sub_verbalizer.json'))['actions']
    raw2name = {v['raw_id']: v['name'] for v in sub.values()}
    out = {}
    for sp in ('train', 'test'):
        for line in open(f'{FD}/finediving_{sp}_label.txt'):
            if line.strip():
                r = json.loads(line)
                for x in r['raw_value']:
                    out[r['video']] = [(raw2name.get(q['action_id'], q['action_id']), sec(q['span'][0]), sec(q['span'][1])) for q in x['actions']]
    return out, sorted(set(raw2name.values()))


def harvest(model, ds, device, held, fd_subs, max_windows=0, bs=1):   # inference path: one window per call
    import torch
    rows = collections.defaultdict(list)
    idx = list(range(len(ds)))[:max_windows or None]
    t0 = time.time()
    for i0 in range(0, len(idx), bs):
        items = [ds[i] for i in idx[i0:i0 + bs]]
        with torch.no_grad():
            model(items)
        pre, post, mask = model._probe_pre[0], model._probe_post, model._probe_mask[0]
        post = post[0] if post is not None else pre
        for b, it in enumerate(items):
            segs = it['segments'].numpy() if hasattr(it['segments'], 'numpy') else np.asarray(it['segments'])
            labs = it['labels'].numpy() if hasattr(it['labels'], 'numpy') else np.asarray(it['labels'])
            if segs is None or len(segs) == 0:
                continue
            T = int(mask[b, 0].sum().item()); step = float(it['feat_stride']) / float(it['fps'])
            clip = it.get('video_name', it['video_id']); w0 = float(it.get('window_start_time', 0.0))
            subs = fd_subs.get(clip) if fd_subs else None
            for k in range(T):
                t = (k + 0.5) * step
                inside = [(e - s, j) for j, (s, e) in enumerate(segs) if s <= t < e]
                if not inside:
                    continue
                _, j = min(inside); s, e = segs[j]
                pos, M, name = -1, 0, ''
                if subs:
                    M = len(subs)
                    for q, (nm, a, bnd) in enumerate(subs):
                        if a <= w0 + t < bnd:
                            pos, name = q, nm; break
                rows['pre'].append(pre[b, :, k].float().cpu().numpy()); rows['post'].append(post[b, :, k].float().cpu().numpy())
                rows['tnorm'].append((t - s) / max(1e-6, e - s)); rows['seg'].append(f'{it["video_id"]}#{j}')
                rows['unseen'].append(int(labs[j]) in held); rows['pos'].append(pos); rows['M'].append(M); rows['name'].append(name)
        if (i0 // bs) % 200 == 0:
            print(f'  harvested {i0 + len(items)}/{len(idx)} windows, {len(rows["tnorm"])} rows, {(time.time()-t0)/60:.1f} min', flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out['pre'] = np.stack(rows['pre']).astype(np.float32); out['post'] = np.stack(rows['post']).astype(np.float32)
    return out


def fit_linear(X, y, seed, epochs=100, lr=3e-3, bs=4096):
    import torch
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    w = torch.nn.Linear(X.shape[1], 1); opt = torch.optim.Adam(w.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; opt.zero_grad(); loss = ((w(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
    return lambda Z: w(torch.from_numpy(Z)).squeeze(-1).detach().numpy()


def fit_mlp(X, y, ncls, seed, epochs=30, lr=1e-3, bs=4096):
    import torch
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)); D = X.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.ReLU(), torch.nn.Linear(D, ncls)); opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; opt.zero_grad(); loss = torch.nn.functional.cross_entropy(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    return lambda Z: net(torch.from_numpy(Z)).argmax(-1).detach().numpy()


def kendall_by_group(pred, tr, val, min_loc):
    from scipy.stats import kendalltau
    res = {}
    for grp, flag in (('seen', False), ('unseen', True)):
        taus = []
        for seg in np.unique(val['seg'][val['unseen'] == flag]):
            m = val['seg'] == seg
            if m.sum() < min_loc:
                continue
            t, _ = kendalltau(pred[m], val['tnorm'][m]); taus.append(0.0 if np.isnan(t) else t)
        res[grp] = {'kendall_tau': float(np.mean(taus)) if taus else None, 'n_segments': len(taus)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['fd', 'fg'], required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--attr', default='configs/finediving/fd_dive_attribute_table_s1.json')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--min_loc', type=int, default=5); ap.add_argument('--max_windows', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    import torch
    if a.device == 'cpu':
        torch.set_num_threads(max(8, os.cpu_count() // 2))
    os.environ['A6_PROBE_STASH'] = '1'
    from libs.core import load_config
    from libs.datasets import make_dataset
    from libs.modeling import make_meta_arch
    cfg = load_config(a.config)
    attr = a.attr if a.dataset == 'fd' else cfg['model'].get('aux_attribute_table_path')
    held = set(int(x[1:]) if isinstance(x, str) else int(x) for x in (json.load(open(attr)).get('held_out_classes', []) if attr else []))
    cfg['dataset']['held_out_class_ids'] = sorted(held)
    cfg['model']['active_learning_method'] = cfg['active_learning_method']
    dsets = {}
    for split, is_tr in (('train', True), ('val', False)):
        d = make_dataset(cfg['dataset_name'], is_tr, cfg[f'{split}_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
        d.set_feature_cache(a.cache); dsets[split] = d
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model.backbone.register_buffer('cached_pad_feat', torch.load(os.path.join(a.cache, 'pad_feat.pt'), map_location='cpu'), persistent=False)
    src = a.ckpt if not os.path.isdir(a.ckpt) else os.path.join(a.ckpt, 'mp_rank_00_model_states.pt')
    sd = torch.load(src, map_location='cpu'); sd = sd.get('module', sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] {src}: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}', flush=True)
    model = model.to(a.device).eval()
    fd_subs, sub_names = load_fd_subs() if a.dataset == 'fd' else ({}, [])
    data = {}
    for split in ('train', 'val'):
        print(f'[{split}] {len(dsets[split])} windows', flush=True)
        data[split] = harvest(model, dsets[split], a.device, held, fd_subs, a.max_windows)
        print(f'[{split}] rows {len(data[split]["tnorm"])}, unseen rows {int(data[split]["unseen"].sum())}', flush=True)
    res = {'meta': vars(a), 'rows': {s: int(len(data[s]['tnorm'])) for s in data}}
    tr, va = data['train'], data['val']
    for feat in ('pre', 'post'):
        mu, sd_ = tr[feat].mean(0), tr[feat].std(0) + 1e-6
        Xtr, Xva = (tr[feat] - mu) / sd_, (va[feat] - mu) / sd_
        taus = {'feature': [], 'shuffled': []}
        for seed in a.seeds:
            f = fit_linear(Xtr, tr['tnorm'], seed); taus['feature'].append(kendall_by_group(f(Xva), tr, va, a.min_loc))
            perm = np.random.RandomState(seed).permutation(len(Xtr)); f2 = fit_linear(Xtr[perm], tr['tnorm'], seed)
            taus['shuffled'].append(kendall_by_group(f2(Xva), tr, va, a.min_loc))
        res[f'tau_{feat}'] = {v: {g: {'kendall_tau': float(np.mean([r[g]['kendall_tau'] for r in taus[v] if r[g]['kendall_tau'] is not None] or [np.nan])),
                                    'n_segments': taus[v][0][g]['n_segments']} for g in ('seen', 'unseen')} for v in taus}
        print(f'TAU [{feat}] feature: seen {res[f"tau_{feat}"]["feature"]["seen"]["kendall_tau"]:.3f} (n={res[f"tau_{feat}"]["feature"]["seen"]["n_segments"]}) '
              f'unseen {res[f"tau_{feat}"]["feature"]["unseen"]["kendall_tau"]:.3f} (n={res[f"tau_{feat}"]["feature"]["unseen"]["n_segments"]}) | '
              f'shuffled: seen {res[f"tau_{feat}"]["shuffled"]["seen"]["kendall_tau"]:.3f} unseen {res[f"tau_{feat}"]["shuffled"]["unseen"]["kendall_tau"]:.3f}', flush=True)
        if a.dataset == 'fd':
            for target, ncls, key in (('pos', int(max(tr['M'].max(), va['M'].max())), 'position'), ('name', len(sub_names), 'subaction')):
                ytr = tr['pos'] if target == 'pos' else np.array([sub_names.index(n) if n in sub_names else -1 for n in tr['name']])
                yva = va['pos'] if target == 'pos' else np.array([sub_names.index(n) if n in sub_names else -1 for n in va['name']])
                mtr, mva = ytr >= 0, yva >= 0
                accs = {'feature': [], 'shuffled': []}
                for seed in a.seeds:
                    for v in ('feature', 'shuffled'):
                        y = ytr[mtr] if v == 'feature' else ytr[mtr][np.random.RandomState(seed).permutation(mtr.sum())]
                        f = fit_mlp(Xtr[mtr], y, ncls, seed); p = f(Xva[mva])
                        accs[v].append({g: float((p[va['unseen'][mva] == flag] == yva[mva][va['unseen'][mva] == flag]).mean()) for g, flag in (('seen', False), ('unseen', True))})
                res[f'{key}_{feat}'] = {v: {g: float(np.mean([r[g] for r in accs[v]])) for g in ('seen', 'unseen')} for v in accs}
                print(f'{key.upper()} acc [{feat}] ({ncls}-way): seen {res[f"{key}_{feat}"]["feature"]["seen"]:.3f} unseen {res[f"{key}_{feat}"]["feature"]["unseen"]:.3f} | '
                      f'shuffled-label: seen {res[f"{key}_{feat}"]["shuffled"]["seen"]:.3f} unseen {res[f"{key}_{feat}"]["shuffled"]["unseen"]:.3f}', flush=True)
    os.makedirs('data/order_probe', exist_ok=True)
    out = a.out or f'data/order_probe/{a.tag}.json'
    json.dump(res, open(out, 'w'), indent=1); print('->', out)


if __name__ == '__main__':
    main()
