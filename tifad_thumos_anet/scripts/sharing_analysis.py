"""Analysis: does an unseen THUMOS14 class's gain over Ti-FAD track the
sub-actions (child sentences) it shares with the SEEN classes of the split?

Inputs (all on disk):
  data/seen_eval/{vsa3,champ,van_t75}_s{0..9}.json  per-class unseen AP (fused)
  configs/thumos14_qwen_ensemble.json                {class: [root, child_1..10]}
  configs/thumos14_fam_map.json                      {class: family}
CLIP ViT-B/16 text encoder (openai/clip-vit-base-patch16, HF cache), CPU.

Output: data/rep_metrics/sharing_analysis.json + printed report.
"""
import json, os, sys, itertools
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

ROOT = '/data3/xiaodan8/tifad'
os.chdir(ROOT)
MODELS = {'ours': 'vsa3', 'textASA': 'champ', 'tifad': 'van_t75'}
SPLITS = list(range(10))
HI_THR = 0.9

trees = json.load(open('configs/thumos14_qwen_ensemble.json'))
fam = json.load(open('configs/thumos14_fam_map.json'))
classes = sorted(trees)
assert len(classes) == 20

# ---------------- CLIP text embeddings (frozen, ViT-B/16) ----------------
from transformers import CLIPTokenizer, CLIPModel
name = 'openai/clip-vit-base-patch16'
tok = CLIPTokenizer.from_pretrained(name)
clip = CLIPModel.from_pretrained(name).float().eval()

@torch.no_grad()
def embed(sents):
    t = tok(sents, padding=True, truncation=True, max_length=77, return_tensors='pt')
    e = clip.get_text_features(**t)
    return torch.nn.functional.normalize(e, dim=-1).numpy()

child_sents = {c: trees[c][1:] for c in classes}
root_sents = {c: trees[c][0] for c in classes}
def pretty(c):  # CamelCase -> words
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', c).lower()
name_sents = {c: f'a video of action {pretty(c)}' for c in classes}

all_children = [(c, i, s) for c in classes for i, s in enumerate(child_sents[c])]
E_child = embed([s for _, _, s in all_children])          # (200, 512)
E_root = embed([root_sents[c] for c in classes])          # (20, 512)
E_name = embed([name_sents[c] for c in classes])          # (20, 512)
idx = {c: [k for k, (cc, _, _) in enumerate(all_children) if cc == c] for c in classes}
S_child = E_child @ E_child.T
# anisotropy-corrected variant: remove the common mean direction, renormalise
E_c = E_child - E_child.mean(0, keepdims=True)
E_c /= np.linalg.norm(E_c, axis=1, keepdims=True)
S_child_c = E_c @ E_c.T
S_root = E_root @ E_root.T
S_name = E_name @ E_name.T
ci = {c: k for k, c in enumerate(classes)}

# ---------------- per-class AP ----------------
def load(tag, s):
    d = json.load(open(f'data/seen_eval/{tag}_s{s}.json'))
    return d['meta']['seen'], d['meta']['unseen'], d['unseen_fused_per_class']

rows = []
for s in SPLITS:
    seen, unseen, ap_ours = load(MODELS['ours'], s)
    seen2, unseen2, ap_asa = load(MODELS['textASA'], s)
    seen3, unseen3, ap_van = load(MODELS['tifad'], s)
    assert set(unseen) == set(unseen2) == set(unseen3) == set(ap_ours) == set(ap_van)
    seen_idx = [k for c in seen for k in idx[c]]
    for c in unseen:
        # (1) child-level sharing with seen classes
        sub = S_child[np.ix_(idx[c], seen_idx)]              # (10, n_seen*10)
        best = sub.max(1)
        arg = sub.argmax(1)
        share = float(best.mean())
        frac_hi = float((best > HI_THR).mean())
        # best matching pair (one example)
        j = int(best.argmax())
        k_seen = seen_idx[arg[j]]
        ex = {'unseen_child': child_sents[c][j],
              'seen_class': all_children[k_seen][0],
              'seen_child': all_children[k_seen][2],
              'cos': float(best[j])}
        # per-child best seen class (for reporting which seen classes it shares with)
        partners = {}
        for jj in range(len(idx[c])):
            sc = all_children[seen_idx[arg[jj]]][0]
            partners[sc] = partners.get(sc, 0) + 1
        share_c = float(S_child_c[np.ix_(idx[c], seen_idx)].max(1).mean())
        # family-restricted variant
        seen_fam = [x for x in seen if fam[x] == fam[c]]
        if seen_fam:
            sf_idx = [k for x in seen_fam for k in idx[x]]
            subf = S_child[np.ix_(idx[c], sf_idx)]
            share_fam = float(subf.max(1).mean())
            frac_hi_fam = float((subf.max(1) > HI_THR).mean())
        else:
            share_fam, frac_hi_fam = float('nan'), float('nan')
        # name-only similarity (max over seen classes)
        sidx = [ci[x] for x in seen]
        name_sim = float(S_name[ci[c], sidx].max())
        root_sim = float(S_root[ci[c], sidx].max())
        rows.append(dict(split=s, cls=c, family=fam[c], n_seen_same_family=len(seen_fam),
                         share=share, share_c=share_c, share_fam=share_fam, frac_hi=frac_hi, frac_hi_fam=frac_hi_fam,
                         name_sim=name_sim, root_sim=root_sim,
                         ap_ours=ap_ours[c], ap_asa=ap_asa[c], ap_tifad=ap_van[c],
                         gain_ours=ap_ours[c] - ap_van[c], gain_asa=ap_asa[c] - ap_van[c],
                         example=ex, partners=partners))

# ---------------- correlations ----------------
def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return dict(n=int(len(x)), spearman=None, pearson=None)
    rs, ps = spearmanr(x, y); rp, pp = pearsonr(x, y)
    return dict(n=int(len(x)), spearman=float(rs), spearman_p=float(ps),
                pearson=float(rp), pearson_p=float(pp))

pairs = {}
for sk in ['share', 'share_c', 'share_fam', 'frac_hi', 'name_sim', 'root_sim']:
    for gk in ['gain_ours', 'gain_asa']:
        pairs[f'{sk}__{gk}'] = corr([r[sk] for r in rows], [r[gk] for r in rows])

# per-class averages (paired over splits where class is unseen)
per_class = {}
for c in classes:
    rr = [r for r in rows if r['cls'] == c]
    def m(k):
        v = [r[k] for r in rr]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float('nan')
    if not rr:
        continue
    ex = max((r['example'] for r in rr), key=lambda e: e['cos'])
    partners = {}
    for r in rr:
        for k, v in r['partners'].items():
            partners[k] = partners.get(k, 0) + v
    per_class[c] = dict(family=fam[c], n_splits_unseen=len(rr),
                        share=m('share'), share_c=m('share_c'), share_fam=m('share_fam'), frac_hi=m('frac_hi'),
                        name_sim=m('name_sim'), root_sim=m('root_sim'),
                        ap_ours=m('ap_ours'), ap_asa=m('ap_asa'), ap_tifad=m('ap_tifad'),
                        gain_ours=m('gain_ours'), gain_asa=m('gain_asa'),
                        example=ex, partners=dict(sorted(partners.items(), key=lambda kv: -kv[1])))
covered = [c for c in classes if c in per_class]
never_unseen = [c for c in classes if c not in per_class]
pairs_cls = {}
for sk in ['share', 'share_c', 'share_fam', 'frac_hi', 'name_sim', 'root_sim']:
    for gk in ['gain_ours', 'gain_asa']:
        pairs_cls[f'{sk}__{gk}'] = corr([per_class[c][sk] for c in covered],
                                        [per_class[c][gk] for c in covered])
# also: does sharing predict absolute unseen AP (not gain)?
pairs_ap = {f'{sk}__ap_{mk}': corr([r[sk] for r in rows], [r[f'ap_{mk}'] for r in rows])
            for sk in ['share', 'name_sim'] for mk in ['ours', 'asa', 'tifad']}

# global stats of child-child similarity for context
off = S_child[~np.eye(len(S_child), dtype=bool)]
within = np.concatenate([S_child[np.ix_(idx[c], idx[c])][~np.eye(10, dtype=bool)] for c in classes])
offc = S_child_c[~np.eye(len(S_child), dtype=bool)]
ctx = dict(child_cos_all_pairs_mean=float(off.mean()), child_cos_all_pairs_p90=float(np.percentile(off, 90)),
           centered_child_cos_all_pairs_mean=float(offc.mean()),
           share_c_min=float(min(r['share_c'] for r in rows)), share_c_max=float(max(r['share_c'] for r in rows)),
           child_cos_within_class_mean=float(within.mean()),
           share_min=float(min(r['share'] for r in rows)), share_max=float(max(r['share'] for r in rows)))

out = dict(models=MODELS, clip=name, hi_thr=HI_THR, n_pairs=len(rows),
           classes_covered=covered, classes_never_unseen=never_unseen,
           correlations_pairs=pairs, correlations_classes=pairs_cls, correlations_absAP_pairs=pairs_ap,
           context=ctx, per_class=per_class, rows=rows)
os.makedirs('data/rep_metrics', exist_ok=True)
json.dump(out, open('data/rep_metrics/sharing_analysis.json', 'w'), indent=1)

# ---------------- report ----------------
def f(x, w=6, d=2):
    return f'{x:{w}.{d}f}' if np.isfinite(x) else ' ' * (w - 3) + 'n/a'
print(f'\n=== Sub-action sharing vs. unseen gain (THUMOS14, 10x 75/25 splits, n={len(rows)} class-split pairs) ===')
print(f'CLIP {name}; sharing = mean over 10 children of max cos to any SEEN child; hi = frac children with best cos > {HI_THR}')
print(f'context: child-child cos mean {ctx["child_cos_all_pairs_mean"]:.3f}, p90 {ctx["child_cos_all_pairs_p90"]:.3f}, within-class mean {ctx["child_cos_within_class_mean"]:.3f}; sharing range [{ctx["share_min"]:.3f}, {ctx["share_max"]:.3f}]')
print('\n-- correlations over (class, split) pairs --')
for k, v in pairs.items():
    print(f'  {k:24s} n={v["n"]:3d}  Spearman {v["spearman"]:+.3f} (p={v["spearman_p"]:.3f})  Pearson {v["pearson"]:+.3f} (p={v["pearson_p"]:.3f})')
print(f'\n-- correlations over classes (per-class means, n={len(covered)}; never unseen: {never_unseen}) --')
for k, v in pairs_cls.items():
    print(f'  {k:24s} n={v["n"]:3d}  Spearman {v["spearman"]:+.3f} (p={v["spearman_p"]:.3f})  Pearson {v["pearson"]:+.3f} (p={v["pearson_p"]:.3f})')
print('\n-- sharing vs absolute unseen AP (pairs) --')
for k, v in pairs_ap.items():
    print(f'  {k:24s} n={v["n"]:3d}  Spearman {v["spearman"]:+.3f} (p={v["spearman_p"]:.3f})  Pearson {v["pearson"]:+.3f}')
print('\n-- 20 classes sorted by sharing score (means over splits where unseen) --')
print(f'{"class":18s} {"fam":18s} k  share  shCtr  shFam  hi   name  | AP_TiFAD AP_ASA AP_ours | gain_ASA gain_ours | top seen partners')
for c in sorted(covered, key=lambda c: -per_class[c]['share']):
    p = per_class[c]
    part = ', '.join(f'{k}({v})' for k, v in list(p['partners'].items())[:3])
    print(f'{c:18s} {p["family"]:18s} {p["n_splits_unseen"]}  {p["share"]:.3f}  {p["share_c"]:.3f}  {f(p["share_fam"],5,3)}  {p["frac_hi"]:.2f} {p["name_sim"]:.3f} | {f(p["ap_tifad"])}  {f(p["ap_asa"])} {f(p["ap_ours"])} | {p["gain_asa"]:+7.2f}  {p["gain_ours"]:+7.2f}  | {part}')
srt = sorted(covered, key=lambda c: -per_class[c]['share'])
print('\n-- top-3 / bottom-3 by sharing: example shared child pair --')
for c in srt[:3] + srt[-3:]:
    p = per_class[c]; e = p['example']
    print(f'* {c} (share {p["share"]:.3f}, gain_ours {p["gain_ours"]:+.2f}, gain_ASA {p["gain_asa"]:+.2f})')
    print(f'    unseen child : {e["unseen_child"]}')
    print(f'    seen child   : [{e["seen_class"]}] {e["seen_child"]}   cos={e["cos"]:.3f}')
# LongJump / HighJump / PoleVault run-up + landing sharing (static, regardless of split)
print('\n-- jump-family child overlap (max cos between child sentences, split-independent) --')
for a, b in [('LongJump', 'HighJump'), ('LongJump', 'PoleVault'), ('HighJump', 'PoleVault')]:
    sub = S_child[np.ix_(idx[a], idx[b])]
    j, k = np.unravel_index(sub.argmax(), sub.shape)
    print(f'  {a} vs {b}: mean-of-max {sub.max(1).mean():.3f}, frac>{HI_THR} {(sub.max(1) > HI_THR).mean():.2f}; best pair cos={sub[j, k]:.3f}')
    print(f'     [{a}] {child_sents[a][j]}')
    print(f'     [{b}] {child_sents[b][k]}')
print(f'\nNote: classes never unseen in the 10 splits (no gain measurable): {never_unseen}')
