"""SIBLING-vs-NON-SIBLING feature-distance test ("are
features of segments of sibling classes closer than features of segments of
non-sibling classes under a fixed ancestor?").  THUMOS14 75/25 splits 0-2,
UNSEEN-class test GT segments, frozen level-0 cls-trunk features already
dumped for the representation-validation study.  CPU only.

Representations (same rows, same grid, same loader override
`ds.subset_labels = seen+unseen; ds.data_list, ds.label_dict = ds._load_json_db(...)`):
  ours   : data/rep_metrics/champ_test_s{n}.npz          (scripts/rep_metrics.py)
  tifad  : data/phase_pseudo/probe_dump_variants/vanilla/split{n}_test.npz
  pda    : data/phase_pseudo/probe_dump_variants/pdanotif/split{n}_test.npz
                                                       (scripts/repvalid_variant.py)
The variant dumps carry no class id; the champ dump does (cls per row).  The
(video, in_seg) -> class lookup is taken from the champ dump and VERIFIED
against every variant dump row-for-row (video, seg, t, unseen flag must be
identical, which they are: same loader order, same level-0 point grid).

Ancestor of each class: configs/thumos14_fam_map.json (9 families).

Definitions (per split, cosine distance d = 1 - cos):
  segment vector  = mean of the level-0 features inside the GT segment.
  gallery         = ALL test GT segments (seen + unseen classes); THUMOS
                    unseen classes mostly have SEEN siblings, so seen-class
                    segments are legitimate siblings / non-siblings.
  same-video pairs are EXCLUDED (as in the retrieval metric of
                    scripts/rep_metrics.py); this also removes the
                    CliffDiving-subset-of-Diving duplicate-location artifact
                    (both are 'aquatic dive', overlapping segments share rows).
                    An include-same-video variant is also stored.
  for each unseen segment i:
    d_same = mean d to other segments of the same class
    d_sib  = mean d to segments of a different class with the same ancestor
    d_non  = mean d to segments with a different ancestor
    (i is dropped from the sib statistics if it has no eligible sibling)
  reported: mean d_same, d_sib, d_non, margin = d_non - d_sib,
            frac(d_sib < d_non) over unseen segments,
            NN: fraction of unseen segments whose nearest OTHER-class segment
            is a sibling; chance = per-query fraction of eligible other-class
            segments that are siblings, averaged over queries.
  per-location variant: the same on raw level-0 locations (query = unseen
  locations, gallery = all locations), subsampled per seed.

Usage:  python scripts/sibling_metric.py       -> data/rep_metrics/sibling.json
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPS = {
    'ours': 'data/rep_metrics/champ_test_s{n}.npz',
    'tifad': 'data/phase_pseudo/probe_dump_variants/vanilla/split{n}_test.npz',
    'pda': 'data/phase_pseudo/probe_dump_variants/pdanotif/split{n}_test.npz',
}
FAM = 'configs/thumos14_fam_map.json'
OUT = 'data/rep_metrics/sibling.json'
SPLITS = [0, 1, 2]
SEEDS = [0, 1, 2]
LOC_Q = 5000      # per-location: unseen query locations per seed
LOC_G = 30000     # per-location: gallery locations per seed


def load_rows(rep, n):
    z = np.load(REPS[rep].format(n=n), allow_pickle=True)
    if rep == 'ours':
        return {'feat': z['feat'], 't': z['t'], 'video': z['video'], 'seg': z['seg'],
                'cls': z['cls'].astype(np.int64), 'unseen': z['unseen'],
                'classes': list(z['classes'])}
    m = z['level'] == 0
    return {'feat': z['feat'][m], 't': z['t'][m], 'video': z['video'][m],
            'seg': z['in_seg'][m], 'unseen': z['in_unseen'][m]}


def attach_labels(ref, d, rep):
    """Verify variant rows == champ rows (video, seg, t, unseen) and copy cls."""
    for k in ('video', 'seg', 't', 'unseen'):
        assert d[k].shape == ref[k].shape, (rep, k, d[k].shape, ref[k].shape)
        assert np.array_equal(d[k], ref[k]), (rep, k, 'row mismatch vs champ dump')
    d['cls'] = ref['cls']
    d['classes'] = ref['classes']
    return d


def cos_dist(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return 1.0 - A @ B.T


def sib_stats(D, q_idx, cls, anc, vid, unseen, exclude_same_video=True):
    """D: (nq, N) cosine distances of query rows q_idx (into the N gallery)."""
    N = D.shape[1]
    out = {'d_same': [], 'd_sib': [], 'd_non': [], 'margin': [], 'ratio': [], 'sib_lt_non': [],
           'nn_is_sib': [], 'nn_chance': []}
    n_no_sib = 0
    for r, i in enumerate(q_idx):
        elig = np.ones(N, bool)
        elig[i] = False
        if exclude_same_video:
            elig &= vid != vid[i]
        same = elig & (cls == cls[i])
        sib = elig & (anc == anc[i]) & (cls != cls[i])
        non = elig & (anc != anc[i])
        d = D[r]
        if non.sum() == 0:
            continue
        d_non = float(d[non].mean())
        out['d_non'].append(d_non)
        if same.sum():
            out['d_same'].append(float(d[same].mean()))
        if sib.sum() == 0:
            n_no_sib += 1
            continue
        d_sib = float(d[sib].mean())
        out['d_sib'].append(d_sib)
        out['margin'].append(d_non - d_sib)          # per query, queries with a sibling
        out['ratio'].append(d_sib / max(d_non, 1e-8))  # scale-free: <1 means siblings closer
        out['sib_lt_non'].append(float(d_sib < d_non))
        other = sib | non
        j = np.flatnonzero(other)[np.argmin(d[other])]
        out['nn_is_sib'].append(float(sib[j]))
        out['nn_chance'].append(float(sib.sum() / other.sum()))
    res = {k: (float(np.mean(v)) if v else float('nan')) for k, v in out.items()}
    res['n_query'] = int(len(q_idx))
    res['n_query_with_sib'] = int(len(out['d_sib']))
    res['n_query_no_sib'] = int(n_no_sib)
    return res


def pool_segments(d):
    key = d['video'].astype(np.int64) * 1000003 + d['seg'].astype(np.int64)
    order = np.argsort(key, kind='stable')
    _, starts = np.unique(key[order], return_index=True)
    groups = np.split(order, starts[1:])
    P = np.stack([d['feat'][g].astype(np.float32).mean(0) for g in groups])
    first = np.asarray([g[0] for g in groups])
    return P, d['cls'][first], d['unseen'][first], d['video'][first], [len(g) for g in groups]


def main():
    fam = json.load(open(FAM))
    res = {'definitions': __doc__, 'per_split': {}, 'summary': {}}
    acc = {rep: {'segment': [], 'segment_incl_same_video': [], 'location': []} for rep in REPS}
    for n in SPLITS:
        ref = load_rows('ours', n)
        classes = ref['classes']
        anc_names = sorted(set(fam.values()))
        cls_anc = np.asarray([anc_names.index(fam[c]) for c in classes])
        unseen_cls = sorted(set(int(c) for c in ref['cls'][ref['unseen']]))
        seen_cls = [c for c in range(len(classes)) if c not in unseen_cls]
        # sibling availability per unseen class (excluding itself)
        sib_info = {}
        for c in unseen_cls:
            sibs = [classes[k] for k in range(len(classes)) if cls_anc[k] == cls_anc[c] and k != c]
            sib_info[classes[c]] = {'ancestor': fam[classes[c]],
                                    'siblings': sibs,
                                    'sibling_status': ['unseen' if k in unseen_cls else 'seen'
                                                       for k in range(len(classes))
                                                       if cls_anc[k] == cls_anc[c] and k != c]}
        split_res = {'unseen_classes': [classes[c] for c in unseen_cls],
                     'seen_classes': [classes[c] for c in seen_cls],
                     'unseen_class_siblings': sib_info, 'reps': {}}
        for rep in REPS:
            d = ref if rep == 'ours' else attach_labels(ref, load_rows(rep, n), rep)
            P, cls, uns, vid, sizes = pool_segments(d)
            anc = cls_anc[cls]
            q = np.flatnonzero(uns)
            D = cos_dist(P[q], P)
            seg = sib_stats(D, q, cls, anc, vid, uns, True)
            seg_in = sib_stats(D, q, cls, anc, vid, uns, False)
            seg['n_segments_total'] = int(len(P))
            seg['n_segments_unseen'] = int(len(q))
            seg['n_segments_seen'] = int(len(P) - len(q))
            # per-location, subsampled per seed
            loc_seeds = []
            X = d['feat']
            r_cls, r_anc, r_vid, r_uns = d['cls'], cls_anc[d['cls']], d['video'], d['unseen']
            all_idx = np.arange(len(X))
            uns_idx = np.flatnonzero(r_uns)
            for s in SEEDS:
                rs = np.random.RandomState(s)
                g = np.sort(rs.choice(all_idx, min(LOC_G, len(all_idx)), replace=False))
                qq = rs.choice(uns_idx, min(LOC_Q, len(uns_idx)), replace=False)
                # make sure queries are in the gallery (so self can be excluded by index)
                g = np.union1d(g, qq)
                pos = {int(v): k for k, v in enumerate(g)}
                q_pos = np.asarray([pos[int(v)] for v in qq])
                Dl = cos_dist(X[g[q_pos]].astype(np.float32), X[g].astype(np.float32))
                loc_seeds.append(sib_stats(Dl, q_pos, r_cls[g], r_anc[g], r_vid[g], r_uns[g], True))
            loc = {k: float(np.mean([r[k] for r in loc_seeds])) for k in loc_seeds[0]}
            loc_std = {k: float(np.std([r[k] for r in loc_seeds])) for k in loc_seeds[0]}
            loc['n_locations_total'] = int(len(X))
            loc['n_locations_unseen'] = int(len(uns_idx))
            split_res['reps'][rep] = {'segment': seg, 'segment_incl_same_video': seg_in,
                                      'location': loc, 'location_std_over_seeds': loc_std}
            acc[rep]['segment'].append(seg)
            acc[rep]['segment_incl_same_video'].append(seg_in)
            acc[rep]['location'].append(loc)
            print(f"split {n} {rep:6s} segs={len(P)} unseen={len(q)} | "
                  f"d_same={seg['d_same']:.4f} d_sib={seg['d_sib']:.4f} d_non={seg['d_non']:.4f} "
                  f"margin={seg['margin']:.4f} frac(sib<non)={seg['sib_lt_non']:.3f} "
                  f"NN-sib={seg['nn_is_sib']:.3f} (chance {seg['nn_chance']:.3f}) "
                  f"| loc: margin={loc['margin']:.4f} frac={loc['sib_lt_non']:.3f} "
                  f"NN-sib={loc['nn_is_sib']:.3f} (chance {loc['nn_chance']:.3f})", flush=True)
        res['per_split'][str(n)] = split_res
    keys = ['d_same', 'd_sib', 'd_non', 'margin', 'ratio', 'sib_lt_non', 'nn_is_sib', 'nn_chance',
            'n_query', 'n_query_with_sib', 'n_query_no_sib']
    for rep in REPS:
        res['summary'][rep] = {}
        for lvl in ('segment', 'segment_incl_same_video', 'location'):
            rows = acc[rep][lvl]
            res['summary'][rep][lvl] = {
                'mean_over_splits': {k: float(np.mean([r[k] for r in rows])) for k in keys},
                'std_over_splits': {k: float(np.std([r[k] for r in rows])) for k in keys}}
        loc_std = [res['per_split'][str(n)]['reps'][rep]['location_std_over_seeds'] for n in SPLITS]
        res['summary'][rep]['location']['mean_std_over_seeds'] = {
            k: float(np.mean([s[k] for s in loc_std])) for k in keys}
    res['meta'] = {'splits': SPLITS, 'seeds': SEEDS, 'loc_query_per_seed': LOC_Q,
                   'loc_gallery_per_seed': LOC_G, 'reps': REPS, 'fam_map': FAM}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"[save] {OUT}")
    print("\n==== mean over splits 0-2 (segment level, same-video pairs excluded) ====")
    hdr = f"{'rep':6s} {'d_same':>7s} {'d_sib':>7s} {'d_non':>7s} {'margin':>7s} {'sib/non':>7s} {'P(sib<non)':>10s} {'NN-sib':>7s} {'chance':>7s}"
    print(hdr)
    for rep in REPS:
        m = res['summary'][rep]['segment']['mean_over_splits']
        sd = res['summary'][rep]['segment']['std_over_splits']
        print(f"{rep:6s} {m['d_same']:7.4f} {m['d_sib']:7.4f} {m['d_non']:7.4f} {m['margin']:7.4f}±{sd['margin']:.3f} "
              f"{m['ratio']:7.3f}±{sd['ratio']:.3f} {m['sib_lt_non']:10.3f}±{sd['sib_lt_non']:.3f} "
              f"{m['nn_is_sib']:7.3f}±{sd['nn_is_sib']:.3f} {m['nn_chance']:7.3f}   (± = std over splits)")
    print("\n==== mean over splits 0-2 (per-location, 3 seeds) ====")
    print(hdr)
    for rep in REPS:
        m = res['summary'][rep]['location']['mean_over_splits']
        s = res['summary'][rep]['location']['mean_std_over_seeds']
        print(f"{rep:6s} {m['d_same']:7.4f} {m['d_sib']:7.4f} {m['d_non']:7.4f} {m['margin']:7.4f}±{s['margin']:.4f} "
              f"{m['ratio']:7.3f}±{s['ratio']:.3f} {m['sib_lt_non']:10.3f}±{s['sib_lt_non']:.3f} "
              f"{m['nn_is_sib']:7.3f}±{s['nn_is_sib']:.3f} {m['nn_chance']:7.3f}   (± = std over seeds)")
    print("\n==== unseen queries per split: total / with a sibling in gallery / without ====")
    for n in SPLITS:
        sp = res['per_split'][str(n)]
        m = sp['reps']['ours']['segment']
        print(f"split {n}: segments {m['n_segments_total']} (seen {m['n_segments_seen']}, unseen {m['n_segments_unseen']}); "
              f"unseen with sibling {m['n_query_with_sib']}, without {m['n_query_no_sib']}")
        for c, info in sp['unseen_class_siblings'].items():
            print(f"   {c:18s} [{info['ancestor']}] siblings: "
                  + (', '.join(f'{s}({st})' for s, st in zip(info['siblings'], info['sibling_status'])) or 'NONE'))


if __name__ == '__main__':
    main()
