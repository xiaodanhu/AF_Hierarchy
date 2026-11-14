#!/usr/bin/env python
"""STAGE 5 — phrase-level zero-shot protocol asset generator (THUMOS14).

Generates, per split n in 0..4 (held-out = 2 multi-member phrases drawn with
np.random.RandomState(n) from the 6 multi-member phrases; singletons never
held out):

  splits/train_phrase/THUMOS14/train/split_{n}.list        seen ACTION names (arm B)
  splits/train_phrase/THUMOS14/test/split_{n}.list         2 held-out PHRASE names
  splits/train_phrase/THUMOS14/train_flat/split_{n}.list   seen PHRASE names (arm A)
  splits/train_phrase/THUMOS14/test_flat/split_{n}.list    copy of test list (arm A)
  data/phrase_anno/thumos14_phrase_{n}.json      Test videos: held-out-phrase segments
                                                 relabeled to the PHRASE name; all other
                                                 labels untouched (filtered by the label
                                                 map downstream — verified behavior of
                                                 thumos14.py:137 and metrics.py
                                                 remove_subset_labels)
  data/phrase_anno/thumos14_phraseflat_{n}.json  same, plus Validation (train) videos
                                                 relabeled action -> phrase (arm A)
  configs/thumos14_i3d_phrase_{n}.yaml           arm B config
  configs/thumos14_i3d_phraseflat_{n}.yaml       arm A config

Split-independent:
  data/phrase_anno/thumos14_phrase_cls_scores.json  ext scores: 'class' = 20 actions
      (original order) + 9 phrases (sorted); per-video vector extended with
      phrase score = max over member actions' scores. postprocess_results maps
      by NAME so ordering only needs to match this file's own 'class' list.
  configs/thumos14_phrasemix_ensemble.json          merge of the action ensemble
      (thumos14_qwen_ensemble.json) and the phrase ensemble
      (thumos14_phrase_ensemble.json): 29 keys x 11 sentences.

Exact-duplicate (label, segment) pairs after relabeling are dropped
(CliffDiving/Diving both map to 'aquatic dive' and share boundaries in THUMOS).
"""
import copy
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNO = '/data3/xiaodan8/thumos/annotations/thumos14.json'
SCORES = '/data3/xiaodan8/thumos/annotations/thumos14_cls_scores.json'
FAM_MAP = os.path.join(ROOT, 'configs/thumos14_fam_map.json')
ACT_ENS = os.path.join(ROOT, 'configs/thumos14_qwen_ensemble.json')
PHR_ENS = os.path.join(ROOT, 'configs/thumos14_phrase_ensemble.json')
MIX_ENS = os.path.join(ROOT, 'configs/thumos14_phrasemix_ensemble.json')
SPLIT_DIR = os.path.join(ROOT, 'splits/train_phrase/THUMOS14')
ANNO_DIR = os.path.join(ROOT, 'data/phrase_anno')
N_SPLITS = 5

CFG_TEMPLATE = """dataset_name: thumos
train_split: ['validation']
val_split: ['test']
dataset: {{
  json_file: {json_file},
  feat_folder: /data3/xiaodan8/thumos/i3d_features,
  subset_file: '{subset_file}',
  file_prefix: ~,
  file_ext: .npy,
  input_dim: 2048,
  feat_stride: 4,
  num_frames: 16,
  trunc_thresh: 0.5,
  crop_ratio: [0.9, 1.0],
  max_seq_len: 2304,
  data_split: 75,
}}

model: {{
  fpn_type: identity,
  max_buffer_len_factor: 6.0,
  n_mha_win_size: 19,
  fpn_with_ln: true,
  embd_dim: 256,
  embd_with_ln: True,
  fpn_dim: 256,
  fpn_start_level: 0,
  head_dim: 256,
  topk_ratio: 0.01,
  threshold: 100,
  text_model: "openai/clip-vit-base-patch16"
}}
opt: {{
  learning_rate: 0.0001,
  epochs: 16,
  weight_decay: 0.01,
  schedule_type: multistep,
  schedule_gamma: 0.1,
  schedule_steps: [10],
  prior_prob: 0.3,
  num_pred: 300,
}}
loader: {{
  batch_size: 2
}}

train_cfg: {{
  init_loss_norm: 100,
  clip_grad_l2norm: 0.1,
  cls_prior_prob: 0.01,
  center_sample: radius,
  center_sample_radius: 1.5,
}}

test_cfg: {{
  voting_thresh: 0.7,
  pre_nms_topk: 2000,
  max_seg_num: 2000,
  min_score: 0.001,
  ext_score_file: {ext_score_file},
}}

output_folder: ./ckpt/
"""


def dedup(ants):
    seen, out = set(), []
    for a in ants:
        key = (a['label'], round(float(a['segment'][0]), 4),
               round(float(a['segment'][1]), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def write_list(path, names):
    with open(path, 'w') as f:
        for c in names:
            f.write(c + '\n')


def main():
    fam = json.load(open(FAM_MAP))
    members = {}
    for act, phr in fam.items():
        members.setdefault(phr, []).append(act)
    multi = sorted(p for p, m in members.items() if len(m) > 1)
    single = sorted(p for p, m in members.items() if len(m) == 1)
    assert len(multi) == 6 and len(single) == 3, (multi, single)

    anno = json.load(open(ANNO))

    for sub in ['train', 'test', 'train_flat', 'test_flat']:
        os.makedirs(os.path.join(SPLIT_DIR, sub), exist_ok=True)
    os.makedirs(ANNO_DIR, exist_ok=True)

    summary = []
    for n in range(N_SPLITS):
        rs = np.random.RandomState(n)
        held = sorted(rs.choice(multi, 2, replace=False).tolist())
        seen_actions = sorted(a for a, p in fam.items() if p not in held)
        seen_phrases = sorted(p for p in members if p not in held)

        write_list(os.path.join(SPLIT_DIR, 'train', f'split_{n}.list'),
                   seen_actions)
        write_list(os.path.join(SPLIT_DIR, 'test', f'split_{n}.list'), held)
        write_list(os.path.join(SPLIT_DIR, 'train_flat', f'split_{n}.list'),
                   seen_phrases)
        write_list(os.path.join(SPLIT_DIR, 'test_flat', f'split_{n}.list'),
                   held)

        # ---- arm B annotation: relabel held-out-phrase segments in Test
        # videos to the phrase name; everything else untouched.
        db_b = copy.deepcopy(anno)
        n_relab = {p: 0 for p in held}
        for vid, v in db_b['database'].items():
            if v['subset'].lower() != 'test':
                continue
            for a in v.get('annotations', []):
                p = fam.get(a['label'])
                if p in held:
                    a['label'] = p
                    n_relab[p] += 1
            v['annotations'] = dedup(v.get('annotations', []))
        for p in held:
            assert n_relab[p] > 0, f'split {n}: no eval GT for phrase {p!r}'

        # ---- arm A annotation: same + Validation (train) videos relabeled
        # to phrases wholesale (held-out phrases then filtered by the
        # train_flat list downstream).
        db_a = copy.deepcopy(db_b)
        for vid, v in db_a['database'].items():
            if v['subset'].lower() != 'validation':
                continue
            for a in v.get('annotations', []):
                if a['label'] in fam:
                    a['label'] = fam[a['label']]
            v['annotations'] = dedup(v.get('annotations', []))

        pb = os.path.join(ANNO_DIR, f'thumos14_phrase_{n}.json')
        pa = os.path.join(ANNO_DIR, f'thumos14_phraseflat_{n}.json')
        json.dump(db_b, open(pb, 'w'))
        json.dump(db_a, open(pa, 'w'))

        # ---- configs
        ext = os.path.join(ANNO_DIR, 'thumos14_phrase_cls_scores.json')
        with open(os.path.join(ROOT, f'configs/thumos14_i3d_phrase_{n}.yaml'),
                  'w') as f:
            f.write(CFG_TEMPLATE.format(
                json_file=pb,
                subset_file='splits/train_phrase/THUMOS14/{mode}/split_{r3}.list',
                ext_score_file=ext))
        with open(os.path.join(ROOT,
                               f'configs/thumos14_i3d_phraseflat_{n}.yaml'),
                  'w') as f:
            f.write(CFG_TEMPLATE.format(
                json_file=pa,
                subset_file='splits/train_phrase/THUMOS14/{mode}_flat/split_{r3}.list',
                ext_score_file=ext))

        summary.append((n, held, len(seen_actions), len(seen_phrases),
                        dict(n_relab)))

    # ---- merged phrase-level external scores (split-independent)
    sc = json.load(open(SCORES))
    acts = sc['class']
    aidx = {a: i for i, a in enumerate(acts)}
    phrases = sorted(members)
    new_cls = list(acts) + phrases
    new_res = {}
    for vid, vec in sc['results'].items():
        vec = list(vec)
        for p in phrases:
            vec.append(max(vec[aidx[a]] for a in members[p]))
        new_res[vid] = vec
    json.dump({'class': new_cls, 'results': new_res},
              open(os.path.join(ANNO_DIR, 'thumos14_phrase_cls_scores.json'),
                   'w'))

    # ---- merged prompt ensemble (actions + phrases, uniform K)
    act_ens = json.load(open(ACT_ENS))
    phr_ens = json.load(open(PHR_ENS))
    ks = {len(v) for v in act_ens.values()} | {len(v) for v in phr_ens.values()}
    assert ks == {11}, f'non-uniform ensemble sizes: {ks}'
    assert set(phr_ens) == set(members), \
        f'phrase ensemble keys mismatch: {set(phr_ens) ^ set(members)}'
    mix = dict(act_ens)
    mix.update(phr_ens)
    assert len(mix) == 29
    json.dump(mix, open(MIX_ENS, 'w'), indent=1)

    print(f'ext scores: {len(new_cls)} classes x {len(new_res)} videos')
    print(f'ensemble: {len(mix)} keys x 11 sentences')
    for n, held, na, np_, relab in summary:
        print(f'split {n}: held-out={held}  seen_actions={na} '
              f'seen_phrases={np_}  test GT relabeled={relab}')


if __name__ == '__main__':
    main()
