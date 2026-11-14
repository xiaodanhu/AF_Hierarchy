#!/usr/bin/env python
"""FineGym PER-LEVEL hierarchical zero-shot protocol asset generator.

PROTOCOL (documented choice)
----------------------------
Hierarchy levels on gym99 (99 element classes, event-clip videos in
data/finegym/finegym_gym99_tal.json):

  SUPER-ACTIVITY = apparatus  (4: VT / FX / BB / UB)   -- only used as a
                   sampling constraint and as the SEM_DEGRADE parent stem.
  ACTIVITY       = gym99 "set" (14 sets; ids from
                   /data3/xiaodan8/FineGym/gym99_categories.txt, readable
                   names below follow annotation/set_categories.txt).
                   CHOICE: sets (14) are the activities. Apparatus (4) were
                   rejected as activities because holding out an apparatus
                   removes an entire visual domain (camera/equipment), which
                   confounds semantic transfer with domain transfer; sets
                   share the apparatus context with seen siblings.
  ACTION         = gym99 element class (99).
  PHASE          = the 10 phase sentences per element in
                   configs/finegym_qwen_ensemble.json (sentence 0 is the
                   class prompt).

Splits (5, seeded): split n holds out 3 ENTIRE sets (all their member
elements) drawn with np.random.RandomState(SEED_BASE + n), constrained to
pairwise DISTINCT apparatus (avoids degenerate splits that strip one
apparatus of most of its sets). SEED_BASE=13 was chosen deterministically
(documented sweep over {0,7,13,29,42,97}) as the smallest base whose 5
splits (a) repeat no set more than 3x, (b) cover 9 distinct sets, and
(c) never draw the singleton VT set -- holding out VT would remove the
entire apparatus and confound semantic transfer with domain transfer,
mirroring the reason apparatus were rejected as the activity level.

Generated per split n in 0..4 (mirrors scripts/gen_phrase_splits.py):

  splits/train_activity/FineGym/train/split_{n}.list       seen-set ELEMENT
                                                           names (arm B)
  splits/train_activity/FineGym/test/split_{n}.list        3 held-out SET
                                                           (activity) names
  splits/train_activity/FineGym/train_flat/split_{n}.list  seen SET names
                                                           (arm A)
  splits/train_activity/FineGym/test_flat/split_{n}.list   copy of test (arm A)
  data/level_anno/finegym_setlevel_{n}.json   Test videos: segments of
      held-out-set elements relabeled to the SET name; all other labels
      untouched (filtered downstream by the subset list -- verified
      behavior of finegym.py:131 and metrics.py remove_subset_labels).
  data/level_anno/finegym_setflat_{n}.json    same, plus Validation (train)
      videos relabeled element -> set wholesale (arm A).
  configs/finegym_i3d_setlevel_{n}.yaml       arm B config (element-trained)
  configs/finegym_i3d_setflat_{n}.yaml        arm A config (activity-trained)

Split-independent:
  configs/finegym_set_map.json       {element: activity} (99 keys; also
                                     usable as TIFAD_FAMCE_MAP / TIFAD_VIDACT_MAP)
  configs/finegym_set_members.json   {activity: [member elements]} -- the
                                     FINEGYM_LEVEL_MAP consumed by the
                                     FINEGYM_LEVEL_COMPOSE=1 eval expansion.
  configs/finegym_set_ensemble.json  14 activities x 11 sentences.
      AUTHORING RECIPE: sentence 0 is a hand-written set-level prompt
      (below); sentence k (k=1..10) is sentence k of member element
      (k-1) mod M -- i.e. real, video-groundable phase sentences drawn
      round-robin from the member elements' ensembles so the set embedding
      covers member diversity while keeping the uniform K=11 the h19
      machinery asserts.
  configs/finegym_setmix_ensemble.json  element ensemble (99) + set
      ensemble (14) = 113 keys x 11 (arm B compose eval embeds BOTH set
      names and member element names).
  configs/finegym_set_arith.json     {activity: {parent: apparatus stem}}
      so TIFAD_SEM_DEGRADE has a parent stem for activity classes (arm A).

EVAL: activity-level unseen mAP is the PRIMARY metric (ANETdetection over
the 3 held-out set names, tIoU 0.3:0.1:0.7). Arm B additionally uses
element-level predictions as AUXILIARY support via FINEGYM_LEVEL_COMPOSE=1
(see libs/modeling/text.py + meta_archs.py): each eval activity expands to
[activity name] + member element names before the text tower, and the
per-snippet logits are max-pooled back to the activity -- so an unseen
activity is detected either by its own name or by any member element name.
"""
import copy
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNO = os.path.join(ROOT, 'data/finegym/finegym_gym99_tal.json')
CATS = '/data3/xiaodan8/FineGym/gym99_categories.txt'
ELEM_ENS = os.path.join(ROOT, 'configs/finegym_qwen_ensemble.json')
SPLIT_DIR = os.path.join(ROOT, 'splits/train_activity/FineGym')
ANNO_DIR = os.path.join(ROOT, 'data/level_anno')
N_SPLITS = 5
N_HELD = 3
SEED_BASE = 13

# gym99 set id -> (activity name, apparatus tag, hand-written sentence 0).
# Names follow /data3/xiaodan8/FineGym/annotation/set_categories.txt via the
# ordinal alignment of sorted gym99 set ids [1,21,22,24,25,31..35,41..44]
# onto set_categories rows 0..13 (verified: set 24 members are salto-forward
# elements = FX_front_salto, set 33 = BB_flight_salto, etc.).
SET_INFO = {
    1:  ("(VT) vault", "VT",
         "a video of a gymnastics vault: the gymnast sprints down the runway,"
         " punches the springboard, and flies over the vaulting table to a"
         " stuck landing"),
    21: ("(FX) leap, jump or hop", "FX",
         "a video of a leap, jump or hop element on floor exercise: the"
         " gymnast springs from the floor into a split, straddle or ring"
         " shape and lands back on her feet"),
    22: ("(FX) turn", "FX",
         "a video of a turn element on floor exercise: the gymnast rises"
         " onto one leg and pirouettes her whole body around the vertical"
         " axis"),
    24: ("(FX) salto forward", "FX",
         "a video of a forward salto on floor exercise: the gymnast punches"
         " from the floor and somersaults forward in the air before"
         " landing"),
    25: ("(FX) salto backward", "FX",
         "a video of a backward salto on floor exercise: after a round-off"
         " and flic-flac the gymnast somersaults backward in the air before"
         " landing"),
    31: ("(BB) leap, jump or hop", "BB",
         "a video of a leap, jump or hop element on the balance beam: the"
         " gymnast springs from the beam into a split or straddle shape and"
         " lands back on the beam"),
    32: ("(BB) turn", "BB",
         "a video of a turn element on the balance beam: the gymnast rises"
         " onto one leg on the beam and pirouettes around the vertical"
         " axis"),
    33: ("(BB) flight salto", "BB",
         "a video of an acrobatic salto on the balance beam: the gymnast"
         " leaves the beam, somersaults in the air, and lands back on the"
         " beam"),
    34: ("(BB) flight handspring", "BB",
         "a video of a handspring or flic-flac element on the balance beam:"
         " the gymnast passes through a hand-support flight phase and lands"
         " back on the beam"),
    35: ("(BB) dismount", "BB",
         "a video of a balance beam dismount: the gymnast tumbles along the"
         " beam and saltos off to land on the mat"),
    41: ("(UB) circle", "UB",
         "a video of a circle element on the uneven bars: the gymnast swings"
         " or circles around a bar, often finishing in a handstand"),
    42: ("(UB) flight on the same bar", "UB",
         "a video of a release element on the uneven bars: the gymnast"
         " releases a bar, flies, and regrasps the SAME bar"),
    43: ("(UB) transition flight", "UB",
         "a video of a transition flight element on the uneven bars: the"
         " gymnast flies between the low bar and the high bar"),
    44: ("(UB) dismount", "UB",
         "a video of an uneven bars dismount: the gymnast releases the bar,"
         " saltos in the air, and lands on the mat"),
}

APPARATUS_STEM = {
    "VT": "a video of a gymnastics element on the vaulting table",
    "FX": "a video of a gymnastics element on the floor exercise",
    "BB": "a video of a gymnastics element on the balance beam",
    "UB": "a video of a gymnastics element on the uneven bars",
}

CFG_TEMPLATE = """dataset_name: finegym
train_split: ['validation']
val_split: ['test']
dataset: {{
  json_file: {json_file},
  feat_folder: /data3/xiaodan8/FineGym/tifad_event_feats,
  subset_file: '{subset_file}',
  file_prefix: v_,
  file_ext: .npy,
  input_dim: 2048,
  default_fps: 30,
  feat_stride: 2,
  num_frames: 32,
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
    # ---- taxonomy: element -> set id (Clabel order preserved per set)
    elem2set, members = {}, {}
    for line in open(CATS):
        parts = [p.strip() for p in line.strip().split(';')]
        sid = int(parts[1].split(':')[1])
        name = parts[3].strip()
        elem2set[name] = sid
        members.setdefault(sid, []).append(name)
    assert len(elem2set) == 99 and set(members) == set(SET_INFO)

    elem_ens = json.load(open(ELEM_ENS))
    assert set(elem_ens) == set(elem2set), "ensemble keys != taxonomy"
    K = len(next(iter(elem_ens.values())))
    assert K == 11

    act_of = {e: SET_INFO[s][0] for e, s in elem2set.items()}
    act_members = {SET_INFO[s][0]: list(ms) for s, ms in members.items()}
    for a, ms in act_members.items():
        tag = a[1:3]  # "(VT) ..." -> VT
        assert all(m.startswith(f'({tag})') for m in ms), (a, ms)

    os.makedirs(ANNO_DIR, exist_ok=True)
    for sub in ['train', 'test', 'train_flat', 'test_flat']:
        os.makedirs(os.path.join(SPLIT_DIR, sub), exist_ok=True)

    # ---- split-independent assets
    json.dump(act_of, open(os.path.join(ROOT, 'configs/finegym_set_map.json'),
                           'w'), indent=1)
    json.dump(act_members,
              open(os.path.join(ROOT, 'configs/finegym_set_members.json'),
                   'w'), indent=1)

    set_ens = {}
    for sid, ms in members.items():
        act, tag, s0 = SET_INFO[sid]
        sents = [s0]
        for k in range(1, K):
            sents.append(elem_ens[ms[(k - 1) % len(ms)]][k])
        set_ens[act] = sents
    json.dump(set_ens,
              open(os.path.join(ROOT, 'configs/finegym_set_ensemble.json'),
                   'w'), indent=1)
    mix = dict(elem_ens)
    mix.update(set_ens)
    assert len(mix) == 113
    json.dump(mix,
              open(os.path.join(ROOT, 'configs/finegym_setmix_ensemble.json'),
                   'w'), indent=1)
    set_arith = {SET_INFO[s][0]: {'parent': APPARATUS_STEM[SET_INFO[s][1]]}
                 for s in SET_INFO}
    json.dump(set_arith,
              open(os.path.join(ROOT, 'configs/finegym_set_arith.json'),
                   'w'), indent=1)

    anno = json.load(open(ANNO))
    sorted_sids = sorted(SET_INFO)

    summary = []
    for n in range(N_SPLITS):
        rs = np.random.RandomState(SEED_BASE + n)
        perm = rs.permutation(sorted_sids)
        held_sids, apps = [], set()
        for sid in perm:
            app = SET_INFO[int(sid)][1]
            if app in apps:
                continue
            held_sids.append(int(sid))
            apps.add(app)
            if len(held_sids) == N_HELD:
                break
        held = sorted(SET_INFO[s][0] for s in held_sids)
        seen_elems = sorted(e for e, s in elem2set.items()
                            if s not in held_sids)
        seen_acts = sorted(SET_INFO[s][0] for s in SET_INFO
                           if s not in held_sids)

        write_list(os.path.join(SPLIT_DIR, 'train', f'split_{n}.list'),
                   seen_elems)
        write_list(os.path.join(SPLIT_DIR, 'test', f'split_{n}.list'), held)
        write_list(os.path.join(SPLIT_DIR, 'train_flat', f'split_{n}.list'),
                   seen_acts)
        write_list(os.path.join(SPLIT_DIR, 'test_flat', f'split_{n}.list'),
                   held)

        # ---- arm B annotation: relabel held-out-set segments in Test videos
        db_b = copy.deepcopy(anno)
        n_relab = {a: 0 for a in held}
        for vid, v in db_b['database'].items():
            if v['subset'].lower() != 'test':
                continue
            for a in v.get('annotations', []):
                if elem2set.get(a['label']) in held_sids:
                    # keep 'label_id' untouched -- both finegym.py and
                    # metrics.py key on 'label' when a subset list is given
                    a['label'] = act_of[a['label']]
                    n_relab[a['label']] += 1
            v['annotations'] = dedup(v.get('annotations', []))
        for a in held:
            assert n_relab[a] > 0, f'split {n}: no eval GT for activity {a!r}'

        # ---- arm A annotation: same + Validation relabeled wholesale
        db_a = copy.deepcopy(db_b)
        for vid, v in db_a['database'].items():
            if v['subset'].lower() != 'validation':
                continue
            for a in v.get('annotations', []):
                if a['label'] in act_of:
                    a['label'] = act_of[a['label']]
            v['annotations'] = dedup(v.get('annotations', []))

        pb = os.path.join(ANNO_DIR, f'finegym_setlevel_{n}.json')
        pa = os.path.join(ANNO_DIR, f'finegym_setflat_{n}.json')
        json.dump(db_b, open(pb, 'w'))
        json.dump(db_a, open(pa, 'w'))

        with open(os.path.join(ROOT,
                               f'configs/finegym_i3d_setlevel_{n}.yaml'),
                  'w') as f:
            f.write(CFG_TEMPLATE.format(
                json_file=pb,
                subset_file='splits/train_activity/FineGym/{mode}/split_{r3}.list'))
        with open(os.path.join(ROOT,
                               f'configs/finegym_i3d_setflat_{n}.yaml'),
                  'w') as f:
            f.write(CFG_TEMPLATE.format(
                json_file=pa,
                subset_file='splits/train_activity/FineGym/{mode}_flat/split_{r3}.list'))

        summary.append((n, held, len(seen_elems), len(seen_acts),
                        dict(n_relab)))

    print(f'set ensemble: {len(set_ens)} x {K}; mix: {len(mix)} x {K}')
    for n, held, ne, na, relab in summary:
        print(f'split {n}: held-out={held}  seen_elements={ne} '
              f'seen_activities={na}  test GT relabeled={relab}')


if __name__ == '__main__':
    main()
