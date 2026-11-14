"""Generate THUMOS14 ZSL metadata: verbalizer (7 phrases) + attribute tables
with per-split held-out classes (10 random splits x {75/25, 50/50}, seed 42).

Class ids are the canonical THUMOS14 alphabetical ordering (c00..c19).

Attribute design (26 binary flags in 8 families, mapped onto the generic
specialist slots so ALL FineGym model plumbing is reused unchanged):

  family (slot)                      candidates
  motion (motion_category, 6-way)    throw_release / strike / jump_launch / lift / kick / catch
  implement (named_element, 6-way)   none / ball / striking_tool / field_implement / barbell / pole
  force (direction, 4-way)           overhead / downward / horizontal / upward
  environment (body_shape, 6-way)    aquatic_pool / aquatic_natural / track_field / court / table / grass_pitch
  has_runup (multiplicity, binary)
  aerial_phase (inverted_phase, binary)
  rotational_windup (hands_on_apparatus, binary)
  two_handed (leg_configuration, binary)

Within-phrase sibling separability (the property that makes ZSL composition
work): every pair of classes sharing a phrase differs in >=1 attribute.
Verified programmatically below.
"""

import json
import os
import random

OUT_DIR = '/data3/xiaodan8/actionformer6/configs/thumos_zsl'

CLASSES = [
    'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
    'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
    'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault',
    'Shotput', 'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking',
]

PHRASES = {
    'p00': ('field_throw',        [10, 12, 15, 18]),   # Hammer, Javelin, Shotput, Discus
    'p01': ('jump_field',         [11, 13, 14]),       # HighJump, LongJump, PoleVault
    'p02': ('aquatic_dive',       [4, 7]),             # CliffDiving, Diving
    'p03': ('implement_strike',   [2, 6, 9, 17]),      # Billiards, CricketShot, Golf, Tennis
    'p04': ('ball_throw',         [0, 5]),             # BaseballPitch, CricketBowling
    'p05': ('court_jump_strike',  [1, 19]),            # Dunk, VolleyballSpiking
    'p06': ('other',              [3, 8, 16]),         # CleanAndJerk, FrisbeeCatch, SoccerPenalty
}

MOTION = ['motion_throw_release', 'motion_strike', 'motion_jump_launch',
          'motion_lift', 'motion_kick', 'motion_catch']
IMPL = ['impl_none', 'impl_ball', 'impl_striking_tool', 'impl_field_implement',
        'impl_barbell', 'impl_pole']
FORCE = ['force_overhead', 'force_downward', 'force_horizontal', 'force_upward']
ENV = ['env_aquatic_pool', 'env_aquatic_natural', 'env_track_field',
       'env_court', 'env_table', 'env_grass_pitch']
BINS = ['has_runup', 'aerial_phase', 'rotational_windup', 'two_handed']
ATTR_NAMES = MOTION + IMPL + FORCE + ENV + BINS

# Per class: (motion, implement, force, environment, runup, aerial, rot_windup, two_handed)
SPEC = {
    0:  ('throw_release', 'ball',            'overhead',   'grass_pitch',    0, 0, 1, 0),  # BaseballPitch
    1:  ('jump_launch',   'ball',            'downward',   'court',          1, 1, 0, 0),  # BasketballDunk
    2:  ('strike',        'striking_tool',   'horizontal', 'table',          0, 0, 0, 0),  # Billiards
    3:  ('lift',          'barbell',         'upward',     'court',          0, 0, 0, 1),  # CleanAndJerk
    4:  ('jump_launch',   'none',            'downward',   'aquatic_natural',0, 1, 0, 0),  # CliffDiving
    5:  ('throw_release', 'ball',            'overhead',   'grass_pitch',    1, 0, 0, 0),  # CricketBowling
    6:  ('strike',        'striking_tool',   'horizontal', 'grass_pitch',    0, 0, 0, 1),  # CricketShot
    7:  ('jump_launch',   'none',            'downward',   'aquatic_pool',   0, 1, 0, 0),  # Diving
    8:  ('catch',         'none',            'horizontal', 'grass_pitch',    1, 0, 0, 0),  # FrisbeeCatch
    9:  ('strike',        'striking_tool',   'horizontal', 'grass_pitch',    0, 0, 1, 1),  # GolfSwing
    10: ('throw_release', 'field_implement', 'horizontal', 'track_field',    0, 0, 1, 1),  # HammerThrow
    11: ('jump_launch',   'none',            'upward',     'track_field',    1, 1, 0, 0),  # HighJump
    12: ('throw_release', 'field_implement', 'overhead',   'track_field',    1, 0, 0, 0),  # JavelinThrow
    13: ('jump_launch',   'none',            'horizontal', 'track_field',    1, 1, 0, 0),  # LongJump
    14: ('jump_launch',   'pole',            'upward',     'track_field',    1, 1, 0, 1),  # PoleVault
    15: ('throw_release', 'field_implement', 'overhead',   'track_field',    0, 0, 1, 0),  # Shotput
    16: ('kick',          'ball',            'horizontal', 'grass_pitch',    1, 0, 0, 0),  # SoccerPenalty
    17: ('strike',        'striking_tool',   'horizontal', 'court',          0, 0, 0, 0),  # TennisSwing
    18: ('throw_release', 'field_implement', 'horizontal', 'track_field',    0, 0, 1, 0),  # ThrowDiscus
    19: ('strike',        'none',            'downward',   'court',          1, 1, 0, 0),  # VolleyballSpiking
}

SPECIALIST_SPECS = {
    'motion_category':    {'candidates': MOTION, 'binary': False},
    'named_element':      {'candidates': IMPL,   'binary': False},
    'direction':          {'candidates': FORCE,  'binary': False},
    'body_shape':         {'candidates': ENV,    'binary': False},
    'multiplicity':       {'candidates': ['has_runup'],          'binary': True},
    'inverted_phase':     {'candidates': ['aerial_phase'],       'binary': True},
    'hands_on_apparatus': {'candidates': ['rotational_windup'],  'binary': True},
    'leg_configuration':  {'candidates': ['two_handed'],         'binary': True},
}


def attr_vector(cid):
    m, im, fo, en, run, aer, rot, twoh = SPEC[cid]
    v = [0] * len(ATTR_NAMES)
    v[ATTR_NAMES.index(f'motion_{m}')] = 1
    v[ATTR_NAMES.index(f'impl_{im}')] = 1
    v[ATTR_NAMES.index(f'force_{fo}')] = 1
    v[ATTR_NAMES.index(f'env_{en}')] = 1
    v[ATTR_NAMES.index('has_runup')] = run
    v[ATTR_NAMES.index('aerial_phase')] = aer
    v[ATTR_NAMES.index('rotational_windup')] = rot
    v[ATTR_NAMES.index('two_handed')] = twoh
    return v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- sanity: within-phrase sibling separability ----
    print("Within-phrase sibling separability check:")
    ok = True
    for pid, (pname, cids) in PHRASES.items():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = attr_vector(cids[i]), attr_vector(cids[j])
                ndiff = sum(x != y for x, y in zip(a, b))
                status = 'OK' if ndiff > 0 else '!! IDENTICAL !!'
                if ndiff == 0:
                    ok = False
                print(f"  {pname}: {CLASSES[cids[i]]} vs {CLASSES[cids[j]]}: "
                      f"{ndiff} differing attrs {status}")
    assert ok, "sibling separability violated"

    # ---- verbalizer ----
    action_to_phrase = {}
    for pid, (_, cids) in PHRASES.items():
        for c in cids:
            action_to_phrase[c] = pid
    verbalizer = {
        'dataset': 'thumos14',
        # 'short' + actions' activity-by-short-name match hierarchy_utils schema
        'activities': {'a0': {'id': 'a0', 'name': 'Sports (SP)', 'short': 'SP'}},
        'phrases': {pid: {'id': pid, 'name': name, 'activity': 'a0', 'set': i}
                    for i, (pid, (name, _)) in enumerate(PHRASES.items())},
        'actions': {f'c{c:02d}': {'id': f'c{c:02d}', 'name': CLASSES[c],
                                  'phrase': action_to_phrase[c], 'activity': 'SP'}
                    for c in range(20)},
    }
    vpath = os.path.join(OUT_DIR, 'thumos14_verbalizer.json')
    with open(vpath, 'w') as f:
        json.dump(verbalizer, f, indent=1)
    print(f"\nwrote {vpath}")

    # ---- attribute tables per split ----
    base_actions = {
        f'c{c:02d}': {'name': CLASSES[c], 'attrs': attr_vector(c),
                      'rotation_count': 0.0}
        for c in range(20)
    }
    rng = random.Random(42)
    split_summary = {}
    for tag, n_unseen in [('75', 5), ('50', 10)]:
        for s in range(10):
            unseen = sorted(rng.sample(range(20), n_unseen))
            table = {
                'attribute_names': ATTR_NAMES,
                'specialist_specs': SPECIALIST_SPECS,
                'actions': base_actions,
                'held_out_classes': [f'c{c:02d}' for c in unseen],
            }
            path = os.path.join(OUT_DIR, f'thumos14_attrs_split{s}_{tag}.json')
            with open(path, 'w') as f:
                json.dump(table, f, indent=1)
            split_summary[f'split{s}_{tag}'] = [CLASSES[c] for c in unseen]
    with open(os.path.join(OUT_DIR, 'splits_summary.json'), 'w') as f:
        json.dump(split_summary, f, indent=1)
    print(f"wrote 20 split tables + splits_summary.json to {OUT_DIR}")
    print("\nsplit0_75 unseen:", split_summary['split0_75'])
    print("split0_50 unseen:", split_summary['split0_50'])


if __name__ == '__main__':
    main()
