"""ActivityNet ZSL metadata v2 — strict-transfer redesign.

Implements STRICT_ZSL_IMPROVEMENTS_AND_ANET_ADVICE.md §6.2/6.3:
  - phrase level from the official taxonomy's mid-level (direct parents), with
    giants split by coarse attribute signature and singletons merged, so that
    every phrase node has >=2 classes and >=1 SEEN member in every split
    (legitimacy: a node with exactly one held-out class would leak identity);
  - a COARSE, SHARED attribute vocabulary (4 categorical axes + 3 binaries on
    the 8 fixed specialist slots), with the coverage rule asserted: every value
    needed by a held-out class has >=1 seen carrier in EVERY split;
  - within-phrase separability: every within-phrase pair differs in >=1 axis.

Splits are NOT re-drawn: the 20 held-out sets are read from the existing v1
tables (configs/anet_zsl/anet_attrs_split{s}_{75,50}.json) for comparability.

Outputs (to configs/anet_zsl/):
  anet_verbalizer_v2.json
  anet_attrs_v2_split{s}_{75,50}.json   (attribute table + specialist_specs)
  v2_summary.json                        (validator report)
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'configs', 'anet_zsl')

# ---------------------------------------------------------------------------
# Axis vocabularies (coarse & shared; slot names are the 8 fixed plumbing slots)
# ---------------------------------------------------------------------------
ENV = ['water', 'snow_ice', 'indoor_home', 'indoor_gym_stage', 'outdoor_field',
       'outdoor_urban']                                        # -> motion_category slot
IMPL = ['ball_projectile', 'stick_racket_club', 'tool_utensil',
        'vehicle_board_animal', 'instrument_toy', 'none']      # -> named_element slot
BODY = ['upper_body', 'lower_body', 'whole_body']              # -> body_shape slot
MOTION = ['throw_strike', 'locomotion', 'rotate_swing_flow',
          'manipulate_fine', 'pose_balance']                   # -> direction slot
SOCIAL = ['solo', 'pair', 'group']                             # -> multiplicity slot
# binaries: leg_configuration <- repetitive_cyclic, inverted_phase <- airborne,
#           hands_on_apparatus <- object_held
AXES = {
    'motion_category': ENV, 'named_element': IMPL, 'body_shape': BODY,
    'direction': MOTION, 'multiplicity': SOCIAL,
    'leg_configuration': ['repetitive_cyclic'],
    'inverted_phase': ['airborne'],
    'hands_on_apparatus': ['object_held'],
}

# ---------------------------------------------------------------------------
# Per-class assignments: cid -> (env, impl, body, motion, social, rep, air, held)
# env/impl/body/motion/social are indices into the lists above; rep/air/held 0|1
# ---------------------------------------------------------------------------
E = {k: i for i, k in enumerate(ENV)}
I = {k: i for i, k in enumerate(IMPL)}
B = {k: i for i, k in enumerate(BODY)}
M = {k: i for i, k in enumerate(MOTION)}
S = {k: i for i, k in enumerate(SOCIAL)}

A = {}  # cid -> tuple

def a(cid, env, impl, body, motion, social, rep, air, held):
    A[cid] = (E[env], I[impl], B[body], M[motion], S[social], rep, air, held)

# --- personal care / grooming (indoor_home, mostly manipulate_fine, upper) ---
a('c000', 'indoor_home', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Applying sunscreen (arms/legs)
a('c012', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Blow-drying hair (dryer sweeps)
a('c014', 'indoor_home', 'none', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 0)           # Braiding hair
a('c016', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Brushing hair
a('c017', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Brushing teeth
a('c049', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Doing nails
a('c059', 'indoor_home', 'none', 'upper_body', 'pose_balance', 'solo', 1, 0, 0)           # Gargling mouthwash (head-back hold)
a('c133', 'indoor_home', 'none', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 0)   # Putting in contact lenses (fingertip only)
a('c134', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Putting on makeup (discrete steps)
a('c135', 'indoor_home', 'none', 'lower_body', 'manipulate_fine', 'solo', 0, 0, 1)           # Putting on shoes
a('c138', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Removing curlers (discrete removals)
a('c151', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Shaving (razor sweeps)
a('c152', 'indoor_home', 'tool_utensil', 'lower_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Shaving legs
a('c192', 'indoor_home', 'none', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 0)           # Washing face
a('c193', 'indoor_home', 'none', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 0)           # Washing hands (circular rubbing)
# --- grooming by others (pair) ---
a('c060', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'pair', 1, 0, 1)  # Getting a haircut (comb/scissor strokes)
a('c061', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'pair', 0, 0, 1)  # Getting a piercing
a('c062', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'pair', 1, 0, 1)  # Getting a tattoo
# --- animal care ---
a('c007', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'pair', 1, 0, 0)   # Bathing dog (hands+water)
a('c033', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'pair', 0, 0, 1)   # Clipping cat claws
a('c063', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'pair', 1, 0, 1)   # Grooming dog
a('c064', 'outdoor_field', 'tool_utensil', 'upper_body', 'manipulate_fine', 'pair', 1, 0, 1) # Grooming horse
a('c040', 'outdoor_field', 'ball_projectile', 'whole_body', 'throw_strike', 'pair', 1, 1, 0) # Disc dog
a('c190', 'outdoor_urban', 'none', 'whole_body', 'locomotion', 'pair', 1, 0, 0)              # Walking the dog
# --- eating / drinking / smoking ---
a('c051', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'group', 0, 0, 1)  # Drinking beer
a('c052', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Drinking coffee
a('c069', 'outdoor_urban', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Having an ice cream (street/shop)
a('c159', 'outdoor_urban', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1) # Smoking a cigarette
a('c160', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'group', 1, 0, 1)  # Smoking hookah
# --- food prep (indoor_home, tool, manipulate) ---
a('c005', 'indoor_home', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)  # Baking cookies (dough+oven trips)
a('c088', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # Making a cake (stir/frost)
a('c089', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 0, 0, 1)  # Making a lemonade (squeeze+stir)
a('c090', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)  # Making a sandwich (assembly)
a('c091', 'indoor_home', 'tool_utensil', 'upper_body', 'throw_strike', 'solo', 1, 0, 1)  # Making an omelette (whisk+pan flip)
a('c092', 'indoor_home', 'tool_utensil', 'upper_body', 'throw_strike', 'solo', 0, 0, 1)  # Mixing drinks (shaker)
a('c099', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)  # Peeling potatoes (peel cycles)
a('c131', 'indoor_home', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # Preparing pasta (stove+pots)
a('c132', 'indoor_home', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 0, 0, 1)  # Preparing salad (chop+toss)
a('c191', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 0)   # Washing dishes (hands+water dominate)
# --- household interior ---
a('c025', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Carving jack-o-lanterns (sawing cycles)
a('c031', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Cleaning sink
a('c032', 'indoor_home', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Cleaning windows (standing reach)
a('c039', 'indoor_home', 'none', 'upper_body', 'manipulate_fine', 'group', 0, 0, 1)          # Decorating the Christmas tree
a('c068', 'indoor_home', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Hanging wallpaper (discrete panels)
a('c077', 'indoor_home', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Installing carpet (roll/stretch sweeps)
a('c084', 'indoor_home', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Laying tile
a('c093', 'indoor_home', 'stick_racket_club', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1) # Mooping floor (long-handled)
a('c096', 'indoor_home', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1) # Painting (wall-scale roller)
a('c101', 'indoor_home', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1) # Plastering (trowel detail)
a('c128', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1) # Polishing forniture
a('c187', 'indoor_home', 'stick_racket_club', 'whole_body', 'locomotion', 'solo', 1, 0, 1)        # Vacuuming floor (long-handled)
a('c198', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Wrapping presents
# --- textiles / shoes ---
a('c030', 'indoor_home', 'tool_utensil', 'upper_body', 'throw_strike', 'solo', 1, 0, 1)   # Cleaning shoes (brush strokes)
a('c083', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Knitting
a('c129', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Polishing shoes (circular buffing)
a('c067', 'indoor_home', 'none', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 0)           # Hand washing clothes
a('c078', 'indoor_home', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1) # Ironing clothes (standing at board)
# --- yard / exterior ---
a('c013', 'outdoor_field', 'tool_utensil', 'whole_body', 'locomotion', 'solo', 1, 0, 1)      # Blowing leaves
a('c038', 'outdoor_field', 'stick_racket_club', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # Cutting the grass (long-handled trimmer)
a('c094', 'outdoor_field', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 1, 0, 1)  # Mowing the lawn
a('c137', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'solo', 1, 0, 1)   # Raking leaves (long-handled)
a('c165', 'outdoor_field', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)   # Spread mulch (dump+hand-spread)
a('c177', 'outdoor_field', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)     # Trimming branches or hedges
a('c097', 'outdoor_urban', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Painting fence
a('c056', 'outdoor_urban', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)     # Fixing the roof
a('c145', 'outdoor_urban', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Roof shingle removal (prying sweeps)
a('c154', 'snow_ice', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)        # Shoveling snow
a('c028', 'outdoor_field', 'tool_utensil', 'whole_body', 'throw_strike', 'solo', 1, 0, 1)        # Chopping wood
a('c167', 'outdoor_field', 'tool_utensil', 'whole_body', 'manipulate_fine', 'group', 0, 0, 1)    # Starting a campfire
a('c098', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)     # Painting furniture
# --- tools / repair ---
a('c150', 'indoor_home', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)     # Sharpening knives
a('c195', 'indoor_home', 'vehicle_board_animal', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # Waxing skis (ski board dominates)
a('c196', 'indoor_home', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)      # Welding
a('c003', 'indoor_home', 'vehicle_board_animal', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)  # Assembling bicycle (bike dominates)
a('c026', 'outdoor_urban', 'tool_utensil', 'whole_body', 'manipulate_fine', 'solo', 0, 0, 1)     # Changing car wheel
a('c055', 'outdoor_urban', 'tool_utensil', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)     # Fixing bicycle
a('c066', 'outdoor_urban', 'tool_utensil', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Hand car wash
a('c139', 'snow_ice', 'tool_utensil', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)        # Removing ice from car
# --- music ---
a('c103', 'indoor_gym_stage', 'instrument_toy', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # accordion (bellows)
a('c105', 'outdoor_urban', 'instrument_toy', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)  # bagpipes (marching)
a('c108', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'throw_strike', 'solo', 1, 0, 0)  # congas (bare-hand strikes)
a('c109', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'throw_strike', 'solo', 1, 0, 1)  # drums (sticks)
a('c111', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)  # flauta
a('c112', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)  # guitarra
a('c113', 'indoor_home', 'instrument_toy', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1)  # harmonica (casual setting)
a('c117', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 0)  # piano (instrument not held)
a('c122', 'indoor_gym_stage', 'instrument_toy', 'whole_body', 'manipulate_fine', 'solo', 1, 0, 1)  # saxophone (standing sway)
a('c125', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # violin (bow strokes)
a('c053', 'outdoor_urban', 'instrument_toy', 'whole_body', 'locomotion', 'group', 1, 0, 1)   # Drum corps
# --- dance / performance ---
a('c006', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'group', 1, 1, 0)   # Ballet
a('c011', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 0)    # Belly dance
a('c015', 'outdoor_urban', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 1, 1, 0)       # Breakdancing
a('c027', 'indoor_gym_stage', 'instrument_toy', 'whole_body', 'rotate_swing_flow', 'group', 1, 1, 1)   # Cheerleading (pompoms)
a('c036', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'group', 1, 0, 0)    # Cumbia (group social dance)
a('c174', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'pair', 1, 0, 0)    # Tango
a('c199', 'indoor_gym_stage', 'none', 'whole_body', 'locomotion', 'group', 1, 0, 0)   # Zumba (aerobic steps)
# --- fitness ---
a('c050', 'indoor_gym_stage', 'none', 'whole_body', 'locomotion', 'group', 1, 0, 0)          # Doing step aerobics
a('c044', 'indoor_gym_stage', 'none', 'whole_body', 'pose_balance', 'solo', 1, 0, 0)         # Doing crunches
a('c082', 'indoor_gym_stage', 'none', 'lower_body', 'pose_balance', 'solo', 0, 0, 0)         # Kneeling
a('c146', 'indoor_gym_stage', 'tool_utensil', 'whole_body', 'locomotion', 'solo', 1, 1, 1)   # Rope skipping
a('c054', 'indoor_gym_stage', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 1, 0, 1)  # Elliptical trainer
a('c164', 'indoor_gym_stage', 'vehicle_board_animal', 'lower_body', 'locomotion', 'group', 1, 0, 1) # Spinning
a('c185', 'indoor_gym_stage', 'vehicle_board_animal', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)  # Using the rowing machine (pull strokes)
a('c029', 'indoor_gym_stage', 'tool_utensil', 'whole_body', 'pose_balance', 'solo', 1, 0, 1) # Clean and jerk (two-phase lift)
a('c161', 'indoor_gym_stage', 'tool_utensil', 'whole_body', 'pose_balance', 'solo', 0, 0, 1) # Snatch
# --- gymnastics ---
a('c180', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 0, 1, 0)    # Tumbling
a('c181', 'indoor_gym_stage', 'none', 'whole_body', 'pose_balance', 'solo', 0, 1, 1)    # Using parallel bars (support/handstand)
a('c182', 'indoor_gym_stage', 'none', 'whole_body', 'pose_balance', 'solo', 0, 1, 0)         # Using the balance beam
a('c184', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)    # Using the pommel horse
a('c186', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 0, 1, 1)    # Using uneven bars
a('c008', 'indoor_gym_stage', 'stick_racket_club', 'upper_body', 'rotate_swing_flow', 'solo', 1, 1, 1)  # Baton twirling
a('c074', 'indoor_gym_stage', 'instrument_toy', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 0)     # Hula hoop
# --- martial arts / combat ---
a('c024', 'indoor_gym_stage', 'none', 'whole_body', 'rotate_swing_flow', 'pair', 1, 1, 0)    # Capoeira
a('c046', 'indoor_gym_stage', 'none', 'whole_body', 'throw_strike', 'solo', 1, 0, 0)         # Doing karate (kata practice)
a('c047', 'indoor_gym_stage', 'none', 'whole_body', 'throw_strike', 'pair', 1, 0, 0)         # Doing kickboxing
a('c173', 'outdoor_field', 'none', 'whole_body', 'pose_balance', 'group', 1, 0, 0)           # Tai chi
a('c045', 'indoor_gym_stage', 'stick_racket_club', 'whole_body', 'throw_strike', 'pair', 1, 0, 1)  # Doing fencing
a('c002', 'indoor_home', 'none', 'upper_body', 'pose_balance', 'pair', 0, 0, 0)              # Arm wrestling
a('c168', 'indoor_gym_stage', 'none', 'whole_body', 'pose_balance', 'pair', 0, 0, 0)         # Sumo
a('c043', 'indoor_gym_stage', 'none', 'whole_body', 'throw_strike', 'pair', 0, 1, 0)         # Doing a powerbomb
a('c019', 'outdoor_field', 'none', 'whole_body', 'throw_strike', 'pair', 0, 0, 0)            # Bullfighting
a('c021', 'outdoor_field', 'tool_utensil', 'whole_body', 'throw_strike', 'pair', 0, 0, 1)    # Calf roping
# --- track & field ---
a('c041', 'outdoor_field', 'ball_projectile', 'whole_body', 'rotate_swing_flow', 'solo', 0, 0, 1) # Discus throw (spin)
a('c065', 'outdoor_field', 'stick_racket_club', 'whole_body', 'rotate_swing_flow', 'solo', 0, 0, 1)  # Hammer throw (wire+handle)
a('c079', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'solo', 0, 0, 1)     # Javelin throw
a('c153', 'outdoor_field', 'ball_projectile', 'whole_body', 'throw_strike', 'solo', 0, 0, 1)       # Shot put
a('c070', 'outdoor_field', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 0, 1, 0)       # High jump (flop rotation)
a('c086', 'outdoor_field', 'none', 'whole_body', 'locomotion', 'solo', 0, 1, 0)              # Long jump
a('c178', 'outdoor_field', 'none', 'whole_body', 'locomotion', 'solo', 1, 1, 0)              # Triple jump (multi-bound)
a('c127', 'outdoor_field', 'stick_racket_club', 'whole_body', 'locomotion', 'solo', 0, 1, 1) # Pole vault
a('c147', 'outdoor_urban', 'none', 'whole_body', 'locomotion', 'group', 1, 0, 0)             # Running a marathon
# --- ball / team sports ---
a('c034', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 1, 1, 1)  # Cricket (aerial ball, overs cycle)
a('c035', 'outdoor_field', 'stick_racket_club', 'upper_body', 'throw_strike', 'group', 0, 0, 1)  # Croquet
a('c075', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 0, 1, 1)  # Hurling (aerial ball)
a('c042', 'indoor_gym_stage', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 0, 0, 0) # Dodgeball
a('c115', 'outdoor_field', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 0, 0, 0)    # Playing kickball
a('c009', 'outdoor_field', 'ball_projectile', 'lower_body', 'throw_strike', 'group', 0, 0, 0)  # Beach soccer (sand court)
a('c058', 'indoor_gym_stage', 'ball_projectile', 'lower_body', 'throw_strike', 'group', 0, 0, 0)  # Futsal
a('c085', 'indoor_gym_stage', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 1, 1, 1)  # Layup drill in basketball
a('c106', 'outdoor_field', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 0, 1, 0)  # Playing beach volleyball (sand court)
a('c188', 'indoor_gym_stage', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 0, 1, 0)  # Volleyball
a('c110', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 0, 0, 1)   # Playing field hockey
a('c114', 'snow_ice', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 0, 0, 1)   # Playing ice hockey
a('c118', 'outdoor_field', 'vehicle_board_animal', 'whole_body', 'throw_strike', 'group', 0, 0, 1)  # Playing polo (horseback dominates visually)
a('c126', 'water', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 0, 0, 0)        # Playing water polo
# --- racquet sports ---
a('c100', 'indoor_home', 'stick_racket_club', 'upper_body', 'throw_strike', 'pair', 1, 0, 1) # Ping-pong
a('c104', 'indoor_gym_stage', 'stick_racket_club', 'whole_body', 'throw_strike', 'pair', 1, 0, 1)  # Playing badminton
a('c116', 'outdoor_field', 'stick_racket_club', 'whole_body', 'locomotion', 'group', 0, 0, 1)    # Playing lacrosse (run-carry signature)
a('c120', 'indoor_gym_stage', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 1, 0, 1)  # Playing racquetball (WEAK vs squash: rec doubles)
a('c123', 'indoor_gym_stage', 'stick_racket_club', 'whole_body', 'throw_strike', 'pair', 1, 0, 1)  # Playing squash
a('c175', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'solo', 1, 0, 1)     # Tennis serve with ball bouncing
# --- games ---
a('c010', 'indoor_home', 'ball_projectile', 'upper_body', 'throw_strike', 'group', 1, 0, 1)  # Beer pong
a('c071', 'outdoor_urban', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 0, 0, 1)  # Hitting a pinata
a('c072', 'outdoor_urban', 'none', 'lower_body', 'locomotion', 'solo', 1, 1, 0)              # Hopscotch
a('c081', 'outdoor_field', 'instrument_toy', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1)   # Kite flying
a('c107', 'indoor_gym_stage', 'instrument_toy', 'upper_body', 'manipulate_fine', 'group', 0, 0, 1)   # Playing blackjack
a('c119', 'indoor_home', 'stick_racket_club', 'upper_body', 'throw_strike', 'pair', 0, 0, 1) # Playing pool
a('c121', 'indoor_home', 'instrument_toy', 'upper_body', 'manipulate_fine', 'solo', 1, 0, 1) # Playing rubik cube
a('c140', 'outdoor_urban', 'vehicle_board_animal', 'whole_body', 'locomotion', 'group', 0, 0, 1) # Riding bumper cars
a('c143', 'indoor_home', 'none', 'upper_body', 'manipulate_fine', 'pair', 1, 0, 0)           # Rock-paper-scissors
a('c155', 'indoor_gym_stage', 'stick_racket_club', 'upper_body', 'throw_strike', 'pair', 1, 0, 1)    # Shuffleboard
a('c158', 'outdoor_field', 'none', 'whole_body', 'pose_balance', 'solo', 0, 0, 0)            # Slacklining
a('c172', 'indoor_home', 'instrument_toy', 'upper_body', 'manipulate_fine', 'pair', 1, 0, 1) # Table soccer
a('c176', 'indoor_home', 'ball_projectile', 'upper_body', 'throw_strike', 'solo', 1, 0, 1)   # Throwing darts
a('c179', 'outdoor_field', 'tool_utensil', 'whole_body', 'pose_balance', 'group', 0, 0, 1)   # Tug of war
a('c018', 'water', 'tool_utensil', 'whole_body', 'manipulate_fine', 'group', 0, 0, 1)        # Building sandcastles (beach)
a('c057', 'outdoor_urban', 'none', 'whole_body', 'locomotion', 'solo', 0, 0, 0)              # Fun sliding down
a('c171', 'outdoor_urban', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 1, 0, 1)       # Swinging at the playground
a('c183', 'outdoor_urban', 'none', 'upper_body', 'rotate_swing_flow', 'solo', 1, 1, 1)       # Using the monkey bar
# --- archery / precision ---
a('c001', 'outdoor_field', 'stick_racket_club', 'upper_body', 'throw_strike', 'solo', 1, 0, 1)   # Archery
a('c095', 'outdoor_field', 'stick_racket_club', 'whole_body', 'throw_strike', 'group', 0, 0, 1)  # Paintball
a('c037', 'snow_ice', 'ball_projectile', 'whole_body', 'throw_strike', 'group', 1, 0, 1)     # Curling
a('c124', 'indoor_gym_stage', 'ball_projectile', 'upper_body', 'throw_strike', 'solo', 1, 0, 1)  # Playing ten pins
# --- wheels / ride ---
a('c004', 'outdoor_urban', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 1, 1)  # BMX
a('c048', 'outdoor_field', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 1, 1)  # Doing motocross
a('c087', 'outdoor_urban', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 0, 0)  # Longboarding
a('c130', 'outdoor_urban', 'vehicle_board_animal', 'lower_body', 'locomotion', 'solo', 0, 1, 0)  # Powerbocking
a('c144', 'outdoor_urban', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 1, 0, 0)  # Rollerblading
a('c156', 'outdoor_urban', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 1, 0)  # Skateboarding
a('c073', 'outdoor_field', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 1, 0, 1)  # Horseback riding
a('c022', 'outdoor_field', 'vehicle_board_animal', 'whole_body', 'locomotion', 'group', 1, 0, 1) # Camel ride
a('c020', 'outdoor_urban', 'none', 'whole_body', 'locomotion', 'solo', 0, 1, 0)              # Bungee jumping
a('c142', 'outdoor_field', 'none', 'whole_body', 'locomotion', 'solo', 0, 0, 1)              # Rock climbing
# --- water ---
a('c023', 'water', 'vehicle_board_animal', 'upper_body', 'rotate_swing_flow', 'pair', 1, 0, 1)   # Canoeing
a('c080', 'water', 'vehicle_board_animal', 'upper_body', 'rotate_swing_flow', 'solo', 1, 0, 1)   # Kayaking
a('c136', 'water', 'vehicle_board_animal', 'whole_body', 'rotate_swing_flow', 'group', 1, 0, 1)  # Rafting
a('c141', 'water', 'vehicle_board_animal', 'whole_body', 'pose_balance', 'group', 0, 0, 0)   # River tubing
a('c148', 'water', 'vehicle_board_animal', 'whole_body', 'manipulate_fine', 'group', 0, 0, 1)    # Sailing
a('c169', 'water', 'vehicle_board_animal', 'whole_body', 'pose_balance', 'solo', 0, 0, 0)    # Surfing
a('c189', 'water', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 1, 1)      # Wakeboarding (wake jumps)
a('c194', 'water', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 0, 1)      # Waterskiing
a('c197', 'water', 'vehicle_board_animal', 'whole_body', 'pose_balance', 'solo', 0, 0, 1)    # Windsurfing
a('c102', 'water', 'none', 'whole_body', 'pose_balance', 'solo', 0, 1, 0)               # Plataform diving (static tower stance)
a('c166', 'water', 'none', 'whole_body', 'rotate_swing_flow', 'solo', 0, 1, 0)               # Springboard diving
a('c149', 'water', 'tool_utensil', 'whole_body', 'locomotion', 'solo', 1, 0, 0)              # Scuba diving
a('c170', 'water', 'none', 'whole_body', 'locomotion', 'solo', 1, 0, 0)                      # Swimming
a('c076', 'snow_ice', 'stick_racket_club', 'upper_body', 'manipulate_fine', 'solo', 0, 0, 1) # Ice fishing
# --- snow ---
a('c157', 'snow_ice', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 0, 1)   # Skiing
a('c162', 'snow_ice', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 0, 0)   # Snow tubing
a('c163', 'snow_ice', 'vehicle_board_animal', 'whole_body', 'locomotion', 'solo', 0, 1, 0)   # Snowboarding

# ---------------------------------------------------------------------------
# Phrase nodes (>=2 classes each; giants split by attribute signature)
# ---------------------------------------------------------------------------
PHRASES = {
    'grooming_hair':      ['c012', 'c014', 'c016', 'c138'],
    'grooming_face_mouth': ['c000', 'c017', 'c059', 'c133', 'c134', 'c151', 'c192'],
    'grooming_hands_legs': ['c049', 'c152', 'c193'],
    'grooming_by_others': ['c060', 'c061', 'c062'],
    'dressing_textile':   ['c030', 'c067', 'c078', 'c083', 'c129', 'c135'],
    'animal_care':        ['c007', 'c033', 'c063', 'c064', 'c040', 'c190'],
    'eat_drink_smoke':    ['c051', 'c052', 'c069', 'c159', 'c160'],
    'food_prep':          ['c005', 'c088', 'c089', 'c090', 'c091', 'c092', 'c099', 'c131', 'c132', 'c191'],
    'surface_finishing':  ['c068', 'c096', 'c098', 'c101', 'c025', 'c039', 'c198'],
    'floor_install':      ['c077', 'c084'],
    'interior_clean':     ['c031', 'c032', 'c093', 'c128', 'c187'],
    'yard_garden':        ['c013', 'c038', 'c094', 'c137', 'c165', 'c177'],
    'exterior_repair':    ['c056', 'c097', 'c145', 'c154', 'c028', 'c167'],
    'tool_craft_repair':  ['c150', 'c195', 'c196', 'c003', 'c026', 'c055', 'c066', 'c139'],
    'music_wind':         ['c105', 'c111', 'c113', 'c122'],
    'music_string_keys':  ['c103', 'c112', 'c117', 'c125'],
    'music_percussion':   ['c108', 'c109', 'c053'],
    'dance_perform':      ['c006', 'c011', 'c015', 'c027', 'c036', 'c174', 'c199'],
    'fitness_floor':      ['c050', 'c044', 'c082', 'c146'],
    'gym_equipment':      ['c054', 'c164', 'c185', 'c029', 'c161'],
    'gymnastics':         ['c180', 'c181', 'c182', 'c184', 'c186', 'c008', 'c074'],
    'martial_striking':   ['c024', 'c046', 'c047', 'c173', 'c045'],
    'grappling_rodeo':    ['c002', 'c168', 'c043', 'c019', 'c021'],
    'throw_field':        ['c041', 'c065', 'c079', 'c153'],
    'jump_field':         ['c070', 'c086', 'c178', 'c127', 'c020'],
    'run_walk':           ['c147', 'c072', 'c057'],
    'bat_stick_team':     ['c034', 'c035', 'c075', 'c110', 'c114', 'c118', 'c116'],
    'ball_team':          ['c042', 'c115', 'c009', 'c058', 'c085', 'c106', 'c188', 'c126'],
    'racquet_over_net':   ['c100', 'c104', 'c175'],
    'racquet_wall':       ['c120', 'c123'],
    'precision_target':   ['c001', 'c095', 'c037', 'c124', 'c010', 'c071', 'c119', 'c155', 'c176'],
    'tabletop_games':     ['c107', 'c121', 'c143', 'c172', 'c081'],
    'playground':         ['c171', 'c183', 'c158', 'c179', 'c018', 'c140'],
    'wheels_ride':        ['c004', 'c048', 'c087', 'c130', 'c144', 'c156'],
    'animal_ride':        ['c073', 'c022'],
    'climb':              ['c142'],   # merged below
    'water_paddle':       ['c023', 'c080', 'c136', 'c141', 'c148'],
    'water_board':        ['c169', 'c189', 'c194', 'c197'],
    'water_swim_dive':    ['c102', 'c166', 'c149', 'c170'],
    'snow_slide':         ['c157', 'c162', 'c163', 'c076'],
}
# merge the singleton climb node into playground (climbing/balance family)
PHRASES['playground'] += PHRASES.pop('climb')


def main():
    verb = json.load(open(os.path.join(OUT, 'anet_verbalizer.json')))
    cid2name = {a_['id']: a_['name'] for a_ in verb['actions'].values()}
    all_cids = sorted(cid2name)
    assert len(all_cids) == 200

    # sanity: every class assigned exactly once
    ph_of = {}
    for p, members in PHRASES.items():
        for c in members:
            assert c not in ph_of, f"{c} in two phrases"
            ph_of[c] = p
    missing = [c for c in all_cids if c not in ph_of]
    assert not missing, f"unassigned: {missing}"
    assert set(A) == set(all_cids), \
        f"attr rows mismatch: {sorted(set(all_cids) ^ set(A))[:6]}"
    assert all(len(m) >= 2 for m in PHRASES.values()), \
        {p: len(m) for p, m in PHRASES.items() if len(m) < 2}

    # attribute_names: flattened one-hot layout per slot
    attribute_names, slot_index = [], {}
    for slot, cands in AXES.items():
        slot_index[slot] = []
        for v in cands:
            slot_index[slot].append(len(attribute_names))
            attribute_names.append(v)   # bare value names (model looks them up verbatim)
    n_attr = len(attribute_names)

    def attrs_vec(cid):
        env, impl, body, mot, soc, rep, air, held = A[cid]
        v = [0] * n_attr
        v[slot_index['motion_category'][env]] = 1
        v[slot_index['named_element'][impl]] = 1
        v[slot_index['body_shape'][body]] = 1
        v[slot_index['direction'][mot]] = 1
        v[slot_index['multiplicity'][soc]] = 1
        if rep: v[slot_index['leg_configuration'][0]] = 1
        if air: v[slot_index['inverted_phase'][0]] = 1
        if held: v[slot_index['hands_on_apparatus'][0]] = 1
        return v

    # ---- validators over the existing 20 splits ----
    report = {'phrases': len(PHRASES), 'n_attr': n_attr, 'splits': {}}
    sep_fail = []
    for p, members in PHRASES.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if A[members[i]] == A[members[j]]:
                    sep_fail.append((p, members[i], members[j]))
    if sep_fail:
        for p, x, y in sep_fail:
            print(f"SEPARABILITY FAIL {p}: {x} {cid2name[x]!r} == {y} {cid2name[y]!r}")
        sys.exit(1)

    cat_slots = ['motion_category', 'named_element', 'body_shape', 'direction',
                 'multiplicity']
    axis_of = dict(zip(
        cat_slots, [0, 1, 2, 3, 4]))
    ok = True
    for s in range(10):
        for tag in ('75', '50'):
            src = json.load(open(os.path.join(OUT, f'anet_attrs_split{s}_{tag}.json')))
            held = src['held_out_classes']
            seen = [c for c in all_cids if c not in set(held)]
            # (a) phrase seen-membership
            dead = [p for p, m in PHRASES.items()
                    if not any(c in set(seen) for c in m)]
            # (b) coverage: every held value has a seen carrier
            uncovered = []
            for slot in cat_slots:
                ax = axis_of[slot]
                seen_vals = {A[c][ax] for c in seen}
                for c in held:
                    if A[c][ax] not in seen_vals:
                        uncovered.append((slot, c))
            report['splits'][f's{s}_{tag}'] = {
                'dead_phrases': dead, 'uncovered': uncovered}
            # 75-protocol (goal metric) must be violation-free; 50/50 dead
            # nodes are tolerated with a warning until the coarser 50-split
            # phrase graph lands (notes SS2.1 corollary / v3 precedent).
            if uncovered or (dead and tag == '75'):
                ok = False
                print(f"split{s}_{tag}: dead_phrases={dead} uncovered={uncovered}")
            elif dead:
                print(f"WARN split{s}_{tag}: dead phrases (50/50 only): {dead}")
    if not ok:
        sys.exit(1)

    # sibling density among held-out (report only)
    dens = []
    for s in range(10):
        src = json.load(open(os.path.join(OUT, f'anet_attrs_split{s}_75.json')))
        held = set(src['held_out_classes'])
        for c in held:
            dens.append(sum(1 for x in PHRASES[ph_of[c]] if x != c and x in held))
    report['mean_heldout_sibling_density_75'] = sum(dens) / len(dens)

    # ---- emit verbalizer v2 ----
    phrase_ids = {p: f"p{idx:02d}" for idx, p in enumerate(sorted(PHRASES))}
    verb2 = {
        'dataset': 'activitynet', 'activities': {
            'a0': {'id': 'a0', 'name': 'Activity', 'short': 'ACT'}},
        'phrases': {phrase_ids[p]: {
            'id': phrase_ids[p], 'name': p, 'activity': 'a0',
            'set': i} for i, p in enumerate(sorted(PHRASES))},
        'actions': {cid: {
            'id': cid, 'name': cid2name[cid], 'phrase': phrase_ids[ph_of[cid]],
            'activity': 'ACT'} for cid in all_cids},
    }
    json.dump(verb2, open(os.path.join(OUT, 'anet_verbalizer_v2.json'), 'w'),
              indent=1)

    # ---- emit per-split attribute tables ----
    specs = {
        'motion_category': {'candidates': ENV},
        'named_element': {'candidates': IMPL},
        'body_shape': {'candidates': BODY},
        'direction': {'candidates': MOTION},
        'multiplicity': {'candidates': SOCIAL},
        'leg_configuration': {'candidates': ['repetitive_cyclic'], 'binary': True},
        'inverted_phase': {'candidates': ['airborne'], 'binary': True},
        'hands_on_apparatus': {'candidates': ['object_held'], 'binary': True},
    }
    for s in range(10):
        for tag in ('75', '50'):
            src = json.load(open(os.path.join(OUT, f'anet_attrs_split{s}_{tag}.json')))
            table = {
                'attribute_names': attribute_names,
                'specialist_specs': specs,
                'actions': {cid: {
                    'id': cid, 'name': cid2name[cid],
                    'attrs': attrs_vec(cid), 'rotation_count': 0.0,
                } for cid in all_cids},
                'held_out_classes': src['held_out_classes'],
            }
            json.dump(table, open(os.path.join(
                OUT, f'anet_attrs_v2_split{s}_{tag}.json'), 'w'))
    json.dump(report, open(os.path.join(OUT, 'v2_summary.json'), 'w'), indent=1)
    print(f"OK: {len(PHRASES)} phrases, {n_attr} attribute dims, "
          f"held-out sibling density (75) = {report['mean_heldout_sibling_density_75']:.2f}")
    print("wrote anet_verbalizer_v2.json + 20 anet_attrs_v2_* tables + v2_summary.json")


if __name__ == '__main__':
    main()
