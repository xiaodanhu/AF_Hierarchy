"""Generate ActivityNet-1.3 ZSL metadata: verbalizer (phrase hierarchy) +
attribute tables with per-split held-out classes (10 random splits x
{75/25 -> 50 unseen, 50/50 -> 100 unseen}, seed 42).

Class ids are the canonical ActivityNet alphabetical ordering (c000..c199),
matching the cNNN action_ids in activitynet_{train,test}_label.txt.

Phrase design: 27 mid-level groups derived from the official ActivityNet-1.3
taxonomy (immediate-parent clusters, merged/split for coherence — the 26-class
"Playing sports" cluster is split by sport family; tiny singleton clusters are
merged into their semantic neighbors).

Attribute design (8 families mapped onto the generic specialist slots so ALL
FineGym/THUMOS model plumbing is reused unchanged):

  family (slot)                          candidates
  motion (motion_category, 15-way)       walk_run_jump / swim_paddle / glide_ride /
                                         ride_steer / climb_hang / throw_catch /
                                         aim_release / strike_swing / kick /
                                         lift_press / rhythmic_body / hand_manipulate /
                                         cut_shape_build / wipe_scrub_wash / ingest_inhale
  object (named_element, ~60-way)        fine-grained primary implement/apparatus
  posture (direction, 5-way)             standing_moving / standing_stationary /
                                         seated / bent_kneeling / lying_inverted
  environment (body_shape, 9-way)        water_body / snow_ice / indoor_home /
                                         indoor_gym_studio / sports_field_court /
                                         outdoor_yard_street / outdoor_nature /
                                         stage_venue / workshop_garage
  social (multiplicity, 3-way)           solo / pair / group
  uses_water_liquid (inverted_phase, binary)
  handheld_implement (hands_on_apparatus, binary)
  repetitive_cyclic (leg_configuration, binary)

Within-phrase sibling separability (the property that makes ZSL composition
work): every pair of classes sharing a phrase differs in >=1 attribute.
Verified programmatically below.
"""

import json
import os
import random

OUT_DIR = '/data3/xiaodan8/actionformer6/configs/anet_zsl'
CLASSES_TXT = '/data3/xiaodan8/Activitynet/classes.txt'

MOTION = ['walk_run_jump', 'swim_paddle', 'glide_ride', 'ride_steer',
          'climb_hang', 'throw_catch', 'aim_release', 'strike_swing', 'kick',
          'lift_press', 'rhythmic_body', 'hand_manipulate', 'cut_shape_build',
          'wipe_scrub_wash', 'ingest_inhale']
POSTURE = ['standing_moving', 'standing_stationary', 'seated', 'bent_kneeling',
           'lying_inverted']
ENV = ['water_body', 'snow_ice', 'indoor_home', 'indoor_gym_studio',
       'sports_field_court', 'outdoor_yard_street', 'outdoor_nature',
       'stage_venue', 'workshop_garage']
SOCIAL = ['solo', 'pair', 'group']
BINS = ['uses_water_liquid', 'handheld_implement', 'repetitive_cyclic']

PHRASES = {
    'p00': 'water_boarding_boating',
    'p01': 'swim_dive',
    'p02': 'snow_ice_sports',
    'p03': 'wheel_board_riding',
    'p04': 'animal_riding_rodeo',
    'p05': 'athletics_field',
    'p06': 'gymnastics_apparatus',
    'p07': 'adventure_climb_swing',
    'p08': 'team_ball_sports',
    'p09': 'racquet_sports',
    'p10': 'target_precision_games',
    'p11': 'combat_strength_sports',
    'p12': 'weight_gym_workout',
    'p13': 'dance_performance',
    'p14': 'music_instruments',
    'p15': 'hair_head_grooming',
    'p16': 'face_body_care',
    'p17': 'food_drink_prep',
    'p18': 'eat_drink_smoke',
    'p19': 'interior_cleaning',
    'p20': 'laundry_textile_craft',
    'p21': 'surface_finish_repair',
    'p22': 'workshop_tool_craft',
    'p23': 'garden_yard_chores',
    'p24': 'vehicle_maintenance',
    'p25': 'animal_care',
    'p26': 'leisure_games_misc',
}

# Per class: (phrase, motion, object, posture, environment, social,
#             uses_water, handheld, repetitive)
SPEC = {
    0:   ('p16', 'wipe_scrub_wash', 'cosmetic_bottle',    'standing_stationary', 'outdoor_nature',      'solo',  0, 1, 1),  # Applying sunscreen
    1:   ('p10', 'aim_release',     'bow_arrow',          'standing_stationary', 'sports_field_court',  'solo',  0, 1, 0),  # Archery
    2:   ('p11', 'lift_press',      'none',               'seated',              'indoor_home',         'pair',  0, 0, 0),  # Arm wrestling
    3:   ('p24', 'cut_shape_build', 'bicycle',            'standing_stationary', 'workshop_garage',     'solo',  0, 1, 0),  # Assembling bicycle
    4:   ('p03', 'ride_steer',      'bicycle',            'seated',              'outdoor_urban_track', 'solo',  0, 1, 1),  # BMX
    5:   ('p17', 'cut_shape_build', 'batter_dough',       'standing_stationary', 'indoor_home',         'solo',  0, 0, 0),  # Baking cookies
    6:   ('p13', 'rhythmic_body',   'none',               'standing_moving',     'stage_venue',         'group', 0, 0, 0),  # Ballet
    7:   ('p25', 'wipe_scrub_wash', 'animal',             'bent_kneeling',       'indoor_home',         'solo',  1, 1, 1),  # Bathing dog
    8:   ('p13', 'hand_manipulate', 'baton_stick',        'standing_moving',     'sports_field_court',  'group', 0, 1, 1),  # Baton twirling
    9:   ('p08', 'kick',            'ball_large',         'standing_moving',     'outdoor_nature',      'group', 0, 0, 0),  # Beach soccer
    10:  ('p10', 'throw_catch',     'ball_small',         'standing_stationary', 'indoor_home',         'group', 0, 0, 0),  # Beer pong
    11:  ('p13', 'rhythmic_body',   'none',               'standing_stationary', 'stage_venue',         'solo',  0, 0, 1),  # Belly dance
    12:  ('p15', 'hand_manipulate', 'hair_dryer',         'seated',              'indoor_home',         'solo',  0, 1, 1),  # Blow-drying hair
    13:  ('p23', 'wipe_scrub_wash', 'leaf_blower',        'standing_moving',     'outdoor_yard_street', 'solo',  0, 1, 1),  # Blowing leaves
    14:  ('p15', 'hand_manipulate', 'none',               'seated',              'indoor_home',         'pair',  0, 0, 1),  # Braiding hair
    15:  ('p13', 'rhythmic_body',   'none',               'lying_inverted',      'outdoor_yard_street', 'group', 0, 0, 0),  # Breakdancing
    16:  ('p15', 'wipe_scrub_wash', 'hair_brush',         'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Brushing hair
    17:  ('p16', 'wipe_scrub_wash', 'toothbrush',         'standing_stationary', 'indoor_home',         'solo',  1, 1, 1),  # Brushing teeth
    18:  ('p26', 'cut_shape_build', 'sand_bucket',        'bent_kneeling',       'outdoor_nature',      'group', 0, 1, 0),  # Building sandcastles
    19:  ('p04', 'hand_manipulate', 'cape_textile',       'standing_moving',     'sports_field_court',  'pair',  0, 1, 0),  # Bullfighting
    20:  ('p07', 'walk_run_jump',   'harness_rope',       'lying_inverted',      'outdoor_nature',      'solo',  0, 0, 0),  # Bungee jumping
    21:  ('p04', 'throw_catch',     'lasso_rope',         'seated',              'sports_field_court',  'pair',  0, 1, 0),  # Calf roping
    22:  ('p04', 'ride_steer',      'animal',             'seated',              'outdoor_nature',      'pair',  0, 0, 1),  # Camel ride
    23:  ('p00', 'swim_paddle',     'paddle_boat_small',  'bent_kneeling',       'water_body',          'pair',  1, 1, 1),  # Canoeing
    24:  ('p11', 'rhythmic_body',   'none',               'lying_inverted',      'indoor_gym_studio',   'pair',  0, 0, 1),  # Capoeira
    25:  ('p22', 'cut_shape_build', 'knife_blade',        'standing_stationary', 'indoor_home',         'solo',  0, 1, 0),  # Carving jack-o-lanterns
    26:  ('p24', 'lift_press',      'car_vehicle',        'bent_kneeling',       'outdoor_yard_street', 'solo',  0, 1, 0),  # Changing car wheel
    27:  ('p13', 'walk_run_jump',   'pom_poms',           'standing_moving',     'indoor_gym_studio',   'group', 0, 1, 0),  # Cheerleading
    28:  ('p22', 'cut_shape_build', 'axe',                'standing_stationary', 'outdoor_yard_street', 'solo',  0, 1, 1),  # Chopping wood
    29:  ('p12', 'lift_press',      'barbell',            'standing_stationary', 'indoor_gym_studio',   'solo',  0, 1, 0),  # Clean and jerk
    30:  ('p19', 'wipe_scrub_wash', 'shoe_footwear',      'seated',              'indoor_home',         'solo',  1, 1, 1),  # Cleaning shoes
    31:  ('p19', 'wipe_scrub_wash', 'cloth_sponge',       'standing_stationary', 'indoor_home',         'solo',  1, 1, 1),  # Cleaning sink
    32:  ('p19', 'wipe_scrub_wash', 'window_squeegee',    'standing_stationary', 'indoor_home',         'solo',  1, 1, 1),  # Cleaning windows
    33:  ('p25', 'cut_shape_build', 'animal',             'seated',              'indoor_home',         'solo',  0, 1, 0),  # Clipping cat claws
    34:  ('p08', 'strike_swing',    'cricket_bat',        'standing_moving',     'sports_field_court',  'group', 0, 1, 0),  # Cricket
    35:  ('p10', 'strike_swing',    'mallet',             'standing_stationary', 'outdoor_yard_street', 'pair',  0, 1, 0),  # Croquet
    36:  ('p13', 'rhythmic_body',   'none',               'standing_moving',     'stage_venue',         'group', 0, 0, 1),  # Cumbia
    37:  ('p02', 'throw_catch',     'curling_stone',      'bent_kneeling',       'snow_ice',            'group', 0, 1, 0),  # Curling
    38:  ('p23', 'cut_shape_build', 'trimmer_scythe',     'standing_moving',     'outdoor_yard_street', 'solo',  0, 1, 1),  # Cutting the grass
    39:  ('p22', 'hand_manipulate', 'decor_ornaments',    'standing_stationary', 'indoor_home',         'group', 0, 0, 0),  # Decorating the Christmas tree
    40:  ('p25', 'throw_catch',     'disc_frisbee',       'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 0),  # Disc dog
    41:  ('p05', 'throw_catch',     'discus_disc',        'standing_moving',     'sports_field_court',  'solo',  0, 1, 0),  # Discus throw
    42:  ('p08', 'throw_catch',     'ball_large',         'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 0),  # Dodgeball
    43:  ('p11', 'lift_press',      'none',               'standing_moving',     'stage_venue',         'pair',  0, 0, 0),  # Doing a powerbomb
    44:  ('p12', 'rhythmic_body',   'none',               'lying_inverted',      'indoor_gym_studio',   'solo',  0, 0, 1),  # Doing crunches
    45:  ('p11', 'strike_swing',    'sword_foil',         'standing_moving',     'indoor_gym_studio',   'pair',  0, 1, 0),  # Doing fencing
    46:  ('p11', 'strike_swing',    'none',               'standing_moving',     'indoor_gym_studio',   'pair',  0, 0, 0),  # Doing karate
    47:  ('p11', 'kick',            'none',               'standing_moving',     'indoor_gym_studio',   'pair',  0, 0, 0),  # Doing kickboxing
    48:  ('p03', 'ride_steer',      'motorbike',          'seated',              'outdoor_nature',      'group', 0, 1, 1),  # Doing motocross
    49:  ('p16', 'hand_manipulate', 'nail_tool',          'seated',              'indoor_home',         'solo',  0, 1, 0),  # Doing nails
    50:  ('p12', 'walk_run_jump',   'step_platform',      'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 1),  # Doing step aerobics
    51:  ('p18', 'ingest_inhale',   'drink_vessel',       'seated',              'indoor_home',         'group', 1, 1, 0),  # Drinking beer
    52:  ('p18', 'ingest_inhale',   'drink_vessel',       'seated',              'indoor_home',         'solo',  1, 1, 0),  # Drinking coffee
    53:  ('p14', 'strike_swing',    'marching_drums',     'standing_moving',     'outdoor_yard_street', 'group', 0, 1, 1),  # Drum corps
    54:  ('p12', 'ride_steer',      'elliptical_machine', 'standing_moving',     'indoor_gym_studio',   'solo',  0, 1, 1),  # Elliptical trainer
    55:  ('p24', 'cut_shape_build', 'bicycle',            'bent_kneeling',       'workshop_garage',     'solo',  0, 1, 0),  # Fixing bicycle
    56:  ('p21', 'cut_shape_build', 'hand_tool',          'bent_kneeling',       'outdoor_yard_street', 'solo',  0, 1, 0),  # Fixing the roof
    57:  ('p07', 'glide_ride',      'playground_slide',   'seated',              'outdoor_yard_street', 'solo',  0, 0, 0),  # Fun sliding down
    58:  ('p08', 'kick',            'ball_large',         'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 0),  # Futsal
    59:  ('p16', 'ingest_inhale',   'cosmetic_bottle',    'standing_stationary', 'indoor_home',         'solo',  1, 1, 0),  # Gargling mouthwash
    60:  ('p15', 'cut_shape_build', 'scissors_clipper',   'seated',              'indoor_home',         'pair',  0, 1, 0),  # Getting a haircut
    61:  ('p16', 'hand_manipulate', 'needle_ink',         'seated',              'indoor_home',         'pair',  0, 1, 0),  # Getting a piercing
    62:  ('p16', 'hand_manipulate', 'needle_ink',         'lying_inverted',      'indoor_home',         'pair',  0, 1, 1),  # Getting a tattoo
    63:  ('p25', 'wipe_scrub_wash', 'animal',             'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Grooming dog
    64:  ('p25', 'wipe_scrub_wash', 'animal',             'standing_stationary', 'outdoor_yard_street', 'solo',  0, 1, 1),  # Grooming horse
    65:  ('p05', 'throw_catch',     'hammer_wire',        'standing_moving',     'sports_field_court',  'solo',  0, 1, 1),  # Hammer throw
    66:  ('p24', 'wipe_scrub_wash', 'car_vehicle',        'standing_moving',     'outdoor_yard_street', 'pair',  1, 1, 1),  # Hand car wash
    67:  ('p20', 'wipe_scrub_wash', 'textile_garment',    'bent_kneeling',       'indoor_home',         'solo',  1, 0, 1),  # Hand washing clothes
    68:  ('p21', 'cut_shape_build', 'wallpaper_roll',     'standing_stationary', 'indoor_home',         'solo',  0, 0, 0),  # Hanging wallpaper
    69:  ('p18', 'ingest_inhale',   'food_solid',         'standing_stationary', 'indoor_home',         'solo',  0, 1, 0),  # Having an ice cream
    70:  ('p05', 'walk_run_jump',   'none',               'lying_inverted',      'sports_field_court',  'solo',  0, 0, 0),  # High jump
    71:  ('p10', 'strike_swing',    'baton_stick',        'standing_moving',     'outdoor_yard_street', 'group', 0, 1, 0),  # Hitting a pinata
    72:  ('p26', 'walk_run_jump',   'none',               'standing_moving',     'outdoor_yard_street', 'group', 0, 0, 1),  # Hopscotch
    73:  ('p04', 'ride_steer',      'animal',             'seated',              'outdoor_nature',      'solo',  0, 1, 1),  # Horseback riding
    74:  ('p26', 'rhythmic_body',   'hoop_ring',          'standing_stationary', 'outdoor_yard_street', 'solo',  0, 0, 1),  # Hula hoop
    75:  ('p08', 'strike_swing',    'hurling_stick',      'standing_moving',     'sports_field_court',  'group', 0, 1, 0),  # Hurling
    76:  ('p02', 'hand_manipulate', 'fishing_rod',        'seated',              'snow_ice',            'solo',  0, 1, 0),  # Ice fishing
    77:  ('p21', 'cut_shape_build', 'carpet_roll',        'bent_kneeling',       'indoor_home',         'solo',  0, 1, 0),  # Installing carpet
    78:  ('p20', 'hand_manipulate', 'clothes_iron',       'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Ironing clothes
    79:  ('p05', 'throw_catch',     'javelin_spear',      'standing_moving',     'sports_field_court',  'solo',  0, 1, 0),  # Javelin throw
    80:  ('p00', 'swim_paddle',     'paddle_boat_small',  'seated',              'water_body',          'solo',  1, 1, 1),  # Kayaking
    81:  ('p26', 'hand_manipulate', 'kite_line',          'standing_moving',     'outdoor_nature',      'solo',  0, 1, 0),  # Kite flying
    82:  ('p12', 'rhythmic_body',   'none',               'bent_kneeling',       'indoor_gym_studio',   'solo',  0, 0, 1),  # Kneeling
    83:  ('p20', 'hand_manipulate', 'knitting_needles',   'seated',              'indoor_home',         'solo',  0, 1, 1),  # Knitting
    84:  ('p21', 'cut_shape_build', 'floor_tile',         'bent_kneeling',       'indoor_home',         'solo',  0, 1, 0),  # Laying tile
    85:  ('p08', 'walk_run_jump',   'ball_large',         'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 1),  # Layup drill in basketball
    86:  ('p05', 'walk_run_jump',   'none',               'standing_moving',     'sports_field_court',  'solo',  0, 0, 0),  # Long jump
    87:  ('p03', 'glide_ride',      'skate_board',        'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 1),  # Longboarding
    88:  ('p17', 'hand_manipulate', 'batter_dough',       'standing_stationary', 'indoor_home',         'solo',  0, 0, 0),  # Making a cake
    89:  ('p17', 'hand_manipulate', 'pitcher_liquid',     'standing_stationary', 'indoor_home',         'solo',  1, 0, 0),  # Making a lemonade
    90:  ('p17', 'cut_shape_build', 'bread_filling',      'standing_stationary', 'indoor_home',         'solo',  0, 1, 0),  # Making a sandwich
    91:  ('p17', 'hand_manipulate', 'pan_stove',          'standing_stationary', 'indoor_home',         'solo',  0, 1, 0),  # Making an omelette
    92:  ('p17', 'hand_manipulate', 'cocktail_shaker',    'standing_stationary', 'indoor_home',         'solo',  1, 1, 1),  # Mixing drinks
    93:  ('p19', 'wipe_scrub_wash', 'mop_broom',          'standing_moving',     'indoor_home',         'solo',  1, 1, 1),  # Mooping floor
    94:  ('p23', 'ride_steer',      'mower_machine',      'standing_moving',     'outdoor_yard_street', 'solo',  0, 1, 1),  # Mowing the lawn
    95:  ('p10', 'aim_release',     'gun_marker',         'standing_moving',     'outdoor_nature',      'group', 0, 1, 0),  # Paintball
    96:  ('p21', 'wipe_scrub_wash', 'paint_brush_roller', 'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Painting
    97:  ('p21', 'wipe_scrub_wash', 'paint_brush_roller', 'standing_stationary', 'outdoor_yard_street', 'solo',  0, 1, 1),  # Painting fence
    98:  ('p21', 'wipe_scrub_wash', 'paint_brush_roller', 'bent_kneeling',       'indoor_home',         'solo',  0, 1, 1),  # Painting furniture
    99:  ('p17', 'cut_shape_build', 'vegetable_fruit',    'seated',              'indoor_home',         'solo',  0, 1, 1),  # Peeling potatoes
    100: ('p09', 'strike_swing',    'tabletennis_paddle', 'standing_stationary', 'indoor_gym_studio',   'pair',  0, 1, 0),  # Ping-pong
    101: ('p21', 'wipe_scrub_wash', 'trowel_plaster',     'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Plastering
    102: ('p01', 'walk_run_jump',   'diving_platform',    'lying_inverted',      'water_body',          'solo',  1, 0, 0),  # Plataform diving
    103: ('p14', 'hand_manipulate', 'accordion',          'seated',              'stage_venue',         'solo',  0, 1, 1),  # Playing accordion
    104: ('p09', 'strike_swing',    'badminton_racquet',  'standing_moving',     'indoor_gym_studio',   'pair',  0, 1, 0),  # Playing badminton
    105: ('p14', 'ingest_inhale',   'bagpipes',           'standing_moving',     'outdoor_yard_street', 'solo',  0, 1, 1),  # Playing bagpipes
    106: ('p08', 'strike_swing',    'ball_large',         'standing_moving',     'outdoor_nature',      'group', 0, 0, 0),  # Playing beach volleyball
    107: ('p26', 'hand_manipulate', 'game_cards',         'seated',              'stage_venue',         'group', 0, 1, 0),  # Playing blackjack
    108: ('p14', 'strike_swing',    'hand_drum',          'standing_stationary', 'stage_venue',         'solo',  0, 0, 1),  # Playing congas
    109: ('p14', 'strike_swing',    'drum_kit',           'seated',              'stage_venue',         'solo',  0, 1, 1),  # Playing drums
    110: ('p08', 'strike_swing',    'hockey_stick',       'standing_moving',     'sports_field_court',  'group', 0, 1, 0),  # Playing field hockey
    111: ('p14', 'ingest_inhale',   'flute',              'standing_stationary', 'stage_venue',         'solo',  0, 1, 1),  # Playing flauta
    112: ('p14', 'hand_manipulate', 'guitar',             'seated',              'stage_venue',         'solo',  0, 1, 1),  # Playing guitarra
    113: ('p14', 'ingest_inhale',   'harmonica',          'seated',              'stage_venue',         'solo',  0, 1, 1),  # Playing harmonica
    114: ('p02', 'strike_swing',    'hockey_stick',       'standing_moving',     'snow_ice',            'group', 0, 1, 0),  # Playing ice hockey
    115: ('p08', 'kick',            'ball_large',         'standing_moving',     'sports_field_court',  'group', 0, 0, 0),  # Playing kickball
    116: ('p08', 'throw_catch',     'lacrosse_stick',     'standing_moving',     'sports_field_court',  'group', 0, 1, 0),  # Playing lacrosse
    117: ('p14', 'hand_manipulate', 'piano',              'seated',              'stage_venue',         'solo',  0, 0, 1),  # Playing piano
    118: ('p04', 'strike_swing',    'mallet',             'seated',              'sports_field_court',  'group', 0, 1, 0),  # Playing polo
    119: ('p10', 'strike_swing',    'cue_stick',          'bent_kneeling',       'indoor_home',         'pair',  0, 1, 0),  # Playing pool
    120: ('p09', 'strike_swing',    'racquetball_racquet','standing_moving',     'indoor_gym_studio',   'pair',  0, 1, 0),  # Playing racquetball
    121: ('p26', 'hand_manipulate', 'puzzle_cube',        'seated',              'indoor_home',         'solo',  0, 1, 1),  # Playing rubik cube
    122: ('p14', 'ingest_inhale',   'saxophone',          'standing_stationary', 'stage_venue',         'solo',  0, 1, 1),  # Playing saxophone
    123: ('p09', 'strike_swing',    'squash_racquet',     'standing_moving',     'indoor_gym_studio',   'pair',  0, 1, 0),  # Playing squash
    124: ('p10', 'throw_catch',     'bowling_ball',       'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 0),  # Playing ten pins
    125: ('p14', 'hand_manipulate', 'violin_bow',         'standing_stationary', 'stage_venue',         'solo',  0, 1, 1),  # Playing violin
    126: ('p01', 'throw_catch',     'ball_large',         'lying_inverted',      'water_body',          'group', 1, 0, 0),  # Playing water polo
    127: ('p05', 'walk_run_jump',   'pole_vault_pole',    'lying_inverted',      'sports_field_court',  'solo',  0, 1, 0),  # Pole vault
    128: ('p19', 'wipe_scrub_wash', 'cloth_sponge',       'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Polishing forniture
    129: ('p19', 'wipe_scrub_wash', 'shoe_footwear',      'seated',              'indoor_home',         'solo',  0, 1, 1),  # Polishing shoes
    130: ('p03', 'walk_run_jump',   'jumping_stilts',     'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 1),  # Powerbocking
    131: ('p17', 'hand_manipulate', 'pan_stove',          'standing_stationary', 'indoor_home',         'solo',  1, 1, 0),  # Preparing pasta
    132: ('p17', 'cut_shape_build', 'vegetable_fruit',    'standing_stationary', 'indoor_home',         'solo',  0, 1, 0),  # Preparing salad
    133: ('p16', 'hand_manipulate', 'lens_small',         'standing_stationary', 'indoor_home',         'solo',  0, 0, 0),  # Putting in contact lenses
    134: ('p16', 'hand_manipulate', 'cosmetic_brush',     'seated',              'indoor_home',         'solo',  0, 1, 1),  # Putting on makeup
    135: ('p16', 'hand_manipulate', 'shoe_footwear',      'seated',              'indoor_home',         'solo',  0, 0, 0),  # Putting on shoes
    136: ('p00', 'swim_paddle',     'raft_tube',          'seated',              'water_body',          'group', 1, 1, 1),  # Rafting
    137: ('p23', 'wipe_scrub_wash', 'rake',               'standing_moving',     'outdoor_yard_street', 'solo',  0, 1, 1),  # Raking leaves
    138: ('p15', 'hand_manipulate', 'hair_curlers',       'seated',              'indoor_home',         'solo',  0, 1, 0),  # Removing curlers
    139: ('p24', 'wipe_scrub_wash', 'car_vehicle',        'standing_stationary', 'snow_ice',            'solo',  0, 1, 1),  # Removing ice from car
    140: ('p03', 'ride_steer',      'bumper_car',         'seated',              'indoor_gym_studio',   'group', 0, 1, 0),  # Riding bumper cars
    141: ('p00', 'glide_ride',      'raft_tube',          'seated',              'water_body',          'solo',  1, 0, 0),  # River tubing
    142: ('p07', 'climb_hang',      'harness_rope',       'standing_moving',     'outdoor_nature',      'solo',  0, 0, 1),  # Rock climbing
    143: ('p26', 'hand_manipulate', 'none',               'standing_stationary', 'indoor_home',         'pair',  0, 0, 1),  # Rock-paper-scissors
    144: ('p03', 'glide_ride',      'roller_skates',      'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 1),  # Rollerblading
    145: ('p21', 'cut_shape_build', 'hand_tool',          'bent_kneeling',       'outdoor_yard_street', 'solo',  0, 1, 1),  # Roof shingle removal
    146: ('p12', 'walk_run_jump',   'rope_line',          'standing_stationary', 'indoor_gym_studio',   'solo',  0, 1, 1),  # Rope skipping
    147: ('p05', 'walk_run_jump',   'none',               'standing_moving',     'outdoor_yard_street', 'group', 0, 0, 1),  # Running a marathon
    148: ('p00', 'ride_steer',      'sail_craft',         'seated',              'water_body',          'group', 1, 1, 0),  # Sailing
    149: ('p01', 'swim_paddle',     'scuba_gear',         'lying_inverted',      'water_body',          'pair',  1, 0, 1),  # Scuba diving
    150: ('p22', 'cut_shape_build', 'knife_blade',        'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Sharpening knives
    151: ('p15', 'cut_shape_build', 'razor',              'standing_stationary', 'indoor_home',         'solo',  1, 1, 1),  # Shaving
    152: ('p16', 'cut_shape_build', 'razor',              'seated',              'indoor_home',         'solo',  1, 1, 1),  # Shaving legs
    153: ('p05', 'throw_catch',     'shot_ball',          'standing_stationary', 'sports_field_court',  'solo',  0, 1, 0),  # Shot put
    154: ('p23', 'lift_press',      'shovel_spade',       'standing_moving',     'snow_ice',            'solo',  0, 1, 1),  # Shoveling snow
    155: ('p10', 'throw_catch',     'cue_stick',          'standing_stationary', 'indoor_gym_studio',   'pair',  0, 1, 0),  # Shuffleboard
    156: ('p03', 'glide_ride',      'skate_board',        'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 0),  # Skateboarding
    157: ('p02', 'glide_ride',      'snow_ski',           'standing_moving',     'snow_ice',            'solo',  0, 1, 1),  # Skiing
    158: ('p07', 'rhythmic_body',   'rope_line',          'standing_moving',     'outdoor_nature',      'solo',  0, 0, 0),  # Slacklining
    159: ('p18', 'ingest_inhale',   'cigarette',          'standing_stationary', 'indoor_home',         'solo',  0, 1, 1),  # Smoking a cigarette
    160: ('p18', 'ingest_inhale',   'hookah_pipe',        'seated',              'indoor_home',         'group', 0, 1, 1),  # Smoking hookah
    161: ('p12', 'lift_press',      'barbell',            'bent_kneeling',       'indoor_gym_studio',   'solo',  0, 1, 0),  # Snatch
    162: ('p02', 'glide_ride',      'raft_tube',          'seated',              'snow_ice',            'solo',  0, 0, 0),  # Snow tubing
    163: ('p02', 'glide_ride',      'snow_board',         'standing_moving',     'snow_ice',            'solo',  0, 0, 0),  # Snowboarding
    164: ('p12', 'ride_steer',      'spin_bike',          'seated',              'indoor_gym_studio',   'group', 0, 1, 1),  # Spinning
    165: ('p23', 'lift_press',      'shovel_spade',       'bent_kneeling',       'outdoor_yard_street', 'solo',  0, 1, 0),  # Spread mulch
    166: ('p01', 'walk_run_jump',   'diving_springboard', 'lying_inverted',      'water_body',          'solo',  1, 0, 0),  # Springboard diving
    167: ('p23', 'hand_manipulate', 'none',               'bent_kneeling',       'outdoor_nature',      'group', 0, 0, 0),  # Starting a campfire
    168: ('p11', 'lift_press',      'none',               'bent_kneeling',       'stage_venue',         'pair',  0, 0, 0),  # Sumo
    169: ('p00', 'glide_ride',      'surf_board',         'standing_moving',     'water_body',          'solo',  1, 0, 0),  # Surfing
    170: ('p01', 'swim_paddle',     'none',               'lying_inverted',      'water_body',          'solo',  1, 0, 1),  # Swimming
    171: ('p07', 'glide_ride',      'playground_swing',   'seated',              'outdoor_yard_street', 'solo',  0, 1, 1),  # Swinging at the playground
    172: ('p10', 'hand_manipulate', 'game_table_prop',    'standing_stationary', 'indoor_home',         'pair',  0, 1, 1),  # Table soccer
    173: ('p11', 'rhythmic_body',   'none',               'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 1),  # Tai chi
    174: ('p13', 'rhythmic_body',   'none',               'standing_moving',     'stage_venue',         'pair',  0, 0, 0),  # Tango
    175: ('p09', 'strike_swing',    'tennis_racquet',     'standing_stationary', 'sports_field_court',  'pair',  0, 1, 0),  # Tennis serve with ball bouncing
    176: ('p10', 'aim_release',     'dart_small',         'standing_stationary', 'indoor_home',         'group', 0, 1, 0),  # Throwing darts
    177: ('p23', 'cut_shape_build', 'garden_shears',      'standing_stationary', 'outdoor_yard_street', 'solo',  0, 1, 1),  # Trimming branches or hedges
    178: ('p05', 'walk_run_jump',   'none',               'standing_moving',     'sports_field_court',  'solo',  0, 0, 1),  # Triple jump
    179: ('p11', 'lift_press',      'rope_line',          'bent_kneeling',       'outdoor_yard_street', 'group', 0, 1, 1),  # Tug of war
    180: ('p06', 'walk_run_jump',   'none',               'lying_inverted',      'indoor_gym_studio',   'solo',  0, 0, 0),  # Tumbling
    181: ('p06', 'climb_hang',      'parallel_bars',      'lying_inverted',      'indoor_gym_studio',   'solo',  0, 1, 1),  # Using parallel bars
    182: ('p06', 'walk_run_jump',   'balance_beam',       'standing_moving',     'indoor_gym_studio',   'solo',  0, 0, 0),  # Using the balance beam
    183: ('p07', 'climb_hang',      'playground_bars',    'standing_moving',     'outdoor_yard_street', 'solo',  0, 0, 1),  # Using the monkey bar
    184: ('p06', 'lift_press',      'pommel_horse',       'lying_inverted',      'indoor_gym_studio',   'solo',  0, 1, 1),  # Using the pommel horse
    185: ('p12', 'swim_paddle',     'rowing_machine',     'seated',              'indoor_gym_studio',   'solo',  0, 1, 1),  # Using the rowing machine
    186: ('p06', 'climb_hang',      'uneven_bars',        'lying_inverted',      'indoor_gym_studio',   'solo',  0, 1, 1),  # Using uneven bars
    187: ('p19', 'wipe_scrub_wash', 'vacuum_cleaner',     'standing_moving',     'indoor_home',         'solo',  0, 1, 1),  # Vacuuming floor
    188: ('p08', 'strike_swing',    'ball_large',         'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 0),  # Volleyball
    189: ('p00', 'glide_ride',      'tow_rope_board',     'standing_moving',     'water_body',          'pair',  1, 1, 0),  # Wakeboarding
    190: ('p25', 'walk_run_jump',   'animal',             'standing_moving',     'outdoor_yard_street', 'solo',  0, 1, 1),  # Walking the dog
    191: ('p19', 'wipe_scrub_wash', 'dish_ware',          'standing_stationary', 'indoor_home',         'solo',  1, 1, 1),  # Washing dishes
    192: ('p16', 'wipe_scrub_wash', 'none',               'bent_kneeling',       'indoor_home',         'solo',  1, 0, 1),  # Washing face
    193: ('p16', 'wipe_scrub_wash', 'none',               'standing_stationary', 'indoor_home',         'solo',  1, 0, 1),  # Washing hands
    194: ('p00', 'glide_ride',      'tow_rope_ski',       'standing_moving',     'water_body',          'pair',  1, 1, 0),  # Waterskiing
    195: ('p22', 'wipe_scrub_wash', 'snow_ski',           'standing_stationary', 'workshop_garage',     'solo',  0, 1, 1),  # Waxing skis
    196: ('p22', 'cut_shape_build', 'welding_torch',      'bent_kneeling',       'workshop_garage',     'solo',  0, 1, 0),  # Welding
    197: ('p00', 'ride_steer',      'sail_craft',         'standing_moving',     'water_body',          'solo',  1, 1, 0),  # Windsurfing
    198: ('p20', 'cut_shape_build', 'paper_wrap',         'standing_stationary', 'indoor_home',         'solo',  0, 0, 0),  # Wrapping presents
    199: ('p13', 'rhythmic_body',   'none',               'standing_moving',     'indoor_gym_studio',   'group', 0, 0, 1),  # Zumba
}

# 'outdoor_urban_track' was used for BMX to separate it from motocross via env;
# fold it into the canonical env list as its own value (dirt/skate park track).
ENV = ENV + ['outdoor_urban_track']


def build_object_list():
    objs = sorted({s[2] for s in SPEC.values()})
    return objs


OBJECTS = build_object_list()
ATTR_NAMES = (
    [f'motion_{m}' for m in MOTION]
    + [f'obj_{o}' for o in OBJECTS]
    + [f'posture_{p}' for p in POSTURE]
    + [f'env_{e}' for e in ENV]
    + [f'social_{s}' for s in SOCIAL]
    + BINS
)

SPECIALIST_SPECS = {
    'motion_category':    {'candidates': [f'motion_{m}' for m in MOTION],   'binary': False},
    'named_element':      {'candidates': [f'obj_{o}' for o in OBJECTS],     'binary': False},
    'direction':          {'candidates': [f'posture_{p}' for p in POSTURE], 'binary': False},
    'body_shape':         {'candidates': [f'env_{e}' for e in ENV],         'binary': False},
    'multiplicity':       {'candidates': [f'social_{s}' for s in SOCIAL],   'binary': False},
    'inverted_phase':     {'candidates': ['uses_water_liquid'],  'binary': True},
    'hands_on_apparatus': {'candidates': ['handheld_implement'], 'binary': True},
    'leg_configuration':  {'candidates': ['repetitive_cyclic'],  'binary': True},
}


def attr_vector(cid):
    _, m, ob, po, en, so, water, hand, rep = SPEC[cid]
    v = [0] * len(ATTR_NAMES)
    v[ATTR_NAMES.index(f'motion_{m}')] = 1
    v[ATTR_NAMES.index(f'obj_{ob}')] = 1
    v[ATTR_NAMES.index(f'posture_{po}')] = 1
    v[ATTR_NAMES.index(f'env_{en}')] = 1
    v[ATTR_NAMES.index(f'social_{so}')] = 1
    v[ATTR_NAMES.index('uses_water_liquid')] = water
    v[ATTR_NAMES.index('handheld_implement')] = hand
    v[ATTR_NAMES.index('repetitive_cyclic')] = rep
    return v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = [l.strip() for l in open(CLASSES_TXT) if l.strip()]
    assert len(names) == 200

    # ---- coverage: every class exactly once, phrases valid ----
    assert sorted(SPEC.keys()) == list(range(200)), "SPEC must cover c000..c199"
    for cid, s in SPEC.items():
        assert s[0] in PHRASES, f"unknown phrase {s[0]} for c{cid:03d}"
        assert s[1] in MOTION and s[3] in POSTURE and s[4] in ENV and s[5] in SOCIAL, \
            f"bad attr value for c{cid:03d}: {s}"

    phrase_members = {pid: [] for pid in PHRASES}
    for cid, s in SPEC.items():
        phrase_members[s[0]].append(cid)

    # ---- sanity: within-phrase sibling separability ----
    print("Within-phrase sibling separability check:")
    ok = True
    for pid, cids in phrase_members.items():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = attr_vector(cids[i]), attr_vector(cids[j])
                ndiff = sum(x != y for x, y in zip(a, b))
                if ndiff == 0:
                    ok = False
                    print(f"  !! {PHRASES[pid]}: {names[cids[i]]} vs "
                          f"{names[cids[j]]}: IDENTICAL")
    n_pairs = sum(len(c) * (len(c) - 1) // 2 for c in phrase_members.values())
    print(f"  checked {n_pairs} within-phrase pairs across "
          f"{len(PHRASES)} phrases: {'ALL SEPARABLE' if ok else 'VIOLATIONS'}")
    assert ok, "sibling separability violated"

    # ---- verbalizer ----
    verbalizer = {
        'dataset': 'activitynet',
        # 'short' + actions' activity-by-short-name match hierarchy_utils schema
        'activities': {'a0': {'id': 'a0', 'name': 'Activities (AC)', 'short': 'AC'}},
        'phrases': {pid: {'id': pid, 'name': name, 'activity': 'a0', 'set': i}
                    for i, (pid, name) in enumerate(PHRASES.items())},
        'actions': {f'c{c:03d}': {'id': f'c{c:03d}', 'name': names[c],
                                  'phrase': SPEC[c][0], 'activity': 'AC'}
                    for c in range(200)},
    }
    vpath = os.path.join(OUT_DIR, 'anet_verbalizer.json')
    with open(vpath, 'w') as f:
        json.dump(verbalizer, f, indent=1)
    print(f"\nwrote {vpath}")
    print(f"phrases: {len(PHRASES)}, attributes: {len(ATTR_NAMES)}, "
          f"objects: {len(OBJECTS)}")

    # ---- attribute tables per split ----
    base_actions = {
        f'c{c:03d}': {'name': names[c], 'attrs': attr_vector(c),
                      'rotation_count': 0.0}
        for c in range(200)
    }
    rng = random.Random(42)
    split_summary = {}
    for tag, n_unseen in [('75', 50), ('50', 100)]:
        for s in range(10):
            unseen = sorted(rng.sample(range(200), n_unseen))
            table = {
                'attribute_names': ATTR_NAMES,
                'specialist_specs': SPECIALIST_SPECS,
                'actions': base_actions,
                'held_out_classes': [f'c{c:03d}' for c in unseen],
            }
            path = os.path.join(OUT_DIR, f'anet_attrs_split{s}_{tag}.json')
            with open(path, 'w') as f:
                json.dump(table, f, indent=1)
            split_summary[f'split{s}_{tag}'] = [names[c] for c in unseen]
    with open(os.path.join(OUT_DIR, 'splits_summary.json'), 'w') as f:
        json.dump(split_summary, f, indent=1)
    print(f"wrote 20 split tables + splits_summary.json to {OUT_DIR}")
    print("\nsplit0_75 unseen (first 10):", split_summary['split0_75'][:10])


if __name__ == '__main__':
    main()
