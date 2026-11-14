# Stage 4: THUMOS14 Supervision-Asset Audit

Exhaustive manual audit of the AI-generated zero-shot supervision assets for all 20
THUMOS14 classes, verified independently against sports domain knowledge (rules,
standard technique, and what the actions actually look like on video).

Files audited:
- `configs/thumos14_ttb_attrs.json` (26-value attribute vocabulary, per-class sets)
- `configs/thumos14_qwen_ensemble.json` (sentence 0 = hierarchy prompt; sentences 1-10 = phases)
- `configs/thumos14_fam_map.json` + `configs/thumos14_arith.json` parents
  (cross-checked against `thumos14_nest_map.json` / `thumos14_parent_map.json`)

Severity scale:
- **breaks-transfer** — factually wrong signal in a discriminative channel: a wrong
  attribute assignment (corrupts the attribute specialist's training positives AND the
  class's composition score), or a phase/hierarchy claim that contradicts the class's
  signature visual.
- **minor** — a factually wrong clause among many correct ones; dilutes but is unlikely
  to dominate the class embedding.
- **cosmetic** — imprecise/awkward wording, negligible impact.

Deliverables: corrected attribute file written to `configs/thumos14_ttb_attrs_v2.json`.
The ensemble json is NOT rewritten; sentence corrections are listed in Section 2 only.

---

## Error counts

| Severity        | Attrs (Sec. 1) | Ensemble (Sec. 2) | Families (Sec. 3) | Total |
|-----------------|---------------:|------------------:|------------------:|------:|
| breaks-transfer | 2              | 5                 | 0                 | **7** |
| minor           | 3 (missing)    | 30                | 0                 | **33**|
| cosmetic        | 3 (notes)      | 11                | 2                 | **16**|

---

## 1. thumos14_ttb_attrs.json — attribute audit

Vocabulary (26 values, grouped): motion {catch, jump_launch, kick, lift, strike,
throw_release}; impl {ball, barbell, field_implement, none, pole, striking_tool};
force {downward, horizontal, overhead, upward}; env {aquatic_natural, aquatic_pool,
court, grass_pitch, table, track_field}; booleans {aerial_phase, has_runup,
rotational_windup, two_handed}.

Inferred assignment convention (validated across all 20 classes): `impl_ball` is
assigned iff the athlete's **body directly contacts/projects the ball** (BaseballPitch,
BasketballDunk, CricketBowling, SoccerPenalty); tool-mediated ball sports get the tool
(`impl_striking_tool`) instead. This convention is used to judge missing attributes.

### 1.1 Wrong assignments (fixed in v2)

| # | Class | Attribute | Verdict | Severity |
|---|-------|-----------|---------|----------|
| A1 | **Shotput** | `force_overhead` | **WRONG.** The shot is held at the neck/under the jaw and *pushed* up-and-out at ~37-40 deg; by rule (WA TR32.14) it is put from the shoulder with one hand and may not be taken behind the shoulder line — the arm is **never overhead** (that is javelin/baseball/cricket-bowling territory). Wrong positives corrupt the `force_overhead` specialist, and the class prompt actively mismatches the visuals. Fixed to `force_horizontal`, which also matches its true siblings ThrowDiscus/HammerThrow (similar release angles, same throwing-circle setting). | breaks-transfer |
| A2 | **CleanAndJerk** | `env_court` | **WRONG.** Olympic weightlifting happens on an indoor wooden platform/stage, not on any court. Weightlifting clips as positives corrupt the `env_court` specialist shared by BasketballDunk / TennisSwing / VolleyballSpiking. No vocabulary value fits a gym platform, so v2 **removes** the env attribute for this class (the scorer averages over however many attrs a class has; a missing group is safe, a wrong positive is not). | breaks-transfer |

### 1.2 Missing assignments (added in v2)

| # | Class | Missing attribute | Justification | Severity |
|---|-------|-------------------|---------------|----------|
| A3 | **VolleyballSpiking** | `motion_jump_launch` | A spike *is* a jump: approach, two-foot takeoff, strike at the apex. Its own family is "court jump **strike**", and its sibling BasketballDunk has `motion_jump_launch`; the asymmetry was unjustified. | minor |
| A4 | **VolleyballSpiking** | `impl_ball` (replacing `impl_none`) | The hand strikes the ball directly — exactly the convention under which SoccerPenalty (foot strikes ball) got `impl_ball`. `impl_none` removed: it denotes body-only actions (dives, jumps), and a large ball at net height is the class's most salient object. | minor |
| A5 | **TennisSwing** | `rotational_windup` | Tennis ground strokes have a pronounced backswing with torso/hip rotation — the same rationale under which GolfSwing (and BaseballPitch) carry the attribute. | minor |

### 1.3 Verified-correct highlights (audit questions from the task)

- **BaseballPitch `rotational_windup`** — correct: the wind-up/delivery involves clear hip-shoulder rotation.
- **SoccerPenalty `motion_kick`** — correct, present; `has_runup` also correctly present (penalty run-up).
- **HammerThrow: `has_runup` correctly ABSENT** — hammer is thrown from a 2.135 m circle with 3-4 turns in place; there is no run-up. Also correctly `two_handed` (only throw gripped with both hands).
- **JavelinThrow `has_runup`** correctly present; `rotational_windup` correctly absent (linear crossover run, no spin).
- **Shotput/ThrowDiscus `has_runup` correctly ABSENT** (circle throws); **`rotational_windup`** correct for both (spin technique; shot also via rotational style, and even the glide coils the trunk).
- All 20 env assignments other than A2 check out (Billiards→table, CliffDiving→natural water, Diving→pool, athletics→track_field, etc.).

### 1.4 Notes (no change made; cosmetic / judgment calls)

- **BaseballPitch `env_grass_pitch`**: the mound itself is dirt, but the surrounding field is grass; acceptable under this coarse vocabulary.
- **FrisbeeCatch `impl_none`**: the disc is equipment but no vocabulary value fits (not a ball, not a striking tool); `impl_none` kept as least-bad. The disc is instead carried by the text channel ("small projectile").
- **FrisbeeCatch**: `aerial_phase` deliberately NOT added despite the hierarchy sentence claiming "body airborne" — most catches are grounded; the *sentence* is the overclaim (see E-FC below), not the attrs.
- **Diving `has_runup`**: springboard approach + hurdle is a borderline run-up; left absent (platform dives start standing).
- **Billiards `two_handed`**: both hands are on the cue (grip + bridge) but only one grips; left absent to keep `two_handed` = "two-handed grip".
- **CleanAndJerk `force_overhead`**: the jerk does finish overhead, but `force_overhead` is otherwise an overhead *arm-swing/throw* attribute; left absent to avoid diluting that specialist.

---

## 2. thumos14_qwen_ensemble.json — sentence audit

Sentence index 0 is the hierarchy prompt (identical to `arith.json` "full" — verified
string-equal for all 20 classes); 1-10 are phase sentences. Corrections below are
**report-only** (per task, the ensemble json is not rewritten). "s3" = sentence index 3.

### Clean classes (no factual errors found)
**BaseballPitch** (full sequence windup→stride→release→follow-through is correct),
**CliffDiving** (feet-first-compatible entry, resurfacing order correct; s8-10 are
post-dive padding but not wrong), **LongJump** (run-up→board→hang/hitch-kick flight→
feet-first sand landing all correct).

### BasketballDunk
- s3 "arms swing downward as the ball **arcs toward the rim**" + s4 "hand **makes
  contact** with the ball, initiating a downward motion" — **minor**: in a dunk the
  ball is *held* throughout; these two sentences describe a separate free-flying ball
  being struck (reads like a tip-in/goaltend). Correction: "the player brings the held
  ball above the rim" / "the player drives the held ball downward".
- s7 "feet trailing behind" — cosmetic.

### Billiards
- s2 "The player's **left hand grips the cue stick near the butt**" — **minor**: hands
  swapped. The rear (dominant) hand grips the butt; the lead hand forms the *bridge*
  near the tip. Correction: "the rear hand grips the cue near the butt while the lead
  hand forms a bridge on the table".
- s4→s5 order — **minor**: s4 has the tip contacting the ball, then s5 has the cue
  "moving forward ... propelling the ball" — the forward stroke precedes contact.
  Swap the stroke and the contact.
- s3 "tip positioned just **above** the white ball" — cosmetic (tip is aimed *behind*
  the cue ball).
- s5 "cue stick moves forward in a smooth **arc**" — cosmetic (a cue stroke is straight).

### CleanAndJerk  (worst temporal structure in the file)
- s1 "Athlete stands upright, **barbell on shoulders, arms extended vertically**" —
  **breaks-transfer**: the lift *starts with the bar on the floor* (this defines the
  action's onset for TAL), and "on shoulders + arms extended vertically" is
  self-contradictory (rack position has bent arms). Correction: "Athlete crouches over
  the barbell on the floor, back flat, hands gripping the bar outside the knees."
- s2 dip ("dips knees, barbell lowers") placed **before** s3 lift-off from the ground —
  **minor** (scrambled order): the jerk dip belongs after the clean, i.e. after s5/s6.
- s6 "extends **arms fully**, barbell held **at chest**" — **minor**: physically
  impossible; in the rack position the elbows are high and arms bent. Correction:
  "Athlete stands erect with the bar racked on the front of the shoulders, elbows high."
- s8-s9 (lower slightly, then re-extend overhead) — **minor**: describes a press-out,
  which is a *failed/illegal* jerk; the real phases are split-under → recover feet →
  motionless hold. 

### CricketBowling
- Missing run-up — **minor**: the phases begin from a standing "ready stance"; a
  bowler's long run-up is the class's most visible temporal signature (and the attrs
  correctly carry `has_runup`). Add an initial "bowler sprints toward the crease"
  sentence.
- s3 "slight **bend in the elbow**" — **minor**: the defining feature of bowling vs
  throwing is the (near-)straight arm; describing elbow bend blurs it with
  BaseballPitch. Correction: "the straight arm rotates backward in a full circle".
- s5 "arm swings forward ..., **elbow high**, as the ball is released" — **minor**:
  "elbow high" is a javelin/baseball throwing cue; a bowler releases at the *top* of a
  straight-arm windmill. Correction: "the straight arm windmills over the top and the
  ball is released at the highest point".

### CricketShot
- s10 "The **bat lies on the grass**, with the batsman's hands still extended" —
  **minor**: batsmen do not drop the bat after a shot; it stays in hand (they run with
  it). Correction: "the batsman completes the follow-through, bat still in hand, and
  sets off for a run".

### Diving
- s6 "The diver's **body becomes fully horizontal**" just before entry —
  **breaks-transfer**: a competitive dive entry is *vertical* (a horizontal body at the
  water is a belly flop); this contradicts the class's signature visual. Correction:
  "the diver's body straightens into a vertical line, arms extended beyond the head
  toward the water".
- s2 "leaps forward ... as they **begin to descend**" — **minor**: the takeoff rises
  first; descent does not begin at the leap.
- s5 "legs straighten and extend **downward**, arms **trailing behind**" — **minor**:
  for the (default) head-first entry the body is head-down with arms extended *beyond
  the head*, not trailing.
- s7 "**feet are just above the water** ... preparing for entry" — **minor**: implies
  feet-first entry; most competitive dives enter head-first (and this contradicts s8's
  streamlined rip entry framing).
- s1 "stands at the edge of a **swimming pool**" vs s2 "leaps from the **diving
  board**" — cosmetic inconsistency (divers start on the board/platform, not the pool
  edge).
- s9 "arms flare outward as they submerge" — cosmetic (rip entry keeps arms locked).

### FrisbeeCatch
- s0 (hierarchy) "**with the body airborne**" — **minor** overclaim: many/most catches
  are made standing or on the run; airborne layout catches are the exception. Same
  overclaim propagates to s4 "torso rotates **mid-air**" (**minor**). Note the attrs
  file correctly does *not* assert `aerial_phase` — text and attrs currently disagree.

### GolfSwing
- s2 "backswing ... lifting the club **over the left shoulder**" — **minor**: for the
  (default, right-handed) golfer the backswing goes over the *right* shoulder; the
  *follow-through* finishes over the left. Sides reversed.
- s4 "the **right arm fully extends backward**" at the top — **minor**: arms swapped;
  at the top of a right-handed backswing the *left* arm is straight and the right arm
  folds.
- s5 "left arm bends inward, the **club head near the hip**, preparing for the
  downswing" — **minor**: describes no real checkpoint (at the top the club is above
  the shoulders) and scrambles the order after s3-s4 already reached the top.
- s10 "club head **resting on the ground**" at the finish — cosmetic (finish position
  holds the club over the lead shoulder).

### HammerThrow
- s10 "Hammer lands on the ground, visible as a small, **spinning disc**" — **minor**
  (wrong equipment): a hammer is a *ball on a wire with a handle*, not a disc —
  contamination from ThrowDiscus, exactly the confusion the supervision should be
  separating. Correction: "the ball-and-wire hammer lands in the field".
- s8 "hammer flying forward in a **straight line**" — cosmetic (projectile arc,
  released upward at ~40 deg).
- Correctly has **no run-up** anywhere (matches reality and the attrs) and correctly
  keeps both hands on the handle until simultaneous release.

### HighJump
- s4 "feet and knees **above the bar**" placed *before* s7 "head and shoulders clearing
  the bar" — **minor** (order reversed): in the Fosbury flop the head/shoulders cross
  first and the feet flick up *last*.
- s2 "body **arching over the bar**, arms extended forward" at takeoff — cosmetic
  (flop arches backward, bar-side up; arching also cannot happen at takeoff).
- s9 "landing softly ... into a **tucked position**" — cosmetic (floppers land flat on
  the upper back/shoulders).

### JavelinThrow
- s1 "stands upright, holding javelin **with both hands**" + no run-up anywhere —
  **breaks-transfer**: the javelin is gripped with *one* hand (carried overhead at ear
  level), and the long accelerating run-up with crossover steps is the class's temporal
  signature (attrs correctly say `has_runup`; the phases never show it). Correction:
  "Athlete sprints down the runway, javelin held overhead in one hand, tip pointing
  slightly up."
- s2 "begins to **pivot** ... as the body **rotates** toward the throw" — **minor**:
  javelin technique is linear (withdrawal during crossovers + block); "pivot/rotate"
  is discus/hammer contamination.

### PoleVault
- s7 "pole **begins to bend again**" and s8 "**pole bends sharply**, athlete's body
  descends" — **minor** x2: the pole bends once and recoils straight during the ascent;
  it never re-bends, and by the descent the vaulter has *released* the pole (s8-s9
  keep them attached).
- s10 "lands **feet-first** on mat" — **minor**: vaulters land on the back/shoulders on
  the mat, never feet-first.
- Missing the two most distinctive moments — full body inversion (head-down) and bar
  clearance (the bar is never mentioned) — **minor** (coverage gap).
- s1 "runs forward **with pole planted** in ground" — cosmetic phase-merge (run first,
  then plant into the box).

### Shotput
- s4 "Shotput is lifted high, **arm fully extended overhead**" — **breaks-transfer**:
  as in A1, the shot never goes overhead; it stays at the neck until the final
  up-and-out push. This sentence hard-codes the same error as the wrong attr.
  Correction: "the shot stays tucked against the neck as the hips and chest drive
  around and up".
- s2 "Right arm **extends backward**, shoulder rotated" — **minor**: illegal/wrong —
  the putting arm stays bent with the shot at the neck; taking it behind the shoulder
  line is a foul.
- s5 "Right arm **swings** forward in a powerful **arc**, elbow locked, as the athlete
  releases" — **minor**: a put is a *push* (elbow extends from behind the shot), not a
  locked-elbow swing.
- s1 "held close to the **chest**" — cosmetic (neck/under the jaw).
- s7 "leaves the hand ... **trailing behind** the athlete's body" — cosmetic
  (it travels forward).

### SoccerPenalty
- s0 (hierarchy) "using a **small projectile**" — **minor**: a soccer ball (~22 cm) is
  not a small projectile; this clause was likely copied from the frisbee/ball-throw
  template and blurs the class against genuinely small-object classes.
- Missing run-up — **minor**: s1 stands at the spot and s2 immediately swings; the
  short penalty run-up (attrs: `has_runup`) is skipped. Insert "the player takes a
  short run-up toward the ball".
- s9-s10 "**no other players in frame** / no player in frame" — **minor**: a penalty by
  definition has a goalkeeper on the line (and the keeper is a strong visual cue for
  the class); asserting an empty frame is wrong and anti-discriminative.

### TennisSwing
- s5 "racket head begins to **decelerate** as it approaches the ball" — **minor**:
  backwards; the racket *accelerates* into contact (and this contradicts s3's own
  "accelerates" claim).
- s6 "wrist snaps sharply **backward** ... to prepare for contact" — **minor**: the
  wrist releases *forward/over* the ball (topspin); a backward snap before contact is
  not a real technique.

### ThrowDiscus
- s6 "Discus is swung **upward and backward** in a **horizontal arc**" — **minor**:
  self-contradictory direction, and it appears *after* s4 already completed the turn
  with the arm swinging forward — a backswing re-inserted mid-delivery. 
- s3 "discus pulled **taut against torso**" — cosmetic (the throwing arm trails long
  and away from the body).
- Notably the prompts never claim a spin direction, so the classic
  clockwise/counterclockwise trap is avoided; turn initiation off the left foot (s2)
  is correct for the default right-handed thrower.

### VolleyballSpiking
- s2 "The player's **fingers grip the ball** as it hovers mid-air above the net" —
  **breaks-transfer**: a spiker *never grips* the ball (that is a held-ball violation);
  the ball is struck with an open palm. "Gripping a hovering ball" describes a catch,
  not a spike — a categorically wrong visual for the class. Correction: "the set ball
  hangs above the net as the player's hitting arm cocks back".
- s5 "elbow **bends sharply** as the forearm accelerates toward the ball" — **minor**:
  backwards; the elbow *extends* from the cocked position through contact.
- s8 "the player's body **continues upward** ... after the strike" — **minor**: contact
  happens at the jump apex; after the strike the player descends.
- Missing the 3-4 step approach run (attrs: `has_runup`) — **minor**: s1 starts at the
  leap.

---

## 3. Family map + arith parents

`thumos14_fam_map.json`, the "a kind of ..." clauses in `thumos14_arith.json`, and the
7-parent `thumos14_nest_map.json` are mutually consistent (verified programmatically:
arith "full" == ensemble sentence 0 for all 20 classes; nest_map's "other" bucket =
exactly the three singleton families).

Family-by-family verdict — **no wrong groupings found**:
- **ball throw** {BaseballPitch, CricketBowling} — correct siblings (hand-projected ball, overhead arm).
- **court jump strike** {BasketballDunk, VolleyballSpiking} — correct (indoor court, jump, downward strike at a high target).
- **implement strike** {Billiards, CricketShot, GolfSwing, TennisSwing} — semantically correct (hand-held tool strikes a ball). *Cosmetic note*: Billiards is visually the odd one out (indoor table, bent posture, no swing), but `env_table` + "bent posture" already carry that separation.
- **aquatic dive** {CliffDiving, Diving} — correct.
- **field throw** {HammerThrow, JavelinThrow, Shotput, ThrowDiscus} — correct (the four athletics throws).
- **jump field** {HighJump, LongJump, PoleVault} — correct members. *Cosmetic note*: the name "jump field" is garbled English ("field jump" / "athletics jump" would read better in prompts, e.g. "a kind of jump field" is ungrammatical); left as-is since renaming changes every prompt string.
- Singletons {CleanAndJerk, FrisbeeCatch, SoccerPenalty} — correctly isolated; no plausible sibling exists in the 20-class set. (`thumos14_parent_map.json` calls CleanAndJerk "weight lift" vs fam_map's "clean and jerk" — both singleton, no functional difference.)

---

## 4. v2 change list (applied to `configs/thumos14_ttb_attrs_v2.json`)

1. **Shotput**: `force_overhead` -> `force_horizontal` (A1, breaks-transfer).
2. **CleanAndJerk**: removed `env_court` (A2, breaks-transfer); class now has no env attribute by design.
3. **VolleyballSpiking**: added `motion_jump_launch`; replaced `impl_none` with `impl_ball` (A3/A4, minor).
4. **TennisSwing**: added `rotational_windup` (A5, minor).

All other 16 classes verified correct and left unchanged. If the ensemble sentences are
regenerated, prioritize the 5 breaks-transfer sentence fixes (CleanAndJerk s1,
Diving s6, Shotput s4, VolleyballSpiking s2, JavelinThrow s1) plus the SoccerPenalty
"small projectile" hierarchy clause, since sentence 0 feeds both the ensemble and
`arith.json`.
