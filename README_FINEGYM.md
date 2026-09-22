# FineGym zero-shot detection: Ti-FAD re-implementation and the 4-level hierarchy champion

> **This branch:** the 4-level hierarchy champion with the action–sub-action
> attention (ASA) in **both** the text and the video branch, at two detection
> levels.
> * `l_d=3` (element level, 20 unseen / 79 seen): `bash scripts/run_finegym_t3.sh champ`
>   → `champ2` → `champ3` (or `bash scripts/run_fg3_champ3_vsa3gpu.sh` for stage 3).
> * `l_d=2` (element-set level, 4 unseen / 10 seen, two levels of sub-actions in
>   both branches): `nohup bash scripts/run_fg2_chain.sh > logs/fg2_chain.out &`
>   (the whole program: three champion stages, the flat-video ablation and the
>   Ti-FAD baseline); `bash scripts/run_fg2_chain.sh table` prints the results.

Both systems share the `ActionFormerWithCLIP` backbone (CLIP ViT-B/32 image
encoder trained end-to-end on raw frames + convolutional-transformer detector)
and a frozen CLIP text encoder with one learned projection; they differ only in
how text enters the model and in the ASA blocks. (A name-embedding-only config,
`finegym_t3_base.yaml`, is kept for reference but is not part of the comparison.)

Hierarchy: apparatus (4) > element set, "phrase" (14) > element, "action" (99)
> temporal phases (10 per element, generated). The detection level `l_d` is
the level whose instances the detector localises; every level below it is
composed by ASA without any boundary annotation, every level above it is
supervised through the ancestor loss.

| system | configs | what it adds |
|---|---|---|
| Ti-FAD (baseline) | `finegym_t3_tifad.yaml` (`l_d=3`), `finegym_t2_tifad.yaml` (`l_d=2`) | one prompt per class name + per-level text–video cross-attention + foreground head; none of our components |
| ours, `l_d=3` | `finegym_t3_champ.yaml` (stage 1), `finegym_t3_champ_stage2.yaml` (stage 2), `finegym_t3_champ_stage3_vsa.yaml` (stage 3) | 11-sentence hierarchical prompt, text ASA (element ← 10 phases), video ASA (proposed segment → 10 equal chunks), ancestor losses at the element-set and apparatus levels with learned uncertainty weights, duration prior |
| ours, `l_d=2` | `finegym_t2_champ.yaml`, `finegym_t2_champ_stage2.yaml`, `finegym_t2_champ_stage3_nested.yaml` (+ `_flat.yaml` ablation) | the same, with **two levels of sub-actions in both branches**: text = phrase ← member elements ← phases; video = proposed segment → K=3 element chunks → K'=10 phase chunks each; ancestor loss at the apparatus level |

## Model components (where to look)

* `libs/modeling/text_pathway.py` — `TextClsPathway`: frozen CLIP sentence
  embeddings, the projection, and the text ASA. Level-3 block: `attr_attn`
  (element node attends over [node; phases]) and `phase_attn` (banded
  neighbour attention among the 10 phases). `nested=True` adds the level-2
  block `attr_attn2` / `phase_attn2` (phrase node over its member elements,
  members refined by neighbour attention) with **separate parameters per
  level**. Stem degradation replaces the whole prompt by the immediate
  ancestor's name (element-set name at `l_d=3`, apparatus name at `l_d=2`).
* `libs/modeling/vsa.py` — the video ASA. `VideoSubtreeAttn` (one level): the
  regression head's detached span at every location is cut into K equal
  chunks, mean-pooled with integral features, refined by banded neighbour
  attention, then the location feature attends over [itself; chunks]; the
  result replaces the feature the cosine classifier reads. `VideoSubtreeAttnNested`
  (two levels, `video_subtree_nested: True`): K element chunks × K' phase
  leaves, leaves → element chunk, then element chunks → location, with four
  distinct attention modules. All output projections are zero-initialised, so
  a stage starts byte-identical to the previous one.
* `libs/modeling/meta_archs.py` — wiring, the ancestor loss (`activity_align_levels`:
  log-sum-exp over member columns, one Kendall uncertainty weight per level),
  the held-out masking of the classification loss, the duration prior at
  inference.
* `libs/datasets/finegym_slide.py` — sliding-window dataset. `detection_level:
  action | phrase` selects the detection level (phrase instances = contiguous
  runs of same-set elements inside a routine). `held_out_mode: keep |
  background` (see Protocol).

## Protocol

* `l_d=3`: 20 of the 99 elements are held out (`held_out_classes` in
  `configs/finegym_attribute_table_v2.json`); `l_d=2`: 4 of the 14 element
  sets — FX turns, FX salto forward, BB dismounts, UB flight on the same bar
  (`configs/finegym_phrase_attribute_table.json`). `train_shard.py` injects
  the ids into the dataset.
* Held-out segments in training videos:
  * `held_out_mode: background` (the `l_d=2` configs; the THUMOS14/ActivityNet
    convention): held-out segments are **removed from the training
    annotation** — they are background for the classification, boundary and
    ancestor losses; videos and windows are kept. Ancestor labels therefore
    cover seen video only.
  * `held_out_mode: keep` (the original `l_d=3` configs): held-out segments
    keep their ancestor labels (ancestor-level supervision) and the
    class-agnostic boundary supervision, and contribute no element-level
    classification loss. Set `held_out_mode: background` in the `t3` configs
    to run `l_d=3` under the stricter protocol.
* Metric: mAP on the unseen classes at tIoU 0.3–0.7 (`[ZSL video]
  held-out-class mAP` in the log); seen-class mAP is printed alongside.

## Data layout

`libs/datasets/finegym_slide.py` reads FineGym from `/data3/xiaodan8/FineGym`
(`data_root` near the top of `FineGymSlideDataset.__init__`; edit if your
copy lives elsewhere):

```
FineGym/
  annotation/Dec16/gym99_train_label.txt      # paths also set in the configs
  annotation/Dec16/gym99_val_label.txt
  RGB/<youtube_id>/<7-digit frame index>.jpg  # cached JPG frames
  video_metadata_cache_train_raw.json         # {youtube_id: {fps, total_frames, duration}}
  video_metadata_cache_val_raw.json
  cache_fg3_champ2/, cache_fg2_champ2/        # frozen CLIP features for stage 3 (built by the scripts)
```

The metadata caches are built automatically from the raw videos when present;
without raw videos, `scripts/gen_finegym_metadata_cache.py` reconstructs them
from the annotation database.

## Prompts and assets (all in `configs/`)

* `finegym_qwen_ensemble.json` — the 11 sentences per element (sentence 0:
  element + element set + apparatus; sentences 1–10: phases in temporal order).
* `finegym_phrase_ensemble.json` — 11 sentences per element set; the nested
  text branch uses sentence 0 as the phrase node and the member elements'
  ensembles as its children.
* `finegym_zsl_verbalizer.json` — element / element-set / apparatus names and
  the tree; `finegym_phrase_verbalizer.json` — the 14 element sets as detection
  classes with their `members`; `finegym_phrase_verbalizer_named.json` — the
  same with apparatus-qualified names for the Ti-FAD name-only prompt (FineGym
  set names collide across apparatus: "dismount", "turn", "leap, jump or hop").
* `finegym_attribute_table_v2.json`, `finegym_phrase_attribute_table.json` —
  per-class attributes (set attributes = union over members) and the held-out
  lists.
* `finegym_t3_duration_prior.json`, `finegym_t2_duration_prior.json` — seen-class
  log-duration statistics and the attribute-transferred priors of the unseen
  classes (`scripts/gen_finegym_duration_prior.py [--level phrase]`).
* `deepspeed_config_bf16_mb16.json` (3 GPUs × 16), `_mb12.json` (4 GPUs × 12),
  `_mb24_acc2.json` (1 GPU, stage 3) — bf16, ZeRO-2, effective batch 48 in all
  cases.

CLIP text embeddings are cached under `./cache_text_emb/` on first use.

## Training

Stage 1 (both systems): 20 epochs = 5 warm-up + 15 cosine, AdamW, lr 1e-4,
weight decay 0.05, effective batch 48, bf16, gradient checkpointing on the
CLIP encoder (about 11 GB per GPU at 16 windows per GPU).

Champion stages (paper Appendix "Training schedule"):

1. **Stage 1** — detector, CLIP image encoder, text projection and the node
   cross-attentions (`attr_attn`, and `attr_attn2` at `l_d=2`), ancestor
   losses, duration prior.
2. **Stage 2** — the banded neighbour attentions of the text ASA only
   (`--init_from <stage-1 best> --freeze_except text_pathway.phase_attn`,
   which matches `phase_attn` and `phase_attn2`), 8 epochs (2 warm-up + 6).
   Zero-initialised output projections: stage 2 starts exactly at stage 1
   (`scripts/verify_t3_stage2.py`, `scripts/tests/test_text_nested.py`).
3. **Stage 3** — the video ASA only (`--freeze_except video_subtree_attn`),
   15 epochs (2 warm-up + 13), on the cached CLIP-encoder features of the
   stage-2 model (`scripts/cache_fg3_clip_feats.py`, bitwise-verified against
   the frame path). The chain scripts first run an *inert* check (stage-2
   model on cached features reproduces the stage-2 numbers) and an
   *init-equality* check (zero-initialised video branch reproduces them too).
   Best epoch by unseen video mAP (`--select_metric video_held_mAP`).

```bash
# l_d=3
bash scripts/run_finegym_t3.sh tifad             # Ti-FAD re-implementation
bash scripts/run_finegym_t3.sh champ             # stage 1
bash scripts/run_finegym_t3.sh champ2            # stage 2
bash scripts/run_fg3_champ3_vsa3gpu.sh           # cache + checks + stage 3
# l_d=2 (everything, with GPU gating and retries)
nohup bash scripts/run_fg2_chain.sh > logs/fg2_chain.out 2>&1 &
bash scripts/run_fg2_chain.sh table
```

Logs go to `logs/fg3_*.log` / `logs/fg2_*.log`; checkpoints to
`ckpt/<config>_<output>/`.

## Tests

* `python scripts/verify_t3_systems.py`, `python scripts/verify_t3_stage2.py`
  — CPU checks of the `l_d=3` systems (asset resolution, one train/eval
  forward each, init-equality).
* `python scripts/tests/test_text_nested.py` — nested text branch (flat path
  unchanged bitwise, init-equality, gradients, freeze substrings).
* `python scripts/tests/test_vsa_nested.py --gpu 0` — nested video branch
  (init-equality in fp32/bf16, index safety at the pyramid boundary, gradients,
  memory).
* `bash scripts/tests/smoke_fg2.sh [gpu]` — every `l_d=2` config for two
  batches on one GPU, chained through `--init_from`/`--freeze_except` like the
  real program (`FG_SMOKE_ITERS` caps the train/eval loops; unset in real runs).
