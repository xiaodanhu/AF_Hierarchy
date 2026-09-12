# FineGym zero-shot detection: Ti-FAD re-implementation and the 4-level hierarchy champion

> **This branch:** the Ti-FAD re-implementation — run `bash scripts/run_finegym_t3.sh tifad`.

This branch contains everything needed to train and evaluate the FineGym
comparison (Table 3) between the Ti-FAD baseline and our 4-level hierarchy
champion. Both systems share the `ActionFormerWithCLIP` backbone (CLIP
ViT-B/32 image encoder trained end-to-end on raw frames +
convolutional-transformer detector) and a frozen CLIP text encoder with one
learned projection; they differ only in how text enters the model. (A
name-embedding-only config, `finegym_t3_base.yaml`, is kept for reference
but is not part of the comparison.)

| system | config | what it adds |
|---|---|---|
| Ti-FAD (baseline) | `configs/finegym_t3_tifad.yaml` | one prompt per element name + per-level text–video cross-attention + foreground head; none of our components |
| ours, 4-level champion | `configs/finegym_t3_champ.yaml` | + 11-sentence hierarchical prompt ensemble, action–phase attention, auxiliary losses at the apparatus and element-set levels (3-way learned uncertainty weighting), duration prior |

Hierarchy used by the champion: apparatus (4) > element set (14) > element (99,
the actions the detector predicts) > phases (10 per element, generated).

## Protocol

* 20 of the 99 elements are held out as unseen actions
  (`held_out_classes` in `configs/finegym_attribute_table_v2.json`; injected
  by `train_shard.py`). The remaining 79 are seen.
* Leakage-aware: segments of held-out elements keep their apparatus and
  element-set labels during training (parent-level supervision) but
  contribute no element-level classification loss.
* Metric: mAP on the 20 unseen elements at tIoU 0.3–0.7 (`[ZSL video]
  held-out-class mAP` in the log); seen-element mAP is printed alongside.

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
```

The metadata caches are built automatically from the raw videos when present;
without raw videos, `scripts/gen_finegym_metadata_cache.py` reconstructs them
from the annotation database.

## Prompts and assets (all in `configs/`)

* `finegym_qwen_ensemble.json` — the 11 sentences per element used by the
  champion (sentence 0: element + element set + apparatus; sentences 1–10:
  phases in temporal order).
* `finegym_zsl_verbalizer.json` — element / element-set / apparatus names and
  the tree.
* `finegym_attribute_table_v2.json` — per-element attributes and the
  held-out list.
* `finegym_t3_duration_prior.json` — seen-element log-duration statistics
  (`scripts/gen_finegym_duration_prior.py` regenerates it).
* `deepspeed_config_bf16_mb16.json` — bf16, ZeRO-2, micro-batch 16, no
  gradient accumulation.

CLIP text embeddings are cached under `./cache_text_emb/` on first use.

## Training

Schedule (identical for both systems): 20 epochs = 5 warm-up + 15 cosine,
AdamW, lr 1e-4, weight decay 0.05, 16 windows per GPU on 3 GPUs (48 per
step), bf16, gradient checkpointing on the CLIP encoder. About 11 GB per GPU.

```bash
bash scripts/run_finegym_t3.sh tifad          # Ti-FAD re-implementation
bash scripts/run_finegym_t3.sh champ          # 4-level champion
```

Logs go to `logs/fg3_<system>.log`; checkpoints to
`ckpt/finegym_t3_<system>_fg3_<system>/`. The best epoch is selected by the
held-out mAP printed each epoch.

`python scripts/verify_t3_systems.py` runs a CPU-only check of the
systems (asset resolution, one train/eval forward each, and the init-equality
test that the champion equals the baseline at initialization).
