#!/bin/bash
# End-to-end smoke test of every l_d=2 config on ONE GPU: 1 epoch of 2
# training batches + a 2-batch eval (FG_SMOKE_ITERS=2, see
# libs/utils/train_utils_deepspeed.py), chained exactly like the real
# program (stage 2 initialised from the stage-1 smoke checkpoint, stage 3
# from stage 2). Checks: dataset builds in phrase mode (14 classes, held-out
# 2/3/9/11 as background), text/video nested modules construct, losses are
# finite, eval reports [ZSL video] seen (10) / held-out (4), init_from reports
# only the expected missing keys, freeze_except selects only the expected
# tensors. Usage: bash scripts/tests/smoke_fg2.sh [gpu]   (default GPU 3)
cd /data3/xiaodan8/actionformer6
G=${1:-3}
S=/tmp/claude-1002/-data3-xiaodan8-actionformer6/15f23ac7-2198-4c7c-b3a0-fcf5333478bb/scratchpad/smoke_fg2
mkdir -p "$S" logs
mk() {  # mk <cfg name>: 1-epoch copy of the config (warm-up 1) in $S
  sed -e 's/^  epochs: [0-9]*,.*/  epochs: 1,/' -e 's/^  warmup_epochs: [0-9]*,/  warmup_epochs: 1,/' \
      "configs/$1.yaml" > "$S/$1.yaml"
  grep -q "warmup_epochs" "$S/$1.yaml" || sed -i 's/^opt: {/opt: {\n  warmup_epochs: 1,/' "$S/$1.yaml"
}
run() {  # run <tag> <cfg name> <extra args...>
  local tag=$1 cfg=$2; shift 2
  mk "$cfg"
  echo "=== smoke $tag ($cfg) $(date '+%T') ==="
  FG_SMOKE_ITERS=2 CUDA_VISIBLE_DEVICES=$G deepspeed --master_port=$((28000 + RANDOM % 900)) train_shard.py \
    --config "$S/$cfg.yaml" --output smoke_$tag "$@" > "logs/smoke_fg2_$tag.log" 2>&1
  local rc=$?
  echo "rc=$rc"
  grep -h "\[FineGymSlide\]\|\[ZSL\]\|\[t3-\|\[init_from\]\|\[freeze_except\]\|\[ZSL video\]\|Error\|error\|Traceback\|final_loss\|Loss " "logs/smoke_fg2_$tag.log" | grep -v "^\s*$" | head -60
  return $rc
}
run champ1 finegym_t2_champ || exit 1
CK1=ckpt/finegym_t2_champ_smoke_champ1/vit_best_model; [ -d $CK1 ] || CK1=ckpt/finegym_t2_champ_smoke_champ1/epoch_001
run champ2 finegym_t2_champ_stage2 --init_from $CK1 --freeze_except text_pathway.phase_attn || exit 1
CK2=ckpt/finegym_t2_champ_stage2_smoke_champ2/vit_best_model; [ -d $CK2 ] || CK2=ckpt/finegym_t2_champ_stage2_smoke_champ2/epoch_001
run nested finegym_t2_champ_stage3_nested --init_from $CK2 --freeze_except video_subtree_attn --select_metric video_held_mAP -c 1 || exit 1
run flat finegym_t2_champ_stage3_flat --init_from $CK2 --freeze_except video_subtree_attn --select_metric video_held_mAP -c 1 || exit 1
run tifad finegym_t2_tifad || exit 1
echo "=== ALL SMOKE RUNS PASSED $(date '+%T') ==="
