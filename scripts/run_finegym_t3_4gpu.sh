#!/bin/bash
# 4-GPU launcher (local). Usage: bash scripts/run_finegym_t3_4gpu.sh <base|tifad|champ|champ2|tifad_p|champ_p> [port=29710] [resume_tag]
# champ2 = stage 2: phase-level neighbour attention, init from best stage-1 ckpt, everything else frozen.
set -u
sys=${1:?usage: run_finegym_t3_4gpu.sh <base|tifad|champ|champ2> [port]}
port=${2:-29710}
resume=${3:-}          # optional DeepSpeed ckpt tag, e.g. epoch_001
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs ckpt
export PATH=/home/jupyter/xhu3/.conda/envs/qwen/bin:$PATH
# 2026-09-13: lossless 256-short-side PNG frame store (scripts/build_rgb256_store.py); bit-identical
# model input, ~20x cheaper decode. Used only once the build finished (MANIFEST present).
STORE=/home/jupyter/xhu3/video/dataset/finegym/RGB256png
if [ -f "$STORE/MANIFEST.json" ]; then export FINEGYM_RGB_ROOT=$STORE FINEGYM_FRAME_EXT=.png; fi
cfg=$sys; extra=""; cfgfile=""
if [ "$sys" = champ2 ]; then
  cfg=champ_stage2
  extra="--init_from ckpt/finegym_t3_champ_fg3_champ/vit_best_model --freeze_except text_pathway.phase_attn"
fi
# 2026-09-14 phrase-level (element-set) protocol: tifad_p / champ_p use configs/finegym_t3p_*.yaml
case "$sys" in
  tifad_p) cfgfile=configs/finegym_t3p_tifad.yaml ;;
  champ_p) cfgfile=configs/finegym_t3p_champ.yaml ;;
  *)       cfgfile="configs/finegym_t3_${cfg}.yaml" ;;
esac
echo "[$(date)] START fg3_${sys} on GPUs 0-3" >> logs/fg3_chain.log
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port="$port" train_shard.py \
  --config "$cfgfile" --output "fg3_${sys}" $extra ${resume:+--resume $resume} \
  > "logs/fg3_${sys}.log" 2>&1
rc=$?
echo "[$(date)] DONE  fg3_${sys} (exit $rc)" >> logs/fg3_chain.log
