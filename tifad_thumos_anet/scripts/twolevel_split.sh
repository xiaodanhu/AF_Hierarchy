#!/bin/bash
# TWO-LEVEL SUB-ACTIONS (2026-09-25): one dispatcher job = the full champion
# recipe for ONE cell and ONE split with sub-phases (K'=3 per phase) in BOTH
# branches. Stages (each skipped when its ckpt + "All done!" log exist):
#   1  stage 1  : detector + text ASA node cross-attentions at both text levels
#                 (attr_attn, attr_attn_sub); envs = the champion stage-1 recipe
#                 + TIFAD_SUB_ENSEMBLE
#   2  stage 2  : neighbour attentions at both text levels (chain_attn,
#                 chain_attn_sub), TIFAD_INIT_FROM stage 1, TIFAD_FREEZE_EXCEPT=chain_attn
#      eval 2   : fused eval of stage 2 (= the "text ASA only" two-level model)
#   3  stage 3  : NESTED video ASA (K=10 chunks x K'=3 leaves), init stage 2,
#                 TIFAD_FREEZE_EXCEPT=vsa
#      eval 3   : fused eval of the full two-level model
# Usage: bash scripts/twolevel_split.sh <t75|t50|a75|a50> <split>
# Outputs: ckpt/<cfgstem>_tl{1,2,3}_<n>, logs/tl{1,2,3}_<cell>_s<n>.log, logs/tl{2,3}e_<cell>_s<n>.log
cd /data3/xiaodan8/tifad
cell=$1; n=$2
case $cell in
  t75) cfg=configs/thumos14_i3d.yaml;    stem=thumos14_i3d;    ENS=configs/thumos14_qwen_ensemble.json; SUB=configs/thumos14_subphase_ensemble.json
       BASE="TIFAD_ATTR_ATTN=1 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1"
       FUSE="TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3" ;;
  t50) cfg=configs/thumos14_i3d_50.yaml; stem=thumos14_i3d_50; ENS=configs/thumos14_qwen_ensemble.json; SUB=configs/thumos14_subphase_ensemble.json
       BASE="TIFAD_ATTR_ATTN=1 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1"
       FUSE="TIFAD_FUSE_ALPHA=0.45 TIFAD_DURPRIOR=0.3" ;;
  a75) cfg=configs/anet_i3d_75.yaml;     stem=anet_i3d_75;     ENS=configs/anet_qwen_ensemble.json;     SUB=configs/anet_subphase_ensemble.json
       BASE="TIFAD_ATTR_ATTN=1 TIFAD_ARITH=configs/anet_arith.json"
       FUSE="TIFAD_FUSE_ALPHA=0.15 TIFAD_DURPRIOR=0.15 TIFAD_DURPRIOR_SRC=allseen TIFAD_DURPRIOR_SUBSET=training TIFAD_DURPRIOR_JSON=/data3/xiaodan8/Activitynet/anet1.3_i3d_filtered.json" ;;
  a50) cfg=configs/anet_i3d_50.yaml;     stem=anet_i3d_50;     ENS=configs/anet_qwen_ensemble.json;     SUB=configs/anet_subphase_ensemble.json
       BASE="TIFAD_ATTR_ATTN=1 TIFAD_ARITH=configs/anet_arith.json"
       FUSE="TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.15 TIFAD_DURPRIOR_SRC=allseen TIFAD_DURPRIOR_SUBSET=training TIFAD_DURPRIOR_JSON=/data3/xiaodan8/Activitynet/anet1.3_i3d_filtered.json" ;;
  *) echo "unknown cell $cell"; exit 1 ;;
esac
TXT="TIFAD_PROMPT_ENSEMBLE=$ENS TIFAD_SUB_ENSEMBLE=$SUB $BASE"
TR="$TXT TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3"
CHAIN="TIFAD_ATTNDESIGN=phasechain"
VSA="TIFAD_VSA=1 TIFAD_VSA_NESTED=1 TIFAD_VSA_K=10 TIFAD_VSA_KP=${TL_KP:-3}"
SMOKE=${TIFAD_MAX_EPOCHS:+TIFAD_MAX_EPOCHS=$TIFAD_MAX_EPOCHS}
done_stage() { [ -f "ckpt/${stem}_$1_$n/mAP_best.pth.tar" ] && grep -q "All done" "logs/$1_${cell}_s$n.log" 2>/dev/null; }
run() { echo "[$(date '+%F %T')] $cell s$n: $1"; }

if ! done_stage tl1; then run "stage 1"
  env $TR $SMOKE python train_eval.py $cfg --output tl1 --n $n > logs/tl1_${cell}_s$n.log 2>&1 || exit 1
fi
if ! done_stage tl2; then run "stage 2"
  env $TR $CHAIN $SMOKE TIFAD_INIT_FROM=ckpt/${stem}_tl1_{n}/mAP_best.pth.tar TIFAD_FREEZE_EXCEPT=chain_attn \
    python train_eval.py $cfg --output tl2 --n $n > logs/tl2_${cell}_s$n.log 2>&1 || exit 1
fi
if ! grep -q "Avearge mAP\|Average mAP" logs/tl2e_${cell}_s$n.log 2>/dev/null; then run "eval 2"
  env $TXT $CHAIN $FUSE python eval.py $cfg ckpt/${stem}_tl2_$n --n $n > logs/tl2e_${cell}_s$n.log 2>&1 || exit 1
fi
if ! done_stage tl3; then run "stage 3"
  env $TR $CHAIN $VSA $SMOKE TIFAD_INIT_FROM=ckpt/${stem}_tl2_{n}/mAP_best.pth.tar TIFAD_FREEZE_EXCEPT=vsa \
    python train_eval.py $cfg --output tl3 --n $n > logs/tl3_${cell}_s$n.log 2>&1 || exit 1
fi
if ! grep -q "Avearge mAP\|Average mAP" logs/tl3e_${cell}_s$n.log 2>/dev/null; then run "eval 3"
  env $TXT $CHAIN $VSA $FUSE python eval.py $cfg ckpt/${stem}_tl3_$n --n $n > logs/tl3e_${cell}_s$n.log 2>&1 || exit 1
fi
run "DONE"
