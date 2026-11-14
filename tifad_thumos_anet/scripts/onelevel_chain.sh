#!/bin/bash
# ONE-LEVEL champion chain for a THUMOS14 75/25 split with recipe overrides
# stage 1 (detector + text node attention) ->
# stage 2 (neighbour attention, frozen rest) -> fused eval -> stage 3 (video
# ASA on the frozen detector) -> fused eval. Same envs as the champion
# (run_vsa_final.sh T_TR / T_EV / F_T75) except the overrides.
# Usage: bash scripts/onelevel_chain.sh <tag> <split> [ENS=<ensemble json>] [EXTRA="<envs for every stage>"]
#   tag       : output stem suffix -> ckpt/thumos14_i3d_<tag>{1,2,3}_<n>, logs/<tag>{1,2,3}_s<n>.log
#   ENS       : sentence ensemble (default configs/thumos14_qwen_ensemble.json), e.g. the K=3 / K=5 files
#   EXTRA     : e.g. "TIFAD_CHAIN_FULL=1" (mask ablation) or "TIFAD_SHUFFLE_CHILDREN=train" (order ablation)
#   STAGES    : subset of "1 2 3" to run (default all); stage 2 needs stage 1 (or INIT1=<ckpt stem> to reuse
#               an existing stage-1 family, e.g. INIT1=ckpt/thumos14_i3d_famce_uw_{n}), stage 3 needs stage 2.
cd /data3/xiaodan8/tifad
tag=$1; n=$2; shift 2
ENS=configs/thumos14_qwen_ensemble.json; EXTRA=""; STAGES="1 2 3"; INIT1=""
for kv in "$@"; do declare -g "$kv"; done   # KEY=VALUE args (values may contain spaces when passed as one word)
cfg=configs/thumos14_i3d.yaml; stem=thumos14_i3d
BASE="TIFAD_PROMPT_ENSEMBLE=$ENS TIFAD_ATTR_ATTN=1 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 $EXTRA"
TR="$BASE TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3"
CHAIN="TIFAD_ATTNDESIGN=phasechain"
VSA="TIFAD_VSA=1"
FUSE="TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3"
SMOKE=${TIFAD_MAX_EPOCHS:+TIFAD_MAX_EPOCHS=$TIFAD_MAX_EPOCHS}
done_stage() { [ -f "ckpt/${stem}_$1_$n/mAP_best.pth.tar" ] && grep -q "All done" "logs/$1_s$n.log" 2>/dev/null; }
run() { echo "[$(date '+%F %T')] $tag s$n: $1"; }
s1=${tag}1; s2=${tag}2; s3=${tag}3
[ -n "$INIT1" ] && s1ck="$INIT1/mAP_best.pth.tar" || s1ck="ckpt/${stem}_${s1}_{n}/mAP_best.pth.tar"

if [[ " $STAGES " == *" 1 "* ]] && [ -z "$INIT1" ] && ! done_stage $s1; then run "stage 1"
  env $TR $SMOKE python train_eval.py $cfg --output $s1 --n $n > logs/${s1}_s$n.log 2>&1 || exit 1
fi
if [[ " $STAGES " == *" 2 "* ]] && ! done_stage $s2; then run "stage 2"
  env $TR $CHAIN $SMOKE TIFAD_INIT_FROM=$s1ck TIFAD_FREEZE_EXCEPT=chain_attn \
    python train_eval.py $cfg --output $s2 --n $n > logs/${s2}_s$n.log 2>&1 || exit 1
  env $BASE $CHAIN $FUSE python eval.py $cfg ckpt/${stem}_${s2}_$n --n $n > logs/${s2}e_s$n.log 2>&1 || exit 1
fi
if [[ " $STAGES " == *" 3 "* ]] && ! done_stage $s3; then run "stage 3"
  env $TR $CHAIN $VSA $SMOKE TIFAD_INIT_FROM=ckpt/${stem}_${s2}_{n}/mAP_best.pth.tar TIFAD_FREEZE_EXCEPT=vsa \
    python train_eval.py $cfg --output $s3 --n $n > logs/${s3}_s$n.log 2>&1 || exit 1
  env $BASE $CHAIN $VSA $FUSE python eval.py $cfg ckpt/${stem}_${s3}_$n --n $n > logs/${s3}e_s$n.log 2>&1 || exit 1
fi
run "DONE"
