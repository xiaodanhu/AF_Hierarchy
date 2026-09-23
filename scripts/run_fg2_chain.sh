#!/bin/bash
# FineGym l_d=2 (PHRASE-level detection) program, 2026-09-22 (reviewer item 1:
# two-level sub-actions in the VIDEO branch; user decisions: held-out phrases
# FX_turns/FX_front_salto/BB_dismounts/UB_flight_same_bar, separate ASA
# parameters per level, held-out segments = background in training, all four
# GPUs on FineGym). 2026-09-23: held_out_mode exclude (unseen spans leave every
# loss) + the ancestor loss over SEEN member columns only (meta_archs); TAG=fg2b
# is the corrected rerun (SKIP_TIFAD=1: the fg2 Ti-FAD baseline is unaffected).
#
# Phases (markers in ckpt/fg2_*_done.marker; each phase retries until its marker):
#   1  champ1   stage 1 champion  configs/finegym_t2_champ.yaml           GPUs 0-3, mb12
#   2  champ2   stage 2 (text neighbour attention)                        GPUs 0-3, mb12
#      2i       init-equality of stage 2 (flag on, zero-init, eval only == stage-1 best)
#   3  cache    frozen CLIP features of the stage-2 model                 GPU 3
#   4a tifad    Ti-FAD re-implementation at l_d=2 (finegym_t2_tifad.yaml) GPUs 0-2, mb16
#   4b (GPU 3, sequential) inert check -> init-equality nested -> train nested
#                          -> init-equality flat -> train flat
# Logs: logs/fg2_<phase>.log, chain log logs/fg2_chain.log.
# Result summary: bash scripts/run_fg2_chain.sh table   (TAG=<tag> selects a rerun, default fg2)
cd /data3/xiaodan8/actionformer6
PY=/data/xiaodan8/anaconda3/envs/parsing2/bin/python
QF=/data3/xiaodan8/tifad/queue/jobs.txt
PAUSE=/data3/xiaodan8/tifad/queue/PAUSE
TAG=${TAG:-fg2}   # run tag: markers ckpt/${TAG}_*_done.marker, outputs ${TAG}_*, logs logs/${TAG}_*.log
LOG=logs/${TAG}_chain.log
CACHE=/data3/xiaodan8/FineGym/cache_${TAG}_champ2
CK1=ckpt/finegym_t2_champ_${TAG}_champ1/vit_best_model
CK2=ckpt/finegym_t2_champ_stage2_${TAG}_champ2/vit_best_model
MIN_FREE=${MIN_FREE:-13000}
mkdir -p logs ckpt
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_gpus() {   # wait_gpus "0 1 2 3": block until each listed GPU has >= MIN_FREE MB free
  while :; do
    ok=1
    for g in $1; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      [ "$f" -ge "$MIN_FREE" ] || ok=0
    done
    [ "$ok" -eq 1 ] && return
    sleep 120
  done
}
ds() {   # ds <gpus,csv> <args...>: deepspeed launcher on those GPUs
  local gpus=$1; shift
  CUDA_VISIBLE_DEVICES=$gpus deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py "$@"
}
run_phase() {   # run_phase <name> <gpus,csv> <args...>: retry until the marker exists
  local name=$1 gpus=$2; shift 2
  until [ -f "ckpt/${TAG}_${name}_done.marker" ]; do
    wait_gpus "${gpus//,/ }"
    log "=== $name starting on GPUs $gpus ==="
    ds "$gpus" "$@" > "logs/${TAG}_${name}.log" 2>&1
    local rc=$?
    log "=== $name rc=$rc ==="
    grep -h "\[ZSL video\]" "logs/${TAG}_${name}.log" | tail -2 | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then touch "ckpt/${TAG}_${name}_done.marker"; else sleep 600; fi
  done
}
grep_best() { grep -h "\[select\]\|Best model saved\|\[ZSL video\]" "$1" | tail -4; }

case "${1:-run}" in
table)
  echo "=== FineGym l_d=2 (phrase detection; 4 unseen / 10 seen) ==="
  for p in champ1 champ2 champ2_init inert nested_init nested flat_init flat tifad; do
    f=logs/${TAG:-fg2}_$p.log; [ -f "$f" ] || continue
    printf "%-12s " "$p"; grep -h "\[ZSL video\]" "$f" | tail -2 | tr '\n' ' '; echo
  done
  echo "--- best epochs (selection metric) ---"
  for p in champ1 champ2 nested flat tifad; do f=logs/${TAG:-fg2}_$p.log; [ -f "$f" ] && { echo "-- $p"; grep -h "\[select\]\|Best model saved" "$f" | tail -2; }; done
  exit 0 ;;
run) ;;
*) echo "usage: $0 [run|table]"; exit 1 ;;
esac

touch "$PAUSE"   # keep the tifad dispatcher off the GPUs while this program runs
log "=== FG2 chain starting (PAUSE held) ==="

# ---- 1. stage 1 champion, 4 GPUs ----
run_phase champ1 0,1,2,3 --config configs/finegym_t2_champ.yaml --output ${TAG}_champ1

# ---- 2i. stage-2 init-equality (zero-init neighbour attention == stage-1 best), eval only ----
if [ ! -f ckpt/${TAG}_champ2_init_done.marker ]; then
  wait_gpus "0 1 2 3"
  log "=== champ2_init (eval only; must reproduce the stage-1 best [ZSL video] numbers) ==="
  ds 0,1,2,3 --config configs/finegym_t2_champ_stage2.yaml --output ${TAG}_champ2_init \
    --init_from $CK1 --freeze_except text_pathway.phase_attn --eval_only > logs/${TAG}_champ2_init.log 2>&1
  log "champ2_init rc=$?"; grep -h "\[ZSL video\]" logs/${TAG}_champ2_init.log | tee -a "$LOG"
  grep -h "\[ZSL video\]" logs/${TAG}_champ1.log | tail -2 | sed 's/^/stage-1 best (for comparison; last eval lines): /' | tee -a "$LOG"
  touch ckpt/${TAG}_champ2_init_done.marker
fi

# ---- 2. stage 2 (neighbour attention at both text levels), 4 GPUs ----
run_phase champ2 0,1,2,3 --config configs/finegym_t2_champ_stage2.yaml --output ${TAG}_champ2 \
  --init_from $CK1 --freeze_except text_pathway.phase_attn

# ---- 3. cache frozen CLIP features of the stage-2 model (GPU 3) ----
if [ ! -f "$CACHE/meta.json" ]; then
  wait_gpus "3"
  log "=== cache -> $CACHE (logs/${TAG}_cache.log) ==="
  CUDA_VISIBLE_DEVICES=3 $PY scripts/cache_fg3_clip_feats.py \
    --config configs/finegym_t2_champ_stage3_nested.yaml --ckpt $CK2 --out $CACHE \
    --verify 8 --workers 24 --batch 8 > logs/${TAG}_cache.log 2>&1
  rc=$?; log "cache rc=$rc"; grep -h "\[verify\|done:" logs/${TAG}_cache.log | tail -3 | tee -a "$LOG"
  [ $rc -eq 0 ] || { log "caching failed; abort"; rm -f "$PAUSE"; exit 1; }
fi

# ---- 4a. Ti-FAD baseline at l_d=2 on GPUs 0,1,2 (background, same recipe as l_d=3) ----
# SKIP_TIFAD=1 for reruns whose fix does not touch the baseline (e.g. TAG=fg2b,
# the seen-only ancestor loss): the fg2 Ti-FAD result stays valid.
if [ "${SKIP_TIFAD:-0}" = 1 ]; then
  log "=== tifad skipped (SKIP_TIFAD=1) ==="; TIFAD_PID=
else
  ( run_phase tifad 0,1,2 --config configs/finegym_t2_tifad.yaml --output ${TAG}_tifad ) &
  TIFAD_PID=$!
fi

# ---- 4b. video branch on GPU 3: inert -> nested (init, train) -> flat (init, train) ----
if [ ! -f ckpt/${TAG}_inert_done.marker ]; then
  wait_gpus "3"
  log "=== inert check (stage-2 config, flag off, cached features, eval only == stage-2 best) ==="
  ds 3 --config configs/finegym_t2_champ_stage2.yaml --output ${TAG}_inert \
    --init_from $CK2 --freeze_except text_pathway.phase_attn --feature_cache $CACHE \
    --eval_only > logs/${TAG}_inert.log 2>&1
  log "inert rc=$?"; grep -h "\[ZSL video\]" logs/${TAG}_inert.log | tee -a "$LOG"
  touch ckpt/${TAG}_inert_done.marker
fi
for arm in nested flat; do
  if [ ! -f ckpt/${TAG}_${arm}_init_done.marker ]; then
    wait_gpus "3"
    log "=== ${arm}_init (zero-init video branch, eval only == inert numbers) ==="
    ds 3 --config configs/finegym_t2_champ_stage3_${arm}.yaml --output ${TAG}_${arm}_init \
      --init_from $CK2 --freeze_except video_subtree_attn --feature_cache $CACHE \
      --eval_only > logs/${TAG}_${arm}_init.log 2>&1
    log "${arm}_init rc=$?"; grep -h "\[ZSL video\]" logs/${TAG}_${arm}_init.log | tee -a "$LOG"
    touch ckpt/${TAG}_${arm}_init_done.marker
  fi
  run_phase $arm 3 --config configs/finegym_t2_champ_stage3_${arm}.yaml --output ${TAG}_${arm} \
    --init_from $CK2 --freeze_except video_subtree_attn --feature_cache $CACHE \
    --select_metric video_held_mAP -c 1
  grep_best logs/${TAG}_${arm}.log | tee -a "$LOG"
done

[ -n "$TIFAD_PID" ] && wait $TIFAD_PID
rm -f "$PAUSE"
log "=== FG2 CHAIN COMPLETE ==="
bash scripts/run_fg2_chain.sh table | tee -a "$LOG"
