#!/bin/bash
# FineDiving v3 chain (split s1, keep protocol). Recipe v2 (mb8 x 4 = 32, lr 8e-5, 5+30 ep, no text dropout / stem
# degrade) + LEARNABLE shared sub-action deltas (compositional ZSL: CLIP text cannot separate somersault counts).
#   champ3 -> cache3 -> nested3 (video ASA K3xKp5) -> champ1lvl3 -> cache1lvl3 -> flat1lvl3 -> tifad_as (control) -> ref (learned head, seen ceiling)
# Markers ckpt/fd3_<phase>_done.marker; log logs/fd_chain.log (shared with the v1 chain for the monitor).
cd /home/jupyter/xhu3/video/Dec14/af_fg3_v3 || exit 1
export PATH=/home/jupyter/xhu3/.conda/envs/qwen/bin:$PATH
PY=/home/jupyter/xhu3/.conda/envs/qwen/bin/python
LOG=logs/fd_chain.log; C=configs/finediving/v3
mkdir -p logs ckpt
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ds() { local gpus=$1; shift; CUDA_VISIBLE_DEVICES=$gpus deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py "$@"; }
run_phase() {   # run_phase <name> <gpus> <args...>
  local name=$1 gpus=$2; shift 2
  until [ -f "ckpt/fd3_${name}_done.marker" ]; do
    local cfgstem=$(basename "$(echo "$@" | sed -n 's/.*--config \([^ ]*\).*/\1/p')" .yaml)
    local outname=$(echo "$@" | sed -n 's/.*--output \([^ ]*\).*/\1/p')
    local resume=""; [ -f "ckpt/${cfgstem}_${outname}/latest" ] && resume="--resume $(cat ckpt/${cfgstem}_${outname}/latest)"
    [ -n "$resume" ] && log "$name: resuming with $resume"
    log "=== $name starting on GPUs $gpus ==="
    ds "$gpus" "$@" $resume > "logs/fd3_${name}.log" 2>&1; local rc=$?
    log "=== $name rc=$rc ==="; grep -h "\[ZSL video\]  \(seen\|held-out\)" "logs/fd3_${name}.log" | tail -2 | tee -a "$LOG"
    ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9
    if [ "$rc" -eq 0 ]; then touch "ckpt/fd3_${name}_done.marker"; else
      grep -hE "Error|Traceback" "logs/fd3_${name}.log" | tail -2 | tee -a "$LOG"; sleep 600; fi
  done
}
cache_phase() {  # cache_phase <name> <config> <ckpt> <out>
  local name=$1 cfg=$2 ck=$3 out=$4
  [ -f "ckpt/fd3_${name}_done.marker" ] && return
  log "=== $name -> $out (from $ck, GPU 3) ==="
  CUDA_VISIBLE_DEVICES=3 $PY scripts/cache_fg3_clip_feats.py --config $cfg --ckpt $ck --out $out --verify 8 --workers 12 --batch 4 > logs/fd3_${name}.log 2>&1
  local rc=$?; log "$name rc=$rc"; grep -hE "\[verify .* worst|Traceback|Error" logs/fd3_${name}.log | tail -2 | tee -a "$LOG"
  [ "$rc" -eq 0 ] && [ -f "$out/meta.json" ] && touch "ckpt/fd3_${name}_done.marker"
}
log "=== FD v3 chain starting (split s1, keep, recipe v2 + learnable sub-action deltas) ==="
run_phase champ3 0,1,2,3 --config $C/fd_champ_s1.yaml --output fd3_s1_champ
CK=ckpt/fd_champ_s1_fd3_s1_champ/vit_best_model; CACHE=/home/jupyter/xhu3/video/dataset/finediving/cache_fd3_s1
cache_phase cache3 $C/fd_champ_stage3_nested_s1.yaml $CK $CACHE
[ -f ckpt/fd3_cache3_done.marker ] && run_phase nested3 0,1,2,3 --config $C/fd_champ_stage3_nested_s1.yaml --output fd3_s1_nested --init_from $CK --freeze_except video_subtree_attn --feature_cache $CACHE --select_metric video_held_mAP -c 1 || log "cache3 failed; nested3 skipped"
run_phase champ1lvl3 0,1,2,3 --config $C/fd_champ1lvl_s1.yaml --output fd3_s1_champ1lvl
CK1=ckpt/fd_champ1lvl_s1_fd3_s1_champ1lvl/vit_best_model; CACHE1=/home/jupyter/xhu3/video/dataset/finediving/cache_fd3_s1_1lvl
cache_phase cache1lvl3 $C/fd_champ1lvl_stage3_flat_s1.yaml $CK1 $CACHE1
[ -f ckpt/fd3_cache1lvl3_done.marker ] && run_phase flat1lvl3 0,1,2,3 --config $C/fd_champ1lvl_stage3_flat_s1.yaml --output fd3_s1_flat1lvl --init_from $CK1 --freeze_except video_subtree_attn --feature_cache $CACHE1 --select_metric video_held_mAP -c 1 || log "cache1lvl3 failed; flat1lvl3 skipped"
run_phase tifad_as3 0,1,2,3 --config $C/fd_tifad_as_s1.yaml --output fd3_s1_tifad_as
run_phase ref3 0,1,2,3 --config $C/fd_ref_learnedhead_s1.yaml --output fd3_s1_ref
log "=== FD V3 CHAIN COMPLETE ==="
for n in champ3 nested3 champ1lvl3 flat1lvl3 tifad_as3 ref3; do echo "$n: $(grep -h '\[ZSL video\]  \(seen\|held-out\)' logs/fd3_${n}.log 2>/dev/null | tail -2 | tr '\n' ' ')" | tee -a "$LOG"; done
