#!/bin/bash
# Quick queue v2 same 10-epoch (2+8) schedule for comparability, but every run is STOPPED right
# after its epoch-5 eval (the 2nd [ZSL video] pair) has printed and the best checkpoint has been saved.
# Adopts a run that is already training. QUEUE lines: <name> <config> [extra args]; marker ckpt/fg2_quick_<name>_done.marker.
cd /home/jupyter/xhu3/video/Dec14/af_fg3_v3 || exit 1
export PATH=/home/jupyter/xhu3/.conda/envs/qwen/bin:$PATH FINEGYM_RGB_ROOT=/home/jupyter/xhu3/video/dataset/finegym/RGB256png FINEGYM_FRAME_EXT=.png
LOG=logs/fg2_chain.log; log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
clean() { pkill -9 -f "[d]eepspeed.launcher.launch" 2>/dev/null; for p in $(pgrep -f "[t]rain_shard.py --local_rank"); do kill -9 $p 2>/dev/null; done
          ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9; }
while :; do
  next=$(while read -r name cfg extra; do [ -z "$name" ] && continue; [ -f "ckpt/fg2_quick_${name}_done.marker" ] || { echo "$name $cfg $extra"; break; }; done < configs/quick/QUEUE.txt)
  [ -z "$next" ] && break
  set -- $next; name=$1; cfg=$2; shift 2; extra="$*"; L=logs/fg2_s1_quick_$name.log
  [ -f "$cfg" ] || { log "quick $name: config $cfg missing, waiting 5 min"; sleep 300; continue; }
  if pgrep -f "train_shard.py --config $cfg" >/dev/null; then log "=== QUICK $name: adopting the running job ==="
  else log "=== QUICK $name starting on GPUs 0,1,2,3 ($cfg $extra); stops after the epoch-5 eval ==="
       CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config "$cfg" --output fg2_s1_quick_$name $extra > "$L" 2>&1 &
       sleep 30; fi
  # wait for 2 video-level evals (epochs 1 and 5) or for the job to end
  while pgrep -f "train_shard.py --config $cfg" >/dev/null && [ "$(grep -c 'ZSL video.*held-out-class mAP' "$L" 2>/dev/null)" -lt 2 ]; do sleep 30; done
  if [ "$(grep -c 'ZSL video.*held-out-class mAP' "$L" 2>/dev/null)" -ge 2 ]; then
    # let the post-eval checkpoint save finish: wait for training to resume (an epoch-5 iteration line) or 4 min
    for i in $(seq 48); do grep -q "Epoch: \[005\]\[000" "$L" && break; pgrep -f "train_shard.py --config $cfg" >/dev/null || break; sleep 5; done
    log "=== QUICK $name: epoch-5 eval reached, stopping ==="; pkill -f "[t]rain_shard.py --config $cfg"; sleep 8; clean
  else log "=== QUICK $name: job ended before the epoch-5 eval (crash?) ==="; clean; echo FAILED > "ckpt/fg2_quick_${name}_done.marker"; fi
  grep -h "\[ZSL video\]" "$L" | tail -2 | tee -a "$LOG"; touch "ckpt/fg2_quick_${name}_done.marker"
done
log "=== QUICK QUEUE EMPTY (chain remains paused; resume with: setsid nohup bash scripts/run_fg2_splits_local.sh &) ==="
