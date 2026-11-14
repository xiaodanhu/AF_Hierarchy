#!/bin/bash
# After bg: run keep + recursive video ASA (stage-3 nested, full 2+13 recipe, eval every epoch)
# and its novel-only eval BEFORE the remaining quick runs; then restore the queue, relaunch the runner and arm the
# final evaluation pass (v8).
cd /home/jupyter/xhu3/video/Dec14/af_fg3_v3 || exit 1
export PATH=/home/jupyter/xhu3/.conda/envs/qwen/bin:$PATH FINEGYM_RGB_ROOT=/home/jupyter/xhu3/video/dataset/finegym/RGB256png FINEGYM_FRAME_EXT=.png
LOG=logs/fg2_chain.log; log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
until [ -f ckpt/fg2_quick_bg_done.marker ] && ! pgrep -f 'run_fg2_quick_queue_v[2][.]sh' >/dev/null; do sleep 30; done
sleep 10; ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9
CACHE=/home/jupyter/xhu3/video/dataset/finegym/cache_fg2_s1_keep
log "=== keep_stage3_nested starting (keep + recursive video ASA, full stage-3 recipe, eval every epoch) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/quick/fg2_s1_keep_stage3_nested.yaml --output fg2_s1_keep_stage3_nested --init_from ckpt/fg2_s1_keep_fg2_s1_quick_keep/epoch_005 --freeze_except video_subtree_attn --feature_cache $CACHE --select_metric video_held_mAP -c 1 > logs/fg2_s1_keep_stage3_nested.log 2>&1
rc=$?; log "keep_stage3_nested rc=$rc"; grep -h "\[ZSL video\]  \(seen\|held-out\)" logs/fg2_s1_keep_stage3_nested.log | tail -4 | tee -a "$LOG"
ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9
D=ckpt/fg2_s1_keep_stage3_nested_fg2_s1_keep_stage3_nested; CK=$D/vit_best_model
if [ -d "$CK" ]; then
  log "=== EVAL keep_stage3_nested FG3_ZSL_NOVEL_ONLY=1 (eval only, best ckpt) ==="
  FG3_ZSL_NOVEL_ONLY=1 CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/quick/fg2_s1_keep_stage3_nested.yaml --output fg2_s1_quick_calib_keep_stage3_nested_NOVEL_ONLY_1 --init_from $CK --feature_cache $CACHE --eval_only > logs/fg2_s1_quick_calib_keep_stage3_nested_NOVEL_ONLY_1.log 2>&1
  grep -h "\[ZSL video\]" logs/fg2_s1_quick_calib_keep_stage3_nested_NOVEL_ONLY_1.log | tail -3 | tee -a "$LOG"
  ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9
else log "keep_stage3_nested: no best ckpt at $CK"; fi
cat configs/quick/QUEUE_rest.txt >> configs/quick/QUEUE.txt
log "=== quick queue restored (partial_fg_distill, partial_fg_frozenclip, bg_oln_gate, partial_ascol); runner relaunched ==="
setsid nohup bash scripts/run_fg2_quick_queue_v2.sh > /dev/null 2>&1 < /dev/null &
sleep 5
setsid nohup bash scripts/run_fg2_quick_calib_v8.sh > /dev/null 2>&1 < /dev/null &
