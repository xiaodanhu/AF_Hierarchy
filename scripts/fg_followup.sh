#!/bin/bash
# After the FineDiving tail run: (1) FineGym quick Ti-FAD + FIXED-weight parent loss (l_d=2, keep, 5-epoch quick protocol,
# via the quick runner); (2) concurrently on GPU 3, cache the element-level (l_d=3) STAGE-1 champion features; then
# (3) FineGym one-level champion + flat video ASA (K=10, 2+13 ep, eval every epoch) on that checkpoint.
cd /home/jupyter/xhu3/video/Dec14/af_fg3_v3 || exit 1
export PATH=/home/jupyter/xhu3/.conda/envs/qwen/bin:$PATH FINEGYM_RGB_ROOT=/home/jupyter/xhu3/video/dataset/finegym/RGB256png FINEGYM_FRAME_EXT=.png
PY=/home/jupyter/xhu3/.conda/envs/qwen/bin/python; LOG=logs/fg2_chain.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
until grep -q "FD V3 TAIL COMPLETE" logs/fd_chain.log; do sleep 60; done
sleep 20; ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9
log "=== FG follow-up: quick tifad_keep_asfix (runner v2) + l_d=3 stage-1 cache (GPU 3) ==="
setsid nohup bash scripts/run_fg2_quick_queue_v2.sh > /dev/null 2>&1 < /dev/null &
CK1=/home/jupyter/xhu3/video/Dec14/af_fg3_champ/ckpt/finegym_t3_champ_fg3_champ/vit_best_model
CACHE=/home/jupyter/xhu3/video/dataset/finegym/cache_fg3_champ1
if [ ! -f ckpt/fg3_cache1_done.marker ]; then
  sleep 90
  CUDA_VISIBLE_DEVICES=3 $PY scripts/cache_fg3_clip_feats.py --config configs/finegym_t3_champ_stage3_vsa_s1init.yaml --ckpt $CK1 --out $CACHE --verify 8 --workers 8 --batch 4 > logs/fg3_cache1.log 2>&1
  rc=$?; log "fg3 cache1 rc=$rc"; grep -hE "\[verify .* worst|Traceback|Error" logs/fg3_cache1.log | tail -2 | tee -a "$LOG"
  [ "$rc" -eq 0 ] && [ -f "$CACHE/meta.json" ] && touch ckpt/fg3_cache1_done.marker
fi
until [ -f ckpt/fg2_quick_tifad_keep_asfix_done.marker ] && ! pgrep -f 'run_fg2_quick_queue_v[2][.]sh' >/dev/null; do sleep 60; done
[ -f ckpt/fg3_cache1_done.marker ] || { log "fg3 cache1 missing; one-level video ASA not run"; exit 1; }
until [ -f ckpt/fg3_vsa1_done.marker ]; do
  R=""; [ -f ckpt/finegym_t3_champ_stage3_vsa_s1init_fg3_s1_vsa1/latest ] && R="--resume $(cat ckpt/finegym_t3_champ_stage3_vsa_s1init_fg3_s1_vsa1/latest)"
  log "=== fg3_vsa1 starting (one-level champion + flat video ASA K=10, init stage-1 fg3_champ, 2+13 ep, eval every epoch) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config configs/finegym_t3_champ_stage3_vsa_s1init.yaml --output fg3_s1_vsa1 --init_from $CK1 --freeze_except video_subtree_attn --feature_cache $CACHE --select_metric video_held_mAP -c 1 $R > logs/fg3_vsa1.log 2>&1; rc=$?
  log "=== fg3_vsa1 rc=$rc ==="; grep -h "\[ZSL video\]  \(seen\|held-out\)" logs/fg3_vsa1.log | tail -2 | tee -a "$LOG"
  ps -eo pid,ppid,args | awk '$2==1 && /python[^ ]* -c from multiprocessing.spawn/ {print $1}' | xargs -r kill -9
  [ "$rc" -eq 0 ] && touch ckpt/fg3_vsa1_done.marker || sleep 600
done
log "=== FG FOLLOW-UP COMPLETE ==="
