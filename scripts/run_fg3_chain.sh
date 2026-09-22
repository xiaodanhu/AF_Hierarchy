#!/bin/bash
# FineGym Table-3: three systems SEQUENTIALLY, each on GPUs 0+1+2 (GPU2 allowed
# since 2026-09-12; GPU3 NEVER) via
# deepspeed data parallel (user directive). GPU3 NEVER touched.
#
# v2 (2026-09-10): the old 8.5GB gate let FG3 start while tifad dispatcher
# jobs were still training; those jobs grow over time and OOM-killed FG3
# base at epoch 1 (and FG3's footprint OOMed two ANet seqchain cells).
# Now each system (a) waits until the tifad dispatcher is fully drained
# (empty queue AND no dispjob processes) and both GPUs have >=13GB free,
# (b) holds queue/PAUSE so the dispatcher places nothing while FG3 trains,
# and (c) RETRIES on nonzero exit instead of silently moving on.
cd /data3/xiaodan8/actionformer6
QF=/data3/xiaodan8/tifad/queue/jobs.txt
PAUSE=/data3/xiaodan8/tifad/queue/PAUSE
# champ2 = champion stage 2 (APA phase-level attention): initialised from the
# best stage-1 champion checkpoint, everything else frozen (paper Sec. 3.3).
for sys in tifad champ champ2; do   # base dropped 2026-09-12 (user): compare ours vs Ti-FAD only
  extra=""
  cfgname=$sys
  if [ "$sys" = champ2 ]; then
    cfgname=champ_stage2
    extra="--init_from ckpt/finegym_t3_champ_fg3_champ/vit_best_model --freeze_except text_pathway.phase_attn"
  fi
  until [ -f "ckpt/fg3_${sys}_done.marker" ]; do
    while :; do
      # NB: grep -c prints 0 AND exits 1 on an empty file, so no "|| echo 0"
      # (that produced "0\n0" and broke the numeric test — chain stalled 09-12)
      nq=$(grep -c . "$QF" 2>/dev/null); nq=${nq:-0}
      nj=$(pgrep -fc "[d]ispjob_gpu"); nj=${nj:-0}
      f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
      f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
      f2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)
      [ "$nq" -eq 0 ] && [ "$nj" -eq 0 ] && [ "$f0" -ge 13000 ] && [ "$f1" -ge 13000 ] && [ "$f2" -ge 13000 ] && break
      sleep 300
    done
    touch "$PAUSE"
    echo "=== FG3 $sys starting $(date) ===" >> logs/fg3_chain.log
    CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --master_port=2971$((RANDOM % 10)) train_shard.py \
      --config configs/finegym_t3_${cfgname}.yaml --output fg3_${sys} $extra \
      > logs/fg3_${sys}.log 2>&1
    rc=$?
    rm -f "$PAUSE"
    echo "=== FG3 $sys rc=$rc $(date) ===" >> logs/fg3_chain.log
    if [ "$rc" -eq 0 ]; then
      touch "ckpt/fg3_${sys}_done.marker"
    else
      sleep 600   # back off, then re-enter the gate and retry this system
    fi
  done
done
echo "=== FG3 CHAIN COMPLETE $(date) ===" >> logs/fg3_chain.log
grep -H "held-out" logs/fg3_*.log | tail -20 >> logs/fg3_chain.log
