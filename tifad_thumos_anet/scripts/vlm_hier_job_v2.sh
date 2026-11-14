#!/bin/bash
# C3 hierarchical-VLM chain on ONE dedicated A4000. Dispatcher: write "0" to queue/MAXJOBS_gpu<g>
# first so the card drains to this job; the wrapper waits for >= 11 GB free.
#   bash scripts/vlm_hier_job.sh thumos 0   # prior -> ground (self-label cuts) -> train -> infer tuned -> infer zero-shot -> eval
#   bash scripts/vlm_hier_job.sh fd         # prior -> ground (self-label member spans) -> train -> infer tuned -> infer zero-shot -> eval
#   bash scripts/vlm_hier_job.sh fd_oracle  # train on annotated sub-action spans (ceiling) -> infer tuned -> eval
# Settings tree = one annotated level + our definition tree, children self-labelled with
# variable lengths (video + LLM prior, never the equal split); oracle = all annotated levels.
# v2 the FineDiving zero-shot pass runs on the 150-window subsample data/vlm_hier/fd_s1/zs_subset_ids.txt
# (75 windows with a held-out dive + 75 without, seed 0); the tuned model is ALSO evaluated on that subsample for a
# like-for-like comparison, besides the full test set. v1 kept as scripts/vlm_hier_job_v1.sh.
# Every stage is resumable; a finished stage is skipped on re-run.
set=$1; n=${2:-0}
cd /data3/xiaodan8/tifad || exit 1
H=/data/xiaodan8/anaconda3/envs/hierarchy/bin/python
P2=/data/xiaodan8/anaconda3/envs/parsing2/bin/python
case $set in
  thumos)    ds=thumos; DATA=data/vlm_hier/thumos_s$n;   TAG=thumos_s$n;   PRIOR=configs/thumos14_step_prior.json;   TF=$DATA/train_nested.jsonl ;;
  fd)        ds=fd;     DATA=data/vlm_hier/fd_s1;        TAG=fd_s1;        PRIOR=configs/finediving_step_prior.json; TF=$DATA/train_nested.jsonl ;;
  fd_oracle) ds=fd;     DATA=data/vlm_hier/fd_s1_oracle; TAG=fd_s1_oracle; PRIOR="";                                 TF=$DATA/train.jsonl ;;
  *) echo "unknown setting $set"; exit 1 ;;
esac
IDS=""; [ "$ds" = fd ] && [ -f data/vlm_hier/fd_s1/zs_subset_ids.txt ] && IDS=data/vlm_hier/fd_s1/zs_subset_ids.txt
CK=ckpt/vlm_hier/$TAG; LOG=logs/vlm_hier; mkdir -p $CK $LOG
log() { echo "[$(date '+%F %T')] vlm_hier $TAG: $*" >> $LOG/${TAG}_chain.log; }
for _i in $(seq 1 720); do
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0})
  [ "${_free:-0}" -ge 11000 ] && break
  sleep 60
done
log "start (free ${_free} MB)"
if [ -n "$PRIOR" ] && ! grep -q "^done ->" $LOG/${TAG}_prior.log 2>/dev/null; then
  log "prior"; $H scripts/vlm_hier/gen_step_prior.py --dataset $ds --device 8bit --out $PRIOR >> $LOG/${TAG}_prior.log 2>&1 || { log "prior FAILED"; exit 1; }
fi
if [ -n "$PRIOR" ] && ! grep -q "^done ->" $LOG/${TAG}_ground.log 2>/dev/null; then
  log "ground"; $H scripts/vlm_hier/ground.py --dataset $ds --data $DATA --prior $PRIOR >> $LOG/${TAG}_ground.log 2>&1 || { log "ground FAILED"; exit 1; }
fi
if [ ! -d $CK/final ]; then
  RES=""; last=$(ls -d $CK/epoch[0-9]* 2>/dev/null | sort -V | tail -1); [ -n "$last" ] && RES="--resume_from $last"   # reboot recovery: continue from the last saved epoch
  log "train ${RES:+(resume from $last)}"; $H scripts/vlm_hier/train.py --dataset $ds --data $DATA --train_file $TF --out $CK $RES >> $LOG/${TAG}_train.log 2>&1 || { log "train FAILED"; exit 1; }
fi
if ! grep -q "^done ->" $LOG/${TAG}_infer.log 2>/dev/null; then
  log "infer tuned"; $H scripts/vlm_hier/infer.py --dataset $ds --data $DATA --adapter $CK/final --out $LOG/${TAG}_pred.jsonl >> $LOG/${TAG}_infer.log 2>&1 || { log "infer FAILED"; exit 1; }
fi
ZS=$LOG/${ds}_zs_pred.jsonl                      # zero-shot base model, shared by fd / fd_oracle
if [ "$set" != fd_oracle ] && ! grep -q "^done ->" $LOG/${ds}_zs_infer.log 2>/dev/null; then
  log "infer zero-shot${IDS:+ (subsample $IDS)}"; $H scripts/vlm_hier/infer.py --dataset $ds --data $DATA --adapter '' --out $ZS ${IDS:+--ids $IDS} >> $LOG/${ds}_zs_infer.log 2>&1 || log "infer zero-shot FAILED"
fi
log "eval"
if [ "$ds" = thumos ]; then
  { for src in greedy all; do $P2 scripts/vlm_hier/eval_thumos.py --n $n --pred $LOG/${TAG}_pred.jsonl --src $src; done
    [ -f $ZS ] && for src in greedy all; do $P2 scripts/vlm_hier/eval_thumos.py --n $n --pred $ZS --src $src; done; } > $LOG/${TAG}_eval.log 2>&1
else
  TP=/data3/xiaodan8/tifad/$LOG/${TAG}_pred.jsonl; TZ=/data3/xiaodan8/tifad/$ZS; TI=""; [ -n "$IDS" ] && TI=/data3/xiaodan8/tifad/$IDS
  (cd /data3/xiaodan8/actionformer6 && {
     for src in greedy all; do $P2 scripts/vlm_hier_eval_fd.py --pred $TP --src $src; done                                   # tuned, full test set
     if [ -n "$TI" ]; then for src in greedy all; do $P2 scripts/vlm_hier_eval_fd.py --pred $TP --src $src --ids $TI; done; fi  # tuned, zero-shot subsample (like-for-like)
     if [ -f "$TZ" ]; then for src in greedy all; do $P2 scripts/vlm_hier_eval_fd.py --pred $TZ --src $src ${TI:+--ids $TI}; done; fi   # zero-shot base model, subsample
   }) > $LOG/${TAG}_eval.log 2>&1
fi
log "DONE"; grep -h "RESULT\|CHILDREN\|SUB-ACTIONS" $LOG/${TAG}_eval.log >> $LOG/${TAG}_chain.log
