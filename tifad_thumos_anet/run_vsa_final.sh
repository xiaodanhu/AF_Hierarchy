#!/bin/bash
# VSA FINAL PROGRAM (2026-09-18: video-side subtree attention
# v1 -- libs/modeling/vsa.py, TIFAD_VSA=1, the stage-3 'vsa3' recipe trained
# on the frozen final champion with only vsa.* trainable -- is used).
# Dispatcher queue only
# (queue/jobs.txt, GPUs 0-3, 4 jobs/GPU, 6 GB gate).
#
# Stage-3 recipe (= logs/vsa3_s0.log banner: INITFROM seqchain, FREEZE
# except vsa, VSA K=10 heads=4; epochs from the cell's yaml):
#   env <cell train envs> TIFAD_VSA=1 TIFAD_INIT_FROM=<champion ckpt {n}>
#       TIFAD_FREEZE_EXCEPT=vsa python train_eval.py <cfg> --output <tag> --n n
# Champion checkpoints (ordered-phase seqchain stage, the 4-cell champion):
# Ablations (THUMOS 75/25, VSA in the full model):
#   vsa3nodp : vsa3 ckpts evaluated fused WITHOUT TIFAD_DURPRIOR (eval only)
#   vsa3noas : vsa3 stage on the 'no activity supervision' variant = the h19
#              (logs/ablnoAS_s*.log: no FAMCE, no chain_attn), then fused eval
# Seen-class mAP companions: scripts/seen_eval_proto.py (= scripts/seen_eval.py
# with the cell's fusion settings) on the champion and the +VSA ckpts of
# t50 / a75 / a50 (t75 is produced by run_seen_eval.sh).
# Tags: vsa50_s{n} vsaA75_s{n} vsaA50_s{n} vsa3noas_s{n} (train),
#       vsa50e/vsaA75e/vsaA50e/vsa3noase/vsa3nodp_s{n} (fused evals),
#       seenev_{vsa50,champ50,vsaA75,champA75,vsaA50,champA50}_s{n}.
# Usage: bash run_vsa_final.sh smoke      (ANet split-0 1-epoch train + init-equality eval)
#        bash run_vsa_final.sh queue_anet | queue_thumos | queue_nodp | queue_seen_champ
#        nohup bash run_vsa_final.sh watch > logs/vsa_final_watch.log 2>&1 &
# Table: python scripts/vsa_final_table.py
cd /data3/xiaodan8/tifad
Q() { flock /tmp/dispatch.lock tee -a queue/jobs.txt > /dev/null; }
mkdir -p queue/evq data/seen_eval logs

# ---- cell env strings (train = the recipe's own banners; eval = the champion eval cmds)
T_TR="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain"
T_EV="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain"
A_TR="TIFAD_PROMPT_ENSEMBLE=configs/anet_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_ARITH=configs/anet_arith.json TIFAD_ATTNDESIGN=phasechain"
A_EV="TIFAD_PROMPT_ENSEMBLE=configs/anet_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_ARITH=configs/anet_arith.json TIFAD_ATTNDESIGN=phasechain"
H19_TR="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3"
H19_EV="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1"
ADP="TIFAD_DURPRIOR=0.15 TIFAD_DURPRIOR_SRC=allseen TIFAD_DURPRIOR_SUBSET=training TIFAD_DURPRIOR_JSON=/data3/xiaodan8/Activitynet/anet1.3_i3d_filtered.json"
F_T75="TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3"
F_T50="TIFAD_FUSE_ALPHA=0.45 TIFAD_DURPRIOR=0.3"
F_A75="TIFAD_FUSE_ALPHA=0.15 $ADP"
F_A50="TIFAD_FUSE_ALPHA=0.35 $ADP"
SEQ="TIFAD_VSA=1 TIFAD_FREEZE_EXCEPT=vsa"

# cell -> cfg / train envs / eval envs / fuse / champion ckpt stem / vsa ckpt stem / tags
cell_def() {  # prints: cfg|tr|ev|fuse|champ_ckpt_fmt|vsa_ckpt_fmt|train_tag|eval_tag|seen_champ_arm|seen_vsa_arm
  case $1 in
    t50) echo "configs/thumos14_i3d_50.yaml|$T_TR|$T_EV|$F_T50|ckpt/thumos14_i3d_50_seqchain50_{n}|ckpt/thumos14_i3d_50_vsa50_{n}|vsa50|vsa50e|champ50|vsa50" ;;
    a75) echo "configs/anet_i3d_75.yaml|$A_TR|$A_EV|$F_A75|ckpt/anet_i3d_75_seqchainA75_{n}|ckpt/anet_i3d_75_vsaA75_{n}|vsaA75|vsaA75e|champA75|vsaA75" ;;
    a50) echo "configs/anet_i3d_50.yaml|$A_TR|$A_EV|$F_A50|ckpt/anet_i3d_50_seqchainA50_{n}|ckpt/anet_i3d_50_vsaA50_{n}|vsaA50|vsaA50e|champA50|vsaA50" ;;
    noas) echo "configs/thumos14_i3d.yaml|$H19_TR|$H19_EV|$F_T75|ckpt/thumos14_i3d_s1_{n}|ckpt/thumos14_i3d_vsa3noas_{n}|vsa3noas|vsa3noase|-|-" ;;
    *) echo "unknown cell $1" >&2; return 1 ;;
  esac
}
field() { echo "$1" | cut -d'|' -f$2; }
sub_n() { echo "${1//\{n\}/$2}"; }

train_line() {  # cell n
  local d; d=$(cell_def $1) || return 1
  local cfg tr champ tag
  cfg=$(field "$d" 1); tr=$(field "$d" 2); champ=$(field "$d" 5); tag=$(field "$d" 7)
  printf '%s_s%d\tenv %s %s TIFAD_INIT_FROM=%s/mAP_best.pth.tar python train_eval.py %s --output %s --n %d > logs/%s_s%d.log 2>&1\n' \
    $tag $2 "$tr" "$SEQ" "$champ" $cfg $tag $2 $tag $2
}
eval_line() {  # cell n
  local d; d=$(cell_def $1) || return 1
  local cfg ev fuse vsa etag
  cfg=$(field "$d" 1); ev=$(field "$d" 3); fuse=$(field "$d" 4); vsa=$(field "$d" 6); etag=$(field "$d" 8)
  printf '%s_s%d\tenv %s TIFAD_VSA=1 %s python eval.py %s %s --n %d > logs/%s_s%d.log 2>&1\n' \
    $etag $2 "$ev" "$fuse" $cfg "$(sub_n "$vsa" $2)" $2 $etag $2
}
seen_line() {  # cell n champ|vsa
  local d; d=$(cell_def $1) || return 1
  local cfg ev fuse ck arm extra
  cfg=$(field "$d" 1); ev=$(field "$d" 3); fuse=$(field "$d" 4)
  if [ "$3" = champ ]; then ck=$(field "$d" 5); arm=$(field "$d" 9); extra=""
  else ck=$(field "$d" 6); arm=$(field "$d" 10); extra=" TIFAD_VSA=1"; fi
  ck=$(sub_n "$ck" $2)/mAP_best.pth.tar
  [ -f data/seen_eval/${arm}_s$2.json ] && return 0   # already measured
  printf 'seenev_%s_s%d\tenv PYTHONUTF8=1 LC_ALL=C.UTF-8 python scripts/seen_eval_proto.py --fuse "%s" --config %s --ckpt %s --split %d --tag %s --envs "%s%s" --out data/seen_eval/%s_s%d.json > logs/seenev_%s_s%d.log 2>&1\n' \
    $arm $2 "$fuse" $cfg $ck $2 $arm "$ev" "$extra" $arm $2 $arm $2
}
nodp_line() {  # n : vsa3 ckpt, fused, NO durprior
  printf 'vsa3nodp_s%d\tenv %s TIFAD_VSA=1 TIFAD_FUSE_ALPHA=0.35 python eval.py configs/thumos14_i3d.yaml ckpt/thumos14_i3d_vsa3_%d --n %d > logs/vsa3nodp_s%d.log 2>&1\n' $1 "$T_EV" $1 $1 $1
}

SPLITS="0 1 2 3 4 5 6 7 8 9"
case "$1" in
  smoke)  # ANet 75/25 split 0: 1-epoch training (train path on the 192-step grid) + init-equality eval (eval path; must reproduce sqeA75_s0 38.48)
    { printf 'vsaA75_smoke\tenv %s %s TIFAD_MAX_EPOCHS=1 TIFAD_INIT_FROM=ckpt/anet_i3d_75_seqchainA75_{n}/mAP_best.pth.tar python train_eval.py configs/anet_i3d_75.yaml --output vsaA75smoke --n 0 > logs/vsaA75_smoke.log 2>&1\n' "$A_TR" "$SEQ"
      printf 'vsaA75_smokeev\tenv %s TIFAD_VSA=1 TIFAD_EVAL_NONSTRICT=1 %s python eval.py configs/anet_i3d_75.yaml ckpt/anet_i3d_75_seqchainA75_0 --n 0 > logs/vsaA75_smokeev.log 2>&1\n' "$A_EV" "$F_A75"
    } | Q ;;
  queue_anet)   for n in $SPLITS; do train_line a75 $n; train_line a50 $n; done | Q ;;
  queue_thumos) for n in $SPLITS; do train_line t50 $n; train_line noas $n; done | Q ;;
  queue_nodp)   for n in $SPLITS; do nodp_line $n; done | Q ;;
  queue_seen_champ) for c in t50 a75 a50; do for n in $SPLITS; do seen_line $c $n champ; done; done | Q ;;
  queue_seen_one) seen_line $2 $3 ${4:-champ} | Q ;;   # cell n champ|vsa (single job, e.g. wrapper test)
  print) for n in 0; do train_line a75 $n; train_line a50 $n; train_line t50 $n; train_line noas $n; eval_line a75 $n; eval_line t50 $n; eval_line noas $n; seen_line t50 $n champ; seen_line a75 $n vsa; nodp_line $n; done ;;
  watch)  # queue each fused eval + seen eval as its training lands; exits when all 40 evals + 30 seen jobs are queued
    log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
    log "watcher up (pid $$)"
    while :; do
      for c in t50 a75 a50 noas; do
        d=$(cell_def $c); ttag=$(field "$d" 7); etag=$(field "$d" 8); vsa=$(field "$d" 6)
        for n in $SPLITS; do
          [ -f queue/evq/${etag}_s$n ] && continue
          grep -q "All done" logs/${ttag}_s$n.log 2>/dev/null || continue
          [ -f "$(sub_n "$vsa" $n)/mAP_best.pth.tar" ] || continue
          { eval_line $c $n; [ $c != noas ] && seen_line $c $n vsa; } | Q
          touch queue/evq/${etag}_s$n; log "queued ${etag}_s$n (+seen)"
        done
      done
      k=$(ls queue/evq | grep -c "^vsa50e_s\|^vsaA75e_s\|^vsaA50e_s\|^vsa3noase_s")
      [ "$k" -ge 40 ] && { log "all 40 evals queued; exiting"; exit 0; }
      sleep 300
    done ;;
  feed)  # HEADROOM FEEDER: the dispatcher stacks 4 jobs/GPU as soon as 6 GB is
         # free, before earlier jobs claim their memory -> ANet VSA trainings
         # (4-6 GB each, growing) OOM'd. This loop hands the remaining
         # trainings and the champion seen-evals to queue/jobs.txt ONE AT A
         # TIME, only when the FIFO is empty and the GPU the dispatcher would
         # pick next (first g with >=6 GB free and <4 jobs) has >= NEED MB
         # free; crashed jobs (Traceback / no process, no 'All done') get their
         # log archived as <tag>.fail<k>.log and are fed again. Exits when
         # every wanted job is done. Placement itself stays the dispatcher's.
    NEED_TRAIN=${NEED_TRAIN:-8000}; NEED_SEEN=${NEED_SEEN:-7000}
    log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
    log "feeder up (pid $$) NEED_TRAIN=$NEED_TRAIN NEED_SEEN=$NEED_SEEN"
    is_running() { pgrep -f -- "$1" > /dev/null; }
    archive() { local k; k=$(ls logs/$1.fail*.log 2>/dev/null | wc -l); mv logs/$1.log logs/$1.fail$((k+1)).log; log "FAILED $1 -> logs/$1.fail$((k+1)).log (will re-feed)" >&2; }   # stderr: state() is called in $(...)
    # job state: prints done|running|queued|pending   ($1 tag $2 done-test $3 pgrep pattern)
    state() {
      local tag=$1 f=logs/$1.log
      eval "$2" && { echo done; return; }
      grep -qP "^${tag}\t" queue/jobs.txt 2>/dev/null && { echo queued; return; }
      if [ -f "$f" ]; then
        if is_running "$3"; then
          grep -q "Traceback\|out of memory" "$f" && ! grep -q "All done" "$f" && { echo dying; return; }
          echo running; return
        fi
        [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 90 ] && { echo running; return; }
        archive "$tag"
      fi
      echo pending
    }
    while :; do
      pend=(); nleft=0
      for c in a75 a50 t50 noas; do
        d=$(cell_def $c); ttag=$(field "$d" 7); vsa=$(field "$d" 6)
        for n in $SPLITS; do
          st=$(state ${ttag}_s$n "[ -f $(sub_n "$vsa" $n)/mAP_best.pth.tar ] && grep -q 'All done' logs/${ttag}_s$n.log 2>/dev/null" "python train_eval.py .* --output $ttag --n $n( |$)")
          [ "$st" = done ] || nleft=$((nleft+1))
          [ "$st" = pending ] && pend+=("train $c $n")
        done
      done
      for c in t50 a75 a50; do
        d=$(cell_def $c); arm=$(field "$d" 9)
        for n in $SPLITS; do
          st=$(state seenev_${arm}_s$n "[ -f data/seen_eval/${arm}_s$n.json ]" "seen_eval_proto.py .*--out data/seen_eval/${arm}_s$n.json")
          [ "$st" = done ] || nleft=$((nleft+1))
          [ "$st" = pending ] && pend+=("seen $c $n")
        done
      done
      [ "$nleft" -eq 0 ] && { log "all wanted jobs done; feeder exiting"; exit 0; }
      if [ ${#pend[@]} -gt 0 ] && [ ! -s queue/jobs.txt ] && [ ! -f queue/PAUSE ]; then
        set -- ${pend[0]}; kind=$1; c=$2; n=$3
        need=$NEED_TRAIN; [ $kind = seen ] && need=$NEED_SEEN
        for g in 0 1 2 3; do
          free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
          njobs=$(pgrep -fc "[d]ispjob_gpu$g" || true)
          [ "$free" -ge 6000 ] && [ "$njobs" -lt 4 ] || continue
          if [ "$free" -ge "$need" ]; then
            if [ $kind = train ]; then train_line $c $n | Q; else seen_line $c $n champ | Q; fi
            log "fed $kind $c s$n (expect GPU$g, free=${free}MB, ${#pend[@]} pending, $nleft not done)"
            sleep 150
          fi
          break   # the dispatcher would take this GPU first; if it lacks headroom, wait
        done
      fi
      sleep 60
    done ;;
  *) echo "usage: $0 smoke|queue_anet|queue_thumos|queue_nodp|queue_seen_champ|queue_seen_one|feed|watch|print"; exit 1 ;;
esac
