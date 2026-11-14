#!/bin/bash
# SEEN-VOCAB CONTROL sweep (2026-09-18): for every candidate checkpoint,
# scripts/seen_eval.py measures the standard UNSEEN mAP and the SEEN-class
# TIFAD_DURPRIOR=0.3) and unfused, one dispatcher job per (arm, split).
# Dispatcher queue only (queue/jobs.txt, GPUs 0-3). Tags seenev_<arm>_s<n>,
# outputs data/seen_eval/<arm>_s<n>.json, logs logs/seenev_<arm>_s<n>.log.
# Arms (env strings from the runs' own log banners / run_*.sh):
#   champ       ckpt seqchain_n, champion envs (dropout/degrade 0), s0-9
#   vsa3        ckpt vsa3_n + TIFAD_VSA=1, s0-9
#   xm / v2a    VSA v2 arms (xm: cross-modal match only; v2a: + clip attn)
#   v3a-d       VSA v3 arms (softphase + posemb + align; a: 4-way UW,
#               b: no-self + out_init 1e-3, c: min_lvl 2, d: mix 1.0)
#   v4b_geo0.25 / v4b_prod0.5 / v4e_geo0.25   quality head combinations
#   v5ex        v5e phase-profile boundary head, bound only
#   v5cj        joint contrastive trunk, plain detection (extent 0)
#   v5t_mean / v5t_whiten   de-anisotropised phase embeddings
#   reanchor    eval-only post-NMS re-anchoring on the champion
# Table: python scripts/seen_eval_table.py
# Usage: bash run_seen_eval.sh queue [arm ...]     (default: everything)
cd /data3/xiaodan8/tifad
CH="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0 TIFAD_SEM_DEGRADE=0 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain"
V3="TIFAD_LEAFLOSS=1 TIFAD_LEAFLOSS_HEAD=1 TIFAD_LEAFLOSS_W=1.0 TIFAD_VSA=1 TIFAD_VSA_XM=1 TIFAD_VSA_SOFTPHASE=1 TIFAD_VSA_POSEMB=1 TIFAD_VSA_ALIGN_W=0.3"
QB="TIFAD_QH=1 TIFAD_QH_PHASEMAP=1"
mkdir -p data/seen_eval
Q() { flock /tmp/dispatch.lock tee -a queue/jobs.txt; }

# arm -> "ckpt-dir-stem<TAB>extra envs" (ckpt = ckpt/thumos14_i3d_<stem>_<n>/mAP_best.pth.tar)
arm_def() {
  case $1 in
    champ)       printf 'seqchain\t' ;;
    vsa3)        printf 'vsa3\tTIFAD_VSA=1' ;;
    xm)          printf 'xm\tTIFAD_VSA=1 TIFAD_VSA_XM=1 TIFAD_VSA_XM_ONLY=1' ;;
    v2a)         printf 'v2a\tTIFAD_VSA=1 TIFAD_VSA_XM=1' ;;
    v3a)         printf 'v3a\t%s TIFAD_VSA_ALIGN_UW=1' "$V3" ;;
    v3b)         printf 'v3b\t%s TIFAD_VSA_NOSELF=1 TIFAD_VSA_OUTINIT=0.001' "$V3" ;;
    v3c)         printf 'v3c\t%s TIFAD_VSA_MINLVL=2' "$V3" ;;
    v3d)         printf 'v3d\t%s TIFAD_VSA_SOFTPHASE_MIX=1.0' "$V3" ;;
    v4b_geo0.25) printf 'v4b\t%s TIFAD_QH_COMB=geo TIFAD_QH_BETA=0.25' "$QB" ;;
    v4b_prod0.5) printf 'v4b\t%s TIFAD_QH_COMB=prod TIFAD_QH_BETA=0.5' "$QB" ;;
    v4e_geo0.25) printf 'v4e\t%s TIFAD_QH_LVLFEAT=1 TIFAD_QH_COMB=geo TIFAD_QH_BETA=0.25' "$QB" ;;
    v5ex)        printf 'v5e\tTIFAD_PH_SRC=head TIFAD_PH_BOUND=1' ;;
    v5cj)        printf 'v5cj\tTIFAD_PH_SRC=head TIFAD_PH_TARGET=decile TIFAD_PH_CTR=1 TIFAD_PH_CTR_MODE=joint TIFAD_PH_EXTENT=0' ;;
    v5t_mean)    printf 'v5t_mean\tTIFAD_TXT_DEANISO=mean' ;;
    v5t_whiten)  printf 'v5t_whiten\tTIFAD_TXT_DEANISO=whiten' ;;
    reanchor)    printf 'seqchain\tTIFAD_REANCHOR=1' ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}
arm_splits() {
  case $1 in
    champ|vsa3) echo "0 1 2 3 4 5 6 7 8 9" ;;
    *) echo "0 1 2" ;;
  esac
}

job() {  # arm split -> queue line (skips if the json already exists)
  local arm=$1 n=$2 def stem extra ck
  def=$(arm_def $arm) || return 1
  stem=${def%%$'\t'*}; extra=${def#*$'\t'}
  ck=ckpt/thumos14_i3d_${stem}_$n/mAP_best.pth.tar
  [ -f "$ck" ] || { echo "[skip] $arm s$n: no $ck" >&2; return 0; }
  [ -f data/seen_eval/${arm}_s$n.json ] && { echo "[skip] $arm s$n: json exists" >&2; return 0; }
  printf 'seenev_%s_s%d\tenv PYTHONUTF8=1 LC_ALL=C.UTF-8 python scripts/seen_eval.py --ckpt %s --split %d --tag %s --envs "%s%s" --out data/seen_eval/%s_s%d.json > logs/seenev_%s_s%d.log 2>&1\n' \
    $arm $n $ck $n $arm "$CH" "${extra:+ $extra}" $arm $n $arm $n
}

ALL="champ vsa3 xm v2a v3a v3b v3c v3d v4b_geo0.25 v4b_prod0.5 v4e_geo0.25 v5ex v5cj v5t_mean v5t_whiten reanchor"
case "$1" in
  queue) shift; arms="${*:-$ALL}"
    { for arm in $arms; do for n in $(arm_splits $arm); do job $arm $n; done; done; } | Q ;;
  print) shift; arms="${*:-$ALL}"
    for arm in $arms; do for n in $(arm_splits $arm); do job $arm $n; done; done ;;
  *) echo "usage: $0 queue|print [arm ...]"; exit 1 ;;
esac
