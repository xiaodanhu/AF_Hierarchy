#!/bin/bash
# Re-queue control jobs whose final result is missing (2026-09-25 disk-full recovery).
# Chains skip completed stages; eval jobs are cheap to repeat.
cd /data3/xiaodan8/tifad
Q() { flock /tmp/dispatch.lock tee -a queue/jobs.txt > /dev/null; }
has_map() { grep -q "Ave.*mAP" "$1" 2>/dev/null; }
ADP="TIFAD_DURPRIOR=0.15 TIFAD_DURPRIOR_SRC=allseen TIFAD_DURPRIOR_SUBSET=training TIFAD_DURPRIOR_JSON=/data3/xiaodan8/Activitynet/anet1.3_i3d_filtered.json"
CH="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain TIFAD_VSA=1 TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3"
T="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain"
E="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain"
n_q=0
{
[ -f data/rep_metrics/vsa3_post.json ] || { printf 'probe_post\tenv PYTHONUTF8=1 LC_ALL=C.UTF-8 python scripts/repvalid_variant.py --tag vsa3_post --ckpt ckpt/thumos14_i3d_vsa3_{n}/mAP_best.pth.tar --config configs/thumos14_i3d.yaml --envs "TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_ATTNDESIGN=phasechain TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_VSA=1 TIFAD_PROBE_STASH_POST=1" --splits 0 1 2 > logs/probe_post.log 2>&1\n'; }
for n in 0 1 2 3 4 5 6 7 8 9; do
  has_map logs/vanasp_a75_s$n.log || printf 'vanasp_a75_s%d\tenv TIFAD_FUSE_ALPHA=0.15 %s python eval.py configs/anet_i3d_75.yaml ckpt/anet_i3d_75_van_%d --n %d > logs/vanasp_a75_s%d.log 2>&1\n' $n "$ADP" $n $n $n
  has_map logs/vanasp_a50_s$n.log || printf 'vanasp_a50_s%d\tenv TIFAD_FUSE_ALPHA=0.35 %s python eval.py configs/anet_i3d_50.yaml ckpt/anet_i3d_50_van_%d --n %d > logs/vanasp_a50_s%d.log 2>&1\n' $n "$ADP" $n $n $n
  has_map logs/shufeval_s$n.log || printf 'shufeval_s%d\tenv %s TIFAD_SHUFFLE_CHILDREN=eval TIFAD_SHUFFLE_SEED=0 python eval.py configs/thumos14_i3d.yaml ckpt/thumos14_i3d_vsa3_%d --n %d > logs/shufeval_s%d.log 2>&1\n' $n "$CH" $n $n $n
  has_map logs/nohp50e_s$n.log || printf 'nohp50_s%d\tenv TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 python train_eval.py configs/thumos14_i3d_50.yaml --output nohp50 --n %d > logs/nohp50_s%d.log 2>&1 && env TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_FUSE_ALPHA=0.45 TIFAD_DURPRIOR=0.3 python eval.py configs/thumos14_i3d_50.yaml ckpt/thumos14_i3d_50_nohp50_%d --n %d > logs/nohp50e_s%d.log 2>&1\n' $n $n $n $n $n $n
  has_map logs/vsajointe_s$n.log || printf 'vsajoint_s%d\tenv %s TIFAD_VSA=1 TIFAD_INIT_FROM=ckpt/thumos14_i3d_seqchain_{n}/mAP_best.pth.tar TIFAD_FREEZE_EXCEPT=vsa TIFAD_LR_MULT_REST=0.1 python train_eval.py configs/thumos14_i3d.yaml --output vsajoint --n %d > logs/vsajoint_s%d.log 2>&1 && env %s TIFAD_VSA=1 TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3 python eval.py configs/thumos14_i3d.yaml ckpt/thumos14_i3d_vsajoint_%d --n %d > logs/vsajointe_s%d.log 2>&1\n' $n "$T" $n $n "$E" $n $n $n
  for k in 3 5; do grep -q DONE logs/k${k}_chain_s$n.log 2>/dev/null || printf 'k%d_s%d\tbash scripts/onelevel_chain.sh k%d %d ENS=configs/thumos14_qwen_ensemble_k%d.json >> logs/k%d_chain_s%d.log 2>&1\n' $k $n $k $n $k $k $n; done
  grep -q DONE logs/chainfull_chain_s$n.log 2>/dev/null || printf 'chainfull_s%d\tbash scripts/onelevel_chain.sh chainfull %d INIT1=ckpt/thumos14_i3d_famce_uw_{n} STAGES="2 3" EXTRA=TIFAD_CHAIN_FULL=1 >> logs/chainfull_chain_s%d.log 2>&1\n' $n $n $n
  grep -q DONE logs/shuf_chain_s$n.log 2>/dev/null || printf 'shuftrain_s%d\tbash scripts/onelevel_chain.sh shuf %d EXTRA=TIFAD_SHUFFLE_CHILDREN=train >> logs/shuf_chain_s%d.log 2>&1\n' $n $n $n
done
} > /tmp/requeue.txt
# stale partial checkpoints of unfinished stages are overwritten by the chains; drop stage logs without "All done" so done_stage re-runs them
for f in logs/k[35][123]_s*.log logs/chainfull[23]_s*.log logs/shuf[123]_s*.log logs/nohp50_s*.log logs/vsajoint_s*.log; do [ -f "$f" ] && ! grep -q "All done" "$f" && rm -f "$f"; done
cat /tmp/requeue.txt | Q
echo "requeued $(wc -l < /tmp/requeue.txt) jobs; queue now $(wc -l < queue/jobs.txt)"
cut -f1 /tmp/requeue.txt | sed 's/_s[0-9]*$//' | sort | uniq -c
