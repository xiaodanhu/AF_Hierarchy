#!/bin/bash
# queue the fused+durprior eval of each vsa3 split as its training lands (GPUs 0/1/2 via dispatcher; GPU3 never)
cd /data3/xiaodan8/tifad
CH="TIFAD_PROMPT_ENSEMBLE=configs/thumos14_qwen_ensemble.json TIFAD_ATTR_ATTN=1 TIFAD_PROMPT_DROPOUT=0.3 TIFAD_SEM_DEGRADE=0.3 TIFAD_FAMCE=0.3 TIFAD_FAMCE_MAP=configs/thumos14_fam_map.json TIFAD_FAMCE_UW=1 TIFAD_ATTNDESIGN=phasechain TIFAD_VSA=1 TIFAD_FUSE_ALPHA=0.35 TIFAD_DURPRIOR=0.3"
mkdir -p queue/evq
while :; do
  for n in 0 1 2 3 4 5 6 7 8 9; do
    [ -f queue/evq/vsa3e_s$n ] && continue
    grep -q "All done" logs/vsa3_s$n.log 2>/dev/null || continue
    [ -f ckpt/thumos14_i3d_vsa3_$n/mAP_best.pth.tar ] || continue
    printf 'vsa3e_s%d\tenv %s python eval.py configs/thumos14_i3d.yaml ckpt/thumos14_i3d_vsa3_%d --n %d > logs/vsa3e_s%d.log 2>&1\n' $n "$CH" $n $n $n | flock /tmp/dispatch.lock tee -a queue/jobs.txt > /dev/null
    touch queue/evq/vsa3e_s$n
  done
  [ "$(ls queue/evq | grep -c vsa3e_)" -ge 10 ] && exit 0
  sleep 300
done
