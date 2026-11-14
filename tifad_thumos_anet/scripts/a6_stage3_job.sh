#!/bin/bash
# FineDiving / FineGym stage-3 variant (video branch only, frozen
# stage-2 champion, cached CLIP features) on ONE GPU in actionformer6, as a dispatcher job.
# usage (dispatcher): <tag><TAB>bash scripts/a6_stage3_job.sh <fd|fg> <tag> <config file name>
# Same training call as fd_stage3_job.sh / the fgMEAN run of 11-07, but the slot is held
# with gpu_hold.sh until the card has the memory (fdK4 OOMed on 11-07 with the plain 8 GB wait).
ds=$1; tag=$2; cfg=$3
source /data3/xiaodan8/tifad/scripts/gpu_hold.sh
cd /data3/xiaodan8/actionformer6 || exit 1
case $ds in
  fd) CFG=configs/finediving/v3/$cfg; CK=ckpt/fd_champ_s1_a4000_fda_s1_champ/vit_best_model; CACHE=/data3/xiaodan8/FineDiving/cache_fd3_s1; MIN=9000 ;;
  fg) CFG=configs/$cfg; CK=ckpt/finegym_t3_champ_stage2_fg3_champ2/vit_best_model; CACHE=/data3/xiaodan8/FineGym/cache_fg3_champ2; MIN=11000 ;;
  *) echo "unknown dataset $ds"; exit 1 ;;
esac
[ -f "$CFG" ] || { echo "missing config $CFG"; exit 1; }
trap 'release_card $tag' EXIT
hold_card "$tag"
wait_free $MIN 2880 || echo "[$(date '+%F %T')] $tag: no $MIN MB free after 48 h, launching anyway" >> logs/$tag.wait.log
deepspeed --master_port=$((29000 + RANDOM % 900)) train_shard.py --config $CFG \
  --output $tag --init_from $CK --freeze_except video_subtree_attn --feature_cache $CACHE \
  --select_metric video_held_mAP -c 1 > logs/$tag.log 2>&1 &
pid=$!
sleep 240; release_card "$tag"
wait $pid
