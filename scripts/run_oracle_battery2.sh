#!/bin/bash
# Battery 2: fixed suppress-only phrase oracle + composition-weight (alpha) sweep
# + top-k cap probe. All on the split0_75 checkpoint, old-arch frozen config.
cd /data3/xiaodan8/actionformer6
CFG=configs/anet_zsl/eval_arch224_split0_75.yaml
CKPT=ckpt/activitynet_i3d_zsl_split0_75_anet_smoke/epoch_020
E=scripts/eval_anet_zsl.py
CUDA_VISIBLE_DEVICES=0 python $E --config $CFG --ckpt $CKPT --oracle phrase       > logs/oracle/b2_phrasefix.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python $E --config $CFG --ckpt $CKPT --oracle all          > logs/oracle/b2_allfix.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python $E --config $CFG --ckpt $CKPT --oracle none --alpha 0.3 > logs/oracle/b2_alpha03.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python $E --config $CFG --ckpt $CKPT --oracle none --alpha 1.0 > logs/oracle/b2_alpha10.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python $E --config $CFG --ckpt $CKPT --oracle none --max-seg-num 200 > logs/oracle/b2_seg200.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python $E --config $CFG --ckpt $CKPT --oracle all --alpha 0.3 > logs/oracle/b2_allfix_a03.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python $E --config $CFG --ckpt $CKPT --oracle none --alpha 0.05 > logs/oracle/b2_alpha005.log 2>&1 &
wait
echo "battery2 done $(date)"
