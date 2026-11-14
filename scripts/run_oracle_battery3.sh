#!/bin/bash
# Battery 3: coarse-level score propagation probe (the "long actions have no
# held-out scores at their FPN level" hypothesis).
cd /data3/xiaodan8/actionformer6
CFG=configs/anet_zsl/eval_arch224_split0_75.yaml
CKPT=ckpt/activitynet_i3d_zsl_split0_75_anet_smoke/epoch_020
E=scripts/eval_anet_zsl.py
CUDA_VISIBLE_DEVICES=0 python $E --config $CFG --ckpt $CKPT --oracle none --extend-levels                > logs/oracle/b3_extend.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python $E --config $CFG --ckpt $CKPT --oracle none --extend-levels --alpha 0.3    > logs/oracle/b3_extend_a03.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python $E --config $CFG --ckpt $CKPT --oracle all  --extend-levels                > logs/oracle/b3_extend_all.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python $E --config $CFG --ckpt $CKPT --oracle all  --extend-levels --alpha 0.3    > logs/oracle/b3_extend_all_a03.log 2>&1 &
wait
echo "battery3 done $(date)"
