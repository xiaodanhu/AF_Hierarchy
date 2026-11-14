#!/bin/bash
# Oracle diagnostics battery: 10 evals of the split0_75 checkpoint with
# GT-substituted phrase/specialist predictions. 4 parallel per wave.
cd /data3/xiaodan8/actionformer6
CFG=configs/anet_zsl/eval_arch224_split0_75.yaml
CKPT=ckpt/activitynet_i3d_zsl_split0_75_anet_smoke/epoch_020
run() {  # $1=gpu $2=oracle-spec $3=logname
  CUDA_VISIBLE_DEVICES=$1 python scripts/eval_anet_zsl.py \
    --config $CFG --ckpt $CKPT --oracle "$2" > logs/oracle/$3.log 2>&1
}
run 0 all            oracle_all    & run 1 phrase           oracle_phrase &
run 2 named_element  oracle_object & run 3 motion_category  oracle_motion &
wait
run 0 body_shape     oracle_env    & run 1 direction        oracle_posture &
run 2 multiplicity   oracle_social & run 3 leg_configuration oracle_repetitive &
wait
run 0 inverted_phase oracle_water  & run 1 hands_on_apparatus oracle_handheld &
run 2 phrase,named_element oracle_phrase_object &
wait
echo "battery done $(date)"
