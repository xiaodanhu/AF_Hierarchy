#!/bin/bash
# Video-LLM baseline on ONE GPU: wait for >= 11 GB free, run inference in the LLM env,
# then score the saved predictions in the training env (the evaluator's numpy alias).
cd /data3/xiaodan8/tifad || exit 1
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES)" -ge 11000 ]; do sleep 120; done
/data/xiaodan8/anaconda3/envs/hierarchy/bin/python scripts/vlm_localize_baseline.py --n 0 --no_eval > logs/vlm_s0.log 2>&1 && \
/data/xiaodan8/anaconda3/envs/parsing2/bin/python scripts/vlm_localize_baseline.py --n 0 --eval_json logs/vlm_s0.json >> logs/vlm_s0.log 2>&1
