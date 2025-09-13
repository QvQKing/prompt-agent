#!/bin/bash

export CHECKPOINT_DIR='./checkpoints/Prompt-R1/grpo-qwen2.5-3b-instruct/global_step_320/actor'
export HF_MODEL_PATH='./Qwen/Qwen2.5-3B-Instruct'
export TARGET_DIR='./merge_model/Qwen2.5-3B-Instruct'

python3 verl/scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path "$HF_MODEL_PATH" \
  --local_dir "$CHECKPOINT_DIR" \
  --target_dir "$TARGET_DIR"
