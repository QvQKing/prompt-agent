#!/bin/bash

# 指定 GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 指定模型路径（就是你 merge 完后的 HF 模型目录）
export MODEL_NAME='./merge_model/Qwen2.5-1.5B-Instruct'

# 启动 vLLM 服务
vllm serve $MODEL_NAME \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name agent \
  --port 8000 \
  --tensor-parallel-size 4 
