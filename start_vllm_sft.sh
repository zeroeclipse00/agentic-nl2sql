#!/bin/bash
# Start vLLM server for SFT model on free GPU 1 (port 10081)
# GPU 0 is occupied; GPU 1 is free (~46GB available)

export CUDA_VISIBLE_DEVICES=1

echo "Starting vLLM server for SFT model..."
echo "  Model: /var/tmp/wangshangshu/models/agent-7b-hf"
echo "  Port:  10081"
echo "  GPU:   1 (A40)"

/home/koujianshang/miniconda3/envs/rlfac/bin/vllm serve \
    /var/tmp/wangshangshu/models/agent-7b-hf \
    --host 0.0.0.0 \
    --port 10081 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --served-model-name sft-agent-7b \
    --max-model-len 8192 \
    --dtype bfloat16
