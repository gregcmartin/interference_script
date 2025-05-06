#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
VLLM_PATH=~/tools/vllm

# System-specific optimizations for dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export CUDA_LAUNCH_BLOCKING=0    # Async CUDA operations
export NCCL_P2P_DISABLE=0       # Enable GPU P2P communication

# Navigate to vllm directory
cd $VLLM_PATH

# Run the model with optimized parameters
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \  # Use both GPUs
    --gpu-memory-utilization 0.95 \  # Use 95% of GPU memory
    --max-num-batched-tokens 8192 \  # Adjust based on available memory
    --max-num-seqs 256 \  # Maximum number of concurrent sequences
    --quantization awq \  # Use AWQ quantization if supported
    --dtype float16 \  # Use half precision for better performance
    --trust-remote-code \  # Required for some models
    --port 8000 \  # API server port
    "$@"  # Pass any additional arguments

# Note: vLLM automatically handles many optimizations including:
# - Paged Attention for efficient memory usage
# - Continuous batching for higher throughput
# - PagedAttention for efficient memory management
# - Kernel fusion for faster inference