#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
VLLM_PATH=~/tools/vllm

# System-specific optimizations for dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export CUDA_LAUNCH_BLOCKING=0    # Async CUDA operations
export NCCL_P2P_DISABLE=0       # Enable GPU P2P communication

# Navigate to vllm directory
cd $VLLM_PATH

# Check if Python and vLLM are available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not found"
    exit 1
fi

# Check if vLLM module is installed
if ! python3 -c "import vllm" &> /dev/null; then
    echo "Error: vLLM Python module not found"
    echo "Please ensure vLLM is properly installed:"
    echo "pip install vllm"
    exit 1
fi

# Check if CUDA is available
if ! python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo "Error: CUDA is not available for PyTorch"
    echo "Please ensure CUDA and PyTorch with CUDA support are properly installed"
    exit 1
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists and the path is correct"
    exit 1
fi

# Run the model with optimized parameters
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --quantization awq \
    --dtype float16 \
    --trust-remote-code \
    --port 8000 \
    "$@"  # Pass any additional arguments

# Note: vLLM automatically handles many optimizations including:
# - Paged Attention for efficient memory usage
# - Continuous batching for higher throughput
# - PagedAttention for efficient memory management
# - Kernel fusion for faster inference