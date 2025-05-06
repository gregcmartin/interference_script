#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
VLLM_PATH=~/tools/vllm
VENV_PATH=$VLLM_PATH/venv

# System-specific optimizations for dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export CUDA_LAUNCH_BLOCKING=0    # Async CUDA operations
export NCCL_P2P_DISABLE=0       # Enable GPU P2P communication

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please ensure the virtual environment is properly set up"
    exit 1
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists and the path is correct"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be available."
    echo "This script is optimized for dual NVIDIA 3090 GPUs."
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Run vLLM using the entrypoints module
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

# Deactivate virtual environment
deactivate