#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
IK_LLAMA_PATH=~/tools/ik_llama.cpp
EXECUTABLE=$IK_LLAMA_PATH/build/bin/llama-cli

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads
export BLAS_NUM_THREADS=$(nproc)
export IK_LLAMA_TURBO=1         # Enable turbo mode for faster inference

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: llama-cli executable not found at $EXECUTABLE"
    echo "Please ensure ik_llama.cpp is properly compiled"
    echo "Available binaries in build/bin:"
    ls $IK_LLAMA_PATH/build/bin/
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

# Run the model with optimized parameters
$EXECUTABLE \
    --model $MODEL_PATH \
    --n-gpu-layers -1 \
    --threads $(nproc) \
    --ctx-size 4096 \
    --batch-size 512 \
    --parallel 2 \
    --memory-f32 \
    --mlock \
    --mul-mat-q \
    --tensor-split 0.5,0.5 \
    --rope-scaling dynamic \
    "$@"  # Pass any additional arguments