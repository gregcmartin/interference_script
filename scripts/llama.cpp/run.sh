#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
LLAMA_PATH=~/tools/llama.cpp

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads
export BLAS_NUM_THREADS=$(nproc)

# Navigate to llama.cpp directory
cd $LLAMA_PATH

# Run the model with optimized parameters
# --n-gpu-layers -1     : Automatically determine optimal layer split between GPU/CPU
# --threads $(nproc)    : Use all available CPU threads
# --ctx-size 4096       : Default context size, adjust based on requirements
# --batch-size 512      : Larger batch size for better GPU utilization
# --parallel 2          : Utilize both GPUs
./main \
    --model $MODEL_PATH \
    --n-gpu-layers -1 \
    --threads $(nproc) \
    --ctx-size 4096 \
    --batch-size 512 \
    --parallel 2 \
    --temp 0.7 \
    --repeat_penalty 1.1 \
    "$@"  # Pass any additional arguments