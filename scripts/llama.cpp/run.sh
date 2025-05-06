#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
LLAMA_PATH=~/tools/llama.cpp

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads
export BLAS_NUM_THREADS=$(nproc)

# Navigate to llama.cpp directory
cd $LLAMA_PATH

# Find the main executable (could be main, server, or built with cmake)
if [ -f "./main" ]; then
    EXECUTABLE="./main"
elif [ -f "./build/bin/main" ]; then
    EXECUTABLE="./build/bin/main"
elif [ -f "./build/main" ]; then
    EXECUTABLE="./build/main"
else
    echo "Error: Could not find llama.cpp executable. Please ensure llama.cpp is properly compiled."
    echo "Expected locations checked:"
    echo "  - ./main"
    echo "  - ./build/bin/main"
    echo "  - ./build/main"
    exit 1
fi

# Run the model with optimized parameters
$EXECUTABLE \
    --model $MODEL_PATH \
    --n-gpu-layers -1 \
    --threads $(nproc) \
    --ctx-size 4096 \
    --batch-size 512 \
    --parallel 2 \
    --temp 0.7 \
    --repeat_penalty 1.1 \
    "$@"  # Pass any additional arguments