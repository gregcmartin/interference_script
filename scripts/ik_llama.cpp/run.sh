#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
IK_LLAMA_PATH=~/tools/ik_llama.cpp

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads
export BLAS_NUM_THREADS=$(nproc)
export IK_LLAMA_TURBO=1         # Enable turbo mode for faster inference

# Navigate to ik_llama.cpp directory
cd $IK_LLAMA_PATH

# Run the model with optimized parameters
./main \
    --model $MODEL_PATH \
    --n-gpu-layers -1 \     # Auto-determine GPU layer split
    --threads $(nproc) \    # Use all CPU threads
    --ctx-size 4096 \       # Default context size
    --batch-size 512 \      # Large batch size for GPU utilization
    --parallel 2 \          # Use both GPUs
    --memory-f32 \         # Use FP32 for memory keys
    --mlock \              # Lock memory to prevent swapping
    --mul-mat-q \          # Quantized matrix multiplication
    --tensor-split 0.5,0.5 \ # Even split between GPUs
    --rope-scaling dynamic \ # Dynamic rope scaling
    "$@"  # Pass any additional arguments

# Note: ik_llama.cpp specific features:
# - Improved kernel optimizations
# - Better memory management
# - Enhanced parallel processing
# - Optimized matrix operations