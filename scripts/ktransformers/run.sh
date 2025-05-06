#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
KTRANSFORMERS_PATH=~/tools/ktransformers

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads
export KOTLIN_OPTS="-Xmx64g"     # Use half of system RAM for JVM

# Navigate to ktransformers directory
cd $KTRANSFORMERS_PATH

# Run the model with optimized parameters
./gradlew run --args=" \
    --model $MODEL_PATH \
    --gpu-layers -1 \           # Auto-determine GPU layer split
    --num-gpus 2 \             # Use both GPUs
    --batch-size 32 \          # Adjust based on memory
    --context-size 4096 \      # Default context size
    --memory-map true \        # Memory mapping for large models
    --parallel-processing true \
    --precision float16 \      # Use half precision
    $@"                        # Pass additional arguments

# Note: ktransformers specific optimizations:
# - Uses Kotlin coroutines for parallel processing
# - JVM memory management optimized for large models
# - GPU memory management through CUDA bindings
# - Automatic batching for inference