#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
EXLLAMA_PATH=~/tools/exllamav2

# System-specific optimizations for dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize CUDA memory allocation

# Navigate to exllama directory
cd $EXLLAMA_PATH

# Run the model with optimized parameters
python3 example.py \
    --model $MODEL_PATH \
    --max_seq_len 4096 \
    --gpu-split auto \  # Automatically split across GPUs
    --tensor-parallel \  # Enable tensor parallelism
    --compress-pos-emb 4 \  # Compress positional embeddings for memory efficiency
    --alpha_value 1 \  # Default value for sampling temperature
    --rope-scaling dynamic \  # Dynamic rope scaling for better long context handling
    "$@"  # Pass any additional arguments

# Note: ExLlamaV2 automatically handles most GPU memory optimizations
# The script assumes example.py exists in the exllamav2 directory
# Adjust parameters based on specific model requirements and system performance