#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
EXLLAMA_PATH=~/tools/exllamav2

# System-specific optimizations for dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize CUDA memory allocation

# Navigate to exllama directory
cd $EXLLAMA_PATH

# Check for required Python files
if [ ! -f "example.py" ] && [ ! -f "exllama.py" ]; then
    echo "Error: Could not find ExLlamaV2 Python scripts."
    echo "Please ensure ExLlamaV2 is properly installed and you're in the correct directory:"
    echo "Current location: $(pwd)"
    echo "Expected files:"
    echo "  - example.py"
    echo "  - exllama.py"
    exit 1
fi

# Determine which script to use
if [ -f "example.py" ]; then
    SCRIPT="example.py"
else
    SCRIPT="exllama.py"
fi

# Check if Python and required packages are available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not found"
    exit 1
fi

# Run the model with optimized parameters
python3 $SCRIPT \
    --model $MODEL_PATH \
    --max_seq_len 4096 \
    --gpu-split auto \
    --tensor-parallel \
    --compress-pos-emb 4 \
    --alpha_value 1 \
    --rope-scaling dynamic \
    "$@"  # Pass any additional arguments

# Note: ExLlamaV2 automatically handles most GPU memory optimizations