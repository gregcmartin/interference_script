#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
KTRANS_PATH=~/tools/ktransformers
VENV_PATH=$KTRANS_PATH/venv

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please ensure the virtual environment is properly set up"
    exit 1
fi

# Check if ktransformers module exists
if [ ! -d "$VENV_PATH/lib/python"*/site-packages/ktransformers ]; then
    echo "Error: KTransformers module not found in virtual environment"
    echo "Please ensure KTransformers is properly installed in the virtual environment"
    exit 1
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists and the path is correct"
    exit 1
fi

# Activate virtual environment and run KTransformers
source "$VENV_PATH/bin/activate"

# Run KTransformers using the built-in local_chat.py script
python3 -m ktransformers.local_chat \
    --model $MODEL_PATH \
    --gpu-layers -1 \
    --num-gpus 2 \
    --batch-size 32 \
    --context-size 4096 \
    --memory-map true \
    --parallel-processing true \
    --precision float16 \
    "$@"  # Pass any additional arguments

# Deactivate virtual environment
deactivate