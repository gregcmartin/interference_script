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

# Find the main executable (check various possible locations)
POSSIBLE_LOCATIONS=(
    "./main"
    "./build/bin/main"
    "./build/main"
    "./ik_llama"
    "./build/bin/ik_llama"
    "./build/ik_llama"
)

EXECUTABLE=""
for loc in "${POSSIBLE_LOCATIONS[@]}"; do
    if [ -f "$loc" ]; then
        EXECUTABLE="$loc"
        break
    fi
done

if [ -z "$EXECUTABLE" ]; then
    echo "Error: Could not find ik_llama.cpp executable."
    echo "Please ensure ik_llama.cpp is properly compiled."
    echo "Checked locations:"
    for loc in "${POSSIBLE_LOCATIONS[@]}"; do
        echo "  - $loc"
    done
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

# Note: ik_llama.cpp specific features:
# - Improved kernel optimizations
# - Better memory management
# - Enhanced parallel processing
# - Optimized matrix operations