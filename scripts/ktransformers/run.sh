#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
KTRANSFORMERS_PATH=~/tools/ktransformers

# System-specific optimizations for Xeon + dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export OMP_NUM_THREADS=$(nproc)  # Use all CPU threads
export KOTLIN_OPTS="-Xmx64g"     # Use half of system RAM for JVM

# Navigate to ktransformers directory
cd $KTRANSFORMERS_PATH

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Error: Java is required but not found"
    echo "Please install Java 11 or higher"
    exit 1
fi

# Check if Gradle is available
if [ -f "./gradlew" ]; then
    GRADLE_CMD="./gradlew"
elif command -v gradle &> /dev/null; then
    GRADLE_CMD="gradle"
else
    echo "Error: Gradle not found. Please ensure either:"
    echo "1. gradlew exists in the ktransformers directory"
    echo "2. or gradle is installed system-wide"
    exit 1
fi

# Check if the build file exists
if [ ! -f "build.gradle.kts" ] && [ ! -f "build.gradle" ]; then
    echo "Error: No Gradle build file found"
    echo "Expected either build.gradle.kts or build.gradle"
    echo "Current location: $(pwd)"
    exit 1
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists and the path is correct"
    exit 1
fi

# Run the model with optimized parameters
$GRADLE_CMD run --args=" \
    --model $MODEL_PATH \
    --gpu-layers -1 \
    --num-gpus 2 \
    --batch-size 32 \
    --context-size 4096 \
    --memory-map true \
    --parallel-processing true \
    --precision float16 \
    $@"

# Note: ktransformers specific optimizations:
# - Uses Kotlin coroutines for parallel processing
# - JVM memory management optimized for large models
# - GPU memory management through CUDA bindings
# - Automatic batching for inference