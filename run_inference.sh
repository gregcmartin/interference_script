#!/bin/bash

# Function to display usage
show_usage() {
    echo "Usage: ./run_inference.sh [engine] [additional arguments]"
    echo "Available engines:"
    echo "  llama      - Run using llama.cpp"
    echo "  exllama    - Run using ExLlamaV2"
    echo "  vllm       - Run using vLLM"
    echo "  ktrans     - Run using KTransformers"
    echo "  ikllama    - Run using ik_llama.cpp"
    echo ""
    echo "Example: ./run_inference.sh llama --prompt 'Hello, world!'"
}

# Function to check and set script permissions
check_permissions() {
    local script_path=$1
    if [ ! -f "$script_path" ]; then
        echo "Error: Script not found at $script_path"
        exit 1
    fi
    
    if [ ! -x "$script_path" ]; then
        echo "Setting executable permission for $script_path"
        chmod +x "$script_path"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to set executable permission for $script_path"
            exit 1
        fi
    fi
}

# Check if engine argument is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Get the engine argument
ENGINE=$1
shift  # Remove the engine argument, leaving remaining args

# Verify model path exists
MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists and the path is correct"
fi

# Run the appropriate engine
case $ENGINE in
    "llama")
        SCRIPT="scripts/llama.cpp/run.sh"
        check_permissions "$SCRIPT"
        "$SCRIPT" "$@"
        ;;
    "exllama")
        SCRIPT="scripts/exllamav2/run.sh"
        check_permissions "$SCRIPT"
        "$SCRIPT" "$@"
        ;;
    "vllm")
        SCRIPT="scripts/vllm/run.sh"
        check_permissions "$SCRIPT"
        "$SCRIPT" "$@"
        ;;
    "ktrans")
        SCRIPT="scripts/ktransformers/run.sh"
        check_permissions "$SCRIPT"
        "$SCRIPT" "$@"
        ;;
    "ikllama")
        SCRIPT="scripts/ik_llama.cpp/run.sh"
        check_permissions "$SCRIPT"
        "$SCRIPT" "$@"
        ;;
    *)
        echo "Unknown engine: $ENGINE"
        show_usage
        exit 1
        ;;
esac