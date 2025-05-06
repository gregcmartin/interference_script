#!/bin/bash

# Make all inference scripts executable
chmod +x scripts/llama.cpp/run.sh
chmod +x scripts/exllamav2/run.sh
chmod +x scripts/vllm/run.sh
chmod +x scripts/ktransformers/run.sh
chmod +x scripts/ik_llama.cpp/run.sh

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

# Check if engine argument is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Get the engine argument
ENGINE=$1
shift  # Remove the engine argument, leaving remaining args

# Run the appropriate engine
case $ENGINE in
    "llama")
        scripts/llama.cpp/run.sh "$@"
        ;;
    "exllama")
        scripts/exllamav2/run.sh "$@"
        ;;
    "vllm")
        scripts/vllm/run.sh "$@"
        ;;
    "ktrans")
        scripts/ktransformers/run.sh "$@"
        ;;
    "ikllama")
        scripts/ik_llama.cpp/run.sh "$@"
        ;;
    *)
        echo "Unknown engine: $ENGINE"
        show_usage
        exit 1
        ;;
esac