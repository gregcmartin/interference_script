#!/bin/bash

MODEL_PATH=~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf
EXLLAMA_PATH=~/tools/exllamav2
VENV_PATH=$EXLLAMA_PATH/venv

# System-specific optimizations for dual 3090s
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize CUDA memory allocation

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please ensure the virtual environment is properly set up"
    exit 1
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists and the path is correct"
    exit 1
fi

# Create a temporary Python script that uses the actual module structure
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << 'EOF'
from exllamav2.model import ExLlamaV2
from exllamav2.generator import ExLlamaV2StreamingGenerator
from exllamav2.config import ExLlamaV2Config
import sys

def main():
    # Initialize model configuration
    config = ExLlamaV2Config()
    config.model_path = sys.argv[1]
    config.max_seq_len = 4096
    config.gpu_split = "auto"
    config.tensor_parallel = True
    config.compress_pos_emb = 4
    config.alpha_value = 1
    config.rope_scaling = "dynamic"

    # Load the model
    model = ExLlamaV2(config)

    # Create generator
    generator = ExLlamaV2StreamingGenerator(model)
    generator.settings.temperature = 0.7
    generator.settings.top_p = 0.9
    generator.settings.top_k = 50

    # Get input prompt
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello, how are you?"

    # Generate response
    output_text = ""
    for chunk in generator.generate_simple(prompt, max_new_tokens=256):
        print(chunk, end="", flush=True)
        output_text += chunk
    print()  # Final newline

if __name__ == "__main__":
    main()
EOF

# Activate virtual environment and run ExLlamaV2
source "$VENV_PATH/bin/activate"
python3 "$TMP_SCRIPT" "$MODEL_PATH" "$@"
deactivate

# Cleanup
rm "$TMP_SCRIPT"