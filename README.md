This is a collection of scripts to optimize running GGUF models on a Xeon processor server with 128GB system RAM and dual 3090 GPUs.

The idea is we offer simple reliable ways to generate inference on GGUF models loadbalancing the model to the maximum resources available and max context window given the system's hardware.

Model Path:
~/models/Qwen3-30B-A3B-Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf

# Supported Inference Engines

The following inference engines are supported, with optimized configurations for the target hardware:

1. **llama.cpp**
   - Automatic GPU layer splitting
   - Multi-GPU support with parallel processing
   - Optimized batch size for dual 3090s
   - Full CPU thread utilization

2. **ExLlamaV2**
   - Tensor parallelism for dual GPUs
   - Dynamic rope scaling
   - Compressed positional embeddings
   - Automatic GPU memory optimization

3. **vLLM**
   - Tensor parallel inference across GPUs
   - PagedAttention for efficient memory usage
   - AWQ quantization support
   - Continuous batching for higher throughput

4. **KTransformers**
   - JVM memory optimization
   - Parallel processing with Kotlin coroutines
   - GPU memory management via CUDA
   - Float16 precision support

5. **ik_llama.cpp**
   - Enhanced kernel optimizations
   - Improved memory management
   - Dynamic rope scaling
   - Even tensor split across GPUs

All inference engines are compiled from source from latest GitHub repos and reside in ~/tools.

# Usage

The repository provides a unified interface to run inference using any of the supported engines:

```bash
./run_inference.sh [engine] [additional arguments]
```

Available engine options:
- llama   (llama.cpp)
- exllama (ExLlamaV2)
- vllm    (vLLM)
- ktrans  (KTransformers)
- ikllama (ik_llama.cpp)

Example:
```bash
# Run inference using llama.cpp
./run_inference.sh llama --prompt "Hello, world!"

# Run inference using vLLM
./run_inference.sh vllm --prompt "Hello, world!"
```

Each engine's script is optimized for:
- Dual NVIDIA 3090 GPUs
- Xeon processor utilization
- 128GB system RAM
- Ubuntu latest

The scripts automatically handle:
- GPU device selection
- Memory optimization
- Thread allocation
- Batch size configuration
- Model loading and inference

This repo exists in ~/tools/interference
