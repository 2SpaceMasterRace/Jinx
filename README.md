# Introduction:

Jinx is a high-performance LLM inference and serving framework for large language models. 

## Background
This is a toy project to understand LLM inference from a systems point of view. It's a lightweight mix of vLLM and SGLang. Honestly this is a JAX version of skyzh's [Tiny LLM - LLM Serving in a Week](https://skyzh.github.io/tiny-llm/). I'll also probably build my own model if I got the time and host it to completely understand the machine learning systems stack.

# Specs
Inspired by John Carmack's .plan files.

**TODO**: Now **[0/3]**
- **TODO** Model Implementation **[0/7]**
  - [ ] Attention
  - [ ] RoPE
  - [ ] Grouped Query Attention
  - [ ] RMSNorm and MLP
  - [ ] Load the Model
  - [ ] Generate Responses (aka Decoding)
  - [ ] Sampling
- **TODO** Inference System **[0/7]**
  - [ ] Key-Value Cache
  - [ ] Continuous Batching
  - [ ] Chunked Prefill
  - [ ] Quantized Matmul and Linear - CPU
  - [ ] Quantized Matmul and Linear - GPU
  - [ ] Flash Attention 2 - CPU
  - [ ] Flash Attention 2 - GPU
- **TODO** Advanced Features I **[0/7]**
  - [ ] Paged Attention 
  - [ ] MoE (Mixture of Experts)
  - [ ] Speculative Decoding
  - [ ] RAG Pipeline
  - [ ] AI Agent / Tool Calling
  - [ ] Long Context

**TODO**: Later **[0/4]**
- **TODO** Optimization Suite **[0/6]**
  - [ ] Overlap Scheduling
  - [ ] Tensor Parallelism
  - [ ] JIT CUDA kernels
  - [ ] Torch compilation
  - [ ] CUDA graph
  - [ ] Prefix caching
- **TODO** TVM-FFI Integration **[0/4]**
  - [ ] Custom CUDA kernels
  - [ ] Communication primitives
  - [ ] Symbolic tensor matching
  - [ ] PDL kernel launches
- **TODO** Advanced Features II **[0/4]**
  - [ ] Quantized/compressed KV cache
  - [ ] Prefix/prompt cache
  - [ ] Fine tuning support
  - [ ] Smaller kernels (softmax, silu, etc)
- **TODO** Model Deployment **[0/3]**
  - [ ] Serve OSS models (GPT, DeepSeek) via Modal
  - [ ] Online and offline serving modes
  - [ ] Streaming output

**HOLD** Hardware Support **[0/4]**
- [ ] NVIDIA GPUs (GB200/B300/H100/A100/Spark)
- [ ] AMD GPUs (MI355/MI300)
- [ ] Apple Silicon (M2+) 
- [ ] Google TPUs

## Archive



# Installation 
### Model Download 
I'll be using the smaller Qwen2-0.5B-Instruct model and maybe if I get access to compute in the future, I'll use the bigger Qwen2-7B-Instruct model. You'll need the huggingface-cli for this as the model parameters are hosted there.

```bash
# On macOS and Linux:
> curl -LsSf https://hf.co/cli/install.sh | bash

# Once installed, you can check that the CLI is correctly set up: 
> hf --help

# After authenticating your cli, download the parameters:
> huggingface-cli login
> huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX
> huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
```

# Quick Start 
WIP. Will be publishing as  package.

# Benchmarks
See bench.py for benchmarks.

**Test Configuration**:
- **Hardware:** Apple M4 (16GB)
- **Model:** Qwen2-0.5B-Instruct
- **Total Requests:** 256 sequences
- **Input Length:** Randomly sampled between 100–1024 tokens
- **Output Length:** Randomly sampled between 100–1024 tokens


# References
[1] Blog | LMSYS Org  
    https://lmsys.org/blog/

[2] vLLM (vllm-project/vllm)  
    https://github.com/vllm-project/vllm

[3] Nano vLLM (GeeeekExplorer/nano-vllm)  
    https://github.com/GeeeekExplorer/nano-vllm

[4] mini-sglang (sgl-project/mini-sglang): A compact implementation of SGLang  
    https://github.com/sgl-project/mini-sglang

[5] SGLang (sgl-project/sglang): Fast serving framework for large language models  
    https://github.com/sgl-project/sglang

[6] LightLLM (ModelTC/LightLLM): Lightweight Python-based LLM inference and serving  
    https://github.com/ModelTC/lightllm

[7] FlashInfer (flashinfer-ai/flashinfer): Kernel library for LLM serving  
    https://github.com/flashinfer-ai/flashinfer