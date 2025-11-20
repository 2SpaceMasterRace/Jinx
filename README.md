# Introduction:

This project is designed for systems engineers who want to understand how LLMs work. Since most of the open source projects that serve LLMs are highly optimized with CUDA kernels and other low-level optimizations, it is not easy to understand the whole picture by looking at a codebase of 100k lines of code. 

Therefore, I decided to implement an LLM serving project from scratch -- with only matrix manipulations APIs, so that I can understand what it takes to load those LLM model parameters and do the math magic to generate text. You can think of this course as an JAX version of Tiny LLM - LLM Serving in a Week.

# Prerequisites

You should have some experience with the basics of deep learning and have some idea of how PyTorch works. Some recommended resources are:

    CMU Intro to Machine Learning -- this course teaches you the basics of machine learning
    CMU Deep Learning Systems -- this course teaches you how to build PyTorch from scratch

# Environment Setup

This project uses JAX, a library for array-oriented numerical computation (à la NumPy), with automatic differentiation and JIT compilation to enable high-performance machine learning research. In theory you can also do this course with PyTorch or numpy, but we just don't have the test infra to support them. We test your implementation against PyTorch's CPU implementation and MLX's implementation to ensure correctness.

This course is divided into 3 weeks. We will serve the Qwen2-7B-Instruct model and optimize it throughout the course.

# Roadmap
    Week 1: serve Qwen2 with purely matrix manipulation APIs. Just Python.
    Week 2: optimizations, implement C++/Metal custom kernels to make the model run faster.
    Week 3: more optimizations, batch the requests to serve the model with high throughput. try to serve gpt-oss and deepseek and other oss models using modal prolly


