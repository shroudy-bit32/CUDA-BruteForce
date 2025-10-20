# CUDA-BruteForce

A highly optimized and parallelized CUDA implementation of a brute-force password cracker using SHA-256 hashing.  
This program leverages GPU power to explore large password spaces at high speed.

## 🔧 Build Instructions

To compile the optimized and working version of the CUDA brute-force implementation, use the following `nvcc` command:

```bash
nvcc -O3 -gencode arch=compute_89,code=sm_89 -Xptxas -v --use_fast_math --maxrregcount=28 kernel.cu -o test.exe
```

> ⚠️ **Note:** Make sure to adjust the architecture flags (`arch=compute_XX,code=sm_XX`) based on your GPU model.  
> This ensures compatibility and optimal performance.

## ✅ Common GPU Architectures

- **Turing** → `arch=compute_75,code=sm_75`  
  _(e.g., RTX 2060 / 2070 / 2080)_

- **Ampere** → `arch=compute_80,code=sm_80` or `arch=compute_86,code=sm_86`  
  _(e.g., RTX 3060 / 3070 / 3080 / 3090)_

- **Ada (Lovelace)** → `arch=compute_89,code=sm_89`  
  _(e.g., RTX 4070 / 4080 / 4090)_

## 📦 Requirements

- **CUDA Toolkit** 11.8 or higher (recommended for Ada GPUs)
- **Visual Studio** (on Windows) or `nvcc` CLI on Linux
- **GPU with CUDA Compute Capability 7.5+**
- Proper NVIDIA GPU driver installed

## Features

- Password generation using configurable character sets
- Constant memory usage for target hash and charset
- Fully parallel brute-force evaluation with per-thread hashing
- Register-optimized kernel with `--maxrregcount` to avoid spills
- Host-device mapped flags for fast communication

## Benchmark

> Example performance on RTX 4080 Laptop (charSet length = 32, password length = 7):

- ⏱️ ~6.4 seconds to find correct password
- 👷‍♂️ 29696 threads used per chunk
- 🚀 Achieves high occupancy with register-limited kernel

  ## 📄 License

This project is licensed under the MIT License.  
See the full license text in the [LICENSE](LICENSE) file.
