# Phase 3: DDPM CUDA — Kernel 1 (Forward Diffusion)

Standalone educational CUDA extension for PyTorch 2.7 + CUDA 12.8, built for NVIDIA RTX 5080 (SM 12.0). Does **not** modify BruteForce_DDPM or Optimized_DDPM.

## Check the toolchain

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))'
which nvcc
nvcc --version
```

You need NVIDIA CUDA Toolkit `nvcc` compatible with the CUDA version used by PyTorch, a suitable host C++ compiler, Python build tools, and the same Python environment where PyTorch is installed. CUDA 12.8 is the intended configuration.

## Build

```bash
cd ~/Diffusion/CUDA_DDPM
export TORCH_CUDA_ARCH_LIST="12.0"
MAX_JOBS=4 python3 setup.py build_ext --inplace
```

Do not build under the Python interpreter outside your `(rail3d)` environment.

## Test

```bash
python3 test_kernel.py
```

## Benchmark

```bash
python3 benchmark.py --batch-size 128 --warmup 100 --repeats 1000
```

## Files

- `csrc/ddpm.cpp`: Python binding + input validation and output allocation.
- `csrc/ddpm_kernel.cu`: Actual CUDA kernel + kernel launcher.
- `setup.py`: Compiles `.cpp` and `.cu` into `ddpm_cuda*.so`.
- `test_kernel.py`: Compares exact same noise/timesteps/schedule against PyTorch for correctness.
- `benchmark.py`: Times the reference PyTorch expression vs a fused CUDA kernel, with CUDA events.
- `integration_example.py`: Optional compatible wrapper for Phase 2 scheduler API.

## Mathematics

`x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon`

Input x0 and epsilon: contiguous float32 CUDA tensors `[B,C,H,W]`; `t`: int64 CUDA `[B]`; schedule arrays: float32 CUDA `[T]`. This extension returns `x_t`. The Python wrapper samples epsilon and returns `(x_t, epsilon)` to match the Phase 2 API.

No CUDA backward is implemented: in DDPM training, inputs to this diffusion step normally do not require gradients, so training learns through the U-Net prediction loss. Do not use this extension as a differentiable custom autograd op without adding backward.

## Practical notes

- Verify `0 <= t[b] < T`; the kernel expects valid indices.
- GPU floating-point results can differ from PyTorch by small rounding amounts, especially due to fused multiply-add operations.
- Even if the fused kernel is faster, forward diffusion takes a tiny fraction of DDPM training time compared with the U-Net.
- This is Phase 3's learning exercise, not a replacement for U-Net convolutions or the entire sampling/training system.
- Do not turn on `--use_fast_math` while validating numerical parity.
