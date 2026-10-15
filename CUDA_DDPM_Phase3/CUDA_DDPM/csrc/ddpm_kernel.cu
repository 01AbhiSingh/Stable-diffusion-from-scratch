#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

// One CUDA thread computes one pixel of x_t.
// x_t[i] = sqrt(alpha_bar[t_b]) * x0[i]
//        + sqrt(1 - alpha_bar[t_b]) * epsilon[i]
// b = image index corresponding to flattened pixel i.
__global__ void ddpm_forward_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ epsilon,
    const int64_t* __restrict__ timesteps,
    const float* __restrict__ sqrt_alpha_bar,
    const float* __restrict__ sqrt_one_minus_alpha_bar,
    float* __restrict__ output,
    int64_t elements,
    int64_t pixels_per_image) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;

    const int64_t batch_idx = i / pixels_per_image;
    const int64_t timestep = timesteps[batch_idx];

    const float a = sqrt_alpha_bar[timestep];
    const float b = sqrt_one_minus_alpha_bar[timestep];
    output[i] = a * x0[i] + b * epsilon[i];
}

void ddpm_forward_cuda(
    const at::Tensor& x0,
    const at::Tensor& epsilon,
    const at::Tensor& timesteps,
    const at::Tensor& sqrt_alpha_bar,
    const at::Tensor& sqrt_one_minus_alpha_bar,
    at::Tensor& output) {

    const c10::cuda::CUDAGuard device_guard(x0.device());
    const int64_t elements = x0.numel();
    const int64_t pixels_per_image = x0.numel() / x0.size(0);

    constexpr int threads = 256;
    const int64_t block_count = (elements + threads - 1) / threads;
    TORCH_CHECK(block_count <= 2147483647LL, "tensor too large for this kernel launch");
    const dim3 blocks(static_cast<unsigned int>(block_count));

    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    ddpm_forward_kernel<<<blocks, threads, 0, stream>>>(
        x0.data_ptr<float>(),
        epsilon.data_ptr<float>(),
        timesteps.data_ptr<int64_t>(),
        sqrt_alpha_bar.data_ptr<float>(),
        sqrt_one_minus_alpha_bar.data_ptr<float>(),
        output.data_ptr<float>(),
        elements,
        pixels_per_image);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
