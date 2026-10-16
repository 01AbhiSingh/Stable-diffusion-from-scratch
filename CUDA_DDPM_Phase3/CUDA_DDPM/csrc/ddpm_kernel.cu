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

// ============================================================
// DDPM REVERSE DIFFUSION KERNEL
// ============================================================
//
// mean = (1 / sqrt(alpha_t)) *
//        (x_t - (beta_t / sqrt(1-alpha_bar_t)) * eps_pred)
//
// t > 0:
//     x_prev = mean + sqrt(posterior_variance_t) * noise
//
// t == 0:
//     x_prev = mean
//
// One CUDA thread processes one tensor element.
// All images in the batch share the same reverse timestep.
// ============================================================

__global__ void ddpm_reverse_kernel(
    const float* __restrict__ x_t,
    const float* __restrict__ epsilon_pred,
    const float* __restrict__ noise,

    const float* __restrict__ betas,
    const float* __restrict__ sqrt_one_minus_alpha_bar,
    const float* __restrict__ sqrt_recip_alpha,
    const float* __restrict__ posterior_variance,

    float* __restrict__ output,

    int64_t timestep,
    int64_t elements
) {
    // --------------------------------------------------------
    // 1. Global thread index
    // --------------------------------------------------------

    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i >= elements)
        return;

    // --------------------------------------------------------
    // 2. Get scheduler coefficients
    // --------------------------------------------------------

    const float beta = betas[timestep];

    const float scale =
        sqrt_one_minus_alpha_bar[timestep];

    const float recip_alpha =
        sqrt_recip_alpha[timestep];

    // --------------------------------------------------------
    // 3. Predict reverse mean
    // --------------------------------------------------------

    const float mean =
        recip_alpha *
        (
            x_t[i] -
            (beta / scale) * epsilon_pred[i]
        );

    // --------------------------------------------------------
    // 4. Add posterior noise when timestep > 0
    // --------------------------------------------------------

    if (timestep > 0) {

        const float sigma =
            sqrtf(posterior_variance[timestep]);

        output[i] = mean + sigma * noise[i];

    } else {

        // Final reverse step: no additional noise
        output[i] = mean;
    }
}
// ============================================================
// REVERSE CUDA KERNEL LAUNCHER
// ============================================================

void ddpm_reverse_cuda(
    const at::Tensor& x_t,
    const at::Tensor& epsilon_pred,
    const at::Tensor& noise,

    const at::Tensor& betas,
    const at::Tensor& sqrt_one_minus_alpha_bar,
    const at::Tensor& sqrt_recip_alpha,
    const at::Tensor& posterior_variance,

    int64_t timestep,
    at::Tensor& output
) {
    // Ensure execution uses the correct CUDA device
    const c10::cuda::CUDAGuard device_guard(x_t.device());

    // --------------------------------------------------------
    // Total number of elements
    // --------------------------------------------------------

    const int64_t elements = x_t.numel();

    // --------------------------------------------------------
    // Configure CUDA execution
    // --------------------------------------------------------

    constexpr int threads = 256;

    const int64_t block_count =
        (elements + threads - 1) / threads;

    TORCH_CHECK(
        block_count <= 2147483647LL,
        "Tensor too large for CUDA kernel launch"
    );

    const dim3 blocks(
        static_cast<unsigned int>(block_count)
    );

    // --------------------------------------------------------
    // Use PyTorch's current CUDA stream
    // --------------------------------------------------------

    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream().stream();

    // --------------------------------------------------------
    // Launch reverse diffusion kernel
    // --------------------------------------------------------

    ddpm_reverse_kernel<<<blocks, threads, 0, stream>>>(
        x_t.data_ptr<float>(),
        epsilon_pred.data_ptr<float>(),
        noise.data_ptr<float>(),

        betas.data_ptr<float>(),
        sqrt_one_minus_alpha_bar.data_ptr<float>(),
        sqrt_recip_alpha.data_ptr<float>(),
        posterior_variance.data_ptr<float>(),

        output.data_ptr<float>(),

        timestep,
        elements
    );

    // Check kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
