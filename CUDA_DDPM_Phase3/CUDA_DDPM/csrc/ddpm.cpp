#include <torch/extension.h>

// Implemented in ddpm_kernel.cu.
void ddpm_forward_cuda(
    const at::Tensor& x0,
    const at::Tensor& epsilon,
    const at::Tensor& timesteps,
    const at::Tensor& sqrt_alpha_bar,
    const at::Tensor& sqrt_one_minus_alpha_bar,
    at::Tensor& output);

// Implemented in ddpm_kernel.cu
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
);

at::Tensor forward(
    const at::Tensor& x0,
    const at::Tensor& epsilon,
    const at::Tensor& timesteps,
    const at::Tensor& sqrt_alpha_bar,
    const at::Tensor& sqrt_one_minus_alpha_bar) {

    TORCH_CHECK(x0.is_cuda(), "x0 must be a CUDA tensor");
    TORCH_CHECK(x0.scalar_type() == at::kFloat, "x0 must have dtype float32");
    TORCH_CHECK(x0.dim() == 4, "x0 must be [B, C, H, W]");
    TORCH_CHECK(x0.is_contiguous(), "x0 must be contiguous");
    TORCH_CHECK(x0.numel() > 0, "x0 must not be empty");

    TORCH_CHECK(epsilon.device() == x0.device(), "epsilon must be on the same GPU");
    TORCH_CHECK(epsilon.scalar_type() == at::kFloat, "epsilon must be float32");
    TORCH_CHECK(epsilon.sizes() == x0.sizes(), "epsilon must match x0's shape");
    TORCH_CHECK(epsilon.is_contiguous(), "epsilon must be contiguous");

    TORCH_CHECK(timesteps.device() == x0.device(), "timesteps must be on the same GPU");
    TORCH_CHECK(timesteps.scalar_type() == at::kLong, "timesteps must have dtype int64");
    TORCH_CHECK(timesteps.dim() == 1 && timesteps.size(0) == x0.size(0),
                "timesteps must have shape [B]");
    TORCH_CHECK(timesteps.is_contiguous(), "timesteps must be contiguous");

    TORCH_CHECK(sqrt_alpha_bar.device() == x0.device(), "sqrt_alpha_bar must be on the same GPU");
    TORCH_CHECK(sqrt_one_minus_alpha_bar.device() == x0.device(),
                "sqrt_one_minus_alpha_bar must be on the same GPU");
    TORCH_CHECK(sqrt_alpha_bar.scalar_type() == at::kFloat &&
                sqrt_one_minus_alpha_bar.scalar_type() == at::kFloat,
                "schedule arrays must be float32");
    TORCH_CHECK(sqrt_alpha_bar.dim() == 1 && sqrt_one_minus_alpha_bar.dim() == 1 &&
                sqrt_alpha_bar.numel() == sqrt_one_minus_alpha_bar.numel() &&
                sqrt_alpha_bar.numel() > 0,
                "schedule arrays must be nonempty 1-D tensors of equal length");
    TORCH_CHECK(sqrt_alpha_bar.is_contiguous() && sqrt_one_minus_alpha_bar.is_contiguous(),
                "schedule arrays must be contiguous");

    // timesteps must be valid indices [0, T-1]. The caller is responsible for
    // ensuring this; a GPU-side range check would add synchronization overhead.
    auto output = at::empty_like(x0);
    ddpm_forward_cuda(x0, epsilon, timesteps, sqrt_alpha_bar,
                      sqrt_one_minus_alpha_bar, output);
    return output;
}

// ============================================================
// DDPM REVERSE DIFFUSION - C++ WRAPPER
// ============================================================

at::Tensor reverse(
    const at::Tensor& x_t,
    const at::Tensor& epsilon_pred,
    const at::Tensor& noise,

    const at::Tensor& betas,
    const at::Tensor& sqrt_one_minus_alpha_bar,
    const at::Tensor& sqrt_recip_alpha,
    const at::Tensor& posterior_variance,

    int64_t timestep
) {
    // --------------------------------------------------------
    // 1. Validate input tensor
    // --------------------------------------------------------

    TORCH_CHECK(
        x_t.is_cuda(),
        "x_t must be a CUDA tensor"
    );

    TORCH_CHECK(
        x_t.scalar_type() == at::kFloat,
        "x_t must be float32"
    );

    TORCH_CHECK(
        x_t.dim() == 4,
        "x_t must have shape [B, C, H, W]"
    );

    TORCH_CHECK(
        x_t.is_contiguous(),
        "x_t must be contiguous"
    );

    TORCH_CHECK(
        x_t.numel() > 0,
        "x_t must not be empty"
    );

    // --------------------------------------------------------
    // 2. Validate epsilon prediction and noise
    // --------------------------------------------------------

    auto check_image = [&](const at::Tensor& tensor,
                           const char* name) {

        TORCH_CHECK(
            tensor.is_cuda(),
            name, " must be CUDA"
        );

        TORCH_CHECK(
            tensor.device() == x_t.device(),
            name, " must be on the same GPU as x_t"
        );

        TORCH_CHECK(
            tensor.scalar_type() == at::kFloat,
            name, " must be float32"
        );

        TORCH_CHECK(
            tensor.sizes() == x_t.sizes(),
            name, " must have the same shape as x_t"
        );

        TORCH_CHECK(
            tensor.is_contiguous(),
            name, " must be contiguous"
        );
    };

    check_image(epsilon_pred, "epsilon_pred");
    check_image(noise, "noise");

    // --------------------------------------------------------
    // 3. Validate scheduler arrays
    // --------------------------------------------------------

    const int64_t T = betas.numel();

    TORCH_CHECK(
        T > 0,
        "betas must not be empty"
    );

    auto check_schedule = [&](const at::Tensor& tensor,
                              const char* name) {

        TORCH_CHECK(
            tensor.is_cuda(),
            name, " must be CUDA"
        );

        TORCH_CHECK(
            tensor.device() == x_t.device(),
            name, " must be on the same GPU as x_t"
        );

        TORCH_CHECK(
            tensor.scalar_type() == at::kFloat,
            name, " must be float32"
        );

        TORCH_CHECK(
            tensor.dim() == 1,
            name, " must be 1-D"
        );

        TORCH_CHECK(
            tensor.numel() == T,
            name, " must have length T"
        );

        TORCH_CHECK(
            tensor.is_contiguous(),
            name, " must be contiguous"
        );
    };

    check_schedule(betas, "betas");

    check_schedule(
        sqrt_one_minus_alpha_bar,
        "sqrt_one_minus_alpha_bar"
    );

    check_schedule(
        sqrt_recip_alpha,
        "sqrt_recip_alpha"
    );

    check_schedule(
        posterior_variance,
        "posterior_variance"
    );

    // --------------------------------------------------------
    // 4. Validate timestep
    // --------------------------------------------------------

    TORCH_CHECK(
        timestep >= 0 && timestep < T,
        "timestep must be between 0 and T-1"
    );

    // --------------------------------------------------------
    // 5. Allocate output tensor
    // --------------------------------------------------------

    auto output = at::empty_like(x_t);

    // --------------------------------------------------------
    // 6. Execute CUDA kernel
    // --------------------------------------------------------

    ddpm_reverse_cuda(
        x_t,
        epsilon_pred,
        noise,

        betas,
        sqrt_one_minus_alpha_bar,
        sqrt_recip_alpha,
        posterior_variance,

        timestep,
        output
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def(
        "forward",
        &forward,
        "DDPM forward diffusion (CUDA, float32)"
    );

    m.def(
        "reverse",
        &reverse,
        "DDPM reverse diffusion (CUDA, float32)"
    );
}
