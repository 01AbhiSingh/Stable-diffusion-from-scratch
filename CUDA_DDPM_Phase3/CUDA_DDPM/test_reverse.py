import torch
import ddpm_cuda


# ============================================================
# 1. CREATE DDPM SCHEDULER
# ============================================================

def create_scheduler(T=1000):

    device = "cuda"

    betas = torch.linspace(
        1e-4,
        0.02,
        T,
        device=device,
        dtype=torch.float32
    )

    alphas = 1.0 - betas

    alpha_bar = torch.cumprod(alphas, dim=0)

    alpha_bar_prev = torch.cat([
        torch.ones(1, device=device),
        alpha_bar[:-1]
    ])

    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar
    )

    sqrt_recip_alpha = torch.rsqrt(alphas)

    posterior_variance = (
        betas
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar)
    )

    return (
        betas,
        sqrt_one_minus_alpha_bar,
        sqrt_recip_alpha,
        posterior_variance
    )


# ============================================================
# 2. PYTORCH REFERENCE IMPLEMENTATION
# ============================================================

def pytorch_reverse(
    x_t,
    epsilon_pred,
    noise,
    betas,
    sqrt_one_minus_alpha_bar,
    sqrt_recip_alpha,
    posterior_variance,
    timestep
):

    beta = betas[timestep]

    scale = sqrt_one_minus_alpha_bar[timestep]

    mean = sqrt_recip_alpha[timestep] * (
        x_t - (beta / scale) * epsilon_pred
    )

    if timestep == 0:
        return mean

    sigma = torch.sqrt(
        posterior_variance[timestep]
    )

    return mean + sigma * noise


# ============================================================
# 3. TEST CUDA AGAINST PYTORCH
# ============================================================

def run_test(shape, timestep, scheduler):

    x_t = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32
    )

    epsilon_pred = torch.randn_like(x_t)

    noise = torch.randn_like(x_t)

    betas, scale, recip_alpha, variance = scheduler

    # PyTorch reference
    expected = pytorch_reverse(
        x_t,
        epsilon_pred,
        noise,
        betas,
        scale,
        recip_alpha,
        variance,
        timestep
    )

    # Our CUDA implementation
    result = ddpm_cuda.reverse(
        x_t,
        epsilon_pred,
        noise,
        betas,
        scale,
        recip_alpha,
        variance,
        timestep
    )

    # Compare outputs
    torch.testing.assert_close(
        result,
        expected,
        atol=3e-6,
        rtol=1e-5
    )

    max_error = (
        result - expected
    ).abs().max().item()

    print(
        f"PASS timestep={timestep:3d} "
        f"shape={shape} "
        f"max_error={max_error:.3e}"
    )

    # At timestep zero, noise must not affect the result
    if timestep == 0:

        different_noise = torch.randn_like(x_t)

        result2 = ddpm_cuda.reverse(
            x_t,
            epsilon_pred,
            different_noise,
            betas,
            scale,
            recip_alpha,
            variance,
            timestep
        )

        assert torch.equal(result, result2), (
            "Noise incorrectly affects timestep 0"
        )


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":

    assert torch.cuda.is_available()

    torch.manual_seed(42)

    print("GPU:", torch.cuda.get_device_name(0))

    scheduler = create_scheduler()

    shapes = [
        (1, 1, 32, 32),
        (16, 1, 32, 32),
        (128, 1, 32, 32),
        (7, 3, 33, 35)
    ]

    timesteps = [
        999,
        500,
        100,
        1,
        0
    ]

    for shape in shapes:
        for timestep in timesteps:
            run_test(shape, timestep, scheduler)

    print("\nAll CUDA reverse diffusion tests passed!")
