"""
DDPM reverse sampling using PyTorch U-Net + CUDA kernel.
"""

import torch
import ddpm_cuda


@torch.no_grad()
def sample_step_cuda(
    model,
    x_t,
    timestep: int,
    scheduler,
    noise=None
):

    # --------------------------------------------------------
    # 1. Validate timestep and device
    # --------------------------------------------------------

    if not (0 <= timestep < scheduler.T):
        raise IndexError(f"Invalid timestep: {timestep}")

    if not x_t.is_cuda:
        raise ValueError("CUDA sampling requires a GPU tensor")

    # --------------------------------------------------------
    # 2. Create timestep tensor for U-Net
    # --------------------------------------------------------

    t = torch.full(
        (x_t.shape[0],),
        timestep,
        device=x_t.device,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # 3. Predict noise using PyTorch U-Net
    # --------------------------------------------------------

    predicted_noise = model(x_t, t).sample

    # Our CUDA kernel currently accepts FP32 only.
    predicted_noise = predicted_noise.float().contiguous()

    # --------------------------------------------------------
    # 4. Prepare Gaussian noise
    # --------------------------------------------------------

    if noise is None:

        if timestep == 0:
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn_like(x_t)

    if noise.shape != x_t.shape:
        raise ValueError(
            "noise must have the same shape as x_t"
        )

    # --------------------------------------------------------
    # 5. Execute our custom CUDA reverse kernel
    # --------------------------------------------------------

    x_prev = ddpm_cuda.reverse(
        x_t.float().contiguous(),
        predicted_noise,
        noise.float().contiguous(),
        scheduler.betas.contiguous(),
        scheduler.sqrt_one_minus_alpha_bar.contiguous(),
        scheduler.sqrt_recip_alpha.contiguous(),
        scheduler.posterior_variance.contiguous(),
        int(timestep)
    )

    return x_prev
