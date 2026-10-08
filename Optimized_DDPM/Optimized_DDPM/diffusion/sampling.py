"""Manual ancestral DDPM reverse transition (posterior variance choice)."""
import torch


@torch.no_grad()
def sample_step(model, x_t, timestep: int, scheduler, noise=None):
    """One reverse step. `timestep` is a scalar Python int shared by batch.

    For timestep > 0, noise may be injected for deterministic parity tests.
    At timestep = 0, no random noise is added.
    """
    if timestep < 0 or timestep >= scheduler.T:
        raise IndexError(f"timestep {timestep} outside [0,{scheduler.T - 1}]")
    t = torch.full((x_t.shape[0],), timestep, dtype=torch.long, device=x_t.device)
    predicted_noise = model(x_t, t).sample.float()
    beta = scheduler.betas[timestep]
    scale = scheduler.sqrt_one_minus_alpha_bar[timestep]
    mean = scheduler.sqrt_recip_alpha[timestep] * (
        x_t.float() - (beta / scale) * predicted_noise
    )
    if timestep == 0:
        return mean
    if noise is None:
        noise = torch.randn_like(x_t)
    if noise.shape != x_t.shape:
        raise ValueError("Noise must have shape identical to x_t")
    sigma = torch.sqrt(scheduler.posterior_variance[timestep])
    return mean + sigma * noise


@torch.inference_mode()
def generate(model, scheduler, shape, *, initial_noise=None, on_step=None):
    """Start x_T~N(0,I), repeatedly apply learned reverse transitions."""
    if initial_noise is None:
        x = torch.randn(shape, device=scheduler.device, dtype=torch.float32)
    else:
        x = initial_noise.clone()
        if tuple(x.shape) != tuple(shape):
            raise ValueError("initial_noise must match requested shape")
    for step in range(scheduler.T - 1, -1, -1):
        x = sample_step(model, x, step, scheduler)
        if on_step is not None:
            on_step(step, x)
    return x
