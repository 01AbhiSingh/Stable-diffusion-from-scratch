"""Manual DDPM constants, indexed by Python timestep t in [0,T-1].

Python t=0 corresponds to the paper's FIRST diffusion step. The initial
clean sample x_0 from the paper is named x0 in our forward_pass API.
"""
import torch


class ddpmScheduler:
    def __init__(self, diffusion_config, device="cpu"):
        self.device = torch.device(device)
        self.T = int(diffusion_config["timesteps"])
        if diffusion_config.get("beta_schedule", "linear") != "linear":
            raise ValueError("Only linear beta schedule is implemented")

        self.betas = torch.linspace(
            float(diffusion_config["beta_start"]),
            float(diffusion_config["beta_end"]),
            self.T,
            device=self.device,
            dtype=torch.float32,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat(
            (torch.ones(1, device=self.device, dtype=self.betas.dtype),
             self.alpha_bar[:-1]), dim=0,
        )
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.rsqrt(self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        # Posterior q(x_(t-1)|x_t,x0) variance is zero at index zero.
        # At sampling index zero we always return the mean without new noise.

    def extract(self, values, timesteps, target_shape):
        """[T] indexed by [B] -> [B,1,1,1] for [B,C,H,W]."""
        if timesteps.ndim != 1 or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a torch.long tensor of shape [B]")
        if timesteps.shape[0] != target_shape[0]:
            raise ValueError("Batch dimensions disagree")
        return values.gather(0, timesteps).reshape(
            timesteps.shape[0], *([1] * (len(target_shape) - 1))
        )
