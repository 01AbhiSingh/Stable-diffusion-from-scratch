"""Reference integration snippet for the existing Optimized_DDPM pipeline.

Copy forward_scheduler_cuda into a module in your optimized project only AFTER
standalone correctness tests pass. This extension supports FP32 CUDA inputs.
"""
import torch
import ddpm_cuda


def forward_scheduler_cuda(x0, t, scheduler, epsilon=None):
    if epsilon is None:
        epsilon = torch.randn_like(x0)
    xt = ddpm_cuda.forward(
        x0.contiguous(),
        epsilon.contiguous(),
        t.contiguous(),
        scheduler.sqrt_alpha_bar.contiguous(),
        scheduler.sqrt_one_minus_alpha_bar.contiguous(),
    )
    return xt, epsilon
