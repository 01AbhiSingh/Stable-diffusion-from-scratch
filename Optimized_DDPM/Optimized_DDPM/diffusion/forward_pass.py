"""Manually implement q(x_t|x_0) with no diffusers scheduler."""
import torch


def forward_scheduler(x0, t, scheduler, epsilon=None):
    """x0:[B,C,H,W], t:[B]; returns (x_t, noise_used)."""
    if epsilon is None:
        epsilon = torch.randn_like(x0)
    if epsilon.shape != x0.shape or epsilon.device != x0.device:
        raise ValueError("epsilon and x0 must have the same shape and device")
    signal = scheduler.extract(scheduler.sqrt_alpha_bar, t, x0.shape)
    noise_scale = scheduler.extract(scheduler.sqrt_one_minus_alpha_bar, t, x0.shape)
    return signal * x0 + noise_scale * epsilon, epsilon
