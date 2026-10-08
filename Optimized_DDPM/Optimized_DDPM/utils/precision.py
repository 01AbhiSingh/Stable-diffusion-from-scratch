"""AMP setup and context managers for training/sampling."""
from contextlib import nullcontext
import torch


def resolve_precision(requested, device):
    if device.type != "cuda":
        if requested != "none":
            print("[precision] CPU detected; using FP32 (AMP disabled).")
        return "none"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16. Use precision: fp16 or none")
    return requested


def amp_context(precision, device):
    if device.type != "cuda" or precision == "none":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )


def make_scaler(precision, device):
    # BF16 has sufficient exponent range and normally does NOT need a scaler.
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
