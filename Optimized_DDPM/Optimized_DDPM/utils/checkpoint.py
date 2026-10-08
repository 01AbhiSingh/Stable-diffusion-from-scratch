"""Atomic saves and weights-only loads (train/resume and old Phase 1 models)."""
import os
from pathlib import Path
import torch


def save_checkpoint(path, *, epoch, global_step, model, optimizer, scaler, ema, config, loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "epoch": epoch,  # completed 1-based epoch
        "global_step": global_step,
        "loss": float(loss),
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_checkpoint(path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location, weights_only=True)


def load_training_state(path, *, model, optimizer, scaler, ema, config):
    ckpt = read_checkpoint(path, map_location="cpu")
    if "config" in ckpt:
        previous = ckpt["config"]
        for key in ("diffusion", "model"):
            if previous[key] != config[key]:
                raise ValueError(f"Cannot resume: {key} config changed")
        if any(previous["dataset"][k] != config["dataset"][k]
               for k in ("image_size", "channels")):
            raise ValueError("Cannot resume: data shape/config changed")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema is not None:
        if ckpt.get("ema_state_dict") is None:
            raise ValueError("EMA enabled but checkpoint has no EMA state")
        ema.load_state_dict(ckpt["ema_state_dict"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        states = ckpt["cuda_rng_state"]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)
    return int(ckpt["epoch"]), int(ckpt.get("global_step", 0))
