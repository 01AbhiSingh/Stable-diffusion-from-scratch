#!/usr/bin/env python3
"""Small synthetic microbenchmark for forward noise, network and reverse step.

Times include GPU synchronization; does not measure a full data loading epoch.
"""
import argparse
import time
import torch
import torch.nn.functional as F

from diffusion.forward_pass import forward_scheduler
from diffusion.sampling import sample_step
from diffusion.scheduler import ddpmScheduler
from models.unet import get_model
from utils.config import load_config
from utils.precision import amp_context, resolve_precision


def timed(label, fn, *, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000 / repeats
    peak = (torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0)
    print(f"{label:24s} {ms:9.3f} ms/iter   peak_allocated={peak:.3f} GiB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mnist.yaml")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--precision", choices=["none", "bf16", "fp16"], default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision or cfg["optimization"]["precision"], device)
    sched = ddpmScheduler(cfg["diffusion"], device)
    model = get_model(cfg).to(device)
    x = torch.randn(args.batch_size, cfg["dataset"]["channels"],
                    cfg["dataset"]["image_size"], cfg["dataset"]["image_size"],
                    device=device)
    t = torch.randint(0, sched.T, (args.batch_size,), device=device)
    print(f"device={device} precision={precision} batch={args.batch_size}")

    with torch.no_grad():
        timed("forward_diffusion", lambda: forward_scheduler(x, t, sched),
              device=device, warmup=args.warmup, repeats=args.repeats)
        def net():
            with amp_context(precision, device):
                return model(x, t).sample
        timed("UNet_forward", net, device=device,
              warmup=args.warmup, repeats=args.repeats)
        timed("one_reverse_step", lambda: sample_step(model, x, 500, sched),
              device=device, warmup=args.warmup, repeats=args.repeats)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        with amp_context(precision, device):
            pred = model(x, t).sample
            loss = F.mse_loss(pred.float(), torch.randn_like(x))
        loss.backward()
        model.zero_grad(set_to_none=True)
    timed("UNet_forward_backward", forward_backward, device=device,
          warmup=args.warmup, repeats=args.repeats)


if __name__ == "__main__":
    main()
