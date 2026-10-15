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

    # ============================================================
    # 1. ARGUMENTS
    # ============================================================

    p = argparse.ArgumentParser()

    p.add_argument("--config", default="configs/mnist.yaml")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--compile",action="store_true",help="Enable torch.compile")
    p.add_argument("--compile-mode",choices=["default", "reduce-overhead", "max-autotune"],default="default")
    p.add_argument("--precision",choices=["none", "bf16", "fp16"],default=None)
    p.add_argument("--bench-mode",choices=["inference", "training", "all"],default="all")
    args = p.parse_args()

    # ============================================================
    # 2. CONFIGURATION
    # ============================================================

    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    precision = resolve_precision(
        args.precision or cfg["optimization"]["precision"],
        device
    )

    print(
        f"device={device} "
        f"precision={precision} "
        f"batch={args.batch_size}"
    )

    # ============================================================
    # 3. SCHEDULER
    # ============================================================

    sched = ddpmScheduler(
        cfg["diffusion"],
        device
    )

    # ============================================================
    # 4. MODEL
    # ============================================================

    model = get_model(cfg).to(device)

    if args.compile:

        # Allow more specialized compiled graphs
        # for U-Net blocks with different channel dimensions.
        torch._dynamo.config.recompile_limit = 32

        print(
            f"Compiling model: mode={args.compile_mode}"
        )

        model = torch.compile(
            model,
            backend="inductor",
            mode=args.compile_mode
        )

    model.eval()

    # ============================================================
    # 5. SYNTHETIC INPUT
    # ============================================================

    x = torch.randn(
        args.batch_size,
        cfg["dataset"]["channels"],
        cfg["dataset"]["image_size"],
        cfg["dataset"]["image_size"],
        device=device
    )

    t = torch.randint(
        0,
        sched.T,
        (args.batch_size,),
        device=device
    )

    # Fixed noise target for repeatable backward benchmarking
    epsilon_target = torch.randn_like(x)

    # ============================================================
    # 6. INFERENCE BENCHMARKS
    # ============================================================

    if args.bench_mode in ("inference", "all"):

        print("\n========== INFERENCE BENCHMARK ==========")

        model.eval()

        with torch.no_grad():

            # ----------------------------------------------------
            # Forward diffusion
            # ----------------------------------------------------

            timed(
                "forward_diffusion",
                lambda: forward_scheduler(x, t, sched),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats
            )

            # ----------------------------------------------------
            # U-Net forward
            # ----------------------------------------------------

            def net():
                with amp_context(precision, device):
                    return model(x, t).sample

            timed(
                "UNet_forward",
                net,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats
            )

            # ----------------------------------------------------
            # One reverse diffusion step
            # ----------------------------------------------------

            def reverse_step():
                with amp_context(precision, device):
                    return sample_step(
                        model,
                        x,
                        500,
                        sched
                    )

            timed(
                "one_reverse_step",
                reverse_step,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats
            )

    # ============================================================
    # 7. TRAINING BENCHMARK
    # ============================================================

    if args.bench_mode in ("training", "all"):

        print("\n========== TRAINING BENCHMARK ==========")

        model.train()

        epsilon_target = torch.randn_like(x)

        def forward_backward():

            model.zero_grad(set_to_none=True)

            # ----------------------------------------------------
            # Forward pass
            # ----------------------------------------------------

            with amp_context(precision, device):

                epsilon_pred = model(
                    x,
                    t
                ).sample

                loss = F.mse_loss(
                    epsilon_pred.float(),
                    epsilon_target
                )

            # ----------------------------------------------------
            # Backward pass
            # ----------------------------------------------------

            loss.backward()

            model.zero_grad(set_to_none=True)

        timed(
            "UNet_forward_backward",
            forward_backward,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats
        )

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()