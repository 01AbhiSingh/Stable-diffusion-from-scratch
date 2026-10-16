import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion.sampling import sample_step
from diffusion.scheduler import ddpmScheduler
from models.unet import get_model
from utils.checkpoint import read_checkpoint
from utils.config import load_config, project_path
from utils.precision import amp_context, resolve_precision


# ============================================================
# 1. CUDA EXTENSION LOCATION
# ============================================================

DEFAULT_CUDA_DIR = (
    Path(__file__).resolve().parents[2]
    / "CUDA_DDPM_Phase3"
    / "CUDA_DDPM"
)


# ============================================================
# 2. IMAGE NORMALIZATION
# ============================================================

def to_image_range(x):

    return (
        x.float().clamp(-1.0, 1.0) + 1.0
    ) / 2.0


# ============================================================
# 3. COMPLETE REVERSE SAMPLING
# ============================================================

@torch.inference_mode()
def run_sampling(
    model,
    scheduler,
    start_noise,
    generator,
    precision,
    device,
    use_cuda=False,
    trace = None
):

    from diffusion.cuda_sampling import sample_step_cuda

    x = start_noise.clone()

    zero_noise = torch.zeros_like(x)

    torch.cuda.synchronize(device)

    start = time.perf_counter()

    with amp_context(precision, device):

        for timestep in range(
            scheduler.T - 1,
            -1,
            -1
        ):

            # Generate identical noise sequences
            # using separately seeded generators.

            if timestep == 0:

                noise = zero_noise

            else:

                noise = torch.randn(
                    x.shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32
                )

            # --------------------------------------------
            # Select implementation
            # --------------------------------------------

            if use_cuda:

                x = sample_step_cuda(
                    model,
                    x,
                    timestep,
                    scheduler,
                    noise=noise
                )

            else:

                x = sample_step(
                    model,
                    x,
                    timestep,
                    scheduler,
                    noise=noise
                )
            # Record intermediate samples for numerical comparison
            if trace is not None and timestep in (
                999, 750, 500, 250, 100, 10, 0
            ):
                trace[timestep] = x.detach().clone()
    torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    return x, elapsed


# ============================================================
# 4. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/mnist.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/last.pt"
    )

    parser.add_argument(
        "--cuda-dir",
        type=Path,
        default=DEFAULT_CUDA_DIR
    )

    parser.add_argument(
        "--output-dir",
        default="cuda_comparison"
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--precision",
        choices=["none", "bf16", "fp16"],
        default=None
    )

    parser.add_argument(
        "--no-ema",
        action="store_true"
    )

    parser.add_argument(
        "--compile",
        action="store_true"
    )

    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead"],
        default="reduce-overhead"
    )

    args = parser.parse_args()
    pytorch_trace = {}
    cuda_trace = {}
    # --------------------------------------------------------
    # 5. DEVICE + CUDA EXTENSION
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")

    cuda_dir = args.cuda_dir.expanduser().resolve()

    if not cuda_dir.exists():
        raise FileNotFoundError(
            f"CUDA extension directory not found: {cuda_dir}"
        )

    sys.path.insert(0, str(cuda_dir))

    import ddpm_cuda

    # --------------------------------------------------------
    # 6. CONFIGURATION
    # --------------------------------------------------------

    cfg = load_config(args.config)

    precision = resolve_precision(
        args.precision or cfg["optimization"]["precision"],
        device
    )

    # --------------------------------------------------------
    # 7. LOAD TRAINED CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = project_path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = read_checkpoint(
        checkpoint_path,
        map_location="cpu"
    )

    model = get_model(cfg).to(device)

    ema = checkpoint.get("ema_state_dict")

    if (
        not args.no_ema
        and cfg["sampling"]["use_ema"]
        and ema is not None
    ):

        weights = ema["model"]
        print("Using EMA weights")

    else:

        weights = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model")
        )

        if weights is None:
            raise KeyError(
                "Checkpoint missing model weights"
            )

        print("Using regular model weights")

    model.load_state_dict(weights, strict=True)

    model.eval()

    # --------------------------------------------------------
    # 8. OPTIONAL TORCH.COMPILATION
    # --------------------------------------------------------

    if args.compile:

        torch._dynamo.config.recompile_limit = 32

        model = torch.compile(
            model,
            mode=args.compile_mode
        )

        print(
            f"torch.compile enabled: {args.compile_mode}"
        )

    # --------------------------------------------------------
    # 9. DDPM SCHEDULER
    # --------------------------------------------------------

    scheduler = ddpmScheduler(
        cfg["diffusion"],
        device=device
    )

    # --------------------------------------------------------
    # 10. PREPARE INITIAL NOISE
    # --------------------------------------------------------

    B = (
        args.num_images
        or int(cfg["sampling"]["num_images"])
    )

    if B <= 0:
        raise ValueError("--num-images must be positive")

    shape = (
        B,
        int(cfg["dataset"]["channels"]),
        int(cfg["dataset"]["image_size"]),
        int(cfg["dataset"]["image_size"])
    )

    # Same initial noise for both implementations.

    initial_generator = torch.Generator(
        device=device
    ).manual_seed(args.seed)

    x_initial = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=initial_generator
    )

    # Independent generators with identical seeds.
    # Both produce the same noise at every reverse step.

    pytorch_generator = torch.Generator(
        device=device
    ).manual_seed(args.seed + 1)

    cuda_generator = torch.Generator(
        device=device
    ).manual_seed(args.seed + 1)

    # --------------------------------------------------------
    # 11. U-NET WARMUP
    # --------------------------------------------------------

    print("\nWarming up U-Net...")

    with torch.inference_mode(), amp_context(precision, device):

        warmup_t = torch.full(
            (B,),
            scheduler.T - 1,
            device=device,
            dtype=torch.long
        )

        for _ in range(3):

            _ = model(
                x_initial,
                warmup_t
            ).sample

    torch.cuda.synchronize(device)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {shape}")
    print(f"Precision: {precision}")
    print(f"Diffusion steps: {scheduler.T}")

    # ============================================================
    # SAME-INPUT REVERSE DIFFUSION PARITY TEST
    # ============================================================

    print("\n========== SAME-INPUT CUDA PARITY ==========")

    with torch.inference_mode():

        check_noise = torch.randn_like(x_initial)

        for step in [999, 500, 100, 10, 0]:

            t_batch = torch.full(
                (B,),
                step,
                device=device,
                dtype=torch.long
            )

            # Predict noise ONCE.
            # Both implementations use this exact prediction.
            with amp_context(precision, device):
                eps_pred = model(x_initial, t_batch).sample

            eps_pred = eps_pred.float().contiguous()

            # PyTorch reference calculation
            mean = scheduler.sqrt_recip_alpha[step] * (
                x_initial
                - (
                    scheduler.betas[step]
                    / scheduler.sqrt_one_minus_alpha_bar[step]
                ) * eps_pred
            )

            if step > 0:
                pytorch_output = (
                    mean
                    + torch.sqrt(scheduler.posterior_variance[step])
                    * check_noise
                )
            else:
                pytorch_output = mean

            # CUDA calculation
            cuda_output = ddpm_cuda.reverse(
                x_initial.contiguous(),
                eps_pred,
                check_noise.contiguous(),
                scheduler.betas.contiguous(),
                scheduler.sqrt_one_minus_alpha_bar.contiguous(),
                scheduler.sqrt_recip_alpha.contiguous(),
                scheduler.posterior_variance.contiguous(),
                step
            )

            error = (pytorch_output - cuda_output).abs()

            print(
                f"t={step:3d} "
                f"max_error={error.max().item():.9f} "
                f"mean_error={error.mean().item():.9f}"
            )
    from diffusion.cuda_sampling import sample_step_cuda

    print("\n========== ACTUAL SAMPLER PARITY ==========")

    with torch.inference_mode(), amp_context(precision, device):

        for step in [999, 500, 100, 10, 0]:

            noise = torch.randn_like(x_initial)

            # Existing PyTorch sampler
            pytorch_result = sample_step(
                model,
                x_initial,
                step,
                scheduler,
                noise=noise
            )

            # Our custom CUDA sampler
            cuda_result = sample_step_cuda(
                model,
                x_initial,
                step,
                scheduler,
                noise=noise
            )

            error = (pytorch_result - cuda_result).float().abs()

            print(
                f"t={step:3d} "
                f"max_error={error.max().item():.9f} "
                f"mean_error={error.mean().item():.9f}"
            )
    # ========================================================
    # 12. PYTORCH SAMPLING
    # ========================================================

    print("\nRunning PyTorch sampling...")

    reference, pytorch_seconds = run_sampling(
        model,
        scheduler,
        x_initial,
        pytorch_generator,
        precision,
        device,
        use_cuda=False,
        trace=pytorch_trace
    )

    # ========================================================
    # 13. CUDA SAMPLING
    # ========================================================

    print("\nRunning CUDA sampling...")

    candidate, cuda_seconds = run_sampling(
        model,
        scheduler,
        x_initial,
        cuda_generator,
        precision,
        device,
        use_cuda=True,
        trace=cuda_trace
    )
    # ========================================================
    # 14. NUMERICAL COMPARISON
    # ========================================================

    delta = (
        candidate - reference
    ).float()

    max_error = delta.abs().max().item()

    mean_error = delta.abs().mean().item()

    rmse = delta.square().mean().sqrt().item()

    print(
        "\n========== COMPLETE SAMPLING COMPARISON =========="
    )

    print(f"PyTorch sampling: {pytorch_seconds:.3f} s")

    print(f"CUDA sampling:    {cuda_seconds:.3f} s")

    print(
        f"Speedup:          "
        f"{pytorch_seconds / cuda_seconds:.3f}x"
    )

    print(f"Max abs error:    {max_error:.6g}")

    print(f"Mean abs error:   {mean_error:.6g}")

    print(f"RMSE:             {rmse:.6g}")
    print("\n========== ERROR BY TIMESTEP ==========")

    for timestep in sorted(pytorch_trace.keys(), reverse=True):

        ref = pytorch_trace[timestep]
        out = cuda_trace[timestep]

        error = (out - ref).float().abs()

        print(
            f"t={timestep:3d} "
            f"max_error={error.max().item():.8f} "
            f"mean_error={error.mean().item():.8f}"
        )

    # ========================================================
    # 15. SAVE GENERATED IMAGES
    # ========================================================

    output_dir = project_path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    nrow = min(B, 8)

    save_image(
        to_image_range(reference).cpu(),
        output_dir / "samples_pytorch.png",
        nrow=nrow
    )

    save_image(
        to_image_range(candidate).cpu(),
        output_dir / "samples_cuda.png",
        nrow=nrow
    )

    print(f"\nImages saved to: {output_dir}")


if __name__ == "__main__":
    main()
