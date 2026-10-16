import argparse
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

from models.unet import get_model
from utils.config import load_config, project_path
from utils.checkpoint import read_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = load_config("configs/mnist.yaml")

    # Load our trained EMA model
    model = get_model(cfg).to(device)

    checkpoint = read_checkpoint(
        project_path("checkpoints/last.pt"),
        map_location="cpu"
    )

    ema = checkpoint.get("ema_state_dict")

    if ema is not None:
        weights = ema["model"]
    else:
        weights = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model")
        )

    model.load_state_dict(weights, strict=True)
    model.eval()

    if args.compile:
        torch._dynamo.config.recompile_limit = 32
        model = torch.compile(
            model,
            mode="reduce-overhead"
        )

    # Input matching our DDPM sampling workload
    x = torch.randn(
        args.batch_size, 1, 32, 32,
        device=device
    )

    t = torch.full(
        (args.batch_size,),
        500,
        device=device,
        dtype=torch.long
    )

    # Warmup before profiling
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        for _ in range(20):
            model(x, t).sample

    torch.cuda.synchronize()

    # Profile U-Net execution
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            for _ in range(5):
                model(x, t).sample

            torch.cuda.synchronize()

    print("\n========== TOP CUDA OPERATIONS ==========")

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20
        )
    )

    output_dir = Path("profiler_results")
    output_dir.mkdir(exist_ok=True)

    name = "compiled" if args.compile else "eager"

    trace_path = output_dir / f"unet_{name}.json"

    prof.export_chrome_trace(str(trace_path))

    print(f"\nProfiler trace saved: {trace_path}")


if __name__ == "__main__":
    main()
