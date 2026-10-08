#!/usr/bin/env python3
"""Sample DDPM manually from a Phase 1 or Phase 2 checkpoint."""
import argparse
import torch
from torchvision.utils import save_image

from diffusion.sampling import generate
from diffusion.scheduler import ddpmScheduler
from models.unet import get_model
from utils.checkpoint import read_checkpoint
from utils.config import load_config, project_path
from utils.precision import amp_context, resolve_precision


def to_image_range(x):
    return (x.float().clamp(-1.0, 1.0) + 1.0) / 2.0


def main():
    p = argparse.ArgumentParser(description="Manual ancestral DDPM sampler")
    p.add_argument("--config", default="configs/mnist.yaml")
    p.add_argument("--checkpoint", default="checkpoints/last.pt")
    p.add_argument("--output", default=None)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--precision", choices=["none", "bf16", "fp16"], default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args.precision or cfg["optimization"]["precision"], device)
    ckpt_path = project_path(args.checkpoint)
    ckpt = read_checkpoint(ckpt_path, map_location="cpu")
    model = get_model(cfg).to(device)
    ema = ckpt.get("ema_state_dict")
    if not args.no_ema and cfg["sampling"]["use_ema"] and ema is not None:
        weights = ema["model"]
        print("Using EMA weights")
    else:
        weights = ckpt["model_state_dict"]
        print("Using standard model weights")
    model.load_state_dict(weights, strict=True)
    model.eval()
    scheduler = ddpmScheduler(cfg["diffusion"], device=device)
    total = args.num_images or int(cfg["sampling"]["num_images"])
    batch_size = int(cfg["sampling"]["batch_size"])
    output_file = project_path(args.output or cfg["sampling"]["output"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    snapshot_steps = set(cfg["sampling"]["intermediate_timesteps"])
    capture = bool(cfg["sampling"]["save_intermediates"])
    all_images = []

    # Chunk generation so larger num_images doesn't exhaust VRAM.
    for start in range(0, total, batch_size):
        B = min(batch_size, total - start)
        shape = (B, cfg["dataset"]["channels"],
                 cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])

        def on_step(step, x):
            if start == 0 and capture and step in snapshot_steps:
                snapshots[step] = x.detach().float().cpu()
            if step % 100 == 0:
                print(f"batch={start // batch_size + 1} timestep={step}")

        with torch.inference_mode(), amp_context(precision, device):
            result = generate(model, scheduler, shape, on_step=on_step)
        all_images.append(to_image_range(result).cpu())

    images = torch.cat(all_images, dim=0)
    save_image(images, output_file, nrow=4)
    print(f"Saved {total} samples: {output_file}")
    if snapshots:
        directory = output_file.parent / (output_file.stem + "_steps")
        directory.mkdir(parents=True, exist_ok=True)
        for step, tensor in sorted(snapshots.items(), reverse=True):
            save_image(to_image_range(tensor), directory / f"step_{step:04d}.png", nrow=4)
        print(f"Saved intermediate images: {directory}")


if __name__ == "__main__":
    main()
