"""Read experiment YAML and validate the minimum DDPM invariants."""
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]


def project_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_DIR / p


def load_config(path="configs/mnist.yaml"):
    with project_path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    d, ds, m, tr, opt = (cfg[key] for key in (
        "diffusion", "dataset", "model", "training", "optimization"))
    if ds["name"].upper() != "MNIST":
        raise ValueError("This Phase 2 example implements MNIST only")
    if ds["channels"] != 1:
        raise ValueError("MNIST must have one image channel")
    if int(d["timesteps"]) < 2:
        raise ValueError("timesteps must be at least two")
    if d["beta_schedule"] != "linear":
        raise ValueError("Only a manual linear beta schedule is implemented")
    if not 0 < float(d["beta_start"]) <= float(d["beta_end"]) < 1:
        raise ValueError("Require 0 < beta_start <= beta_end < 1")
    nlevels = len(m["block_out_channels"])
    if nlevels != len(m["down_block_types"]) or nlevels != len(m["up_block_types"]):
        raise ValueError("Model channel/down/up lists must have equal lengths")
    if ds["image_size"] % (2 ** (nlevels - 1)):
        raise ValueError("Image size must be divisible by 2**(num_levels-1)")
    if ds["batch_size"] < 1 or tr["grad_accum_steps"] < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    if opt["precision"] not in ("none", "bf16", "fp16"):
        raise ValueError("precision must be none/bf16/fp16")
    if not 0.0 <= float(opt["ema"]["decay"]) < 1.0:
        raise ValueError("EMA decay must be in [0,1)")
    return cfg
