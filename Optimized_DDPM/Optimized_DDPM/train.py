#!/usr/bin/env python3
"""Entry point: python train.py --config configs/mnist.yaml [--resume ...]."""
import argparse
import random
import torch

from data_pipeline.mnist import make_mnist_loader
from diffusion.scheduler import ddpmScheduler
from models.unet import get_model
from trainers.trainer import Trainer
from utils.checkpoint import load_training_state
from utils.config import load_config, project_path
from utils.ema import EMA
from utils.precision import make_scaler, resolve_precision


def main():
    parser = argparse.ArgumentParser(description="Train optimized epsilon-prediction DDPM")
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--resume", default=None, help="last.pt or epoch_XXXX.pt")
    parser.add_argument("--smoke-test", action="store_true", help="one train batch, no checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["optimization"]["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(cfg["optimization"]["tf32"])
    precision = resolve_precision(cfg["optimization"]["precision"], device)
    print(f"device={device} precision={precision} torch={torch.__version__}")

    loader = make_mnist_loader(cfg, device)
    scheduler = ddpmScheduler(cfg["diffusion"], device=device)
    model = get_model(cfg).to(device)
    train_model = model

    if cfg["optimization"]["compile"]["enabled"]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile not available in this torch installation")
        train_model = torch.compile(model, mode=cfg["optimization"]["compile"]["mode"])
        print("torch.compile enabled: first iteration may take much longer")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = make_scaler(precision, device)
    ema_cfg = cfg["optimization"]["ema"]
    ema = EMA(model, ema_cfg["decay"]) if ema_cfg["enabled"] else None
    start_epoch = global_step = 0
    if args.resume:
        start_epoch, global_step = load_training_state(
            project_path(args.resume), model=model, optimizer=optimizer,
            scaler=scaler, ema=ema, config=cfg,
        )
        print(f"Resumed at completed epoch={start_epoch}, global_step={global_step}")

    trainer = Trainer(
        model=model, train_model=train_model, optimizer=optimizer,
        scaler=scaler, ema=ema, scheduler=scheduler, dataloader=loader,
        config=cfg, device=device, precision=precision,
        start_epoch=start_epoch, global_step=global_step,
    )
    trainer.train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
