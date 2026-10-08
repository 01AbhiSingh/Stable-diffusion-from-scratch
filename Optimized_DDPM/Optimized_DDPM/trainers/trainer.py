"""DDPM epsilon-prediction trainer with accumulation, AMP and EMA."""
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from diffusion.forward_pass import forward_scheduler
from utils.checkpoint import save_checkpoint
from utils.config import project_path
from utils.precision import amp_context


class Trainer:
    def __init__(self, *, model, train_model, optimizer, scaler, ema,
                 scheduler, dataloader, config, device, precision,
                 start_epoch=0, global_step=0):
        self.model = model  # uncompiled underlying model for checkpoint/EMA
        self.train_model = train_model  # may be torch.compile(model)
        self.optimizer = optimizer
        self.scaler = scaler
        self.ema = ema
        self.scheduler = scheduler
        self.loader = dataloader
        self.config = config
        self.device = device
        self.precision = precision
        self.start_epoch = start_epoch
        self.global_step = global_step

    def train(self, smoke_test=False):
        cfg = self.config["training"]
        total_epochs = self.start_epoch + 1 if smoke_test else int(cfg["epochs"])
        accum = int(cfg["grad_accum_steps"])
        log_every = max(1, int(cfg["log_every"]))
        ckpt_dir = project_path(self.config["checkpoint"]["directory"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, total_epochs):
            self.train_model.train()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize()
            started = time.perf_counter()
            loss_sum = torch.zeros((), device=self.device)
            samples_seen = 0
            num_batches = 1 if smoke_test else len(self.loader)
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, (x0, _) in enumerate(self.loader):
                if batch_idx >= num_batches:
                    break
                x0 = x0.to(self.device, non_blocking=True)
                B = x0.shape[0]
                t = torch.randint(0, self.scheduler.T, (B,),
                                  device=self.device, dtype=torch.long)
                x_t, epsilon = forward_scheduler(x0, t, self.scheduler)

                # Final partial accumulation group must use its real size.
                group_start = (batch_idx // accum) * accum
                group_size = min(accum, num_batches - group_start)
                with amp_context(self.precision, self.device):
                    prediction = self.train_model(x_t, t).sample
                    # MSE evaluated in FP32, including BF16/FP16 predictions.
                    raw_loss = F.mse_loss(prediction.float(), epsilon)
                    loss = raw_loss / group_size
                self.scaler.scale(loss).backward()
                loss_sum += raw_loss.detach() * B
                samples_seen += B

                should_step = (batch_idx + 1) % accum == 0 or batch_idx + 1 == num_batches
                if should_step:
                    if cfg.get("max_grad_norm") is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float(cfg["max_grad_norm"]))
                    old_scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
                    # A GradScaler overflow skips the optimizer step.
                    stepped = (not self.scaler.is_enabled()) or new_scale >= old_scale
                    self.optimizer.zero_grad(set_to_none=True)
                    if stepped:
                        self.global_step += 1
                        if self.ema is not None:
                            self.ema.update(self.model)

                if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                    print(f"epoch={epoch+1}/{total_epochs} "
                          f"batch={batch_idx+1}/{num_batches} "
                          f"mse={raw_loss.item():.5f}")

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            epoch_loss = (loss_sum / max(samples_seen, 1)).item()
            peak = (torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    if self.device.type == "cuda" else 0.0)
            print(f"epoch={epoch+1} avg_mse={epoch_loss:.6f} "
                  f"seconds={elapsed:.2f} images/s={samples_seen/max(elapsed,1e-9):.1f} "
                  f"peak_allocated_GiB={peak:.3f}")

            if smoke_test:
                print("[smoke-test] One batch completed; no checkpoint written.")
                return

            save_checkpoint(
                ckpt_dir / "last.pt", epoch=epoch + 1, global_step=self.global_step,
                model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                ema=self.ema, config=self.config, loss=epoch_loss,
            )
            every = int(self.config["checkpoint"]["save_every"])
            if every > 0 and (epoch + 1) % every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch+1:04d}.pt",
                    epoch=epoch+1, global_step=self.global_step,
                    model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                    ema=self.ema, config=self.config, loss=epoch_loss,
                )
