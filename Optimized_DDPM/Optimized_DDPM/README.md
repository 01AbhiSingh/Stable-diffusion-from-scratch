# Phase 2 — Optimized DDPM (MNIST)

Complete standalone Python DDPM project. **Scheduler, closed-form forward diffusion,
reverse ancestral DDPM step, training objective and sampling loop are implemented manually**.
The only external DDPM-related component is `diffusers.UNet2DModel`, which is **just**
the timestep-conditioned noise predictor; no prebuilt diffusion scheduler or pipeline is used.

## Install

```bash
cd Optimized_DDPM
# Activate a venv with PyTorch + torchvision installed for your system/CUDA.
python -m pip install -r requirements.txt
python -c 'import torch, diffusers; print(torch.__version__, diffusers.__version__)'
```

## Run

```bash
# Verify scheduler, q(x_t|x0), and reverse math (no dataset/GPU needed).
python -m unittest discover -s tests -v

# Check full training pipeline with exactly one batch (no checkpoint saved).
python train.py --config configs/mnist.yaml --smoke-test

# Train 15 epochs; save checkpoints/last.pt each completed epoch.
python train.py --config configs/mnist.yaml

# Continue from complete epoch stored in last.pt.
python train.py --resume checkpoints/last.pt

# Generate images and optional intermediate grids with EMA model weights.
python sample.py --checkpoint checkpoints/last.pt --num-images 16

# For an old Phase 1 checkpoint (no EMA stored), sample using raw weights.
python sample.py --checkpoint ../BruteForce_DDPM/checkpoints/ddpm_mnist_epoch_15.pt --no-ema

# Compare timing under same GPU/batch with config precision switched.
python benchmark.py --precision none --batch-size 16
python benchmark.py --precision bf16 --batch-size 16
```

Paths inside YAML and CLI checkpoint/output paths are relative to this project folder.
By default MNIST downloads to `Optimized_DDPM/data/` (not in Git).

## Source reading order

1. `configs/mnist.yaml` — experiment controls. Defaults preserve the Phase 1 U-Net.
2. `diffusion/scheduler.py` — beta, alpha, alpha_bar and posterior variance.
3. `diffusion/forward_pass.py` — `q(x_t|x_0)` and epsilon target.
4. `diffusion/sampling.py` — DDPM reverse mean and posterior-variance injection.
5. `models/unet.py` — model factory, no hand-built U-Net.
6. `trainers/trainer.py` — DDPM training, AMP, gradient accumulation, EMA.
7. `train.py` and `sample.py` — CLI entrypoints.
8. `utils/` — EMA and checkpoint internals; `benchmark.py` — microbenchmarks.

## Mathematical parity

- `T=1000`, linear betas `[0.0001, 0.02]`, and epsilon MSE match Phase 1.
- The model architecture matches the original `(64,128,256)` channels and
  three ordinary DownBlock2D / UpBlock2D stages with two layers/block.
- At Python index 0 the first noise step has occurred, not the untouched x0.
- Reverse sampling at index 0 adds *no noise*.
- The DDPM variance choice is `posterior_variance` (`beta_tilde`), not `beta`.
- Given identical inputs, noise, timesteps, precision, and model weights,
  Phase 1 and Phase 2 math should agree to floating-point tolerance.

## Optimizations and controls

- AMP `precision: bf16` (preferred on supported NVIDIA GPUs), `fp16` (GradScaler),
  or `none` (FP32). FP32 diffusion coefficients are kept outside autocast.
- `grad_accum_steps`: averages loss over the accumulation group, including
  a shorter final group. `max_grad_norm` clips unscaled gradients.
- EMA tracks *successful optimizer updates*, and EMA weights are selectable
  during sampling. EMA is optional via config.
- DataLoader uses pinned memory, persistent workers and prefetch with workers > 0.
- `torch.compile` is **off by default** because initial compile is expensive
  and is not a guaranteed speedup on short MNIST training runs.
- Checkpoints: `last.pt` (always) and `epoch_XXXX.pt` at configured intervals;
  includes optimizer/scaler/EMA states. If resuming, `training.epochs` is the
  **total** desired number of epochs, not an additional number.
- `benchmark.py` uses CUDA synchronization for actual GPU timings; forward
  and backward timing is a synthetic microbenchmark, not throughput for data loading.

## Notes

- BF16 requires supported GPU hardware; use `precision: fp16` or `none` otherwise.
- Mixed precision and `torch.compile` can change floating-point results, so
  check numerical tolerances rather than expecting bitwise FP32 equivalence.
- EMA can improve sampling quality, but samples depend on training and hardware;
  the project makes **no guaranteed** accuracy/speed claims.
- Keep `BruteForce_DDPM/` frozen as the mathematical golden reference.
- Phase 1 checkpoints can be used for **sampling** as shown above. To resume
  training use a Phase 2 checkpoint that has optimizer and EMA states.
- This example does not implement DDP, automated hyperparameter search,
  sophisticated image metrics (FID), or latent diffusion.
