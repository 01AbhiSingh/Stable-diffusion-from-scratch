import argparse
import torch
import ddpm_cuda

from benchmark import benchmark_cuda_events
from test_reverse import create_scheduler, pytorch_reverse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--timestep", type=int, default=500)

    args = parser.parse_args()

    assert torch.cuda.is_available()

    torch.manual_seed(42)

    shape = (args.batch_size, 1, 32, 32)

    scheduler = create_scheduler()

    betas, scale, recip_alpha, variance = scheduler

    assert 0 <= args.timestep < betas.numel()

    x_t = torch.randn(shape, device="cuda")
    epsilon_pred = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)

    timestep = args.timestep

    # --------------------------------------------------------
    # PyTorch implementation
    # --------------------------------------------------------

    def pytorch_fn():

        return pytorch_reverse(
            x_t,
            epsilon_pred,
            noise,
            betas,
            scale,
            recip_alpha,
            variance,
            timestep
        )

    # --------------------------------------------------------
    # CUDA implementation
    # --------------------------------------------------------

    def cuda_fn():

        return ddpm_cuda.reverse(
            x_t,
            epsilon_pred,
            noise,
            betas,
            scale,
            recip_alpha,
            variance,
            timestep
        )

    # --------------------------------------------------------
    # Correctness verification
    # --------------------------------------------------------

    torch.testing.assert_close(
        cuda_fn(),
        pytorch_fn(),
        atol=3e-6,
        rtol=1e-5
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    pytorch_ms = benchmark_cuda_events(
        pytorch_fn,
        args.warmup,
        args.repeats
    )

    cuda_ms = benchmark_cuda_events(
        cuda_fn,
        args.warmup,
        args.repeats
    )

    print("\n========== REVERSE DIFFUSION BENCHMARK ==========")

    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"Shape:     {shape}")
    print(f"Timestep:  {timestep}")

    print(
        f"PyTorch reverse: {pytorch_ms * 1000:.3f} us"
    )

    print(
        f"CUDA reverse:    {cuda_ms * 1000:.3f} us"
    )

    print(
        f"Speedup:         {pytorch_ms / cuda_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
