import argparse
import torch
import ddpm_cuda


def pytorch_forward(x0, eps, t, sqrt_ab, sqrt_1m_ab):
    a = sqrt_ab[t].view(-1, 1, 1, 1)
    b = sqrt_1m_ab[t].view(-1, 1, 1, 1)
    return a * x0 + b * eps


def benchmark_cuda_events(fn, warmup=100, repeat=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--repeats", type=int, default=1000)
    args = p.parse_args()
    assert torch.cuda.is_available()

    torch.manual_seed(123)
    dev = "cuda"
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar).contiguous()
    sqrt_1m_ab = torch.sqrt(1 - alpha_bar).contiguous()
    shape = (args.batch_size, args.channels, args.image_size, args.image_size)
    x0 = torch.randn(*shape, device=dev)
    eps = torch.randn_like(x0)
    t = torch.randint(0, T, (args.batch_size,), device=dev)

    reference = lambda: pytorch_forward(x0, eps, t, sqrt_ab, sqrt_1m_ab)
    fused = lambda: ddpm_cuda.forward(x0, eps, t, sqrt_ab, sqrt_1m_ab)
    torch.testing.assert_close(reference(), fused(), atol=3e-6, rtol=1e-5)

    reference_ms = benchmark_cuda_events(reference, args.warmup, args.repeats)
    fused_ms = benchmark_cuda_events(fused, args.warmup, args.repeats)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {shape}  repeats={args.repeats}")
    print(f"PyTorch forward:  {reference_ms * 1000:.2f} microseconds/call")
    print(f"CUDA fused:       {fused_ms * 1000:.2f} microseconds/call")
    print(f"Speedup:          {reference_ms / fused_ms:.2f}x")
    print("Note: CUDA-event timing of repeated Python launches; not a complete DDPM training benchmark.")


if __name__ == "__main__":
    main()
