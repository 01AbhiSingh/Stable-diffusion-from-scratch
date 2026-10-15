import torch
import ddpm_cuda


def reference(x0, eps, t, sqrt_ab, sqrt_1m_ab):
    a = sqrt_ab[t].view(-1, 1, 1, 1)
    b = sqrt_1m_ab[t].view(-1, 1, 1, 1)
    return a * x0 + b * eps


def run_test(batch, channels, height, width):
    torch.manual_seed(42)
    device = "cuda"
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar).contiguous()
    sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar).contiguous()

    x0 = torch.randn(batch, channels, height, width, device=device)
    eps = torch.randn_like(x0)
    t = torch.randint(0, T, (batch,), device=device, dtype=torch.int64)
    t[0] = 0
    if batch > 1:
        t[-1] = T - 1

    result = ddpm_cuda.forward(x0, eps, t, sqrt_ab, sqrt_1m_ab)
    expected = reference(x0, eps, t, sqrt_ab, sqrt_1m_ab)

    torch.testing.assert_close(result, expected, atol=3e-6, rtol=1e-5)
    max_error = (result - expected).abs().max().item()
    print(f"PASS shape={tuple(x0.shape)} max_abs_error={max_error:.3g}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    print("GPU:", torch.cuda.get_device_name(0))
    for shape in [(1, 1, 32, 32), (16, 1, 32, 32), (128, 1, 32, 32),
                  (7, 3, 33, 35)]:
        run_test(*shape)
    print("All CUDA forward diffusion tests passed.")
