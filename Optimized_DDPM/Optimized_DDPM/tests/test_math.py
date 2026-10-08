"""Run from Optimized_DDPM: python -m unittest discover -s tests -v"""
import unittest
from types import SimpleNamespace
import torch

from diffusion.scheduler import ddpmScheduler
from diffusion.forward_pass import forward_scheduler
from diffusion.sampling import sample_step, generate


CFG = {"timesteps": 1000, "beta_schedule": "linear", "beta_start": 1e-4,
       "beta_end": 0.02}


class ZeroNoisePredictor(torch.nn.Module):
    def forward(self, x, t):
        return SimpleNamespace(sample=torch.zeros_like(x))


class TestMath(unittest.TestCase):
    def setUp(self):
        self.scheduler = ddpmScheduler(CFG, "cpu")
        torch.manual_seed(7)

    def test_scheduler(self):
        s = self.scheduler
        self.assertEqual(s.T, 1000)
        self.assertTrue(torch.allclose(s.alphas, 1 - s.betas))
        self.assertTrue(torch.allclose(s.alpha_bar, s.alphas.cumprod(0)))
        self.assertTrue(torch.all(s.alpha_bar[1:] <= s.alpha_bar[:-1]))
        self.assertEqual(s.posterior_variance[0].item(), 0.0)
        self.assertTrue(torch.all(s.posterior_variance >= 0))

    def test_forward_equals_formula(self):
        x0 = torch.randn(4, 1, 16, 16)
        eps = torch.randn_like(x0)
        t = torch.tensor([0, 10, 500, 999], dtype=torch.long)
        xt, returned = forward_scheduler(x0, t, self.scheduler, epsilon=eps)
        a = self.scheduler.sqrt_alpha_bar[t].reshape(4, 1, 1, 1)
        b = self.scheduler.sqrt_one_minus_alpha_bar[t].reshape(4, 1, 1, 1)
        self.assertTrue(torch.equal(returned, eps))
        self.assertTrue(torch.allclose(xt, a*x0 + b*eps, atol=1e-6))
        self.assertEqual(xt.shape, x0.shape)

    def test_reverse_deterministic_and_stochastic(self):
        model = ZeroNoisePredictor()
        x = torch.randn(2, 1, 8, 8)
        at_first = sample_step(model, x, 0, self.scheduler)
        expected_first = x * self.scheduler.sqrt_recip_alpha[0]
        self.assertTrue(torch.allclose(at_first, expected_first, atol=1e-6))
        z = torch.zeros_like(x)
        at_t = sample_step(model, x, 99, self.scheduler, noise=z)
        expected = x * self.scheduler.sqrt_recip_alpha[99]
        self.assertTrue(torch.allclose(at_t, expected, atol=1e-6))

    def test_generate_shape_small_T(self):
        cfg = dict(CFG, timesteps=4)
        model = ZeroNoisePredictor()
        scheduler = ddpmScheduler(cfg)
        result = generate(model, scheduler, (2, 1, 8, 8))
        self.assertEqual(tuple(result.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
