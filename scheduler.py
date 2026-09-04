import torch

class ddpmScheduler:
    def __init__(self, T = 1000, beta_start = 1e-4,beta_end = 0.02, device = "cpu"):

        self.T = T
        self.device =device
        #beta_schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device =device)
        self.alphas = 1 - self.betas

        #alspha_bar 

        self.alpha_bar = torch.cumprod(self.alphas, dim = 0)

        #fwd coeff
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)

        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar)

        #rev coeff
        self.sqrt_recip_alpha = torch.sqrt(1/self.alphas)

        alpha_bar_prev = torch.cat(
            [
                torch.tensor([1.0], device=device),
                self.alpha_bar[:-1]
            ]
        )

        self.posterior_vairance = (self.betas * (1-alpha_bar_prev)/(1-self.alpha_bar))