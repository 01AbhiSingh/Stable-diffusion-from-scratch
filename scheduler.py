import torch

class ddpmScheduler:
    def __init__(sellf, T = 1000, beta_start = 1e-4,beta_end = 0.2):

        self.T = T

        #beta_schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas

        #alspha_bar 

        self.alpha_bar = torch.cumprod(self.alphas, dim = 0)

        #fwd coeff
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)

        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar)

        #rev coeff
        self.sqrt_recip_alpha = torch.sqrt(1/self.aphas)

        