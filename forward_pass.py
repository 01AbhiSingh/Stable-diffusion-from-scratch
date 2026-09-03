import torch

def forward_scheduler(x0, t, scheduler):
    """
        Performs q(x_t | x_0)

        Args:

            x0:
                clean image tensor
                Shape:
                [B,C,H,W]

            t:
                timestep tensor
                Shape:
                [B]

            scheduler:
                DDPM scheduler object


        Returns:

            xt:
                noisy image

            epsilon:
                noise added
    """

    #Sample Gaussian noise
    epsilon = torch.randn_like(x0)

    # 2. Get alpha_bar coefficients
    # for selected timestep
    sqrt_alpha_bar_t = (scheduler.sqrt_alpha_bar[t])
    sqrt_one_minus_alpha_bar_t = (scheduler.sqrt_one_minus_alpha_bar[t])


    # 3. Reshape for broadcasting
    # [B] -> [B,1,1,1]

    sqrt_alpha_bar_t = (sqrt_alpha_bar_t[:, None, None, None])

    sqrt_one_minus_alpha_bar_t = (sqrt_one_minus_alpha_bar_t[:, None, None, None])


    # 4. Apply DDPM equation
    xt = (sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t *epsilon)

    return xt, epsilon