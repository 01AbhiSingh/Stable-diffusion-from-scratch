import torch 

@torch.no_grad()

def sample_step(model,x_t,t,scheduler):
    """
    Performs one reverse DDPM step:

    x_t -> x_(t-1)


    Args:

        model:
            noise prediction network

        x_t:
            current noisy image
            [B,C,H,W]

        t:
            timestep
            [B]

        scheduler:
            DDPM scheduler


    Returns:

        x_prev:
            previous timestep image
    """


    # 1. Predict noise
    # epsilon_theta(x_t,t)

    epsilon_pred= model(x_t, t).sample

    #2extract scheduler values
    beta_t =scheduler.betas[t]
    alpha_t =scheduler.alphas[t]
    alpha_bar_t=scheduler.alpha_bar[t]

    # 3. Reshape for broadcasting
    #
    # [B] -> [B,1,1,1]
    beta_t = beta_t[:, None, None, None]
    alpha_t = alpha_t[:, None, None, None]
    alpha_bar_t = (alpha_bar_t[:, None, None, None])

    # ------------------------------------------------
    # 4. Reverse mean equation
    #
    # μθ =
    #
    # 1/sqrt(alpha_t)
    #
    # (
    # x_t -
    # beta_t/sqrt(1-alpha_bar_t)
    # * epsilon_pred
    # )
    #
    # ------------------------------------------------


    mean = (1 / torch.sqrt(alpha_t)) * (x_t-(beta_t/torch.sqrt(1 - alpha_bar_t))*epsilon_pred)

    # ------------------------------------------------
    # 5. Add noise
    #
    # only if t > 0
    # ------------------------------------------------

    if t[0] > 0:

        posterior_variance = (scheduler.posterior_variance[t])

        posterior_variance = (posterior_variance[:,None,None,None])
        noise = torch.randn_like(x_t)


        x_prev = (mean+torch.sqrt(posterior_variance)*noise)
    
    else:

        # Final step:
        # no additional noise

        x_prev = mean



    return x_prev





