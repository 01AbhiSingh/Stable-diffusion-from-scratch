import torch

from scheduler import ddpmScheduler
from forward_pass import forward_scheduler


scheduler = ddpmScheduler()


x0 = torch.randn(
    1,
    1,
    28,
    28
)


epsilon = torch.randn_like(x0)


# =============================
# t = 0
# =============================

t0 = torch.tensor(
    [0],
    dtype=torch.long
)


xt0, _ = forward_scheduler(
    x0,
    t0,
    scheduler,
    epsilon
)


difference_from_x0 = (
    xt0 - x0
).abs().mean()


print(
    "t=0 mean difference from x0:",
    difference_from_x0.item()
)


# =============================
# t = 999
# =============================

t999 = torch.tensor(
    [999],
    dtype=torch.long
)


xt999, _ = forward_scheduler(
    x0,
    t999,
    scheduler,
    epsilon
)


difference_from_noise = (
    xt999 - epsilon
).abs().mean()


print(
    "t=999 mean difference from epsilon:",
    difference_from_noise.item()
)
print(
    "alpha_bar[0]:",
    scheduler.alpha_bar[0]
)

print(
    "alpha_bar[999]:",
    scheduler.alpha_bar[999]
)


print(
    "signal coefficient t=0:",
    scheduler.sqrt_alpha_bar[0]
)

print(
    "noise coefficient t=0:",
    scheduler.sqrt_one_minus_alpha_bar[0]
)


print(
    "signal coefficient t=999:",
    scheduler.sqrt_alpha_bar[999]
)

print(
    "noise coefficient t=999:",
    scheduler.sqrt_one_minus_alpha_bar[999]
)