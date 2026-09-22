import torch

from scheduler import ddpmScheduler
from forward_pass import forward_scheduler



# Create scheduler

scheduler = ddpmScheduler()



# Fake image batch

x0 = torch.randn(
    4,
    1,
    28,
    28
)



# Random timestep for each image

t = torch.randint(
    0,
    1000,
    (4,)
)



# Apply diffusion

xt, epsilon = forward_scheduler(
    x0,
    t,
    scheduler
)



print("---------------------")

print(
    "x0 shape:",
    x0.shape
)


print(
    "t shape:",
    t.shape
)


print(
    "xt shape:",
    xt.shape
)


print(
    "epsilon shape:",
    epsilon.shape
)


print("---------------------")


print(
    "Timesteps:",
    t
)