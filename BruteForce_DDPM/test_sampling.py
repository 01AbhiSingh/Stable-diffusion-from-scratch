import torch

from scheduler import ddpmScheduler
from model import get_model
from sampling import sample_step



device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


scheduler = ddpmScheduler(
    T=1000,
    device=device
)


model = get_model().to(device)


# Load your trained checkpoint

checkpoint = torch.load(
    "./checkpoints/ddpm_mnist_epoch_15.pt",
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()



# Start from random noise

x_t = torch.randn(
    4,
    1,
    32,
    32,
    device=device
)



# One reverse step

t = torch.full(
    (4,),
    999,
    device=device,
    dtype=torch.long
)



x_prev = sample_step(
    model,
    x_t,
    t,
    scheduler
)



print(
    "Current:",
    x_t.shape
)

print(
    "Previous:",
    x_prev.shape
)