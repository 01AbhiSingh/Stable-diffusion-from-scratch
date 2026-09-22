import torch
import torchvision.utils as vutils

from model import get_model
from scheduler import ddpmScheduler
from sampling import sample_step



# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)



# ============================================================
# PARAMETERS
# ============================================================

num_samples = 16

checkpoint_path = (
    "./checkpoints/"
    "ddpm_mnist_epoch_15.pt"
)



# ============================================================
# SCHEDULER
# ============================================================

scheduler = ddpmScheduler(
    T=1000,
    device=device
)



# ============================================================
# MODEL
# ============================================================

model = get_model().to(device)



checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()



# ============================================================
# START FROM PURE GAUSSIAN NOISE
# ============================================================

x = torch.randn(
    num_samples,
    1,
    32,
    32,
    device=device
)


print(
    "Initial noise:",
    x.shape
)



# ============================================================
# REVERSE DIFFUSION LOOP
# ============================================================


for timestep in reversed(
    range(scheduler.T)
):

    t = torch.full(
        (
            num_samples,
        ),
        timestep,
        device=device,
        dtype=torch.long
    )


    x = sample_step(
        model,
        x,
        t,
        scheduler
    )


    if timestep % 100 == 0:

        print(
            "Sampling timestep:",
            timestep
        )



# ============================================================
# POST PROCESS
# ============================================================

# Model works in [-1,1]

# Convert back to [0,1]

x = (
    x.clamp(-1,1)
    + 1
) / 2



# ============================================================
# SAVE IMAGE GRID
# ============================================================

vutils.save_image(
    x,
    "generated_samples.png",
    nrow=4
)


print(
    "Saved generated_samples.png"
)	
