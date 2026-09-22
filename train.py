import os

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scheduler import ddpmScheduler
from forward_pass import forward_scheduler
from model import get_model


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

batch_size = 64
epochs = 15
learning_rate = 1e-4

checkpoint_dir = "./checkpoints"

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)


# ============================================================
# DATASET TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# ============================================================
# MNIST DATASET
# ============================================================

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


print(
    "Training samples:",
    len(dataset)
)


# ============================================================
# DDPM SCHEDULER
# ============================================================

scheduler = ddpmScheduler(
    T=1000,
    device=device
)


# ============================================================
# MODEL
# ============================================================

model = get_model().to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# TRAINING MODE
# ============================================================

model.train()


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(epochs):

    epoch_loss = 0.0


    for batch_idx, (x0, _) in enumerate(dataloader):

        # ----------------------------------------------------
        # Move clean images to GPU
        # ----------------------------------------------------

        x0 = x0.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Sample timestep for every image
        #
        # t shape:
        # [B]
        # ----------------------------------------------------

        t = torch.randint(
            low=0,
            high=scheduler.T,
            size=(x0.shape[0],),
            device=device,
            dtype=torch.long
        )


        # ----------------------------------------------------
        # Forward diffusion
        #
        # q(x_t | x_0)
        # ----------------------------------------------------

        xt, epsilon = forward_scheduler(
            x0,
            t,
            scheduler
        )


        # ----------------------------------------------------
        # Predict noise
        #
        # epsilon_theta(x_t, t)
        # ----------------------------------------------------

        epsilon_pred = model(
            xt,
            t
        ).sample


        # ----------------------------------------------------
        # DDPM LOSS
        #
        # || epsilon - epsilon_theta ||^2
        # ----------------------------------------------------

        loss = F.mse_loss(
            epsilon_pred,
            epsilon
        )


        # ----------------------------------------------------
        # GRADIENT UPDATE
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        epoch_loss += loss.item()


        if batch_idx % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}"
            )


    # ========================================================
    # AVERAGE EPOCH LOSS
    # ========================================================

    average_loss = (
        epoch_loss /
        len(dataloader)
    )


    print(
        f"\nEpoch {epoch + 1} completed"
    )

    print(
        f"Average loss: {average_loss:.6f}\n"
    )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"ddpm_mnist_epoch_{epoch + 1}.pt"
    )


    torch.save(
        {
            "epoch": epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                average_loss,
        },

        checkpoint_path
    )


    print(
        "Saved:",
        checkpoint_path
    )


print("\nTraining complete.")