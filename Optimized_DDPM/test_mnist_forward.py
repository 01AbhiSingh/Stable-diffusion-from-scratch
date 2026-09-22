import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusers import UNet2DModel

from scheduler import ddpmScheduler
from forward_pass import forward_scheduler

from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print (device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset,batch_size = 16, shuffle=True)

scheduler = ddpmScheduler(T = 1000, device = device)

model = get_model().to(device)

optmizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

for step, (x0,_) in enumerate(dataloader):
    x0 = x0.to(device)

    # --------------------------------
    # Sample timestep independently
    # for every image
    #
    # Shape:
    # [B]
    # --------------------------------


    t = torch.randint(
        low = 0,
        high = scheduler.T,
        size = (x0.shape[0],),
        device =device
    )

    # --------------------------------
    # Forward diffusion
    #
    # x0 -> xt
    # --------------------------------

    xt, epsilon = forward_scheduler(
        x0,
        t,
        scheduler
    )

    epsilon_pred = model(xt, t).sample

    # --------------------------------
    # DDPM loss
    #
    # || epsilon - epsilon_theta ||^2
    # --------------------------------

    loss = F.mse_loss(
        epsilon_pred,
        epsilon
    )

    # --------------------------------
    # Backpropagation
    # --------------------------------
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()

    # --------------------------------
    # Debug information
    # --------------------------------

    print(
        f"Step: {step} | "
        f"Loss: {loss.item():.6f}"
    )

    if step == 9:
        break

