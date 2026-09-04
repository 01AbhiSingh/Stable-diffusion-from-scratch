import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from scheduler import ddpmScheduler
from forward_pass import forward_scheduler


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5,))])

dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download=True,
    transform=transform
)

x0, label = dataset[0]

print("Label:", label)
print("Single image shape:", x0.shape)
# Add batch dimension
#
# Before:
# [C,H,W]
#
# After:
# [B,C,H,W]
print("before",x0[0])
x0 = x0.unsqueeze(0)
print("after",x0[0])


print("Batch image shape:", x0.shape)

scheduler = ddpmScheduler()

timesteps = [0,250,500,750,999]


plt.figure(figsize=(15, 3))


# Original image

plt.subplot(1, 6, 1)

plt.imshow(
    x0[0, 0].cpu(),
    cmap="gray"
)

plt.title("x0")
plt.axis("off")

for i , timestep in enumerate(timesteps):
    t = torch.tensor([timestep], dtype=torch.long)

    xt, epsilon = forward_scheduler(x0, t, scheduler)

    plt.subplot(1,6,i+2)
    plt.imshow(xt[0,0].detach().cpu(),cmap = "gray")
    plt.title(f"t={timestep}")

    plt.axis("off")

plt.tight_layout()
plt.show()