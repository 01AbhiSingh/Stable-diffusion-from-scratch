"""MNIST dataset and throughput-oriented DataLoader defaults."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.config import project_path


def make_mnist_loader(config, device):
    ds = config["dataset"]
    preprocessing = transforms.Compose([
        transforms.Resize((ds["image_size"], ds["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(
        root=str(project_path(ds["root"])),
        train=True,
        download=True,
        transform=preprocessing,
    )
    workers = int(ds["num_workers"])
    opts = dict(
        batch_size=int(ds["batch_size"]),
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        pin_memory=(device.type == "cuda" and bool(ds["pin_memory"])),
    )
    if workers > 0:
        opts.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **opts)
