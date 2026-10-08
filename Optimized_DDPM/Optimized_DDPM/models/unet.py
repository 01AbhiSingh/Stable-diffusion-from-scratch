"""Network factory only; DDPM maths live in diffusion/ modules.

To load Phase 1 checkpoints, leave block types/channels/layers unchanged.
"""
from diffusers import UNet2DModel


def get_model(config):
    ds, m = config["dataset"], config["model"]
    return UNet2DModel(
        sample_size=int(ds["image_size"]),
        in_channels=int(ds["channels"]),
        out_channels=int(ds["channels"]),
        layers_per_block=int(m["layers_per_block"]),
        block_out_channels=tuple(m["block_out_channels"]),
        down_block_types=tuple(m["down_block_types"]),
        up_block_types=tuple(m["up_block_types"]),
    )
