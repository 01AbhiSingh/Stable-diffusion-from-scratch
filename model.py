from diffusers import UNet2DModel

def get_model():
    model = UNet2DModel(
        sample_size=32,
        in_channels = 1,
        out_channels = 1,

        layers_per_block = 2,

        block_out_channels=(64, 128, 256),

        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ),

        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
)
    return model