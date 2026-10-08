"""Optional integration test; automatically skipped if diffusers is missing."""
import importlib.util
import unittest
import torch


@unittest.skipUnless(importlib.util.find_spec("diffusers"), "diffusers not installed")
class TestModel(unittest.TestCase):
    def test_unet_io(self):
        from models.unet import get_model
        from utils.config import load_config
        cfg = load_config()
        # Smaller network for a quick CPU interface test.
        cfg["model"]["block_out_channels"] = [32, 32]
        cfg["model"]["down_block_types"] = ["DownBlock2D", "DownBlock2D"]
        cfg["model"]["up_block_types"] = ["UpBlock2D", "UpBlock2D"]
        cfg["model"]["layers_per_block"] = 1
        model = get_model(cfg).eval()
        with torch.no_grad():
            predicted = model(torch.randn(1, 1, 32, 32), torch.tensor([50])).sample
        self.assertEqual(tuple(predicted.shape), (1, 1, 32, 32))
