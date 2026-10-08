"""Save/load/resume state is compatible with weights_only=True."""
import tempfile
import unittest
from pathlib import Path
import torch

from utils.checkpoint import save_checkpoint, load_training_state, read_checkpoint
from utils.config import load_config
from utils.ema import EMA
from utils.precision import make_scaler


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        cfg = load_config()
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = make_scaler("none", torch.device("cpu"))
        ema = EMA(model, 0.9)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "last.pt"
            save_checkpoint(checkpoint, epoch=2, global_step=13, model=model,
                            optimizer=optimizer, scaler=scaler, ema=ema,
                            config=cfg, loss=0.7)
            saved = read_checkpoint(checkpoint)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["global_step"], 13)
            copy_model = torch.nn.Linear(3, 2)
            copy_optim = torch.optim.Adam(copy_model.parameters(), lr=1e-4)
            copy_ema = EMA(copy_model, 0.9)
            epoch, step = load_training_state(
                checkpoint, model=copy_model, optimizer=copy_optim,
                scaler=scaler, ema=copy_ema, config=cfg,
            )
            self.assertEqual((epoch, step), (2, 13))
            self.assertTrue(torch.equal(model.weight, copy_model.weight))
