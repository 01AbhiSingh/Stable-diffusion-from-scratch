"""Exponential moving average of network weights for stable DDPM sampling."""
import copy
import torch


class EMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, training_model):
        # Parameters are float32 master weights under AMP.
        for ema_param, param in zip(self.model.parameters(), training_model.parameters()):
            ema_param.lerp_(param.detach(), 1.0 - self.decay)
        # Copy non-learnable buffers (GroupNorm-based UNet has few/none).
        for ema_buf, buf in zip(self.model.buffers(), training_model.buffers()):
            ema_buf.copy_(buf)
        self.updates += 1

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.updates = int(state["updates"])
