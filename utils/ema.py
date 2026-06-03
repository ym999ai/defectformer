"""
Exponential Moving Average (EMA) of model weights.

EMA maintains a shadow copy of the model with time-averaged parameters.
During inference the EMA model is more stable and consistently gives
+0.3-0.5% AP over the last-epoch checkpoint.

Usage
-----
    ema = ModelEMA(model, decay=0.9999)
    # after every optimizer step:
    ema.update(model)
    # for validation / final eval:
    metrics = evaluate(ema.module, ...)
"""

from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module  = deepcopy(model)
        self.module.eval()
        self.decay   = decay
        self.updates = 0

    # ramp up decay during the first few hundred steps to avoid
    # the shadow model diverging at the very start of training
    def _decay_now(self) -> float:
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self._decay_now()
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)
        # copy non-param buffers (running mean/var of BN) exactly
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    # ── Delegate common attrs so callers can treat EMA like the model ──
    def predict(self, *a, **kw):
        return self.module.predict(*a, **kw)

    def eval(self):
        return self.module.eval()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self.module.load_state_dict(sd, strict=strict)

    def to(self, device):
        self.module = self.module.to(device)
        return self

    def __call__(self, *a, **kw):
        return self.module(*a, **kw)
