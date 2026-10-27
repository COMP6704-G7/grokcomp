"""Optimiser registry. Each member adds their class here; main.py looks names up in REGISTRY
first and falls back to torch.optim (so --optimizer AdamW / Adam still work)."""
import torch

from .orth import OrthoAdamW

REGISTRY = {
    "OrthoAdamW": OrthoAdamW,
}


def build_optimizer(name, params, **kwargs):
    if name in REGISTRY:
        return REGISTRY[name](params, **kwargs)
    # torch.optim classes do not accept our extra options
    for key in ("scope", "rescale", "skip_1d"):
        kwargs.pop(key, None)
    return getattr(torch.optim, name)(params, **kwargs)
