"""Optimiser registry. main.py looks names up in REGISTRY first and falls back to torch.optim
(so --optimizer AdamW / Adam / SGD still work).

    OrthoAdamW  ⊥AdamW, "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697)
    Lion        "Symbolic Discovery of Optimization Algorithms" (arXiv:2302.06675)
    EGDAdam     EGD in its paper's transformer setup (arXiv:2510.04930): U V^T on MLP weights, then Adam
    EGD         plain gradient descent on U V^T (shared routine, unit tests)
    FPD         not an optimiser: AdamW + Function-Preserving Decay cycles, see fpd.py
"""
import inspect

import torch

from .egd import EGD, EGDAdam
from .lion import Lion
from .orth import OrthoAdamW

REGISTRY = {
    "OrthoAdamW": OrthoAdamW,
    "Lion": Lion,
    "EGD": EGD,
    "EGDAdam": EGDAdam,
}


def build_optimizer(name, params, **kwargs):
    """Build `name`, passing only the keyword arguments its constructor names
    (Lion takes no eps, EGD/SGD take no betas, torch classes take no ⊥ options)."""
    cls = REGISTRY.get(name) or getattr(torch.optim, name)
    accepted = inspect.signature(cls.__init__).parameters
    return cls(params, **{k: v for k, v in kwargs.items() if k in accepted})
