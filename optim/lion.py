"""Lion: one momentum buffer and a decoupled weight decay update.

Algorithm reference: "Symbolic Discovery of Optimization Algorithms", arXiv:2302.06675.
The separately downloaded official implementation is used only in tests.
"""
import math

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two numbers in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or parameter.is_complex():
                    raise RuntimeError("Lion supports dense real gradients only")
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                direction = (beta1 * momentum + (1 - beta1) * gradient).sign_()
                parameter.add_(direction, alpha=-group["lr"])
                momentum.mul_(beta2).add_(gradient, alpha=1 - beta2)
        return loss
