"""Function-Preserving Decay (FPD) for the modular multiplication experiment in main.py.

`--optimizer FPD` trains with Adam. Once the training accuracy reaches `--start_train_acc`, it runs one FPD cycle
every `--fpd_every` steps: cache the answer distributions on the training set, shrink all parameters by `--fpd_s`,
then restore the cached distributions with `--fpd_steps` steps of a separate Adam. The main Adam's state is kept.
To first order the net move is a decay inside the null space of the training-prediction Jacobian, without forming
any projector.

    python main.py --label fpd --optimizer FPD --stop_after_grok 200
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def answer_log_probs(m: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(m(inputs[:-1])[-1], dim=-1)


def param_norm(m: nn.Module) -> float:
    return torch.sqrt(sum(p.detach().float().square().sum() for p in m.parameters())).item()


def function_preserving_decay_(m: nn.Module, inputs: torch.Tensor, s: float, steps: int, lr: float) -> Dict[str, float]:
    """Shrink the weights, then restore the training-set predictions."""
    with torch.no_grad():
        target = answer_log_probs(m, inputs)
    kl = lambda log_probs: F.kl_div(log_probs, target, log_target=True, reduction="batchmean")

    stats = {"norm_before": param_norm(m)}
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(1 - s)
        stats["norm_shrunk"] = param_norm(m)
        stats["kl_shrunk"] = kl(answer_log_probs(m, inputs)).item()

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        loss = kl(answer_log_probs(m, inputs))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        log_probs = answer_log_probs(m, inputs)
        stats["kl"] = kl(log_probs).item()
        stats["agree"] = (log_probs.argmax(-1) == target.argmax(-1)).float().mean().item()
    stats["norm_after"] = param_norm(m)
    return stats


class FPD:
    """The calls main.py makes into this file. Without `--optimizer FPD` they change nothing."""

    def __init__(self, model: nn.Module, train_data: torch.Tensor, args):
        self.model, self.args = model, args
        self.train_inputs = train_data.to(next(model.parameters()).device)
        self.enabled = args.optimizer == "FPD"
        # Unified study: AdamW, so that --weight_decay is decoupled (lr * wd) like every other arm.
        # At --weight_decay 0 (FPD's own setting) AdamW and Adam are the same update.
        self.base_optimizer = "AdamW" if self.enabled else args.optimizer
        self.active = False
        self.history = []

    def step(self, i: int, train_acc: List[float]) -> None:
        args = self.args
        if self.enabled and train_acc and train_acc[-1] >= args.start_train_acc:
            self.active = True
        if self.active and i % args.fpd_every == 0:
            stats = function_preserving_decay_(
                self.model, self.train_inputs, s=args.fpd_s, steps=args.fpd_steps, lr=args.fpd_lr,
            )
            self.history.append({"step": i, **stats})

    def should_stop(self, val_acc: List[float]) -> bool:
        # True once val >= 0.95 has held for the last stop_after_grok epochs; 0 disables early stopping.
        n = self.args.stop_after_grok
        return n > 0 and len(val_acc) >= n and min(val_acc[-n:]) >= 0.95


def add_args(parser) -> None:
    parser.add_argument("--start_train_acc", type=float, default=0.99)
    parser.add_argument("--fpd_s", type=float, default=0.1)
    parser.add_argument("--fpd_every", type=int, default=25)
    parser.add_argument("--fpd_steps", type=int, default=20)
    parser.add_argument("--fpd_lr", type=float, default=1e-3)
    parser.add_argument("--stop_after_grok", type=int, default=0)


def label_suffix(args) -> str:
    return f"_FPD_s{args.fpd_s:g}_T{args.fpd_every}_K{args.fpd_steps}" if args.optimizer == "FPD" else ""
