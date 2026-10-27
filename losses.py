"""Softmax and StableMax cross-entropy, computed in a chosen float precision.

StableMax (Prieto et al., ICLR 2025) replaces exp in the softmax with
    s(x) = x + 1         for x >= 0
    s(x) = 1 / (1 - x)   for x <  0
which grows linearly instead of exponentially, so large logits cannot push the
probabilities to exactly 0/1 in floating point (Softmax Collapse).
Matches utils.py in the authors' repository.
"""
import torch
import torch.nn.functional as F

DTYPES = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


def _s(x, eps=1e-30):
    return torch.where(x < 0, 1 / (1 - x + eps), x + 1)


def log_stablemax(x, dim=-1):
    s_x = _s(x)
    return torch.log(s_x / s_x.sum(dim=dim, keepdim=True))


def _cast(logits, dtype):
    # MPS has no float64; do the loss on the CPU (autograd follows the copy)
    if dtype == torch.float64 and logits.device.type == "mps":
        logits = logits.cpu()
    return logits.to(dtype)


def cross_entropy(logits, targets, kind="softmax", dtype="float32"):
    logits = _cast(logits, DTYPES[dtype])
    targets = targets.to(logits.device)
    if kind == "softmax":
        logp = F.log_softmax(logits, dim=-1)
    elif kind == "stablemax":
        logp = log_stablemax(logits, dim=-1)
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return -logp.gather(-1, targets[:, None]).mean()


@torch.no_grad()
def softmax_collapse_fraction(logits, targets):
    """Share of samples whose float32 softmax loss is exactly 0, i.e. no gradient flows from them.
    This is the Softmax Collapse measure from Prieto et al."""
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None].to(logits.device))
    return (logp == 0).float().mean().item()
