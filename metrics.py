"""Measurements shared by every optimiser, so the team's results are comparable.

Grok-timing definitions (evaluated once per epoch, steps = optimiser updates):
    fit_step       first step with train accuracy >= thresh
    grok_step      first step with val accuracy >= thresh
    sustained_step first step after which val accuracy stays >= thresh until the end of the run
    delay          grok_step - fit_step (length of the plateau)
A run that never reaches the threshold reports None, not a number.
"""
import torch


@torch.no_grad()
def weight_norm(model):
    """Total L2 norm of all trainable parameters."""
    return torch.sqrt(sum((p.float() ** 2).sum() for p in model.parameters() if p.requires_grad)).item()


@torch.no_grad()
def spectral_norm_max(model):
    """Largest singular value over all 2-D weight matrices (embeddings, attention, MLP, head)."""
    best = 0.0
    for p in model.parameters():
        if p.requires_grad and p.dim() == 2:
            best = max(best, torch.linalg.svdvals(p.float().cpu())[0].item())
    return best


@torch.no_grad()
def grad_weight_cos(model):
    """Gradient-norm-weighted mean over tensors of cos(g, w).

    Positive: the gradient points along the weights, so a descent step shrinks them.
    Negative: a descent step grows the weights. Prieto et al. call steps that mostly
    scale the weights up (and so the logits) naive loss minimisation; ⊥AdamW removes them.
    """
    nums, dens = [], []
    for p in model.parameters():
        if p.grad is None:
            continue
        w, g = p.reshape(-1).float(), p.grad.reshape(-1).float()
        gn = g.norm()
        nums.append(torch.dot(w, g) / (w.norm() + 1e-30))  # = cos * ||g||
        dens.append(gn)
    if not dens:
        return 0.0
    den = torch.stack(dens).sum()
    return (torch.stack(nums).sum() / (den + 1e-30)).item()  # one device sync per call


def grok_summary(steps, train_acc, val_acc, thresh=0.99):
    def first(xs):
        return next((s for s, a in zip(steps, xs) if a >= thresh), None)

    fit, grok = first(train_acc), first(val_acc)
    sustained = None
    for s, a in zip(reversed(steps), reversed(val_acc)):
        if a < thresh:
            break
        sustained = s
    return {
        "thresh": thresh,
        "fit_step": fit,
        "grok_step": grok,
        "sustained_step": sustained,
        "delay": grok - fit if grok is not None and fit is not None else None,
        "final_train_acc": train_acc[-1] if train_acc else None,
        "final_val_acc": val_acc[-1] if val_acc else None,
    }
