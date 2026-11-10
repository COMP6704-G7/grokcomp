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
    Negative: a descent step grows the weights. "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697) calls steps that mostly
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


def optimizer_state_bytes(optimizer):
    """Bytes held in the optimiser's persistent per-parameter state tensors (Adam: 2 buffers, Lion: 1)."""
    return int(sum(v.numel() * v.element_size() for st in optimizer.state.values()
                   for v in st.values() if torch.is_tensor(v) and v.dim() > 0))


def unified_metrics(steps, train_acc, val_acc, every=10, thresholds=(0.95, 0.99), consec=3):
    """Metrics of the unified study, for each threshold (None = not reached within the budget).

    first      first epoch-end with acc >= thr (evaluated every epoch = 10 steps)
    consec3    start of the first run of `consec` evaluations in a row >= thr, evaluations taken
               every `every` epochs (= every 100 steps), as in the Lion / EGD protocols
    sustained  first step after which acc stays >= thr to the end of the run
    min_after  lowest val acc after `consec3` (stability after grokking)
    """
    out = {}
    sub = list(range(0, len(steps), every))
    for thr in thresholds:
        k = f"{int(round(thr * 100))}"
        first = lambda xs: next((s for s, a in zip(steps, xs) if a >= thr), None)
        run, c3, c3_idx = 0, None, None
        for j in sub:
            run = run + 1 if val_acc[j] >= thr else 0
            if run == consec:
                c3_idx = sub[sub.index(j) - consec + 1]
                c3 = steps[c3_idx]
                break
        sustained = None
        for s, a in zip(reversed(steps), reversed(val_acc)):
            if a < thr:
                break
            sustained = s
        out[f"train{k}_first"] = first(train_acc)
        out[f"val{k}_first"] = first(val_acc)
        out[f"val{k}_consec3"] = c3
        out[f"val{k}_sustained"] = sustained
        out[f"val{k}_min_after"] = min(val_acc[c3_idx:]) if c3_idx is not None else None
    n = len(val_acc)
    tail = val_acc[max(0, n - n // 10):]  # last 10% of the run
    out["final_val_acc"] = val_acc[-1] if val_acc else None
    out["tail10_mean_val_acc"] = sum(tail) / len(tail) if tail else None
    return out
