"""⊥AdamW: AdamW on gradients projected orthogonal to the weights.

Prieto et al., "Grokking at the Edge of Numerical Stability", ICLR 2025 (arXiv:2501.04697).
Reference code: github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py

Once the training set is fitted, the gradient lines up with the weights themselves
("naive loss minimisation"): following it only scales the logits up, which lowers the loss
without changing any prediction and eventually causes Softmax Collapse. ⊥Grad removes the
component of the gradient along w before the AdamW update:

    g_perp = g - (<w, g> / <w, w>) * w
    g_perp <- g_perp * ||g|| / ||g_perp||      (rescale, as in the authors' code)

Options beyond the paper, for our ablations:
    scope   "tensor" (authors): project each parameter tensor separately.
            "global": project the concatenation of all parameters in a group at once.
    rescale True (authors): give g_perp the norm of the original g. False: keep the shorter g_perp.
    skip_1d True: leave 1-D tensors (biases, LayerNorm gains) unprojected.

Every step also records cos(g, w), the share of the gradient that points along the weights,
in `self.last_stats` so the training loop can log it.
"""
import torch

EPS = 1e-30


class OrthoAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0,
                 scope="tensor", rescale=True, skip_1d=False, **kwargs):
        if scope not in ("tensor", "global"):
            raise ValueError(f"scope must be 'tensor' or 'global', got {scope!r}")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        self.scope = scope
        self.rescale = rescale
        self.skip_1d = skip_1d
        self.last_stats = {}

    def _targets(self, group):
        return [p for p in group["params"]
                if p.grad is not None and not (self.skip_1d and p.dim() <= 1)]

    @torch.no_grad()
    def orthogonalize_(self):
        """Project every gradient in place. Returns the gradient-norm-weighted mean cos(g, w)."""
        cos_sum, weight_sum = 0.0, 0.0
        for group in self.param_groups:
            params = self._targets(group)
            if not params:
                continue
            if self.scope == "tensor":
                for p in params:
                    w, g = p.reshape(-1), p.grad.reshape(-1)
                    c, gn = _project_(w, g, self.rescale)
                    cos_sum += c * gn
                    weight_sum += gn
            else:
                w = torch.cat([p.reshape(-1) for p in params])
                g = torch.cat([p.grad.reshape(-1) for p in params])
                c, gn = _project_(w, g, self.rescale)
                cos_sum += c * gn
                weight_sum += gn
                offset = 0
                for p in params:
                    n = p.numel()
                    p.grad.copy_(g[offset:offset + n].view_as(p.grad))
                    offset += n
        self.last_stats = {"cos_gw": cos_sum / weight_sum if weight_sum > 0 else 0.0}
        return self.last_stats["cos_gw"]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.orthogonalize_()
        super().step()
        return loss


def _project_(w, g, rescale):
    """Remove the w-component of g in place. Returns (cos(g, w), ||g||) measured before projection."""
    g_norm = g.norm()
    w_dot_g = torch.dot(w, g)
    w_sq = torch.dot(w, w) + EPS
    cos = (w_dot_g / (w_sq.sqrt() * (g_norm + EPS))).item()
    g.sub_(w * (w_dot_g / w_sq))
    if rescale:
        g.mul_(g_norm / (g.norm() + EPS))
    return cos, g_norm.item()
