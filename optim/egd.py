"""Egalitarian Gradient Descent (EGD).

"Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking", ICLR 2026 (arXiv:2510.04930).

A 2-D gradient G = U S V^T is replaced by U V^T: every non-zero singular direction gets the same step
size. Singular values are clamped at `eps` before inverting, so G_egd = U diag(s / max(s, eps)) V^T.

EGDAdam is the method used in the experiment: the paper's transformer setup as in the authors' code,
U V^T on the MLP weight matrices of every block, then a plain Adam step with weight decay added to
the gradient. EGD (plain gradient descent on U V^T for every 2-D matrix) provides the shared
`egalitarian` routine and is kept for unit tests.
"""
import torch


class EGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, eps=1e-6, rank=None):
        if lr < 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("invalid EGD hyper-parameters")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      eps=eps, rank=rank))

    @staticmethod
    def egalitarian(g, eps=1e-6, rank=None):
        """U diag(s / max(s, eps)) V^T of a 2-D gradient (rank-r randomized SVD if `rank`)."""
        g32 = g.float()
        if rank is None or rank >= min(g32.shape):
            u, s, vh = EGD._svd(g32)
        else:
            u, s, v = torch.svd_lowrank(g32, q=rank, niter=2)
            vh = v.T
        return ((u * (s / s.clamp_min(eps))) @ vh).to(g.dtype)

    @staticmethod
    def _svd(g):
        """Thin SVD that survives non-convergence (seen once at lr 0.3 on CUDA, v0 tuning).

        Same result whenever the default solver converges; otherwise retry in float64 on the CPU,
        then with a relative 1e-7 jitter (changes the direction by ~1e-7, far below float32 noise).
        """
        if not torch.any(g):  # zero gradient (train loss underflowed to 0): direction is 0 anyway
            k = min(g.shape)
            return g.new_zeros(g.shape[0], k), g.new_zeros(k), g.new_zeros(k, g.shape[1])
        try:
            return torch.linalg.svd(g, full_matrices=False)
        except RuntimeError:  # torch._C._LinAlgError subclasses RuntimeError
            pass
        if not torch.isfinite(g).all():
            raise FloatingPointError("EGD: non-finite gradient (the run diverged)")
        g64 = g.detach().double().cpu()
        try:
            u, s, vh = torch.linalg.svd(g64, full_matrices=False)
        except RuntimeError:
            scale = g64.abs().max().clamp_min(1e-30)
            u, s, vh = torch.linalg.svd(g64 + 1e-7 * scale * torch.randn_like(g64), full_matrices=False)
        return u.to(g), s.to(g), vh.to(g)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d = self.egalitarian(p.grad, group["eps"], group["rank"]) if p.dim() == 2 else p.grad
                if mom > 0:
                    buf = self.state[p].get("momentum")
                    if buf is None:
                        buf = self.state[p]["momentum"] = d.clone()
                    else:
                        buf.mul_(mom).add_(d)
                    d = buf
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(d, alpha=-lr)
        return loss


class EGDAdam(torch.optim.Adam):
    """EGD as the authors ran it on the Grokfast transformer (egd/grokfast-comparison/main_modmult.py,
    branch egd; App. D of the paper): only the MLP weight matrices of each block get the egalitarian
    gradient U V^T (clamp 1e-9, as there), then a plain torch Adam step with *coupled* L2 weight decay
    (their run line: --mode egd --weight_decay 0.005, lr 1e-3, betas 0.9/0.98). Every other tensor
    gets its raw gradient. `egd_params` are the tensors to precondition (main.py passes mlp[0], mlp[2]).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0,
                 egd_params=(), egd_eps=1e-9):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.egd_ids = {id(p) for p in egd_params}
        self.egd_eps = egd_eps
        if not self.egd_ids:
            raise ValueError("EGDAdam needs egd_params (the MLP weight matrices)")

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and id(p) in self.egd_ids:
                    p.grad.copy_(EGD.egalitarian(p.grad, self.egd_eps))
        return super().step(closure)
