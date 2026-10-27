"""Checks for ⊥AdamW and StableMax.  Run:  python -m pytest tests -q   (or python tests/test_orth.py)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import cross_entropy, log_stablemax  # noqa: E402
from optim import OrthoAdamW  # noqa: E402


def _authors_orthograd_(params):
    """Copy of OrthoGrad._orthogonalize_gradients from the authors' repo, used as the reference."""
    with torch.no_grad():
        for p in params:
            w, g = p.view(-1), p.grad.view(-1)
            proj = torch.dot(w, g) / (torch.dot(w, w) + 1e-30)
            g_orth = g - proj * w
            p.grad.copy_((g_orth * (g.norm(2) / (g_orth.norm(2) + 1e-30))).view_as(p.grad))


def _params(seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(5, 3)), torch.nn.Parameter(torch.randn(7))]
    for p in ps:
        p.grad = torch.randn_like(p) + 0.5 * p.detach()  # gradient partly aligned with w
    return ps


def test_projection_is_orthogonal_and_norm_preserving():
    ps = _params()
    before = [p.grad.norm().item() for p in ps]
    opt = OrthoAdamW(ps, lr=1e-3)
    cos = opt.orthogonalize_()
    assert cos > 0.1  # the test gradient was built to point partly along w
    for p, n in zip(ps, before):
        assert abs(torch.dot(p.detach().view(-1), p.grad.view(-1)).item()) < 1e-5
        assert abs(p.grad.norm().item() - n) < 1e-5


def test_matches_authors_code():
    ours, ref = _params(1), _params(1)
    OrthoAdamW(ours, lr=1e-3).orthogonalize_()
    _authors_orthograd_(ref)
    for a, b in zip(ours, ref):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_global_scope_is_orthogonal_to_all_weights():
    ps = _params(2)
    OrthoAdamW(ps, lr=1e-3, scope="global").orthogonalize_()
    w = torch.cat([p.detach().view(-1) for p in ps])
    g = torch.cat([p.grad.view(-1) for p in ps])
    assert abs(torch.dot(w, g).item()) < 1e-5


def test_no_rescale_shrinks_gradient():
    ps = _params(3)
    before = [p.grad.norm().item() for p in ps]
    OrthoAdamW(ps, lr=1e-3, rescale=False).orthogonalize_()
    assert all(p.grad.norm().item() < n for p, n in zip(ps, before))


def test_step_minimises_a_quadratic():
    # f(w) = ||w - c||^2 with c not parallel to w: ⊥AdamW still reduces it.
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([3.0, 0.5]))
    c = torch.tensor([-1.0, 2.0])
    opt = OrthoAdamW([w], lr=0.05)
    f0 = ((w - c) ** 2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        ((w - c) ** 2).sum().backward()
        opt.step()
    assert ((w - c) ** 2).sum().item() < 0.5 * f0


def test_stablemax_is_a_distribution_and_does_not_collapse():
    x = torch.tensor([[1e4, 0.0, -3.0]], dtype=torch.float32)
    p = log_stablemax(x).exp()
    assert torch.allclose(p.sum(), torch.tensor(1.0))
    assert p[0, 1] > 0  # softmax gives exactly 0 here in float32
    assert torch.softmax(x, -1)[0, 1] == 0
    y = torch.tensor([0])
    assert cross_entropy(x, y, kind="stablemax").item() > 0
    assert cross_entropy(x, y, kind="softmax").item() == 0  # Softmax Collapse: zero loss, zero gradient


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
