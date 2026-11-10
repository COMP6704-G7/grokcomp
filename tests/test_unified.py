"""Checks for the optimisers ported into the unified study.  python -m pytest tests -q  (or python tests/test_unified.py)"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metrics import optimizer_state_bytes, unified_metrics  # noqa: E402
from optim import EGD, Lion, build_optimizer  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "vendor"))
from lion_official import Lion as LionOfficial  # noqa: E402


def _pair(seed=0):
    torch.manual_seed(seed)
    a = [torch.nn.Parameter(torch.randn(6, 4)), torch.nn.Parameter(torch.randn(5))]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    return a, b


def test_lion_matches_official():
    a, b = _pair()
    oa = Lion(a, lr=3e-4, betas=(0.9, 0.99), weight_decay=5.0)
    ob = LionOfficial(b, lr=3e-4, betas=(0.9, 0.99), weight_decay=5.0)
    for t in range(5):
        torch.manual_seed(100 + t)
        grads = [torch.randn_like(p) for p in a]
        for p, q, g in zip(a, b, grads):
            p.grad, q.grad = g.clone(), g.clone()
        oa.step(), ob.step()
    for p, q in zip(a, b):
        assert torch.allclose(p, q, atol=1e-7)


def test_egd_direction_has_unit_singular_values():
    torch.manual_seed(0)
    g = torch.randn(12, 7)
    d = EGD.egalitarian(g)
    s = torch.linalg.svdvals(d)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    assert torch.allclose(d, u @ vh, atol=1e-5)


def test_egd_svd_fallback(monkeypatch=None):
    torch.manual_seed(1)
    g = torch.randn(9, 5)
    ref = EGD.egalitarian(g)
    real = torch.linalg.svd
    calls = {"n": 0}

    def flaky(x, *a, **k):
        calls["n"] += 1
        if x.dtype == torch.float32:
            raise getattr(torch._C, "_LinAlgError", RuntimeError)("forced non-convergence")
        return real(x, *a, **k)

    torch.linalg.svd = flaky
    try:
        d = EGD.egalitarian(g)
    finally:
        torch.linalg.svd = real
    assert calls["n"] == 2 and d.dtype == g.dtype
    assert torch.allclose(d, ref, atol=1e-5)


def test_egd_zero_gradient_gives_zero_direction():
    d = EGD.egalitarian(torch.zeros(7, 4))
    assert d.shape == (7, 4) and not torch.any(d)


def test_egd_step_is_sgd_on_egalitarian_gradient_plus_decoupled_decay():
    a, _ = _pair()
    w0 = [p.detach().clone() for p in a]
    for p in a:
        p.grad = torch.randn_like(p)
    expect = [w * (1 - 0.1 * 0.5) - 0.1 * (EGD.egalitarian(p.grad) if p.dim() == 2 else p.grad)
              for w, p in zip(w0, a)]
    EGD(a, lr=0.1, weight_decay=0.5).step()
    for p, e in zip(a, expect):
        assert torch.allclose(p, e, atol=1e-6)


def test_build_optimizer_filters_kwargs():
    common = dict(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-8, scope="tensor",
                  rescale=True, skip_1d=False, momentum=0.0, rank=None)
    for name in ["AdamW", "Adam", "OrthoAdamW", "Lion", "EGD", "SGD"]:
        build_optimizer(name, [torch.nn.Parameter(torch.randn(3, 3))], **common)


def test_state_bytes_lion_half_of_adamw():
    sizes = {}
    for name in ["AdamW", "Lion"]:
        p = torch.nn.Parameter(torch.randn(10, 10))
        opt = build_optimizer(name, [p], lr=1e-3, betas=(0.9, 0.99))
        p.grad = torch.randn_like(p)
        opt.step()
        sizes[name] = optimizer_state_bytes(opt)
    assert sizes["AdamW"] == 2 * sizes["Lion"] == 800


def test_unified_metrics():
    steps = [10 * (k + 1) for k in range(60)]
    val = [0.0] * 20 + [0.96] * 10 + [0.5] * 10 + [1.0] * 20
    m = unified_metrics(steps, [1.0] * 60, val, every=10)
    assert m["val95_first"] == 210
    assert m["val95_consec3"] is None or m["val95_consec3"] >= 210
    assert m["val99_sustained"] == 410
    assert m["train99_first"] == 10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_egd_adam_preconditions_only_listed_matrices():
    from optim.egd import EGD, EGDAdam
    torch.manual_seed(0)
    a, b = torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(8, 4))
    ga, gb = torch.randn(8, 4), torch.randn(8, 4)
    a.grad, b.grad = ga.clone(), gb.clone()
    opt = EGDAdam([a, b], lr=1e-3, weight_decay=0.005, egd_params=[a])
    ref_a, ref_b = torch.nn.Parameter(a.detach().clone()), torch.nn.Parameter(b.detach().clone())
    ref_a.grad, ref_b.grad = EGD.egalitarian(ga, 1e-9), gb.clone()
    ref = torch.optim.Adam([ref_a, ref_b], lr=1e-3, betas=(0.9, 0.98), weight_decay=0.005)
    opt.step(); ref.step()
    assert torch.allclose(a, ref_a, atol=1e-7) and torch.allclose(b, ref_b, atol=1e-7)


def test_egd_adam_arm_uses_fixed_coupled_weight_decay():
    import sweep
    _, cmd = sweep.uni_run("x", "egd_adam", 2e-3, 1)
    assert cmd[cmd.index("--optimizer") + 1] == "EGDAdam"
    assert float(cmd[cmd.index("--weight_decay") + 1]) == 0.005
