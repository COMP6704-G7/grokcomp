"""Checks for the shared grok-timing definitions and the gradient–weight cosine."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import grad_weight_cos, grok_summary  # noqa: E402


def test_grok_summary_first_sustained_and_delay():
    steps = [10, 20, 30, 40, 50, 60]
    train = [0.5, 0.995, 1.0, 1.0, 1.0, 1.0]
    val = [0.0, 0.1, 0.995, 0.5, 0.995, 1.0]  # crosses at 30, falls back, holds from 50
    s = grok_summary(steps, train, val)
    assert (s["fit_step"], s["grok_step"], s["sustained_step"], s["delay"]) == (20, 30, 50, 10)


def test_grok_summary_never_grokked():
    s = grok_summary([1, 2], [1.0, 1.0], [0.2, 0.3])
    assert s["grok_step"] is None and s["sustained_step"] is None and s["delay"] is None


def test_grad_weight_cos_sign():
    m = torch.nn.Linear(4, 3, bias=False)
    m.weight.grad = m.weight.detach().clone()        # gradient = w  -> cos = +1
    assert abs(grad_weight_cos(m) - 1) < 1e-5
    m.weight.grad = -m.weight.detach().clone()       # gradient = -w -> cos = -1
    assert abs(grad_weight_cos(m) + 1) < 1e-5
