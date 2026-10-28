import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from lion_experiment.optim import Lion

spec = importlib.util.spec_from_file_location(
    "official", Path(__file__).resolve().parents[1] / "vendor/lion_official.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def test_hand_calculated_two_steps():
    """Independent arithmetic check; no official implementation is invoked.

    The second gradient opposes the existing momentum. Updating the momentum
    before constructing the direction would reverse the expected signs.
    """
    parameter = torch.nn.Parameter(torch.tensor([1., -2., 3., 4.], dtype=torch.float64))
    optimizer = Lion([parameter], lr=.1, betas=(.5, .75), weight_decay=.2)
    parameter.grad = torch.tensor([2., -4., 0., 0.], dtype=torch.float64)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([.88, -1.86, 2.94, 3.92],
        dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"],
        torch.tensor([.5, -1., 0., 0.], dtype=torch.float64), rtol=0, atol=0)
    parameter.grad = torch.tensor([-.375, .75, 0., 0.], dtype=torch.float64)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([.7624, -1.7228, 2.8812, 3.8416],
        dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"],
        torch.tensor([.28125, -.5625, 0., 0.], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_official_gradient_sequence(dtype):
    generator = torch.Generator().manual_seed(3407)
    params = [torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype))
              for shape in [(5, 7), (7,), (3,)]]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    def groups(ps):
        return [dict(params=ps[:2], lr=2e-4, weight_decay=5),
                dict(params=ps[2:], lr=3e-4, weight_decay=0)]
    actual, expected = Lion(groups(params)), official.Lion(groups(refs))
    for step in range(40):
        for index, (p, q) in enumerate(zip(params, refs)):
            grad = torch.randn(p.shape, generator=generator, dtype=dtype)
            if step % 7 == 0:
                grad.zero_()
            p.grad = grad.clone() if index != 2 or step % 3 else None
            q.grad = None if p.grad is None else p.grad.clone()
        actual.step()
        expected.step()
        for p, q in zip(params, refs):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            if p in actual.state:
                torch.testing.assert_close(actual.state[p]["exp_avg"],
                                           expected.state[q]["exp_avg"], rtol=0, atol=0)


def test_state_restore_and_closure():
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = Lion([p])
    def closure():
        optimizer.zero_grad()
        loss = p.square().sum()
        loss.backward()
        return loss
    assert optimizer.step(closure).item() == 3
    q = torch.nn.Parameter(p.detach().clone())
    restored = Lion([q])
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    p.grad = q.grad = torch.tensor([1., -1., 0.])
    optimizer.step()
    restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [dict(lr=-1), dict(lr=float('nan')),
    dict(betas=(1, .99)), dict(weight_decay=-1)])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        Lion([torch.nn.Parameter(torch.ones(1))], **kwargs)
