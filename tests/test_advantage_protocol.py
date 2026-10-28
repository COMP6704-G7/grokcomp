"""Selection-policy regressions, independent of the optimizer's implementation."""
import pytest

from scripts.confirm_advantage import selection_key, tail_accuracy


def summary(hit, accuracy, lr=.001):
    return {"val95": {"step": hit} if hit is not None else None,
            "final_val_acc": accuracy, "lr": lr}


def rows(accuracy):
    return [{"step": step, "val_acc": accuracy} for step in range(0,10001,100)]


def test_primary_threshold_has_priority_over_high_final_point():
    # A single lucky 99% point cannot outrank a confirmed 95% crossing.
    confirmed = selection_key(summary(9000,.96),rows(.96))
    unconfirmed = selection_key(summary(None,.999),rows(.999))
    assert confirmed < unconfirmed


def test_earlier_primary_crossing_is_not_replaced_by_secondary_accuracy():
    assert selection_key(summary(3000,.96),rows(.96)) < selection_key(summary(4000,1.),rows(1.))


def test_tail_window_excludes_step8000_and_selection_uses_whole_window():
    assert tail_accuracy([{"step":8000,"val_acc":1.},
        {"step":8100,"val_acc":.5},{"step":10000,"val_acc":.7}]) == pytest.approx(.6)
    stable = rows(.98)
    lucky = rows(.6)
    lucky[-1]["val_acc"] = 1.
    assert selection_key(summary(None,.98),stable) < selection_key(summary(None,1.),lucky)
