import ast
import json
from pathlib import Path

import torch

from lion_experiment.metrics import crossing, summarize
from lion_experiment.model import multiplication_mod_p_data
from lion_experiment.train import ROOT, run, setup


def test_exhaustive_mod97_data():
    data = multiplication_mod_p_data(97, 97, 98)
    assert data.shape == (5, 9312)
    assert torch.equal(data[-1], data[0] * data[2] % 97)
    assert (data[2] != 0).all()
    assert torch.unique(data[[0, 2]], dim=1).shape[1] == 9312


def test_model_and_split_match_group_original():
    config = json.loads((ROOT / "configs/lion_study.json").read_text())
    source = (ROOT / "results/diagnostics/source_snapshot/group_main_93b2248.py").read_text()
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                   and node.name in {"Block", "Decoder", "multiplication_mod_p_data"}]
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "original_main.py", "exec"), namespace)
    torch.manual_seed(2)
    original = namespace["Decoder"](num_tokens=99)
    expected_split = torch.randperm(9312).split(4656)
    current, train, valid, identity = setup(config, 2, torch.device("cpu"))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(current.state_dict()[key], value, rtol=0, atol=0)
    assert identity["train_indices"] == expected_split[0].tolist()
    assert identity["valid_indices"] == expected_split[1].tolist()
    assert train.shape[1] == valid.shape[1] == 4656


def test_crossings_streak_reset_start_and_censoring():
    rows = [dict(step=i * 100, wall_seconds=i * 2, train_acc=1., val_acc=a)
            for i, a in enumerate([1., .99, .98, .99, .995, 1.])]
    assert crossing(rows, "val_acc", .99) == dict(step=300, wall_seconds=6, confirmed_step=500)
    assert crossing(rows[:-1], "val_acc", .99) is None
    assert summarize(rows)["generalization_delay"] == 200
    negative = [dict(step=i * 100, wall_seconds=i, train_acc=t, val_acc=v)
                for i, (t, v) in enumerate([(0, 0), (0, 1), (1, 1), (1, 1), (1, 1)], 1)]
    assert summarize(negative)["generalization_delay"] == -100


def test_resume_replays_identically(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/lion_study.json").read_text())
    config.update(dim=16, num_layers=1, num_heads=2, max_steps=5,
                  eval_every=2, eval_batch_size=2048, checkpoint_every=2,
                  cpu_threads=1, stop_at_99=False)
    baseline = tmp_path / "continuous"
    run(config, "lion", .0002, 0, baseline, "cpu", phase="smoke")
    original_step = __import__('lion_experiment.train', fromlist=['Lion']).Lion.step
    calls = 0
    def interrupt(self, closure=None):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt()
        return original_step(self, closure)
    monkeypatch.setattr("lion_experiment.train.Lion.step", interrupt)
    interrupted = tmp_path / "interrupted"
    import pytest
    with pytest.raises(KeyboardInterrupt):
        run(config, "lion", .0002, 0, interrupted, "cpu", phase="smoke")
    monkeypatch.setattr("lion_experiment.train.Lion.step", original_step)
    run(config, "lion", .0002, 0, interrupted, "cpu", resume=True, phase="smoke")
    first = torch.load(baseline / "checkpoint.pt", weights_only=False)
    second = torch.load(interrupted / "checkpoint.pt", weights_only=False)
    for key, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][key], rtol=0, atol=0)
    assert second["segments"] == 2
    assert [r["val_acc"] for r in first["rows"]] == [r["val_acc"] for r in second["rows"]]
