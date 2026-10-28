"""Controlled decay checks; same model, split, sampler, LR and fixed horizon.

This is a diagnostic, not a replacement formal result or an optimizer contest.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.train import atomic_json, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--optimizer", choices=["lion", "adamw", "both"], default="both")
parser.add_argument("--decays", type=float, nargs="+", default=[1.0, 0.1, 0.01, 0.0])
args = parser.parse_args()
base = json.loads((ROOT / "configs/lion_study.json").read_text())
base.update(max_steps=args.steps, stop_at_99=False)
target = ROOT / "results/diagnostics/decay_sweep"
target.mkdir(parents=True, exist_ok=True)
atomic_json(target / "design.json", {"seed": 0, "steps": args.steps,
    "adamw_lr": .001, "lion_lr": .0002, "reference_adamw_decays": args.decays,
    "lion_decay_multiplier": 5, "early_stopping": False,
    "purpose": "Vary decay only within each optimizer; matched peak eta*lambda between optimizers"})
for decay in args.decays:
    config = {**base, "adamw_weight_decay": decay, "lion_weight_decay": 5 * decay}
    for opt in (["adamw", "lion"] if args.optimizer == "both" else [args.optimizer]):
        lr = .001 if opt == "adamw" else .0002
        result = run(config, opt, lr, 0, target / f"{opt}_refwd{decay:g}",
                     "cuda", resume=True, phase="diagnostic")
        print("DIAGNOSTIC SUMMARY", json.dumps(result), flush=True)
