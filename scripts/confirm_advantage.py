"""Prospective low-decay confirmation with equal-count LR tuning and fresh seeds."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.train import atomic_json, run


def tail_accuracy(rows):
    cutoff = rows[-1]["step"] - 2000
    tail = [r["val_acc"] for r in rows if r["step"] > cutoff]
    return sum(tail) / len(tail)


def selection_key(summary, rows):
    hit = summary["val95"]
    return (hit is None, hit["step"] if hit else float("inf"),
            -tail_accuracy(rows), -summary["final_val_acc"], summary["lr"])


def run_label(opt, seed, lr):
    return f"{opt}_seed{seed}_lr{lr:.0e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/advantage_validation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/advantage_validation")
    parser.add_argument("--stage", choices=["tune", "confirm", "all"], default="all")
    args = parser.parse_args()
    design = json.loads(args.config.read_text())
    if design["tuning_seed"] in design["confirmation_seeds"]:
        raise ValueError("Confirmation seeds must be distinct from the tuning seed")
    if len(set(design["confirmation_seeds"])) != 3:
        raise ValueError("Exactly three distinct confirmation seeds are required")
    if any(len(design["candidates"][opt]) != 3 for opt in ["lion", "adamw"]):
        raise ValueError("Equal tuning budget requires three candidates per optimizer")
    if design["protocol"]["stop_at_99"] or design["early_stopping"]:
        raise ValueError("This design requires a fixed observation horizon")
    args.output.mkdir(parents=True, exist_ok=True)
    locked = args.output / "design_locked.json"
    if locked.exists():
        lock = json.loads(locked.read_text())
        if lock["design"] != design:
            raise ValueError("The design has changed since locking; use a separate study directory")
    else:
        atomic_json(locked, {"design": design,
            "locked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "design_sha256": hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()})
    frozen_path = args.output / "frozen_selection.json"
    if args.stage in ["tune", "all"] and not frozen_path.exists():
        chosen, evidence = {}, {}
        # Match candidate counts, horizons and selection rule for both methods.
        for opt in ["adamw", "lion"]:
            candidates = []
            for lr in design["candidates"][opt]:
                cfg = {**design["protocol"], opt + "_weight_decay": design["target_eta_lambda"] / lr}
                path = args.output / "tuning" / run_label(opt, design["tuning_seed"], lr)
                summary = run(cfg, opt, lr, design["tuning_seed"], path,
                              "cuda", resume=True, phase="advantage_tuning")
                rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
                candidates.append((summary, rows))
            winner, _ = min(candidates, key=lambda pair: selection_key(*pair))
            chosen[opt] = {"lr": winner["lr"], "weight_decay": winner["weight_decay"]}
            evidence[opt] = [{"summary": s, "tail_mean_val_acc": tail_accuracy(r)} for s,r in candidates]
        atomic_json(frozen_path, {"selected": chosen, "tuning_evidence": evidence,
            "selection_rule": design["selection_rule"],
            "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "design_sha256": json.loads(locked.read_text())["design_sha256"]})
    if args.stage in ["confirm", "all"]:
        if not frozen_path.exists():
            raise RuntimeError("Complete tuning before confirming")
        frozen = json.loads(frozen_path.read_text())
        if frozen["design_sha256"] != json.loads(locked.read_text())["design_sha256"]:
            raise ValueError("Frozen configuration does not match the locked design")
        cfg = {**design["protocol"],
            "adamw_lr": frozen["selected"]["adamw"]["lr"],
            "adamw_weight_decay": frozen["selected"]["adamw"]["weight_decay"],
            "lion_weight_decay": frozen["selected"]["lion"]["weight_decay"]}
        for index, seed in enumerate(design["confirmation_seeds"]):
            pair = []
            for opt in (["lion", "adamw"] if index % 2 else ["adamw", "lion"]):
                lr = frozen["selected"][opt]["lr"]
                pair.append(run(cfg, opt, lr, seed, args.output / "confirmation" / run_label(opt, seed, lr),
                    "cuda", resume=True, phase="advantage_confirmation"))
            for key in ["initial_weights_sha256", "split_sha256"]:
                if pair[0][key] != pair[1][key]:
                    raise RuntimeError(f"Paired initialization mismatch: {key}")
        atomic_json(args.output / "complete.json", {"seeds": design["confirmation_seeds"],
            "selected": frozen["selected"], "horizon": cfg["max_steps"]})
        print('Fresh-seed confirmation complete. Run scripts/analyze_advantage.py.', flush=True)


if __name__ == "__main__":
    main()
