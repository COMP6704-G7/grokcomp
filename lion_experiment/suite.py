"""Sequential tuning and paired comparison on a single GPU."""
import argparse
import hashlib
import json
from pathlib import Path

from .train import ROOT, atomic_json, run


def run_name(optimizer, seed, lr):
    return f"{optimizer}_seed{seed}_lr{lr:.0e}"


def selection_key(result):
    # Prespecified: success, earliest val99, then val95, final accuracy, lower lr.
    return (result["val99"] is None,
            result["val99"]["step"] if result["val99"] else float("inf"),
            result["val95"]["step"] if result["val95"] else float("inf"),
            -result["final_val_acc"], result["lr"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lion_study.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lion_study")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=["pilot", "tune", "formal", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if config["tuning_seed"] in config["formal_seeds"]:
        raise ValueError("Formal seeds must be new and distinct from the tuning seed")
    if len(set(config["formal_seeds"])) != 3:
        raise ValueError("The plan requires exactly three distinct formal seeds")
    if args.stage == "pilot":
        pilot = {**config, "max_steps": 1000}
        for opt, lr in [("lion", config["lion_lrs"][1]), ("adamw", config["adamw_lr"])]:
            run(pilot, opt, lr, config["tuning_seed"],
                args.output / "pilot" / run_name(opt, config["tuning_seed"], lr),
                args.device, args.resume, "pilot")
        return
    freeze_path = args.output / "frozen_selection.json"
    if args.stage in ["tune", "all"] and not freeze_path.exists():
        candidates = []
        for lr in config["lion_lrs"]:
            candidates.append(run(config, "lion", lr, config["tuning_seed"],
                                  args.output / "tuning" / run_name("lion", config["tuning_seed"], lr),
                                  args.device, args.resume, "tuning"))
        winner = min(candidates, key=selection_key)
        atomic_json(freeze_path, {"selected_lr": winner["lr"], "protocol": config,
            "selection_rule": "val99 success, first val99 step, first val95 step, final val_acc, lower lr",
            "tuning_results": candidates})
    if args.stage in ["formal", "all"]:
        if not freeze_path.exists():
            raise RuntimeError("Run tuning first to freeze the learning rate")
        frozen = json.loads(freeze_path.read_text())
        if frozen["protocol"] != config:
            raise ValueError("Study protocol changed after learning-rate selection")
        # One process, one GPU, no concurrent runs: timing conditions are matched.
        for seed in config["formal_seeds"]:
            paired = []
            for opt, lr in [("lion", frozen["selected_lr"]), ("adamw", config["adamw_lr"])]:
                paired.append(run(config, opt, lr, seed,
                    args.output / "formal" / run_name(opt, seed, lr), args.device, args.resume, "formal"))
            for key in ["initial_weights_sha256", "split_sha256"]:
                if paired[0][key] != paired[1][key]:
                    raise RuntimeError(f"Paired comparison identity mismatch: {key}")
        from .analyze import analyze
        analyze(args.output)


if __name__ == "__main__":
    main()
