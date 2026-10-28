"""Verify shipped source, raw evidence, selection and paired protocol without training."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.metrics import summarize
from lion_experiment.train import digest_tensors, setup
from scripts.confirm_advantage import selection_key


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = read(ROOT / "docs/SOURCE_MANIFEST.json")
    source = hashlib.sha256()
    for path in sorted((ROOT / "lion_experiment").glob("*.py")):
        relative = str(path.relative_to(ROOT))
        require(sha(path) == manifest["files"][relative], f"Source changed: {relative}")
        source.update(path.name.encode() + path.read_bytes())
    require(source.hexdigest() == manifest["experiment_source_sha256"], "Combined source hash differs")
    reference = read(ROOT / "vendor/lion_official.provenance.json")
    require(sha(ROOT / "vendor/lion_official.py") == reference["sha256"], "Official reference changed")
    require(sha(ROOT / "results/diagnostics/source_snapshot/group_main_93b2248.py") ==
            read(ROOT / "docs/RELEASE_PROVENANCE.json")["original_group_main_sha256"],
            "Original group source snapshot changed")
    study = ROOT / "results/advantage_validation"
    audit = read(study / "AUDIT_MANIFEST.json")
    require(audit["experiment_source_sha256"] == source.hexdigest(), "Run/source hash mismatch")
    for relative, expected in audit["per_run_logs"].items():
        require(sha(ROOT / relative) == expected, f"Recorded log changed: {relative}")
    for name in ["design_locked", "frozen_selection"]:
        require(sha(study / (name + ".json")) == audit[name + "_sha256"], f"Changed {name}")
    for name, path in [("runner", "scripts/confirm_advantage.py"), ("analysis", "scripts/analyze_advantage.py")]:
        require(sha(ROOT / path) == audit[name + "_sha256"], f"Recorded {name} source changed")
    locked, frozen = read(study / "design_locked.json"), read(study / "frozen_selection.json")
    design = locked["design"]
    require(design == read(ROOT / "configs/advantage_validation.json"), "Config differs from lock")
    design_hash = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    require(design_hash == locked["design_sha256"] == frozen["design_sha256"], "Design digest mismatch")
    require(design["tuning_seed"] not in design["confirmation_seeds"], "Tuning/confirmation overlap")
    counts = {"tuning": 0, "confirmation": 0}
    all_runs = {}
    for phase in counts:
        by_opt = {"lion": [], "adamw": []}
        for path in sorted((study / phase).glob("*/summary.json")):
            summary = read(path)
            cfg, metadata = read(path.parent / "config.json"), read(path.parent / "metadata.json")
            rows = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
            opt = summary["optimizer"]
            require(cfg["optimizer"] == opt and cfg["seed"] == summary["seed"], f"Identity: {path}")
            require(cfg["phase"] == summary["phase"] == "advantage_" + phase, f"Phase: {path}")
            require(summary["steps"] == design["protocol"]["max_steps"] == 10000, f"Incomplete: {path}")
            require(summary["segments"] == 1 and not cfg["protocol"]["stop_at_99"], f"Stopping/timing: {path}")
            require(metadata["source_sha256"] == source.hexdigest(), f"Source metadata: {path}")
            require([r["step"] for r in rows] == list(range(0, 10001, 100)), f"Missing evaluations: {path}")
            recomputed = summarize(rows, design["protocol"]["consecutive"])
            for key in ["train99", "val95", "val99", "generalization_delay"]:
                require(summary[key] == recomputed[key], f"Threshold metric {key}: {path}")
            require(summary["final_val_acc"] == rows[-1]["val_acc"], f"Final accuracy: {path}")
            train, valid = metadata["train_indices"], metadata["valid_indices"]
            require(len(train) == len(valid) == 4656 and not set(train).intersection(valid)
                    and set(train).union(valid) == set(range(9312)), f"Invalid split: {path}")
            require(digest_tensors([torch.tensor(train), torch.tensor(valid)]) == metadata["split_sha256"],
                    f"Split digest: {path}")
            require(abs(summary["lr"] * summary["weight_decay"] - design["target_eta_lambda"]) < 1e-14,
                    f"Shrinkage mismatch: {path}")
            require(cfg["lr"] == summary["lr"] and
                    cfg["protocol"][opt + "_weight_decay"] == summary["weight_decay"], f"Config: {path}")
            expected = dict(design["protocol"])
            if phase == "tuning":
                expected[opt + "_weight_decay"] = design["target_eta_lambda"] / summary["lr"]
                require(summary["seed"] == design["tuning_seed"], f"Tuning seed: {path}")
            else:
                expected.update(adamw_lr=frozen["selected"]["adamw"]["lr"],
                                adamw_weight_decay=frozen["selected"]["adamw"]["weight_decay"],
                                lion_weight_decay=frozen["selected"]["lion"]["weight_decay"])
                require(summary["seed"] in design["confirmation_seeds"] and
                        summary["lr"] == frozen["selected"][opt]["lr"], f"Unfrozen confirmation: {path}")
            require(cfg["protocol"] == expected, f"Unexpected protocol field: {path}")
            by_opt[opt].append((summary, rows, metadata, cfg))
            counts[phase] += 1
        require(counts[phase] == 6 and all(len(v) == 3 for v in by_opt.values()), f"Run count: {phase}")
        all_runs[phase] = by_opt
    for opt, group in all_runs["tuning"].items():
        require(sorted(r[0]["lr"] for r in group) == sorted(design["candidates"][opt]), f"Candidates: {opt}")
        winner = min(group, key=lambda r: selection_key(r[0], r[1]))[0]
        require(frozen["selected"][opt] == {"lr": winner["lr"], "weight_decay": winner["weight_decay"]},
                f"Selection not reproducible: {opt}")
    for seed in design["confirmation_seeds"]:
        pair = [next(r for r in all_runs["confirmation"][opt] if r[0]["seed"] == seed)
                for opt in ["lion", "adamw"]]
        for key in ["source_sha256", "initial_weights_sha256", "split_sha256", "torch", "cuda", "gpu"]:
            require(pair[0][2][key] == pair[1][2][key], f"Pair mismatch, seed {seed}: {key}")
        _, _, _, identity = setup(design["protocol"], seed, torch.device("cpu"))
        for key in ["initial_weights_sha256", "split_sha256"]:
            require(pair[0][2][key] == pair[0][0][key] == pair[1][0][key] == identity[key],
                    f"Reconstructed initialization/split mismatch, seed {seed}: {key}")
    require(audit["completed_runs"] == 12 and audit["total_updates"] == 120000, "Audit totals")
    print(json.dumps({"verified": True, "runs": counts, "total_updates": 120000,
                      "training_source_sha256": source.hexdigest(),
                      "checks": ["source/reference/log integrity", "frozen candidate selection",
                                 "thresholds", "complete split", "paired initialization",
                                 "matched shrinkage", "fixed horizon"]}, indent=2))


if __name__ == "__main__":
    verify()
