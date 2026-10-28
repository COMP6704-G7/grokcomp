"""Continue the six original formal checkpoints to a fixed 10k-step horizon.

Hyperparameters and RNG/sampler state are preserved. This changes observation
length only; it does not tune against the three formal seeds. Timing is segmented.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.train import atomic_json, run

target = ROOT / "results/diagnostics/formal_stability"
target.mkdir(parents=True, exist_ok=True)
origins = []
for source in sorted((ROOT / "results/lion_study/formal").iterdir()):
    if not (source / "summary.json").exists():
        continue
    original_config = json.loads((source / "config.json").read_text())
    result = json.loads((source / "summary.json").read_text())
    config = {**original_config["protocol"], "max_steps": 10000, "stop_at_99": False}
    destination = target / source.name
    destination.mkdir(exist_ok=True)
    if not (destination / "checkpoint.pt").exists():
        shutil.copy2(source / "checkpoint.pt", destination / "checkpoint.pt")
    atomic_json(destination / "continuation_origin.json", {
        "original_run": str(source.relative_to(ROOT)), "original_stop_step": result["steps"],
        "new_horizon": 10000, "only_protocol_changes": {"max_steps": 10000, "stop_at_99": False},
        "note": "Preserves checkpoint weights, optimizer, scheduler, RNG and sampler; no LR/decay change. Timing has two segments."})
    origins.append(source.name)
    run(config, result["optimizer"], result["lr"], result["seed"], destination,
        "cuda", resume=True, phase="stability")
atomic_json(target / "complete.json", {"runs": origins, "new_horizon": 10000,
    "claim": "Checks only the original fixed-baseline comparison, not the supplementary tuned baseline"})
print('All six original formal checkpoints continued without retuning.', flush=True)
