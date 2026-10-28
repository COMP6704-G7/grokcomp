"""Independent checks of saved checkpoints, split separation and target access."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.model import Decoder, multiplication_mod_p_data

torch.set_num_threads(4)
data = multiplication_mod_p_data(97, 97, 98)
results = []
for path in sorted((ROOT / "results/lion_study/formal").glob("*/checkpoint.pt")):
    metadata = json.loads((path.parent / "metadata.json").read_text())
    summary = json.loads((path.parent / "summary.json").read_text())
    train_ids, valid_ids = metadata["train_indices"], metadata["valid_indices"]
    assert not set(train_ids).intersection(valid_ids)
    assert set(train_ids).union(valid_ids) == set(range(9312))
    model = Decoder(num_tokens=99).eval()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    valid = data[:, valid_ids]
    with torch.no_grad():
        # Independent prediction code; never calls train.evaluate.
        predictions = torch.cat([model(batch[:4])[-1].argmax(-1)
                                 for batch in valid.split(512, dim=1)])
        accuracy = (predictions == valid[4]).double().mean().item()
        assert abs(accuracy - summary["final_val_acc"]) < 1e-12
        generator = torch.Generator().manual_seed(8204)
        shuffled = valid[4, torch.randperm(valid.shape[1], generator=generator)]
        shuffled_accuracy = (predictions == shuffled).double().mean().item()
        # A future answer token cannot influence the preceding '=' prediction.
        full = valid[:, :64].clone()
        changed = full.clone()
        changed[-1] = (changed[-1] + 1) % 97
        before = model(full)[3]
        after = model(changed)[3]
        causal_max_error = (before - after).abs().max().item()
        assert causal_max_error < 1e-5
    results.append(dict(run=path.parent.name, split_disjoint=True,
        independent_val_acc=accuracy, saved_val_acc=summary["final_val_acc"],
        shuffled_label_acc=shuffled_accuracy, causal_max_error=causal_max_error))
out = ROOT / "results/diagnostics"
out.mkdir(exist_ok=True)
(out / "checkpoint_audit.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
