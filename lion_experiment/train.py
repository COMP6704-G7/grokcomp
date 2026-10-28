"""Train identical group models with fixed-checkpoint full-set evaluation."""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import summarize
from .model import Decoder, multiplication_mod_p_data
from .optim import Lion

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_tensors(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(config, seed, device):
    torch.set_num_threads(config["cpu_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    torch.backends.cudnn.benchmark = False
    # Preserve original group RNG ordering: initialize model, then split data.
    model = Decoder(config["dim"], config["num_layers"], config["num_heads"],
                    config["p"] + 2, 5)
    initial_hash = digest_tensors(model.state_dict().values())
    data = multiplication_mod_p_data(config["p"], config["p"], config["p"] + 1)
    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
    split_hash = digest_tensors([train_idx, valid_idx])
    return model.to(device), data[:, train_idx].to(device), data[:, valid_idx].to(device), {
        "initial_weights_sha256": initial_hash, "split_sha256": split_hash,
        "train_indices": train_idx.tolist(), "valid_indices": valid_idx.tolist()}


@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval()
    loss_sum, correct = 0.0, 0
    for batch in data.split(batch_size, dim=1):
        logits = model(batch[:-1])[-1]
        loss_sum += F.cross_entropy(logits, batch[-1], reduction="sum").item()
        correct += (logits.argmax(-1) == batch[-1]).sum().item()
    return loss_sum / data.shape[1], correct / data.shape[1]


def run(config, optimizer_name, lr, seed, output, device_name="cuda", resume=False,
        phase="formal"):
    if config["p"] != 97 or config["dtype"] != "float32":
        raise ValueError("This study fixes mod-97 and FP32")
    if any(config[k] <= 0 for k in ["max_steps", "batch_size", "eval_every", "consecutive"]):
        raise ValueError("Steps, batch size, evaluation interval and streak must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for a smoke test")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"protocol": config, "optimizer": optimizer_name, "lr": lr,
                     "seed": seed, "phase": phase, "device": device_name}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError(f"Configuration differs from existing run: {output}")
        if not resume:
            raise FileExistsError(f"Run exists; use --resume: {output}")
        if (output / "summary.json").exists():
            return json.loads((output / "summary.json").read_text())
    atomic_json(config_path, configuration)

    model, train, valid, identity = setup(config, seed, device)
    if optimizer_name == "lion":
        optimizer = Lion(model.parameters(), lr=lr, betas=config["lion_betas"],
                         weight_decay=config["lion_weight_decay"])
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=config["adamw_betas"],
                                     weight_decay=config["adamw_weight_decay"], foreach=False)
    else:
        raise ValueError(optimizer_name)
    warmup = config["warmup_steps"]
    # Match group's LambdaLR convention: the very first update uses lr=0.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, update / warmup) if warmup else 1.0)
    generator = torch.Generator().manual_seed(seed + 1000000)
    order = torch.empty(0, dtype=torch.long)
    cursor, step, training_seconds, elapsed, segments, peak = 0, 0, 0.0, 0.0, 1, 0
    rows = []
    checkpoint = output / "checkpoint.pt"
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["sampler_rng"])
        order, cursor, step = state["order"], state["cursor"], state["step"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        rows = state["rows"]
        training_seconds, elapsed = state["training_seconds"], state["elapsed"]
        segments, peak = state["segments"] + 1, state["peak_memory_bytes"]

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # Hash actual code, since working-tree additions may not be committed.
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "lion_experiment").glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    metadata = {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cuda": torch.version.cuda, "git_revision": revision,
                "source_sha256": source_hash.hexdigest(),
                "parameter_count": sum(p.numel() for p in model.parameters()), **identity}
    atomic_json(output / "metadata.json", metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()

    def wall():
        return elapsed + time.perf_counter() - started

    def log_evaluation(log):
        train_loss, train_acc = evaluate(model, train, config["eval_batch_size"])
        val_loss, val_acc = evaluate(model, valid, config["eval_batch_size"])
        synchronize(device)
        row = {"step": step, "wall_seconds": wall(), "training_seconds": training_seconds,
               "train_loss": train_loss, "val_loss": val_loss,
               "train_acc": train_acc, "val_acc": val_acc,
               "lr": optimizer.param_groups[0]["lr"],
               "eta_lambda": optimizer.param_groups[0]["lr"] * optimizer.param_groups[0]["weight_decay"]}
        if not all(np.isfinite(row[k]) for k in ["train_loss", "val_loss"]):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        rows.append(row)
        log.write(json.dumps(row, allow_nan=False) + "\n")
        log.flush()
        print(f"{output.name} step={step} train={train_acc:.4f} val={val_acc:.4f} "
              f"wall={row['wall_seconds']:.1f}s", flush=True)

    def save_checkpoint():
        synchronize(device)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "sampler_rng": generator.get_state(), "order": order, "cursor": cursor,
                    "step": step, "rows": rows, "training_seconds": training_seconds,
                    "elapsed": wall(), "segments": segments,
                    "peak_memory_bytes": max(peak, torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else None}, temporary)
        temporary.replace(checkpoint)

    # Rebuild log from the checkpoint to discard work after its last safe save.
    with (output / "metrics.jsonl").open("w") as log:
        for row in rows:
            log.write(json.dumps(row) + "\n")
        if not rows:
            log_evaluation(log)
            save_checkpoint()
        try:
            while step < config["max_steps"]:
                synchronize(device)
                tick = time.perf_counter()
                if cursor >= len(order):
                    order = torch.randperm(train.shape[1], generator=generator)
                    cursor = 0
                indices = order[cursor:cursor + config["batch_size"]].to(device)
                cursor += len(indices)
                batch = train[:, indices]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch[:-1])[-1]
                loss = F.cross_entropy(logits, batch[-1])
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                synchronize(device)
                training_seconds += time.perf_counter() - tick
                if step % config["eval_every"] == 0 or step == config["max_steps"]:
                    log_evaluation(log)
                    if config["stop_at_99"] and summarize(rows, config["consecutive"])["val99"]:
                        break
                if step % config["checkpoint_every"] == 0:
                    save_checkpoint()
        except BaseException:
            # An interrupted optimizer may have partially updated parameters.
            # Resume from the last completed checkpoint, discarding later work.
            raise
    save_checkpoint()
    hits = summarize(rows, config["consecutive"])
    summary = {"optimizer": optimizer_name, "lr": lr, "weight_decay": optimizer.param_groups[0]["weight_decay"],
               "eta_lambda": lr * optimizer.param_groups[0]["weight_decay"], "seed": seed, "phase": phase,
               "status": "success" if hits["val99"] else "budget_exhausted", "steps": step,
               "max_steps": config["max_steps"], "wall_seconds": wall(),
               "training_seconds": training_seconds,
               "mean_step_ms": 1000 * training_seconds / step,
               "peak_memory_bytes": max(peak, torch.cuda.max_memory_allocated(device))
               if device.type == "cuda" else None,
               "optimizer_state_bytes": sum(v.numel() * v.element_size()
                                            for state in optimizer.state.values()
                                            for v in state.values() if torch.is_tensor(v)),
               "segments": segments, "final_train_acc": rows[-1]["train_acc"],
               "final_val_acc": rows[-1]["val_acc"],
               "initial_weights_sha256": identity["initial_weights_sha256"],
               "split_sha256": identity["split_sha256"], **hits}
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lion_study.json")
    parser.add_argument("--optimizer", choices=["lion", "adamw"], required=True)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--phase", choices=["smoke", "pilot", "tuning", "formal"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ["max_steps", "eval_every"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    lr = args.lr if args.lr is not None else (config["lion_lrs"][1] if args.optimizer == "lion" else config["adamw_lr"])
    run(config, args.optimizer, lr, args.seed, args.output, args.device, args.resume, args.phase)


if __name__ == "__main__":
    main()
