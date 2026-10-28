"""Consecutive-evaluation thresholds; censored runs remain missing."""
import statistics


def crossing(rows, column, threshold, consecutive=3):
    streak = []
    for row in rows:
        # Step zero is a diagnostic, not an evaluation after training.
        if row["step"] == 0:
            continue
        streak = streak + [row] if row[column] >= threshold else []
        if len(streak) == consecutive:
            return {"step": streak[0]["step"], "wall_seconds": streak[0]["wall_seconds"],
                    "confirmed_step": row["step"]}
    return None


def summarize(rows, consecutive=3):
    hits = {name: crossing(rows, col, threshold, consecutive) for name, col, threshold in [
        ("train99", "train_acc", .99), ("val95", "val_acc", .95), ("val99", "val_acc", .99)]}
    hits["generalization_delay"] = (
        hits["val99"]["step"] - hits["train99"]["step"]
        if hits["val99"] and hits["train99"] else None)
    return hits


def mean_std(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}
