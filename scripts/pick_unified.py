"""Freeze one lr per method from the uni_tune runs (tuning seed only) into results/uni_selected.json.

Rule, the same for every method and fixed before any comparison run:
    reached val95 (3 evaluations in a row) > earlier val95_consec3 > higher mean val acc over the
    last 10% of the run > lower lr.
    python scripts/pick_unified.py [--out_dir DIR]
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep import UNI_ARMS, lr_tag  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default=".")
a = ap.parse_args()
os.chdir(a.out_dir)

by_arm = {}
for f in glob.glob("results/summary_uni_tune_*.json"):
    s = json.load(open(f))
    arm = re.match(r"uni_tune_(\w+?)_lr", s["label"]).group(1)
    if arm not in UNI_ARMS:  # results of methods no longer in the study
        continue
    u = s["unified"]
    by_arm.setdefault(arm, []).append({
        "label": s["label"], "lr": s["args"]["lr"], "weight_decay": s["args"]["weight_decay"],
        "val95_consec3": u["val95_consec3"], "val99_consec3": u["val99_consec3"],
        "tail10_mean_val_acc": u["tail10_mean_val_acc"], "final_val_acc": u["final_val_acc"],
    })


def key(c):
    t = c["val95_consec3"]
    return (t is None, t if t is not None else 0, -(c["tail10_mean_val_acc"] or 0), c["lr"])


def tried(arm, lr):
    """A tuning run was attempted at this lr (summary or stdout exists), even if it crashed."""
    return bool(glob.glob(f"results/stdout_uni_tune_{arm}_lr{lr_tag(lr)}_s*.txt"))


sel = {}
for arm, cands in sorted(by_arm.items()):
    cands.sort(key=key)
    best = cands[0]["lr"]
    # the grid is every lr attempted, including runs that crashed or diverged (no summary)
    lrs = [c["lr"] for c in cands] + [float(re.search(r"_lr([0-9.e-]+)_s", f).group(1))
                                      for f in glob.glob(f"results/stdout_uni_tune_{arm}_lr*_s*.txt")]
    lo, hi = min(lrs), max(lrs)
    at_edge = len(cands) > 1 and best in (lo, hi)
    # one factor-2 step past the edge, unless that lr was already tried (e.g. it crashed)
    nxt = None
    if at_edge and cands[0]["val95_consec3"] is not None:
        nxt = float(f"{(best * 2 if best == hi else best / 2):.0e}")
        nxt = None if tried(arm, nxt) else nxt
    sel[arm] = {"lr": best, "candidates": cands, "at_edge": at_edge, "extend_lr": [nxt] if nxt else []}
    print(f"{arm:6s} lr={best:.0e}  val95_consec3={cands[0]['val95_consec3']}  "
          f"edge={at_edge}  next={nxt}  ({len(cands)} candidates)")
sel["_frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
json.dump(sel, open("results/uni_selected.json", "w"), indent=2)
