"""Analysis of the unified experiment: tables + figures from the run outputs, nothing re-run.

    python scripts/analyze_unified.py --out_dir OUT

Reads   OUT/results/summary_uni_*.json, res_uni_*.pt, log_uni_*.jsonl, stdout_uni_*.txt, uni_selected.json
Writes  OUT/analysis/unified_analysis.json, uni_curves.png and uni_stability.png

Safe to re-run at any time: runs that are not finished yet (no summary) are reported as "running" or
"pending", never filled in. Light CPU work only (reads ~150 small files).
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sweep import (UNI_ARMS, UNI_DECAY_EL, UNI_DECAY_SEEDS, UNI_ETA_LAMBDA, UNI_FIXED_WD, UNI_SEEDS,  # noqa: E402
                   UNI_TUNE_SEED, lr_tag)

ARMS = ["adam", "adamw", "lion", "orth", "fpd", "egd_adam"]  # adam: reference without decay
MAIN5 = ["adamw", "lion", "orth", "fpd", "egd_adam"]  # the methods compared on the page and in the report
NAME = {"adam": "Adam (no decay)", "adamw": "AdamW", "lion": "Lion", "orth": "⊥AdamW + StableMax",
        "fpd": "FPD (AdamW base)",
        "egd_adam": "EGD (paper setup: MLP weights, Adam)"}
SOURCE = {"adam": "main (Grokfast default)", "adamw": "main (Grokfast, standard setting)", "lion": "Lion",
          "orth": "⊥AdamW", "fpd": "fpd",
          "egd_adam": "egd (grokfast-comparison/main_modmult.py)"}
N_TRAIN = 4656  # 9,312 equations x 50 %
BATCH = 512
BUDGET = 20000
SEL_THR = 0.95
SHORT = {"adam": "Adam, no decay", "adamw": "AdamW", "lion": "Lion", "orth": "⊥AdamW + StableMax", "fpd": "FPD", "egd_adam": "EGD"}
COLORS = {"adam": "#7f7f7f", "adamw": "#1f77b4", "lion": "#2ca02c", "orth": "#d62728", "fpd": "#9467bd", "egd_adam": "#ff7f0e"}


def jload(p):
    with open(p) as f:
        return json.load(f)


def summary(R, label):
    p = os.path.join(R, f"summary_{label}.json")
    return jload(p) if os.path.exists(p) else None


def log_rows(R, label):
    p = os.path.join(R, f"log_{label}.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line being written right now
    return rows


def res(R, label):
    p = os.path.join(R, f"res_{label}.pt")
    if not os.path.exists(p):
        return None
    try:
        return torch.load(p, weights_only=False, map_location="cpu")
    except Exception:  # file being rewritten by a live run
        return None


def active_labels():
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return sorted(set(re.findall(r"--label (uni_\S+)", out)))


def run_status(R, label, active):
    """finished / no_grok / diverged / crashed / running / not_started, with what is known so far."""
    s = summary(R, label)
    rows = log_rows(R, label)
    last = rows[-1] if rows else None
    if s is not None:
        u = s["unified"]
        tl = last["train_loss"] if last else None
        if u["val95_consec3"] is not None:
            st = "grokked"
        elif tl is not None and (not np.isfinite(tl) or tl > 10):
            st = "diverged"
        else:
            st = "no_grok"
        return {"status": st, "summary": s}
    so = os.path.join(R, f"stdout_{label}.txt")
    if not os.path.exists(so):
        return {"status": "not_started"}
    txt = open(so, errors="replace").read()
    if "Traceback" in txt and label not in active:
        why = "non-finite values (NaN)" if ("non-finite" in txt or "nan" in txt.lower().split("traceback")[-1]) else "error"
        return {"status": "crashed", "reason": why, "last_step": last["step"] if last else None,
                "last_train_loss": last["train_loss"] if last else None}
    out = {"status": "running", "last_step": last["step"] if last else None}
    if last:
        out.update(last_val_acc=last["val_acc"], last_train_loss=last["train_loss"],
                   diverging=bool(not np.isfinite(last["train_loss"]) or last["train_loss"] > 10))
        out["val95_reached_so_far"] = any(r["val_acc"] >= SEL_THR for r in rows)
    return out


def post_grok_stability(r, grok_step):
    """From the per-10-step history: dips below 95 % after the val95 (3-in-a-row) step."""
    its, val, tr = r["its"], r["val_acc"], r["train_acc"]
    g = next(i for i, s in enumerate(its) if s >= grok_step)
    post, post_tr = val[g:], tr[g:]
    below = [v < SEL_THR for v in post]
    eps, cur, longest = 0, 0, 0
    for b in below:
        if b:
            cur += 1
            if cur == 1:
                eps += 1
            longest = max(longest, cur)
        else:
            cur = 0
    last_dip = max((its[g + i] for i, b in enumerate(below) if b), default=None)
    return {"frac_below95": sum(below) / len(below), "dip_episodes": eps, "longest_dip_steps": 10 * longest,
            "min_val_after": min(post), "min_train_after": min(post_tr), "last_dip_step": last_dip,
            "n_evals_after": len(post)}


def mechanism(rows, grok_step):
    if not rows:
        return {}
    at = lambda s: next((x for x in rows if x["step"] >= s), rows[-1])
    out = {"w_norm_start": rows[0]["w_norm"], "w_norm_end": rows[-1]["w_norm"],
           "w_norm_max": max(x["w_norm"] for x in rows),
           "softmax_collapse_max": max(x["softmax_collapse"] for x in rows),
           "softmax_collapse_end": rows[-1]["softmax_collapse"],
           "spec_norm_end": rows[-1]["spec_norm_max"]}
    if grok_step is not None:
        out["w_norm_at_grok"] = at(grok_step)["w_norm"]
        after = [x["cos_gw"] for x in rows if x["step"] >= grok_step and x["cos_gw"] is not None]
        out["cos_gw_median_after"] = float(np.median(after)) if after else None
    return out


def agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": float(statistics.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "n": len(xs)}


def selected_lr(sel, arm):
    if arm == "adam":
        return UNI_ARMS["adam"][4][0]
    return sel[arm]["lr"] if sel and arm in sel else None


def main_arm(R, arm, lr, active, el=None):
    """All seeds of one arm at one lr (Phase 2), or of one decay cell (Phase 3, el given)."""
    seeds = UNI_SEEDS if el is None else UNI_DECAY_SEEDS
    runs, hist = [], {}
    for s in seeds:
        if el is None or el == (UNI_FIXED_WD[arm] if arm in UNI_FIXED_WD else UNI_ARMS[arm][3]):
            label = f"uni_main_{arm}_lr{lr_tag(lr)}_s{s}"
        elif arm in UNI_FIXED_WD:  # el is the weight_decay value itself
            label = f"uni_decay_{arm}_lr{lr_tag(lr)}_wd{el:g}_s{s}"
        else:
            label = f"uni_decay_{arm}_lr{lr_tag(lr)}_el{el:g}_s{s}"
        st = run_status(R, label, active)
        row = {"seed": s, "label": label, "status": st["status"]}
        if "summary" in st:
            sm, u = st["summary"], st["summary"]["unified"]
            row.update(val95=u["val95_consec3"], val99=u["val99_consec3"], val95_first=u["val95_first"],
                       train99=u["train99_first"], tail10=u["tail10_mean_val_acc"], final=u["final_val_acc"],
                       min_after=u["val95_min_after"], state_bytes=sm["optimizer_state_bytes"],
                       fpd_cycles=sm.get("fpd_cycles", 0), fpd_extra=sm.get("fpd_extra_fullbatch_passes", 0),
                       sec_per_step=sm["sec_per_step"], wd=sm["args"]["weight_decay"])
            if el is None:
                r = res(R, label)
                if r is not None:
                    hist[s] = r
                    if u["val95_consec3"] is not None:
                        row.update(post_grok_stability(r, u["val95_consec3"]))
                row.update(mechanism(log_rows(R, label), u["val95_consec3"]))
        else:
            row.update({k: v for k, v in st.items() if k != "status"})
        runs.append(row)
    done = [r for r in runs if r["status"] in ("grokked", "no_grok", "diverged")]
    gro = [r for r in done if r["status"] == "grokked"]
    out = {"arm": arm, "lr": lr, "n_seeds": len(seeds), "n_done": len(done), "n_grokked": len(gro),
           "complete": len(done) == len(seeds), "runs": runs,
           "val95": agg([r["val95"] for r in gro]), "val99": agg([r.get("val99") for r in gro]),
           "n_val99": sum(r.get("val99") is not None for r in gro),
           "tail10": agg([r.get("tail10") for r in done])}
    if el is None and done:
        for k in ("frac_below95", "dip_episodes", "longest_dip_steps", "min_val_after", "min_train_after",
                  "last_dip_step"):
            out[k] = agg([r.get(k) for r in gro])
        for k in ("w_norm_start", "w_norm_at_grok", "w_norm_end", "w_norm_max", "softmax_collapse_max",
                  "softmax_collapse_end", "cos_gw_median_after", "spec_norm_end"):
            out[k] = agg([r.get(k) for r in done])
        out["state_bytes"] = next((r["state_bytes"] for r in done if "state_bytes" in r), None)
        out["fpd_extra"] = agg([r.get("fpd_extra") for r in done])
        out["fpd_cycles"] = agg([r.get("fpd_cycles") for r in done])
        base_passes = BUDGET * BATCH / N_TRAIN
        out["base_epoch_equiv"] = base_passes
        out["fpd_extra_ratio"] = (out["fpd_extra"]["median"] / base_passes) if out["fpd_extra"]["median"] else 0.0
        out["sec_per_step_parallel"] = agg([r.get("sec_per_step") for r in done])
        out["wd"] = next((r["wd"] for r in done if "wd" in r), None)
    return out, hist


# ------------------------------------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
    return plt


def fig_curves(plt, main, hists, out):
    fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.7), sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, MAIN5):
        H = hists.get(arm, {})
        m = main[arm]
        if not H:
            ax.text(0.5, 0.5, "pending", ha="center", transform=ax.transAxes)
        else:
            n = min(len(h["its"]) for h in H.values())
            x = np.array(next(iter(H.values()))["its"][:n])
            V = np.array([h["val_acc"][:n] for h in H.values()]) * 100
            T = np.array([h["train_acc"][:n] for h in H.values()]) * 100
            ax.fill_between(x, V.min(0), V.max(0), color=COLORS[arm], alpha=0.25, lw=0)
            ax.plot(x, V.mean(0), color=COLORS[arm], lw=1.2, label="new questions (validation)")
            ax.plot(x, T.mean(0), color="#444444", lw=0.8, ls="--", label="training questions")
        ax.axhline(95, color="#999999", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"{SHORT[arm]}\nlr {m['lr']:g} · {m['n_grokked']}/{m['n_done']} grokked", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("step (log scale)")
    axes[0].set_ylabel("% answered correctly")
    axes[0].legend(fontsize=6, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "uni_curves.png"))
    plt.close(fig)


def fig_stability(plt, main, hists, out, bin_steps=50):
    from matplotlib.colors import ListedColormap
    rows, labels = [], []
    nb = BUDGET // bin_steps
    for arm in MAIN5:
        for r in main[arm]["runs"]:
            h = hists.get(arm, {}).get(r["seed"])
            if h is None:
                continue
            code = np.zeros(nb)  # 0 = before grokking / never grokked
            its = np.array(h["its"])
            val = np.array(h["val_acc"])
            if r.get("val95") is not None:
                code[:] = 0
                for b in range(nb):
                    sel = (its > b * bin_steps) & (its <= (b + 1) * bin_steps)
                    if not sel.any() or its[sel][0] < r["val95"]:
                        continue
                    code[b] = 2 if (val[sel] < SEL_THR).any() else 1
            rows.append(code)
            labels.append(f"{SHORT[arm]} · seed {r['seed']}")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 0.17 * len(rows) + 0.9))
    cmap = ListedColormap(["#e3e3e3", "#cfe3f3", "#b2182b"])
    ax.imshow(np.array(rows), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2,
              extent=[0, BUDGET, len(rows) - 0.5, -0.5])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(False)
    k = 0
    for arm in MAIN5:
        k += len(hists.get(arm, {}))
        ax.axhline(k - 0.5, color="white", lw=1.5)
    ax.set_xlabel("training step")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e3e3e3", label="not grokked yet / never"),
                       Patch(color="#cfe3f3", label="grokked, ≥ 95% on new questions"),
                       Patch(color="#b2182b", label="grokked, then a dip below 95%")],
              fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "uni_stability.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".", help="the directory given to sweep.py (holds results/)")
    a = ap.parse_args()
    R, OUT = os.path.join(a.out_dir, "results"), os.path.join(a.out_dir, "analysis")
    os.makedirs(OUT, exist_ok=True)
    active = active_labels()
    sel = jload(os.path.join(R, "uni_selected.json")) if os.path.exists(os.path.join(R, "uni_selected.json")) else {}
    sel0p = os.path.join(R, "uni_selected_v0.json")
    sel0 = jload(sel0p) if os.path.exists(sel0p) else {}

    # Phase 1 tuning
    tune = {}
    for arm in ARMS[1:]:
        runs = []
        for lr in UNI_ARMS[arm][4]:
            label = f"uni_tune_{arm}_lr{lr_tag(lr)}_s{UNI_TUNE_SEED}"
            st = run_status(R, label, active)
            row = {"lr": lr, "status": st["status"], "label": label}
            if "summary" in st:
                u = st["summary"]["unified"]
                row.update(val95=u["val95_consec3"], val99=u["val99_consec3"], tail10=u["tail10_mean_val_acc"],
                           wd=st["summary"]["args"]["weight_decay"])
            else:
                row.update({k: v for k, v in st.items() if k != "status"})
            runs.append(row)
        running = [r for r in runs if r["status"] in ("running", "not_started")]
        v0 = sel0.get(arm, {}).get("lr") if isinstance(sel0.get(arm), dict) else None
        tune[arm] = {"runs": runs, "selected_lr": selected_lr(sel, arm), "v0_lr": v0,
                     "lr_pending": bool(running), "running_lrs": [r["lr"] for r in running],
                     "at_edge": sel.get(arm, {}).get("at_edge") if isinstance(sel.get(arm), dict) else None}

    # Phase 2 main
    main_, hists = {}, {}
    for arm in ARMS:
        lr = selected_lr(sel, arm)
        main_[arm], hists[arm] = main_arm(R, arm, lr, active)
        main_[arm]["lr_pending"] = tune.get(arm, {}).get("lr_pending", False)

    # Phase 3 decay ablation
    decay = {}
    for arm in ARMS[1:]:
        lr = selected_lr(sel, arm)
        cells = {}
        fixed = arm in UNI_FIXED_WD
        values = [0.0, UNI_FIXED_WD[arm]] if fixed else list(UNI_DECAY_EL)
        for el in values:
            c, _ = main_arm(R, arm, lr, active, el=el)
            c["reused_from_main"] = el == (UNI_FIXED_WD[arm] if fixed else UNI_ARMS[arm][3])
            cells[f"{el:g}"] = c
        decay[arm] = {"lr": lr, "default_eta_lambda": UNI_ARMS[arm][3], "cells": cells,
                      "axis": "weight_decay (coupled L2, Adam)" if fixed else "eta_lambda",
                      "values": values, "default_wd": UNI_FIXED_WD.get(arm),
                      "complete": all(c["complete"] for c in cells.values())}

    # timing (one run per arm alone on the GPU)
    timing = {}
    for arm in ARMS:
        lr = selected_lr(sel, arm)
        label = f"uni_timing_{arm}_lr{lr_tag(lr)}_s1"
        st = run_status(R, label, active)
        row = {"lr": lr, "status": st["status"]}
        if "summary" in st:
            sm = st["summary"]
            row.update(sec_per_step=sm["sec_per_step"], wall_sec=sm["wall_sec"],
                       val95=sm["unified"]["val95_consec3"])
            if row["val95"] is not None:
                row["sec_to_val95"] = row["val95"] * sm["sec_per_step"]
        timing[arm] = row

    # peak GPU memory (one run per arm, alone; also a second wall-clock sample)
    memory = {}
    for arm in ARMS:
        lr = selected_lr(sel, arm)
        label = f"uni_timing_{arm}_lr{lr_tag(lr)}_s1"
        st = run_status(R, label, active)
        if "summary" in st and st["summary"].get("peak_gpu_mem_allocated_bytes") is None:
            older = run_status(R, f"uni_mem_{arm}_lr{lr_tag(lr)}_s1", active)  # timing runs made before memory was logged
            st = older if "summary" in older else st
        row = {"lr": lr, "status": st["status"]}
        if "summary" in st:
            sm = st["summary"]
            row.update(peak_alloc_bytes=sm.get("peak_gpu_mem_allocated_bytes"),
                       peak_reserved_bytes=sm.get("peak_gpu_mem_reserved_bytes"),
                       state_bytes=sm["optimizer_state_bytes"], sec_per_step=sm["sec_per_step"],
                       val95=sm["unified"]["val95_consec3"])
        memory[arm] = row

    pending = {
        "lr": [a for a in ARMS[1:] if tune[a]["lr_pending"]],
        "main": [a for a in ARMS if not main_[a]["complete"] or main_[a]["lr_pending"]],
        "decay": [a for a in decay if not decay[a]["complete"]],
        "timing": [a for a in ARMS if timing[a]["status"] not in ("grokked", "no_grok", "diverged")],
        "memory": [a for a in ARMS if memory[a]["status"] not in ("grokked", "no_grok", "diverged")],
    }
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M %Z"), "results_dir": R, "active_runs": active,
        "protocol": {"budget": BUDGET, "batch": BATCH, "n_train": N_TRAIN, "eta_lambda": UNI_ETA_LAMBDA,
                     "tune_seed": UNI_TUNE_SEED, "seeds": UNI_SEEDS, "eval_every_steps_consec": 100,
                     "threshold": SEL_THR},
        "arms": {a: {"name": NAME[a], "source": SOURCE[a], "lr_grid": UNI_ARMS[a][4],
                     "eta_lambda": UNI_ARMS[a][3], "optimizer": UNI_ARMS[a][0], "loss": UNI_ARMS[a][1],
                     "loss_dtype": UNI_ARMS[a][2]} for a in ARMS},
        "selected_frozen_at": sel.get("_frozen_at"), "tune": tune, "main": main_, "decay": decay,
        "timing": timing, "memory": memory, "pending": pending,
    }

    plt = setup_mpl()
    fig_curves(plt, main_, hists, OUT)
    fig_stability(plt, main_, hists, OUT)
    figs = ["uni_curves.png", "uni_stability.png"]
    out["figures"] = figs
    with open(os.path.join(OUT, "unified_analysis.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {OUT}/unified_analysis.json and {len(figs)} figures; pending: {pending}; active: {len(active)} runs")


if __name__ == "__main__":
    main()
