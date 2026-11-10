# How optimisers affect grokking: a unified comparison

Five optimisers trained under one protocol on the same grokking task, so their results can be compared directly:

| Method | `--optimizer` | Idea |
|---|---|---|
| AdamW (baseline) | `AdamW` | Adam with decoupled weight decay |
| Lion | `Lion` | sign of an interpolated momentum, one state buffer ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)) |
| ⊥AdamW + StableMax | `OrthoAdamW`, `--loss stablemax` | drop the gradient component along the weights; StableMax instead of softmax ([arXiv:2501.04697](https://arxiv.org/abs/2501.04697)) |
| FPD | `FPD` | Adam plus periodic shrink-and-restore cycles that decay the weights while keeping the training predictions (`fpd.py`) |
| EGD | `EGDAdam` | gradient G = U S Vᵀ of each MLP weight replaced by U Vᵀ, then Adam with weight decay 0.005, the paper's transformer setup ([arXiv:2510.04930](https://arxiv.org/abs/2510.04930)) |

Adam without weight decay is run as a reference.

## Run everything

```bash
pip install -r requirements.txt
bash scripts/run_all.sh OUT_DIR 12
```

`OUT_DIR` receives all outputs; `12` is the number of runs trained in parallel on one GPU (each is small and CPU-bound).
Set `DEVICE=cpu` or `PYTHON=/path/to/python` to override the defaults (`cuda`, `python`). The script runs, in order:

1. the unit tests;
2. learning-rate tuning on seed 100 (23 runs);
3. `scripts/pick_unified.py`: freeze one learning rate per method, and extend a grid by one factor-2 step (at most twice) if the pick is at its edge;
4. the main runs, seeds 1–5 (30 runs), and the decay runs, decay off or on at the frozen learning rate, seeds 1–3 (15 runs);
5. one run per method alone on the GPU for time per step and peak GPU memory (6 runs);
6. `scripts/analyze_unified.py`: tables (`OUT_DIR/analysis/unified_analysis.json`) and figures (`uni_curves.png`, `uni_stability.png`).

Every run writes `OUT_DIR/results/summary_<label>.json` and is skipped if that file exists, so an interrupted run of the script can be restarted with the same command.

## Protocol

- **Task:** x · y mod 97 with y ≠ 0 (9,312 equations), 50% for training, the rest for validation.
- **Network:** 2-layer decoder-only transformer, width 128, 4 heads, 422,784 parameters (from the Grokfast code, [arXiv:2405.20233](https://arxiv.org/abs/2405.20233)).
- **Training:** batch 512, β = (0.9, 0.98) (Lion: (0.9, 0.99)), 10-step linear warm-up, constant learning rate, exactly 20,000 steps with no early stopping.
- **Decay:** decoupled decay is matched as η·λ = 10⁻³ for AdamW and Lion; ⊥AdamW and FPD use none; EGD uses λ = 0.005 inside Adam, as in its paper's setup.
- **Seeds:** a seed fixes the initial weights, the split and the batch order, and is shared by all methods. Learning rates are tuned on seed 100 only, by one rule: reached val95 > earlier val95 > higher validation accuracy over the last 10% > smaller learning rate.
- **val95 (val99):** the first step that starts three consecutive evaluations, 100 steps apart, with validation accuracy ≥ 95% (99%).
- **Stability after grokking:** share of 10-step evaluations below 95% after val95, number of dips, lowest validation and training accuracy.
- **Cost:** optimiser-state bytes, FPD's extra full-batch passes, time per step and peak GPU memory (PyTorch, per process).

## Results (seeds 1–5)

| Method | Selected lr | Grokked | val95 step, median | Seconds to val95 | Peak GPU memory |
|---|---|---|---|---|---|
| AdamW | 2e-3 | 5/5 | 1,210 | 18 | 84 MiB |
| Lion | 1e-4 | 5/5 | 2,610 | 52 | 83 MiB |
| ⊥AdamW + StableMax | 3e-4 | 4/5 | 2,810 | 57 | 84 MiB |
| FPD | 2e-3 | 5/5 | 610 | 17 | 592 MiB |
| EGD | 4e-3 | 5/5 | 510 | 18 | 84 MiB |

Adam without decay does not grok within 20,000 steps (0/5). AdamW, FPD and EGD keep dipping below 95% after grokking; ⊥AdamW does not.
With decay off for every method, FPD, ⊥AdamW and EGD still grok within a few thousand steps while AdamW and Lion need 10,000–20,000, so the ranking is not explained by decay alone.

## Files

| Path | Content |
|---|---|
| `main.py` | one training run: data, model, training loop, logging, summary |
| `sweep.py` | the grids (`uni_tune`, `uni_extend`, `uni_main`, `uni_decay`, `uni_main_decay`, `uni_timing`) and a parallel runner |
| `optim/` | `OrthoAdamW` (`orth.py`), `Lion` (`lion.py`), `EGD` and `EGDAdam` (`egd.py`); `__init__.py` builds any optimiser by name |
| `fpd.py` | Function-Preserving Decay cycles |
| `losses.py` | softmax and StableMax cross-entropy, Softmax Collapse share |
| `metrics.py` | val95/val99, stability, weight-norm and optimiser-state measures |
| `scripts/` | `run_all.sh`, `pick_unified.py`, `analyze_unified.py` |
| `tests/` | unit tests, run first by `run_all.sh` |
| `vendor/` | the official Lion implementation, used only by a test to check `optim/lion.py` |

## Licence

The training code is derived from the Grokfast code and keeps its MIT licence (`LICENSE`). `vendor/lion_official.py` is under the Apache 2.0 licence (`vendor/LICENSE.lion`).
