# ⊥AdamW + StableMax

Owner: Eric Wang · COMP6704 Group 7 · branch `⊥AdamW`
Reproduces Prieto et al., *Grokking at the Edge of Numerical Stability*, ICLR 2025 ([arXiv:2501.04697](https://arxiv.org/abs/2501.04697), [authors' code](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability)).

## The method in brief

After a network has memorised its training set, cross-entropy keeps lowering the loss by making every score larger. That never changes a prediction ("naive loss minimisation"). For networks that are (roughly) homogeneous, the cheapest way to do it is to scale the weights up, so the gradient turns to point along the weights. Eventually the float32 softmax rounds the loss to exactly 0 ("Softmax Collapse"), the gradient vanishes, and learning stops before the network generalises. The paper's two fixes:

| Piece | File | Rule |
|---|---|---|
| **⊥AdamW** | `optim/orth.py` | For each parameter tensor: `g⊥ = g − (⟨w,g⟩/⟨w,w⟩)·w`, rescaled to ‖g‖, then a normal AdamW step |
| **StableMax** | `losses.py` | Softmax with `exp` replaced by `s(x) = x+1` (x ≥ 0), `1/(1−x)` (x < 0) |

Together they make the model grok **without weight decay**.

Implementation notes, checked against the authors' code (`tests/test_orth.py` compares the two gradient for gradient):
- The paper's Eq. 12 projects the whole parameter vector; the authors' code projects **each tensor separately** and **rescales** the result to the original gradient's length. The default here follows the code; `--orth_scope global` and `--orth_no_rescale` give the alternatives.
- Weight decay, if switched on, stays AdamW's decoupled decay and is not projected.
- StableMax runs compute the loss in float64, as the authors do (`--loss_dtype float64`).

## Flags added to `main.py`

All original Grokfast commands still work; these are additions.

| Flag | Default | Meaning |
|---|---|---|
| `--optimizer OrthoAdamW` | | ⊥AdamW (any `torch.optim` name still works, e.g. `AdamW`) |
| `--loss {softmax,stablemax}` | softmax | output normalisation inside the cross-entropy |
| `--loss_dtype {float16,float32,float64}` | float32 | precision of the loss computation |
| `--orth_scope {tensor,global}` | tensor | project per tensor (authors' code) or all weights at once (paper's Eq. 12) |
| `--orth_no_rescale` | off | keep the shorter projected gradient |
| `--orth_skip_1d` | off | leave biases and LayerNorm parameters unprojected |
| `--train_frac` | 0.5 | share of the 9,312 equations used for training |
| `--device` | auto | `cuda`, `cpu`, `mps` |
| `--thresh` | 0.99 | accuracy that counts as fitted / grokked |
| `--log_every` | 10 | epochs between JSON log lines (diagnostics are computed only then) |
| `--stop_after` | 0 | stop once test accuracy has stayed ≥ thresh for this many epochs |
| `--raw_label` | off | use `--label` as given, without the automatic suffix |

## What each run writes (`results/`, git-ignored)

- `log_<label>.jsonl`, one line every `--log_every` epochs: step, train/test loss and accuracy, total weight norm, largest singular value of any weight matrix, **cos(g, w)** before any projection (negative = the descent step inflates the weights), the **share of training samples in Softmax Collapse** (float32 loss exactly 0), wall time.
- `summary_<label>.json`: fit step (train ≥ 99%), grok step (test ≥ 99% first), sustained step (test stays ≥ 99% to the end), delay = grok − fit, seconds per step, all arguments.

## Experiments

| ID | Grid | Runs | Question |
|---|---|---|---|
| E0 | `repro/anchor/run_anchor.sh` | 8 | The authors' own Figure 1 and 6 runs, their code at commit 720d244, unchanged |
| E1 | `sweep.py orth_e1 --with_reference` | 9 + 9 | ⊥AdamW + StableMax (wd 0) vs AdamW (wd 1), lr ∈ {3e-4, 1e-3, 3e-3}, 3 seeds |
| E2 | `sweep.py orth_e2 --best_lr <lr>` | 9 | weight decay 0 / 0.1 / 1 on top of ⊥AdamW + StableMax |
| E3 | `sweep.py orth_e3 --best_lr <lr>` | 9 | training fraction 0.3 / 0.5 / 0.7 |
| E4 | `sweep.py orth_e4 --best_lr <lr>` | 39 | ⊥ × StableMax × wd; global vs per-tensor, rescale vs not, skip 1-D; softmax float32 vs float64 |

`scripts/pick_best_lr.py` chooses `<lr>` from E1: the learning rate where every seed grokked with the smallest mean sustained step.
E2–E4 compute the loss in float64 (so rounding cannot cause collapse) except the rows labelled float32; the E1 AdamW reference uses float32, as in grokfast.

## Results, first round (A6000, 2026-10-28)

Sustained step = test accuracy stays ≥ 99% from then on; ranges are over 3 seeds.

- **E0** reproduces the paper: ⊥AdamW groks the MLP by epoch 120 and the transformer by epoch 600 with no weight decay; AdamW + wd 1.5 needs 1,200; plain AdamW stays at 4%; StableMax alone reaches 98.5% after 80k epochs.
- **E1** best lr = 3e-4: ⊥AdamW + StableMax (wd 0) sustained at 2,100–4,000 vs AdamW (wd 1) at 5,800–8,600, with the weight norm flat at ~122. At lr 1e-3 / 3e-3 the weight norm still grows (to 340 / ~2,100) and grokking becomes slow or unstable: Adam's per-coordinate scaling and the finite step size leak the update outward.
- **E2** wd 0.1 is slightly faster (2,000–3,000). wd 1 groks fast but one seed collapses to chance (weights shrink to ~1): with the radial gradient removed nothing opposes the decay.
- **E3** 70% data: sustained at 1,250–1,400. 30%: memorises by step ~700 but does not grok within 100k steps.
- **E4** ⊥ alone (softmax) groks (1,800–3,200); StableMax alone and plain AdamW do not (test ~41–50% at 100k). ⊥ + wd 1 collapses in 2/3 (softmax) and 1/3 (StableMax) seeds. Global projection (Eq. 12 literally) is much slower (48k–67k) than per-tensor; no-rescale fails 1/3; skipping 1-D params is slightly slower.

The full write-up with figures is the Group project part of the course companion page.

## Running

```bash
python -m pytest -q tests                        # 9 checks, seconds, CPU is fine
bash scripts/gpu_job.sh anchor                   # E0
bash scripts/gpu_job.sh all 8 4                  # E1 → best lr → E2, E3, E4, 8 GPU + 4 CPU workers
python summarize.py orth_e1                      # table + figures in results/
```

Set `PYTHON` to the environment's interpreter. `scripts/gpu_job.sh` unsets `LD_LIBRARY_PATH`, because a system CUDA there shadows the libraries bundled with pip-installed PyTorch.

**Throughput.** Each run is a 0.4M-parameter model driven by a Python loop, so it is limited by the CPU, not the GPU. On an A6000 workstation (12 cores) one run does about 45 steps/s on the GPU; 10 runs sharing one GPU reach only about 78 steps/s in total, and 11 single-thread CPU runs about 62 steps/s. The sweeps therefore mix both: `--jobs N` runs on the GPU plus `--cpu_jobs M` on the CPU, every process limited to one CPU thread (`OMP_NUM_THREADS=1`). CPU and GPU runs are not bit-identical, so the device is recorded in each summary's `args`.

Interrupted sweeps can be restarted: a run whose `summary_<label>.json` exists is skipped. A run that exits with an error is retried once (a CUDA context occasionally fails to start when many runs launch on one GPU).
