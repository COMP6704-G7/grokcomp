# Reproduction guide

Run from the repository root after the README installation. The exact recorded environment is documented in `TESTED_ENVIRONMENT.json`; measured runs used PyTorch 2.5.1+cu124, Python3.10, FP32 and an A100-SXM4-80GB. The task is generated locally, with no dataset or private-repository download.

## Current study

```bash
python -m pytest -q
python scripts/verify_release.py
python scripts/analyze_advantage.py
CUDA_VISIBLE_DEVICES=0 python scripts/confirm_advantage.py --output runs/current_reproduction
python scripts/analyze_advantage.py --root runs/current_reproduction
```

The first three commands test implementation/protocol, validate the archived evidence, and reconstruct current figures/tables. The final two retrain all 12 runs into a fresh directory and analyze them. An A100 is not necessary for execution, but reported timing/memory values are hardware-specific and floating-point trajectories can vary across runtimes. Do not compare CPU smoke timings with the reported CUDA measurements.

Keep all runs sequential on the same idle GPU. `CUDA_VISIBLE_DEVICES` chooses the physical GPU; the code uses its visible CUDA device. Both methods share the seed's initial weights, split and independently generated sample ordering. Full data are evaluated on the same checkpoint every100 updates; step0 is diagnostic. Failed95/99 crossings remain `null`, not artificially successful at the budget. `summary.status` is a legacy **val99** field; for the current primary95 criterion use `summary.val95` or `analysis/per_run.csv:primary_val95_success`.

`--stage tune` runs the six candidates; `--stage confirm` requires the frozen selection. `--stage all` performs both. Rerunning the same output resumes from completed checkpoints or skips completed summaries. Use a **new directory** for an actual fresh replication; the checked-in archive deliberately has summaries but omits full checkpoints. Never change configuration files inside an existing run. Do not use `python -O` for analysis because validation includes Python assertions.

The candidate rule is defined before confirmation in `configs/advantage_validation.json`. Reproduction creates new lock/freeze timestamps; these naturally differ from the archive. Metric values should be checked with the same software/hardware rather than requiring binary-identical timings or PDF metadata.

## Reconstruct earlier figures and statistics

These commands consume included logs without checkpoints:

```bash
python -m lion_experiment.analyze --root results/lion_study
python -m lion_experiment.analyze --root results/matched_budget_study
python scripts/analyze_diagnostics.py
python scripts/analyze_stability.py
```

The generic analyzer's output title is the initial comparison's title; the saved supplementary report additionally labels its retrospective design. Historical raw results are unchanged by this release. See `STUDY_HISTORY.md` for the distinctions.

## Retrain earlier studies and checkpoint audits

The historical scripts use fixed paths under `results/`; use a **separate clean clone**, move its archived `results/` directory aside, and let scripts create fresh results. This avoids changing the distributed evidence.

```bash
mv results results_recorded
mkdir -p results/diagnostics/source_snapshot
cp results_recorded/diagnostics/source_snapshot/group_main_93b2248.py results/diagnostics/source_snapshot/
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.suite --stage pilot
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.suite --stage all
CUDA_VISIBLE_DEVICES=0 python scripts/matched_budget.py
CUDA_VISIBLE_DEVICES=0 python scripts/decay_diagnostics.py
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_original.py --budget 50000
python scripts/audit_training.py
CUDA_VISIBLE_DEVICES=0 python scripts/continue_stability.py
python scripts/analyze_diagnostics.py
python scripts/analyze_stability.py
```

The source-fixture copy must precede original-script reproduction, as in the sequence above. The original script imports the included `grokfast.py` and `tqdm`; it has its original epoch metrics, not synchronized train/validation metrics. The raw `.pt` curve file is included in the archive and is only loaded from this trusted repository.

The independent saved-weight audit and exact state continuation require **regenerated** `checkpoint.pt` files from the initial study. The distributed archive retains the completed audit output, but does not pretend an audit can be rerun without weights. The CPU test for interruption/resume independently exercises exact checkpoint restoration without any archived training weights.

## Documents and attribution

```bash
python -m pip install -r requirements-docs.txt
python scripts/export_contribution.py
python scripts/build_advantage_deliverables.py
# Optional: generate PDFs using a locally installed LibreOffice.
libreoffice --headless --convert-to pdf --outdir deliverables deliverables/Lion_Contribution.docx
```

The contribution exporter renders the consolidated member chapter, not a complete group report. The advantage builder regenerates the current six-slide deck and result note from current archived statistics. Existing PDFs are included for immediate use. LibreOffice, CJK fonts and their versions affect layout; they are not required for training or log analysis.

No training-package files were changed during release organization; `SOURCE_MANIFEST.json` and the current study manifest record their exact digests. `verify_release.py` checks these against the current files and all twelve recorded log hashes. It additionally checks official reference integrity, source snapshot integrity, candidate selection, summary metrics, fresh seed separation, splits and paired initialization.
