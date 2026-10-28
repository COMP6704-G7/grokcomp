# Study history and interpretation

All stages below are retained, including results that do not favor Lion. They answer different questions and must not be pooled as interchangeable replications.

| Stage | Design | Main observation | Evidence |
|---|---|---|---|
| Initial local comparison | AdamW lr .001 / decay 1; Lion lr selected on seed0 / decay5; new seeds1–3; stop after confirmed99 | AdamW first val99 mean 1366.7 steps vs Lion 1566.7; Lion uses half the moment storage | `results/lion_study/analysis/` |
| Supplementary equal-count search | Added after seeing initial results; 3 AdamW candidates, original Lion results reused | Selected AdamW lr .002; mean val99 933.3 vs Lion 1566.7 | `results/matched_budget_study/analysis/` |
| Original-code reproduction | Original Adam / decay0, seed0, 50,000 steps | Train99.63%, val87.84%; did not reproduce upstream README's threshold timing | `results/diagnostics/original_reproduction_summary.json` |
| Decay diagnostic | Seed0; fixed10k; four decay levels each optimizer, matched ηλ across methods | Generalization changes substantially with decay; low-decay Lion exceeds corresponding AdamW within the horizon | `results/diagnostics/decay_comparison.csv` |
| Six-checkpoint continuation | Continue original formal runs to10k with original states and sampler | All original AdamW seeds fall below95 after first99 confirmation; all original Lion seeds stay≥99. Retrospective stability is not the prespecified first-crossing metric | `results/diagnostics/formal_stability/` |
| Current fresh-seed confirmation | Low shrinkage identified exploratorily; lock design before seed7 tuning; freeze then confirm on4–6; 3 candidates each; all fixed10k; ηλ=1e−5 | Lion val95 success3/3 vs AdamW0/3; tail98.76% vs41.84%. No universal superiority claim | `results/advantage_validation/analysis/` |

The initial study's baseline was a local choice inferred from comments, not an agreed group setting: the original CLI actually defaults to Adam with zero decay. The diagnostic record corrects that distinction. The current study is a prospective fresh-seed check within a previously selected regime, not an entirely unselected discovery.

Original protocol notes in `PROTOCOL.md`, `ALGORITHM.md` and legacy deliverables describe the earlier stages. Current parameters are in `ADVANTAGE_VALIDATION_PROTOCOL.md` and `configs/advantage_validation.json`. In this release, previously linked repeated Lion evidence is materialized, with identical data.
