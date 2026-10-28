## Lion under matched low per-step shrinkage

A protocol locked before confirmation compared custom Lion with PyTorch AdamW on three fresh seeds (4–6). Each optimizer received three 10,000-step learning-rate trials on seed 7. All candidates used identical per-step shrinkage ηλ=1e−5, paired group-model initialization and splits, and full-set evaluation every 100 steps. Configuration selection prioritized the first three-checkpoint 95% validation crossing, with identical tie breakers. No early stopping was used.

Lion reached 95% validation accuracy in 3/3 confirmation runs, compared with 0/3 for AdamW. Mean validation accuracy over steps 8100–10000 was 98.76% versus 41.84%, a difference of +56.92 percentage points. Persistent optimizer state was approximately 1.61 versus 3.23 MiB. Failed threshold crossings are right-censored and excluded from successful-crossing means.

The low-shrinkage regime was identified in earlier exploratory work; these results confirm behavior within that regime on unused seeds, not universal superiority or a statistically significant effect. No confirmation outcomes were used to revise the locked configuration.
