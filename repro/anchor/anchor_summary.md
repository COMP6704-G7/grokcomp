| run | final test acc | best test acc | first epoch test ≥ 99% |
|---|---|---|---|
| MLP · ⊥AdamW · softmax · wd=0 · loss float32 · lr=0.01 · 300 ep | 100.0% | 100.0% | 120 |
| transformer · ⊥AdamW · softmax · wd=0 · loss float32 · lr=0.001 · 5000 ep | 100.0% | 100.0% | 600 |
| transformer · AdamW · softmax · wd=0 · loss float32 · lr=0.001 · 5000 ep | 4.4% | 4.4% | never |
| transformer · AdamW · softmax · wd=1.5 · loss float32 · lr=0.001 · 5000 ep | 100.0% | 100.0% | 1200 |
| MLP · AdamW · softmax · wd=0 · loss float32 · lr=0.005 · 500 ep | 0.1% | 0.7% | never |
| MLP · ⊥AdamW · softmax · wd=0 · loss float32 · lr=0.005 · 500 ep | 79.2% | 79.4% | never |
| MLP · AdamW · softmax · wd=0 · loss float64 · lr=0.01 · 80000 ep | 0.5% | 0.7% | never |
| MLP · AdamW · stablemax · wd=0 · loss float64 · lr=0.01 · 80000 ep | 98.5% | 98.5% | never |
