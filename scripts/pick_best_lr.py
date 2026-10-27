"""Pick ⊥AdamW's learning rate from E1: among lrs where every seed grokked (sustained),
the one with the smallest mean sustained step. Prints the lr, e.g. 0.001."""
import glob
import json
import re
import statistics
from collections import defaultdict

runs = defaultdict(list)
for f in glob.glob("results/summary_orth_e1_orth_lr*_s*.json"):
    s = json.load(open(f))
    runs[s["args"]["lr"]].append(s["sustained_step"])
ok = {lr: statistics.mean(v) for lr, v in runs.items() if v and all(x is not None for x in v)}
if not ok:  # fall back to the most seeds grokked, then the earliest
    ok = {lr: (-sum(x is not None for x in v), statistics.mean([x for x in v if x is not None] or [1e12]))
          for lr, v in runs.items()}
print(min(ok, key=ok.get))
