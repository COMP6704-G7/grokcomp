"""Run the exact archived group main.py with its actual Adam/decay=0 defaults."""
import argparse
import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--budget", type=int, default=50000)
parser.add_argument("--label", default="diagnostic_original_adam0")
args = parser.parse_args()
source = ROOT / "results/diagnostics/source_snapshot/group_main_93b2248.py"
if not source.exists():
    raise FileNotFoundError("The archived original source is required: " + str(source))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
sys.argv = [str(source), "--label", args.label, "--budget", str(args.budget)]
runpy.run_path(str(source), run_name="__main__")
