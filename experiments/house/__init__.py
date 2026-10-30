"""House Stage 1 experiments: run (run_ols + CLI), evaluation (shared data prep and metrics), tracking (saved runs).

Run from the repository root (`python -m experiments.house.run ...`), or put the root on sys.path in a notebook.
house.regression lives in src/, which isn't installed, so importing this package puts src/ on sys.path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
