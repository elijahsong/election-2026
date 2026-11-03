"""House Stage 1 experiments: run (run_ols / run_wls / run_ols_fixed + CLI), evaluation (shared data prep and
metrics), tracking (saved runs), ensemble.

From a notebook, put src/ on sys.path and `from house.experiments import evaluation, tracking`. CLI, from the repository
root: `PYTHONPATH=src python -m house.experiments.run ...`.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root (results/ lives there)
