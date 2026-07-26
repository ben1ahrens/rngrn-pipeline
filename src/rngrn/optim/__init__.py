"""optim/ — the outer loops around fit() (Stages 3 & 6).

sweep     : enumerate config overrides -> fit() per point -> run index.
benchmark : aggregate the run index into a cross-run comparison table.
"""
from .sweep import run_sweep, grid_points          # noqa: F401
from .benchmark import build_table, to_markdown     # noqa: F401
