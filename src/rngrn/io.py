"""io.py — run directories, atomic writes, checkpoints, the run index.

Run layout (three-ID scheme in nn-research-codebase-principles.md §5):
  experiments/runs/<run_id>/
    config/frozen_config.yaml     # the FULLY RESOLVED config that actually ran
    checkpoints/                  # model state
    results/                      # train_results.json, analysis_results.json, ...
    figures/

The run index (experiments/runs.jsonl) is an append-only ledger, one row per run,
so sweeps and the benchmark harness are queryable without globbing directories.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone


def new_run_id(name: str = "run") -> str:
    return f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def run_dir(root: str, run_id: str) -> str:
    d = os.path.join(root, "runs", run_id)
    for sub in ("config", "checkpoints", "results", "figures"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def atomic_write_json(path: str, obj: dict):
    """Write JSON to a temp file and rename on success (never a half-written file)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def save_results(rdir: str, name: str, obj: dict):
    """Each producer writes its OWN result file (train_results / analysis_results / ...)."""
    atomic_write_json(os.path.join(rdir, "results", name), obj)


def append_run_index(root: str, row: dict, backend: str = "jsonl"):
    """Append one run row to the run index. backend in {'jsonl','sqlite'} (index.py)."""
    from .index import open_index
    open_index(root, "runs", backend).append(row)


def read_run_index(root: str, backend: str = "jsonl") -> list[dict]:
    from .index import open_index
    return open_index(root, "runs", backend).read()


def save_checkpoint(rdir: str, model, extra: dict | None = None):
    import torch
    payload = dict(state_dict=model.state_dict(), N=model.N, form=model.form,
                   n_hill=model.n_hill, dispersion_backend=model.dispersion_backend)
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(rdir, "checkpoints", "model.pt"))


def load_checkpoint(rdir: str, map_location="cpu"):
    """Load a checkpoint. `map_location` defaults to 'cpu' so a checkpoint saved on a CUDA
    host still loads on a CPU-only one; pass e.g. 'cuda' or a torch.device to override."""
    import torch
    from .model import RNGRN
    payload = torch.load(os.path.join(rdir, "checkpoints", "model.pt"), weights_only=False,
                         map_location=map_location)
    model = RNGRN(N=payload["N"], form=payload["form"], n_hill=payload["n_hill"],
                  dispersion_backend=payload.get("dispersion_backend", "eig"))
    model.load_state_dict(payload["state_dict"])
    return model, payload
