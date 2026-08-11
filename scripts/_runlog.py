"""_runlog.py — shared auditability helper for the experiment scripts.

The library (io.py + index.py + config.py) already has three orthogonal IDs
(dataset_hash, config_id, run_id), a frozen config, a provenance block, atomic JSON
writes and an append-only run index. The experiment scripts (scripts/exp*.py), which
produced every scientific result in this project, had none of it: bare JSON/CSV in
experiments/ with no run id, no git sha, no config hash. This module adds that layer
WITHOUT touching what any script computes -- it only records more about a run that
already happened, alongside the script's existing output.
"""
from __future__ import annotations
import hashlib
import json
import os

from rngrn.io import new_run_id, atomic_write_json
from rngrn.index import open_index
from rngrn.utils import provenance


def args_config_id(args: dict) -> str:
    """sha256 of the canonical (sorted-key) JSON of the script's resolved argument
    dict, truncated to 12 hex chars -- same recipe as Config.config_id() in
    config.py (sha256 of canonical JSON, truncated)."""
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def start(script: str, args: dict) -> dict:
    """Mint a run_id + config_id + provenance block for one script invocation.

    Call once near the top of a script's __main__ block, after argparse has resolved
    `args` (pass vars(args)). Returns a dict meant to be written alongside the
    script's normal output (`write_meta`) and merged into the run index (`record`).
    """
    return dict(
        run_id=new_run_id(script),
        script=script,
        config_id=args_config_id(args),
        args=args,
        provenance=provenance(),
    )


def write_meta(out_path: str, run: dict):
    """Write the run's identity/provenance block to a sidecar JSON next to the
    script's normal output (e.g. experiments/exp11_robustness_baseline.csv ->
    experiments/exp11_robustness_baseline.runlog.json). Never touches the script's
    own output file."""
    root, _ = os.path.splitext(out_path)
    atomic_write_json(f"{root}.runlog.json", run)


def record(runs_root: str, run: dict, row: dict, backend: str = "jsonl"):
    """Append one flat-scalar row to the SAME run index the library uses
    (<runs_root>/runs.jsonl or index.db, per `backend`, see index.py).

    `row` is the caller's own flat-scalar summary fields. Run-identity fields are
    merged in LAST via dict.update() so they always win a key collision with a
    scorer field, per project convention.
    """
    merged = dict(row)
    merged.update(run_id=run["run_id"], script=run["script"],
                  config_id=run["config_id"],
                  git_revision=run["provenance"].get("git_revision"))
    open_index(runs_root, "runs", backend).append(merged)
