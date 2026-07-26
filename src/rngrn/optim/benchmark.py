"""benchmark.py — the cross-run harness (Stage 6).

Reads the append-only run index (experiments/runs.jsonl) and aggregates it into a
comparison table: configs x seeds x targets, with the priority-order columns and
an IDENTIFIABILITY headline metric (spread of an outcome across seeds that all
reproduce the pattern — the metric that respects a degenerate inverse problem).
Pure aggregation: it reconstructs the table from the ledger alone.
"""
from __future__ import annotations
from collections import defaultdict
from statistics import mean, pstdev

from .. import io as IO

# columns surfaced in the comparison table (priority order + identifiability)
COLUMNS = ["config_id", "source", "dataset", "N", "m", "form", "strategy", "n_seeds",
           "kstar_rel_err_mean", "recovered_turing_frac", "sign_match_frac_mean",
           "loss_mean", "kstar_identifiability_std"]


def _dataset_of(row):
    """The TRUE data identity of a run, source-appropriate (not the unused `system`
    default). Falls back to legacy `system`/`dataset_id` for rows written before the
    source/dataset_label columns existed."""
    return (row.get("dataset_label") or row.get("dataset_id")
            or row.get("system") or row.get("dataset_hash"))


def _group_key(row):
    return (row.get("config_id"), row.get("source"), _dataset_of(row),
            row.get("N"), row.get("m"), row.get("form"), row.get("strategy"))


def build_table(runs_root="experiments", backend="jsonl") -> list[dict]:
    """Aggregate the run index into one row per (config x target), averaged over seeds."""
    rows = IO.read_run_index(runs_root, backend=backend)
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    table = []
    for key, members in groups.items():
        cfg_id, source, dataset, N, m, form, strategy = key
        kstar_errs = [x["kstar_rel_err"] for x in members if _isnum(x.get("kstar_rel_err"))]
        turing = [1.0 if x.get("recovered_turing") else 0.0 for x in members]
        signs = [x["sign_match_frac"] for x in members if _isnum(x.get("sign_match_frac"))]
        losses = [x["loss"] for x in members if _isnum(x.get("loss"))]
        # identifiability: spread of recovered k* across seeds that all landed in-regime
        kstars_ok = [x["kstar_model"] for x in members
                     if x.get("recovered_turing") and _isnum(x.get("kstar_model"))]
        table.append(dict(
            config_id=cfg_id, source=source, dataset=dataset, N=N, m=m, form=form,
            strategy=strategy, n_seeds=len(members),
            kstar_rel_err_mean=_safe(mean, kstar_errs),
            recovered_turing_frac=_safe(mean, turing),
            sign_match_frac_mean=_safe(mean, signs),
            loss_mean=_safe(mean, losses),
            kstar_identifiability_std=_safe(pstdev, kstars_ok) if len(kstars_ok) > 1 else float("nan"),
        ))
    return table


def _isnum(v):
    return isinstance(v, (int, float)) and v == v  # not None, not NaN


def _safe(fn, xs):
    return float(fn(xs)) if xs else float("nan")


def to_markdown(table: list[dict]) -> str:
    if not table:
        return "_(run index empty)_"
    hdr = "| " + " | ".join(COLUMNS) + " |\n" + "|" + "---|" * len(COLUMNS) + "\n"
    body = ""
    for row in table:
        body += "| " + " | ".join(
            (f"{row.get(c):.4g}" if isinstance(row.get(c), float) else str(row.get(c)))
            for c in COLUMNS) + " |\n"
    return hdr + body
