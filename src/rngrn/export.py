"""export.py — the run record as TIDY (long) CSV, ready for pandas / seaborn / ggplot.

WHY LONG AND NOT WIDE. The indexes are one row per run with ~120 columns, which is a record,
not a plotting frame: to draw "kstar_fft_rel_err against adam_steps, faceted by form" from a
wide row you must first know which of the 120 columns are identifiers and which are
measurements, and reshape. Tidy format states that explicitly — ONE OBSERVATION PER ROW:

    <identifier columns...>, variable, value, value_num

`variable` names the measurement, `value` is its verbatim text (so a categorical observation
like morphology_pred survives), and `value_num` is the numeric reading or empty when the
observation is not numeric. Booleans are numeric (0/1) as well as text, because a bar chart
of `recovered_turing` wants the number and a legend wants the word.

Four tables, one file each:
  runs_tidy.csv            one row per (run_id, variable)         <- runs.jsonl
  target_reports_tidy.csv  one row per (report_id, variable)      <- target_reports.jsonl
  lgen_tidy.csv            one row per (run_id, L, variable)      <- lgen_eval.jsonl
  history_tidy.csv         one row per (run_id, step, member, variable)  <- the run npz

The HYPERPARAMETERS stay as identifier columns on the runs table (see RUN_ID_COLS), which is
what makes an outcome-vs-hyperparameter plot a one-liner: group by the hyperparameter column,
filter `variable` to the outcome. A sweep already writes one run row per cell (optim/sweep.py
calls fit() per point), so the sweep axis is a column like any other.

Stdlib `csv` only — a tidy export must not be the reason a plotting environment needs pandas.
"""
from __future__ import annotations
import csv
import os

import numpy as np

from .index import open_index
from . import plotdata as PD

# IDENTIFIER columns of the runs table: what you group, facet or colour BY. Everything else on
# the row becomes an observation. The hyperparameters are here deliberately — they are the x
# axis of a tuning plot, not a measurement of the run.
RUN_ID_COLS = (
    # `arm_id` is an IDENTIFIER, not a measurement: it is the seed-independent config
    # identity optim.benchmark groups on (D-EVID-13). Without it here the tidy export melts
    # it into an observation row and you cannot group or facet by arm — the one thing it
    # exists for.
    "run_id", "config_id", "arm_id", "git_sha", "_ts",
    # what problem
    "source", "dataset_label", "dataset_id", "sample_key", "system", "arm",
    "N", "m", "form", "observed_idx", "hidden_idx",
    # HYPERPARAMETERS / execution path
    "seed", "model_seed", "strategy", "adam_steps", "adam_lr", "lbfgs_steps", "grad_clip",
    "n_restarts_requested", "tau", "jac_floor", "split_hinges", "hinge_k_min_frac",
    "staging_off_frac", "staging_ramp_frac", "staging_keys", "detach_xstar",
    "w_kstar", "w_turing", "w_anticollapse", "w_anchor", "w_resid", "w_morphology",
    "w_param_prior", "dratio_centre", "dratio_spread", "ratio_update_every",
    "batched", "device", "dispersion_backend", "model_init", "nondim",
    "d_init_from_kstar", "deterministic", "n_grid", "history_every", "morphology_rollout",
)

# `n_seeds_requested` is what aggregate_target_report actually emits; this tuple named
# `n_seeds` (which only benchmark.build_table emits, into a DIFFERENT table), so
# target_reports_tidy.csv shipped a permanently empty identifier column while the seed count
# was demoted to an observation row — "group by n_seeds" collapsed to one NaN bucket.
REPORT_ID_COLS = ("report_id", "config_id", "git_sha", "_ts",
                  "dataset_id", "sample_key", "form", "n_seeds_requested", "seeds")

LGEN_ID_COLS = ("run_id", "git_sha", "_ts", "L", "L_over_L_train", "L_train",
                "is_L_train", "n", "dx", "grid_rule", "n_grid")

TIDY_TAIL = ("variable", "value", "value_num")


def _num(v):
    """The numeric reading of a value, or '' when it has none. Bools count as 0/1."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    return ""


def _melt(rows, id_cols) -> list:
    """One dict per (row, non-identifier column). The whole of tidy, in six lines."""
    out = []
    for row in rows:
        ids = {c: row.get(c) for c in id_cols}
        for k, v in row.items():
            if k in id_cols:
                continue
            out.append(dict(ids, variable=k, value=("" if v is None else str(v)),
                            value_num=_num(v)))
    return out


def write_csv(path: str, id_cols, records) -> int:
    """Write tidy records to `path`. Returns the number of DATA rows written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(id_cols) + list(TIDY_TAIL)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    return len(records)


# --------------------------------------------------------------------------------------
# per-table melts
# --------------------------------------------------------------------------------------
def tidy_runs(runs_root: str, backend: str = "jsonl") -> list:
    return _melt(open_index(runs_root, "runs", backend).read(), RUN_ID_COLS)


def tidy_target_reports(runs_root: str, backend: str = "jsonl") -> list:
    rows = open_index(runs_root, "target_reports", backend).read()
    # `per_seed` and `frozen_config` are nested records, not observations: a per-seed audit
    # trail flattened into (variable, value) pairs would be unreadable and the frozen config
    # is already on disk per run. Dropped from the melt, deliberately and not silently.
    rows = [{k: v for k, v in r.items() if k not in ("per_seed", "frozen_config")}
            for r in rows]
    return _melt(rows, REPORT_ID_COLS)


def tidy_lgen(runs_root: str, backend: str = "jsonl") -> list:
    return _melt(open_index(runs_root, "lgen_eval", backend).read(), LGEN_ID_COLS)


# --------------------------------------------------------------------------------------
# the training trajectory — read from each run's npz, not from an index
# --------------------------------------------------------------------------------------
HISTORY_ID_COLS = ("run_id", "step", "member", "is_best_member")


def tidy_history(runs_root: str, backend: str = "jsonl", members: str = "best") -> list:
    """One row per (run_id, step, member, variable) over every run whose npz has a history.

    members='best' (DEFAULT) exports only the member that BECAME the run's answer, which is
    what a learning curve for "this run" means and keeps the file at ~S*(Q+P) rows. 'all'
    exports every recorded member — S*B*(Q+P) rows, ~112k for one 400-step 64-restart run —
    for a figure about the spread across restarts. Nothing is dropped from the npz either
    way: the arrays hold every member and this flag chooses what the CSV projects.

    A member's rows are emitted only for the steps it was actually recorded at; the NaN cells
    of a member that died mid-run are skipped rather than exported as blank observations.
    """
    if members not in ("best", "all"):
        raise ValueError(f"members must be 'best' or 'all'; got {members!r}")
    out = []
    runs_dir = os.path.join(runs_root, "runs")
    if not os.path.isdir(runs_dir):
        return out
    for run_id in sorted(os.listdir(runs_dir)):
        path = PD.plot_arrays_path(os.path.join(runs_dir, run_id))
        if not os.path.exists(path):
            continue
        z = PD.load_plot_arrays(path)
        if "hist_step" not in z:
            continue
        best = z.get("meta", {}).get("history_best_member")
        steps = z["hist_step"]
        names = list(z["hist_scalar_names"]) + list(z["hist_param_names"])
        block = np.concatenate([z["hist_scalars"], z["hist_params"]], axis=2)  # (S,B,Q+P)
        wanted = ([int(best)] if members == "best" and best is not None
                  else list(range(block.shape[1])))
        for si, step in enumerate(steps):
            for b in wanted:
                col = block[si, b, :]
                if not np.any(np.isfinite(col)):
                    continue        # this member had no record at this step (dead or unborn)
                ids = dict(run_id=run_id, step=int(step), member=int(b),
                           is_best_member=(best is not None and int(b) == int(best)))
                for name, v in zip(names, col):
                    fv = float(v)
                    out.append(dict(ids, variable=name,
                                    value=("" if not np.isfinite(fv) else repr(fv)),
                                    value_num=("" if not np.isfinite(fv) else fv)))
    return out


# --------------------------------------------------------------------------------------
# the whole export
# --------------------------------------------------------------------------------------
def export_all(runs_root: str, out_dir: str, backend: str = "jsonl",
               history_members: str = "best") -> dict:
    """Write every tidy table under `out_dir`. Returns {filename: n_data_rows}."""
    jobs = (
        ("runs_tidy.csv", RUN_ID_COLS, tidy_runs(runs_root, backend)),
        ("target_reports_tidy.csv", REPORT_ID_COLS, tidy_target_reports(runs_root, backend)),
        ("lgen_tidy.csv", LGEN_ID_COLS, tidy_lgen(runs_root, backend)),
        ("history_tidy.csv", HISTORY_ID_COLS,
         tidy_history(runs_root, backend, members=history_members)),
    )
    return {name: write_csv(os.path.join(out_dir, name), ids, recs)
            for name, ids, recs in jobs}
