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

# columns surfaced in the comparison table (priority order + identifiability).
# `kstar_rel_err_mean` is the HEADLINE wavelength column (vs the linear answer-key k*);
# `kstar_fft_rel_err_mean` immediately follows it as the clearly-labelled SECONDARY
# diagnostic (vs the FFT-measured k*). They are different references — never average or
# substitute one for the other. See validate.score_recovery.
COLUMNS = ["config_id", "source", "dataset", "N", "m", "form", "strategy", "n_seeds",
           "kstar_rel_err_mean", "kstar_fft_rel_err_mean",
           "recovered_turing_frac", "sign_match_frac_mean",
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
        turing = [1.0 if x.get("recovered_turing") else 0.0 for x in members]
        signs = [x["sign_match_frac"] for x in members if _isnum(x.get("sign_match_frac"))]
        losses = [x["loss"] for x in members if _isnum(x.get("loss"))]
        # identifiability: spread of recovered k* across seeds that all landed in-regime
        kstars_ok = [x["kstar_model"] for x in members
                     if x.get("recovered_turing") and _isnum(x.get("kstar_model"))]
        table.append(dict(
            config_id=cfg_id, source=source, dataset=dataset, N=N, m=m, form=form,
            strategy=strategy, n_seeds=len(members),
            kstar_rel_err_mean=_col_mean(members, "kstar_rel_err"),           # headline
            kstar_fft_rel_err_mean=_col_mean(members, "kstar_fft_rel_err"),   # secondary
            recovered_turing_frac=_safe(mean, turing),
            sign_match_frac_mean=_safe(mean, signs),
            loss_mean=_safe(mean, losses),
            kstar_identifiability_std=_safe(pstdev, kstars_ok) if len(kstars_ok) > 1 else float("nan"),
        ))
    return table


# --------------------------------------------------------------------------------------
# Identifiability degradation table (the two N=3 experiments)
# --------------------------------------------------------------------------------------

DEGRADATION_COLUMNS = [
    "arm", "dataset", "n_true", "n_model", "n_runs",
    "kstar_rel_err_mean",              # HEADLINE: vs the linear answer-key k*
    "kstar_fft_rel_err_mean",          # SECONDARY diagnostic: vs the FFT-measured k*
    "recovered_turing_frac",
    "subblock_sign_match_mean",        # comparison valid across ALL arms
    "sign_match_aligned_mean",         # permutation-aligned (hidden-channel arm)
    "spare_inert_frac",                # over-parameterised arm only
    "kstar_identifiability_std",
]


def degradation_table(runs_root="experiments", backend="jsonl") -> list[dict]:
    """Compare experiment ARMS: fully-observed control vs hidden-channel vs over-parameterised.

    This is the headline output of the two identifiability experiments. Grouping is by
    (arm, dataset, n_true, n_model) so an arm is never silently averaged with a different
    one. Columns that do not apply to an arm are NaN by construction, not by failure:
      * `sign_match_aligned_mean` is meaningful only where a same-size true J exists
        (the hidden-channel and fully-observed arms).
      * `spare_inert_frac` is meaningful only in the over-parameterised arm.
      * `subblock_sign_match_mean` restricts to the OBSERVED species and is therefore the
        one column comparable across every arm — read this for cross-arm degradation.

    The two wavelength columns are against DIFFERENT references and must be read as such:
    `kstar_rel_err_mean` (headline) compares the recovered dispersion-relation k* to the
    generator's linear k*; `kstar_fft_rel_err_mean` (secondary) compares it to the k*
    measured off the frame by FFT, which is quantised onto the half-integer FFT-bin grid and
    so differs from the linear k* by an offset of EITHER SIGN — a non-zero floor there is
    expected even for a perfect recovery. Never average the two together.
    """
    rows = IO.read_run_index(runs_root, backend=backend)
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("arm"), _dataset_of(r), r.get("n_true"), r.get("n_model"))].append(r)

    out = []
    for (arm, dataset, n_true, n_model), members in groups.items():
        kstars_ok = [x["kstar_model"] for x in members
                     if x.get("recovered_turing") and _isnum(x.get("kstar_model"))]
        out.append(dict(
            arm=arm, dataset=dataset, n_true=n_true, n_model=n_model, n_runs=len(members),
            kstar_rel_err_mean=_col_mean(members, "kstar_rel_err"),           # headline
            kstar_fft_rel_err_mean=_col_mean(members, "kstar_fft_rel_err"),   # secondary
            recovered_turing_frac=_safe(mean, [1.0 if x.get("recovered_turing") else 0.0
                                               for x in members]),
            subblock_sign_match_mean=_col_mean(members, "subblock_sign_match"),
            sign_match_aligned_mean=_col_mean(members, "sign_match_frac_aligned"),
            spare_inert_frac=_safe(mean, [1.0 if x.get("spare_species_inert") else 0.0
                                          for x in members
                                          if x.get("spare_species_inert") is not None]),
            kstar_identifiability_std=(_safe(pstdev, kstars_ok) if len(kstars_ok) > 1
                                       else float("nan")),
        ))
    # stable, readable ordering: controls first, then the degraded arms
    order = {"fully_observed": 0, "hidden_channel": 1, "overparameterised": 2,
             "underparameterised": 3}
    out.sort(key=lambda r: (order.get(r["arm"], 9), str(r["dataset"])))
    return out


def _col_mean(members, key):
    return _safe(mean, [x[key] for x in members if _isnum(x.get(key))])


def degradation_markdown(table: list[dict]) -> str:
    if not table:
        return "_(no runs indexed yet)_"
    hdr = ("| " + " | ".join(DEGRADATION_COLUMNS) + " |\n"
           + "|" + "---|" * len(DEGRADATION_COLUMNS) + "\n")
    body = ""
    for row in table:
        body += "| " + " | ".join(
            (f"{row.get(c):.4g}" if isinstance(row.get(c), float) else str(row.get(c)))
            for c in DEGRADATION_COLUMNS) + " |\n"
    return hdr + body


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
