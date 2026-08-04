"""benchmark.py — the cross-run harness (Stage 6).

Reads the append-only run index (experiments/runs.jsonl) and aggregates it into a
comparison table: configs x seeds x targets, with the priority-order columns and
an IDENTIFIABILITY headline metric (spread of an outcome across seeds that all
reproduce the pattern — the metric that respects a degenerate inverse problem).
Pure aggregation: it reconstructs the table from the ledger alone.
"""
from __future__ import annotations
import json
from collections import defaultdict
from statistics import mean, pstdev

import numpy as np

from .. import io as IO
from ..scoring import reproducibility as REPRO

# columns surfaced in the comparison table (priority order + identifiability).
# `kstar_fft_rel_err_mean` is the HEADLINE wavelength column (vs the FFT-measured
# answer-key k* — owner decision 2026-07-29, reversing the 2026-07-26 decision, see
# docs/STATE_OF_THE_SCIENCE.md line 499); `kstar_rel_err_mean` immediately follows it as
# the clearly-labelled SECONDARY diagnostic (vs the linear answer-key k*). They are
# different references — never average or substitute one for the other. Neither is
# interpretable without ITS OWN leak control next to it (see validate.score_recovery's
# leak-instrumentation note): an image-blind L-only predictor beats measured recovery.
# EACH COLUMN TAKES THE CONTROL COMPUTED AGAINST ITS OWN REFERENCE (D-EVID-7):
# `kstar_fft_rel_err_mean` pairs with `trivial_kstar_fft_err_mean`, `kstar_rel_err_mean`
# with `trivial_kstar_err_mean`. Pairing the headline with the linear control is the
# category error that made an unmeasured baseline read as a 14-order-of-magnitude win.
# `arm_id` replaces `config_id` here (D-EVID-13): a row of this table is an ARM aggregated
# over its seeds, and config_id is per-run because it hashes train.seed. `seeds` records
# WHICH seeds went in, because `n_seeds` alone cannot be audited.
COLUMNS = ["arm_id", "source", "dataset", "sample_key", "N", "m", "form", "strategy",
           "n_seeds", "n_unique_seeds", "seeds",
           "kstar_fft_rel_err_mean", "trivial_kstar_fft_err_mean",
           "kstar_rel_err_mean", "trivial_kstar_err_mean",
           "recovered_turing_frac", "sign_match_frac_mean",
           "loss_mean", "kstar_identifiability_std"]


def _dataset_of(row):
    """The TRUE data identity of a run, source-appropriate (not the unused `system`
    default). Falls back to legacy `system`/`dataset_id` for rows written before the
    source/dataset_label columns existed."""
    return (row.get("dataset_label") or row.get("dataset_id")
            or row.get("system") or row.get("dataset_hash"))


def _group_key(row):
    """Identity of an experiment ARM + target, so K seed replicates land in ONE group.

    THIS USED TO KEY ON `config_id` (D-EVID-13). `Config.config_id()` hashes the whole
    config INCLUDING `train.seed`, and both `optim/sweep.py` and `optim/target_report.py`
    set that per seed — so every K-seed replicate became K groups of one, `n_seeds` was
    always 1, and `kstar_identifiability_std` (the spread ACROSS seeds) was always NaN. The
    column two docs instruct the reader to weigh "as seriously as the means" could never be
    computed. `degradation_table` was unaffected because it keys differently, which is why
    the defect was invisible.

    `arm_id` is the same hash with the seed fields neutralised, so it groups replicates
    while still separating anything that genuinely differs — including `data.sample_key`,
    which never appeared in this key on its own and reached it only via `config_id`. Keying
    on the coarse tuple below WITHOUT an arm identity would pool different targets and
    different step budgets into one mean: a worse defect than the one being fixed.

    LEGACY ROWS (written before `arm_id` existed) fall back to `config_id`. They then still
    split per seed, exactly as before. That is deliberate: such a row carries no
    seed-independent identity, and guessing one risks pooling genuinely different configs.
    An honest ungrouped row beats a plausible wrong mean.
    """
    identity = row.get("arm_id") or ("legacy_config_id", row.get("config_id"))
    return (identity, row.get("source"), _dataset_of(row), row.get("sample_key"),
            row.get("N"), row.get("m"), row.get("form"), row.get("strategy"))


def build_table(runs_root="experiments", backend="jsonl") -> list[dict]:
    """Aggregate the run index into one row per (config x target), averaged over seeds."""
    rows = IO.read_run_index(runs_root, backend=backend)
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    table = []
    for key, members in groups.items():
        identity, source, dataset, sample_key, N, m, form, strategy = key
        arm_id = identity if isinstance(identity, str) else None   # None => legacy row
        turing = [1.0 if x.get("recovered_turing") else 0.0 for x in members]
        signs = [x["sign_match_frac"] for x in members if _isnum(x.get("sign_match_frac"))]
        losses = [x["loss"] for x in members if _isnum(x.get("loss"))]
        # identifiability: spread of recovered k* across seeds that all landed in-regime
        kstars_ok = [x["kstar_model"] for x in members
                     if x.get("recovered_turing") and _isnum(x.get("kstar_model"))]
        # WHICH seeds, not just how many — n_seeds alone cannot be audited, and a group of
        # 3 built from seeds [0,0,0] would be indistinguishable from one built from [0,1,2].
        seed_vals = [x["seed"] for x in members if x.get("seed") is not None]
        seeds = sorted(seed_vals)
        # DUPLICATES ARE WEIGHTED. Run records are append-only and tracked, so re-running a
        # seed legitimately lands a second row in the same group — and every mean, fraction
        # and standard deviation below then counts it twice. `n_unique_seeds` makes that
        # visible rather than leaving `n_seeds=3` to read as three independent draws. Which
        # duplicate to drop is an ops decision (latest? best? both are defensible), so this
        # reports the discrepancy instead of silently resolving it.
        n_unique = len(set(seed_vals))
        table.append(dict(
            arm_id=arm_id, source=source, dataset=dataset, sample_key=sample_key,
            N=N, m=m, form=form,
            strategy=strategy, n_seeds=len(members), n_unique_seeds=n_unique,
            seeds=json.dumps(seeds),
            kstar_fft_rel_err_mean=_col_mean(members, "kstar_fft_rel_err"),   # headline
            trivial_kstar_fft_err_mean=_col_mean(members, "trivial_kstar_fft_err"),
            kstar_rel_err_mean=_col_mean(members, "kstar_rel_err"),           # secondary
            trivial_kstar_err_mean=_col_mean(members, "trivial_kstar_err"),
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
    "kstar_fft_rel_err_mean",          # HEADLINE: vs the FFT-measured answer-key k*
    "trivial_kstar_fft_err_mean",      # its LEAK CONTROL, same reference (D-EVID-7)
    "kstar_rel_err_mean",              # SECONDARY diagnostic: vs the linear answer-key k*
    "trivial_kstar_err_mean",          # its LEAK CONTROL, same reference
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
    `kstar_fft_rel_err_mean` (headline) compares the recovered dispersion-relation k* to the
    k* measured off the frame by FFT, which is quantised onto the half-integer FFT-bin grid
    and so differs from the linear k* by an offset of EITHER SIGN — a non-zero floor there
    is expected even for a perfect recovery; `kstar_rel_err_mean` (secondary) compares it to
    the generator's linear k* instead. Never average the two together.

    READ EACH NEXT TO ITS OWN LEAK CONTROL — `kstar_fft_rel_err_mean` with
    `trivial_kstar_fft_err_mean`, `kstar_rel_err_mean` with `trivial_kstar_err_mean`. An
    image-blind predictor using only L (never the frame) scores far below measured recovery
    on the LINEAR reference and several percent on the FFT one, so reading the headline
    against the linear control overstates the win by orders of magnitude (D-EVID-7). Neither
    column is evidence of recovery on its own — see validate.score_recovery's
    leak-instrumentation note.
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
            kstar_fft_rel_err_mean=_col_mean(members, "kstar_fft_rel_err"),   # headline
            trivial_kstar_fft_err_mean=_col_mean(members, "trivial_kstar_fft_err"),
            kstar_rel_err_mean=_col_mean(members, "kstar_rel_err"),           # secondary
            trivial_kstar_err_mean=_col_mean(members, "trivial_kstar_err"),
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


# --------------------------------------------------------------------------------------
# Topology reproducibility table (unit 3 — the user's #1 metric: for ONE target, does
# the model consistently learn the SAME topology across seeds?)
# --------------------------------------------------------------------------------------
#
# The per-run pieces (`repro_sign_vector`, `repro_N`, `repro_kstar`, `repro_D_ratio`) are
# written by validate.score_recovery -> scoring.reproducibility.per_run_fields on EVERY
# run. This table is the cross-run aggregation scoring.reproducibility itself cannot do
# (it scores K seeds against each other, not one run in isolation) — grouped the same way
# as `build_table` (one row per config x target, i.e. per K-seed group), because that is
# exactly "K seeds recovered on ONE target".

REPRODUCIBILITY_COLUMNS = [
    "arm_id", "source", "dataset", "sample_key", "N", "m", "form", "strategy", "K",
    "topology_consistency", "mean_agreement", "modal_fraction",
    "kstar_spread", "Dratio_spread",
]


def _row_to_sign_matrix(row):
    """Re-hydrate one run's stored sign vector back into an (N, N) array.

    `sign_structure` is idempotent on values already in {-1, 0, 1} (see
    scoring.reproducibility.sign_structure's docstring), so feeding this straight back
    into `reproducibility_report` as a "Jacobian" reproduces the same sign structure
    that was thresholded once, at score time, in `per_run_fields`.
    """
    n = int(row["repro_N"])
    flat = json.loads(row["repro_sign_vector"])
    return np.array(flat, dtype=float).reshape(n, n)


def reproducibility_table(runs_root="experiments", backend="jsonl") -> list[dict]:
    """Cross-seed topology reproducibility, one row per (config x target) group.

    A group needs >= 2 seeds carrying `repro_sign_vector` (written by
    scoring.reproducibility.per_run_fields) to report anything; groups with fewer are
    SKIPPED (not NaN-padded into the table) because a single seed has nothing to be
    reproducible with — see scoring.reproducibility.reproducibility_report.

    STATUS: this function has NO CALLERS repo-wide and no CLI flag. It is kept working
    rather than deleted only so that "dead" never quietly becomes "broken" — the D-EVID-13
    commit widened `_group_key` to an 8-tuple, this unpack still expected 7, and the
    function raised `ValueError` on every call while that commit's message claimed it
    "inherits the fix". It did not; it was broken by it, and nothing caught that because
    there is no caller and no test. Wiring it up or deleting it is still an open decision.

    LIMITATION, unfixed on purpose: `_row_to_sign_matrix` re-thresholds an ALREADY-COLLAPSED
    sign vector, which is the D-EVID-12 no-op. This table therefore cannot honour a
    `sign_zero_rtol` other than the one used at score time. Do NOT wire it up without
    porting `target_report._sign_matrix_from_metric`'s raw-Jacobian path first.
    """
    rows = IO.read_run_index(runs_root, backend=backend)
    groups = defaultdict(list)
    for r in rows:
        if "repro_sign_vector" in r:
            groups[_group_key(r)].append(r)

    out = []
    for key, members in groups.items():
        identity, source, dataset, sample_key, N, m, form, strategy = key
        arm_id = identity if isinstance(identity, str) else None
        if len(members) < 2:
            continue
        J_list = [_row_to_sign_matrix(r) for r in members]
        kstar_list = [float(r["repro_kstar"]) for r in members]
        dratio_list = [float(r["repro_D_ratio"]) for r in members]
        rep = REPRO.reproducibility_report(J_list, kstar_list, dratio_list)
        out.append(dict(
            arm_id=arm_id, source=source, dataset=dataset, sample_key=sample_key,
            N=N, m=m, form=form,
            strategy=strategy, K=rep["K"],
            topology_consistency=rep["topology_consistency"],
            mean_agreement=rep["mean_agreement"],
            modal_fraction=rep["modal_fraction"],
            kstar_spread=rep["kstar_spread"],
            Dratio_spread=rep["Dratio_spread"],
        ))
    return out


def reproducibility_markdown(table: list[dict]) -> str:
    if not table:
        return "_(no group with >= 2 seeds carrying repro_sign_vector yet)_"
    hdr = ("| " + " | ".join(REPRODUCIBILITY_COLUMNS) + " |\n"
           + "|" + "---|" * len(REPRODUCIBILITY_COLUMNS) + "\n")
    body = ""
    for row in table:
        body += "| " + " | ".join(
            (f"{row.get(c):.4g}" if isinstance(row.get(c), float) else str(row.get(c)))
            for c in REPRODUCIBILITY_COLUMNS) + " |\n"
    return hdr + body


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
