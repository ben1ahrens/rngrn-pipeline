"""target_report.py — the target-report driver: the measurement instrument for the
whole goal (unit B5).

Every metric the project's goal needs already exists somewhere in the codebase
(scoring/reproducibility.py, eval/analysis.py, validate.py, scoring/plausibility.py), but
nothing ran them TOGETHER on one target and produced a single auditable artefact. This
module is that instrument: given one target (dataset_id + sample_key), a model form, and
K seeds, it runs recovery K times and emits ONE report combining

  REPRODUCIBILITY (the user's #1 metric)  -- topology_consistency and the modal sign
    structure across the K seeds (scoring/reproducibility.py). Needs no answer key.
  ROBUSTNESS  -- the turing_volume_* columns (eval/analysis.py), aggregated across the
    seeds that reached the Turing regime. Read against the generator-system baseline in
    docs/ROBUSTNESS_MEASUREMENT.md section 4.2 (median 0.935 at 10% noise, 1.000 at 4.8%)
    -- never against zero.
  PATTERN  -- morphology_match/morphology_distance and k*, ALWAYS next to the leak control
    computed against the SAME reference: kstar_fft_rel_err with trivial_kstar_fft_err and
    kstar_fft_bin_width_rel_fft; kstar_rel_err with trivial_kstar_err and
    kstar_fft_bin_width -- see validate.py's leak note (D-EVID-7).
  VIABILITY  -- plausibility_score and the per-parameter in-box flags
    (scoring/plausibility.py).
  the success/convergence RATE itself -- how many of K seeds produced a scored recovery,
    and how many reached the Turing regime.

DO NOT DEFINE A PASS THRESHOLD HERE. This module reports numbers and their controls; pass
conditions are pre-registered elsewhere, deliberately, so they cannot be chosen after
seeing results. There is no "success: true" field.

AUDITABILITY IS THE POINT (this project is headed for a paper). The report carries the
git sha, the config_id and full frozen config of the shared per-target recipe, every seed
requested, the run_id (or the error) of each constituent seed, and the dataset_id +
sample_key -- so any number in the report can be traced back to the run that produced it.
It is written through the EXISTING run-index machinery (index.open_index), as a new
'target_reports' table sitting next to 'runs', not as bare JSON.

TWO LAYERS, deliberately split so the aggregation is unit-testable without ever calling
fit():
  aggregate_target_report(results, ...)  -- pure. Takes a list of per-seed result dicts
    (each {"seed", "run_id", "metric", "error"}), already produced elsewhere. No torch, no
    I/O. This is what tests/test_target_report.py exercises, on SYNTHETIC per-run rows.
  run_target_report(cfg, ...)  -- orchestration. Builds the per-seed configs, runs fit()
    for each seed (multiprocessing over seeds -- they are independent, following the
    pattern in scripts/exp07_long_training.py / exp09_anchor_and_ensemble.py), then calls
    aggregate_target_report and writes the result.

SEED INDEPENDENCE is unit B1's job (restart-seed sharing across s, s+1), not this
module's: `seeds` is taken as an explicit parameter and echoed verbatim in the report so
whatever seed list a caller used is auditable, whether or not the seeds are independent.

PARALLELISM: swap point for unit B2's batched GPU path. `_fit_one_seed` is the entire
per-seed interface (one cfg + one seed in, one result dict out); `run_target_report` only
knows how to map it over seeds with a ProcessPoolExecutor or sequentially. A future
batched backend replaces the `if n_workers > 1: ... else: ...` block without touching
`aggregate_target_report` at all.

FIREWALL: this module is NOT on the recovery side (tests/test_firewall.py's RECOVERY_SIDE
list) -- it orchestrates fit(), which crosses the firewall gate itself (train.py), the
same way optim/sweep.py already does. It reads the answer-key side only indirectly, via
fit()'s own scoring call, after recovery has already finished.
"""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import yaml

from .. import io as IO
from ..config import Config, apply_overrides
from ..index import open_index
from ..scoring import reproducibility as REPRO
from ..utils import provenance

# The four robustness noise levels reported by eval.analysis.robustness_volumes.
_ROBUSTNESS_LEVELS = ("1pct", "4p8pct", "10pct", "20pct")

# The four plausibility parameters scored per-run by scoring.plausibility.plausibility_report.
_PLAUSIBILITY_PARAMS = ("alpha", "delta", "beta", "d_ratio")


# ----------------------------------------------------------------------------------------
# per-seed unit of work -- the swap point noted in the module docstring
# ----------------------------------------------------------------------------------------
def _fit_one_seed(job) -> dict:
    """Run fit() for one (cfg, seed, run_id). Never raises: a seed that fails is recorded
    as a failure (error string, no metric), not dropped -- the caller must keep it in the
    denominator. Runs in a worker process, so all imports are local and thread count is
    pinned to 1 (one core per worker; parallelism is across seeds), matching
    scripts/exp07_long_training.py / exp09_anchor_and_ensemble.py.

    `run_id` is assigned by the CALLER (run_target_report), not left to fit()'s own
    io.new_run_id default: io.new_run_id has only SECOND resolution and no PID/uniqueness
    suffix, so two seeds of the same report that finish fitting within the same wall-clock
    second collide on one run_id -- observed running this driver for real (two workers,
    200 Adam steps): both seeds silently shared one run directory, and whichever finished
    last clobbered the other's frozen_config.yaml/checkpoint/train_results.json on disk,
    even though runs.jsonl still got two distinct rows. Passing an explicit,
    seed-qualified run_id (see run_target_report) makes that collision structurally
    impossible, independent of timing.
    """
    cfg, seed, run_id, runs_root, verbose = job
    import torch
    torch.set_num_threads(1)
    from ..train import fit
    try:
        metric = fit(cfg, runs_root=runs_root, run_id=run_id, verbose=verbose)
        return {"seed": seed, "run_id": metric.get("run_id"), "metric": metric, "error": None}
    except Exception as exc:                          # fail loud IN THE RECORD, not by crashing the batch
        return {"seed": seed, "run_id": None, "metric": None,
                "error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------------------------------
# aggregation -- pure, unit-tested on synthetic per-run rows
# ----------------------------------------------------------------------------------------
def _finite(vals) -> list:
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def _mean_median(vals) -> tuple:
    fv = _finite(vals)
    if not fv:
        return float("nan"), float("nan")
    return float(np.mean(fv)), float(np.median(fv))


def _sign_matrix_from_metric(m: dict, sign_zero_rtol: float) -> np.ndarray:
    """Re-hydrate one run's Jacobian for the reproducibility aggregation, at the REQUESTED
    threshold.

    Prefers `repro_J_vector` — the RAW Jacobian — so `sign_zero_rtol` can genuinely be
    applied here. Falls back to the collapsed `repro_sign_vector` ONLY when the caller
    asks for exactly the threshold that was used at score time.

    WHY THE FALLBACK IS GUARDED (D-EVID-12). `sign_structure` is idempotent on an
    already-collapsed matrix for any rtol < 1, so re-thresholding a sign vector silently
    returns the score-time answer. This function used to do exactly that unconditionally,
    which made `aggregate_target_report(..., sign_zero_rtol=0.02)` compute at 0.05 and
    label the result 0.02 — and turned PREREGISTRATION §3.1's committed 0.02/0.05/0.10
    sensitivity sweep into three identical numbers. A row that cannot honour the requested
    threshold now says so instead of answering the wrong question quietly.
    """
    n = int(m["repro_N"])
    raw = m.get("repro_J_vector")
    if raw is not None:
        J = np.array(json.loads(raw), dtype=float).reshape(n, n)
        return REPRO.sign_structure(J, sign_zero_rtol)

    recorded = m.get("repro_sign_zero_rtol")
    if recorded is None or not np.isclose(float(recorded), float(sign_zero_rtol)):
        raise ValueError(
            f"this run carries no `repro_J_vector` (the raw Jacobian), so it cannot be "
            f"re-scored at sign_zero_rtol={sign_zero_rtol}; it was collapsed at score "
            f"time using rtol={recorded}. Re-thresholding the stored sign vector is a "
            f"no-op and would report the rtol={recorded} answer under the "
            f"rtol={sign_zero_rtol} label (docs/DECISIONS.md D-EVID-12). Either request "
            f"rtol={recorded}, or re-run the target so the raw Jacobian is recorded.")
    return np.array(json.loads(m["repro_sign_vector"]), dtype=float).reshape(n, n)


def aggregate_target_report(results, *, dataset_id, sample_key, form, seeds,
                            sign_zero_rtol: float = REPRO.DEFAULT_SIGN_ZERO_RTOL,
                            report_id=None, config_id=None, frozen_config=None,
                            git_sha=None) -> dict:
    """Combine K per-seed recovery results (on ONE target) into one auditable report.

    Parameters
    ----------
    results : sequence of dict, one per seed, SAME ORDER as `seeds`. Each element:
        {"seed": int, "run_id": str|None, "metric": dict|None, "error": str|None}
        A seed that raised has metric=None and a non-None error; it is counted in every
        denominator (n_seeds_requested) but not in n_recovered. `metric` is exactly what
        train.fit()/validate.score_recovery returns for a real seed -- this function makes
        no distinction between a real recovery and a hand-built dict with the same keys,
        which is what makes it unit-testable without fitting anything.
    seeds : the K seeds requested, in the same order as `results`. Echoed into the report
        verbatim (unit B5 does not judge seed independence -- that is unit B1's job).
    sign_zero_rtol : forwarded to scoring.reproducibility.reproducibility_report.
    report_id, config_id, frozen_config, git_sha : auditability fields supplied by the
        caller (run_target_report fills these from a real Config); left None here they are
        simply recorded as None rather than fabricated.

    Raises
    ------
    ValueError
        `seeds` is empty, `results` is not seed-aligned with `seeds`, or a succeeded
        result's metric is missing the repro_sign_vector fields that
        validate.score_recovery always emits (a broken upstream invariant, not a
        legitimate degenerate case).
    """
    seeds = list(seeds)
    K = len(seeds)
    if K == 0:
        raise ValueError("aggregate_target_report needs at least one seed; got 0")
    if len(results) != K:
        raise ValueError(f"results must have exactly one entry per seed; got "
                         f"{len(results)} results for {K} seeds")
    for i, (seed, r) in enumerate(zip(seeds, results)):
        if r.get("seed") != seed:
            raise ValueError(f"results[{i}]['seed']={r.get('seed')!r} does not match "
                             f"seeds[{i}]={seed!r} -- results must be seed-aligned "
                             f"with `seeds`")

    succeeded = [r for r in results if r.get("error") is None]
    turing_rows = [r for r in succeeded if bool(r["metric"].get("recovered_turing"))]
    n_recovered = len(succeeded)
    n_turing = len(turing_rows)

    out: dict = {
        "report_id": report_id,
        "git_sha": git_sha,
        "config_id": config_id,
        "frozen_config": frozen_config,
        "dataset_id": dataset_id,
        "sample_key": sample_key,
        "form": form,
        "seeds": json.dumps(seeds),
        "n_seeds_requested": K,
        "run_ids": json.dumps({str(seed): r.get("run_id") for seed, r in zip(seeds, results)}),
        "seed_errors": json.dumps({str(seed): r["error"] for seed, r in zip(seeds, results)
                                   if r.get("error") is not None}),
        # ---- success / convergence RATE. Denominator is ALWAYS K (n_seeds_requested): a
        # seed that raised is a failure, not a sample removed from the population.
        "n_recovered": n_recovered,
        "n_turing": n_turing,
        "recovered_frac": n_recovered / K,
        "turing_frac": n_turing / K,
    }

    out.update(_reproducibility_block(succeeded, sign_zero_rtol))
    out.update(_robustness_block(turing_rows))
    out.update(_pattern_block(succeeded))
    out.update(_viability_block(succeeded))
    out["per_seed"] = json.dumps(_per_seed_audit(seeds, results))
    return out


def _reproducibility_block(succeeded, sign_zero_rtol) -> dict:
    """REPRODUCIBILITY -- the user's #1 metric. Needs >= 2 recovered seeds carrying
    repro_sign_vector (see scoring.reproducibility.reproducibility_report). K=1 (or 0)
    successful seeds is a legitimate, DOCUMENTED degenerate case -- topology_consistency
    is NaN with `reproducibility_status="insufficient_seeds"`, never a fake 1.0 (a single
    seed trivially "agrees with itself", which is not the same claim as reproducibility).
    """
    fields = ("repro_sign_vector", "repro_N", "repro_kstar", "repro_D_ratio")
    for r in succeeded:
        missing = [f for f in fields if f not in r["metric"]]
        if missing:
            raise ValueError(
                f"seed {r['seed']}: succeeded metric is missing {missing} -- "
                f"validate.score_recovery always emits these unconditionally "
                f"(scoring.reproducibility.per_run_fields); a succeeded run without them "
                f"indicates a broken upstream invariant, not a case this function should "
                f"paper over")

    base = {
        "reproducibility_sign_zero_rtol": float(sign_zero_rtol),
        "topology_K": len(succeeded),
        "topology_consistency": float("nan"),
        "mean_agreement": float("nan"),
        "modal_fraction": float("nan"),
        "n_distinct_structures": float("nan"),
        "modal_structure": None,
        "kstar_spread": float("nan"),
        "Dratio_spread": float("nan"),
        # §3.1 sensitivity cells, defaulted so every early return emits the same schema
        **{_rtol_key(r): float("nan") for r in PREREGISTERED_SIGN_ZERO_RTOLS},
    }
    if len(succeeded) < 2:
        base["reproducibility_status"] = "insufficient_seeds"
        base["reproducibility_error"] = (
            f"only {len(succeeded)} seed(s) produced a scored recovery; reproducibility "
            f"needs >= 2 seeds to compare against each other (see "
            f"scoring.reproducibility.reproducibility_report)")
        return base

    kstar_list = [float(r["metric"]["repro_kstar"]) for r in succeeded]
    dratio_list = [float(r["metric"]["repro_D_ratio"]) for r in succeeded]
    try:
        # applies sign_zero_rtol to the RAW Jacobian, or raises rather than mislabelling
        J_list = [_sign_matrix_from_metric(r["metric"], sign_zero_rtol) for r in succeeded]
        rep = REPRO.reproducibility_report(J_list, kstar_list, dratio_list, sign_zero_rtol)
    except Exception as exc:                           # fail loud IN THE RECORD
        base["reproducibility_status"] = "error"
        base["reproducibility_error"] = f"{type(exc).__name__}: {exc}"
        return base

    base.update({
        "reproducibility_status": "ok",
        "reproducibility_error": None,
        "topology_K": rep["K"],
        "topology_consistency": rep["topology_consistency"],
        "mean_agreement": rep["mean_agreement"],
        "modal_fraction": rep["modal_fraction"],
        "n_distinct_structures": rep["n_distinct_structures"],
        "modal_structure": json.dumps(rep["modal_structure"]),
        "kstar_spread": rep["kstar_spread"],
        "Dratio_spread": rep["Dratio_spread"],
    })
    base.update(_sensitivity_cells(succeeded, kstar_list, dratio_list))
    return base


# The thresholds PREREGISTRATION.md §3.1 commits to reporting at, so "the conclusion's
# sensitivity to that choice is visible rather than hidden". Emitted from ONE report
# because the threshold is applied POST HOC — re-running recovery per threshold would burn
# K trainings to recompute a number that only needs the stored Jacobians.
PREREGISTERED_SIGN_ZERO_RTOLS = (0.02, 0.05, 0.10)


def _rtol_key(rtol: float) -> str:
    """0.02 -> 'topology_consistency_rtol_0p02'. Matches the turing_volume_4p8pct style.

    Fixed 2 decimals, deliberately: '%g' renders 0.10 as '0.1', which would name the cell
    `..._0p1` and break the visual pairing with `..._0p02` / `..._0p05` in a report meant
    to be read as a three-cell sweep. All three pre-registered thresholds are exact at 2dp.
    """
    return f"topology_consistency_rtol_{rtol:.2f}".replace(".", "p")


def _sensitivity_cells(succeeded, kstar_list, dratio_list) -> dict:
    """The §3.1 sensitivity sweep, as flat scalar columns.

    A cell is NaN when the rows cannot honour that threshold — i.e. they predate
    `repro_J_vector` and were collapsed at a different rtol. NaN is the honest answer
    there; repeating the score-time number under a different label is the defect
    D-EVID-12 records. Cells are computed independently of the headline, so one
    unavailable cell never blanks the report.
    """
    out = {}
    for rtol in PREREGISTERED_SIGN_ZERO_RTOLS:
        try:
            J_list = [_sign_matrix_from_metric(r["metric"], rtol) for r in succeeded]
            rep = REPRO.reproducibility_report(J_list, kstar_list, dratio_list, rtol)
            out[_rtol_key(rtol)] = float(rep["topology_consistency"])
        except Exception:
            # not swallowed: the reason is already explicit on `reproducibility_error`
            # for the headline threshold, and NaN here means "this threshold is not
            # answerable from these rows", which is exactly what a reader needs.
            out[_rtol_key(rtol)] = float("nan")
    return out


def _robustness_block(turing_rows) -> dict:
    """ROBUSTNESS -- turing_volume_* aggregated across seeds that REACHED THE TURING
    REGIME only (recovered_turing == True), per unit B5's brief. Read against
    docs/ROBUSTNESS_MEASUREMENT.md section 4.2's generator-system baseline, never zero."""
    out = {
        "robustness_status": "ok" if turing_rows else "no_turing_seeds",
        "robustness_n_used": len(turing_rows),
    }
    for level in _ROBUSTNESS_LEVELS:
        key = f"turing_volume_{level}"
        vals = [r["metric"].get(key) for r in turing_rows]
        mean_v, med_v = _mean_median(vals)
        out[f"{key}_mean"] = mean_v
        out[f"{key}_median"] = med_v
    return out


def _pattern_block(succeeded) -> dict:
    """PATTERN -- morphology_match / morphology_distance, and k* NEVER without the leak
    control computed against ITS OWN reference, in the same record (D-EVID-7). Aggregated
    over every seed that produced a scored recovery (not gated on Turing -- morphology and
    k* are meaningful whether or not the recovered model itself patterns).

    The headline `kstar_fft_rel_err_*` pairs with `trivial_kstar_fft_err_mean` and the
    resolution floor `kstar_fft_bin_width_rel_fft_mean`; the secondary `kstar_rel_err_*`
    pairs with `trivial_kstar_err_mean` and `kstar_fft_bin_width_mean`. Pairing the headline
    with the linear control is the defect this split fixes: on legacy samples the L-only
    predictor is exact against the linear reference and several percent against the FFT one.
    """
    kstar_fft_mean, kstar_fft_med = _mean_median(
        [r["metric"].get("kstar_fft_rel_err") for r in succeeded])
    kstar_lin_mean, kstar_lin_med = _mean_median(
        [r["metric"].get("kstar_rel_err") for r in succeeded])
    trivial_mean, _ = _mean_median([r["metric"].get("trivial_kstar_err") for r in succeeded])
    binw_mean, _ = _mean_median([r["metric"].get("kstar_fft_bin_width") for r in succeeded])
    trivial_fft_mean, _ = _mean_median(
        [r["metric"].get("trivial_kstar_fft_err") for r in succeeded])
    binw_fft_mean, _ = _mean_median(
        [r["metric"].get("kstar_fft_bin_width_rel_fft") for r in succeeded])

    compared = [r for r in succeeded if r["metric"].get("morphology_scored") == "compared"]
    match_frac = (float(np.mean([bool(r["metric"].get("morphology_match")) for r in compared]))
                 if compared else float("nan"))

    return {
        "pattern_n_used": len(succeeded),
        "kstar_fft_rel_err_mean": kstar_fft_mean,
        "kstar_fft_rel_err_median": kstar_fft_med,
        "kstar_rel_err_mean": kstar_lin_mean,
        "kstar_rel_err_median": kstar_lin_med,
        # LEAK CONTROLS -- one per reference, always alongside the column it controls. See
        # validate.score_recovery's leak-instrumentation note: neither k* column above is
        # evidence of recovery without ITS OWN control read next to it.
        "trivial_kstar_fft_err_mean": trivial_fft_mean,        # controls kstar_fft_rel_err
        "kstar_fft_bin_width_rel_fft_mean": binw_fft_mean,     # its resolution floor
        "trivial_kstar_err_mean": trivial_mean,                # controls kstar_rel_err
        "kstar_fft_bin_width_mean": binw_mean,
        "morphology_n_compared": len(compared),
        "morphology_match_frac": match_frac,
    }


def _viability_block(succeeded) -> dict:
    """VIABILITY -- plausibility_score and the per-parameter in-box flags
    (scoring.plausibility.plausibility_report), aggregated over every scored seed."""
    score_mean, _ = _mean_median([r["metric"].get("plausibility_score") for r in succeeded])
    out = {"plausibility_n_used": len(succeeded), "plausibility_score_mean": score_mean}
    for name in _PLAUSIBILITY_PARAMS:
        key = f"plausibility_{name}_in_box"
        verdicts = [r["metric"].get(key) for r in succeeded]
        verdicts = [v for v in verdicts if v is not None]
        out[f"{key}_frac"] = float(np.mean([bool(v) for v in verdicts])) if verdicts else float("nan")
        out[f"{key}_n_scored"] = len(verdicts)
    return out


def _per_seed_audit(seeds, results) -> list:
    """One row per requested seed -- the full per-seed trace for tracing any aggregate
    number in this report back to the run (or the error) that produced it."""
    rows = []
    for seed, r in zip(seeds, results):
        if r.get("error") is not None:
            rows.append({"seed": seed, "run_id": None, "status": "failed",
                        "error": r["error"]})
            continue
        m = r["metric"]
        rows.append({
            "seed": seed, "run_id": r.get("run_id"), "status": "ok",
            "recovered_turing": bool(m.get("recovered_turing")),
            "kstar_model": m.get("kstar_model"),
            "kstar_fft_rel_err": m.get("kstar_fft_rel_err"),
            "kstar_rel_err": m.get("kstar_rel_err"),
            "trivial_kstar_fft_err": m.get("trivial_kstar_fft_err"),
            "kstar_fft_bin_width_rel_fft": m.get("kstar_fft_bin_width_rel_fft"),
            "trivial_kstar_err": m.get("trivial_kstar_err"),
            "kstar_fft_bin_width": m.get("kstar_fft_bin_width"),
            "morphology_scored": m.get("morphology_scored"),
            "morphology_match": m.get("morphology_match"),
            "morphology_distance": m.get("morphology_distance"),
            "plausibility_score": m.get("plausibility_score"),
        })
    return rows


# ----------------------------------------------------------------------------------------
# orchestration -- calls fit(), not covered by unit tests (see module docstring)
# ----------------------------------------------------------------------------------------
def run_target_report(cfg: Config, dataset_id: str, sample_key: str, form: str, seeds,
                      *, runs_root: str = "experiments", n_workers: int = 1,
                      verbose: bool = False,
                      sign_zero_rtol: float = REPRO.DEFAULT_SIGN_ZERO_RTOL) -> dict:
    """Run recovery K times (one per seed in `seeds`) on ONE target and emit ONE
    auditable report, written into the run-index machinery as a 'target_reports' table.

    `cfg` supplies everything ELSE the recovery needs (model N/m/observed_idx, loss,
    train, solver, tracking) -- dataset_id/sample_key/form are overridden onto it
    explicitly so the target and form are never accidentally left at whatever the config
    file happened to say. `data.source` is forced to 'registry' for the same reason: this
    driver runs against a registered dataset, not a freshly-generated reference system.

    PARALLELISM: seeds are independent, so this uses a ProcessPoolExecutor over seeds
    (n_workers > 1) exactly like scripts/exp07_long_training.py / exp09; n_workers=1 (the
    default) runs sequentially in-process, which is what a test importing this function
    without wanting subprocess machinery should use.
    """
    seeds = list(seeds)
    if len(seeds) == 0:
        raise ValueError("run_target_report needs at least one seed; got 0")

    target_cfg = apply_overrides(deepcopy(cfg), [
        f"data.source={json.dumps('registry')}",
        f"data.dataset_id={json.dumps(dataset_id)}",
        f"data.sample_key={json.dumps(sample_key)}",
        f"model.form={json.dumps(form)}",
    ])
    config_id = target_cfg.config_id()
    frozen_config = yaml.safe_dump(asdict(target_cfg), sort_keys=False)
    git_sha = provenance()["git_revision"]

    # Seed-qualified run_ids, assigned HERE rather than left to fit()'s own
    # io.new_run_id default -- see _fit_one_seed's docstring: that default has only
    # second resolution and no uniqueness suffix, so parallel seeds finishing in the same
    # wall-clock second collide and clobber each other's on-disk provenance. The seed
    # integer makes collision impossible regardless of timing.
    report_base_id = IO.new_run_id(target_cfg.tracking.run_name)
    jobs = [(apply_overrides(deepcopy(target_cfg), [f"train.seed={seed}"]),
            seed, f"{report_base_id}_seed{seed}", runs_root, verbose) for seed in seeds]

    if n_workers and n_workers > 1:
        # 'spawn', not the Linux default 'fork': a forked worker that touches CUDA after
        # the parent process has already initialised a CUDA context crashes with
        # "Cannot re-initialize CUDA in forked subprocess" (observed running this driver
        # with the sandbox disabled, where CUDA is visible -- see worker brief env fact
        # 4). 'spawn' re-imports cleanly in each worker and has no such failure mode; the
        # extra per-worker startup cost is negligible next to one seed's training time.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            results = list(ex.map(_fit_one_seed, jobs))
    else:
        results = [_fit_one_seed(j) for j in jobs]

    report = aggregate_target_report(
        results, dataset_id=dataset_id, sample_key=sample_key, form=form, seeds=seeds,
        sign_zero_rtol=sign_zero_rtol, report_id=f"target_report_{report_base_id}",
        config_id=config_id, frozen_config=frozen_config, git_sha=git_sha)

    open_index(runs_root, "target_reports", target_cfg.tracking.index_backend).append(report)
    return report
