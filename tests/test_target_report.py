"""test_target_report.py — unit tests for rngrn.optim.target_report.aggregate_target_report.

These test the AGGREGATOR on SYNTHETIC per-seed rows (dicts with exactly the keys
train.fit()/validate.score_recovery would put in `metric`) -- no torch, no fitting, no
config, no real recovery. A failure here means the aggregation logic is wrong, never
that the science is wrong. Same spirit as tests/test_reproducibility_scoring.py.

`run_target_report` (the orchestration layer that actually calls fit()) is intentionally
NOT exercised here -- unit B5's brief is explicit that aggregation is covered on
synthetic rows, and calling fit() is training, not a unit test.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rngrn.optim.target_report import aggregate_target_report
from rngrn.scoring.reproducibility import DEFAULT_SIGN_ZERO_RTOL as REPRO_DEFAULT_RTOL
import rngrn.cli as cli


# --------------------------------------------------------------------------------------
# synthetic per-seed metric builder -- mirrors what train.fit() actually returns
# --------------------------------------------------------------------------------------
def _sign_vector(J):
    """Flatten a small hand-picked sign matrix the way per_run_fields would store it."""
    return json.dumps([int(v) for row in J for v in row])


_SIGN_A = [[-1, -1, 1], [1, -1, -1], [-1, 1, -1]]   # one topology
_SIGN_B = [[-1, 1, -1], [-1, -1, 1], [1, -1, -1]]   # a DIFFERENT topology


def _metric(*, sign=_SIGN_A, kstar=1.0, dratio=10.0, turing=True,
           morphology_scored="compared", morphology_match=True,
           plaus_score=0.75, alpha_ok=True, delta_ok=True, beta_ok=True, dratio_ok=True,
           tv=0.9) -> dict:
    m = {
        "repro_sign_vector": _sign_vector(sign), "repro_N": 3,
        # per_run_fields emits this on EVERY run; the fixture omitted it, which made these
        # synthetic rows less faithful than the real ones (D-EVID-12).
        "repro_sign_zero_rtol": REPRO_DEFAULT_RTOL,
        "repro_kstar": kstar, "repro_D_ratio": dratio,
        "recovered_turing": turing,
        "turing_volume_1pct": tv, "turing_volume_4p8pct": tv,
        "turing_volume_10pct": tv, "turing_volume_20pct": tv,
        "kstar_model": kstar, "kstar_fft_rel_err": 0.05, "kstar_rel_err": 0.03,
        "trivial_kstar_err": 0.45, "kstar_fft_bin_width": 0.17,
        "morphology_scored": morphology_scored, "morphology_match": morphology_match,
        "morphology_distance": 0.1,
        "plausibility_score": plaus_score,
        "plausibility_alpha_in_box": alpha_ok, "plausibility_delta_in_box": delta_ok,
        "plausibility_beta_in_box": beta_ok, "plausibility_d_ratio_in_box": dratio_ok,
    }
    return m


def _result(seed, run_id="r0", metric=None, error=None):
    return {"seed": seed, "run_id": run_id if error is None else None,
            "metric": metric, "error": error}


# --------------------------------------------------------------------------------------
# THE PRE-REGISTERED rtol SWEEP MUST ACTUALLY SWEEP (D-EVID-12)
#
# PREREGISTRATION §3.1: "Results will be reported at 0.02 / 0.05 / 0.10 so the
# conclusion's sensitivity to that choice is visible rather than hidden." The report used
# to re-threshold an ALREADY-COLLAPSED sign vector, which is a no-op for any rtol < 1 —
# so it returned three identical numbers under three different labels.
# --------------------------------------------------------------------------------------
_J_WITH_WEAK_EDGE = [[1.0, -1.0, 0.07], [0.5, -1.0, 0.0], [0.0, 0.5, -1.0]]
_J_NO_EDGE = [[1.0, -1.0, 0.00], [0.5, -1.0, 0.0], [0.0, 0.5, -1.0]]


def _metric_with_raw_J(J, **kw):
    """A metric carrying the RAW Jacobian, as validate.score_recovery now emits."""
    m = _metric(**kw)
    m["repro_J_vector"] = json.dumps([float(v) for row in J for v in row])
    m["repro_sign_zero_rtol"] = 0.05                       # what score time used
    return m


def test_rtol_sweep_produces_different_numbers_through_the_report():
    """End-to-end: the three pre-registered cells must not collapse to one number."""
    results = [_result(0, "r0", _metric_with_raw_J(_J_WITH_WEAK_EDGE)),
               _result(1, "r1", _metric_with_raw_J(_J_NO_EDGE))]
    got = {}
    for rtol in (0.02, 0.05, 0.10):
        rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0",
                                      form="nc1", seeds=[0, 1], sign_zero_rtol=rtol)
        assert rep["reproducibility_status"] == "ok", rep.get("reproducibility_error")
        # the label must match what was actually computed
        assert rep["reproducibility_sign_zero_rtol"] == pytest.approx(rtol)
        got[rtol] = rep["topology_consistency"]

    assert got[0.02] == pytest.approx(0.5)
    assert got[0.05] == pytest.approx(0.5)
    assert got[0.10] == pytest.approx(1.0), "the 0.10 cell must differ"
    assert len(set(got.values())) > 1, "the sweep measured nothing"


def test_report_emits_the_three_preregistered_sensitivity_cells():
    """PREREGISTRATION §3.1 commits to reporting at 0.02 / 0.05 / 0.10. Emitting all
    three from ONE report is what makes that possible without re-training three times —
    the threshold is purely post-hoc, so re-running recovery to change it would be waste.
    """
    results = [_result(0, "r0", _metric_with_raw_J(_J_WITH_WEAK_EDGE)),
               _result(1, "r1", _metric_with_raw_J(_J_NO_EDGE))]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=[0, 1])

    assert rep["topology_consistency_rtol_0p02"] == pytest.approx(0.5)
    assert rep["topology_consistency_rtol_0p05"] == pytest.approx(0.5)
    assert rep["topology_consistency_rtol_0p10"] == pytest.approx(1.0)
    # the headline stays the default-rtol number, unchanged
    assert rep["topology_consistency"] == pytest.approx(
        rep["topology_consistency_rtol_0p05"])
    # every cell is a flat scalar (run-index rows are flat scalars)
    for k in ("topology_consistency_rtol_0p02", "topology_consistency_rtol_0p05",
              "topology_consistency_rtol_0p10"):
        assert isinstance(rep[k], float)


def test_sensitivity_cells_are_nan_when_the_raw_J_is_unavailable():
    """A legacy row cannot support the sweep. NaN says so; a repeated number would lie."""
    results = [_result(0, "r0", _metric()), _result(1, "r1", _metric(sign=_SIGN_B))]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=[0, 1])
    assert rep["reproducibility_status"] == "ok"          # the headline still works
    assert math.isnan(rep["topology_consistency_rtol_0p02"])
    assert math.isnan(rep["topology_consistency_rtol_0p10"])
    # the cell that MATCHES the recorded rtol is computable, so it is not NaN
    assert rep["topology_consistency_rtol_0p05"] == pytest.approx(
        rep["topology_consistency"])


def test_legacy_row_without_raw_J_is_accepted_at_its_own_recorded_rtol():
    """Rows recorded before repro_J_vector existed stay usable — at their own rtol."""
    results = [_result(0, "r0", _metric()), _result(1, "r1", _metric(sign=_SIGN_B))]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=[0, 1],
                                  sign_zero_rtol=REPRO_DEFAULT_RTOL)
    assert rep["reproducibility_status"] == "ok", rep.get("reproducibility_error")
    assert rep["reproducibility_sign_zero_rtol"] == pytest.approx(REPRO_DEFAULT_RTOL)


def test_legacy_row_REFUSES_a_different_rtol_rather_than_mislabelling_it():
    """THE DEFECT, made impossible.

    A row that only carries the collapsed sign vector cannot honour a different rtol. It
    must say so, not compute at the recorded threshold and label the answer with the
    requested one.
    """
    results = [_result(0, "r0", _metric()), _result(1, "r1", _metric(sign=_SIGN_B))]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=[0, 1], sign_zero_rtol=0.02)
    assert rep["reproducibility_status"] == "error"
    err = rep["reproducibility_error"]
    assert "repro_J_vector" in err and "0.02" in err, err
    assert np.isnan(rep["topology_consistency"])


# --------------------------------------------------------------------------------------
# 1. identity / auditability fields are echoed verbatim, never fabricated
# --------------------------------------------------------------------------------------
def test_identity_fields_echoed_verbatim():
    seeds = [0, 1]
    results = [_result(0, "run_a", _metric()), _result(1, "run_b", _metric())]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds, report_id="rep_1", config_id="cfg_abc",
                                  frozen_config="frozen: yaml", git_sha="deadbee")
    assert rep["dataset_id"] == "ds" and rep["sample_key"] == "s0" and rep["form"] == "nc1"
    assert rep["report_id"] == "rep_1" and rep["config_id"] == "cfg_abc"
    assert rep["frozen_config"] == "frozen: yaml" and rep["git_sha"] == "deadbee"
    assert json.loads(rep["seeds"]) == [0, 1]
    assert json.loads(rep["run_ids"]) == {"0": "run_a", "1": "run_b"}
    assert json.loads(rep["seed_errors"]) == {}


# --------------------------------------------------------------------------------------
# 2. a seed that raised is a FAILURE, counted in the denominator, never dropped
# --------------------------------------------------------------------------------------
def test_raised_seed_counts_as_failure_not_dropped_from_denominator():
    seeds = [0, 1, 2]
    results = [
        _result(0, "run_0", _metric(turing=True)),
        _result(1, "run_1", _metric(turing=False)),
        _result(2, error="RuntimeError: NaN loss"),
    ]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="competitive",
                                  seeds=seeds)
    assert rep["n_seeds_requested"] == 3
    assert rep["n_recovered"] == 2                 # only 0 and 1 scored
    assert rep["n_turing"] == 1                    # only seed 0 reached Turing
    assert rep["recovered_frac"] == pytest.approx(2 / 3)   # denominator is 3, NOT 2
    assert rep["turing_frac"] == pytest.approx(1 / 3)
    assert json.loads(rep["seed_errors"]) == {"2": "RuntimeError: NaN loss"}
    per_seed = json.loads(rep["per_seed"])
    assert len(per_seed) == 3
    assert per_seed[2]["status"] == "failed"
    assert per_seed[2]["error"] == "RuntimeError: NaN loss"


# --------------------------------------------------------------------------------------
# 3. K=1 reports a DOCUMENTED NaN, never a fake consistency of 1.0
# --------------------------------------------------------------------------------------
def test_k_equals_1_documents_nan_not_fake_full_consistency():
    seeds = [0]
    results = [_result(0, "run_0", _metric())]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["reproducibility_status"] == "insufficient_seeds"
    assert math.isnan(rep["topology_consistency"])
    assert math.isnan(rep["mean_agreement"])
    assert "reproducibility_error" in rep and rep["reproducibility_error"]
    # the convergence rate itself is still perfectly well defined at K=1
    assert rep["n_recovered"] == 1 and rep["recovered_frac"] == 1.0


def test_zero_recovered_seeds_also_documents_nan():
    seeds = [0, 1]
    results = [_result(0, error="Boom"), _result(1, error="Boom2")]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["n_recovered"] == 0
    assert rep["reproducibility_status"] == "insufficient_seeds"
    assert math.isnan(rep["topology_consistency"])
    assert rep["robustness_status"] == "no_turing_seeds"
    assert math.isnan(rep["turing_volume_10pct_mean"])


# --------------------------------------------------------------------------------------
# 4. reproducibility: identical topologies -> consistency 1.0; a dissenter drags it down
# --------------------------------------------------------------------------------------
def test_reproducibility_full_agreement():
    seeds = [0, 1, 2]
    results = [_result(s, f"run_{s}", _metric(sign=_SIGN_A, kstar=1.0 + 0.01 * s))
              for s in seeds]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["reproducibility_status"] == "ok"
    assert rep["topology_consistency"] == pytest.approx(1.0)
    assert rep["mean_agreement"] == pytest.approx(1.0)
    assert rep["topology_K"] == 3


def test_reproducibility_one_dissenting_seed_lowers_modal_fraction():
    seeds = [0, 1, 2]
    results = [
        _result(0, "run_0", _metric(sign=_SIGN_A)),
        _result(1, "run_1", _metric(sign=_SIGN_A)),
        _result(2, "run_2", _metric(sign=_SIGN_B)),
    ]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["reproducibility_status"] == "ok"
    assert rep["topology_consistency"] == pytest.approx(2 / 3)
    assert rep["n_distinct_structures"] == 2
    assert rep["modal_structure"] is not None


def test_reproducibility_missing_repro_fields_raises_loud():
    seeds = [0, 1]
    bad_metric = {"recovered_turing": True}   # no repro_sign_vector etc.
    results = [_result(0, "run_0", bad_metric), _result(1, "run_1", _metric())]
    with pytest.raises(ValueError, match="repro_sign_vector"):
        aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                seeds=seeds)


# --------------------------------------------------------------------------------------
# 5. robustness is aggregated ONLY over seeds that reached the Turing regime
# --------------------------------------------------------------------------------------
def test_robustness_excludes_non_turing_seeds():
    seeds = [0, 1, 2]
    results = [
        _result(0, "run_0", _metric(turing=True, tv=0.8)),
        _result(1, "run_1", _metric(turing=True, tv=0.6)),
        _result(2, "run_2", _metric(turing=False, tv=0.99)),   # not Turing -> excluded
    ]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["robustness_status"] == "ok"
    assert rep["robustness_n_used"] == 2
    assert rep["turing_volume_10pct_mean"] == pytest.approx(0.7)
    assert rep["turing_volume_10pct_median"] == pytest.approx(0.7)


# --------------------------------------------------------------------------------------
# 6. pattern: k* headline numbers are never emitted without their leak controls
# --------------------------------------------------------------------------------------
def test_pattern_block_carries_leak_controls_alongside_kstar():
    seeds = [0, 1]
    results = [_result(0, "run_0", _metric()), _result(1, "run_1", _metric())]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    for key in ("kstar_fft_rel_err_mean", "kstar_rel_err_mean",
               "trivial_kstar_err_mean", "kstar_fft_bin_width_mean"):
        assert key in rep and not math.isnan(rep[key])
    assert rep["morphology_n_compared"] == 2
    assert rep["morphology_match_frac"] == pytest.approx(1.0)


def test_pattern_block_morphology_not_scored_gives_nan_not_zero():
    seeds = [0]
    results = [_result(0, "run_0", _metric(morphology_scored="target_only",
                                           morphology_match=None))]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["morphology_n_compared"] == 0
    assert math.isnan(rep["morphology_match_frac"])


# --------------------------------------------------------------------------------------
# 7. viability: plausibility aggregation, including a partially-UNCITED parameter
# --------------------------------------------------------------------------------------
def test_viability_block_aggregates_plausibility():
    seeds = [0, 1]
    results = [
        _result(0, "run_0", _metric(plaus_score=1.0, alpha_ok=True, beta_ok=None)),
        _result(1, "run_1", _metric(plaus_score=0.5, alpha_ok=False, beta_ok=None)),
    ]
    rep = aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                  seeds=seeds)
    assert rep["plausibility_score_mean"] == pytest.approx(0.75)
    assert rep["plausibility_alpha_in_box_frac"] == pytest.approx(0.5)
    assert rep["plausibility_alpha_in_box_n_scored"] == 2
    # beta is UNCITED for both seeds -> nothing to score, NaN not 0
    assert rep["plausibility_beta_in_box_n_scored"] == 0
    assert math.isnan(rep["plausibility_beta_in_box_frac"])


# --------------------------------------------------------------------------------------
# 8. input validation: seed alignment
# --------------------------------------------------------------------------------------
def test_empty_seeds_raises():
    with pytest.raises(ValueError, match="at least one seed"):
        aggregate_target_report([], dataset_id="ds", sample_key="s0", form="nc1", seeds=[])


def test_mismatched_results_length_raises():
    with pytest.raises(ValueError, match="one entry per seed"):
        aggregate_target_report([_result(0, "r0", _metric())], dataset_id="ds",
                                sample_key="s0", form="nc1", seeds=[0, 1])


def test_seed_order_mismatch_raises():
    results = [_result(1, "r1", _metric()), _result(0, "r0", _metric())]
    with pytest.raises(ValueError, match="seed-aligned"):
        aggregate_target_report(results, dataset_id="ds", sample_key="s0", form="nc1",
                                seeds=[0, 1])


# --------------------------------------------------------------------------------------
# 9. CLI wiring — no fitting, just checking the subcommand parses correctly
# --------------------------------------------------------------------------------------
def test_cli_target_report_subcommand_parses():
    p = cli.build_parser()
    args = p.parse_args(["target-report", "--config", "configs/m3_registry.yaml",
                         "--dataset-id", "three_gene_val", "--sample-key", "sample_0000",
                         "--form", "nc1", "--seeds", "0", "1", "2"])
    assert args.dataset_id == "three_gene_val"
    assert args.sample_key == "sample_0000"
    assert args.form == "nc1"
    assert args.seeds == [0, 1, 2]
    assert args.workers == 1
    assert args.func is cli.cmd_target_report
