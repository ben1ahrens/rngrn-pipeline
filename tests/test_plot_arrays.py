"""test_plot_arrays.py — the plottable-array artefact, the trajectory recorder, the tidy export.

Covers the three things unit P1 added and the one thing it must not have broken:
  * plotdata: the npz round-trips; a NON-PATTERNED rollout gets no `model_field`; the
    answer-key dispersion curve appears only when the dataset has one; a grid mismatch raises.
  * history: the thinning rule keeps the endpoints; every member gets a lane; a member that
    dies leaves NaN, not a shifted index; the column set is frozen fail-loud.
  * export: the CSV is ACTUALLY TIDY — exactly one observation per row, and (identifiers,
    variable) is unique.
  * the firewall: no recovery-side module may import the scoring-side plotdata/export.
"""
import ast
import csv
import json
import pathlib

import numpy as np
import pytest

from rngrn import export as EX
from rngrn import plotdata as PD
from rngrn.eval.analysis import turing_ok
from rngrn.history import TrainingHistory, param_names
from rngrn.index import open_index
from rngrn.scoring.morphology import FEATURE_ORDER

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "rngrn"


# --------------------------------------------------------------------------------------
# fixtures: a synthetic run's worth of arrays, with no torch model anywhere
# --------------------------------------------------------------------------------------
def _turing_JD():
    """A (J, D) pair that is genuinely Turing-unstable, so sigma(k) has a real peak."""
    J = np.array([[0.5, -1.5], [1.0, -1.2]])
    D = np.array([0.02, 1.0])
    assert turing_ok(J, D)[0]
    return J, D


def _field(n=32, k=4, seed=0):
    x = np.arange(n) * 2 * np.pi * k / n
    base = np.outer(np.sin(x), np.cos(x))
    rng = np.random.default_rng(seed)
    return base + 0.01 * rng.standard_normal((n, n))


def _arrays(model_field=None, rollout_status="ok", with_true=True, kstar_obs=0.5):
    J, D = _turing_JD()
    return PD.build_plot_arrays(
        target_frame=np.stack([_field(seed=0), _field(seed=1)]), L=20.0,
        J_recovered=J, D_recovered=D, kstar_obs=kstar_obs,
        morph_vector_target=[0.1, 0.2, 0.3, 0.4],
        model_field=model_field, rollout_status=rollout_status,
        morph_vector_model=(None if model_field is None else [0.5, 0.6, 0.7, 0.8]),
        J_true=(J * 1.05 if with_true else None), D_true=(D if with_true else None),
        meta=dict(run_id="unit_test_run", git_sha="deadbee"))


# --------------------------------------------------------------------------------------
# plotdata
# --------------------------------------------------------------------------------------
def test_npz_round_trips(tmp_path):
    mf = np.stack([_field(seed=2)] * 2)
    arrays = _arrays(model_field=mf)
    path = PD.plot_arrays_path(str(tmp_path / "run"))
    n_bytes = PD.save_plot_arrays(path, arrays)
    assert n_bytes > 0
    back = PD.load_plot_arrays(path)

    assert isinstance(back["meta"], dict)
    assert back["meta"]["run_id"] == "unit_test_run"
    assert back["meta"]["schema_version"] == PD.ARRAY_SCHEMA_VERSION
    for key in ("target_frame", "model_field", "raps_k", "raps_target", "raps_model",
                "dispersion_k", "dispersion_sigma", "dispersion_sigma_true",
                "morph_vector_target", "morph_vector_model"):
        assert key in back, key
        np.testing.assert_allclose(back[key], arrays[key], rtol=0, atol=0)
    assert list(back["morph_feature_order"]) == list(FEATURE_ORDER)
    # the dtype split: fields/spectra float32, k axes and sigma float64
    assert back["target_frame"].dtype == np.float32
    assert back["raps_target"].dtype == np.float32
    assert back["dispersion_k"].dtype == np.float64
    assert back["dispersion_sigma"].dtype == np.float64


def test_unpatterned_rollout_gets_no_model_field():
    """The guard that matters: a decayed-noise field must never be saved as a pattern."""
    arrays = _arrays(model_field=None, rollout_status="unpatterned")
    for absent in ("model_field", "raps_model", "morph_vector_model"):
        assert absent not in arrays
    meta = json.loads(str(arrays["meta"].item()))
    assert meta["has_model_field"] is False
    assert meta["rollout_status"] == "unpatterned"      # the npz says WHY it is absent
    assert not np.isfinite(meta["kstar_raps_model"])


def test_answer_key_curve_absent_without_an_answer_key():
    arrays = _arrays(with_true=False)
    assert "dispersion_sigma" in arrays
    assert "dispersion_sigma_true" not in arrays
    meta = json.loads(str(arrays["meta"].item()))
    assert not np.isfinite(meta["kstar_dispersion_true"])


def test_model_field_on_a_different_grid_raises():
    with pytest.raises(ValueError, match="different"):
        _arrays(model_field=np.stack([_field(n=16)] * 2))


def test_sigma_of_agrees_with_turing_ok():
    """plotdata.sigma_of must be the SAME curve eval.analysis.turing_ok maxes over."""
    J, D = _turing_JD()
    kgrid = np.linspace(1e-3, 50, 4000)
    sig = PD.sigma_of(J, D, kgrid)
    _, info = turing_ok(J, D, kgrid=kgrid)
    assert sig.shape == kgrid.shape
    np.testing.assert_allclose(sig.max(), info["sig_max"], rtol=1e-12)
    np.testing.assert_allclose(kgrid[int(np.argmax(sig))], info["kstar"], rtol=1e-12)


def test_sigma_of_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching"):
        PD.sigma_of(np.eye(3), np.ones(2), [0.0, 1.0])


# --------------------------------------------------------------------------------------
# the cross-L records
# --------------------------------------------------------------------------------------
def _lgen_out(with_fields=True):
    def row(L, is_train, patterned=True):
        f = np.stack([_field(n=16, seed=int(L))] * 2)
        r = dict(L=float(L), L_over_L_train=L / 10.0, is_L_train=is_train, n=16,
                 dx=L / 16.0, kstar_phys=0.4, periods_per_box=L * 0.4 / (2 * np.pi),
                 pixels_per_wavelength=20.0, kstar_bin_width=0.1, kstar_bin_width_rel=0.25,
                 patterned=patterned, amplitude=0.3, blew_up=False,
                 morph_vector=[1.0, 2.0, 3.0, 4.0], morph_feature_order=list(FEATURE_ORDER),
                 morph_class="spots", morph_margin=0.5, sig_max=0.2, nsteps_run=10,
                 stopped_reason="horizon", seconds=0.1)
        if with_fields:
            r["field"] = f
        return r
    per_L = [row(10.0, True), row(20.0, False)]
    return dict(L_train=10.0, n_grid=16, grid_rule="fixed", ppw_floor=6.0, per_L=per_L,
                summary=dict(kstar_phys_cv=0.0, n_patterned=2, n_L=2,
                             morphology_class_at_L_train="spots",
                             verdict_components=dict(kstar_phys_cv=0.0),
                             L_over_L_train_values=[1.0, 2.0]))


def test_lgen_rows_are_flat_scalars():
    """House rule: index rows are FLAT SCALARS. The morph vector becomes one column each."""
    per_L, summary = PD.lgen_rows(_lgen_out(), run_id="r1", git_sha="abc1234")
    assert len(per_L) == 2
    for row in per_L + [summary]:
        assert row["run_id"] == "r1" and row["git_sha"] == "abc1234"
        for k, v in row.items():
            assert v is None or isinstance(v, (int, float, bool, str)), (k, type(v))
        assert "morph_vector" not in row and "field" not in row
    for name in FEATURE_ORDER:
        assert f"morph_{name}" in per_L[0]
    assert "verdict_components" not in summary


def test_save_lgen_fields_round_trips_and_demands_fields(tmp_path):
    out = _lgen_out()
    path = PD.lgen_fields_path(str(tmp_path / "run"))
    assert PD.save_lgen_fields(path, out, meta=dict(run_id="r1")) > 0
    back = PD.load_plot_arrays(path)
    np.testing.assert_allclose(back["L_values"], [10.0, 20.0])
    assert back["field_00"].shape == (2, 16, 16)
    assert back["field_00"].dtype == np.float32
    assert back["meta"]["run_id"] == "r1"

    with pytest.raises(ValueError, match="keep_fields"):
        PD.save_lgen_fields(str(tmp_path / "x.npz"), _lgen_out(with_fields=False))


# --------------------------------------------------------------------------------------
# the training-history recorder
# --------------------------------------------------------------------------------------
class _FakeModel:
    """Duck-typed RNGRN: only the six constrained-parameter properties are read."""
    def __init__(self, N=2, scale=1.0):
        import torch
        self.N = N
        self.KA = torch.full((N, N), 0.1 * scale)
        self.KR = torch.full((N, N), 0.2 * scale)
        self.alpha = torch.full((N, N), 0.3 * scale)
        self.delta = torch.full((N,), 0.4 * scale)
        self.beta = torch.full((N,), 0.5 * scale)
        self.D = torch.tensor([0.05, 1.0 * scale])


def _parts(total=1.0):
    return dict(total=total, sig_max=0.2, sig_max_pos=0.2, kstar_model=0.4, rel_err=0.1,
                ss_converged=True, L_kstar=0.5, L_turing=0.25, L_anchor=0.1,
                L_anticollapse=0.0, weights_used=dict(kstar=1.0, turing=0.5, anchor=2.0))


def test_should_record_keeps_the_endpoints():
    h = TrainingHistory(every=10, total_steps=37, n_members=2, N=2)
    assert h.should_record(0)               # first, unconditionally
    assert h.should_record(36)              # last training step, though 36 % 10 != 0
    assert h.should_record(37)              # the final evaluation pass
    assert h.should_record(20)
    assert not h.should_record(21)


def test_every_below_one_raises():
    with pytest.raises(ValueError, match="do not construct"):
        TrainingHistory(every=0, total_steps=10, n_members=1, N=2)


def test_serial_history_renders_dense_arrays_with_nan_for_a_dead_member():
    h = TrainingHistory(every=5, total_steps=10, n_members=3, N=2)
    for r in (0, 1):
        for step in (0, 5, 9):
            h.record_serial(step, r, _parts(total=1.0 / (step + 1)), _FakeModel(scale=step + 1))
    h.record_death(2, 3)                     # member 2 never produced a record

    a = h.to_arrays()
    assert a["hist_scalars"].shape == (3, 3, len(a["hist_scalar_names"]))
    assert a["hist_params"].shape == (3, 3, len(param_names(2)))
    assert list(a["hist_step"]) == [0, 5, 9]
    assert list(a["hist_member"]) == [0, 1, 2]
    # the dead member is a lane of NaN, NOT a missing lane that shifts the others
    assert np.all(np.isnan(a["hist_scalars"][:, 2, :]))
    assert np.all(np.isfinite(a["hist_scalars"][:, 0, :]))
    assert a["hist_death_step"][2] == 3.0 and np.isnan(a["hist_death_step"][0])
    # the weights actually in force are recorded, alongside the terms and the parameters
    names = [str(s) for s in a["hist_scalar_names"]]
    for expect in ("total", "L_turing", "w_turing", "d_ratio", "ss_converged"):
        assert expect in names, expect
    assert list(a["hist_param_names"])[:2] == ["KA[0,0]", "KA[0,1]"]
    # D-ratio of D = [0.05, 1.0*scale] at scale 1 is 20
    d = a["hist_scalars"][0, 0, names.index("d_ratio")]
    assert abs(float(d) - 20.0) < 1e-4


def test_history_column_set_is_frozen_fail_loud():
    h = TrainingHistory(every=1, total_steps=3, n_members=1, N=2)
    h.record_serial(0, 0, _parts(), _FakeModel())
    p = _parts()
    p["L_brand_new_term"] = 1.0
    with pytest.raises(ValueError, match="changed mid-run"):
        h.record_serial(1, 0, p, _FakeModel())


def test_empty_history_renders_nothing():
    assert TrainingHistory(every=1, total_steps=3, n_members=1, N=2).to_arrays() == {}


# --------------------------------------------------------------------------------------
# the tidy export
# --------------------------------------------------------------------------------------
def _seed_index(tmp_path):
    idx = open_index(str(tmp_path), "runs", "jsonl")
    idx.append(dict(run_id="r1", form="competitive", seed=0, adam_steps=100,
                    kstar_fft_rel_err=0.05, recovered_turing=True,
                    morphology_pred="spots"))
    idx.append(dict(run_id="r2", form="nc1", seed=1, adam_steps=200,
                    kstar_fft_rel_err=0.11, recovered_turing=False,
                    morphology_pred="stripes"))
    lg = open_index(str(tmp_path), "lgen_eval", "jsonl")
    per_L, _ = PD.lgen_rows(_lgen_out(), run_id="r1", git_sha="abc1234")
    for row in per_L:
        lg.append(row)


def test_runs_export_is_tidy_one_observation_per_row(tmp_path):
    _seed_index(tmp_path)
    recs = EX.tidy_runs(str(tmp_path))
    assert recs, "nothing exported"
    # ONE observation per row: every record has exactly one variable/value pair ...
    for r in recs:
        assert set(r) == set(EX.RUN_ID_COLS) | set(EX.TIDY_TAIL)
    # ... and (identifiers, variable) is UNIQUE, i.e. nothing was emitted twice
    keys = [(r["run_id"], r["variable"]) for r in recs]
    assert len(keys) == len(set(keys))
    # the measurements are present as observations, the hyperparameters as ID columns
    per_run = {rid: {v for r, v in keys if r == rid} for rid in ("r1", "r2")}
    assert "kstar_fft_rel_err" in per_run["r1"]
    assert "adam_steps" not in per_run["r1"]        # a hyperparameter is an axis, not an obs
    r1 = {r["variable"]: r for r in recs if r["run_id"] == "r1"}
    assert r1["kstar_fft_rel_err"]["value_num"] == 0.05
    assert r1["recovered_turing"]["value_num"] == 1                 # bool -> 0/1
    assert r1["morphology_pred"]["value"] == "spots"
    assert r1["morphology_pred"]["value_num"] == ""  # categorical: no numeric reading


def test_lgen_export_is_tidy_per_run_and_L(tmp_path):
    _seed_index(tmp_path)
    recs = EX.tidy_lgen(str(tmp_path))
    keys = [(r["run_id"], r["L"], r["variable"]) for r in recs]
    assert len(keys) == len(set(keys))
    assert {r["L"] for r in recs} == {10.0, 20.0}
    assert "kstar_phys" in {r["variable"] for r in recs}


def test_export_all_writes_the_csv_files(tmp_path):
    _seed_index(tmp_path)
    counts = EX.export_all(str(tmp_path), str(tmp_path / "out"))
    assert counts["runs_tidy.csv"] > 0 and counts["lgen_tidy.csv"] > 0
    with open(tmp_path / "out" / "runs_tidy.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == counts["runs_tidy.csv"]
    assert rows[0]["variable"] and rows[0]["run_id"]


def test_tidy_history_rejects_a_bad_members_choice(tmp_path):
    with pytest.raises(ValueError, match="best"):
        EX.tidy_history(str(tmp_path), members="everything")


# --------------------------------------------------------------------------------------
# the firewall, extended: plotdata/export are SCORING side
# --------------------------------------------------------------------------------------
RECOVERY_SIDE = [
    "model.py", "observables.py", "recover.py",
    "losses/terms.py", "losses/total.py", "losses/weighting.py",
    "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
    "eval/analysis.py", "eval/topology.py",
]


@pytest.mark.parametrize("relpath", RECOVERY_SIDE)
def test_recovery_side_does_not_import_the_scoring_side_writers(relpath):
    """`plotdata` accepts the ANSWER KEY's (J, D) and `export` reads the scored indexes, so
    both are scoring-side. A recovery-side import of either would be a route for truth to
    reach recovery, so it is audited statically here just as tests/test_firewall.py audits
    rd_models / AnswerKey."""
    tree = ast.parse((SRC / relpath).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [f"{node.module or ''}.{a.name}" for a in node.names] + [node.module or ""]
    joined = " ".join(names)
    for forbidden in ("plotdata", "export"):
        assert forbidden not in joined, (
            f"FIREWALL: {relpath} imports '{forbidden}', which is on the scoring side.")


def test_history_is_recovery_side_safe():
    """The recorder runs INSIDE the training loop, so it must import nothing truth-side."""
    tree = ast.parse((SRC / "history.py").read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        elif isinstance(node, ast.Import):
            names += [a.name for a in node.names]
    joined = " ".join(names)
    for forbidden in ("rd_models", "data.solver", "data.cache", "answer_key", "plotdata"):
        assert forbidden not in joined, f"history.py imports '{forbidden}'"
