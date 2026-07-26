"""test_gate_contract.py — the gate reads per-sample geometry and BOTH wavenumbers.

These pin the contract fixed on this branch. The bug they guard against was silent, not
loud: `gate.from_registry` used to take `L` from the caller's config (every file-backed
config hardcoded `L: 100.0`, while real samples span L = 18 to 208) and to read the
answer-key wavenumber from an attribute named `kstar`, which no payload carries — so
`AnswerKey.kstar` was always None and `kstar_rel_err`, the headline metric, was always NaN.
Nothing failed; the number just wasn't real.

Two tiers here, deliberately:
  * SYNTHETIC fixtures (tmp_path) — the contract itself: file wins over caller, missing
    attributes raise. These always run.
  * REAL registered datasets — the measured scaling fact (linear k* at exactly 6.00 FFT
    bins). Skipped when the local datasets are absent, since they are gitignored.
"""
import numpy as np
import h5py
import pytest


TWO_PI = 2.0 * np.pi

# Datasets whose linear k* is pinned to exactly 6.00 FFT bins. Scope matters: this holds
# for all 127 samples of the three_gene family but NOT for the classical families — see
# test_classical_sets_are_not_all_at_six_bins.
SIX_BIN_DATASETS = ["three_gene_train", "three_gene_val", "three_gene_test"]


def _write_sample(g, N=3, H=16, L=100.0, k_star=None, k_star_fft=None,
                  omit=(), rng=None):
    """One sample group in the real payload layout. `omit` drops attributes deliberately."""
    rng = rng or np.random.default_rng(0)
    g.create_dataset("final_frame", data=rng.standard_normal((N, H, H)).astype("float32"))
    g.create_dataset("jacobian", data=rng.standard_normal((N, N)))
    g.create_dataset("x_star", data=rng.standard_normal(N))
    g.create_dataset("D", data=np.array([1.0, 40.0, 20.0])[:N])
    attrs = {"L": float(L),
             "k_star": float(k_star if k_star is not None else 6.0 * TWO_PI / L),
             "k_star_fft": float(k_star_fft if k_star_fft is not None
                                 else 1.08 * 6.0 * TWO_PI / L)}
    for k, v in attrs.items():
        if k not in omit:
            g.attrs[k] = v


def _registered(tmp_path, dataset_id="gate_v1", L=57.0, omit=(), **kw):
    """Build + register a one-sample payload; return (datasets_root, dataset_id)."""
    from rngrn.data import registry as reg
    droot = tmp_path / "datasets"
    dsdir = droot / dataset_id
    dsdir.mkdir(parents=True)
    with h5py.File(dsdir / "payload.h5", "w") as f:
        _write_sample(f.create_group("sample_0000"), L=L, omit=omit, **kw)
    reg.scan(str(droot), backend="jsonl")
    return str(droot), dataset_id


# --------------------------------------------------------------------------------------
# the gate returns the SAMPLE's L, never the caller's
# --------------------------------------------------------------------------------------
def test_gate_returns_sample_L_not_caller_L(tmp_path):
    """The original bug: the config's L: 100.0 reached recovery instead of the file's."""
    from rngrn.data import gate
    droot, did = _registered(tmp_path, L=57.0)
    ri, _ = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2])
    assert ri.L == pytest.approx(57.0)
    assert ri.L != pytest.approx(100.0), "caller/config default leaked into RecoveryInput"


def test_disagreeing_caller_L_warns_and_file_wins(tmp_path):
    """A caller L is a cross-check, not an override: warn loudly, then prefer the file."""
    from rngrn.data import gate
    droot, did = _registered(tmp_path, L=57.0)
    with pytest.warns(UserWarning, match="PREFERRING THE FILE"):
        ri, _ = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                   L=100.0)
    assert ri.L == pytest.approx(57.0)


def test_agreeing_caller_L_does_not_warn(tmp_path):
    """An L that matches the file is a passing cross-check, not a complaint."""
    import warnings
    from rngrn.data import gate
    droot, did = _registered(tmp_path, L=57.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ri, _ = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                   L=57.0)
    assert ri.L == pytest.approx(57.0)


# --------------------------------------------------------------------------------------
# the answer key carries a POPULATED, FINITE k* — and the FFT one alongside it
# --------------------------------------------------------------------------------------
def test_answer_key_kstar_is_populated_and_finite(tmp_path):
    """`AnswerKey.kstar` was permanently None because the gate read the wrong attr name."""
    from rngrn.data import gate
    droot, did = _registered(tmp_path, L=57.0)
    _, ak = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2])
    assert ak.kstar is not None
    assert np.isfinite(ak.kstar) and ak.kstar > 0
    assert ak.kstar == pytest.approx(6.0 * TWO_PI / 57.0)


def test_answer_key_carries_both_wavenumbers_distinctly(tmp_path):
    from rngrn.data import gate
    droot, did = _registered(tmp_path, L=57.0)
    _, ak = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2])
    assert np.isfinite(ak.kstar_fft) and ak.kstar_fft > 0
    # they are different references and must not be conflated
    assert ak.kstar_fft != pytest.approx(ak.kstar)


def test_answer_key_is_not_reachable_from_recovery_input(tmp_path):
    """Firewall spot-check at the new fields: neither k* may ride along on RecoveryInput."""
    from rngrn.data import gate
    droot, did = _registered(tmp_path, L=57.0)
    ri, _ = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2])
    for bad in ("kstar", "kstar_fft", "k_star", "J", "x_star"):
        assert not hasattr(ri, bad), bad


# --------------------------------------------------------------------------------------
# a sample missing L or k_star RAISES — house style, no silent default
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["L", "k_star"])
def test_missing_required_attr_raises(tmp_path, missing):
    from rngrn.data import gate
    droot, did = _registered(tmp_path, dataset_id=f"no_{missing}", omit=(missing,))
    with pytest.raises(KeyError, match=missing):
        gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2])


def test_missing_L_raises_even_when_caller_supplies_one(tmp_path):
    """The caller's L must not become a fallback for a file that has none."""
    from rngrn.data import gate
    droot, did = _registered(tmp_path, dataset_id="no_L2", omit=("L",))
    with pytest.raises(KeyError, match="L"):
        gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2], L=100.0)


def test_missing_kstar_fft_is_tolerated_as_none(tmp_path):
    """The FFT number is a diagnostic, not the headline — absent is legal, not fatal."""
    from rngrn.data import gate
    droot, did = _registered(tmp_path, dataset_id="no_fft", omit=("k_star_fft",))
    _, ak = gate.from_registry(droot, did, "sample_0000", N=3, observed_idx=[0, 1, 2])
    assert ak.kstar_fft is None
    assert np.isfinite(ak.kstar), "the headline k* must still be present"


# --------------------------------------------------------------------------------------
# both k* errors reach a scored run, the run index, and the degradation table
# --------------------------------------------------------------------------------------
def test_both_kstar_errors_are_scored(tmp_path):
    """score_recovery emits the headline AND the secondary error, both finite."""
    from rngrn.validate import score_recovery
    from test_experiment_arms import _Key, _Result

    key = _Key(np.ones((3, 3)), n_true=3, kstar=0.5)
    key.kstar_fft = 0.54
    out = score_recovery(_Result(N=3), key, observed_idx=[0, 1])
    for k in ("kstar_rel_err", "kstar_fft_rel_err", "kstar_true", "kstar_fft_true"):
        assert k in out, k
        assert np.isfinite(out[k]), (k, out[k])
    # different references => different errors
    assert out["kstar_rel_err"] != pytest.approx(out["kstar_fft_rel_err"])
    assert out["kstar_true"] == pytest.approx(0.5)
    assert out["kstar_fft_true"] == pytest.approx(0.54)


def test_absent_kstar_fft_gives_nan_secondary_but_real_headline():
    """A source with no FFT measurement must not poison the headline metric."""
    from rngrn.validate import score_recovery
    from test_experiment_arms import _Key, _Result

    key = _Key(np.ones((3, 3)), n_true=3, kstar=0.5)
    key.kstar_fft = None
    out = score_recovery(_Result(N=3), key, observed_idx=[0, 1])
    assert np.isfinite(out["kstar_rel_err"])
    assert np.isnan(out["kstar_fft_rel_err"])


def test_both_kstar_errors_reach_run_index_and_degradation_table(tmp_path):
    """End-to-end on a REAL registered sample: metric -> runs.jsonl -> both tables.

    This is a WIRING test on a 6-step dry run — the error VALUES are meaningless and are
    deliberately not asserted against any threshold. Only their presence and finiteness.
    """
    import json
    import os
    pytest.importorskip("torch")
    droot = _repo_datasets_root()
    if not os.path.exists(os.path.join(droot, "three_gene_val", "payload.h5")):
        pytest.skip("registered three_gene_val not present locally (datasets are gitignored)")

    from rngrn.config import load_config, apply_overrides
    from rngrn.train import fit
    from rngrn.optim.benchmark import build_table, degradation_table, DEGRADATION_COLUMNS

    cfg = load_config(os.path.join(_repo_root(), "configs", "expA_control_full.yaml"))
    cfg = apply_overrides(cfg, ["train.n_restarts=1", "train.adam_steps=6",
                                "train.lbfgs_steps=0"])
    runs = str(tmp_path / "experiments")
    metric = fit(cfg, runs_root=runs)
    assert np.isfinite(metric["kstar_rel_err"]), "headline k* error is still NaN"
    assert np.isfinite(metric["kstar_fft_rel_err"])

    # the run index row (built by dict.update() merge — duplicate keys must not TypeError)
    rows = [json.loads(l) for l in open(os.path.join(runs, "runs.jsonl"))]
    assert len(rows) == 1
    assert np.isfinite(rows[0]["kstar_rel_err"])
    assert np.isfinite(rows[0]["kstar_fft_rel_err"])
    assert rows[0]["run_id"] == metric["run_id"], "run identity lost in the merge"

    tbl = build_table(runs_root=runs)
    assert np.isfinite(tbl[0]["kstar_rel_err_mean"])
    assert np.isfinite(tbl[0]["kstar_fft_rel_err_mean"])

    deg = degradation_table(runs_root=runs)
    assert "kstar_fft_rel_err_mean" in DEGRADATION_COLUMNS
    assert np.isfinite(deg[0]["kstar_rel_err_mean"])
    assert np.isfinite(deg[0]["kstar_fft_rel_err_mean"])


def test_registry_config_does_not_hardcode_L():
    """The five file-backed configs must not carry an L that silently loses to the file."""
    import os
    import yaml
    for name in ["expA_control_full", "expA_hidden_channel", "expB_control_matched",
                 "expB_overparam", "m3_registry"]:
        path = os.path.join(_repo_root(), "configs", f"{name}.yaml")
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        assert "L" not in (raw.get("data") or {}), (
            f"{name}.yaml sets data.L, which the gate ignores in favour of the sample's "
            f"own L — use data.L_override if you mean an explicit cross-check")


# --------------------------------------------------------------------------------------
# the MEASURED scaling fact, pinned so a rescaled future dataset is noticed loudly
# --------------------------------------------------------------------------------------
def _repo_root():
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[1])


def _repo_datasets_root():
    import os
    return os.path.join(_repo_root(), "data", "datasets")


def _bins(attrs):
    """The linear k* expressed in FFT bins: k* = 2*pi*n/L  =>  n = k* L / 2pi."""
    return float(attrs["k_star"]) * float(attrs["L"]) / TWO_PI


@pytest.mark.parametrize("dataset_id", SIX_BIN_DATASETS)
def test_linear_kstar_sits_at_six_fft_bins(dataset_id):
    """MEASURED FACT: every three_gene sample's linear k* is exactly 6.00 FFT bins.

    i.e. `k_star * L / 2pi == 6.000` — the generator chose L to place the fastest-growing
    mode at the 6th Fourier bin, so L and k_star are not independent on this family.

    Pinned because it silently underwrites how the k* metric behaves: if a future dataset
    picks a different bin, `kstar_rel_err` changes meaning and scale without anything
    failing. Make that noisy rather than invisible. This is an OBSERVATION about the
    supplied data, NOT a requirement the code enforces — the gate reads whatever the file
    stores.
    """
    import os
    path = os.path.join(_repo_datasets_root(), dataset_id, "payload.h5")
    if not os.path.exists(path):
        pytest.skip(f"{dataset_id} not present locally (datasets are gitignored)")
    with h5py.File(path, "r") as f:
        bins = np.array([_bins(f[k].attrs) for k in sorted(f.keys())])
    assert len(bins) > 0
    assert np.allclose(bins, 6.0, rtol=0, atol=1e-6), (
        f"{dataset_id}: linear k* is no longer at exactly 6.00 FFT bins "
        f"(range {bins.min():.6f}..{bins.max():.6f}). If this is an intended rescaling, "
        f"update SIX_BIN_DATASETS and re-read every kstar_rel_err in the run index — the "
        f"metric's scale has changed.")


def test_classical_sets_are_not_all_at_six_bins():
    """Scope guard: the 6.00-bin fact is NOT universal, so don't generalise it.

    15 of the 160 classical samples sit off 6.00 bins, and every one of them is at the
    L = 18.0 floor — the generator clamped the domain instead of reaching 6 bins. A test
    that asserted 6.00 bins across all registered datasets would be wrong.
    """
    import os
    offenders = []
    checked = 0
    for dataset_id in ["three_gene_classical_train", "three_gene_classical_val",
                       "three_gene_classical_test", "two_gene_classical_train",
                       "two_gene_classical_val", "two_gene_classical_test"]:
        path = os.path.join(_repo_datasets_root(), dataset_id, "payload.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for k in sorted(f.keys()):
                checked += 1
                if abs(_bins(f[k].attrs) - 6.0) > 1e-6:
                    offenders.append((dataset_id, k, float(f[k].attrs["L"])))
    if checked == 0:
        pytest.skip("classical datasets not present locally (datasets are gitignored)")
    assert offenders, ("the classical sets used to contain off-6-bin samples; if they no "
                       "longer do, the scope note above is stale")
    assert all(L == pytest.approx(18.0) for _, _, L in offenders), (
        f"off-6-bin samples are no longer all at the L=18.0 floor: {offenders[:5]}")


def test_kstar_fft_lands_on_the_half_bin_grid():
    """Why the SECONDARY metric has a floor: the FFT k* is quantised, not biased.

    `k_star_fft * L / 2pi` lands on the half-integer bin grid for every registered sample
    (to within 3-decimal storage rounding), so its disagreement with the linear k* is a
    grid offset of EITHER sign — which is why kstar_fft_rel_err is a diagnostic and never
    a tuning target.
    """
    import os
    devs = []
    for dataset_id in sorted(os.listdir(_repo_datasets_root())):
        path = os.path.join(_repo_datasets_root(), dataset_id, "payload.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for k in sorted(f.keys()):
                a = f[k].attrs
                if "k_star_fft" not in a:
                    continue
                n = float(a["k_star_fft"]) * float(a["L"]) / TWO_PI
                devs.append(abs(n * 2 - round(n * 2)))
    if not devs:
        pytest.skip("registered datasets not present locally (datasets are gitignored)")
    assert max(devs) < 0.05, f"k_star_fft is off the half-bin grid by {max(devs):.4f} bins"