"""Canonical dataset generation: cv tracing, the saturation gate, the payload contract.

The cv-tracing tests matter more than they look. `simulate_and_classify` is a verbatim port
of the generator that made every registered dataset; any change to it that perturbed the
trajectory would make the canonical sets non-comparable with the entire corpus. These tests
pin that the trace is a read-only observation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import gen_tg3 as G                                                # noqa: E402
import canon_generate as CG                                        # noqa: E402


def tiny_system():
    """A small, fast, known-Turing 3-gene system for integration tests."""
    hits = G.screen_model("double_inhibitor", "multiplicative", n_samples=400, seed=5, want=1)
    if not hits:
        pytest.skip("screen found no Turing system at this seed")
    p = dict(hits[0])
    p["_M"] = G.TOPOLOGIES["double_inhibitor"]
    p["periods_per_box"] = int(p["feasible_periods"][0])
    p["sim_seed"] = 12345
    return p


# --------------------------------------------------------------------------------------
# cv tracing must not perturb the simulation
# --------------------------------------------------------------------------------------
def test_cv_tracing_is_off_by_default_and_changes_nothing():
    p = tiny_system()
    a = G.simulate_and_classify(p, grid=48, Tmax=40.0, seed=7)
    b = G.simulate_and_classify(p, grid=48, Tmax=40.0, seed=7)
    assert np.array_equal(a["final"], b["final"])
    assert "cv_trace" not in a


def test_cv_tracing_records_a_trace_when_requested():
    p = tiny_system()
    r = G.simulate_and_classify(p, grid=48, Tmax=40.0, seed=7, cv_every=50)
    assert len(r["cv_trace"]) == len(r["cv_times"]) >= 2
    assert np.all(np.isfinite(r["cv_trace"]))


def test_tracing_does_not_perturb_the_simulation():
    """The trace must be a read-only observation of the same trajectory."""
    p = tiny_system()
    a = G.simulate_and_classify(p, grid=48, Tmax=40.0, seed=7)
    b = G.simulate_and_classify(p, grid=48, Tmax=40.0, seed=7, cv_every=25)
    assert np.array_equal(a["final"], b["final"])


# --------------------------------------------------------------------------------------
# the saturation gate
# --------------------------------------------------------------------------------------
def test_a_plateaued_trace_is_saturated():
    trace = np.concatenate([np.linspace(0.0, 0.8, 80), np.full(20, 0.8005)])
    assert CG.is_saturated(trace) is True


def test_a_still_growing_trace_is_not_saturated():
    """At larger boxes the pattern needs longer to fill the domain; Tmax=260 was chosen at
    96 and must not be assumed sufficient at 512."""
    assert CG.is_saturated(np.linspace(0.0, 0.8, 100)) is False


def test_saturation_needs_enough_samples_to_judge():
    with pytest.raises(ValueError, match="too short"):
        CG.is_saturated(np.array([0.5, 0.5]))


# --------------------------------------------------------------------------------------
# rebuilding generator input from a stored sample
# --------------------------------------------------------------------------------------
def test_params_from_sample_rebuilds_the_generator_input():
    p = tiny_system()
    fake = {"params": {k: p[k] for k in ("b", "V", "mu", "K", "n", "D")},
            "x_star": np.asarray(p["x_star"]),
            "attrs": {"k_star": p["k_star"], "sim_seed": 99, "grid": 96}}
    fake["params"]["interaction_matrix"] = p["_M"]
    out = CG.params_from_sample(fake, periods=20)
    assert out["periods_per_box"] == 20
    assert out["sim_seed"] == 99
    assert np.allclose(out["_M"], p["_M"])


# --------------------------------------------------------------------------------------
# the canonical payload contract
# --------------------------------------------------------------------------------------
def _record(grid=32):
    rng = np.random.default_rng(0)
    return {"final": rng.random((3, grid, grid)).astype(np.float32),
            "L": 120.0, "dx": 120.0 / grid, "dt_sim": 0.01, "grid": grid,
            "cv0": 0.7, "morphology": "spots", "wavelength": 20.0, "k_star_fft": 0.31,
            "area_frac": 0.2, "n_components": 40, "anisotropy": 0.05,
            "cv_trace": np.linspace(0, 0.7, 20), "cv_times": np.linspace(0, 260, 20),
            "params": {"b": [1., 1., 1.], "V": [1., 1., 1.], "mu": [1., 1., 1.],
                       "K": [1., 1., 1.], "n": 2.0, "D": [1., 30., 40.],
                       "topology": "double_inhibitor", "reaction": "multiplicative",
                       "interaction_matrix": [[1, -1, -1], [1, 0, 0], [1, 0, 0]],
                       "k_star": 0.30, "sim_seed": 7, "periods_per_box": 20,
                       "x_star": [1., 1., 1.], "_M": [[1, -1, -1], [1, 0, 0], [1, 0, 0]]},
            "source_dataset": "three_gene_qvar", "source_key": "sample_0002",
            "system_id": 2, "role": "tuning"}


def test_payload_carries_the_attributes_the_gate_requires(tmp_path):
    """gate.from_registry RAISES if a sample lacks L or k_star -- it refuses to default
    them. Everything else in the payload is optional to the loader."""
    import h5py
    out = CG.write_canonical_payload([_record()], str(tmp_path / "payload.h5"))
    with h5py.File(out, "r") as f:
        g = f["sample_0000"]
        assert "L" in g.attrs and "k_star" in g.attrs
        assert g["final_frame"].shape == (3, 32, 32)
        assert "params_json" in g.attrs and "sim_seed" in g.attrs


def test_payload_stores_no_trajectory(tmp_path):
    """src/rngrn/ never reads `trajectory`; at 512 it would be 6x the storage for nothing."""
    import h5py
    out = CG.write_canonical_payload([_record()], str(tmp_path / "payload.h5"))
    with h5py.File(out, "r") as f:
        assert "trajectory" not in f["sample_0000"]
        assert "cv_trace" in f["sample_0000"]


def test_payload_records_its_provenance(tmp_path):
    """Which corpus sample this was promoted from, and which side of the split it is on."""
    import h5py
    out = CG.write_canonical_payload([_record()], str(tmp_path / "payload.h5"))
    with h5py.File(out, "r") as f:
        g = f["sample_0000"]
        assert g.attrs["source_dataset"] == "three_gene_qvar"
        assert g.attrs["source_key"] == "sample_0002"
        assert g.attrs["role"] == "tuning"


def test_all_three_channels_are_stored(tmp_path):
    """Real images give one observable, but storing all three keeps m=1 vs m=3 a
    controlled comparison rather than a different dataset."""
    import h5py
    out = CG.write_canonical_payload([_record()], str(tmp_path / "payload.h5"))
    with h5py.File(out, "r") as f:
        assert f["sample_0000"]["final_frame"].shape[0] == 3


def test_payload_round_trips_through_the_firewall_gate(tmp_path):
    """The real contract: gate.from_registry must be able to load what we wrote."""
    import json
    from rngrn.data import registry as reg
    from rngrn.data import gate

    root = tmp_path / "datasets"
    (root / "turing_spots").mkdir(parents=True)
    CG.write_canonical_payload([_record()], str(root / "turing_spots" / "payload.h5"))
    reg.scan(str(root))
    ri, ak = gate.from_registry(str(root), "turing_spots", "sample_0000",
                                N=3, observed_idx=(0, 1, 2))
    assert ri.frame.shape == (3, 32, 32)
    assert ri.L == pytest.approx(120.0)
