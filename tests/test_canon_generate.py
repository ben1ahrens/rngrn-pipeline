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
