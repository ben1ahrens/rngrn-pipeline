"""test_viz.py — smoke tests for the viz module figures.

`viz.py` is top-level (outside the firewall): it reads exactly the dict shape
`TrainingHistory.to_arrays()` produces (history.py, Task 9) and draws nothing but
matplotlib figures from it. These tests build that dict by hand — never a real
`TrainingHistory`, `payload.h5` or answer key — so they run on any fresh clone.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np                                                 # noqa: E402
import pytest                                                      # noqa: E402

from rngrn import viz                                              # noqa: E402


def _full_arrays():
    """A minimal but complete `to_arrays()`-shaped dict: 2 members, 3 steps."""
    scalar_names = np.array(["sig_max_pos", "total"])
    scal = np.array([
        [[0.10, 1.00], [0.05, 1.20]],
        [[0.30, 0.80], [0.25, 0.90]],
        [[0.60, 0.50], [np.nan, np.nan]],   # member 1 died before step 2
    ], dtype=np.float32)                                            # (S=3, B=2, Q=2)
    events = np.array(
        [(1, 0, "ignition"), (2, 1, "death")],
        dtype=[("step", "i8"), ("member", "i8"), ("kind", "U16")],
    )
    inv = np.array([[[20.0], [np.nan]], [[18.0], [5.0]]], dtype=np.float32)  # (Si=2,B=2,Qi=1)
    return {
        "hist_step": np.array([0, 1, 2], dtype=np.int64),
        "hist_member": np.arange(2, dtype=np.int64),
        "hist_scalar_names": scalar_names,
        "hist_scalars": scal,
        "hist_param_names": np.array(["D0", "D1"]),
        "hist_params": np.zeros((3, 2, 2), dtype=np.float32),
        "hist_death_step": np.array([np.nan, 2.0]),
        "events": events,
        "invariant_step": np.array([0, 2], dtype=np.int64),
        "invariant_names": np.array(["d_ratio"]),
        "invariants": inv,
    }


def _legacy_arrays():
    """Pre-Task-9 shape: no `events`/`invariant_*` keys at all."""
    a = _full_arrays()
    for key in ("events", "invariant_step", "invariant_names", "invariants"):
        del a[key]
    return a


@pytest.mark.parametrize(
    "arrays", [_full_arrays(), _legacy_arrays(), {}],
    ids=["full", "legacy-only", "empty"],
)
def test_all_four_figures_are_written_and_nonempty(tmp_path, arrays):
    outs = [
        viz.loss_curves(arrays, str(tmp_path / "loss.png")),
        viz.invariant_trajectories(arrays, str(tmp_path / "invariants.png")),
        viz.event_timeline(arrays, str(tmp_path / "events.png")),
        viz.spectral_trace(arrays, 0.4, str(tmp_path / "spectral.png")),
    ]
    for path in outs:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_each_function_returns_the_out_png_path_it_was_given(tmp_path):
    arrays = _full_arrays()
    assert viz.loss_curves(arrays, str(tmp_path / "a.png")) == str(tmp_path / "a.png")
    assert viz.invariant_trajectories(arrays, str(tmp_path / "b.png")) == str(tmp_path / "b.png")
    assert viz.event_timeline(arrays, str(tmp_path / "c.png")) == str(tmp_path / "c.png")
    assert viz.spectral_trace(arrays, 0.4, str(tmp_path / "d.png")) == str(tmp_path / "d.png")


def test_malformed_dict_raises_rather_than_silently_producing_a_blank_figure(tmp_path):
    # hist_scalars shape does not match hist_step/hist_scalar_names -> fail loud, not a
    # silently wrong plot.
    bad = _full_arrays()
    bad["hist_scalars"] = bad["hist_scalars"][:, :, :1]  # drops the "total" column's data
    with pytest.raises((IndexError, ValueError)):
        viz.loss_curves(bad, str(tmp_path / "bad.png"))
