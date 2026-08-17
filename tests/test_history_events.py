"""Task 9: TrainingHistory population events + invariant trajectories.

Extends `TrainingHistory` with a discrete event log (`record_event`, kinds
"ignition"/"deignition"/"stall"/"cull"/"death") and named invariant trajectories
(`record_invariants`), both rendered by `to_arrays()`. A0 protection: every key,
shape and value `to_arrays()` produced before this task must be byte-identical for
a caller that never touches the new methods -- see
`test_legacy_keys_are_untouched_when_new_methods_are_never_called`.
"""
import numpy as np
import pytest

from rngrn.history import TrainingHistory


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
                ss_converged=True, spec_ignited=1.0, L_kstar=0.5, L_turing=0.25,
                L_anchor=0.1, L_anticollapse=0.0,
                weights_used=dict(kstar=1.0, turing=0.5, anchor=2.0))


def _record_legacy(h):
    for step in (0, 1):
        for member in (0, 1):
            h.record_serial(step, member, _parts(total=1.0 / (step + 1)),
                             _FakeModel(scale=step + 1))


def test_events_and_invariants_round_trip_through_to_arrays():
    h = TrainingHistory(every=1, total_steps=1, n_members=2, N=2)
    _record_legacy(h)

    h.record_event(1, 0, "ignition")
    h.record_event(1, 1, "cull")
    h.record_invariants(1, 0, dict(d_ratio=20.0, alpha_over_delta=0.75))

    a = h.to_arrays()

    events = a["events"]
    assert len(events) == 2
    got = {(int(r["step"]), int(r["member"]), str(r["kind"])) for r in events}
    assert got == {(1, 0, "ignition"), (1, 1, "cull")}

    names = list(a["invariant_names"])
    inv = a["invariants"]
    step_idx = list(a["invariant_step"]).index(1)
    assert float(inv[step_idx, 0, names.index("d_ratio")]) == pytest.approx(20.0)
    assert float(inv[step_idx, 0, names.index("alpha_over_delta")]) == pytest.approx(0.75)
    # member 1 never got an invariants row -> NaN, not a shifted/missing lane
    assert np.all(np.isnan(inv[step_idx, 1, :]))


def test_record_death_delegates_to_record_event():
    h = TrainingHistory(every=1, total_steps=1, n_members=2, N=2)
    _record_legacy(h)
    h.record_death(1, 1)

    a = h.to_arrays()
    assert a["hist_death_step"][1] == 1.0          # legacy behaviour preserved
    events = a["events"]
    assert len(events) == 1
    r = events[0]
    assert (int(r["step"]), int(r["member"]), str(r["kind"])) == (1, 1, "death")


def test_invalid_event_kind_raises():
    h = TrainingHistory(every=1, total_steps=1, n_members=1, N=2)
    with pytest.raises(ValueError, match="unknown"):
        h.record_event(0, 0, "not-a-real-kind")


def test_legacy_keys_are_untouched_when_new_methods_are_never_called():
    h_legacy = TrainingHistory(every=1, total_steps=1, n_members=2, N=2)
    _record_legacy(h_legacy)
    baseline = h_legacy.to_arrays()

    h_new = TrainingHistory(every=1, total_steps=1, n_members=2, N=2)
    _record_legacy(h_new)
    h_new.record_event(1, 0, "ignition")
    h_new.record_invariants(1, 0, dict(d_ratio=20.0))
    extended = h_new.to_arrays()

    legacy_keys = ("hist_step", "hist_member", "hist_scalar_names", "hist_scalars",
                   "hist_param_names", "hist_params", "hist_death_step")
    for key in legacy_keys:
        np.testing.assert_array_equal(baseline[key], extended[key])
    # a caller that never touches the new methods still gets the new keys, but empty --
    # additive, never a behaviour change for the legacy keys themselves.
    assert len(baseline["events"]) == 0
    assert baseline["invariants"].shape == (0, 2, 0)
    assert set(extended) >= set(legacy_keys) | {"events", "invariants"}


def test_empty_history_still_renders_nothing():
    assert TrainingHistory(every=1, total_steps=3, n_members=1, N=2).to_arrays() == {}
