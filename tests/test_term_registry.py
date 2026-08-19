"""test_term_registry.py — the loss-term registry (Task 8, R2 redesign).

A0 protection: this refactor moves DEFAULT_WEIGHTS and the term functions onto a
registry (losses/term_registry.py) but must not change a single default value or any
term's arithmetic. `test_default_weights_are_bit_identical_to_the_legacy_dict` is the
pin for that.
"""
from rngrn.losses import term_registry, terms


def test_every_registered_term_is_fully_classified():
    for key in term_registry.LOSS_TERMS.keys():
        t = term_registry.LOSS_TERMS.get(key)
        assert (t.batched_fn is not None) != (t.refusal_reason is not None), key
        assert t.calibration.startswith(("CALIBRATED(", "UNCALIBRATED")), key


def test_default_weights_are_bit_identical_to_the_legacy_dict():
    # A0 protection: the registry refactor may not change a single default.
    assert term_registry.default_weights() == dict(terms.DEFAULT_WEIGHTS)


def test_registry_has_exactly_the_twelve_legacy_keys():
    assert term_registry.LOSS_TERMS.keys() == sorted(terms.DEFAULT_WEIGHTS)


def test_default_weights_match_a_hardcoded_literal_snapshot():
    """The A0 pin that is NOT tautological (R3 Task 8 Step 2 / collision ledger row 35).

    `test_default_weights_are_bit_identical_to_the_legacy_dict` above compares
    `default_weights()` against `terms.DEFAULT_WEIGHTS`, and since Task 8 the latter IS
    `default_weights()` (terms.py: `DEFAULT_WEIGHTS = _term_registry.default_weights()`).
    Both sides move together, so that test cannot catch a flipped default -- exactly the
    damage class that silently reverted `resid` to 0.3 and dropped the anchor weight in the
    phase-A merge.

    The literal below is therefore written out by hand, ONCE, and is the only copy in the
    tree that does not derive from the registry. The twelve legacy entries are transcribed
    verbatim from the pre-registry `terms.DEFAULT_WEIGHTS` dict at merge-base `10cff1b`;
    `kstar_si` is Task 14's thirteenth entry, born registered at 0.0 (docs/REDESIGN_rngrn.md
    §4.4). A deliberate default change must edit this literal in the same commit -- that
    edit is the announcement CLAUDE.md §8 requires, not an obstacle to it.
    """
    expected = {
        "kstar": 1.0,
        "kstar_si": 0.0,
        "turing": 1.0,
        "resid": 0.0,
        "anticollapse": 0.5,
        "anchor": 2.0,
        "morphology": 0.0,
        "param_prior": 0.0,
        "spec_shape": 0.0,
        "spec_aniso": 0.0,
        "spec_amp_mean": 0.0,
        "spec_amp_fluct": 0.0,
        "real_moments": 0.0,
    }
    got = term_registry.default_weights()
    assert got == expected, (
        "a registered term's DEFAULT WEIGHT changed. If that is deliberate, edit the "
        "literal in this test in the same commit and announce the change "
        f"(CLAUDE.md 8). got={got}")


def test_dispersion_side_terms_all_accept_the_hoisted_jacobian():
    """Rows 9/10 of docs/INTEGRATION_r3_collisions.md, as a contract rather than a habit.

    `losses/total.py` evaluates ONE autograd Jacobian per step and threads it as `J=` into
    every dispersion-side term. The failure mode the ledger names is silent: a term added
    later (T14 added two, `kstar_si`/`kstar_si_batched`, after gpu-optim wrote the hoist)
    keeps computing its own Jacobian, and nothing raises -- the value is identical, only
    the cost differs. This pins `J=`/`idx=` as a property of every registered
    dispersion-side entry, serial AND batched, so a new one cannot quietly opt out.

    `anchor` (frame_scale_anchor) is deliberately excluded: it does not read the Jacobian
    at all and must not acquire one. `resid`, `morphology` and the five spectral terms are
    not dispersion-side either.
    """
    import inspect

    needs_J = ("kstar", "kstar_si", "turing", "anticollapse")
    needs_idx = ("kstar", "kstar_si")
    for key in needs_J:
        t = term_registry.LOSS_TERMS.get(key)
        for label, fn in (("fn", t.fn), ("batched_fn", t.batched_fn)):
            assert fn is not None, f"{key}.{label} is unexpectedly None"
            params = inspect.signature(fn).parameters
            assert "J" in params, f"{key}.{label} ({fn.__name__}) does not accept J="
            assert params["J"].default is None, f"{key}.{label} J= must default to None"
            if key in needs_idx:
                assert "idx" in params, f"{key}.{label} ({fn.__name__}) does not accept idx="
                assert params["idx"].default is None, f"{key}.{label} idx= must default None"

    # the control-arm hinges are selected by split_hinges=False, so they are not reachable
    # through the registry -- pin them directly or the control arm silently loses the hoist.
    for fn in (terms.turing_hinges, terms.turing_hinges_batched):
        assert inspect.signature(fn).parameters["J"].default is None, fn.__name__

    anchor = term_registry.LOSS_TERMS.get("anchor")
    for fn in (anchor.fn, anchor.batched_fn):
        assert "J" not in inspect.signature(fn).parameters, (
            f"{fn.__name__} acquired a J= it has no use for")


def test_default_weights_is_a_plain_dict():
    w = term_registry.default_weights()
    assert isinstance(w, dict)
    assert w is not term_registry.default_weights()  # not aliased/mutable-shared
