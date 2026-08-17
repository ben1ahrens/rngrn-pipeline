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


def test_default_weights_is_a_plain_dict():
    w = term_registry.default_weights()
    assert isinstance(w, dict)
    assert w is not term_registry.default_weights()  # not aliased/mutable-shared
