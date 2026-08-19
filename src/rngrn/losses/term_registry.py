"""term_registry.py — the loss-term registry (Task 8, R2 redesign; RECOVERY SIDE).

Names the loss terms `losses/terms.py::DEFAULT_WEIGHTS` and `losses/total.py`
implement, pairing each with its serial callable, its batched twin (or the
reason batching is refused), its current default weight, and an honest calibration
tag. At Task 8 this covered exactly the 12 terms `total.py` wires into
compute_terms/compute_terms_batched, and `default_weights()` was required to equal
the legacy `DEFAULT_WEIGHTS` dict bit-for-bit
(tests/test_term_registry.py::test_default_weights_are_bit_identical_to_the_legacy_dict)
-- that pin still holds because `terms.py` now derives `DEFAULT_WEIGHTS` from
`default_weights()` at import time, rather than carrying a second literal, so the two
are tautologically equal. Task 14 added a 13th entry, `kstar_si`, BORN REGISTERED:
present here with a real serial+batched callable, but not yet wired into
compute_terms/total_loss (docs/REDESIGN_rngrn.md §4.4) -- so a caller that only reads
compute_terms's own term_vals dict still sees the original 12.

Exactly two terms refuse a batched form: `resid` (losses/total.py::compute_terms_batched
raises if compute_resid=True — the batched reaction takes one state vector per member, not
per-pixel states) and `morphology` (losses/terms.py::morphology_consistency is a
non-differentiable numpy diagnostic, never wired into compute_terms/compute_terms_batched
at all — see terms.py's b2-section comment).

The five spectral terms USED TO refuse as well, and no longer do (R3 Phase B, collision
ledger row 28). `losses/spectral.py` now carries a `<key>_batched` twin for each, driven by
`forward.BatchedPatternSolver`'s per-member warm state, so the refusal text this file used
to hold was retired together with `losses/total.py`'s raise. Their `fn`/`batched_fn` arity
is `(u_star, targets, cfg[, members])`, not the `(model, xstar, kgrid, ...)` of every other
entry — a pre-existing asymmetry of the spectral terms, not something the flip introduced.
`tests/test_term_registry.py::test_a_refusing_term_has_no_batched_implementation_sitting_next_to_it`
is what keeps a `refusal_reason` here from outliving the fact it asserts.

Calibration tags follow CLAUDE.md §8/§10: `CALIBRATED(<source>)` only where a source
measured and settled the DEFAULT VALUE itself; `UNCALIBRATED` otherwise, even where a
term's provenance cites a measurement (e.g. `anchor`'s weight 2.0 traces to exp02, but
TUNING.md keeps it tagged [TUNE] and says outright "Sweep it before trusting any
anchor-sensitive result" — that is not a settled default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..registry import Registry
from . import spectral as S
from . import terms as T

LOSS_TERMS: "Registry[LossTerm]" = Registry("loss_terms")


@dataclass(frozen=True)
class LossTerm:
    name: str
    fn: Callable                      # serial callable, existing signature
    batched_fn: Optional[Callable]    # None requires refusal_reason
    refusal_reason: Optional[str]
    default_weight: float
    calibration: str                  # "CALIBRATED(<source>)" or "UNCALIBRATED"


# Quoted VERBATIM from losses/total.py::compute_terms_batched's own raise text (below,
# `total.py` sources its ValueError message from this registry entry so the two can never
# drift apart; see the wiring at the bottom of that module).
_RESID_REFUSAL = (
    "compute_terms_batched cannot compute the stationarity residual: the batched "
    "reaction takes one state vector per member, not per-pixel states. Its default "
    "weight is 0 (exp06 settled it off), so batched recovery is available only for "
    "loss.weights.resid == 0. Use the serial path for residual runs."
)

_MORPHOLOGY_REFUSAL = (
    "morphology_consistency is a non-differentiable numpy diagnostic and is not wired "
    "into compute_terms/compute_terms_batched at all (losses/terms.py's b2-section "
    "comment: 'NOT batched, deliberately ... morphology_consistency (non-differentiable "
    "numpy diagnostic)'); loss.weights.morphology is currently INERT (TUNING.md)."
)


def _register(name, fn, batched_fn, refusal_reason, default_weight, calibration):
    LOSS_TERMS.register(name)(LossTerm(
        name=name, fn=fn, batched_fn=batched_fn, refusal_reason=refusal_reason,
        default_weight=default_weight, calibration=calibration))


_register("kstar", T.kstar_anchor, T.kstar_anchor_batched, None,
           default_weight=1.0, calibration="UNCALIBRATED")
_register("kstar_si", T.kstar_anchor_si, T.kstar_anchor_si_batched, None,
           default_weight=0.0, calibration="UNCALIBRATED")
_register("turing", T.turing_hinges_split, T.turing_hinges_split_batched, None,
           default_weight=1.0, calibration="UNCALIBRATED")
_register("resid", T.stationarity_residual, None, _RESID_REFUSAL,
           default_weight=0.0, calibration="CALIBRATED(exp06 — measured harmful, off)")
_register("anticollapse", T.anticollapse, T.anticollapse_batched, None,
           default_weight=0.5, calibration="UNCALIBRATED")
_register("anchor", T.frame_scale_anchor, T.frame_scale_anchor_batched, None,
           default_weight=2.0, calibration="UNCALIBRATED")
_register("morphology", T.morphology_consistency, None, _MORPHOLOGY_REFUSAL,
           default_weight=0.0, calibration="UNCALIBRATED")
_register("param_prior", T.param_prior, T.param_prior_batched, None,
           default_weight=0.0, calibration="UNCALIBRATED")
for _key in S.SPECTRAL_TERM_KEYS:
    # Batched since R3 Phase B: every one of the five has a `<key>_batched` twin in
    # `losses/spectral.py` (uniform naming, verified). Weights stay 0.0 and the tag stays
    # UNCALIBRATED -- nothing has measured a spectral weight on this data, and the batched
    # twins change neither the arithmetic nor what would calibrate it.
    _register(_key, getattr(S, _key), getattr(S, f"{_key}_batched"), None,
              default_weight=0.0, calibration="UNCALIBRATED")
del _key


def default_weights() -> dict[str, float]:
    """Every registered term's default weight, keyed by name. Replaces the `DEFAULT_WEIGHTS`
    literal that used to live in `terms.py`; that name now aliases this function's
    result at import time, so `terms.DEFAULT_WEIGHTS` stays a valid, bit-identical
    plain dict for every existing caller."""
    return {key: LOSS_TERMS.get(key).default_weight for key in LOSS_TERMS.keys()}
