"""term_registry.py — the loss-term registry (Task 8, R2 redesign; RECOVERY SIDE).

Names the 12 loss terms `losses/terms.py::DEFAULT_WEIGHTS` and `losses/total.py`
already implement, pairing each with its serial callable, its batched twin (or the
reason batching is refused), its current default weight, and an honest calibration
tag. This module adds no new behaviour and changes no default: `default_weights()`
must equal the legacy `DEFAULT_WEIGHTS` dict bit-for-bit (pinned by
tests/test_term_registry.py::test_default_weights_are_bit_identical_to_the_legacy_dict).
`terms.py` now derives `DEFAULT_WEIGHTS` from `default_weights()` at import time,
rather than carrying a second literal.

Two terms refuse a batched form: `resid` (losses/total.py::compute_terms_batched raises
if compute_resid=True — the batched reaction takes one state vector per member, not
per-pixel states) and `morphology` (losses/terms.py::morphology_consistency is a
non-differentiable numpy diagnostic, never wired into compute_terms/compute_terms_batched
at all — see terms.py's b2-section comment). The five spectral terms also refuse: they are
not owned by this file's per-term batched_fn slot but by losses/spectral.py; batching is
refused by losses/total.py::compute_terms_batched for the same reason forward.PatternSolver
cannot be shared across a batched member axis.

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

# Quoted VERBATIM from losses/total.py::compute_terms_batched's own raise text — same
# single-sourcing note as _RESID_REFUSAL above.
_SPECTRAL_REFUSAL = (
    "compute_terms_batched cannot compute the spectral terms (unit U4): "
    "forward.PatternSolver owns per-restart warm-start state, which has no batched "
    "form, and the batched reaction does not broadcast to per-pixel fields. Use the "
    "serial path for spectral runs."
)


def _register(name, fn, batched_fn, refusal_reason, default_weight, calibration):
    LOSS_TERMS.register(name)(LossTerm(
        name=name, fn=fn, batched_fn=batched_fn, refusal_reason=refusal_reason,
        default_weight=default_weight, calibration=calibration))


_register("kstar", T.kstar_anchor, T.kstar_anchor_batched, None,
           default_weight=1.0, calibration="UNCALIBRATED")
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
    _register(_key, getattr(S, _key), None, _SPECTRAL_REFUSAL,
              default_weight=0.0, calibration="UNCALIBRATED")
del _key


def default_weights() -> dict[str, float]:
    """The 12 term default weights, keyed by name. Replaces the `DEFAULT_WEIGHTS`
    literal that used to live in `terms.py`; that name now aliases this function's
    result at import time, so `terms.DEFAULT_WEIGHTS` stays a valid, bit-identical
    plain dict for every existing caller."""
    return {key: LOSS_TERMS.get(key).default_weight for key in LOSS_TERMS.keys()}
