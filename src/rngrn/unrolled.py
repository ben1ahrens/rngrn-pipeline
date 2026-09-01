"""unrolled.py — the truncated-unrolled gradient path (`docs/REDESIGN_rngrn.md` §4.2).

RECOVERY-SIDE. Inputs are the model's own parameters, a field the caller already holds, and
grid geometry. No observed frame, no answer-key quantity, no import of either.

THE SECOND GRADIENT PATH. `forward.PatternSolve` differentiates the patterned steady state
through the implicit-function theorem: exact, O(1) memory, and premised on F(u*) = 0, which
F-D1-5 measured as generically unreachable on incommensurate boxes. This module is the
structural fallback §4.2 specifies for exactly that case: run the ETDRK4 relax as
differentiable torch ops and backpropagate through the unrolled steps. It requires **no
convergence at all** — the loss is defined on the relaxed field after finite time — so the
stall class that bounds the adjoint path cannot touch it.

TRUNCATED, and this is the whole design:

* the warm-started state is **DETACHED** on entry, unconditionally. Whatever produced it —
  a gradient-free relax, or a previous differentiated segment — contributes exactly zero.
* only the final `segment_steps` are differentiated, from a **saturated** warm state. §4.2
  gives the rationale as gradients exploding through the exponential-growth phase of the
  instability — that explosion **did not reproduce** on the measured fixture (no segment
  length in [1, 2048] produced a non-finite field or gradient, either from a saturated or
  from a growth-phase start; D-R3-3). What truncation actually buys here is bounded
  activation memory and a gradient that is still finite-difference-VERIFIABLE; what a
  non-saturated start actually costs is a 35.6% band-power gradient error that no segment
  length up to 1024 repairs. The knob stays a knob for those reasons.
* those steps run under **gradient checkpointing** (`checkpoint_every`), so retained
  activation memory is set by the number of block boundaries rather than by the step count.
  MEASURED at n=32, N=3, float64 (`tests/test_unrolled_grad.py`): 24,576 B/step retained at
  `checkpoint_every=1` — exactly one field — against 773,184 B/step un-checkpointed, a
  **31.5x** reduction, and 0 B/step for the warm-up at any length. Those are MARGINAL bytes
  per differentiated step, taken between two segment lengths; the raw TOTALS differ by only
  1.35x at 6 steps, because the constant in the next paragraph dominates them both.

A COST THIS PATH CARRIES, stated plainly: making the ETDRK4 coefficients differentiable in D
retains the contour-integral graph (M=32 complex128 tensors of shape (N, n, n/2+1)), which
MEASURED 12.59 MB at n=32 — a CONSTANT, independent of segment length, but ~8.7x larger at
n=96 by shape alone (~110 MB, UNMEASURED). It dominates the activation budget at short
segments. Trading it away means giving up the theta_D gradient or checkpointing the
coefficient build as well; neither is done here.

**The segment length is CALIBRATED ONLY INSIDE THIS PATH'S DESIGNED OPERATING REGIME.**
`docs/REDESIGN_rngrn.md` §8 item 14 assigned that calibration to R3; the measured curve
(`scripts/r3_unrolled_segment.py`, committed under
`experiments/redesign_r3/unrolled_segment/`) set `SEGMENT_STEPS_DEFAULT = 128` for a
**saturated** warm start, and for nothing else. Read that constant before using the default —
it carries the three conditions the ruling attaches to it (D-R3-2), the load-bearing one being
that this path **must not be invoked from a non-saturated state at all**.

WHAT IS DIFFERENTIATED AND WHAT IS NOT. The reaction's parameters (KA, KR, alpha, beta,
delta) and the ETDRK4 coefficients (through D, which enters only via the linear operator
-D k^2) are live; the timestep `dt`, the grid `(n, L)` and the initial field are not. dt is
set from the detached |eig(J)|_max timescale and the solve box from the detached k-hat
(`solve_box.py`, §4.3) — both are geometry, and §4.3 requires geometry not to be
differentiated. A finite-difference check of this path must therefore hold dt, the box and
the warm state FIXED while perturbing theta, or it is checking a different function.

SERIAL MODEL ONLY. `model` is an `RNGRN`; the leading axis of `X0` is a batch of INITIAL
CONDITIONS for that one model, not the member axis of a `BatchedRNGRN`. The batched twin
would need `forward._batched_reaction_builder` and `forward._half_coeffs_batched` in place of
the two builders used here; `torch_half_coeffs` refuses a (B, N) D loudly rather than
flattening it into B*N fake species. That twin is not written — Task 14 (fd_ab, D-R3-5) ran
without producing it, so it is an open, un-commissioned R4-scope gap: until it exists, a
batched spectral run must select `gradient_path='adjoint'` deliberately (`recover.py`'s
refusal states this loudly). Plan citation: `docs/PLAN_redesign_R3.md` lives on
`docs/redesign-rngrn` until merged.

NOT WIRED HERE: what is scored on the relaxed field. The spectral targets on the solve box
(`solve_box.interpolate_targets` + `band_bins`, and the model-side band selection that goes
with them) belong to the objective, not to the gradient path, and are Tasks 13/14 of
`docs/PLAN_redesign_R3.md`. This module returns the differentiable field and stops there.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .etdrk4_torch import (_torch_reaction_builder, integrate_etdrk4_rfft_torch,
                           torch_half_coeffs)
from .model import RNGRN

#: Differentiated steps in the truncated segment. ADOPTED by controller ruling (D-R3-2)
#: from the measured gradient-error-vs-segment-length curve,
#: `experiments/redesign_r3/unrolled_segment/results/curve.json`. Three conditions travel
#: with the number and none of them is optional:
#:
#: (a) CALIBRATED on ONE fixture, ONE solve box, ONE seed — the D1/D2 known-Turing checkpoint
#:     at k-hat=0.17607 on the p=8, n=96 commensurate box, 6 FD directions, two placeholder
#:     loss functionals. At 128 the gradient sits at cosine 0.999994 (amp) / 0.999997
#:     (log band power) and relative norm gap 0.394% / 0.287% against the S=2048 reference,
#:     FD-faithful at 9.2e-10 / 3.9e-10 — five orders below D1's 1e-4 acceptance.
#: (b) UNCALIBRATED beyond that point. It is not established as transferable to another
#:     fixture, box, k-hat or objective. Task 14's A/B against the adjoint has since RUN at
#:     the operating point and did not move this number (D-R3-5 rider 3: "Promotion does
#:     not calibrate it; it raises the stakes on it") — it remains uncalibrated beyond
#:     that point.
#: (c) THE PATH MUST NOT BE INVOKED FROM A NON-SATURATED STATE. Measured from a growth-phase
#:     warm start (20.4% of saturated amplitude), the band-power gradient is 35.6% off in
#:     norm at S=128 (cosine 0.936) and NO segment length up to 1024 repairs it — the error
#:     only closes once the segment is itself long enough to reach saturation. §4.3's
#:     stall-switch must therefore hand this path a SATURATED field; a stalled Newton polish
#:     is not the same condition as an unsaturated relax, and only the latter is disqualifying.
SEGMENT_STEPS_DEFAULT = 128


def unrolled_relax(model: RNGRN, X0: torch.Tensor, n: int, L: float, dt: float,
                   segment_steps: int = SEGMENT_STEPS_DEFAULT, warmup_steps: int = 0,
                   checkpoint_every: int | None = None,
                   device: torch.device | None = None) -> torch.Tensor:
    """`warmup_steps` un-differentiated ETDRK4 steps, then `segment_steps` differentiated.

    `X0` is the warm state and **must be SATURATED** — see `SEGMENT_STEPS_DEFAULT` condition
    (c): from a growth-phase field the band-power gradient is 35.6% off in norm at the default
    segment length and no length up to 1024 repairs it. That condition is a CALLER CONTRACT,
    not a check: saturation is a property of the trajectory that produced `X0`, which this
    function does not see, and inventing a single-field proxy for it here would be a threshold
    nothing has calibrated. The caller that ran the relax knows whether its detector fired.

    `X0` is (N, n, n) or (B, N, n, n); the returned field has the same rank. It is DETACHED on
    entry — §4.2's truncation — so a caller may pass a tensor that still carries a graph
    without silently widening the differentiated stretch.

    `checkpoint_every` is the gradient-checkpointing block size in steps: `None` runs the
    segment as one plain graph (every step's activations retained), an integer re-runs each
    block in the backward pass instead of storing it. The two are numerically identical up to
    float64 round-off — MEASURED 9.1e-17 relative on the full theta gradient at n=32 over a
    6-step segment, for block sizes 1, 2 and 3 (`tests/test_unrolled_grad.py`, which asserts
    <= 1e-14) — so the choice is memory-vs-compute only, not numerical.

    Raises on a non-finite field (CLAUDE.md §4) rather than handing the optimiser a NaN
    gradient. The check is ONE `isfinite` on the segment's output, for
    `integrate_etdrk4_rfft_torch`'s reason: every operation in a step is linear or an FFT
    over the whole field, so a non-finite value cannot be erased once it appears. What is
    lost is WHICH step blew up.
    """
    segment_steps = int(segment_steps)
    warmup_steps = int(warmup_steps)
    if segment_steps < 1:
        raise ValueError(
            f"segment_steps must be at least 1 — an unrolled path with nothing unrolled has "
            f"no gradient at all; got {segment_steps!r}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps cannot be negative, got {warmup_steps!r}")
    if checkpoint_every is not None:
        checkpoint_every = int(checkpoint_every)
        if checkpoint_every < 1:
            raise ValueError(
                f"checkpoint_every must be at least 1 step per block, got "
                f"{checkpoint_every!r} (pass None for no checkpointing)")
    if X0.dim() not in (3, 4):
        raise ValueError(
            f"X0 must be (N, n, n) or (B, N, n, n), got shape {tuple(X0.shape)}")

    dev = model.device if device is None else torch.device(device)
    squeeze = X0.dim() == 3
    X = X0.detach().to(dev)
    if squeeze:
        X = X[None]

    reaction_t = _torch_reaction_builder(model, dev, differentiable=True)
    coeffs = torch_half_coeffs(model.D, n, L, dt, dev)

    def steps(x: torch.Tensor, m: int) -> torch.Tensor:
        # check_blowup=False: the driver checks once, below. Inside a checkpointed block the
        # check would also run again on every recompute.
        out, _blew = integrate_etdrk4_rfft_torch(x, reaction_t, n, dt, m, coeffs,
                                                 check_blowup=False)
        return out

    if warmup_steps:
        with torch.no_grad():
            X = steps(X, warmup_steps)
        X = X.detach()

    if checkpoint_every is None:
        X = steps(X, segment_steps)
    else:
        done = 0
        while done < segment_steps:
            m = min(checkpoint_every, segment_steps - done)
            X = checkpoint(steps, X, m, use_reentrant=False)
            done += m

    if not bool(torch.isfinite(X).all()):
        raise RuntimeError(
            f"unrolled segment blew up: the field is non-finite after {warmup_steps} warm-up "
            f"+ {segment_steps} differentiated ETDRK4 steps at dt={dt:g}, n={n}, L={L:g}. "
            f"Refusing to return a NaN gradient path.")
    return X[0] if squeeze else X
