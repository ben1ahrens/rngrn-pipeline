"""r3_stall_survey.py — the population stall-rate DISTRIBUTION on the commensurate solve box.

`docs/REDESIGN_rngrn.md` §4.3 defines the adaptive commensurate solve box (`solve_box.py`,
Task 11) and the two-path gradient switch: a member whose Newton polish meets
`forward.PatternSolver.CONVERGENCE_TOL` (1e-9) gets the adjoint/IFT gradient, one that misses
it (a "stall") gets the truncated-unrolled gradient instead
(`recover._spectral_solve_with_stall_switch`, Task 13). §4.3 leaves the switch's calibration
open: "a per-run counter of ignited-member solves missing the 1e-9 bar" is wired
(`RecoveryResult.n_ignited_solves` / `.n_stalled_solves`), but `stall_switch_fraction=0.20` is
UNCALIBRATED (`docs/PLAN_redesign_R3.md` §8 register item 14) — it wants the DISTRIBUTION of
stall outcomes, not just a single run's mean, because the fraction is meant to be chosen
against the measured gradient-error difference between the two paths (Task 14's FD A/B:
converged-arm adjoint is FD-faithful to 1.7e-06, a stalled member's adjoint gradient is O(1)
wrong, the unrolled path is faithful on both at ~1e-8). This script measures that distribution.

WHY A NEW SCRIPT AND NOT `recover(stall_switch=True)` DIRECTLY. `recover()`'s stall-aware
solver is hard-wired to the CALLER's (data) grid (`n=frame.shape[-1], L=L_model`) —
`_spectral_solve_with_stall_switch`'s own docstring says as much: wiring `solve_box.py`'s
commensurate geometry into `recover()` is Task 14's driver / R4's integration, ruled OUT of
Task 13's scope by the controller (2026-08-19). This task's brief explicitly asks for the
COMMENSURATE box, so this script drives a population directly against Task 11's
`solve_box.geometry`/`needs_retile` and Task 13's stall-switch decision logic, without
touching `recover.py`.

`stall_check_solve` below is a NEAR-VERBATIM COPY of `recover._spectral_solve_with_stall_switch`
— same steady-state / relax / Newton / path-choice / pattern-floor sequence, same constants
— with ONE difference: it also returns the Newton residual, which the original discards after
comparing it to the bar. Duplicating ~25 lines here (rather than editing `recover.py`, which
would be a production-code change for a measurement script's benefit) is the same trade-off
`scripts/r3_fd_ab.py` already made for its own `AdjointPath`/`UnrolledPath` classes. `main()`
runs a ONE-SHOT equivalence check against the real `_spectral_solve_with_stall_switch` at the
same (model, box, seed) before anything else, so drift between the two would be caught rather
than silently trusted (see the FD A/B "solver equivalence" precedent).

POPULATION. Each "member" starts as the tracked known-Turing checkpoint (same one
`scripts/r3_fd_ab.py` uses) displaced by a random unit-direction offset of a given `scale`,
across `--scales` — the same three scales `r3_fd_ab.py`'s stall hunt used (0.1, 0.5, 2.0),
extended here to six and run with many more trials per scale, since a budget-limited hunt of
1 trial/scale (Task 14) found no genuine stall. Each member then runs `--steps` real Adam
steps against the batched-free (SERIAL) five spectral loss terms
(`losses.spectral.spectral_terms_batched`), evaluated on THIS member's own commensurate box,
against a SURROGATE target pattern (a nearby-theta relaxation on the SAME box — the identical
firewall-clean device `r3_fd_ab.py`'s `surrogate_frame` uses: no `payload.h5`, no `AnswerKey`).
The box re-tiles per `solve_box.needs_retile`'s hysteresis (Task 11) exactly as §4.3 specifies,
never on a fixed cadence.

IGNITION GATE — CORRECTED 2026-08-19 (Task 16 review finding I2). A step whose member has not
converged to a steady state is not-ignited exactly as production treats it (that half of the
gate matches `losses.spectral.is_ignited`'s own precondition). Whether the member is
Turing-UNSTABLE ENOUGH to ignite, however, is decided here with `sig.max() > IGNITION_MARGIN`
on this script's own fixed `torch.linspace(1e-3, 10, 2000)` k-grid — this is NOT
`losses.spectral.is_ignited`'s real gate, which reads `sig_max_pos` from
`losses.terms.turing_hinges_split` (a k-FLOOR-limited slice of the dispersion curve). An
earlier version of this docstring claimed the two "mirror" each other; they do not, and the
difference is not cosmetic: `turing_hinges_split`'s k-floor sits near k≈1.0 on this survey's
population, whose `khat` values all fall in [0.101, 0.191] — a literal `sig_max_pos` check
would have de-ignited the ENTIRE population, including every genuine stall this script found.
This is a DELIBERATE DIVERGENCE, kept because it is the only gate that produces a usable
population here, but it makes this script's ignition test strictly MORE PERMISSIVE than
production's — a member this script calls "ignited" is not guaranteed to pass production's own
`is_ignited`.

DENOMINATOR — A SECOND DIVERGENCE (Task 16 review finding I1). A step whose member IS ignited
but whose solve fails for a non-stall reason (steady-state / relax failure, or a homogeneous /
non-finite result — `reason != "ok"`) is recorded in this script's per-step trace but EXCLUDED
from the stall-rate numerator AND denominator. Production does NOT exclude it: `spec_ignited`
is set to `1.0` BEFORE the solve runs (`losses/total.py:49`, unconditional on the outcome), and
`recover._account_for_stall` (`recover.py:330-332`) increments `n_ignited_solves`
UNCONDITIONALLY whenever it is called (i.e. whenever `spec_ignited==1.0`), counting every
non-"ok" reason as a NON-stall in the denominator (`last_stalled` is freshly `False` for every
non-"ok" reason, never stale). So a real run's `n_stalled_solves`/`n_ignited_solves` ratio
COUNTS relax/pattern failures as non-stalls; this script's `stall_fraction` DROPS them from
both counts entirely — a different (arguably more honest, since a relax/pattern failure is not
evidence about the Newton-convergence question at all) but genuinely DIFFERENT definition.
Zero non-"ok" solves occurred in the population run below, so this divergence does not change
the 25.7% headline this time — but it is a divergence, not a match, and a reader must not
assume this script's aggregate is bit-for-bit comparable to a real run's counters without it.

A "recovery kick" (a small fixed-seed random nudge to theta) is applied on any step that
produced no gradient (not ignited, or the solve failed for a non-stall reason) so a member
stuck in a dead patch of theta-space does not burn its whole step budget doing nothing.

WHAT THIS SCRIPT DOES NOT COVER (state plainly, per CLAUDE.md §8/§10):
  * ONE checkpoint (N=3, `m3_registry_20260803_190250_seed3`), ONE architecture/form. The
    stall rate for a different N or a different topology is not measured here.
  * The perturbation scales sample a NEIGHBOURHOOD of one known-Turing point, not the full
    theta space an independent from-scratch training run would visit. A member that starts
    far outside the Turing class (`--scales` 2.0-and-up, per Task 14's hunt) is excluded from
    the ignited population the moment it fails THIS script's ignition gate — which is more
    PERMISSIVE than production's own `is_ignited` (see the divergence note above), not "same
    as production".
  * The "training-like" Adam steps here use ONLY the five spectral terms against a synthetic
    surrogate target — not the full A0 objective (kstar/turing/anticollapse/anchor). This
    affects how realistically theta WANDERS across steps; it does not affect the measured
    quantity itself (Newton residual / stall outcome), which is a pure function of
    (theta, n, L, seed) and does not depend on what loss is driving theta.
  * `--steps` real Adam steps per member is a SHORT population (brief's own word), not a full
    training budget — see the module docstring's cost accounting below for why.
  * **The measurement is made at `solve_box.N_DEFAULT` (n=96) and is GRID-DEPENDENT.** The
    Newton residual — hence "stalled" vs "converged" itself — is a function of the grid the
    forward solve runs on, not a property of theta alone. This n=96 is the grid the controller
    has since RULED for Phase II (owner-delegated, D-R3-6 pending commit as this fix round is
    written); this distribution is a TRANSFER CAVEAT against that ruling, not a pending
    question — it does not automatically carry over if the ruled grid ever changes.

FIREWALL. No `payload.h5`, no `AnswerKey`. Every input is the tracked checkpoint (via
`torch.load`, the `r3_fd_ab.py` precedent) or a synthetic pattern this script itself relaxes.

Run — CPU, threads pinned to 1 (Task 11: the batched CPU relax is ~215x slower at 14 threads;
this script is serial-model-only, same discipline applies to the BLAS/FFT libraries scipy and
numpy use underneath `newton_polish`/relax):

    bash scripts/guarded_run.sh .venv/bin/python scripts/r3_stall_survey.py \\
        --out experiments/redesign_r3/stall_rate
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import time

# BEFORE numpy/torch — see scripts/r3_fd_ab.py's identical comment: torch.set_num_threads(1)
# pins torch's intra-op pool but not the BLAS/FFT threads scipy/numpy use underneath the
# Newton/relax legs.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

import rngrn.solve_box as sb
from rngrn.forward import (PatternSolve, PatternSolver, make_spatial_F, newton_polish,
                           relax_to_pattern_torch, translation_modes)
from rngrn.losses.spectral import (SPECTRAL_TERM_KEYS, SpectralConfig, build_frame_targets,
                                   spectral_terms_batched)
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN, THETA_NAMES
from rngrn.recover import _spectral_solve_with_stall_switch
from rngrn.unrolled import SEGMENT_STEPS_DEFAULT, unrolled_relax

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
NEWTON_ITER_FULL = 30                 # matches BatchedPatternSolver's / r3_fd_ab.py's default
KGRID_MAX, KGRID_N = 10.0, 2000       # matches r3_fd_ab.py's operating_point()
IGNITION_MARGIN = 1e-3                # losses.spectral.SpectralConfig.ignition_margin default
KICK_SCALE = 0.02                     # small fixed-seed nudge for a stuck (non-ignited /
                                       # solve-failed) step -- see module docstring
SCALES_DEFAULT = (0.1, 0.3, 0.6, 1.0, 1.5, 2.0)
MEMBERS_PER_SCALE_DEFAULT = 5
STEPS_DEFAULT = 20
ADAM_LR_DEFAULT = 0.05                # the tracked checkpoint's own train.adam_lr
GRAD_CLIP_DEFAULT = 10.0              # the tracked checkpoint's own train.grad_clip
RELAX_NOISE, RELAX_CHUNK, RELAX_MAX_CHUNKS = 1e-2, 500, 400   # recover.py's own defaults


# ------------------------------------------------------------------------------- fixtures

def load_fixture(pert: np.ndarray | None = None) -> RNGRN:
    """The tracked known-Turing checkpoint, optionally displaced by a flat theta offset.
    Same fixture and load path as `scripts/r3_fd_ab.py::load_fixture`."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    if pert is not None:
        off = 0
        with torch.no_grad():
            for nm in THETA_NAMES:
                p = getattr(m, nm)
                p.add_(torch.from_numpy(pert[off:off + p.numel()]).reshape(p.shape))
                off += p.numel()
    return m


def theta_dim(model) -> int:
    return sum(getattr(model, nm).numel() for nm in THETA_NAMES)


def theta_params(model) -> list:
    return [getattr(model, nm) for nm in THETA_NAMES]


def perturbed_copy(model: RNGRN, pert: np.ndarray) -> RNGRN:
    """A deep copy of `model` with `pert` added to its raw theta -- used for the surrogate
    target's "nearby theta", which must track the MEMBER's current (evolving) state, unlike
    `load_fixture`'s fixed checkpoint base."""
    m2 = copy.deepcopy(model)
    off = 0
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(m2, nm)
            p.add_(torch.from_numpy(pert[off:off + p.numel()]).reshape(p.shape))
            off += p.numel()
    return m2


def member_seed(scale: float, idx: int) -> int:
    import hashlib
    h = hashlib.blake2b(f"stall_survey|{scale!r}|{idx!r}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % 2 ** 32


def initial_model(scale: float, idx: int) -> tuple:
    """(model, seed) -- the member's STARTING perturbed theta, seed-deterministic from
    (scale, idx) alone. Factored out of `run_member` (Task 16 review finding I3c) so a cheap
    init-only re-scan (`init_scan`) can reproduce exactly what a full `run_member` call would
    have started from, without paying for the training loop -- `member_seed` and the
    `np.random.default_rng(seed)` draw below are the ONLY sources of randomness in building
    this starting point, so this is bit-reproducible."""
    seed = member_seed(scale, idx)
    rng = np.random.default_rng(seed)
    dim = theta_dim(load_fixture())
    direction = rng.standard_normal(dim)
    pert0 = direction / np.linalg.norm(direction) * float(scale)
    return load_fixture(pert0), seed


# ------------------------------------------------------------------- cheap dispersion check

def cheap_dispersion(model, kgrid_max: float = KGRID_MAX, kgrid_n: int = KGRID_N) -> dict:
    """The LINEAR stability diagnostic only -- no relax, no Newton -- so this script does not
    pay a full relax+Newton on a member that is not even predicted Turing-unstable.

    CORRECTED 2026-08-19 (Task 16 review finding I2): this does NOT mirror
    `losses.spectral.is_ignited` (`sig_max_pos` from `turing_hinges_split`'s k-floor-limited
    slice) -- it is `sig.max()` on this script's own fixed k-grid, which is strictly MORE
    PERMISSIVE. See the module docstring's "IGNITION GATE" paragraph for why (the real
    `sig_max_pos` k-floor would have de-ignited this entire population). The steady-state
    half of the gate DOES match production: both require `steady_state`'s own convergence
    check to pass before anything downstream is trusted.

    ALWAYS returns a dict (never None, per Task 16 review finding I3) with an "ok" bool, so a
    caller can tell WHY a member did not ignite:
      * `ok=False, failure="steady_state_failed"` -- `steady_state` itself did not converge.
      * `ok=False, failure="exception_<Type>"` -- a downstream numerical exception (e.g. a
        non-finite Jacobian on a badly displaced theta). Caught broadly and TYPED rather than
        left unhandled, because a population survey must not die on one degenerate member --
        but the exception's type is recorded rather than swallowed into an untyped None.
      * `ok=True` with `sig_max <= IGNITION_MARGIN` -- a THIRD, different outcome ("left the
        Turing class"): steady state converged and the dispersion computation succeeded, the
        member is simply not unstable enough. Callers check `sig_max` themselves; this
        function does not conflate that case with either failure above.
    """
    try:
        xs, ok = steady_state(model)
        if not ok:
            return dict(ok=False, failure="steady_state_failed")
        J = model.jacobian(xs, create_graph=False).detach()
        kg = torch.linspace(1e-3, kgrid_max, kgrid_n)
        sig = model.dispersion(xs, kg, J=J).detach()
        sig_max = float(sig.max())
        khat = float(kg[int(sig.argmax())])
        rate = float(np.abs(np.linalg.eigvals(J.cpu().numpy())).max())
        gamma = rate if (np.isfinite(rate) and rate > 0.0) else None
        return dict(ok=True, sig_max=sig_max, khat=khat, xstar=xs.detach().cpu().numpy(),
                   gamma=gamma)
    except Exception as exc:                                              # noqa: BLE001
        return dict(ok=False, failure=f"exception_{type(exc).__name__}")


def _kick(model, seed: int, step: int, scale: float = KICK_SCALE) -> None:
    """A small fixed-seed random nudge to theta, applied on a step that produced no gradient
    (not ignited, or the solve failed for a non-stall reason) -- see module docstring."""
    rng = np.random.default_rng((seed * 7919 + step) % (2 ** 32))
    dim = theta_dim(model)
    d = rng.standard_normal(dim)
    d = d / np.linalg.norm(d) * scale
    off = 0
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(model, nm)
            p.add_(torch.from_numpy(d[off:off + p.numel()]).reshape(p.shape))
            off += p.numel()


# --------------------------------------------------------------- the stall-check solve itself

def stall_check_solve(model, n: int, L: float, seed: int, newton_iter: int = NEWTON_ITER_FULL,
                      segment_steps: int = SEGMENT_STEPS_DEFAULT) -> tuple:
    """(u_star | None, path | None, stalled: bool, residual: float, reason: str) on box (n, L).

    A near-verbatim copy of `recover._spectral_solve_with_stall_switch` -- see the module
    docstring for why this duplicates rather than imports it (it needs the residual, which the
    original discards). ONE further divergence, disclosed here: on a non-finite/non-positive
    `|eig(J)|_max`, `recover.py:246-247` RAISES (`RuntimeError`, fail-loud); this function
    instead returns `reason="bad_jacobian_rate"` (fail-SOFT), because a population survey must
    keep going past one degenerate member rather than abort the whole run on it. Not observed
    in the population run below (`reason_counts` has no `bad_jacobian_rate` entries)."""
    xs, ok = steady_state(model)
    if not ok:
        return None, None, False, float("nan"), "steady_state_failed"
    xstar = xs.detach().cpu().numpy()
    J = model.jacobian(xs, create_graph=False).detach()
    jac_rate = float(np.abs(np.linalg.eigvals(J.cpu().numpy())).max())
    if not (np.isfinite(jac_rate) and jac_rate > 0.0):
        return None, None, False, float("nan"), "bad_jacobian_rate"
    dt = 0.2 / jac_rate
    gamma = jac_rate
    D_np = model.D.detach().cpu().numpy()

    try:
        X_sat = relax_to_pattern_torch(model, xstar, n, L, dt, seed, torch.device("cpu"),
                                       noise=RELAX_NOISE, chunk=RELAX_CHUNK,
                                       max_chunks=RELAX_MAX_CHUNKS)
    except RuntimeError:
        return None, None, False, float("nan"), "relax_failed"

    F_fn = make_spatial_F(model, n, L)
    k2_full = _spectral_k2_local(n, L)
    modes_of = lambda uu: translation_modes(uu, n, L)                       # noqa: E731
    u, res = newton_polish(F_fn, X_sat, modes_of, k2_full, D_np, gamma, n_iter=newton_iter)
    res = float(res)

    if res <= PatternSolver.CONVERGENCE_TOL:
        payload = dict(model=model, u_star=u.detach(), n=n, L=L, k2_full=k2_full,
                       D_np=D_np, gamma=gamma, k2h=None, k2_dev=None, D_dev=None)
        out = PatternSolve.apply(payload, *theta_params(model))
        path, stalled = "adjoint", False
    else:
        out = unrolled_relax(model, X_sat, n, L, dt, segment_steps=segment_steps,
                             device=torch.device("cpu"))
        path, stalled = "unrolled", True

    pattern_floor = max(1e-3, 0.02 * abs(float(xstar[0])))
    amp = float(out[0].detach().std())
    if not np.isfinite(amp):
        return None, None, False, res, "solve_failed"
    if amp <= pattern_floor:
        return None, None, False, res, "not_patterned"
    return out, path, stalled, res, "ok"


def _spectral_k2_local(n: int, L: float) -> np.ndarray:
    """Same formula `eval.numerics._spectral_k2` uses -- imported by name here to avoid a
    second private cross-module import; both are `2*pi*fftfreq(n, d=L/n)` meshed and summed
    in quadrature. Kept as a one-line local so this script's only recover-side reuse point
    (`_spectral_solve_with_stall_switch`) is confined to the ONE-SHOT equivalence check."""
    from rngrn.eval.numerics import _spectral_k2
    return _spectral_k2(n, L)


# ---------------------------------------------------------------------- the loss functional

def eval_terms(u_stack: torch.Tensor, targets: dict, cfg: SpectralConfig) -> dict:
    """The five ACTIVE spectral terms on a (1, N, n, n) stack -> {name: scalar tensor}.
    Identical to `r3_fd_ab.py::eval_terms`."""
    vals, _parts = spectral_terms_batched(u_stack, targets, cfg, members=[0])
    return {k: vals[k][0] for k in SPECTRAL_TERM_KEYS}


def surrogate_frame_on_box(model, box: dict, seed: int) -> np.ndarray | None:
    """A nearby-theta pattern relaxed on THIS box -- `r3_fd_ab.py::surrogate_frame`'s device,
    but anchored to `model`'s CURRENT (evolving) state rather than a fixed checkpoint, since a
    member's target must track its own commensurate box across re-tiles. Returns None if the
    nearby theta fails to reach a usable steady state / pattern (rare; the caller falls back
    to the previous target)."""
    rng = np.random.default_rng((seed + 991) % (2 ** 32))
    dim = theta_dim(model)
    pert = 0.01 * rng.standard_normal(dim)
    nearby = perturbed_copy(model, pert)
    try:
        xs2, ok2 = steady_state(nearby)
        if not ok2:
            return None
        J2 = nearby.jacobian(xs2, create_graph=False).detach()
        rate2 = float(np.abs(np.linalg.eigvals(J2.cpu().numpy())).max())
        if not (np.isfinite(rate2) and rate2 > 0.0):
            return None
        dt2 = 0.2 / rate2
        X = relax_to_pattern_torch(nearby, xs2.detach().cpu().numpy(), box["n"], box["L_solve"],
                                   dt2, seed, torch.device("cpu"),
                                   noise=RELAX_NOISE, chunk=RELAX_CHUNK,
                                   max_chunks=RELAX_MAX_CHUNKS)
    except RuntimeError:
        return None
    return X.detach().numpy()


# ------------------------------------------------------------------------------ one member

def run_member(scale: float, idx: int, steps: int, lr: float, grad_clip: float) -> dict:
    model, seed = initial_model(scale, idx)

    op = cheap_dispersion(model)
    if not op["ok"] or not (op["sig_max"] > IGNITION_MARGIN):
        init_reason = op["failure"] if not op["ok"] else "left_turing_class"
        return dict(scale=float(scale), member=idx, seed=seed, init_ignited=False,
                    init_sig_max=(op["sig_max"] if op["ok"] else None),
                    init_reason=init_reason, n_retile=0, records=[])

    khat_at_tile = op["khat"]
    box = sb.geometry(khat_at_tile)
    cfg = SpectralConfig()
    frame = surrogate_frame_on_box(model, box, seed)
    targets = build_frame_targets(frame, box["L_solve"], khat_at_tile, cfg) \
        if frame is not None else None

    params = theta_params(model)
    opt = torch.optim.Adam(params, lr=lr)
    n_retile = 0
    records = []

    for step in range(steps):
        opv = cheap_dispersion(model)
        if not opv["ok"] or not (opv["sig_max"] > IGNITION_MARGIN):
            step_reason = opv["failure"] if not opv["ok"] else "left_turing_class"
            records.append(dict(step=step, ignited=False, reason=step_reason,
                                sig_max=(opv["sig_max"] if opv["ok"] else None)))
            _kick(model, seed, step)
            continue

        khat_now = opv["khat"]
        retiled = False
        if sb.needs_retile(khat_now, khat_at_tile, p=box["p"]):
            box = sb.geometry(khat_now)
            khat_at_tile = khat_now
            n_retile += 1
            retiled = True
            frame = surrogate_frame_on_box(model, box, seed + step + 1)
            if frame is not None:
                targets = build_frame_targets(frame, box["L_solve"], khat_at_tile, cfg)
            # else: keep the previous targets rather than train against nothing this step;
            # noted via "retile_target_stale" below.

        try:
            out, path, stalled, res, reason = stall_check_solve(
                model, box["n"], box["L_solve"], (seed + 7919 * step) % (2 ** 32))
        except Exception as exc:                                          # noqa: BLE001
            out, path, stalled, res = None, None, False, float("nan")
            reason = f"exception_{type(exc).__name__}"

        rec = dict(step=step, ignited=True, khat=khat_now, sig_max=opv["sig_max"],
                   box_n=box["n"], box_p=box["p"], box_L=box["L_solve"], retiled=retiled,
                   retile_target_stale=(retiled and frame is None),
                   path=path, stalled=bool(stalled), residual=res, reason=reason)

        if reason != "ok" or targets is None:
            records.append(rec)
            _kick(model, seed, step)
            continue

        terms = eval_terms(out[None], targets, cfg)
        loss = sum(terms.values())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        rec["loss"] = float(loss.detach())
        records.append(rec)

    return dict(scale=float(scale), member=idx, seed=seed, init_ignited=True,
               init_sig_max=op["sig_max"], n_retile=n_retile, records=records)


# -------------------------------------------------------------------------------- aggregation

def _quantiles(xs: list) -> dict | None:
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    return dict(n=int(a.size), min=float(a.min()), p10=float(np.quantile(a, 0.10)),
               p50=float(np.quantile(a, 0.50)), p90=float(np.quantile(a, 0.90)),
               max=float(a.max()), mean=float(a.mean()))


def summarize(all_members: list) -> tuple:
    """(summary: dict, per_member: list, solved_records: list) -- fixed from a `-> dict`
    annotation that did not match the actual 3-tuple return (Task 16 review, cheap minor)."""
    solved = [dict(scale=m["scale"], member=m["member"], step=r["step"], path=r["path"],
                   stalled=r["stalled"], residual=r["residual"], khat=r["khat"])
              for m in all_members for r in m["records"]
              if r.get("ignited") and r.get("reason") == "ok"]
    n_total = len(solved)
    n_stalled = sum(1 for r in solved if r["stalled"])
    stall_fraction = (n_stalled / n_total) if n_total else float("nan")

    per_member = []
    for m in all_members:
        s = [r for r in m["records"] if r.get("ignited") and r.get("reason") == "ok"]
        st = sum(1 for r in s if r["stalled"])
        per_member.append(dict(scale=m["scale"], member=m["member"], n_solved=len(s),
                               n_stalled=st, stall_rate=(st / len(s) if s else None)))

    reason_counts: dict = {}
    for m in all_members:
        if not m["init_ignited"]:
            reason_counts["init_not_ignited"] = reason_counts.get("init_not_ignited", 0) + 1
        for r in m["records"]:
            # `reason` is meaningful on BOTH ignited and non-ignited records as of the
            # Task 16 review fix (I3a) -- non-ignited steps now carry the specific reason
            # ("steady_state_failed" / "exception_<Type>" / "left_turing_class") rather than
            # a generic "not_ignited" placeholder, so no ternary override is needed here.
            key = r.get("reason", "unknown")
            reason_counts[key] = reason_counts.get(key, 0) + 1

    by_scale: dict = {}
    for m in all_members:
        s = m["scale"]
        d = by_scale.setdefault(str(s), dict(n_members=0, n_init_ignited=0, n_solved=0,
                                             n_stalled=0))
        d["n_members"] += 1
        d["n_init_ignited"] += int(m["init_ignited"])
        solved_m = [r for r in m["records"] if r.get("ignited") and r.get("reason") == "ok"]
        d["n_solved"] += len(solved_m)
        d["n_stalled"] += sum(1 for r in solved_m if r["stalled"])
    for d in by_scale.values():
        d["stall_fraction"] = (d["n_stalled"] / d["n_solved"]) if d["n_solved"] else None

    return dict(
        n_members=len(all_members),
        n_init_ignited=sum(int(m["init_ignited"]) for m in all_members),
        n_solved_total=n_total, n_stalled_total=n_stalled, stall_fraction=stall_fraction,
        reason_counts=reason_counts, by_scale=by_scale,
        residual_stalled=_quantiles([r["residual"] for r in solved if r["stalled"]]),
        residual_converged=_quantiles([r["residual"] for r in solved if not r["stalled"]]),
        per_member_stall_rate=_quantiles(
            [p["stall_rate"] for p in per_member if p["stall_rate"] is not None]),
    ), per_member, solved


def init_scan(scales: list, members_per_scale: int) -> list:
    """A CHEAP re-scan of every member's STARTING dispersion only -- no relax, no Newton, no
    Adam steps. Added post-hoc (Task 16 review finding I3c) to recover the `init_sig_max` /
    failure-kind evidence behind a "N of M members left the Turing class at init" claim
    WITHOUT re-running the (potentially hours-long, "fresh relax every call") full survey:
    `initial_model` reproduces each member's exact starting perturbation from (scale, idx)
    alone, and `cheap_dispersion` is, by construction, the same cheap linear-algebra check
    `run_member` already pays for at step 0 -- so this is seconds of work reproducing exactly
    what the original run started from, not a new measurement."""
    rows = []
    for scale in scales:
        for idx in range(members_per_scale):
            model, seed = initial_model(scale, idx)
            op = cheap_dispersion(model)
            ignited = bool(op["ok"] and op["sig_max"] > IGNITION_MARGIN)
            rows.append(dict(scale=float(scale), member=idx, seed=seed, ok=op["ok"],
                            sig_max=op.get("sig_max"), khat=op.get("khat"),
                            failure=op.get("failure"), init_ignited=ignited))
    return rows


# -------------------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="run directory; results land in <out>/results/stall_rate.json")
    ap.add_argument("--scales", type=float, nargs="+", default=list(SCALES_DEFAULT))
    ap.add_argument("--members-per-scale", type=int, default=MEMBERS_PER_SCALE_DEFAULT)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--lr", type=float, default=ADAM_LR_DEFAULT)
    ap.add_argument("--grad-clip", type=float, default=GRAD_CLIP_DEFAULT)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--init-scan-only", action="store_true",
                    help="skip the full population survey; only run the cheap per-member "
                         "init_scan (Task 16 review finding I3c) and merge it into "
                         "<out>/results/stall_rate.json's 'init_scan' key")
    a = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)

    if a.init_scan_only:
        rows = init_scan(a.scales, a.members_per_scale)
        n_ignited = sum(r["init_ignited"] for r in rows)
        print(f"init scan: {len(rows)} members, {n_ignited} init-ignited, "
             f"{len(rows) - n_ignited} not init-ignited", flush=True)
        for r in rows:
            print(f"  scale={r['scale']} idx={r['member']}: ok={r['ok']} "
                 f"sig_max={r['sig_max']!r} khat={r['khat']!r} "
                 f"init_ignited={r['init_ignited']} failure={r['failure']!r}", flush=True)
        out_path = pathlib.Path(a.out) / "results" / "stall_rate.json"
        payload = json.loads(out_path.read_text()) if out_path.exists() else {}
        payload["init_scan"] = dict(
            what="cheap re-scan of every member's INITIAL perturbed-theta dispersion only "
                 "(no relax, no Newton, no training steps) -- added post-hoc (Task 16 review "
                 "finding I3) to recover init_sig_max / failure-kind evidence behind the "
                 "'N of M left the Turing class at init' claim, without re-running the full "
                 "training survey. Uses initial_model(), the SAME seed-deterministic "
                 "perturbation run_member starts each member from, so this reproduces "
                 "exactly what the original run's members started at.",
            ignition_margin=IGNITION_MARGIN, rows=rows,
            n_members=len(rows), n_init_ignited=n_ignited)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nmerged init_scan into {out_path}", flush=True)
        return

    t_start = time.perf_counter()

    # ---- one-shot equivalence check: stall_check_solve vs the real recover.py function ----
    base = load_fixture()
    op0 = cheap_dispersion(base)
    if not op0["ok"] or not (op0["sig_max"] > IGNITION_MARGIN):
        raise RuntimeError("the base fixture is not Turing-unstable -- cannot run the "
                           "equivalence check or build the commensurate box")
    box0 = sb.geometry(op0["khat"])
    out_a, path_a, stalled_a, res_a, reason_a = stall_check_solve(
        base, box0["n"], box0["L_solve"], 12345)
    _out_b, path_b, stalled_b, reason_b = _spectral_solve_with_stall_switch(
        load_fixture(), box0["n"], box0["L_solve"], 12345)
    if (path_a, stalled_a, reason_a) != (path_b, stalled_b, reason_b):
        raise RuntimeError(
            f"stall_check_solve does not reproduce recover._spectral_solve_with_stall_switch: "
            f"got (path={path_a!r}, stalled={stalled_a!r}, reason={reason_a!r}) vs the real "
            f"function's (path={path_b!r}, stalled={stalled_b!r}, reason={reason_b!r})")
    print(f"equivalence check OK: path={path_a} stalled={stalled_a} residual={res_a:.3e} "
          f"reason={reason_a} (bar {PatternSolver.CONVERGENCE_TOL:g})", flush=True)

    all_members = []
    for scale in a.scales:
        for idx in range(a.members_per_scale):
            t0 = time.perf_counter()
            rec = run_member(scale, idx, a.steps, a.lr, a.grad_clip)
            dt = time.perf_counter() - t0
            all_members.append(rec)
            solved = [r for r in rec["records"] if r.get("ignited") and r.get("reason") == "ok"]
            st = sum(1 for r in solved if r["stalled"])
            print(f"member scale={scale} idx={idx}: init_ignited={rec['init_ignited']} "
                 f"solved={len(solved)}/{a.steps} stalled={st} n_retile={rec['n_retile']} "
                 f"({dt:.1f}s)", flush=True)

    summary, per_member, records = summarize(all_members)

    out_dir = pathlib.Path(a.out)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    payload = {
        "what": "population stall-rate distribution on the commensurate solve box "
                "(docs/REDESIGN_rngrn.md §4.3; PLAN_redesign_R3 Task 16)",
        "method": {
            "checkpoint": str(CKPT.relative_to(REPO)),
            "scales": list(a.scales), "members_per_scale": a.members_per_scale,
            "steps": a.steps, "adam_lr": a.lr, "grad_clip": a.grad_clip,
            "kick_scale": KICK_SCALE, "newton_iter_full": NEWTON_ITER_FULL,
            "segment_steps": SEGMENT_STEPS_DEFAULT,
            "convergence_bar": PatternSolver.CONVERGENCE_TOL,
            "ignition_margin": IGNITION_MARGIN,
            "device": "cpu", "threads": a.threads, "torch": torch.__version__,
            "argv": list(sys.argv),
            "equivalence_check": {"path": path_a, "stalled": stalled_a,
                                  "residual": res_a, "reason": reason_a},
        },
        "summary": summary,
        "per_member": per_member,
        "records": records,
        # Task 16 review finding I3b: `summary`/`per_member`/`records` all project down to
        # solved (ignited AND reason=="ok") steps, dropping init_sig_max, n_retile, box_*,
        # retiled, loss, and every non-"ok" step record. `all_members` is the RAW per-member,
        # per-step trace `run_member` already builds -- persisted here in full so a reader can
        # audit the evidence behind any "N of M left the Turing class" / "n_retile" claim
        # without re-running anything.
        "all_members": all_members,
        "total_seconds": time.perf_counter() - t_start,
    }
    out = out_dir / "results" / "stall_rate.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
