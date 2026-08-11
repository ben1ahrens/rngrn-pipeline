"""analysis.py — linear stability, Turing conditions, and the robustness cloud.

Reads only the recovered model (firewall-safe). turing_ok evaluates the Turing
conditions on a numpy J, D; robustness_cloud is the payoff metric (fraction of a
log-normal parameter cloud that still patterns + k* spread).

PROVENANCE: repackaged from rngrn_eval.py, which the project tracks as SCAFFOLD
(not yet validated), unlike the VALIDATED model/observables core. The Turing-
condition and dispersion logic here still needs validation by Claude Code (a
science test against the reference answer keys is the natural anchor).
"""
from __future__ import annotations
import numpy as np
import torch


# [UNCALIBRATED] the default scan band. `linspace(1e-3, 50, ...)` is an ABSOLUTE band in
# rad/length, inherited from the scaffold and never calibrated — the same class of defect
# unit B4 removed from recover._kgrid_for, where an absolute 2.0 floor silently dominated
# whenever kstar_obs < 0.25. It is currently wide enough for the registered L range
# (18..220 -> k* 0.17..2.09) and is left as-is so k* values stay comparable, but it is not
# scale-free and nothing tests its boundary. Pass an explicit kgrid for any L outside that
# range. See docs/DECISIONS.md D-EVID-11.
_TURING_KGRID = np.linspace(1e-3, 50, 4000)


def turing_ok(J, D, kgrid=None, tol=1e-9):
    """Evaluate the Turing conditions on numpy J, D. Returns (ok: bool, info dict).

    THE TWO CONDITIONS, both evaluated strictly (corrected 2026-08-04, D-EVID-11):

      1. UNIFORMLY STABLE:  max Re eig(J) < 0      -- i.e. sigma(k=0) < 0
      2. STRUCTURALLY UNSTABLE:  max over k > 0 of sigma(k) > tol

    Condition 1 used to be `tr(J) < 0`. **A negative trace does not imply stability**: the
    trace is the SUM of the eigenvalues, so one eigenvalue can be positive while the sum is
    negative. Worse, the default grid starts at k = 1e-3, where sigma(k) is still
    essentially sigma(0) -- so a system that was merely uniformly unstable satisfied
    condition 2 as well, with its own uniform instability, and `turing_ok` returned True
    with `kstar` pinned to the grid floor.

    MEASURED, 2026-08-04: of 398 converged `model.init='low_basal'` draws at N=3, **206
    (51.8%) passed the loose test and 0 (0.0%) pass this one** -- and all 206 had k* exactly
    at the grid floor. The low-basal init makes inits uniformly UNSTABLE; it does not make
    them Turing-reachable. See docs/DECISIONS.md D-EVID-11.

    This now matches the criterion `_perturb_cloud` has always used (strict eigenvalue test
    at line ~86, structured max over k>0 at line ~92), so `recovered_turing` and the
    `turing_volume_*` columns finally answer the same question.
    `scripts/exp11_robustness_baseline.py:23` recorded the discrepancy without fixing it.

    The loose verdict is returned alongside as `turing_loose` / `stable_uniform_loose`
    rather than discarded, so a row recorded under the old criterion stays interpretable --
    the same convention `_perturb_cloud` uses with `frac_loose` / `frac_loose_only`.
    """
    J = np.asarray(J); D = np.asarray(D)
    if kgrid is None:
        kgrid = _TURING_KGRID
    kgrid = np.asarray(kgrid, float)

    # sigma(0) exactly, independent of whether the caller's grid contains 0.
    eig0 = np.real(np.linalg.eigvals(J))
    sig0 = float(eig0.max())

    kpos = kgrid[kgrid > 0.0]
    if kpos.size == 0:
        raise ValueError("turing_ok needs at least one k > 0 in kgrid; the k=0 mode is the "
                         "uniform mode and can never be the structured instability.")
    sig = np.array([np.max(np.real(np.linalg.eigvals(J - k**2 * np.diag(D)))) for k in kpos])

    tr0 = float(np.trace(J))
    stable_uniform = sig0 < 0.0                    # STRICT
    unstable_struct = bool(sig.max() > tol)
    ok = bool(stable_uniform and unstable_struct)

    return ok, dict(tr0=tr0, sig0=sig0, sig_max=float(sig.max()),
                    kstar=float(kpos[np.argmax(sig)]),
                    stable_uniform=bool(stable_uniform),
                    unstable_struct=unstable_struct,
                    # the superseded criterion, kept visible and explicitly labelled
                    stable_uniform_loose=bool(tr0 < 0.0),
                    turing_loose=bool(tr0 < 0.0 and unstable_struct))


def linear_stability(model, xstar, D=None):
    """Full linear-stability readout at a given steady state (numpy).

    `D` : PHYSICAL diffusivities, overriding `model.D`. Pass it for anything reloaded from a
    checkpoint written on the non-dimensional path, where `theta_D` is log(D/L**2) — see
    `eval.lgen_eval.physical_model_from_checkpoint`, which does the conversion. None keeps
    `model.D`, correct on the dimensional path where the two coincide (D-EVID-14).
    """
    xs_t = torch.as_tensor(np.asarray(xstar, float))
    J = model.jacobian(xs_t, create_graph=False).detach().cpu().numpy()
    if D is None:
        D = model.D.detach().cpu().numpy()
    D = np.asarray(D, dtype=float)
    ok, info = turing_ok(J, D)
    info["J"] = J.tolist(); info["D"] = D.tolist()
    info["turing"] = ok
    return info


# k-grid for the robustness cloud's own dispersion scan. Deliberately separate from
# turing_ok's default kgrid (this one matches scripts/exp11_robustness_baseline.py, the
# already-validated baseline this module is read against).
#
# [UNCALIBRATED] the 4.0 cap is ABSOLUTE, in rad/length — the same class of defect unit B4
# removed from recover._kgrid_for. It is adequate for the registered L range (18..220 ->
# k* 0.17..2.09) and is kept as-is so numbers stay comparable to the exp11 baseline, but it
# is not scale-free and nothing tests its boundary. It is also the mechanism that made
# D-EVID-14 so loud: a D scaled down by L**2 pushes the unstable band far outside this
# window, so the cloud reported ~0 rather than merely a shifted k*.
_CLOUD_KGRID = np.concatenate([[0.0], np.linspace(1e-3, 4.0, 250)])

# Noise levels the run index reports at. 4.8% is Tica et al.'s measured experimental
# parameter CV between biological repeats, included so our number is comparable to theirs.
ROBUSTNESS_SIGMA_LEVELS = {"1pct": 0.01, "4p8pct": 0.048, "10pct": 0.10, "20pct": 0.20}


def _draw_JD_cloud(J, D, sigma_log, rng, n_samples):
    """Draw n_samples independent log-normal multiplicative perturbations of the
    PHYSICAL (J, D). Only NONZERO Jacobian entries and diffusivities are perturbed,
    each with its own lognormal factor (always positive, so a negative J entry stays
    negative and a positive one stays positive) — sign structure and structural zeros
    (including an immobile node, D == 0) are preserved by construction. Returns
    (Jp, Dp) of shape (n_samples, N, N) and (n_samples, N).
    """
    J = np.asarray(J, float); D = np.asarray(D, float)
    mask = J != 0
    fJ = np.ones((n_samples,) + J.shape)
    fJ[:, mask] = rng.lognormal(0.0, sigma_log, size=(n_samples, int(mask.sum())))
    Jp = J[None] * fJ
    Dp = D[None, :] * rng.lognormal(0.0, sigma_log, size=(n_samples, len(D)))
    Dp[:, D == 0] = 0.0
    return Jp, Dp


def _perturb_cloud(J, D, sigma_log, rng, n_samples, kgrid=_CLOUD_KGRID):
    """Vectorized log-normal cloud around a single PHYSICAL (J, D).

    Ported from scripts/exp11_robustness_baseline.py::cloud (lines 60-86), the
    already-validated perturbation model used for the generator-system baseline in
    docs/ROBUSTNESS_MEASUREMENT.md section 4. See _draw_JD_cloud for the perturbation
    itself. Returns both the STRICT criterion (max Re eig(J) < 0, section 3.5) and
    the loose tr(J) < 0 criterion, so the loose criterion's overcount stays visible
    rather than being silently discarded.
    """
    J = np.asarray(J, float); D = np.asarray(D, float)
    n = J.shape[0]
    Jp, Dp = _draw_JD_cloud(J, D, sigma_log, rng, n_samples)

    strict_u = np.linalg.eigvals(Jp).real.max(axis=1) < 0
    loose_u = np.trace(Jp, axis1=1, axis2=2) < 0

    k2 = (kgrid ** 2)[:, None, None]
    Dm = np.zeros((n_samples, n, n)); Dm[:, range(n), range(n)] = Dp
    sig = np.linalg.eigvals(Jp[:, None, :, :] - k2[None] * Dm[:, None, :, :]).real.max(axis=2)
    ipos = sig[:, 1:].argmax(axis=1) + 1
    unst = sig[np.arange(n_samples), ipos] > 1e-9
    kst = kgrid[ipos]

    keep = strict_u & unst
    keep_loose = loose_u & unst
    return dict(
        frac_strict=float(keep.mean()),
        frac_loose=float(keep_loose.mean()),
        frac_loose_only=float((keep_loose & ~keep).mean()),
        n_strict=int(keep.sum()),
        kstar_mean=float(kst[keep].mean()) if keep.any() else float("nan"),
        kstar_std=float(kst[keep].std()) if keep.any() else float("nan"),
    )


def _model_JD(model, xstar=None, D=None):
    """(J, D) at the model's own steady state, as numpy. Solves for x* once if xstar is
    not supplied (a caller that already has it, e.g. a RecoveryResult, should pass it to
    avoid a redundant Newton solve).

    ALWAYS PASS `D` FOR A RECOVERED MODEL (D-EVID-14). `recover(nondim=True)` optimises on
    the unit box, where the learned parameter is D_hat = D/L**2, and it returns the model
    UNCHANGED — so `model.D` is dimensionless there while `RecoveryResult.D_phys` holds the
    physical value. Everything in this module works at the PHYSICAL length scale, so
    reading `model.D` on that path is wrong by L**2 (3600x at L=60) and produces a
    plausible wrong number rather than an error.
    """
    if xstar is None:
        from ..losses.terms import steady_state
        xs, _ = steady_state(model)
    else:
        xs = torch.as_tensor(np.asarray(xstar, float))
    J = model.jacobian(xs, create_graph=False).detach().cpu().numpy()
    if D is None:
        D = model.D.detach().cpu().numpy()
    return J, np.asarray(D, dtype=float)


def robustness_cloud(model, n_samples=200, sigma_log=0.1, seed=0, xstar=None, D=None):
    """Log-normal multiplicative perturbation cloud around the recovered model's
    PHYSICAL (J, D); fraction still strictly Turing, and k* spread among survivors.

    J, D are computed ONCE at the model's own steady state, then perturbed directly
    (see _perturb_cloud) and all n_samples draws are evaluated with batched numpy
    linear algebra. This fixes the defects recorded in docs/ROBUSTNESS_MEASUREMENT.md
    section 3, together:
      3.1/3.2 the perturbation is multiplicative on the PHYSICAL J, D entries, not raw
        theta behind three different link functions, so a given sigma_log means the
        same thing everywhere and a lognormal factor (always positive) cannot flip a
        parameter's sign.
      3.3 dispersion_backend is moot: there is no per-draw model rebuild, so there is
        no backend selection left to silently drop.
      3.4 all n_samples draws are vectorized numpy, not a Python loop of per-draw
        model construction + Newton solve.
      3.5 the reported frac_turing/n_turing use the STRICT criterion; frac_loose and
        frac_loose_only are also returned so the loose criterion's overcount (median
        0.5% of draws at 10% noise, 9.8% at 20%, up to ~70% for a single sample) stays
        visible instead of being silently used.
    """
    J, D = _model_JD(model, xstar, D)
    rng = np.random.default_rng(seed)
    res = _perturb_cloud(J, D, sigma_log, rng, n_samples)
    return dict(frac_turing=res["frac_strict"], n=n_samples, n_turing=res["n_strict"],
                kstar_mean=res["kstar_mean"], kstar_std=res["kstar_std"],
                frac_strict=res["frac_strict"], frac_loose=res["frac_loose"],
                frac_loose_only=res["frac_loose_only"])


def robustness_volumes(model, n_samples=200, seed=0, xstar=None, D=None):
    """Strict-criterion local Turing volume at Tica et al.'s four noise levels (1%,
    4.8% — their measured experimental parameter CV, 10%, 20%), against the recovered
    model's own PHYSICAL (J, D). Read against the generator-system baseline in
    docs/ROBUSTNESS_MEASUREMENT.md section 4.2, never against zero:
        1%: median 1.000, mean 0.993, worst sample 0.538
        4.8%: median 1.000, mean 0.954, worst sample 0.385
        10%: median 0.935, mean 0.879, worst sample 0.232
        20%: median 0.755, mean 0.746, worst sample 0.220
    That baseline used a DIFFERENT perturbation model (directly on an answer-key J, D)
    from Tica's own (on kinetic parameters, re-deriving x* and J); the recovered-model
    number here uses the same direct-(J, D) perturbation as the baseline, so the two
    are comparable to each other, not to Tica's raw percentages.
    """
    J, D = _model_JD(model, xstar, D)
    out = {}
    for label, sigma in ROBUSTNESS_SIGMA_LEVELS.items():
        rng = np.random.default_rng(seed)
        res = _perturb_cloud(J, D, sigma, rng, n_samples)
        out[f"turing_volume_{label}"] = res["frac_strict"]
    return out
