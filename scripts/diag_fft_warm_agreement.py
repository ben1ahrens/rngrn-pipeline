"""diag_fft_warm_agreement.py — fresh-vs-warm branch agreement under theta drift
(D-FFT-11 owner flag 3 evidence).

DIAGNOSTIC SCRIPT (no training, no payload.h5 — the known-Turing theta comes from a
tracked RECOVERED checkpoint under experiments/, i.e. a recovery OUTPUT, never an answer
key). Float64, CPU. Not FORBIDDEN — no payload.h5 opened here (CLAUDE.md §5).

Owner flag under test (docs/DECISIONS.md D-FFT-11, "Owner flags carried forward",
numerics-review finding): "the warm-started Newton chain is BRANCH CONTINUATION with no
dynamical-stability check — over a long drifting theta trajectory it can track a
patterned F=0 branch that has become dynamically unstable, and residual + amplitude
floor both still pass. D1's warm-Newton evidence covers +/-eps FD pairs about a fixed
base, not a long chain; a periodic fresh re-relax cadence or a linear-stability spot
check on u* is an owner decision before any calibration run."

This script drives one `PatternSolver` through a random walk on theta (a proxy for an
Adam trajectory) and at each in-Turing-regime step measures:
  (a) AGREEMENT — do the warm chain and an independent fresh solve (new solver, new
      relax-from-noise seed) land on translation-invariant-equivalent patterns:
      amplitude ratio, k* one-radial-bin agreement (2*pi/L, the repo's own k* tolerance),
      band-limited log-RAPS shape distance?
  (b) STABILITY — perturbed slightly and integrated forward ~1000 ETDRK4 steps, does the
      WARM branch's u* return to itself (dynamically stable) or depart (evidence the warm
      chain is tracking an unstable branch — exactly the flag-3 failure mode)?

--sigma (default 0.01; per-step additive gaussian std, RELATIVE: std = sigma *
leaf.abs().mean()) is an UNCALIBRATED proxy for one Adam step's theta movement — no
measurement ties this scale to a real Adam trajectory. The real evidence run uses
sigma in {0.005, 0.01, 0.02} (three separate invocations of this script).

The band-limited log-RAPS shape distance is the spec_shape-form arithmetic ported from
`scripts/diag_fft_d1.py::make_loss_terms::spec_shape` (sum of squared log differences of
each field's OWN-band-sum-normalised RAPS on [0.60, 1.55]*k*_ref — the CLOSED band,
D-FFT-9 closure 1), evaluated numerically (numpy `observables.raps`, no autodiff needed
here). k*_ref is the BASE (undrifted) fixture's linear k* — fixed for the whole run, so
the band tracks the fixture, not a per-step-drifted estimate.

STABILITY_SHAPE_FLOOR (0.39) is D3's measured PATCH-CONTROL floor
(docs/DIAGNOSTICS_fft.md D3: spec_shape-form log-RAPS distance on the *provisional*
[0.5, 1.5]*k* band, 4-quadrant patches of turing_labyrinth/sample_0000, mean 0.389,
range 0.122-0.621) — reused here as an approximate "same pattern, up to normal
patch-to-patch estimation variation" bar. CAVEAT stated once, honestly: D3 measured that
number on the (0.5, 1.5) provisional band; this script's shape distances (including the
stability check) use the CLOSED (0.60, 1.55) band throughout for internal consistency,
so 0.39 is an approximately-reused floor, not a band-matched calibration.

Usage:
    .venv/bin/python scripts/diag_fft_warm_agreement.py \
        [--ckpt experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/checkpoints/model.pt] \
        [--n 96] [--steps 12] [--sigma 0.01] [--periods 8.0] [--seed 0] \
        [--out experiments/diag_fft/warm]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.model import RNGRN, THETA_NAMES                       # noqa: E402
from rngrn.losses.terms import steady_state                      # noqa: E402
from rngrn.forward import PatternSolver                          # noqa: E402
from rngrn.eval.numerics import integrate_etdrk4_rfft             # noqa: E402
from rngrn.eval.rollout import _reaction_np_builder                # noqa: E402
from rngrn import observables as obs                               # noqa: E402
from rngrn.utils import git_revision                                # noqa: E402

torch.set_default_dtype(torch.float64)

BAND_LO, BAND_HI = 0.60, 1.55        # D-FFT-9 closure 1
STABILITY_SHAPE_FLOOR = 0.39         # D3 patch-control floor — see module docstring
STABILITY_AMP_FRAC_TOL = 0.20
STABILITY_NSTEPS = 1000
STABILITY_NOISE_FRAC = 1e-3          # of the warm branch's own amplitude


# ------------------------------------------------------------------ fixture helpers
def load_known_turing_model(ckpt_path: str) -> RNGRN:
    """A recovered checkpoint (dimensional path only). Mirrors diag_fft_d1.py verbatim —
    fail loud rather than trust the checkpoint's own recorded label."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck.get("nondim"):
        raise ValueError(f"{ckpt_path} was trained on the nondim path — its D is D/L^2 "
                         "and this script assumes physical D; pick a dimensional run")
    model = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    model.load_state_dict(ck["state_dict"])
    return model


def perturb_theta(model: RNGRN, rng: np.random.Generator, sigma: float) -> None:
    """Additive gaussian perturbation on every theta leaf, in place, no_grad,
    CUMULATIVE across calls (a random-walk drift). std = sigma * leaf.abs().mean()
    (UNCALIBRATED Adam-step-scale proxy, see module docstring)."""
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(model, nm)
            std = sigma * float(p.abs().mean())
            p.add_(torch.from_numpy(rng.standard_normal(tuple(p.shape))) * std)


def band_shape(field: np.ndarray, L: float, kstar_ref: float) -> np.ndarray:
    """Own-band-sum-normalised RAPS of a (H, W) real field on [BAND_LO, BAND_HI]*kstar_ref.
    Fails loud on an empty or non-finite/zero band (CLAUDE.md §4)."""
    kcent, power, _ = obs.raps(field, L=L)
    mask = (kcent >= BAND_LO * kstar_ref) & (kcent <= BAND_HI * kstar_ref)
    if not mask.any():
        raise ValueError(
            f"band [{BAND_LO}, {BAND_HI}]*k*_ref={kstar_ref:.4g} contains no RAPS bins — "
            f"k_centers span [{kcent.min():.4g}, {kcent.max():.4g}]")
    band = power[mask]
    total = band.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"band RAPS power is zero or non-finite (sum={total!r})")
    return band / total


def shape_distance(field_a: np.ndarray, field_b: np.ndarray, L: float,
                   kstar_ref: float) -> float:
    """spec_shape-form distance: sum of squared log differences of the two fields' own-
    band-normalised RAPS. Both fields must share (n, L) so the bin grids line up."""
    sa = band_shape(field_a, L, kstar_ref)
    sb = band_shape(field_b, L, kstar_ref)
    if sa.shape != sb.shape:
        raise ValueError(
            f"band shapes disagree in bin count ({sa.shape[0]} vs {sb.shape[0]}) — "
            "fields must share the same grid n and box L")
    return float(np.sum((np.log(sa + 1e-300) - np.log(sb + 1e-300)) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--periods", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/diag_fft/warm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = load_known_turing_model(a.ckpt)
    xs0, ok0 = steady_state(model)
    if not ok0:
        raise RuntimeError("steady state failed on the fixture checkpoint")
    J0 = model.jacobian(xs0, create_graph=False).detach()
    kg_lin = torch.linspace(1e-3, 10.0, 2000)
    sig0 = model.dispersion(xs0, kg_lin, J=J0).detach()
    sig_max0 = float(sig0.max())
    if not (sig_max0 > 0):
        raise RuntimeError(f"fixture is not Turing-unstable: sig_max={sig_max0}")
    kstar_ref = float(kg_lin[int(sig0.argmax())])
    L = a.periods * 2.0 * np.pi / kstar_ref
    one_bin = 2.0 * np.pi / L
    print(f"fixture: sig_max={sig_max0:.4f} k*_lin={kstar_ref:.4f} L={L:.2f} "
          f"one_bin(2pi/L)={one_bin:.4f}", flush=True)

    # 400-point kgrid around the fixture's k*, mirroring recover.py::_kgrid_for's
    # scale-free-floor convention (kmin relative to kstar_ref, not an absolute constant).
    kmin = kstar_ref / 50.0 + 1e-3
    kmax = max(8.0 * kstar_ref, 2.0 * kmin)
    kgrid = torch.linspace(kmin, kmax, 400)

    n = a.n
    warm_solver = PatternSolver(model, n=n, L=L, seed=a.seed)
    rng = np.random.default_rng(a.seed)

    rows: list[dict] = []
    n_turing = n_warm_ok = n_fresh_ok = n_kstar_agree = n_stability_departed = 0
    shape_dists: list[float] = []

    for k in range(1, a.steps + 1):
        perturb_theta(model, rng, a.sigma)

        xs, ok = steady_state(model)
        if not ok:
            rows.append(dict(step=k, in_turing=False, reason="steady_state_failed"))
            print(f"step {k}: steady state failed — theta walked off the model's "
                  f"solvable region, skipping solves this step", flush=True)
            continue
        J = model.jacobian(xs, create_graph=False).detach()
        sig = model.dispersion(xs, kgrid, J=J).detach()
        sig_max_pos = float(sig.max())
        if not (sig_max_pos > 0):
            rows.append(dict(step=k, in_turing=False, sig_max_pos=sig_max_pos))
            print(f"step {k}: sig_max_pos={sig_max_pos:.4e} <= 0 — left the Turing "
                  f"regime, skipping solves (training would not ignite either)",
                  flush=True)
            continue

        n_turing += 1
        row: dict = dict(step=k, in_turing=True, sig_max_pos=sig_max_pos)

        t0 = time.perf_counter()
        u_warm, reason_warm = warm_solver.solve()
        warm_s = time.perf_counter() - t0
        row.update(warm_reason=reason_warm, warm_s=warm_s)
        if reason_warm == "ok":
            n_warm_ok += 1

        fresh_solver = PatternSolver(model, n=n, L=L, seed=a.seed + 1000 + k)
        t0 = time.perf_counter()
        u_fresh, reason_fresh = fresh_solver.solve()
        fresh_s = time.perf_counter() - t0
        row.update(fresh_reason=reason_fresh, fresh_s=fresh_s)
        if reason_fresh == "ok":
            n_fresh_ok += 1

        if reason_warm == "ok" and reason_fresh == "ok":
            u_warm_np = u_warm.detach().numpy()
            u_fresh_np = u_fresh.detach().numpy()
            amp_warm = float(u_warm_np[0].std())
            amp_fresh = float(u_fresh_np[0].std())
            kstar_warm = float(obs.kstar_of(u_warm_np[0], L=L))
            kstar_fresh = float(obs.kstar_of(u_fresh_np[0], L=L))
            kstar_agree = bool(abs(kstar_warm - kstar_fresh) <= one_bin)
            sdist = shape_distance(u_warm_np[0], u_fresh_np[0], L, kstar_ref)
            if kstar_agree:
                n_kstar_agree += 1
            shape_dists.append(sdist)
            row.update(amp_warm=amp_warm, amp_fresh=amp_fresh,
                       amp_ratio_warm_fresh=amp_warm / amp_fresh,
                       kstar_warm=kstar_warm, kstar_fresh=kstar_fresh,
                       kstar_one_bin_agree=kstar_agree, shape_dist_warm_fresh=sdist)
            print(f"step {k}: warm {warm_s:.2f}s fresh {fresh_s:.2f}s "
                  f"amp_ratio(w/f)={amp_warm / amp_fresh:.3f} "
                  f"kstar(w/f)={kstar_warm:.3f}/{kstar_fresh:.3f} "
                  f"1bin_agree={kstar_agree} shape_dist={sdist:.3f}", flush=True)

            # stability check of the WARM branch: perturb its u* slightly, integrate
            # forward, and see whether it relaxes back or departs (flag-3 failure mode).
            jac_rate = float(torch.linalg.eigvals(J).abs().max())
            dt = 0.2 / jac_rate
            D_np = model.D.detach().numpy()
            reaction_np = _reaction_np_builder(model)
            noise = STABILITY_NOISE_FRAC * amp_warm
            X0 = u_warm_np + noise * rng.standard_normal(u_warm_np.shape)
            Xf, blew = integrate_etdrk4_rfft(X0, D_np, reaction_np, n, L, dt,
                                             STABILITY_NSTEPS)
            if blew:
                departed = True
                row.update(stability_blew_up=True, stability_departed=True)
            else:
                sdist_stab = shape_distance(Xf[0], u_warm_np[0], L, kstar_ref)
                amp_stab = float(Xf[0].std())
                amp_frac = abs(amp_stab - amp_warm) / amp_warm
                departed = not (sdist_stab < STABILITY_SHAPE_FLOOR
                               and amp_frac < STABILITY_AMP_FRAC_TOL)
                row.update(stability_blew_up=False, stability_shape_dist=sdist_stab,
                          stability_amp_frac=amp_frac, stability_departed=departed)
            if departed:
                n_stability_departed += 1
            print(f"  stability check: {'DEPARTED' if departed else 'returned'} "
                  f"(blew_up={blew})", flush=True)
        else:
            print(f"step {k}: warm={reason_warm} fresh={reason_fresh} — skipping "
                  f"agreement/stability checks", flush=True)

        rows.append(row)

    summary = dict(
        steps=a.steps, n_in_turing_regime=n_turing, n_warm_ok=n_warm_ok,
        n_fresh_ok=n_fresh_ok, n_kstar_one_bin_agree=n_kstar_agree,
        max_shape_dist_warm_fresh=max(shape_dists) if shape_dists else None,
        median_shape_dist_warm_fresh=float(np.median(shape_dists)) if shape_dists
        else None, n_stability_departed=n_stability_departed,
        stability_shape_floor=STABILITY_SHAPE_FLOOR,
        stability_amp_frac_tol=STABILITY_AMP_FRAC_TOL,
        stability_nsteps=STABILITY_NSTEPS,
    )
    report = dict(
        ckpt=a.ckpt, args=vars(a), git_rev=git_revision(),
        torch_version=torch.__version__, kstar_ref=kstar_ref, L=L,
        sig_max_fixture=sig_max0, one_bin=one_bin, band=(BAND_LO, BAND_HI),
        rows=rows, summary=summary,
    )
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nsummary:", flush=True)
    for kk, vv in summary.items():
        print(f"  {kk}: {vv}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
