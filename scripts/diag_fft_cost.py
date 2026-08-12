"""diag_fft_cost.py — forward-solve cost vs grid size (D-FFT-11 owner flag 1 evidence).

DIAGNOSTIC SCRIPT (no training, no payload.h5 — the known-Turing theta comes from a
tracked RECOVERED checkpoint under experiments/, i.e. a recovery OUTPUT, never an answer
key). Float64, CPU. Not FORBIDDEN — no payload.h5 opened here (CLAUDE.md §5).

Owner flag under test (docs/DECISIONS.md D-FFT-11, "Owner flags carried forward"):
"the forward-solve cost at training grids is UNMEASURED (per-solve ~3-9s at 64^2, an
unrecorded test timing; a relax step-budget cap is an owner call before calibration)".
This script measures, per grid size n: a FRESH solve (relax-from-noise + Newton), three
WARM solves under a small theta perturbation (an Adam-step-scale proxy), and the cost of
ONE backward() through `rngrn.forward.PatternSolve` (one minimal-norm adjoint solve) on a
two-term spectral loss — plus the process RSS high-water mark, so the owner can size a
relax step-budget cap and a training-grid choice against real numbers instead of the
64^2 test timing.

`forward.py` is used exactly as shipped (`PatternSolver`); this script does not modify
it and does not have access to a relax-vs-Newton time breakdown (PatternSolver does not
expose one) — only solve() totals are recorded, as PatternSolver's own docstring
anticipates ("A relax step-budget cap is an owner decision before any calibration run").

`--warm-eps`'s scale (default 1e-3, relative: std = warm_eps * leaf.abs().mean()) is an
UNCALIBRATED proxy for one Adam step's parameter movement — no measurement ties it to a
real Adam trajectory; it exists only to exercise PatternSolver's warm-start (Newton-only)
path under a small, plausible perturbation.

Each grid n reloads the fixture checkpoint fresh (rather than reusing the previous grid's
already-perturbed model): the warm-eps perturbations applied while timing one grid must
not leak into the next grid's FRESH-solve measurement, or "fresh" would silently mean
"fresh relax from a drifted theta" for every grid after the first.

Usage:
    .venv/bin/python scripts/diag_fft_cost.py \
        [--ckpt experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/checkpoints/model.pt] \
        [--grids 96,128,192,256] [--periods 8.0] [--seed 0] [--warm-eps 1e-3] \
        [--out experiments/diag_fft/cost]
"""
from __future__ import annotations
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.model import RNGRN, THETA_NAMES                       # noqa: E402
from rngrn.losses.terms import steady_state                      # noqa: E402
from rngrn.forward import PatternSolver                          # noqa: E402
from rngrn.losses.spectral import (SpectralConfig, build_frame_targets,  # noqa: E402
                                   spectral_terms)
from rngrn import observables as obs                              # noqa: E402
from rngrn.utils import git_revision                               # noqa: E402

torch.set_default_dtype(torch.float64)


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


def kstar_lin_and_L(model: RNGRN, periods: float) -> tuple[float, float, float]:
    """The model's own linear k* and the box size it implies (diag_fft_d1.py::main
    convention: L = periods * 2*pi / k*_lin)."""
    xs, ok = steady_state(model)
    if not ok:
        raise RuntimeError("steady state failed on the fixture checkpoint")
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    sig_max = float(sig.max())
    if not (sig_max > 0):
        raise RuntimeError(f"fixture is not Turing-unstable: sig_max={sig_max}")
    kstar_lin = float(kg[int(sig.argmax())])
    L = periods * 2.0 * np.pi / kstar_lin
    return kstar_lin, L, sig_max


def perturb_theta(model: RNGRN, rng: np.random.Generator, eps: float) -> None:
    """Additive gaussian perturbation on every theta leaf, in place, no_grad.
    std = eps * leaf.abs().mean() (UNCALIBRATED Adam-step-scale proxy, see module
    docstring)."""
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(model, nm)
            std = eps * float(p.abs().mean())
            p.add_(torch.from_numpy(rng.standard_normal(tuple(p.shape))) * std)


def maxrss_gib() -> float:
    """Process RSS high-water mark. ru_maxrss is KB on Linux — meaningful cumulatively
    across ASCENDING grids (it never decreases), not per-grid in isolation."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--grids", default="96,128,192,256")
    ap.add_argument("--periods", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-eps", type=float, default=1e-3)
    ap.add_argument("--warm-count", type=int, default=3,
                    help="warm solves per grid; 0 skips them. Added 2026-08-12 after the "
                         "first run measured warm[1]=5030s at 96^2 (vs fresh 938s) — a "
                         "full warm sweep at 128-256^2 would take days")
    ap.add_argument("--out", default="experiments/diag_fft/cost")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    grids = [int(x) for x in a.grids.split(",")]
    if grids != sorted(grids):
        raise ValueError(
            f"--grids must be ascending ({grids}) — the running RSS high-water mark is "
            "recorded per grid and is only meaningful read that way")

    base_model = load_known_turing_model(a.ckpt)
    kstar_lin, L, sig_max = kstar_lin_and_L(base_model, a.periods)
    print(f"fixture: sig_max={sig_max:.4f} k*_lin={kstar_lin:.4f} L={L:.2f}", flush=True)
    del base_model  # per-grid loop reloads its own copy

    cfg = SpectralConfig()
    report: dict = dict(
        ckpt=a.ckpt, args=vars(a), git_rev=git_revision(),
        torch_version=torch.__version__, kstar_lin=kstar_lin, L=L, sig_max=sig_max,
        grids={},
    )

    for n in grids:
        print(f"=== grid n={n} ===", flush=True)
        model = load_known_turing_model(a.ckpt)
        rng = np.random.default_rng(a.seed * 1000 + n)
        solver = PatternSolver(model, n=n, L=L, seed=a.seed)

        # 1. fresh solve
        t0 = time.perf_counter()
        u, reason = solver.solve()
        fresh_s = time.perf_counter() - t0
        if reason != "ok":
            raise RuntimeError(f"n={n}: fresh solve did not pattern (reason={reason})")
        fresh_amp = float(u[0].detach().std())
        fresh_kstar = float(obs.kstar_of(u[0].detach().numpy(), L=L))
        rss = maxrss_gib()
        print(f"  fresh: {fresh_s:.2f}s amp={fresh_amp:.4f} k*={fresh_kstar:.3f} "
              f"maxrss={rss:.3f} GiB", flush=True)

        # 2. warm solves (--warm-count, cumulative theta drift)
        warm_s: list[float] = []
        for k in range(1, a.warm_count + 1):
            perturb_theta(model, rng, a.warm_eps)
            t0 = time.perf_counter()
            u, reason = solver.solve()
            dt_s = time.perf_counter() - t0
            if reason != "ok":
                raise RuntimeError(
                    f"n={n}: warm solve {k} did not pattern (reason={reason})")
            warm_s.append(dt_s)
            rss = maxrss_gib()
            print(f"  warm[{k}]: {dt_s:.2f}s maxrss={rss:.3f} GiB", flush=True)

        # 3. adjoint timing: one more (still-warm) solve for a differentiable u, a
        # cheap two-term spectral loss built from u's OWN detached stats (no observed
        # frame involved — this is a cost probe, not a fit), then time ONLY backward().
        u, reason = solver.solve()
        if reason != "ok":
            raise RuntimeError(
                f"n={n}: pre-adjoint solve did not pattern (reason={reason})")
        kstar_obs = float(obs.kstar_of(u[0].detach().numpy(), L=L))
        targets = build_frame_targets(u.detach().numpy(), L, kstar_obs, cfg)
        term_vals, _ = spectral_terms(u, targets, cfg)
        loss = term_vals["spec_shape"] + term_vals["spec_amp_fluct"]
        t0 = time.perf_counter()
        loss.backward()
        adjoint_s = time.perf_counter() - t0
        for nm in THETA_NAMES:
            getattr(model, nm).grad = None
        rss = maxrss_gib()
        print(f"  adjoint: {adjoint_s:.3f}s maxrss={rss:.3f} GiB", flush=True)

        report["grids"][str(n)] = dict(
            fresh_s=fresh_s, fresh_amplitude=fresh_amp, fresh_kstar=fresh_kstar,
            warm_s=warm_s, adjoint_s=adjoint_s, maxrss_gib=rss,
            note="PatternSolver exposes no relax-vs-Newton time breakdown; totals "
                 "only — see module docstring",
        )
        # Crash-safe incremental write: each completed grid cell survives a kill
        # (added 2026-08-12 after the first run had to be stopped mid-sweep with
        # nothing on disk).
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(report, f, indent=2)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nper-n summary:", flush=True)
    warm_hdr = " ".join(f"{f'warm{k}':>9}" for k in range(1, a.warm_count + 1))
    print(f"{'n':>6} {'fresh_s':>9} {warm_hdr} {'adjoint_s':>10} {'maxrss_gib':>11}",
          flush=True)
    for n in grids:
        row = report["grids"][str(n)]
        warm_cols = " ".join(f"{w:9.2f}" for w in row["warm_s"])
        print(f"{n:6d} {row['fresh_s']:9.2f} {warm_cols} "
              f"{row['adjoint_s']:10.3f} {row['maxrss_gib']:11.3f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
