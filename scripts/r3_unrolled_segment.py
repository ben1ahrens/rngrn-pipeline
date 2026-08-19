"""r3_unrolled_segment.py — the gradient-error-vs-segment-length curve (PLAN_redesign_R3 T12).

`docs/REDESIGN_rngrn.md` §8 item 14 leaves the truncated-unrolled path's SEGMENT LENGTH
UNCALIBRATED and assigns its calibration to R3. This script is that measurement: the
unrolled gradient of `rngrn.unrolled.unrolled_relax` against central finite differences, as a
function of segment length, at the n=96 commensurate solve box of §4.3.

METHOD (D1's protocol, `scripts/diag_fft_d1.py` check 3, applied to the unrolled path):

* Fixture: the tracked known-Turing checkpoint the D1/D2 diagnostics used. Its dispersion
  argmax k-hat sets the commensurate box, `solve_box.geometry(k-hat)` -> p=8, n=96.
* dt = 0.2/|eig(J)|_max at the base theta and is HELD FIXED across every finite difference.
  This is not a convenience: dt, the box, and the warm state are all DETACHED in the
  analytic path (§4.3 — geometry is not differentiated), so an FD that let them move would
  be differencing a different function and would disagree for a reason that is not an error.
* For each segment length S: the analytic directional derivative g.v from one backward pass,
  against central differences (theta +/- eps*v, same X0/dt/box) over eps in
  {1e-3, 1e-4, 1e-5, 1e-6}, BEST eps per direction, worst over `--ndir` random unit
  directions in the full theta space. This is D1's acceptance shape and its 1e-4 tolerance.
* TWO warm states, because they answer different questions:
  - `saturated` — the operating point the truncated path is designed for (the field after a
    full gradient-free relax on the box);
  - `growth` — a field still in the exponential-growth phase, which is the regime §4.2 warns
    can make gradients explode and is the reason the path is truncated at all.
* Two loss functionals, both PLACEHOLDERS for the trained objective (Task 13/14 own what is
  actually scored on the box; this script owns the gradient path): `amp`, the mean square of
  channel 0, and `log_band_power`, the log of channel-0 power in the annulus
  [0.60, 1.55]*k-hat — `SpectralConfig`'s B_train edges read in the box's own scale-free
  units. `log_band_power` exercises the FFT and the band, but it is NOT `losses/spectral`'s
  RAPS shape term and must not be reported as one.

WHAT THIS DOES NOT MEASURE: whether the unrolled gradient is the RIGHT gradient for training
— it is FD-faithful to the finite-time relax map it defines, which is a different object from
the steady state the adjoint path differentiates. Task 14's A/B against the adjoint is where
those two are compared.

Run — this is the EXACT invocation that regenerated the tracked JSON, and every default below
is the value it was produced with. CPU, ~14 min; no guard needed (this is not a trainer):

    .venv/bin/python scripts/r3_unrolled_segment.py \
        --out experiments/redesign_r3/unrolled_segment/results/curve.json

The ladder MUST reach 2048, because the reference point every `cosine_to_reference` and
`rel_norm_gap` is measured against is the longest length in the ladder. A shorter default
would silently redefine every convergence number against a different reference — which is
exactly what the committed default did before this was fixed. So: the default ladder ends at
2048, and the JSON records both the resolved `argv` and an explicit `reference_segment_steps`
per arm, so a reader never has to infer the reference from the invocation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

import rngrn.solve_box as sb
from rngrn.forward import relax_to_pattern_torch
from rngrn.losses.spectral import SpectralConfig
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN, THETA_NAMES
from rngrn.unrolled import unrolled_relax

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
EPS_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6)
FD_TOL = 1e-4          # D1's acceptance tolerance, inherited unchanged


def load_fixture() -> RNGRN:
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m


def _loss_amp(X: torch.Tensor) -> torch.Tensor:
    return (X[:, 0] ** 2).mean()


def _band_mask(n: int, L: float, khat: float) -> torch.Tensor:
    """Half-spectrum annulus |k|/k-hat in [b_lo, b_hi], SpectralConfig's B_train edges."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k[: n // 2 + 1], k, indexing="xy")
    r = np.sqrt(kx ** 2 + ky ** 2) / khat
    mask = (r >= SpectralConfig.b_lo) & (r <= SpectralConfig.b_hi)
    if not mask.any():
        raise ValueError("band annulus is empty on this box — geometry is wrong")
    return torch.from_numpy(mask)


def make_losses(n: int, L: float, khat: float):
    mask = _band_mask(n, L, khat)

    def log_band_power(X: torch.Tensor) -> torch.Tensor:
        P = torch.fft.rfft2(X[:, 0], dim=(-2, -1)).abs() ** 2
        return torch.log(P[..., mask].sum())

    return {"amp": _loss_amp, "log_band_power": log_band_power}


def theta_flat_grad(model: RNGRN, loss_fn, X0, n, L, dt, S) -> np.ndarray:
    out = unrolled_relax(model, X0, n, L, dt, segment_steps=S, checkpoint_every=1)
    params = [getattr(model, nm) for nm in THETA_NAMES]
    g = torch.autograd.grad(loss_fn(out), params)
    return torch.cat([x.reshape(-1) for x in g]).detach().numpy()


def perturbed_value(v, sgn, eps, losses, X0, n, L, dt, S):
    """Loss values of every functional at theta + sgn*eps*v, sharing one unrolled forward.

    The base theta is re-read from the tracked checkpoint per point (D1's own pattern), so a
    perturbation cannot accumulate across points. Returns None if that segment blew up.
    """
    m2 = load_fixture()
    off = 0
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(m2, nm)
            sz = p.numel()
            p.add_(sgn * eps * torch.from_numpy(v[off:off + sz]).reshape(p.shape))
            off += sz
    try:
        with torch.no_grad():
            u = unrolled_relax(m2, X0, n, L, dt, segment_steps=S)
    except RuntimeError:
        return None
    return {name: float(fn(u)) for name, fn in losses.items()}


def measure_arm(arm: str, X0: torch.Tensor, model: RNGRN, losses, n, L, dt,
                ladder, ndir, seed) -> list[dict]:
    rows = []
    sizes = [getattr(model, nm).numel() for nm in THETA_NAMES]
    total = sum(sizes)
    rng = np.random.default_rng(seed)
    dirs = []
    for _ in range(ndir):
        v = rng.standard_normal(total)
        dirs.append(v / np.linalg.norm(v))
    for S in ladder:
        t0 = time.perf_counter()
        row: dict = {"arm": arm, "segment_steps": int(S)}
        try:
            grads = {name: theta_flat_grad(model, fn, X0, n, L, dt, S)
                     for name, fn in losses.items()}
        except RuntimeError as exc:
            row.update(forward_blew_up=True, note=str(exc)[:200],
                       seconds=time.perf_counter() - t0)
            rows.append(row)
            print(f"[{arm} S={S}] FORWARD BLEW UP")
            continue
        row["forward_blew_up"] = False
        row["grad_norm"] = {k: float(np.linalg.norm(g)) for k, g in grads.items()}
        row["grad_finite"] = {k: bool(np.isfinite(g).all()) for k, g in grads.items()}
        # the FULL gradient (theta is 36 numbers here), so the segment-length recommendation
        # can be made on the gradient's DIRECTION as well as its norm without a re-run
        row["grad"] = {k: g.tolist() for k, g in grads.items()}
        per_loss: dict = {name: {"per_dir": []} for name in losses}
        for d, v in enumerate(dirs):
            vals: dict = {}
            for eps in EPS_SWEEP:
                pair = {}
                for sgn in (+1.0, -1.0):
                    pair[sgn] = perturbed_value(v, sgn, eps, losses, X0, n, L, dt, S)
                vals[eps] = pair
            for name in losses:
                d_an = float(grads[name] @ v)
                best = None
                per_eps = {}
                for eps in EPS_SWEEP:
                    p, m_ = vals[eps][+1.0], vals[eps][-1.0]
                    if p is None or m_ is None:
                        per_eps[str(eps)] = {"fd_blew_up": True}
                        continue
                    d_fd = (p[name] - m_[name]) / (2.0 * eps)
                    rel = abs(d_fd - d_an) / max(abs(d_fd), abs(d_an), 1e-300)
                    per_eps[str(eps)] = {"d_fd": d_fd, "rel_err": rel}
                    if best is None or rel < best[1]:
                        best = (eps, rel, d_fd)
                per_loss[name]["per_dir"].append(
                    {"dir": d, "d_analytic": d_an,
                     "best_eps": None if best is None else best[0],
                     "rel_err": None if best is None else best[1],
                     "d_fd": None if best is None else best[2],
                     "per_eps": per_eps})
        for name in losses:
            errs = [r["rel_err"] for r in per_loss[name]["per_dir"]
                    if r["rel_err"] is not None]
            per_loss[name]["worst_rel_err"] = max(errs) if errs else None
            per_loss[name]["median_rel_err"] = float(np.median(errs)) if errs else None
            per_loss[name]["n_dirs_with_fd"] = len(errs)
        row["losses"] = per_loss
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        print(f"[{arm} S={S}] worst rel err " + ", ".join(
            f"{k} {per_loss[k]['worst_rel_err']!r}" for k in losses)
            + f"  |g| " + ", ".join(f"{k} {row['grad_norm'][k]:.3e}" for k in losses)
            + f"  ({row['seconds']:.1f}s)")
    return rows


def summarise_convergence(rows, losses) -> dict:
    """Each segment length's gradient against the LONGEST measured one, per arm and loss.

    The segment-length recommendation turns on whether a shorter segment gives the SAME
    gradient, which is a statement about direction as much as magnitude — hence cosine
    similarity alongside the relative norm gap. The reference is the longest length in the
    ladder, not a converged limit: it is a measured comparison, not a proof of convergence.
    """
    out: dict = {}
    for arm in sorted({r["arm"] for r in rows}):
        ok = [r for r in rows if r["arm"] == arm and not r["forward_blew_up"]]
        if not ok:
            continue
        ref = max(ok, key=lambda r: r["segment_steps"])
        out[arm] = {"reference_segment_steps": ref["segment_steps"], "per_loss": {}}
        for name in losses:
            gr = np.asarray(ref["grad"][name])
            per = []
            for r in ok:
                g = np.asarray(r["grad"][name])
                per.append({
                    "segment_steps": r["segment_steps"],
                    "cosine_to_reference": float(g @ gr / (np.linalg.norm(g)
                                                           * np.linalg.norm(gr))),
                    "rel_norm_gap": float(np.linalg.norm(g - gr) / np.linalg.norm(gr)),
                    "worst_fd_rel_err": r["losses"][name]["worst_rel_err"],
                })
            out[arm]["per_loss"][name] = per
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ndir", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                    help="segment lengths to measure. The LONGEST is the reference every "
                         "convergence number is quoted against, so shortening this ladder "
                         "changes what those numbers mean; see the module docstring.")
    ap.add_argument("--growth-frac", type=float, default=0.2,
                    help="the growth-phase warm state is the first 5-step chunk whose "
                         "channel-0 amplitude exceeds this fraction of the saturated one")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads; the batched CPU relax is ~215x slower at 14 "
                         "(Task 11 finding), so timings here are pinned at 1")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)

    model = load_fixture()
    xs, ok = steady_state(model)
    assert ok, "steady state failed on the fixture"
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    assert float(sig.max()) > 0, "fixture is not Turing-unstable"
    khat = float(kg[int(sig.argmax())])
    box = sb.geometry(khat)
    n, L = int(box["n"]), float(box["L_solve"])
    dt = 0.2 / float(torch.linalg.eigvals(J).abs().max())
    xstar = xs.detach().numpy()
    losses = make_losses(n, L, khat)

    t0 = time.perf_counter()
    sat = relax_to_pattern_torch(model, xstar, n, L, dt, a.seed,
                                 torch.device("cpu"))[None].detach()
    relax_seconds = time.perf_counter() - t0
    amp_sat = float(sat[0, 0].std(correction=0))
    print(f"saturated in {relax_seconds:.1f}s, channel-0 amplitude {amp_sat:.4g}")

    # The growth-phase warm state: probed from the SAME IC in 5-step chunks (this fixture
    # saturates inside ~100 steps on this box, so a coarser probe lands past the growth phase
    # entirely — measured: at 100 steps the amplitude is already 77% of saturated). Taken at
    # the first chunk whose channel-0 amplitude exceeds `--growth-frac` of the saturated one,
    # i.e. still climbing, which is the regime §4.2 says can explode.
    rng = np.random.default_rng(a.seed)
    X = torch.from_numpy(
        xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, n, n)))[None]
    amp_traj = [float(X[0, 0].std(correction=0))]
    growth, growth_steps = None, 0
    with torch.no_grad():
        for c in range(1, 4001):
            X = unrolled_relax(model, X, n, L, dt, segment_steps=5)
            amp_traj.append(float(X[0, 0].std(correction=0)))
            if amp_traj[-1] > a.growth_frac * amp_sat:
                growth, growth_steps = X.detach(), 5 * c
                break
    if growth is None:
        raise RuntimeError(
            f"never reached {a.growth_frac:g} of the saturated amplitude {amp_sat:.4g}")
    amp_growth = float(growth[0, 0].std(correction=0))
    print(f"growth-phase state at {growth_steps} steps, amplitude {amp_growth:.4g} "
          f"({amp_growth / amp_sat:.1%} of saturated)")

    rows = []
    for arm, X0 in (("saturated", sat), ("growth", growth)):
        rows += measure_arm(arm, X0, model, losses, n, L, dt,
                            a.ladder, a.ndir, a.seed)
    convergence = summarise_convergence(rows, losses)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "what": "unrolled-vs-FD gradient error against segment length "
                "(REDESIGN_rngrn.md §4.2, §8 item 14; PLAN_redesign_R3 Task 12 Step 5)",
        "method": {
            "checkpoint": str(CKPT.relative_to(REPO)),
            "box": box, "khat": khat, "dt": dt, "L_solve": L, "n": n,
            "dt_is_held_fixed_across_fd": True,
            "eps_sweep": list(EPS_SWEEP), "fd_tol_D1": FD_TOL,
            "ndir": a.ndir, "seed": a.seed, "ladder": a.ladder,
            # self-describing provenance: the exact argv this file was produced by, and the
            # reference segment length every cosine_to_reference / rel_norm_gap is quoted
            # against. Without the latter a reader has to infer it from the ladder, and a
            # ladder default that disagreed with the committed run is exactly how that goes
            # wrong.
            "argv": list(sys.argv),
            "reference_segment_steps": max(a.ladder),
            "checkpoint_every_analytic": 1,
            "losses": {"amp": "mean(u_0^2)",
                       "log_band_power": "log sum_{|k|/khat in [%.2f, %.2f]} |FFT(u_0)|^2"
                                         % (SpectralConfig.b_lo, SpectralConfig.b_hi)},
            "warm_states": {"saturated": {"relax_seconds": relax_seconds,
                                          "amplitude": amp_sat},
                            "growth": {"steps": growth_steps, "amplitude": amp_growth,
                                       "frac_of_saturated": amp_growth / amp_sat,
                                       "growth_frac_requested": a.growth_frac,
                                       "amp_every_5_steps": amp_traj}},
            "device": "cpu", "threads": a.threads,
            "torch": torch.__version__,
        },
        "convergence_vs_longest_segment": convergence,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
