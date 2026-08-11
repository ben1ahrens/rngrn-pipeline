#!/usr/bin/env python
"""stage0_bio_viability.py — is configs/bio_box.yaml Turing-EMPTY?

FEASIBILITY PRECONDITION, not a pre-registered criterion. docs/PREREGISTRATION.md §3.4
demands plausibility_score = 1.0 (every CITED row of configs/bio_box.yaml satisfied) AT
THE SAME TIME as the pattern criteria §3.1-3.3. Nobody had checked that region is
non-empty. If it is empty, 3.4 and 3.1-3.3 are mutually unsatisfiable and every GPU-hour
queued behind them is spent on an empty set. Per PREREGISTRATION §5 an unwelcome answer
here does NOT move the box.

WHAT `mu` IS. scripts/gen_tg3.py:93 writes the generator reaction as
    out[i] = b[i] + V[i]*reg - mu[i]*x[i]
and model.py::_reaction_raw's last line writes the RNGRN's as
    beta + prod - delta * x
so the owner's "mu" IS the RNGRN's `delta`, a CITED bio_box row with bounds [0.4, 5.0].

FOUR SUBCOMMANDS, run in this order:
  part1    draw parameter sets INSIDE the box, measure the strictly-Turing acceptance rate.
           THIS is the measurement; its `n` is its denominator and nothing else touches it.
  harvest  an INDEPENDENTLY seeded pass that keeps drawing until it has enough survivors to
           make parts 2 and 3 a band rather than an anecdote. Its own hit/draw counts are a
           second, independent estimate of the same rate.
  part2    sweep delta (= mu) over the box and one decade either side, on harvest survivors
  part3    roll the survivors out on a 64x64 grid and LOOK at the fields

Part 4 (recovery of five REAL targets with the biological prior active) is a separate
driver: scripts/stage0_part4_prior_recovery.sh. Figures: scripts/stage0_figures.py.

Everything is written to experiments/figures_report/stage0/{arrays/*.npz, *.png}. The
arrays are the audit trail behind every panel; both are tracked in git (see .gitignore).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rngrn.model import RNGRN, BatchedRNGRN                      # noqa: E402
from rngrn.losses.terms import steady_state, steady_state_batched  # noqa: E402
from rngrn.scoring.plausibility import (                         # noqa: E402
    d_ratio_of, load_box, plausibility_report)
from rngrn.eval import rollout                                   # noqa: E402
from rngrn.scoring import morphology                             # noqa: E402

OUT = "experiments/figures_report/stage0"
ARR = os.path.join(OUT, "arrays")
N_SPECIES = 3

# ---------------------------------------------------------------------------------------
# k-grid for the dispersion scan. LOG-spaced over 6.5 decades, not eval/analysis.py's
# linspace(1e-3, 50, 4000): with a near-immobile third node (D ~ 1e-6 x D_max) the branch
# that node dominates only turns over at k ~ sqrt(|J|/D_lo) ~ 1e3, so a grid that stops at
# k=50 cannot see whether it grows there. 700 points over [1e-3, 10^3.5] is ~108 per decade,
# which resolves k* to ~1%. k=0 is prepended and EXCLUDED from the instability search so the
# uniform mode can never be reported as the growing one.
# ---------------------------------------------------------------------------------------
KGRID = np.concatenate([[0.0], np.logspace(-3.0, 3.5, 500)])


def turing_verdict_batch(J, D, kgrid=KGRID):
    """Batched `turing_verdict`. J (B,N,N), D (B,N) -> dict of (B,) arrays.

    Identical arithmetic to the serial version below, which part1 re-runs on a subsample
    and asserts against, so the fast path cannot silently drift from the reference.
    """
    J = np.asarray(J, float); D = np.asarray(D, float)
    B, N = D.shape
    ev0 = np.linalg.eigvals(J).real.max(axis=-1)                       # (B,)
    Dm = np.zeros((B, N, N)); Dm[:, range(N), range(N)] = D
    k2 = (kgrid ** 2)[None, :, None, None]
    sig = np.linalg.eigvals(J[:, None] - k2 * Dm[:, None]).real.max(axis=-1)   # (B,K)
    i = 1 + np.argmax(sig[:, 1:], axis=1)
    sig_pos = sig[np.arange(B), i]
    tr = np.trace(J, axis1=1, axis2=2)
    unstable = sig_pos > 1e-9
    return dict(max_re_eig_J=ev0, trace_J=tr, stable_strict=ev0 < 0.0,
                stable_loose=tr < 0.0, sig_max_pos=sig_pos, kstar=kgrid[i],
                unstable=unstable, turing_strict=(ev0 < 0.0) & unstable,
                turing_loose=(tr < 0.0) & unstable)


def turing_verdict(J, D, kgrid=KGRID):
    """STRICT Turing verdict, plus the loose (trace) one on the SAME k-grid.

    strict  : max Re eig(J) < 0  AND  max_{k>0} sigma(k) > 1e-9
    loose   : trace(J) < 0       AND  max_{k>0} sigma(k) > 1e-9

    docs/ROBUSTNESS_MEASUREMENT.md §3 measured the trace test overcounting by up to 70% of
    draws; eval/analysis.py::turing_ok uses the trace test, so its verdict is reported here
    only as `turing_loose`, never led with.
    """
    J = np.asarray(J, float)
    D = np.asarray(D, float)
    ev0 = np.linalg.eigvals(J).real
    M = J[None] - (kgrid ** 2)[:, None, None] * np.diag(D)[None]
    sig = np.linalg.eigvals(M).real.max(axis=-1)
    i = 1 + int(np.argmax(sig[1:]))
    unstable = bool(sig[i] > 1e-9)
    return dict(max_re_eig_J=float(ev0.max()), trace_J=float(np.trace(J)),
                stable_strict=bool(ev0.max() < 0.0), stable_loose=bool(np.trace(J) < 0.0),
                sig_max_pos=float(sig[i]), kstar=float(kgrid[i]), unstable=unstable,
                turing_strict=bool(ev0.max() < 0.0 and unstable),
                turing_loose=bool(np.trace(J) < 0.0 and unstable),
                sigma=sig)


# ---------------------------------------------------------------------------------------
# building a BOX-CONSTRAINED RNGRN: draw the CONSTRAINED value, invert the link function
# ---------------------------------------------------------------------------------------
def _inv_softplus(x):
    return torch.log(torch.expm1(torch.as_tensor(x, dtype=torch.float64)))


def build_model(alpha, delta, beta, D, s, gate_logit, form):
    """An RNGRN whose CONSTRAINED parameters are exactly the ones passed in.

    Inverts model.py:196-215's link functions (the same idiom as
    model.py::_low_basal_raw_params, with different ranges):
      alpha=softplus(theta_alpha), delta=softplus(theta_delta), beta=softplus(theta_beta),
      D=exp(theta_D), KA=softplus(theta_s)*sigmoid(theta_g), KR=softplus(theta_s)*(1-.).
    """
    m = RNGRN(N=N_SPECIES, form=form, n_hill=2, seed=0)
    with torch.no_grad():
        m.theta_alpha.copy_(_inv_softplus(alpha))
        m.theta_delta.copy_(_inv_softplus(delta))
        m.theta_beta.copy_(_inv_softplus(beta))
        m.theta_D.copy_(torch.log(torch.as_tensor(D, dtype=torch.float64)))
        m.theta_s.copy_(_inv_softplus(s))
        m.theta_g.copy_(torch.as_tensor(gate_logit, dtype=torch.float64))
    return m


# D-SAMPLING SCHEMES. d_ratio_of (scoring/plausibility.py) is the ratio of the two MOST
# MOBILE species, so the SMALLEST D never enters it. That is deliberate — it is what makes
# a ~7.5 D-ratio Turing-viable at all (docs/ROBUSTNESS_MEASUREMENT.md §4.4: 127/127
# generator systems stayed strictly Turing when the slowest diffuser was immobilised). The
# two schemes differ ONLY in where the third (slowest) node sits, and both satisfy the
# d_ratio box row by construction, so the acceptance rates are directly comparable.
#   mobile3  : all three species genuinely mobile — D_lo within one decade of D_mid.
#   immobile : a near-immobile third node — D_lo in 1e-4..1e-2 x D_mid.
D_SCHEMES = ("mobile3", "immobile")

# BETA. bio_box.yaml marks beta `source: UNCITED` and it is NEVER scored; a bound must not
# be invented for it. A value is still needed to build a model, so the DEFAULT arm uses the
# generator's own basal draw, scripts/gen_tg3.py:157  b = 10**U(-2, -0.3)  (0.01..0.5) —
# calibrated against the data-generating process this pipeline actually fits, not guessed.
# `--beta-mode` runs the sensitivity arms.
BETA_MODES = {
    "generator": (-2.0, -0.3),     # gen_tg3.py:157, the default
    "low": (-3.0, -3.0),           # fixed 1e-3
    "high": (-0.301, -0.301),      # fixed ~0.5
}

# BINDING (s, gate). No bio_box row exists for KA/KR at all: these are the FREE directions.
# s is drawn log-uniform over four decades, wide enough to bracket the generator's own Hill
# thresholds (gen_tg3.py:160 K = 10**U(-1, 0.6), n in {2,3,4}, so the comparable binding
# scale 1/K^n spans ~1e-2..1e2). The gate logit is the wide zero-mean normal of
# model.py::_low_basal_raw_params, i.e. sub-saturated, non-degenerate activator/repressor
# splits.
S_LOG_RANGE = (-2.0, 2.0)
GATE_LOGIT_SD = 2.5


def draw_batch(rng, box, scheme, beta_mode, B):
    """B parameter sets drawn INSIDE the cited rows of configs/bio_box.yaml.

    The two D-schemes are drawn from the SAME rng stream in the same order, so with a
    common seed `mobile3` and `immobile` differ in the slowest D and in nothing else —
    a paired comparison, which is what makes their acceptance rates directly comparable.
    """
    N = N_SPECIES
    lo_a, hi_a = box["alpha"].low, box["alpha"].high
    lo_d, hi_d = box["delta"].low, box["delta"].high
    lo_r, hi_r = box["d_ratio"].low, box["d_ratio"].high
    alpha = 10.0 ** rng.uniform(np.log10(lo_a), np.log10(hi_a), (B, N, N))
    delta = 10.0 ** rng.uniform(np.log10(lo_d), np.log10(hi_d), (B, N))
    b0, b1 = BETA_MODES[beta_mode]
    beta = (10.0 ** rng.uniform(b0, b1, (B, N)) if b1 > b0
            else np.full((B, N), 10.0 ** b0))
    d_ratio = 10.0 ** rng.uniform(np.log10(lo_r), np.log10(hi_r), B)
    # absolute D scale is a pure length rescale (k* ~ 1/sqrt(D)), so fix D_max = 1 and let
    # part 3 choose L from the resulting k*.
    d_mid = 1.0 / d_ratio
    frac = (rng.uniform(-1.0, 0.0, B) if scheme == "mobile3"
            else rng.uniform(-4.0, -2.0, B))
    D = np.stack([np.ones(B), d_mid, d_mid * 10.0 ** frac], axis=1)
    perm = rng.permuted(np.tile(np.arange(N), (B, 1)), axis=1)   # species order
    D = np.take_along_axis(D, perm, axis=1)
    s = 10.0 ** rng.uniform(*S_LOG_RANGE, size=(B, N, N))
    gate_logit = rng.normal(0.0, GATE_LOGIT_SD, (B, N, N))
    return dict(alpha=alpha, delta=delta, beta=beta, D=D, s=s, gate_logit=gate_logit,
                d_ratio=d_ratio)


def build_batched(pb, form):
    """A BatchedRNGRN whose per-member CONSTRAINED parameters are exactly `pb`."""
    B = pb["alpha"].shape[0]
    proto = RNGRN(N=N_SPECIES, form=form, n_hill=2, seed=0)
    bm = BatchedRNGRN([proto] * B)
    with torch.no_grad():
        bm.theta_alpha.copy_(_inv_softplus(pb["alpha"]))
        bm.theta_delta.copy_(_inv_softplus(pb["delta"]))
        bm.theta_beta.copy_(_inv_softplus(pb["beta"]))
        bm.theta_D.copy_(torch.log(torch.as_tensor(pb["D"], dtype=torch.float64)))
        bm.theta_s.copy_(_inv_softplus(pb["s"]))
        bm.theta_g.copy_(torch.as_tensor(pb["gate_logit"], dtype=torch.float64))
    return bm


def evaluate_batch(pb, form):
    """Steady state + strict Turing verdict for a whole batch of draws.

    A member whose steady state does not converge is a FAILED draw: it keeps its place in
    the denominator with converged=False and turing_strict=False, and is never dropped.
    """
    bm = build_batched(pb, form)
    xs, conv = steady_state_batched(bm)
    J = bm.jacobian(xs, create_graph=False).detach().cpu().numpy()
    D = bm.D.detach().cpu().numpy()
    v = turing_verdict_batch(J, D)
    c = np.asarray(conv.cpu().numpy(), bool)
    for k in ("turing_strict", "turing_loose", "stable_strict", "stable_loose", "unstable"):
        v[k] = v[k] & c
    for k in ("sig_max_pos", "kstar", "max_re_eig_J", "trace_J"):
        v[k] = np.where(c, v[k], np.nan)
    v["converged"] = c
    v["xstar"] = xs.detach().cpu().numpy()
    return v


def evaluate_draw(p, form, box):
    """Build, solve for x*, score against the box, and return the strict Turing verdict.

    A draw whose steady state does not converge is a FAILED draw: it stays in the
    denominator with converged=False and is never silently dropped.
    """
    m = build_model(p["alpha"], p["delta"], p["beta"], p["D"], p["s"], p["gate_logit"], form)
    xs, ok = steady_state(m)
    rep = plausibility_report(m.alpha, m.delta, m.beta, m.D, box=box)
    out = dict(converged=bool(ok), plausibility_score=rep["plausibility_score"],
               d_ratio=rep["plausibility_d_ratio_value"],
               xstar=xs.detach().cpu().numpy().copy())
    if not ok:
        out.update(turing_strict=False, turing_loose=False, sig_max_pos=np.nan,
                   kstar=np.nan, max_re_eig_J=np.nan, trace_J=np.nan,
                   stable_strict=False, stable_loose=False, unstable=False)
        return out, m
    J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
    v = turing_verdict(J, m.D.detach().cpu().numpy())
    v.pop("sigma")
    out.update(v)
    out["J"] = J
    return out, m


# ======================================================================================
# PART 1
# ======================================================================================
PARAM_KEYS = ("alpha", "delta", "beta", "D", "s", "gate_logit", "d_ratio")
VERDICT_KEYS = ("converged", "turing_strict", "turing_loose", "stable_strict",
                "stable_loose", "unstable", "sig_max_pos", "kstar", "max_re_eig_J",
                "trace_J")


def _wilson(k, n, z=1.959963985):
    """Wilson 95% interval for a binomial rate. Used instead of k/n +- 1.96*sqrt(p(1-p)/n)
    because the normal interval is meaningless at the rates measured here (k of order 1),
    where it can cover negative values or collapse to zero width at k=0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def part1(args):
    box = load_box("configs/bio_box.yaml")
    os.makedirs(ARR, exist_ok=True)
    store, t0 = {}, time.time()
    for form in ("competitive", "nc1"):
        for scheme in D_SCHEMES:
            rng = np.random.default_rng(args.seed)
            chunks = []
            done = 0
            while done < args.n:
                B = min(args.batch, args.n - done)
                pb = draw_batch(rng, box, scheme, args.beta_mode, B)
                v = evaluate_batch(pb, form)
                chunks.append((pb, v))
                done += B
                print(f"  {form:12s} {scheme:9s} {done:6d}/{args.n}  "
                      f"{time.time()-t0:7.1f}s", flush=True)
            cell = {k: np.concatenate([c[0][k] for c in chunks]) for k in PARAM_KEYS}
            cell.update({k: np.concatenate([c[1][k] for c in chunks])
                         for k in VERDICT_KEYS})
            cell["xstar"] = np.concatenate([c[1]["xstar"] for c in chunks])
            store[(form, scheme)] = cell

    flat = {}
    for (form, scheme), cell in store.items():
        for k, v in cell.items():
            flat[f"{form}__{scheme}__{k}"] = v
    np.savez_compressed(os.path.join(ARR, f"part1_box_draws_{args.beta_mode}.npz"),
                        kgrid=KGRID, n_per_cell=args.n, seed=args.seed,
                        beta_mode=args.beta_mode, **flat)

    # ---- EQUIVALENCE CHECK: the batched fast path against the serial reference ---------
    # `args.check` draws re-run through the serial RNGRN + losses.terms.steady_state + the
    # serial verdict. The VERDICT must match exactly; max Re eig(J) is only required to
    # agree to 1e-4 absolute, because the two steady-state solvers legitimately stop at
    # different residuals (the serial relaxation fallback accepts |f| < 1e-4, the batched
    # Newton reaches ~1e-10 on the same root), which moves the eigenvalue in the 8th
    # decimal. A verdict disagreement, by contrast, means the fast path is not measuring
    # what the table below claims and is fatal.
    key = ("competitive", "mobile3")
    c = store[key]
    n_chk = min(args.check, len(c["d_ratio"]))
    dis, worst = 0, 0.0
    for i in range(n_chk):
        p = {k: c[k][i] for k in ("alpha", "delta", "beta", "D", "s", "gate_logit")}
        r, _ = evaluate_draw(p, key[0], box)
        if bool(r["turing_strict"]) != bool(c["turing_strict"][i]):
            dis += 1
        if r["converged"] and c["converged"][i]:
            worst = max(worst, abs(r["max_re_eig_J"] - c["max_re_eig_J"][i]))
    print(f"\nbatched-vs-serial equivalence over {n_chk} draws: {n_chk - dis} verdicts "
          f"agree, worst |delta max Re eig J| = {worst:.3g}")
    if dis or worst > 1e-4:
        raise SystemExit(f"batched path disagrees with the serial reference: "
                         f"{dis} verdict flips, worst eig deviation {worst:.3g}")

    # ---- plausibility firewall check: every draw must score 1.0 by construction --------
    p0 = plausibility_report(torch.as_tensor(c["alpha"][0]), torch.as_tensor(c["delta"][0]),
                             torch.as_tensor(c["beta"][0]), torch.as_tensor(c["D"][0]),
                             box=box)
    assert p0["plausibility_score"] == 1.0, p0
    assert p0["plausibility_n_scored"] == 3, p0     # alpha, delta, d_ratio; beta UNCITED

    # ---- the table --------------------------------------------------------------------
    print(f"\nPART 1 — strictly-Turing acceptance INSIDE configs/bio_box.yaml "
          f"(beta_mode={args.beta_mode}, seed={args.seed}, N=3, n_hill=2, "
          f"plausibility_score=1.0 by construction on every draw)")
    hdr = (f"{'form':12s} {'D-scheme':9s} {'n':>7s} {'conv':>7s} {'strict':>7s} "
           f"{'rate':>9s} {'Wilson95':>19s} {'loose':>7s} {'looserate':>10s}")
    print(hdr)
    summary = {}
    for form in ("competitive", "nc1"):
        pooled = dict(n=0, conv=0, strict=0, loose=0)
        for scheme in D_SCHEMES:
            cell = store[(form, scheme)]
            n = len(cell["converged"])
            nc = int(cell["converged"].sum())
            ns = int(cell["turing_strict"].sum())
            nl = int(cell["turing_loose"].sum())
            lo, hi = _wilson(ns, n)
            print(f"{form:12s} {scheme:9s} {n:7d} {nc:7d} {ns:7d} {ns/n:9.5f} "
                  f"[{lo:8.6f},{hi:8.6f}] {nl:7d} {nl/n:10.5f}")
            summary[f"{form}/{scheme}"] = dict(
                n=n, converged=nc, strict=ns, loose=nl, rate_strict=ns / n,
                rate_loose=nl / n, wilson95_strict=[lo, hi])
            for a, b in (("n", n), ("conv", nc), ("strict", ns), ("loose", nl)):
                pooled[a] += b
        lo, hi = _wilson(pooled["strict"], pooled["n"])
        print(f"{form:12s} {'POOLED':9s} {pooled['n']:7d} {pooled['conv']:7d} "
              f"{pooled['strict']:7d} {pooled['strict']/pooled['n']:9.5f} "
              f"[{lo:8.6f},{hi:8.6f}] {pooled['loose']:7d} "
              f"{pooled['loose']/pooled['n']:10.5f}")
        summary[f"{form}/POOLED"] = dict(
            n=pooled["n"], converged=pooled["conv"], strict=pooled["strict"],
            loose=pooled["loose"], rate_strict=pooled["strict"] / pooled["n"],
            rate_loose=pooled["loose"] / pooled["n"], wilson95_strict=[lo, hi])
    summary["_meta"] = dict(beta_mode=args.beta_mode, seed=args.seed,
                            n_per_cell=args.n, n_hill=2, N=N_SPECIES,
                            kgrid=[float(KGRID[1]), float(KGRID[-1]), len(KGRID)],
                            s_log_range=list(S_LOG_RANGE), gate_logit_sd=GATE_LOGIT_SD,
                            batched_serial_check=[n_chk - dis, n_chk],
                            batched_serial_worst_eig_dev=float(worst))
    with open(os.path.join(ARR, f"part1_summary_{args.beta_mode}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return store


# ======================================================================================
# HARVEST — collect enough strictly-Turing base systems for parts 2 and 3
# ======================================================================================
# Part 1's acceptance rates are the MEASUREMENT and its `n` is its denominator; they are
# not touched. But at rates of 5e-5 a 20000-draw cell yields 0-1 survivors, which is an
# anecdote, not a band. This pass draws from an INDEPENDENT seed offset until it has
# `--want` survivors per cell or hits `--max-draws`, and records how many draws that took —
# which is itself a second, independent estimate of the same rate.
def harvest(args):
    box = load_box("configs/bio_box.yaml")
    os.makedirs(ARR, exist_ok=True)
    flat, log = {}, {}
    t0 = time.time()
    for form in ("competitive", "nc1"):
        for scheme in D_SCHEMES:
            rng = np.random.default_rng(args.seed + 10_000)
            keep, seen, hits = [], 0, 0
            while hits < args.want and seen < args.max_draws:
                B = min(args.batch, args.max_draws - seen)
                pb = draw_batch(rng, box, scheme, args.beta_mode, B)
                v = evaluate_batch(pb, form)
                seen += B
                sel = np.where(v["turing_strict"])[0]
                hits += len(sel)
                for i in sel:
                    keep.append(({k: pb[k][i] for k in PARAM_KEYS},
                                 {k: v[k][i] for k in VERDICT_KEYS}))
                print(f"  {form:12s} {scheme:9s} seen={seen:7d} hits={hits:3d}  "
                      f"{time.time()-t0:7.1f}s", flush=True)
            log[f"{form}/{scheme}"] = dict(seen=seen, hits=hits,
                                           rate=hits / seen if seen else float("nan"))
            for k in PARAM_KEYS:
                flat[f"{form}__{scheme}__{k}"] = (
                    np.stack([p[0][k] for p in keep]) if keep
                    else np.zeros((0,) + np.shape(draw_batch(
                        np.random.default_rng(0), box, scheme, args.beta_mode, 1)[k])[1:]))
            for k in VERDICT_KEYS:
                flat[f"{form}__{scheme}__{k}"] = np.array([p[1][k] for p in keep])
    np.savez_compressed(os.path.join(ARR, f"part1_harvest_{args.beta_mode}.npz"),
                        seed=args.seed + 10_000, want=args.want,
                        max_draws=args.max_draws, **flat)
    with open(os.path.join(ARR, f"part1_harvest_{args.beta_mode}.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print("\nHARVEST (independent seed offset +10000; a second estimate of the same rate)")
    for k, v in log.items():
        print(f"  {k:24s} {v['hits']:3d} survivors in {v['seen']:7d} draws "
              f"-> {v['rate']:.6f}")
    return log


# ======================================================================================
# PART 2 — the mu (= delta) axis, swept
# ======================================================================================
# Box [0.4, 5] plus one decade either side, log-spaced. The sweep variable is the
# GEOMETRIC MEAN of the delta vector: the base draw's per-species delta pattern is held and
# rescaled so its geometric mean equals d. Individual delta_i can then sit slightly outside
# the box even when d is inside it, so `all_delta_in_box` is recorded per point.
SWEEP_D = np.logspace(np.log10(0.04), np.log10(50.0), 41)


def _sweep_one(p, form, box, mode, species=None):
    base = np.asarray(p["delta"], float)
    gm = float(np.exp(np.mean(np.log(base))))
    # THE BASE POINT MUST BE ON THE CURVE. The 41-point log grid does not generally contain
    # the base draw's own delta, so without this the sweep can report 0/41 strictly-Turing
    # for a system that IS strictly Turing at the delta it was harvested at — a curve that
    # misses the one point it is known to pass through. Insert it and keep the grid sorted.
    base_pt = gm if mode == "uniform" else float(base[species])
    grid = np.unique(np.concatenate([SWEEP_D, [base_pt]]))
    rows = []
    for d in grid:
        if mode == "uniform":
            delta = base * (d / gm)
        else:                                   # per-species: move ONE delta_i, hold rest
            delta = base.copy()
            delta[species] = d
        q = dict(p, delta=delta)
        res, _ = evaluate_draw(q, form, box)
        lo, hi = box["delta"].low, box["delta"].high
        rows.append(dict(d=float(d), mode=mode, species=-1 if species is None else species,
                         is_base=bool(d == base_pt), delta=delta.copy(),
                         all_delta_in_box=bool(np.all((delta >= lo) & (delta <= hi))),
                         converged=res["converged"], turing_strict=res["turing_strict"],
                         turing_loose=res["turing_loose"],
                         sig_max_pos=res["sig_max_pos"], kstar=res["kstar"],
                         max_re_eig_J=res["max_re_eig_J"]))
    return rows


def survivors(z, form, scheme):
    """Indices of the strictly-Turing draws of one part-1 cell, in draw order."""
    return np.where(z[f"{form}__{scheme}__turing_strict"])[0]


def balanced_picks(z, form, n):
    """Up to `n` survivors of one form, ROUND-ROBINED across the D-schemes.

    Taking `survivors(mobile3)[:n] + survivors(immobile)[:n]` and then truncating to `n`
    silently yields n mobile3 and zero immobile whenever mobile3 has n of its own — which
    would make every part-2 curve and part-3 panel one scheme's anecdote, the exact thing
    the two schemes exist to avoid. Round-robin instead, and fall through to whichever
    scheme still has survivors when the other runs out.
    """
    pools = {s: list(survivors(z, form, s)) for s in D_SCHEMES}
    out = []
    while len(out) < n and any(pools.values()):
        for s in D_SCHEMES:
            if pools[s] and len(out) < n:
                out.append((s, int(pools[s].pop(0))))
    return out


def params_at(z, form, scheme, i):
    return {k: z[f"{form}__{scheme}__{k}"][i]
            for k in ("alpha", "delta", "beta", "D", "s", "gate_logit")}


def part2(args):
    box = load_box("configs/bio_box.yaml")
    z = np.load(os.path.join(ARR, f"part1_harvest_{args.beta_mode}.npz"),
                allow_pickle=False)
    out, meta = {}, {}
    for form in ("competitive", "nc1"):
        picks = balanced_picks(z, form, args.n_base)
        if not picks:
            print(f"PART 2: no strictly-Turing base system for form={form} — "
                  f"nothing to sweep")
            continue
        for j, (scheme, i) in enumerate(picks):
            p = params_at(z, form, scheme, i)
            rows = _sweep_one(p, form, box, "uniform")
            out[f"{form}/base{j}/uniform"] = rows
            print(f"  {form} base{j} ({scheme}, draw {i}) uniform: "
                  f"{sum(r['turing_strict'] for r in rows)}/{len(rows)} strict-Turing",
                  flush=True)
            if j == 0:            # per-species sweep on the first base system of each form
                for sp in range(N_SPECIES):
                    rr = _sweep_one(p, form, box, "per_species", species=sp)
                    out[f"{form}/base{j}/species{sp}"] = rr
                    print(f"  {form} base{j} species{sp}: "
                          f"{sum(r['turing_strict'] for r in rr)}/{len(rr)} strict-Turing",
                          flush=True)
        meta[f"{form}__base_idx"] = np.array([i for _, i in picks])
        meta[f"{form}__base_scheme"] = np.array([s for s, _ in picks])
        meta[f"{form}__base_delta"] = np.stack(
            [params_at(z, form, s, i)["delta"] for s, i in picks])

    flat = dict(sweep_d=SWEEP_D, box_lo=box["delta"].low, box_hi=box["delta"].high, **meta)
    for key, rows in out.items():
        for f in ("d", "turing_strict", "turing_loose", "sig_max_pos", "kstar",
                  "converged", "all_delta_in_box", "max_re_eig_J", "is_base"):
            flat[f"{key}__{f}".replace("/", "__")] = np.array([r[f] for r in rows])
    np.savez_compressed(os.path.join(ARR, f"part2_mu_sweep_{args.beta_mode}.npz"), **flat)
    print(f"wrote {ARR}/part2_mu_sweep_{args.beta_mode}.npz  ({len(out)} curves)")
    return out


# ======================================================================================
# PART 3 — LOOK AT THE PATTERNS, 64x64
# ======================================================================================
# L POLICY: L = PERIODS * 2*pi/k*, i.e. PERIODS full wavelengths of the fastest-growing
# linear mode across the box. PERIODS=6 is the generator's own original choice
# (scripts/gen_tg3.py:66 "The original generator hard-coded 6 periods per box") and sits
# inside its replacement acceptance window PERIODS_CHOICES = {3..14}, so the domain size
# these figures use is the same one the training data was built at.
PERIODS = 6
GRID_N = 64


def part3(args):
    box = load_box("configs/bio_box.yaml")
    zh = np.load(os.path.join(ARR, f"part1_harvest_{args.beta_mode}.npz"),
                 allow_pickle=False)
    z = np.load(os.path.join(ARR, f"part1_box_draws_{args.beta_mode}.npz"),
                allow_pickle=False)
    bank = morphology.default_reference_bank()
    recs, fields, L_ref = [], [], {}
    for form in ("competitive", "nc1"):
        picks = [(s, i, zh) for s, i in balanced_picks(zh, form, args.n_show)]
        # a NON-patterning control: the first converged, NOT-strictly-Turing draw of this
        # form. A figure that only shows successes is not evidence.
        # the NON-patterning control comes from the part-1 population, not the harvest
        # (the harvest keeps only survivors by construction).
        neg = np.where(z[f"{form}__mobile3__converged"]
                       & ~z[f"{form}__mobile3__turing_strict"])[0]
        if neg.size:
            picks.append(("mobile3", int(neg[0]), z))
        for scheme, i, src in picks:
            p = params_at(src, form, scheme, i)
            m = build_model(p["alpha"], p["delta"], p["beta"], p["D"], p["s"],
                            p["gate_logit"], form)
            strict = bool(src[f"{form}__{scheme}__turing_strict"][i])
            ks = float(src[f"{form}__{scheme}__kstar"][i])
            if strict and np.isfinite(ks) and ks > 0:
                L = PERIODS * 2.0 * np.pi / ks
                L_ref.setdefault(form, L)
            else:
                # THE NON-PATTERNING CONTROL. A linearly stable system's argmax over k>0
                # sits at the grid's smallest k, so L = 6*2pi/k* would be a nonsensically
                # huge domain and the panel would be answering a different question. It is
                # run at the SAME L as the first patterned system of its form instead:
                # same domain, different system, which is what a control is for.
                L = L_ref.get(form, PERIODS * 2.0 * np.pi / max(ks, 1e-3))
            L = float(L)
            r = rollout.simulate(m, L=L, n=GRID_N, integrator="etdrk4_rfft",
                                 early_stop=True, seed=args.seed)
            F = np.asarray(r["fields"], float)                       # (N, n, n)
            # WHICH CHANNEL TO LOOK AT. eval/rollout.py::simulate defines `amplitude` and
            # `patterned` on CHANNEL 0 ONLY — correct for a real dataset, where channel 0 is
            # the observed one. Here it is WRONG, and by our own doing: draw_batch SHUFFLES
            # the species order (D-scheme aside, nothing distinguishes them), so channel 0
            # is arbitrary. MEASURED: a competitive/mobile3 survivor has channel-0 std
            # 2.0e-4 and channel-1 std 0.71 — the library calls it unpatterned while a
            # large pattern sits one channel over. So the panel shows the channel with the
            # LARGEST spatial std, and both verdicts are recorded: `patterned` is the
            # library's channel-0 one (unchanged, never overridden) and `patterned_any` is
            # true when ANY channel clears its own floor.
            std_c = F.std(axis=(1, 2))
            xs_c = np.asarray(r.get("xstar_used", np.zeros(F.shape[0])), float) \
                if False else F.mean(axis=(1, 2))
            chan = int(np.argmax(std_c))
            floors = np.maximum(1e-3, 0.02 * np.abs(xs_c))
            patterned_any = bool(np.any(std_c > floors))
            f0 = F[chan]
            # scoring/morphology.py RAISES on a field with (near-)zero spatial variance —
            # skewness and kurtosis are undefined there, and it refuses to return a
            # fabricated class. That is exactly what the non-patterning control produces,
            # so the refusal is caught and RECORDED rather than allowed to abort the
            # gallery or, worse, silently replaced with a made-up label.
            try:
                call = morphology.classify_morphology(f0, bank)
                morph, margin = call.label, float(call.margin)
            except ValueError as exc:
                morph, margin = "undefined", float("nan")
                print(f"    morphology undefined (collapsed field): {exc}", flush=True)
            recs.append(dict(idx=i, form=form, scheme=scheme,
                             turing_strict=strict,
                             d_ratio=float(src[f"{form}__{scheme}__d_ratio"][i]),
                             delta_min=float(p["delta"].min()),
                             delta_max=float(p["delta"].max()),
                             L=L, kstar_lin=ks, kstar_field=float(r["kstar"]),
                             patterned=bool(r["patterned"]), amplitude=float(r["amplitude"]),
                             sig_max=float(r["sig_max"]), stopped_reason=r["stopped_reason"],
                             nsteps_run=int(r["nsteps_run"]), nsteps=int(r["nsteps"]),
                             seconds=float(r["seconds"]), morphology=morph,
                             morph_margin=margin, channel_shown=chan,
                             patterned_any=patterned_any,
                             std_c0=float(std_c[0]), std_c1=float(std_c[1]),
                             std_c2=float(std_c[2]), std_max=float(std_c.max())))
            fields.append(np.asarray(r["fields"], float))
            print(f"  {form:12s} idx{i:6d} {scheme:9s} L={L:7.2f} "
                  f"k*={ks:.4f} patt_lib={r['patterned']!s:5s} patt_any={patterned_any!s:5s} "
                  f"ch={chan} std={np.round(std_c, 5)} "
                  f"stop={r['stopped_reason']:11s} morph={morph} "
                  f"({r['seconds']:.1f}s)", flush=True)

    keys = recs[0].keys()
    np.savez_compressed(
        os.path.join(ARR, f"part3_rollouts_{args.beta_mode}.npz"),
        fields=np.stack(fields), grid_n=GRID_N, periods=PERIODS,
        **{k: np.array([r[k] for r in recs]) for k in keys})
    with open(os.path.join(ARR, f"part3_rollouts_{args.beta_mode}.json"), "w") as fh:
        json.dump(recs, fh, indent=2)
    print(f"wrote {ARR}/part3_rollouts_{args.beta_mode}.npz  ({len(recs)} rollouts)")
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("part1", "harvest", "part2", "part3"))
    ap.add_argument("--n", type=int, default=2000, help="part1 draws per (form, D-scheme)")
    ap.add_argument("--batch", type=int, default=2048, help="part1 batched-member count")
    ap.add_argument("--check", type=int, default=32,
                    help="part1 draws re-run serially as an equivalence check")
    ap.add_argument("--n-base", type=int, default=6, help="part2 base systems per form")
    ap.add_argument("--n-show", type=int, default=6, help="part3 patterned systems per form")
    ap.add_argument("--beta-mode", choices=tuple(BETA_MODES), default="generator")
    ap.add_argument("--want", type=int, default=6, help="harvest survivors per cell")
    ap.add_argument("--max-draws", type=int, default=200000,
                    help="harvest draw cap per cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"part1": part1, "harvest": harvest, "part2": part2,
     "part3": part3}[args.part](args)


if __name__ == "__main__":
    main()
