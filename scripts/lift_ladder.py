#!/usr/bin/env python
"""lift_ladder.py — the V0-V4 lift-validation-ladder campaign driver.

WHAT THIS IS. A thin CLI over `rngrn.eval.ladder`: it selects the population a rung is
measured over, calls the rung's own function, and writes one JSON per rung under
`<out>/results/` plus the arrays a figure would need under `<out>/arrays/`. No rung's
arithmetic lives here — every number comes out of `eval/ladder.py` or `eval/lifted.py`, so
the campaign and the unit tests exercise the same code (docs/REDESIGN_rngrn.md §5.3).

    .venv/bin/python scripts/lift_ladder.py v3 --out experiments/lift_ladder/v3

NOT a trainer, sweep or target-report, so `.claude/hooks/guard_trainer.py` does not require
`scripts/guarded_run.sh` for it. The campaign is nonetheless launched through the guard —
V3 holds several (N,N,n,n) gate arrays at n=512 (18.9 MB each) plus the ETDRK4 work arrays,
and CLAUDE.md §7a's `flock` is what keeps that from landing on top of another agent's pool.

THE THREE POPULATIONS, and why a rung uses the one it does.

  `low_basal`  — `ladder.draw_models(n, form, seed)`, the population V0-V2 are defined over.
                 Newton-tight steady states, both forms, deterministic in (n, form, seed).
                 **0 of 398 such N=3 draws are strictly Turing** (docs/DECISIONS.md
                 D-EVID-11, quoted in `ladder.v1_continuation`), so it is the right
                 population for algebraic, linear and 0-D claims and the WRONG one for any
                 claim about patterning. `--harvest-draws` measures that rate again on this
                 branch rather than inheriting it.
  `harvest`    — the 23 strictly-Turing generator draws Stage 0 harvested inside the cited
                 biological box, stored in
                 `experiments/figures_report/stage0/arrays/part1_harvest_generator.npz`
                 (docs/BIO_VIABILITY.md §1.2): competitive 6 mobile3 + 6 immobile, nc1
                 5 mobile3 + 6 immobile. These are §5.3 V4's "generator draws already
                 harvested" and the only Turing-positive population large enough for V3's
                 patterning and morphology flags to be non-vacuous.
  `d5`         — the 4 Turing-unstable recovered models of diagnostic D5 (seeds 1, 3, 5, 6
                 of `experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed*`;
                 `analysis.json::turing_flags` = [F,T,F,T,F,T,T,F,F,F], the 4 True entries,
                 whose `kstar_model` is 0.2848 against the 6 failures' grid-floor 0.006678).
                 These are §5.3 V4's "the target's own gate-passing recovered models" — the
                 recovered side of the survey, not generator draws.

BOX SIZE IS SET PER MODEL AT p = 8 PERIODS ACROSS THE BOX: `L = 8 * 2*pi / k*_qss`. A fixed
L would put some of these systems below one wavelength per box (their k* spans two orders of
magnitude) and no lifted-vs-QSS comparison survives that. p = 8 is not a free choice either:
it is the target `turing_labyrinth/sample_0000`'s own geometry, at which
docs/SPEC_fourier_training.md §9.1's BINDING one-radial-bin k* tolerance 2*pi/L is exactly
12.5% of k* — so every V3 k* number here is judged at the same relative precision as the
target's, rather than at a precision the box was quietly chosen to make easy.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import time

import numpy as np
import torch

from rngrn.eval import ladder, lifted
from rngrn.eval.lifted import MU_BIO_CENTRAL, MU_BIO_HI, MU_BIO_LO
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
HARVEST_NPZ = REPO / "experiments/figures_report/stage0/arrays/part1_harvest_generator.npz"
D5_RUNS = REPO / "experiments/diag_fft/d5/runs"
D5_PREFIX = "m3_registry_20260812_092043_seed"
D5_TURING_SEEDS = (1, 3, 5, 6)          # analysis.json::turing_flags, the 4 True entries

# V0-V2 population knobs, matching tests/test_lift_ladder.py so a campaign number and a test
# number are the same measurement at a larger sample size.
V0_MUS = [1e-6, 1e-4, 1e-2, 1.0, 1e2]
V0_N_DRAWS = 20
V0_SEED = 101
V1_MUS = (1e-7, 1e-6, 1e-5)
V1_SEEDS = (7, 101, 202, 303)
V1_PER_SEED = 2
V2_MUS = (1e-5, 7.2e-4, 1e-3, 1e-2)
V2_QSS_MUS = (1e-6, 1e-5, 1e-4)
V2_ROLLOUT_HORIZON, V2_GATE_TIMES = 5.0, 1000.0

# V3(a): the spec's {1e-4, 1e-5, 1e-6} plus mu_gate = 1e-3 (owner decision D-REDESIGN-5).
V3_MUS = [1e-3, 1e-4, 1e-5, 1e-6]
V3_PERIODS_PER_BOX = 8
V3_N_FULL = 512          # the grid §5.3 V3 licenses the morphology-class claim at
V3_N_ANCHOR = 128        # V3(b)'s torch grid
FORMS = ("competitive", "nc1")

# V3(b)'S HORIZON IS A STEP BUDGET, NOT A GROWTH TIME, AND THAT IS FORCED BY ITS OWN dt
# POLICY. The §5.2 policy dt = min(0.2/jac_rate, mu/2) is mu/2 = 3.6e-4 at mu_central, up to
# 702x finer than the QSS rollout's growth-rate-aware dt (measured 0.0264-0.2528 on the 23
# harvest systems, experiments/lift_ladder/v3/results/v3.json -- dt/mu spans 26.4-2.528e5
# over the V3_MUS set), while the pattern-formation horizon 40/|sigma_max| is ~700 time
# units. Running that horizon at
# that dt is ~2e6 steps per field, i.e. past `simulate_lifted`'s own 200000 max_steps, and a
# step_budget run is TRUNCATED -- its field is not a statement about the attractor
# (eval/rollout.py::simulate). So V3(b) is scoped to what its dt can actually pay for: a
# COMMON, SHORT horizon T = V3B_STEPS * dt, run at dt and at dt/2 (hence 2x V3B_STEPS at the
# fine end), which is a dt-CONVERGENCE measurement on the transient and is reported as one.
# The attractor-scale statement at mu_central comes from the separate 512^2 run at the QSS
# dt in the same block. §5.3 V3(b) is "report-only until the ladder completes" either way.
V3B_STEPS = 4000

V4_N_MU = 9
V4_CLOUD = 200


# ======================================================================================
# populations
# ======================================================================================
def _inv_softplus(a):
    return torch.log(torch.expm1(torch.as_tensor(np.asarray(a, float))))


def harvest_models():
    """The 23 strictly-Turing Stage-0 generator draws, rebuilt as `RNGRN` instances.

    The npz stores the CONSTRAINED parameters (alpha, delta, beta, s, D) plus the raw gate
    logit; `model.py`'s link functions are inverted here exactly as
    `lifted.cloud_xstar` inverts them for a perturbation cloud — softplus for alpha, delta,
    beta and the binding budget s, log for D, and theta_g copied straight across. The
    round trip is exact to ~5e-16 (docs/BIO_VIABILITY.md §1, measured there, not assumed).
    """
    d = np.load(HARVEST_NPZ, allow_pickle=True)
    models, labels = [], []
    for form in FORMS:
        for scheme in ("mobile3", "immobile"):
            cell = f"{form}__{scheme}"
            if f"{cell}__alpha" not in d:
                continue
            for i in range(d[f"{cell}__alpha"].shape[0]):
                if not bool(d[f"{cell}__turing_strict"][i]):
                    continue
                m = RNGRN(N=3, form=form, n_hill=2, seed=0)
                with torch.no_grad():
                    m.theta_alpha.copy_(_inv_softplus(d[f"{cell}__alpha"][i]))
                    m.theta_delta.copy_(_inv_softplus(d[f"{cell}__delta"][i]))
                    m.theta_beta.copy_(_inv_softplus(d[f"{cell}__beta"][i]))
                    m.theta_s.copy_(_inv_softplus(d[f"{cell}__s"][i]))
                    m.theta_g.copy_(torch.as_tensor(d[f"{cell}__gate_logit"][i]))
                    m.theta_D.copy_(torch.log(torch.as_tensor(d[f"{cell}__D"][i])))
                models.append(m)
                labels.append(f"harvest/{cell}/{i}")
    return models, labels


def d5_models():
    """The 4 Turing-unstable D5 recovered models, from their tracked checkpoints."""
    models, labels = [], []
    for s in D5_TURING_SEEDS:
        rd = D5_RUNS / f"{D5_PREFIX}{s}"
        ck = torch.load(rd / "checkpoints" / "model.pt", map_location="cpu",
                        weights_only=True)
        m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
        m.load_state_dict(ck["state_dict"])
        models.append(m)
        labels.append(f"d5/seed{s}")
    return models, labels


def draw_models_turing_survivors(n_draws: int, seed: int = 0):
    """Strictly-Turing survivors of `ladder.draw_models`, with the RATE they were found at.

    The brief's "draw_models harvest". The expected answer is ZERO — D-EVID-11 measured
    0/398 for this init — and measuring it again on this branch is the point: a rate is a
    result, and 'we looked and found none' is a different statement from 'we assumed none'.
    """
    out, labels = [], []
    n_seen = 0
    for form in FORMS:
        models = ladder.draw_models(n=n_draws, form=form, seed=seed)
        for i, m in enumerate(models):
            xs, _ = steady_state(m)
            n_seen += 1
            if ladder.qss_verdict(m, xs.detach().cpu().numpy())["turing_strict"]:
                out.append(m)
                labels.append(f"low_basal/{form}/{seed}/{i}")
    return out, labels, n_seen


def box_size(model, xstar=None, periods: int = V3_PERIODS_PER_BOX):
    """L such that the QSS k* fits `periods` wavelengths across the box. See the module
    docstring for why p = 8 rather than a fixed L."""
    if xstar is None:
        xs, converged = steady_state(model)
        if not converged:
            raise RuntimeError("box_size: model has no converged steady state")
        xstar = xs.detach().cpu().numpy()
    k = ladder.qss_verdict(model, xstar)["kstar"]
    if not (np.isfinite(k) and k > 0.0):
        raise RuntimeError(f"box_size: QSS k* is {k}, so there is no wavelength to fit a "
                           "box to. Refusing to invent one.")
    return float(periods * 2.0 * np.pi / k), float(k), xstar


# ======================================================================================
# the rungs
# ======================================================================================
def run_v0(_args):
    out = {"mus": V0_MUS, "n_draws": V0_N_DRAWS, "seed": V0_SEED, "forms": {}}
    for form in FORMS:
        models = ladder.draw_models(n=V0_N_DRAWS, form=form, seed=V0_SEED)
        out["forms"][form] = ladder.v0_invariants(models, mus=V0_MUS)
    return out, {}


def run_v1(_args):
    """V1 over the low_basal population, PLUS the controller rider: the Turing-positive
    systems where k* is INTERIOR, where `kstar_grid_offset` is a real measurement rather
    than two grid floors agreeing by construction (`v1_continuation`'s docstring). The
    Turing-positive population is the union of three sources -- `draw_models` survivors
    (measured 0 of 398 for this init, D-EVID-11; re-measured here as `draw_models_turing_rate`),
    the 23 harvest systems (T3 ruling: V1 must also cover the harvest survivors, not the D5
    four alone), and the D5 four -- each row tagged `population` so the three are never
    silently pooled as one."""
    out = {"mus": list(V1_MUS), "seeds": list(V1_SEEDS), "per_seed": V1_PER_SEED,
           "low_basal": [], "turing_positive": []}
    for form in FORMS:
        for seed in V1_SEEDS:
            for i, m in enumerate(ladder.draw_models(n=V1_PER_SEED, form=form, seed=seed)):
                r = ladder.v1_continuation(m, ladder.default_kgrid(m), mus=V1_MUS)
                r.update(form=form, seed=seed, draw=i, label=f"low_basal/{form}/{seed}/{i}")
                out["low_basal"].append(r)

    surv, surv_labels, n_seen = draw_models_turing_survivors(V0_N_DRAWS, seed=V0_SEED)
    hv, hv_labels = harvest_models()
    d5, d5_labels = d5_models()
    out["draw_models_turing_rate"] = dict(
        n_drawn=n_seen, n_strictly_turing=len(surv),
        rate=(len(surv) / n_seen if n_seen else float("nan")),
        note="strict verdict (ladder.qss_verdict), NOT analysis.turing_ok's trace test")
    pops = (["draw_survivor"] * len(surv) + ["harvest"] * len(hv) + ["d5"] * len(d5))
    for m, label, pop in zip(surv + hv + d5, surv_labels + hv_labels + d5_labels, pops):
        r = ladder.v1_continuation(m, ladder.default_kgrid(m), mus=V1_MUS)
        r.update(label=label, form=m.form, population=pop)
        out["turing_positive"].append(r)

    # arrays the viz needs: the per-row scalars stacked, so a figure does not have to
    # re-walk the JSON's row lists (docs/PLAN_redesign.md Task 6 interface: "the arrays the
    # viz needs under arrays/"). All quantities are already in `out`; nothing is recomputed.
    lb, tp = out["low_basal"], out["turing_positive"]
    arrays = dict(
        low_basal_labels=np.array([r["label"] for r in lb], dtype=object),
        low_basal_slow_branch_order=np.array([r["slow_branch_order"] for r in lb]),
        low_basal_kstar_grid_offset=np.array([r["kstar_grid_offset"] for r in lb], dtype=int),
        low_basal_min_fast_mu_product=np.array([r["min_fast_mu_product"] for r in lb]),
        turing_positive_labels=np.array([r["label"] for r in tp], dtype=object),
        turing_positive_population=np.array([r["population"] for r in tp], dtype=object),
        turing_positive_slow_branch_order=np.array([r["slow_branch_order"] for r in tp]),
        turing_positive_kstar_grid_offset=np.array([r["kstar_grid_offset"] for r in tp],
                                                    dtype=int),
        turing_positive_kstar_qss=np.array([r["kstar_qss"] for r in tp]),
        turing_positive_min_qss_overlap=np.array([r["min_qss_overlap"] for r in tp]),
        turing_positive_frac_k_separated=np.array([r["frac_k_separated"] for r in tp]),
        turing_positive_max_mu_D_k2=np.array([r["max_mu_D_k2"] for r in tp]))
    return out, arrays


def _v2_horizon(mu):
    return min(V2_ROLLOUT_HORIZON, V2_GATE_TIMES * mu)


def run_v2(_args):
    out = {"mus": list(V2_MUS), "qss_mus": list(V2_QSS_MUS), "forms": {}}
    for form in FORMS:
        m = ladder.draw_models(n=1, form=form, seed=11)[0]
        temporal = {}
        for mu in V2_MUS:
            r = ladder.v2_temporal(m, mu=mu, T=_v2_horizon(mu),
                                   dts=[mu / 2, mu / 4, mu / 8])
            r["sup_err"] = {str(k): v for k, v in r["sup_err"].items()}
            r["n_steps"] = {str(k): v for k, v in r["n_steps"].items()}
            temporal[str(mu)] = r
        out["forms"][form] = dict(
            temporal=temporal,
            qss_limit=ladder.v2_qss_limit(m, mus=V2_QSS_MUS, T=V2_ROLLOUT_HORIZON),
            rhs_mirror_error=ladder.v2_rhs_mirror_error(m))
    return out, {}


def _v3a_rows(models, labels, n, backend, device, seed):
    rows = []
    for m, label in zip(models, labels):
        L, k, xstar = box_size(m)
        t0 = time.perf_counter()
        r = ladder.v3_spatial(m, mus=V3_MUS, n=n, L=L, seed=seed, xstar=xstar,
                              backend=backend, device=device)
        r.update(label=label, kstar_qss_linear=k, wall_seconds=time.perf_counter() - t0,
                 periods_requested=V3_PERIODS_PER_BOX)
        rows.append(r)
        print(f"  {label}: L={L:.3g} patterned_agree={r['patterned_agree']} "
              f"morph={r['morphology_agree']} kstar_bin={r['kstar_within_one_bin']} "
              f"l2={[f'{v:.2e}' for v in (r['l2_diff_by_mu'][mu] for mu in r['mus'])]} "
              f"({r['wall_seconds']:.0f}s)", flush=True)
    return rows


def _morph_subset(models, labels, per_cell):
    """The first `per_cell` systems of each (form, D-scheme) harvest cell.

    A FIXED, COST-BLIND RULE. The 512^2 block is ~an order of magnitude dearer per system
    than the sweep grid (the QSS control is numpy on the CPU and these horizons run to tens
    of thousands of steps), so it cannot cover all 23. Selecting by stored index rather than
    by anything the run measures — cost, growth rate, how well the arm agrees — is what keeps
    the subset from being a choice about the answer. All four cells are represented.
    """
    seen, out_m, out_l = {}, [], []
    for m, label in zip(models, labels):
        cell = label.rsplit("/", 1)[0]
        if seen.get(cell, 0) >= per_cell:
            continue
        seen[cell] = seen.get(cell, 0) + 1
        out_m.append(m)
        out_l.append(label)
    return out_m, out_l


def run_v3(args):
    """V3(a) over the whole harvest population at the sweep grid, then the 512^2 block:
    V3(a) again on a fixed subset at the grid §5.3 licenses the MORPHOLOGY-class claim at,
    and V3(b) — the mu_central anchor under the §5.2 dt policy with a dt-halving pair on
    128^2, plus 512^2 at the QSS dt."""
    backend = "numpy" if args.cpu else "torch"
    device = "cpu" if args.cpu else "cuda"
    models, labels = harvest_models()
    if args.limit:
        models, labels = models[:args.limit], labels[:args.limit]

    print(f"V3(a): {len(models)} harvest systems at {args.n}^2, backend={backend}",
          flush=True)
    rows = _v3a_rows(models, labels, args.n, backend, device, args.seed)

    sub_m, sub_l = _morph_subset(models, labels, args.morph_per_cell)
    print(f"V3(a) at {args.n_full}^2 (morphology claim): {len(sub_m)} systems", flush=True)
    full_rows = _v3a_rows(sub_m, sub_l, args.n_full, backend, device, args.seed)

    print("V3(b): mu_central anchor, dt = min(0.2/jac_rate, mu/2)", flush=True)
    anchor = []
    for m, label in zip(sub_m, sub_l):
        L, k, xstar = box_size(m)
        D = m.D.detach().cpu().numpy()
        dt_qss, _, _, _, _ = lifted.step_policy(m, xstar, D, L, V3_N_ANCHOR, None, None,
                                                200000)
        dt_policy = min(dt_qss, MU_BIO_CENTRAL / 2.0)
        T_b = V3B_STEPS * dt_policy          # the SAME horizon at both dt (see V3B_STEPS)
        pair = {}
        for tag, dt in (("dt", dt_policy), ("dt_half", 0.5 * dt_policy)):
            r = ladder.v3_spatial(m, mus=[MU_BIO_CENTRAL], n=V3_N_ANCHOR, L=L, seed=args.seed,
                                  xstar=xstar, dt=dt, T=T_b, backend=backend, device=device)
            pair[tag] = r
        d1 = pair["dt"]["l2_diff_by_mu"][MU_BIO_CENTRAL]
        d2 = pair["dt_half"]["l2_diff_by_mu"][MU_BIO_CENTRAL]
        full = ladder.v3_spatial(m, mus=[MU_BIO_CENTRAL], n=args.n_full, L=L, seed=args.seed,
                                 xstar=xstar, backend=backend, device=device)
        anchor.append(dict(label=label, L=L, kstar_qss_linear=k, dt_qss_128=float(dt_qss),
                           dt_policy=float(dt_policy), n_anchor=V3_N_ANCHOR,
                           T_transient=float(T_b), steps_at_dt=V3B_STEPS,
                           l2_at_dt=d1, l2_at_dt_half=d2,
                           dt_halving_ratio=(d1 / d2 if d2 > 0 else float("inf")),
                           anchor_128=pair, full_grid=full, n_full=args.n_full))
        print(f"  {label}: dt={dt_policy:.3g} l2 {d1:.3e} -> {d2:.3e} "
              f"(ratio {d1 / d2 if d2 > 0 else float('inf'):.2f}); "
              f"{args.n_full}^2 patterned_agree={full['patterned_agree']}", flush=True)

    agree = lambda key, rs: dict(
        n=len(rs), n_true=int(sum(bool(r[key]) for r in rs)),
        frac=(float(np.mean([bool(r[key]) for r in rs])) if rs else float("nan")))
    def summarise(rs):
        claimable = [r for r in rs if r["claim_scope_ok"]]
        return dict(
            n=len(rs), n_claim_scope_ok=len(claimable),
            patterned_agree=agree("patterned_agree", rs),
            kstar_within_one_bin=agree("kstar_within_one_bin", rs),
            kstar_within_one_bin_claimable=agree("kstar_within_one_bin", claimable),
            morphology_agree=agree("morphology_agree", rs),
            morphology_agree_claimable=agree("morphology_agree", claimable),
            l2_monotone=agree("l2_monotone", rs),
            l2_median_by_mu={str(mu): float(np.median([r["l2_diff_by_mu"][mu] for r in rs]))
                             for mu in V3_MUS} if rs else {},
            l2_max_by_mu={str(mu): float(np.max([r["l2_diff_by_mu"][mu] for r in rs]))
                          for mu in V3_MUS} if rs else {})

    out = dict(
        mus=V3_MUS, n=args.n, n_full=args.n_full, morph_per_cell=args.morph_per_cell,
        backend=backend, device=device, seed=args.seed,
        periods_per_box=V3_PERIODS_PER_BOX, population="harvest",
        rows=rows, full_rows=full_rows, anchor=anchor,
        summary=summarise(rows), summary_full=summarise(full_rows),
        anchor_summary=dict(
            n=len(anchor),
            dt_halving_ratio_median=(float(np.median([a["dt_halving_ratio"]
                                                      for a in anchor]))
                                     if anchor else float("nan")),
            full_grid_patterned_agree=dict(
                n=len(anchor),
                n_true=int(sum(bool(a["full_grid"]["patterned_agree"]) for a in anchor)))))
    stack = lambda rs, key: np.array([[r[key][mu] for mu in V3_MUS] for r in rs])
    arrays = dict(
        mus=np.array(V3_MUS, float),
        l2=stack(rows, "l2_diff_by_mu"), l2_dev=stack(rows, "l2_diff_dev_by_mu"),
        kstar_abs_diff=stack(rows, "kstar_abs_diff_by_mu"),
        one_bin=np.array([r["one_bin"] for r in rows]),
        kstar_qss=np.array([r["kstar_qss"] for r in rows]),
        labels=np.array(labels, dtype=object),
        full_l2=stack(full_rows, "l2_diff_by_mu"),
        full_kstar_abs_diff=stack(full_rows, "kstar_abs_diff_by_mu"),
        full_one_bin=np.array([r["one_bin"] for r in full_rows]),
        full_labels=np.array(sub_l, dtype=object))
    return _jsonable(out), arrays


def run_v4(args):
    hm, hl = harvest_models()
    dm, dl = d5_models()
    models, labels = hm + dm, hl + dl
    if args.limit:
        models, labels = models[:args.limit], labels[:args.limit]
    print(f"V4: {len(models)} systems, band [{MU_BIO_LO:g}, {MU_BIO_HI:g}], "
          f"n_mu={args.n_mu}, cloud={0 if args.no_robustness else V4_CLOUD}", flush=True)
    out = ladder.v4_survey(models, mu_band=(MU_BIO_LO, MU_BIO_HI), n_mu=args.n_mu,
                           labels=labels, robustness=not args.no_robustness,
                           robustness_n_samples=V4_CLOUD, seed=args.seed)
    mu = np.array(out["mu_grid"], float)
    arrays = dict(
        mu_grid=mu,
        turing=np.array([[e["lifted_turing_by_mu"][m] for m in out["mu_grid"]]
                         for e in out["edges"]], bool),
        kstar=np.array([[e["lifted_kstar_by_mu"][m] for m in out["mu_grid"]]
                        for e in out["edges"]]),
        mu_crit=np.array([e["mu_crit"] for e in out["edges"]]),
        qss_turing=np.array([e["qss_turing"] for e in out["edges"]], bool),
        reentrant=np.array([e["reentrant"] for e in out["edges"]], bool),
        labels=np.array(labels, dtype=object))
    if out["robustness"]:
        arrays["robust_frac_turing"] = np.array(
            [[row["frac_turing"] for row in r["rows"]] for r in out["robustness"]])
        arrays["robust_frac_turing_qss"] = np.array(
            [r["qss"]["frac_turing"] for r in out["robustness"]])
    return _jsonable(out), arrays


RUNGS = {"v0": run_v0, "v1": run_v1, "v2": run_v2, "v3": run_v3, "v4": run_v4}


# ======================================================================================
# io
# ======================================================================================
def _jsonable(o):
    """numpy -> python, and float dict keys -> str, so json.dump does not silently reorder
    or refuse. Applied at the boundary only; the rung functions return native objects."""
    if isinstance(o, dict):
        return {(f"{k:g}" if isinstance(k, float) else str(k)): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _provenance():
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return dict(git_commit=sha, host=platform.node(), torch_version=torch.__version__,
                cuda_available=bool(torch.cuda.is_available()),
                device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                             else None),
                date=time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rung", choices=sorted(RUNGS))
    ap.add_argument("--out", required=True, help="run directory, e.g. experiments/"
                                                 "lift_ladder/v3")
    ap.add_argument("--n", type=int, default=V3_N_ANCHOR,
                    help="V3(a) sweep grid over the whole population (default 128)")
    ap.add_argument("--n-full", type=int, default=V3_N_FULL,
                    help="V3 grid for the morphology claim and the V3(b) anchor (512)")
    ap.add_argument("--morph-per-cell", type=int, default=2,
                    help="systems per (form, D-scheme) cell in the n_full block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap the population (smoke runs)")
    ap.add_argument("--cpu", action="store_true", help="V3 on the numpy integrator")
    ap.add_argument("--n-mu", type=int, default=V4_N_MU)
    ap.add_argument("--no-robustness", action="store_true",
                    help="V4 without the 200-draw parameter cloud")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    payload, arrays = RUNGS[args.rung](args)
    payload = _jsonable(payload)
    payload["provenance"] = _provenance()
    payload["wall_seconds"] = time.perf_counter() - t0
    payload["rung"] = args.rung

    rp = out_dir / "results" / f"{args.rung}.json"
    with open(rp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {rp}")
    if arrays:
        ap_ = out_dir / "arrays" / f"{args.rung}.npz"
        np.savez_compressed(ap_, **arrays)
        print(f"wrote {ap_} ({os.path.getsize(ap_) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
