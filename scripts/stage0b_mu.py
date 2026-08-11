#!/usr/bin/env python
"""stage0b_mu.py — does the Turing pattern survive when the QSS reduction is UNDONE?

WHICH `mu`. The QSS lift's FAST-VARIABLE TIMESCALE (eval/dynamical.py::lift_check,
eval/lifted.py), i.e. TF-promoter binding relative to protein turnover. NOT the generator's
degradation rate, which is the RNGRN's `delta` and is what scripts/stage0_bio_viability.py
swept. docs/STATE_OF_THE_SCIENCE.md records the two-meanings ambiguity; this script is
entirely about the first.

SUBCOMMANDS
  fixedpoint  the lifted fixed point is the QSS one at every mu, and sigma_full -> sigma_QSS
              as mu -> 0. The correctness check on the whole construction.
  mucrit      mu_crit per system over the Stage-0 harvest survivors + the two recovered
              networks, by log-scan + bisection, recording WHICH condition fails.
  dtconv      dt-halving convergence of the lifted PDE integrator, and its agreement with
              the QSS rollout at mu = 1e-4.
  ladder      64x64 QSS field | lifted field at each mu of a ladder, both forms.
  recovered   the same ladder for stage0 part-4 sample_0003/0004, prior-ON.

Everything writes to experiments/figures_report/stage0b/arrays/. Figures:
scripts/stage0b_figures.py.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from rngrn.eval import lifted, rollout                          # noqa: E402
from rngrn.io import load_checkpoint                            # noqa: E402
from rngrn.losses.terms import steady_state                     # noqa: E402
from rngrn.scoring import morphology                            # noqa: E402
from stage0_bio_viability import (                              # noqa: E402
    ARR as STAGE0_ARR, balanced_picks, build_model, params_at)

OUT = "experiments/figures_report/stage0b"
A = os.path.join(OUT, "arrays")

# THE mu LADDER. Six decades, straddling every mu_crit measured here and the biological
# band (docs/TIMESCALE_MU.md §2: mu_bio ~ 1e-5..2e-2, centred ~1e-3).
MU_LADDER = (1e-4, 1e-3, 1e-2, 1e-1, 1e0)
GRID_N = 64
PERIODS = 6            # same domain policy as stage0_bio_viability.py part3
N_PER_FORM = 4

RUNS = "experiments/stage0_bioviab/runs"
RECOVERED = {"sample_0003": "stage0_prior_on_0003_20260803_130206",
             "sample_0004": "stage0_prior_on_0004_20260803_130302"}


def _harvest():
    return np.load(os.path.join(STAGE0_ARR, "part1_harvest_generator.npz"))


def _system(z, form, scheme, i):
    p = params_at(z, form, scheme, i)
    m = build_model(p["alpha"], p["delta"], p["beta"], p["D"], p["s"], p["gate_logit"], form)
    xs, ok = steady_state(m)
    return m, xs.detach().cpu().numpy(), bool(ok)


def _recovered(sample):
    model, payload = load_checkpoint(os.path.join(RUNS, RECOVERED[sample]))
    xs, ok = steady_state(model)
    return model, xs.detach().cpu().numpy(), bool(ok), payload


# ======================================================================================
# fixedpoint — the correctness check
# ======================================================================================
def cmd_fixedpoint(args):
    os.makedirs(A, exist_ok=True)
    z = _harvest()
    mus_fp = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 1e3])
    mus_cv = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    rows, curves = [], {}
    for form in ("competitive", "nc1"):
        for scheme, i in balanced_picks(z, form, N_PER_FORM):
            m, xstar, conv = _system(z, form, scheme, i)
            sq, oq = lifted.qss_dispersion(m, xstar)
            # the k-window the convergence is asserted on: k where the SLOW branch is what
            # sigma reports. At very large k the QSS branch is -D_min k^2 -> -1e7 while the
            # lifted one saturates at -1/mu (the gates do not diffuse), so the two are
            # STRUCTURALLY different there and comparing them is meaningless. Restrict to
            # |sigma_QSS(k)| <= 10, which contains k* and the whole unstable band.
            sel = np.abs(sq) <= 10.0
            fp = [lifted.fixed_point_residual(m, xstar, float(mu)) for mu in mus_fp]
            dev = []
            for mu in mus_cv:
                sl, _ = lifted.lifted_dispersion(m, xstar, float(mu))
                dev.append(float(np.max(np.abs(sl[sel] - sq[sel]))))
            rows.append(dict(form=form, scheme=scheme, idx=int(i), converged=conv,
                             fp_residual=fp, dev=dev, n_k_window=int(sel.sum())))
            print(f"  {form:12s} {scheme:9s} idx{i:<3d} fp_res_max={max(fp):.3e}  "
                  f"dev@mu={dict(zip(mus_cv, np.round(dev, 8)))}", flush=True)
            if form not in curves:
                # the central physics figure's data: sigma_full(k) at several mu, QSS overlaid
                sig = {}
                om = {}
                for mu in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                    s_, o_ = lifted.lifted_dispersion(m, xstar, mu)
                    sig[mu] = s_
                    om[mu] = o_
                curves[form] = dict(scheme=scheme, idx=int(i), sigma_qss=sq, omega_qss=oq,
                                    sigma=sig, omega=om, xstar=xstar)
    flat = dict(kgrid=lifted.KGRID, mus_fp=mus_fp, mus_cv=mus_cv,
                fp_residual=np.array([r["fp_residual"] for r in rows]),
                dev=np.array([r["dev"] for r in rows]),
                form=np.array([r["form"] for r in rows]),
                scheme=np.array([r["scheme"] for r in rows]),
                idx=np.array([r["idx"] for r in rows]))
    for form, c in curves.items():
        flat[f"curve__{form}__sigma_qss"] = c["sigma_qss"]
        flat[f"curve__{form}__omega_qss"] = c["omega_qss"]
        flat[f"curve__{form}__scheme"] = c["scheme"]
        flat[f"curve__{form}__idx"] = c["idx"]
        flat[f"curve__{form}__mus"] = np.array(sorted(c["sigma"]))
        flat[f"curve__{form}__sigma"] = np.stack([c["sigma"][mu] for mu in sorted(c["sigma"])])
        flat[f"curve__{form}__omega"] = np.stack([c["omega"][mu] for mu in sorted(c["omega"])])
    np.savez_compressed(os.path.join(A, "fixedpoint.npz"), **flat)
    print(f"\nwrote {A}/fixedpoint.npz")
    print(f"WORST fixed-point residual over {len(rows)} systems x {len(mus_fp)} mu: "
          f"{np.max(flat['fp_residual']):.3e}")
    for j, mu in enumerate(mus_cv):
        print(f"  mu={mu:8.1e}  worst max|sigma_full - sigma_QSS| over the |sigma|<=10 "
              f"window = {flat['dev'][:, j].max():.3e}")


# ======================================================================================
# mucrit
# ======================================================================================
def cmd_mucrit(args):
    os.makedirs(A, exist_ok=True)
    z = _harvest()
    recs, scans = [], []
    t0 = time.time()
    jobs = []
    for form in ("competitive", "nc1"):
        for scheme, i in balanced_picks(z, form, 12):
            jobs.append((form, scheme, int(i), None))
    for s in RECOVERED:
        jobs.append(("recovered", "prior_on", -1, s))
    for form, scheme, i, sample in jobs:
        if sample is None:
            m, xstar, conv = _system(z, form, scheme, i)
            label = f"{form}/{scheme}/{i}"
            fam = form
        else:
            m, xstar, conv, _ = _recovered(sample)
            label = f"recovered/{sample}"
            fam = m.form
        r = lifted.mu_critical(m, xstar, lo=args.lo, hi=args.hi,
                               per_decade=args.per_decade)
        v0 = lifted.turing_verdict_lifted(m, xstar, args.lo)
        recs.append(dict(label=label, family=fam, form=form, scheme=scheme, idx=i,
                         sample=sample or "", converged=conv, status=r["status"],
                         mu_crit=r["mu_crit"], mu_lo=r["mu_lo"], mu_hi=r["mu_hi"],
                         failure=r["failure"], reentrant=bool(r["reentrant"]),
                         qss_kstar=v0["kstar"], qss_sig=v0["sig_max_pos"],
                         qss_max_re_eig_J=v0["max_re_eig_J"],
                         omega_uniform_at_hi=float(
                             r.get("verdict_hi", {}).get("omega_uniform", np.nan)),
                         omega_kstar_at_hi=float(
                             r.get("verdict_hi", {}).get("omega_at_kstar", np.nan))))
        scans.append(r["scan_ok"])
        print(f"  {label:28s} {r['status']:14s} mu_crit={r['mu_crit']:.4g} "
              f"[{r['mu_lo']:.4g},{r['mu_hi']:.4g}] fail={r['failure']:8s} "
              f"reentrant={r['reentrant']!s:5s}  {time.time()-t0:.1f}s", flush=True)
    scan_mu = 10.0 ** np.arange(np.log10(args.lo), np.log10(args.hi) + 1e-12,
                                1.0 / args.per_decade)
    np.savez_compressed(os.path.join(A, "mucrit.npz"), scan_mu=scan_mu,
                        scan_ok=np.array(scans),
                        **{k: np.array([r[k] for r in recs]) for k in recs[0]})
    with open(os.path.join(A, "mucrit.json"), "w") as fh:
        json.dump(recs, fh, indent=2, default=str)
    print(f"\nwrote {A}/mucrit.npz + .json  ({len(recs)} systems, lo={args.lo}, "
          f"hi={args.hi}, {args.per_decade}/decade, bisection tol_rel=0.02)")


# ======================================================================================
# the PDE: shared helpers
# ======================================================================================
def _L_for(model, xstar, kstar_hint=None):
    """PERIODS wavelengths of the QSS fastest-growing mode, the same policy stage0 part3
    used, so the lifted and QSS runs and the Stage-0 gallery all share a domain size."""
    if kstar_hint is None or not np.isfinite(kstar_hint) or kstar_hint <= 0:
        v = lifted.turing_verdict_lifted(model, xstar, 1e-6)
        kstar_hint = v["kstar"]
    return float(PERIODS * 2.0 * np.pi / kstar_hint)


def _field_stats(F, xstar, L):
    from rngrn import observables as obs
    std_c = F.std(axis=(1, 2))
    mean_c = F.mean(axis=(1, 2))
    chan = int(np.argmax(std_c))
    floors = np.maximum(1e-3, 0.02 * np.abs(mean_c))
    try:
        call = morphology.classify_morphology(F[chan], morphology.default_reference_bank())
        morph, margin = call.label, float(call.margin)
    except ValueError:
        morph, margin = "undefined", float("nan")
    return dict(std_c=std_c, chan=chan, patterned_c0=bool(std_c[0] > floors[0]),
                patterned_any=bool(np.any(std_c > floors)), morphology=morph,
                morph_margin=margin, kstar_field=float(obs.kstar_of(F[chan], L=L)))


# ======================================================================================
# dtconv — the integrator's convergence evidence
# ======================================================================================
def cmd_dtconv(args):
    os.makedirs(A, exist_ok=True)
    z = _harvest()
    out = {}
    rows = []
    for form in ("competitive", "nc1"):
        scheme, i = balanced_picks(z, form, 1)[0]
        m, xstar, _ = _system(z, form, scheme, i)
        L = _L_for(m, xstar)
        for mu in (1e-4, 1e-2):
            base = None
            for half in (1, 2, 4):
                r = lifted.simulate_lifted(m, L=L, mu=mu, n=GRID_N, seed=args.seed,
                                           dt=None if half == 1 else base_dt / half)
                if half == 1:
                    base_dt = r["dt"]
                    base = r["fields"]
                st = _field_stats(np.asarray(r["fields"], float), xstar, L)
                d = (float(np.max(np.abs(r["fields"] - base)))
                     / max(1e-300, float(np.abs(base).max())))
                rows.append(dict(form=form, mu=mu, dt_div=half, dt=r["dt"],
                                 nsteps=r["nsteps_run"], seconds=r["seconds"],
                                 amplitude=r["amplitude"], kstar=st["kstar_field"],
                                 std_max=float(st["std_c"].max()),
                                 morphology=st["morphology"],
                                 rel_field_dev=d,
                                 stopped_reason=r["stopped_reason"]))
                print(f"  {form:12s} mu={mu:.0e} dt/{half}: dt={r['dt']:.4g} "
                      f"steps={r['nsteps_run']} {r['seconds']:.1f}s amp={r['amplitude']:.5g} "
                      f"k*={st['kstar_field']:.4g} morph={st['morphology']} "
                      f"rel|dF|={d:.3e}", flush=True)
                out[f"{form}__mu{mu:.0e}__dt{half}"] = np.asarray(r["fields"], float)
        # ---- the mu -> 0 cross-check against the QSS rollout, SAME seed/noise/L/dt ------
        rq = rollout.simulate(m, L=L, n=GRID_N, integrator="etdrk4_rfft", seed=args.seed)
        rl = lifted.simulate_lifted(m, L=L, mu=1e-6, n=GRID_N, seed=args.seed)
        sq = _field_stats(np.asarray(rq["fields"], float), xstar, L)
        sl = _field_stats(np.asarray(rl["fields"], float), xstar, L)
        d = (float(np.max(np.abs(rl["fields"] - rq["fields"])))
             / max(1e-300, float(np.abs(rq["fields"]).max())))
        rows.append(dict(form=form, mu=1e-6, dt_div=0, dt=rl["dt"],
                         nsteps=rl["nsteps_run"], seconds=rl["seconds"],
                         amplitude=rl["amplitude"], kstar=sl["kstar_field"],
                         std_max=float(sl["std_c"].max()), morphology=sl["morphology"],
                         rel_field_dev=d, stopped_reason=rl["stopped_reason"]))
        out[f"{form}__qss"] = np.asarray(rq["fields"], float)
        out[f"{form}__lift_mu1e-06"] = np.asarray(rl["fields"], float)
        print(f"  {form:12s} QSS vs lifted@mu=1e-6: rel|dF|={d:.3e}  "
              f"k* {sq['kstar_field']:.5g} vs {sl['kstar_field']:.5g}  "
              f"amp {rq['amplitude']:.5g} vs {rl['amplitude']:.5g}  "
              f"morph {sq['morphology']} vs {sl['morphology']}  "
              f"({rq['seconds']:.1f}s vs {rl['seconds']:.1f}s)", flush=True)
    np.savez_compressed(os.path.join(A, "dtconv.npz"), grid_n=GRID_N, seed=args.seed,
                        **out, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    with open(os.path.join(A, "dtconv.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\nwrote {A}/dtconv.npz + .json")


# ======================================================================================
# ladder / recovered — the visual verification
# ======================================================================================
def _ladder(systems, tag, seed):
    recs, fields = [], []
    t0 = time.time()
    for label, m, xstar, kstar_hint in systems:
        L = _L_for(m, xstar, kstar_hint)
        rq = rollout.simulate(m, L=L, n=GRID_N, integrator="etdrk4_rfft", seed=seed)
        base = np.asarray(rq["fields"], float)
        for mu, r in [(0.0, rq)] + [
                (mu, lifted.simulate_lifted(m, L=L, mu=mu, n=GRID_N, seed=seed))
                for mu in MU_LADDER]:
            F = np.asarray(r["fields"], float)
            st = _field_stats(F, xstar, L)
            v = (lifted.turing_verdict_lifted(m, xstar, mu) if mu > 0 else
                 lifted.turing_verdict_lifted(m, xstar, 1e-6))
            recs.append(dict(label=label, mu=float(mu), L=L, form=m.form,
                             patterned_c0=st["patterned_c0"],
                             patterned_any=st["patterned_any"],
                             kstar_field=st["kstar_field"], kstar_lin=v["kstar"],
                             sig_max_pos=v["sig_max_pos"],
                             turing_strict=v["turing_strict"],
                             stable_uniform=v["stable_uniform"],
                             amplitude=float(r["amplitude"]),
                             std_c0=float(st["std_c"][0]), std_c1=float(st["std_c"][1]),
                             std_c2=float(st["std_c"][2]),
                             std_max=float(st["std_c"].max()), chan=st["chan"],
                             morphology=st["morphology"],
                             morph_margin=st["morph_margin"],
                             rel_dev_vs_qss=float(np.max(np.abs(F - base)))
                             / max(1e-300, float(np.abs(base).max())),
                             stopped_reason=r["stopped_reason"],
                             nsteps_run=int(r["nsteps_run"]), dt=float(r["dt"]),
                             seconds=float(r["seconds"])))
            fields.append(F)
            print(f"  {label:26s} mu={mu:8.1e} patt_any={st['patterned_any']!s:5s} "
                  f"k*={st['kstar_field']:8.4g} amp={r['amplitude']:.5g} "
                  f"stdmax={st['std_c'].max():.5g} morph={st['morphology']:10s} "
                  f"stop={r['stopped_reason']:11s} turing={v['turing_strict']!s:5s} "
                  f"({r['seconds']:.1f}s, {time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(A, f"{tag}.npz"), fields=np.stack(fields),
                        grid_n=GRID_N, seed=seed, mu_ladder=np.array(MU_LADDER),
                        **{k: np.array([r[k] for r in recs]) for k in recs[0]})
    with open(os.path.join(A, f"{tag}.json"), "w") as fh:
        json.dump(recs, fh, indent=2, default=str)
    print(f"\nwrote {A}/{tag}.npz + .json  ({len(recs)} fields)")


def cmd_ladder(args):
    os.makedirs(A, exist_ok=True)
    z = _harvest()
    systems = []
    for form in ("competitive", "nc1"):
        for scheme, i in balanced_picks(z, form, N_PER_FORM):
            m, xstar, _ = _system(z, form, scheme, i)
            systems.append((f"{form[:4]}/{scheme}/{i}", m, xstar,
                            float(z[f"{form}__{scheme}__kstar"][i])))
    _ladder(systems, "ladder", args.seed)


def cmd_recovered(args):
    os.makedirs(A, exist_ok=True)
    systems = []
    for s in RECOVERED:
        m, xstar, conv, payload = _recovered(s)
        print(f"  {s}: form={m.form} N={m.N} converged={conv} "
              f"D={np.round(m.D.detach().numpy(), 5)} L_train={payload.get('L')}")
        systems.append((f"rec/{s}", m, xstar, None))
    _ladder(systems, "recovered", args.seed)


# ======================================================================================
# robust — THE DELIVERABLE: is the circuit still ROBUST at finite mu?
# ======================================================================================
# WHICH FINITE mu, and why exactly these two. The scope is a robustness statement AT a
# stated finite mu, not a study of the mu axis, so two values are named and defended and
# the rest of the axis is reported only for context.
#   mu = 1e-3  HEADLINE. mu is (TF-promoter binding time)/(protein turnover time), both in
#              the same units, so it is dimensionless. Live-cell single-molecule tracking
#              puts specific TF-DNA dwell times at ~10 s (Chen et al. 2014, Cell 156:1274 —
#              Sox2 12.0-14.6 s), and measured morphogen clearance half-lives in patterning
#              tissue are ~2-4 h (Muller et al. 2012, Science 336:721 — Nodal/Lefty 95-218
#              min), i.e. mean lifetimes ~2e4 s. 10/2e4 = 5e-4; with the model's own delta
#              of order 1 setting one time unit to one protein lifetime (configs/bio_box.yaml
#              delta row [0.4, 5.0]), 1e-3 is that estimate rounded up one notch.
#   mu = 1e-2  STRESS. One decade slower gates than the estimate, i.e. binding an order of
#              magnitude less separated from turnover than measured. If the verdict holds
#              here it holds comfortably at any defensible mu.
MU_FINITE = (1e-3, 1e-2)
# Context only (demoted per the narrowed scope): the same cloud read across the mu axis.
# It costs one eigen-scan per point because lifted.rescale_mu is exact, so it is free.
MU_CONTEXT = np.logspace(-6.0, 2.0, 17)
ROBUST_LEVELS = ("4p8pct", "10pct")


def cmd_robust(args):
    """turing_volume_{4.8,10}pct on the LIFTED Jacobian at finite mu.

    The perturbation model is lifted.draw_param_cloud: lognormal multiplicative factors on
    the PHYSICAL kinetic parameters, x* and the Jacobian re-derived per draw, strict
    max Re eig(J_full) < 0 for the uniform state. See that docstring for why the reduced-J
    scheme of eval/analysis.py::_draw_JD_cloud cannot be used on a lifted system, and why
    every number here is read against its OWN mu -> 0 column rather than against the QSS
    tables in docs/ROBUSTNESS_MEASUREMENT.md §4.2.

    A Turing-unstable mode whose leading eigenvalue has Im != 0 is a travelling wave, not
    the stationary pattern this project claims. `stationary` is reported beside `turing`
    everywhere and the two are never merged.
    """
    os.makedirs(A, exist_ok=True)
    z = _harvest()
    systems = []
    for form in ("competitive", "nc1"):
        for scheme, i in balanced_picks(z, form, N_PER_FORM):
            m, xstar, _ = _system(z, form, scheme, i)
            systems.append((f"{form[:4]}/{scheme}/{i}", m))
    for s in RECOVERED:                       # the two networks carrying the claim
        m, _x, _c, _p = _recovered(s)
        systems.append((f"rec/{s}", m))

    rows, flat = [], {}
    t0 = time.time()
    for label, m in systems:
        for lvl in ROBUST_LEVELS:
            sigma = lifted.CLOUD_SIGMA_LEVELS[lvl]
            r = lifted.robustness_vs_mu(m, MU_CONTEXT, sigma_log=sigma,
                                        n_samples=args.n_samples, seed=args.seed)
            vol = np.array([q["frac_turing"] for q in r["rows"]])
            sta = np.array([q["frac_stationary"] for q in r["rows"]])
            osc = np.array([q["frac_oscillatory"] for q in r["rows"]])
            key = f"{label}__{lvl}".replace("/", "_")
            flat[f"{key}__vol"] = vol
            flat[f"{key}__stationary"] = sta
            flat[f"{key}__oscillatory"] = osc
            flat[f"{key}__fail_uniform"] = np.array(
                [q["frac_fail_uniform"] for q in r["rows"]])
            flat[f"{key}__fail_band"] = np.array([q["frac_fail_band"] for q in r["rows"]])
            at = {mu: int(np.argmin(np.abs(np.log(MU_CONTEXT / mu)))) for mu in MU_FINITE}
            row = dict(label=label, form=m.form, level=lvl, sigma_log=sigma,
                       bar=lifted.PREREG_BARS[lvl], n=r["n"], n_converged=r["n_converged"],
                       recovered=label.startswith("rec/"),
                       vol_qss=r["qss"]["frac_turing"],
                       stationary_qss=r["qss"]["frac_stationary"],
                       vol_mu0=float(vol[0]))
            for mu, j in at.items():
                row[f"vol_mu{mu:g}"] = float(vol[j])
                row[f"stationary_mu{mu:g}"] = float(sta[j])
                row[f"oscillatory_mu{mu:g}"] = float(osc[j])
                row[f"pass_mu{mu:g}"] = bool(vol[j] >= lifted.PREREG_BARS[lvl])
            # CONTEXT ONLY: how far the volume moves anywhere on the swept axis. A number,
            # not a study — it exists so "flat in mu" is a measurement and not an impression.
            row["vol_max_dev_from_qss"] = float(np.max(np.abs(vol - r["qss"]["frac_turing"])))
            row["oscillatory_max_over_axis"] = float(osc.max())
            rows.append(row)
            print(f"  {label:26s} {lvl:7s} QSS={row['vol_qss']:.3f} "
                  f"mu=1e-3:{row['vol_mu0.001']:.3f} mu=1e-2:{row['vol_mu0.01']:.3f} "
                  f"stat(1e-3)={row['stationary_mu0.001']:.3f} "
                  f"osc_max={row['oscillatory_max_over_axis']:.3f} "
                  f"maxdev={row['vol_max_dev_from_qss']:.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)

    np.savez_compressed(os.path.join(A, "robust.npz"), mu_context=MU_CONTEXT,
                        mu_finite=np.array(MU_FINITE), n_samples=args.n_samples,
                        seed=args.seed, **flat)
    with open(os.path.join(A, "robust.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\nwrote {A}/robust.npz + .json")

    print(f"\nHEADLINE — strict Turing volume on the LIFTED Jacobian at finite mu "
          f"(n={args.n_samples} draws/cell, seed={args.seed}, N=3, n_hill=2)")
    print(f"{'system':26s} {'lvl':7s} {'bar':>4s} {'QSS':>6s} {'mu=1e-3':>8s} "
          f"{'mu=1e-2':>8s} {'stat@1e-3':>9s} {'PASS@1e-3':>9s}")
    for r in rows:
        print(f"{r['label']:26s} {r['level']:7s} {r['bar']:4.2f} {r['vol_qss']:6.3f} "
              f"{r['vol_mu0.001']:8.3f} {r['vol_mu0.01']:8.3f} "
              f"{r['stationary_mu0.001']:9.3f} "
              f"{'yes' if r['pass_mu0.001'] else 'NO':>9s}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fixedpoint", "mucrit", "robust", "dtconv", "ladder",
                                    "recovered"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--lo", type=float, default=1e-6)
    ap.add_argument("--hi", type=float, default=1e4)
    ap.add_argument("--per-decade", type=int, default=8)
    args = ap.parse_args(argv)
    os.chdir(os.path.join(HERE, ".."))
    {"fixedpoint": cmd_fixedpoint, "mucrit": cmd_mucrit, "robust": cmd_robust,
     "dtconv": cmd_dtconv, "ladder": cmd_ladder, "recovered": cmd_recovered}[args.cmd](args)


if __name__ == "__main__":
    main()
