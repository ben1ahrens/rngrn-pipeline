#!/usr/bin/env python
"""run_domain_panel.py — DOMAIN-SIZE INDEPENDENCE panel for the lift paper demo.

WHAT THIS IS. A thin wrapper, not new machinery, per the instruction that spawned it: do not
touch `scripts/lift_paper_demo.py`'s semantics, extend it. That script hardwires the box to
`box_size(model)` at its default `periods=V3_PERIODS_PER_BOX=8` -- the target
`turing_labyrinth/sample_0000`'s own geometry. `lift_ladder.box_size(model, xstar=None,
periods=...)` already takes a `periods` kwarg, so this script calls the SAME primitives
(`box_size`, `rollout.simulate`, `eval.lifted_torch.simulate_lifted_torch`,
`observables.classify`, `ladder.one_radial_bin`) at `periods` in {4, 16} instead of 8, mirrors
`lift_paper_demo._demo_row`'s call pattern line for line, and pulls the existing p=8 row and
final frame from `../results/demo.json` / `../arrays/final_frames.npz` rather than re-running
it.

THE QUESTION. Do the learned parameters pattern at the SAME PHYSICAL WAVELENGTH in boxes the
training never saw? p = 8 was the training target's own geometry; p = 4 and p = 16 are boxes
nothing in training or in the original demo ever constructed. k* is compared in physical units
(rad / length) across all three box sizes, never in a box-relative unit, because the box itself
is what is being varied.

SCOPE. Same as the parent demo: n = 128 supports the pattern-formation statement (amplitude vs
floor, k* within one radial bin) and NOT the 512^2-only morphology-class equality claim
(REDESIGN SS5.3 V3) -- `morphology_claimable` is False here by construction, same as upstream.
mu = 1e-3 (mu_gate, D-REDESIGN-5) only, per the task brief -- this is a domain-size probe, not
a repeat of the mu sweep.

    .venv/bin/python experiments/lift_paper_demo/domain_size/run_domain_panel.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
PARENT = HERE.parent                      # experiments/lift_paper_demo
REPO = PARENT.parents[1]                  # repo root
sys.path.insert(0, str(REPO / "scripts"))

from lift_ladder import D5_TURING_SEEDS, box_size, d5_models  # noqa: E402
from lift_paper_demo import _panel, _prov_for_seed  # noqa: E402

from rngrn import observables as obs  # noqa: E402
from rngrn.eval import ladder, rollout  # noqa: E402
from rngrn.eval.lifted import MU_BIO_CENTRAL, MU_BIO_HI, MU_BIO_LO  # noqa: E402
from rngrn.eval.lifted_torch import simulate_lifted_torch  # noqa: E402

torch.set_default_dtype(torch.float64)

MU_GATE = 1e-3
DEMO_N = 128
DEMO_SEED = 0
NEW_PERIODS = (4, 16)          # the two box sizes training never saw
REFERENCE_PERIODS = 8          # pulled from the existing demo, not re-run
ALL_PERIODS = (4, 8, 16)
REF_RESULTS = PARENT / "results" / "demo.json"
REF_ARRAYS = PARENT / "arrays" / "final_frames.npz"


def _resolution_note(n: int, periods: int) -> dict:
    """points-per-wavelength = n / periods, independent of k* (dx = periods*wavelength/n,
    so wavelength/dx = n/periods algebraically -- verified per-row anyway rather than trusted
    as an identity, since a row's actual k* could disagree with the box's design k* if
    `box_size` and the box were ever computed from different xstar)."""
    ppw = n / periods
    return dict(points_per_wavelength=float(ppw), under_resolved=bool(ppw < 8.0))


def _domain_row(model, label, prov, periods, mu, n, seed, device):
    """One model, one NEW box size, one mu. Mirrors `lift_paper_demo._demo_row`'s call
    pattern exactly, with `periods` threaded into `box_size` instead of its default."""
    L, k_qss_linear, xstar = box_size(model, periods=periods)
    res = _resolution_note(n, periods)

    q = rollout.simulate(model, L=L, n=n, T=None, dt=None, seed=seed, noise=1e-2,
                         xstar=xstar, max_steps=200000)
    if q["blew_up"]:
        raise RuntimeError(f"{label} p={periods}: QSS control blew up at L={L}, n={n}, "
                           f"dt={q['dt']} -- no control to read the lifted arm against.")
    qf = q["fields"]
    bin_width = ladder.one_radial_bin(L)

    r = simulate_lifted_torch(model, L=L, mu=mu, n=n, T=None, dt=None, seed=seed,
                              noise=1e-2, xstar=xstar, max_steps=200000, device=device)
    f = r["fields"]
    d_sim = float(abs(r["kstar"] - q["kstar"]))
    d_lin = float(abs(r["kstar"] - k_qss_linear))

    row = dict(
        label=label, provenance=prov, form=model.form, N=int(model.N),
        periods=int(periods), L=float(L), n=int(n), seed=int(seed), device=device,
        resolution=res,
        kstar_qss_linear=float(k_qss_linear), one_radial_bin=float(bin_width),
        pattern_floor=float(max(1e-3, 0.02 * abs(xstar[0]))),
        qss=dict(patterned=bool(q["patterned"]), morphology=obs.classify(qf[0]),
                 kstar=float(q["kstar"]), amplitude=float(q["amplitude"]),
                 blew_up=bool(q["blew_up"]), stopped_reason=q["stopped_reason"],
                 nsteps_run=int(q["nsteps_run"]), seconds=float(q["seconds"])),
        lifted=dict(
            mu=float(mu),
            patterned=bool(r["patterned"]),
            morphology=(obs.classify(f[0]) if not r["blew_up"] else None),
            morphology_claimable=bool(n >= ladder.V3_MORPHOLOGY_MIN_N),
            morphology_agrees_qss=bool((not r["blew_up"])
                                       and obs.classify(f[0]) == obs.classify(qf[0])),
            kstar=float(r["kstar"]), amplitude=float(r["amplitude"]),
            kstar_abs_diff_vs_qss_sim=d_sim,
            kstar_abs_diff_vs_qss_linear=d_lin,
            kstar_within_one_bin_vs_qss_sim=bool(d_sim <= bin_width),
            kstar_within_one_bin_vs_qss_linear=bool(d_lin <= bin_width),
            patterned_agrees_qss=bool(bool(r["patterned"]) == bool(q["patterned"])),
            rel_l2_vs_qss=float(ladder._rel_l2(f, qf)),
            rel_l2_dev_vs_qss=float(ladder._rel_l2_dev(f, qf)),
            blew_up=bool(r["blew_up"]), stopped_reason=r["stopped_reason"],
            nsteps_run=int(r["nsteps_run"]), dt=float(r["dt"]),
            dt_over_mu=float(r["dt"] / mu), seconds=float(r["seconds"])))

    s = row["lifted"]
    print(f"  {label} p={periods} (L={L:.1f}, ppw={res['points_per_wavelength']:.1f}): "
          f"patterned={s['patterned']} (qss {row['qss']['patterned']}) "
          f"morph={s['morphology']}/{row['qss']['morphology']} "
          f"k*={s['kstar']:.5f} vs qss_sim {row['qss']['kstar']:.5f} "
          f"(|d|={d_sim:.5f}, bin={bin_width:.5f}) "
          f"{s['stopped_reason']} {s['nsteps_run']} steps ({s['seconds']:.0f}s)", flush=True)
    return row, dict(qss=qf, lifted=f)


def _load_reference(seeds):
    """Pull the p=8 row + final frame per seed from the existing demo, rather than re-run."""
    d = json.load(open(REF_RESULTS))
    npz = np.load(REF_ARRAYS)
    mu_key = f"{MU_GATE:g}"
    ref = {}
    for row in d["rows"]:
        s = int(row["provenance"]["seed"])
        if s not in seeds:
            continue
        label = row["label"].replace("/", "_")
        lifted_field = npz[f"{label}__lifted_mu{mu_key}"]
        qss_field = npz[f"{label}__qss"]
        ref[s] = dict(
            row=dict(
                label=row["label"], provenance=row["provenance"], form=row["form"],
                N=row["N"], periods=REFERENCE_PERIODS, L=row["L"], n=row["n"],
                seed=row["seed"], device=d["device"],
                resolution=_resolution_note(row["n"], REFERENCE_PERIODS),
                kstar_qss_linear=row["kstar_qss_linear"], one_radial_bin=row["one_radial_bin"],
                pattern_floor=row["pattern_floor"], qss=row["qss"],
                lifted=row["lifted"][mu_key],
                source="pulled from ../results/demo.json and ../arrays/final_frames.npz, "
                       "NOT re-run (p=8 is the parent demo's own box)"),
            fields=dict(qss=qss_field, lifted=lifted_field))
    return ref


# ======================================================================================
# figure: rows = models, cols = p in {4, 8, 16}
# ======================================================================================
def make_figure(model_labels, rows_by_p, fields_by_p, fig_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow, ncol = len(model_labels), len(ALL_PERIODS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow), squeeze=False)
    for i, label in enumerate(model_labels):
        for j, p in enumerate(ALL_PERIODS):
            row = rows_by_p[p][label]
            field = fields_by_p[p][label]["lifted"][0]
            s = row["lifted"]
            title = f"{label}  p={p}  L={row['L']:.1f}"
            sub = (f"{s['morphology']}   k*={s['kstar']:.3f}   amp={s['amplitude']:.3f}   "
                  f"{'PATTERNS' if s['patterned'] else 'NO PATTERN'}")
            im = _panel(axes[i][j], field, title, sub)
            fig.colorbar(im, ax=axes[i][j], fraction=0.046, pad=0.03)
    fig.suptitle(r"Domain-size independence: lifted field, $\mu$ = "
                f"{MU_GATE:g}, channel 0, across box sizes p (periods per box)",
                fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p_out = fig_dir / "domain_panel.png"
    fig.savefig(p_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p_out}", flush=True)
    return str(p_out)


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
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--n", type=int, default=DEMO_N)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--mu", type=float, default=MU_GATE)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(D5_TURING_SEEDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    if not (MU_BIO_LO <= args.mu <= MU_BIO_HI):
        raise ValueError(f"mu={args.mu:g} outside biological band "
                         f"[{MU_BIO_LO:g}, {MU_BIO_HI:g}]; refusing to run it silently.")

    out_dir = pathlib.Path(args.out)
    for sub in ("results", "arrays", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    models, labels = d5_models()
    keep = [(m, l, s) for m, l, s in zip(models, labels, D5_TURING_SEEDS) if s in args.seeds]
    model_labels = [l for _, l, _ in keep]
    print(f"domain-size panel: {len(keep)} models, mu={args.mu:g}, n={args.n}, "
          f"periods={NEW_PERIODS} (+ p={REFERENCE_PERIODS} pulled from parent demo), "
          f"device={args.device}", flush=True)

    t0 = time.perf_counter()
    rows_by_p = {p: {} for p in NEW_PERIODS}
    fields_by_p = {p: {} for p in NEW_PERIODS}
    arrays = {}
    for m, label, s in keep:
        for p in NEW_PERIODS:
            row, fields = _domain_row(m, label, _prov_for_seed(s), p, args.mu, args.n,
                                      args.seed, args.device)
            rows_by_p[p][label] = row
            fields_by_p[p][label] = fields
            key = label.replace("/", "_")
            arrays[f"{key}__p{p}__qss"] = fields["qss"].astype(np.float32)
            arrays[f"{key}__p{p}__lifted_mu{args.mu:g}"] = fields["lifted"].astype(np.float32)

    ref = _load_reference(set(args.seeds))
    rows_by_p[REFERENCE_PERIODS] = {}
    fields_by_p[REFERENCE_PERIODS] = {}
    for s, entry in ref.items():
        label = f"d5/seed{s}"
        rows_by_p[REFERENCE_PERIODS][label] = entry["row"]
        fields_by_p[REFERENCE_PERIODS][label] = entry["fields"]

    fig_path = make_figure(model_labels, rows_by_p, fields_by_p, out_dir / "figures", args.dpi)

    # per-model drift summary across the three box sizes, in bins AND as % of k*
    drift = {}
    for label in model_labels:
        ref_row = rows_by_p[REFERENCE_PERIODS][label]
        k_ref = ref_row["lifted"]["kstar"]
        entry = {}
        for p in ALL_PERIODS:
            row = rows_by_p[p][label]
            k = row["lifted"]["kstar"]
            bin_w = row["one_radial_bin"]
            d_abs = float(abs(k - k_ref))
            entry[f"p{p}"] = dict(
                kstar=float(k), L=float(row["L"]),
                patterned=bool(row["lifted"]["patterned"]),
                morphology=row["lifted"]["morphology"],
                amplitude=float(row["lifted"]["amplitude"]),
                pattern_floor=float(row["pattern_floor"]),
                points_per_wavelength=row["resolution"]["points_per_wavelength"],
                delta_kstar_vs_p8=d_abs,
                delta_kstar_vs_p8_in_bins=float(d_abs / bin_w),
                delta_kstar_vs_p8_pct=float(100.0 * d_abs / k_ref) if k_ref else None)
        drift[label] = entry

    payload = dict(
        what="domain-size independence panel: same recovered D5 models, lifted at "
             f"mu={args.mu:g}, boxes p in {ALL_PERIODS} periods (p=8 pulled from the "
             "parent demo, not re-run)",
        parent_demo=str(REF_RESULTS.relative_to(REPO)),
        population="d5", mu=float(args.mu), n=int(args.n), seed=int(args.seed),
        device=args.device, periods=list(ALL_PERIODS), periods_run=list(NEW_PERIODS),
        periods_reference=REFERENCE_PERIODS,
        mu_band=[float(MU_BIO_LO), float(MU_BIO_HI)],
        morphology_claim_min_n=int(ladder.V3_MORPHOLOGY_MIN_N),
        scope_note=("n=128 supports the pattern-formation statement (amplitude vs floor, "
                    "k* within one radial bin) and NOT the 512^2-only morphology-class "
                    "equality claim (REDESIGN SS5.3 V3); morphology labels here are "
                    "descriptive. p=16 at n=128 has points-per-wavelength exactly 8.0 -- "
                    "at, not above, the resolution floor; see README for the explicit "
                    "boundary note."),
        rows_by_p={str(p): rows_by_p[p] for p in ALL_PERIODS},
        drift_vs_p8=drift,
        figures=[str(pathlib.Path(fig_path).relative_to(out_dir))],
        provenance=_provenance(),
        wall_seconds=time.perf_counter() - t0)

    rp = out_dir / "results" / "results.json"
    with open(rp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {rp}")
    npz = out_dir / "arrays" / "final_frames.npz"
    np.savez_compressed(npz, **arrays)
    print(f"wrote {npz}")
    print(f"total {payload['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
