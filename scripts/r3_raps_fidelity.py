"""r3_raps_fidelity.py — Task 15 (PLAN_redesign_R3): 96^2-vs-512^2 RAPS band fidelity,
the measurement that may move `n` (`docs/REDESIGN_rngrn.md` §4.3).

THE GAP §4.3 NAMES. Training compares a 96^2 MODEL spectrum (solved on the commensurate
box `solve_box.geometry`, L_solve = p*2*pi/k_hat, p=8, n=96) to a 512^2 OBSERVED spectrum,
with "the model-side estimator's fidelity UNMEASURED at this grid": D2 measured band-limited
agreement with 512^2 at 256^2, not at 96^2 (`docs/DIAGNOSTICS_fft.md` D2). This script
measures that gap as a band-shape DISTANCE (not just D2's one-bin k* pass/fail), comparable
to D3's ~31%/bin estimation floor.

WHAT IS HELD FIXED, WHAT VARIES. The role of "the observed 512^2 frame" is played by the
SAME fixture checkpoint's own 512^2 rollout on the IDENTICAL commensurate box (same L, same
dynamics, same seed) as the 96^2 (and 128^2, 256^2) rollouts being tested. This isolates the
MODEL-SIDE ESTIMATOR's fidelity in isolation from the cross-domain gap: does representing
one pattern at n=96 (12 px/wavelength) reproduce the RAPS band shape a finely resolved n=512
representation of the SAME box shows? That is exactly the quantity §4.3 flags as unmeasured.
The SEPARATE cross-domain gap — interpolating a real dataset's 512^2 OBSERVED frame (a much
larger physical box, unrelated grid) onto the solve box — is Task 11's job, already measured
(`experiments/redesign_r3/solve_box_interp/results/interp_error.json`) and folded into this
report's summary per §4.3's explicit pairing instruction ("measured once per re-tile
geometry at R3, alongside the FD check").

NO payload.h5 IS OPENED — same reasoning as `scripts/diag_fft_d2.py`, whose fixture and
rollout apparatus this reuses directly (`load_fixture`, `linear_kstar`): every fixture is a
tracked RECOVERED checkpoint (a recovery OUTPUT, not the generator or the answer key). No
`tests/test_firewall.py::FORBIDDEN` entry is therefore needed.

MULTI-FIXTURE SWEEP (added after the controller held the single-fixture n-verdict for more
evidence: 1 fixture x 3 seeds could not separate a genuine n-trend from seed noise — n=128's
mean did not beat n=256's). `FIXTURES` lists every Turing-unstable TRACKED checkpoint in
this repo (verified by grepping `git ls-files` for `checkpoints/model.pt` and checking each
with `linear_kstar`; 5 of 11 tracked checkpoints are Turing-unstable — the other 6 collapse
to the k-grid floor, D5's own finding). Two of those 5 land within 0.3% of each other's
k_lin (`d5_seed1` and `d5_seed6`, both descended from the same D5 sweep and evidently
converging to a similar dispersion peak despite different sign structures) — `d5_seed1` is
dropped in favour of the more different `d5_seed3`/`d5_seed5` pair, giving 3 fixtures at
k_lin ~ 0.175 / 0.284 / 0.283. This is every distinct Turing-unstable fixture this repo has
tracked; it is not a free choice among many.

Run one fixture per process (`--tag`), in parallel — `run_all_fixtures.sh` launches all
three. Each writes `.../results/raps_fidelity_<tag>.json`; `--aggregate` (no `--tag`) then
merges every `raps_fidelity_*.json` present into the combined `raps_fidelity.json`.

METHOD, per (fixture, grid, seed):
  1. `diag_fft_d2.load_fixture` + `linear_kstar` → the fixture model and its own dispersion
     argmax k_lin. L = p*2*pi/k_lin (p=8, `solve_box.P_DEFAULT`), fixed across every grid
     for that fixture.
  2. `eval.rollout.simulate` at n in {96, 128, 256, 512}, `early_stop=True` (saturated
     patterns, not mid-transient), same L. The n=512 solve is REQUIRED per seed regardless
     of grid count — it is the reference every other grid is compared against, not merely
     "a grid being tested" — so it cannot be skipped; what IS skipped is treating it as
     an open question (n=512-vs-itself is asserted ~0 every time, cheaply, not re-derived).
  3. `observables.raps` (register item 15's UNWINDOWED primary estimator; the Hann-Welch
     path stays report-only under item 15's ratified both-sides-identical rule) of each
     field. `k_lin` (the fixture's own dispersion argmax, the same k that fixed L in step
     1) plays "k*_obs" here — B_train = [0.60, 1.55]*k*_obs, D-FFT-9 closure 1 — since
     this self-consistency test has no independent "observed" domain (see caveat below);
     each seed's OWN nonlinearly-realised k* is reported separately (drifts a few % from
     k_lin — a real, secondary finding, not folded into the distance metric).
  4. `solve_box.geometry(k_lin)` — the SAME (p=8) box a real training step would build at
     the idealised post-ignition operating point (k_hat == k*_obs); using k_lin keeps
     `L_solve` equal to the field's real L exactly, so the box's bin array aligns
     index-for-index with `observables.raps(field_model, L)`'s own array. `solve_box.
     band_bins` selects its B_train bins (bin WIDTH dk = k_lin/p depends only on p, not
     n, so the band's bin COUNT is identical across every grid tested — Task 11's
     `n_band_bins=7` for this p, reproduced). Two earlier, WRONG forms of this script used
     the per-seed measured k* for the box geometry (misaligns the bin array with L) or as
     `interpolate_targets`'s `kstar_obs` while the box used k_lin (reads the k_lin-vs-
     measured-k* drift as spurious distance) — both caught by the sanity check that n=512
     compared against itself must give ~0 distance; see `raps_band_distance`'s docstring.
  5. `solve_box.interpolate_targets` — the 512^2 RAPS's log-power, sampled on the FINE
     grid, interpolated onto the box's scale-free band-bin positions: the ACTUAL target
     construction the R3 forward-solve geometry performs (dogfooded, not reimplemented).
  6. spec_shape-form distance (`docs/SPEC_fourier_training.md` §5): both sides normalised
     by their OWN B_train sum, sum of squared log-ratio differences over the band bins;
     converted to a "%/bin" figure via `sqrt(distance / n_bins) * 100`, the SAME conversion
     D3 used to turn its 0.389 mean distance into "~31% per-bin power variation"
     (`docs/DIAGNOSTICS_fft.md` D3; reproduced from `experiments/diag_fft/d3/results.json`:
     sqrt(0.38854 / 4) * 100 = 31.17, matching "≈ 31%").

A SUPPLEMENTARY CONTROL, AND WHY IT TURNED OUT DEGENERATE. An independent ETDRK4 rollout at
each n (step 2) is not just a coarser VIEW of one field — it is a separate nonlinear
integration that can settle into a different realised mode/phase from the same seed, so a
first concern was that the rollout-based distance might be measuring pattern-selection
variance rather than grid fidelity. `decimate_field` was added to check this: it
ideal-band-limit-downsamples the SAME converged 512^2 field to each coarser n (no re-solve)
and measures the same distance. It reads EXACTLY 0.00% at every n and seed — and this is
DEGENERATE, not a cleaner measurement: an ideal Fourier truncation is lossless for any
content strictly inside the kept band, and B_train's highest bin sits far below every tested
grid's Nyquist — so the control CANNOT show a nonzero distance by construction, whatever the
true grid-fidelity gap is. What it DOES confirm is that the ROLLOUT-based distances are a
genuine effect of solving at n=96 (etc.), not an artifact of this script's resampling
arithmetic. The ROLLOUT measurement is therefore the PRIMARY reading — matching D2's own
precedent of independent per-grid rollouts — and the decimated 0.00% row is reported as that
confirmatory control, not as an alternative n-verdict.

Usage (CLAUDE.md §7c: no discretised Laplacian; this is CPU, no GPU, no guard needed — it is
not a trainer; pin OMP_NUM_THREADS=1 per fixture and run fixtures in separate PROCESSES for
parallelism, not threads):
    KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python -u scripts/r3_raps_fidelity.py \\
        --tag tune_comp_seed3 --seeds 0,1,2,3,4,5,6,7
    ... (repeat per fixture, in parallel processes) ...
    .venv/bin/python -u scripts/r3_raps_fidelity.py --aggregate

Writes experiments/redesign_r3/raps_fidelity/results/raps_fidelity_<tag>.json per fixture,
and (with --aggregate) the combined raps_fidelity.json. Deterministic given the tracked
fixture checkpoints and the given seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rngrn import observables as obs                              # noqa: E402
from rngrn import solve_box as sb                                 # noqa: E402
from rngrn.eval.rollout import simulate                            # noqa: E402
from diag_fft_d2 import load_fixture, linear_kstar                 # noqa: E402

GRIDS = (96, 128, 256, 512)          # 512 is the reference; ascending for an honest RSS read
CHECK_EVERY = 200
P = sb.P_DEFAULT                     # 8 — D-FFT-9/D1 commensurate periods
B_LO, B_HI = 0.60, 1.55              # D-FFT-9 closure 1

OUT_DIR = os.path.join("experiments", "redesign_r3", "raps_fidelity", "results")

#: Every Turing-unstable TRACKED checkpoint in this repo (see module docstring for how this
#: was enumerated). tag -> checkpoint path.
FIXTURES = {
    "tune_comp_seed3": "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
                       "checkpoints/model.pt",
    "d5_seed3": "experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed3/"
               "checkpoints/model.pt",
    "d5_seed5": "experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed5/"
               "checkpoints/model.pt",
}

#: D3's ~31%/bin RAPS estimation floor (docs/DIAGNOSTICS_fft.md D3), reproduced from the
#: tracked experiments/diag_fft/d3/results.json (not re-derived here — quoted, as
#: measure_interp_error.py also quotes it for the same purpose).
D3_FLOOR_PCT = 31.0

#: Task 11's target-interpolation error (docs/REDESIGN_rngrn.md §4.3's paired measurement),
#: folded into this report per Step 4.
TASK11_INTERP_JSON = os.path.join(
    "experiments", "redesign_r3", "solve_box_interp", "results", "interp_error.json")


def _j(x):
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# --------------------------------------------------------------------------- measurement

def raps_band_distance(field_model: np.ndarray, ref_log_power: np.ndarray,
                       ref_k_centers: np.ndarray, k_hat: float, L: float) -> dict:
    """spec_shape-form log-RAPS band distance between `field_model`'s OWN RAPS (its grid's
    n implied by field_model.shape) and the 512^2 reference RAPS, interpolated onto the
    model's solve-box band-bin positions via `solve_box.interpolate_targets` — SPEC §5,
    exactly as a real training step would build the target at this operating point.

    `k_hat` plays BOTH roles a real training step keeps formally separate: the box's own
    commensurability driver (`L = p*2*pi/k_hat` — box geometry MUST use this value, not a
    per-seed measured k*, or `geometry(...)["L_solve"]` stops equalling the field's actual
    physical L and the box's bin array silently misaligns index-for-index with
    `observables.raps(field_model, L)`'s own array — a real bug this script had, caught by
    the sanity check that n=512 compared against itself must give ~0 distance), AND the
    `kstar_obs` `interpolate_targets` normalises against. §4.3's actual training design
    keeps these SEPARATE (k_hat drifts from the true k*_obs between re-tiles, bounded by
    the re-tile hysteresis |Delta k_hat|/k_hat > 1/(2p)) — but there is no independent
    "observed" domain in this self-consistency test (the fixture provides only itself), so
    k_hat plays k*_obs exactly, the idealised post-ignition operating point the hysteresis
    is designed to hold the real run near. Passing the SEED's own measured k* (which drifts
    a few % from k_hat here via nonlinear finite-size mode selection, reported separately)
    as `kstar_obs` instead would make even n=512-vs-itself read that drift as spurious
    "distance" — the second form this script's bug took before both roles were pinned to
    the same k_hat.
    """
    box = sb.geometry(k_hat, p=P, n=field_model.shape[-1])
    band_idx = sb.band_bins(box, B_LO, B_HI)

    k_mod, p_mod, _ = obs.raps(field_model, L)
    model_band_power = p_mod[band_idx]
    if (model_band_power <= 0).any():
        raise RuntimeError(
            f"non-positive model RAPS power inside the band at n={field_model.shape[-1]} — "
            "log-shape distance undefined")

    target_log = sb.interpolate_targets(ref_log_power, ref_k_centers, k_hat,
                                        box, B_LO, B_HI).numpy()
    target_power = np.exp(target_log)

    s_model = model_band_power / model_band_power.sum()
    s_target = target_power / target_power.sum()
    distance = float(np.sum((np.log(s_model) - np.log(s_target)) ** 2))
    n_bins = int(len(band_idx))
    per_bin_pct = float(np.sqrt(distance / n_bins) * 100.0)
    return dict(n_band_bins=n_bins, distance=distance, per_bin_pct=per_bin_pct,
                L_solve=box["L_solve"])


def decimate_field(field: np.ndarray, n_target: int) -> np.ndarray:
    """Ideal band-limited downsample of a periodic `field` to `n_target x n_target`, on the
    SAME physical L (Fourier-truncate to the central `n_target x n_target` frequency block,
    inverse transform). This is the textbook alias-free decimation operator for a periodic
    signal — the low-frequency content a coarser FFT-basis grid resolves of the SAME field,
    with no independent PDE re-solve. Used as a supplementary, degenerate-but-confirmatory
    control — see the module docstring's "SUPPLEMENTARY CONTROL" paragraph.
    """
    n_src = field.shape[-1]
    if n_target > n_src:
        raise ValueError(f"cannot decimate to a FINER grid ({n_target} > {n_src})")
    F = np.fft.fftshift(np.fft.fft2(field))
    c = n_src // 2
    h = n_target // 2
    Ft = F[c - h:c + h, c - h:c + h]
    down = np.fft.ifft2(np.fft.ifftshift(Ft)) * (n_target * n_target) / (n_src * n_src)
    return np.real(down)


def rollout_and_raps(model, L: float, n: int, seed: int) -> dict:
    rss_before = _rss_kib()
    t0 = time.perf_counter()
    r = simulate(model, L, n=n, seed=seed, integrator="etdrk4_rfft",
                early_stop=True, check_every=CHECK_EVERY)
    seconds = time.perf_counter() - t0
    rss_after = _rss_kib()
    if r["blew_up"]:
        raise RuntimeError(f"rollout blew up at n={n}, seed={seed} — no field to measure")
    field0 = np.asarray(r["fields"])[0]
    k_c, power, kstar = obs.raps(field0, L)
    nonzero = power > 0.0
    return dict(n=n, seed=seed, field0=field0, k_centers=k_c[nonzero],
                log_power=np.log(power[nonzero]), kstar=float(kstar),
                stopped_reason=r["stopped_reason"], patterned=bool(r["patterned"]),
                seconds=seconds, nsteps_run=int(r["nsteps_run"]),
                rss_before_kib=rss_before, rss_after_kib=rss_after,
                sec_per_step=seconds / max(1, int(r["nsteps_run"])))


# --------------------------------------------------------------------------- per-fixture

def run_fixture(ckpt_path: str, tag: str, seeds: tuple[int, ...]) -> dict:
    model = load_fixture(ckpt_path)
    k_lin, sig_max_lin, xstar = linear_kstar(model)
    L = P * 2.0 * np.pi / k_lin
    print(f"[{tag}] fixture: {ckpt_path}")
    print(f"[{tag}]   k*_lin={k_lin:.6f}, sig_max={sig_max_lin:.6f}, L={L:.4f} "
          f"({P} periods/box), seeds={list(seeds)}")

    rows = []
    ref_by_seed: dict[int, dict] = {}
    for seed in seeds:
        r512 = rollout_and_raps(model, L, 512, seed)
        ref_by_seed[seed] = r512
        rows.append(r512)
        print(f"[{tag}]   seed {seed}: n=512 kstar={r512['kstar']:.6f} "
              f"stopped={r512['stopped_reason']} patterned={r512['patterned']} "
              f"{r512['seconds']:.2f}s / {r512['nsteps_run']} steps")

    results_by_grid: dict[str, dict] = {}
    for n in GRIDS:
        grid_rows = []
        for seed in seeds:
            if n == 512:
                r = ref_by_seed[seed]
            else:
                r = rollout_and_raps(model, L, n, seed)
                rows.append(r)
            ref = ref_by_seed[seed]
            dist = raps_band_distance(r["field0"], ref["log_power"], ref["k_centers"],
                                      k_lin, L)
            if n == 512:
                # Sanity check (caught real bugs during development, kept as a live
                # assertion): comparing the n=512 reference against itself must give ~0
                # distance — a nonzero value here means the box/array alignment is wrong
                # again, not a genuine grid-fidelity finding. Cheap; run every time.
                assert dist["per_bin_pct"] < 1.0, (
                    f"[{tag}] n=512 vs itself gave {dist['per_bin_pct']:.2f}%/bin, not ~0 "
                    "— box geometry is misaligned with the model-side RAPS array")
            dk_bin = 2.0 * np.pi / L
            grid_rows.append(dict(
                seed=seed, kstar=_j(r["kstar"]), kstar_512=_j(ref["kstar"]),
                dk_kstar=_j(abs(r["kstar"] - ref["kstar"])),
                within_one_bin=bool(abs(r["kstar"] - ref["kstar"]) <= dk_bin),
                stopped_reason=r["stopped_reason"], patterned=r["patterned"],
                seconds=_j(r["seconds"]), nsteps_run=r["nsteps_run"],
                sec_per_step=_j(r["sec_per_step"]),
                n_band_bins=dist["n_band_bins"], distance=_j(dist["distance"]),
                per_bin_pct=_j(dist["per_bin_pct"]),
            ))
            print(f"[{tag}]   n={n:4d} seed={seed}: per_bin_pct={dist['per_bin_pct']:6.2f}%  "
                  f"(distance={dist['distance']:.4f}, {dist['n_band_bins']} bins)  "
                  f"{r['seconds']:.3f}s/{r['nsteps_run']}steps")
        pcts = [row["per_bin_pct"] for row in grid_rows if row["per_bin_pct"] is not None]
        secs = [row["seconds"] for row in grid_rows]
        steps = [row["nsteps_run"] for row in grid_rows]
        clears_floor = bool(max(pcts) < D3_FLOOR_PCT) if pcts else False
        results_by_grid[str(n)] = dict(
            n=n, rows=grid_rows,
            per_bin_pct_mean=float(np.mean(pcts)) if pcts else None,
            per_bin_pct_max=float(np.max(pcts)) if pcts else None,
            per_bin_pct_min=float(np.min(pcts)) if pcts else None,
            clears_d3_floor_all_seeds=clears_floor,
            seconds_mean=float(np.mean(secs)), seconds_min=float(np.min(secs)),
            seconds_max=float(np.max(secs)),
            steps_mean=float(np.mean(steps)),
        )

    # ---- supplementary, degenerate-but-confirmatory control: decimate the SAME converged
    # 512^2 field to each coarser n (no independent re-solve) — see module docstring.
    decimated_by_grid: dict[str, dict] = {}
    for n in GRIDS:
        if n == 512:
            continue
        grid_rows = []
        for seed in seeds:
            ref = ref_by_seed[seed]
            field_dec = decimate_field(ref["field0"], n)
            dist = raps_band_distance(field_dec, ref["log_power"], ref["k_centers"],
                                      k_lin, L)
            grid_rows.append(dict(seed=seed, n_band_bins=dist["n_band_bins"],
                                  distance=_j(dist["distance"]),
                                  per_bin_pct=_j(dist["per_bin_pct"])))
        pcts = [row["per_bin_pct"] for row in grid_rows]
        decimated_by_grid[str(n)] = dict(
            n=n, rows=grid_rows,
            per_bin_pct_mean=float(np.mean(pcts)), per_bin_pct_max=float(np.max(pcts)),
            per_bin_pct_min=float(np.min(pcts)),
            clears_d3_floor_all_seeds=bool(max(pcts) < D3_FLOOR_PCT))
        dg = decimated_by_grid[str(n)]
        print(f"[{tag}]   [decimated] n={n:4d}: mean={dg['per_bin_pct_mean']:.2e}% "
              f"max={dg['per_bin_pct_max']:.2e}%")
    decimated_by_grid["512"] = dict(n=512, rows=[], per_bin_pct_mean=0.0,
                                    per_bin_pct_max=0.0, per_bin_pct_min=0.0,
                                    clears_d3_floor_all_seeds=True)

    peak_rss_kib = max(_rss_kib(), *(r["rss_after_kib"] for r in rows))

    n_verdict = None
    for n in GRIDS:
        if results_by_grid[str(n)]["clears_d3_floor_all_seeds"]:
            n_verdict = n
            break

    summary = dict(
        tag=tag, fixture=ckpt_path, k_lin=_j(k_lin), sig_max_lin=_j(sig_max_lin), L=_j(L),
        p=P, band=[B_LO, B_HI], seeds=list(seeds), grids=list(GRIDS),
        d3_estimation_floor_pct=D3_FLOOR_PCT,
        per_grid={n: dict(per_bin_pct_mean=results_by_grid[str(n)]["per_bin_pct_mean"],
                          per_bin_pct_max=results_by_grid[str(n)]["per_bin_pct_max"],
                          clears_d3_floor_all_seeds=
                              results_by_grid[str(n)]["clears_d3_floor_all_seeds"],
                          seconds_mean=results_by_grid[str(n)]["seconds_mean"],
                          decimated_per_bin_pct_mean=
                              decimated_by_grid[str(n)]["per_bin_pct_mean"],
                          decimated_per_bin_pct_max=
                              decimated_by_grid[str(n)]["per_bin_pct_max"])
                  for n in GRIDS},
        coarsest_grid_clearing_floor=n_verdict,
        peak_rss_kib=peak_rss_kib,
    )

    out_path = os.path.join(OUT_DIR, f"raps_fidelity_{tag}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, by_grid=results_by_grid,
                       decimated_by_grid=decimated_by_grid), fh, indent=2)

    print(f"[{tag}] D3 floor: ~{D3_FLOOR_PCT:.0f}%/bin")
    for n in GRIDS:
        g = results_by_grid[str(n)]
        verdict = "CLEARS" if g["clears_d3_floor_all_seeds"] else "FAILS"
        print(f"[{tag}]   n={n:4d}: mean={g['per_bin_pct_mean']:6.2f}% "
              f"max={g['per_bin_pct_max']:6.2f}% -> {verdict}  "
              f"(mean {g['seconds_mean']:.2f}s/rollout, {len(seeds)} seeds)")
    print(f"[{tag}] coarsest grid clearing the floor: {n_verdict}")
    print(f"[{tag}] peak RSS: {peak_rss_kib / 1024:.1f} MiB")
    print(f"[{tag}] wrote {out_path}")
    return summary


# --------------------------------------------------------------------------- aggregation

def aggregate() -> int:
    """Merge every experiments/.../raps_fidelity_<tag>.json present into the combined
    raps_fidelity.json, add Task 11's interpolation number (§4.3 pairing), and report
    whether the n-trend separates from seed noise across fixtures."""
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "raps_fidelity_*.json")))
    if not paths:
        raise RuntimeError(f"no per-fixture results found under {OUT_DIR} — run with --tag "
                           "for each fixture in FIXTURES first")
    by_fixture = {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        by_fixture[d["summary"]["tag"]] = d

    task11 = None
    if os.path.exists(TASK11_INTERP_JSON):
        with open(TASK11_INTERP_JSON) as fh:
            task11 = json.load(fh)["summary"]

    # ---- does the n-trend separate from seed noise? For each fixture, is mean(96) >
    # mean(128) > mean(256) (monotonic improvement)? And does the 95%-ish spread (max-min
    # across seeds) at a fixed n dwarf the mean DIFFERENCE between grids?
    trend_rows = []
    for tag, d in by_fixture.items():
        pg = d["summary"]["per_grid"]
        means = {n: pg[n]["per_bin_pct_mean"] for n in ("96", "128", "256")}
        spreads = {n: pg[n]["per_bin_pct_max"] - by_fixture[tag]["by_grid"][n]
                  ["per_bin_pct_min"] for n in ("96", "128", "256")}
        monotonic = means["96"] > means["128"] > means["256"]
        gaps = [means["96"] - means["128"], means["128"] - means["256"]]
        noise_dominates = any(abs(g) < spreads[n] for g, n in
                              zip(gaps, ("96", "128")))
        trend_rows.append(dict(tag=tag, k_lin=d["summary"]["k_lin"], means=means,
                               spreads=spreads, monotonic_96_128_256=monotonic,
                               grid_gaps=gaps, seed_noise_dominates_grid_gap=noise_dominates,
                               coarsest_grid_clearing_floor=
                                   d["summary"]["coarsest_grid_clearing_floor"],
                               n_seeds=len(d["summary"]["seeds"])))

    combined = dict(
        d3_estimation_floor_pct=D3_FLOOR_PCT,
        fixtures=list(by_fixture.keys()),
        by_fixture={tag: d["summary"] for tag, d in by_fixture.items()},
        by_fixture_full={tag: d for tag, d in by_fixture.items()},
        cross_fixture_trend=trend_rows,
        task11_interp_error_summary=task11,
        note=("Multi-fixture sweep (controller-requested, held for more evidence after the "
              "1-fixture/3-seed run could not separate a genuine n-trend from seed noise). "
              "See cross_fixture_trend per fixture for whether the 96/128/256 means are "
              "monotonically decreasing and whether the per-grid seed spread dwarfs the "
              "grid-to-grid mean gap."),
    )
    out_path = os.path.join(OUT_DIR, "raps_fidelity.json")
    with open(out_path, "w") as fh:
        json.dump(combined, fh, indent=2)

    print(f"D3 floor: ~{D3_FLOOR_PCT:.0f}%/bin")
    for row in trend_rows:
        print(f"[{row['tag']}] k_lin={row['k_lin']:.4f} n_seeds={row['n_seeds']}  "
              f"means: 96={row['means']['96']:.2f}% 128={row['means']['128']:.2f}% "
              f"256={row['means']['256']:.2f}%  monotonic={row['monotonic_96_128_256']}  "
              f"coarsest_clearing={row['coarsest_grid_clearing_floor']}  "
              f"seed_noise_dominates={row['seed_noise_dominates_grid_gap']}")
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=sorted(FIXTURES), default=None,
                    help="fixture tag to run (see FIXTURES); omit with --aggregate")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seed list, e.g. 0,1,2,3,4,5,6,7")
    ap.add_argument("--aggregate", action="store_true",
                    help="merge all raps_fidelity_<tag>.json into raps_fidelity.json")
    args = ap.parse_args()

    if args.aggregate:
        return aggregate()
    if args.tag is None:
        raise SystemExit("pass --tag <fixture> or --aggregate")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    run_fixture(FIXTURES[args.tag], args.tag, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
