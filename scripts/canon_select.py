#!/usr/bin/env python
"""canon_select.py — choose the systems that become the canonical Turing datasets.

Selection, not generation. Every decision here is deterministic given a seed, so the
canonical set is reproducible from the corpus rather than hand-picked.

WHY MARGIN AND NOT JUST THE LABEL
---------------------------------
The morphology class boundaries are uncalibrated. ``A > 0.55`` (the stripe test) appears
exactly ONCE in the whole repo, with no control and no test behind it, and ``phi < 0.34``
sits on the Gaussian null (P(z>0.4) = 0.3446), so it separates "positively skewed" from
"not" rather than "spots" from "not spots". Worse, the angular features behind ``A`` are
NOT L-invariant: a measured control drifts ``ang_conc`` 0.190 -> 0.074 across an 8x L span,
enough to flip a class label. So a sample sitting near a boundary carries a label that is
partly an artefact of its box size. Admitting only high-margin, label-stable samples is
what makes the three strata mean something.

WHY THE PERIODICITY GATE IS LOAD-BEARING, NOT A NICETY
------------------------------------------------------
Ranking by margin ALONE selects ``three_gene_qvar/sample_0032`` first among spots, because
its area fraction is 0.032 — the largest possible distance below the 0.34 cut. It is also
the one sample in the corpus that is not a Turing pattern at all: three isolated blobs,
species 1 and 2 flat to cv=0.002, and a monotonically decaying spectrum with no interior
peak. The ``peak_bin >= 3`` gate is the only thing between it and the canonical set.

FIREWALL POSITION
-----------------
Data side. This reads ``payload.h5`` — generating kinetics, the generator's ``k_star`` —
which CLAUDE.md section 5 forbids to recovery-side code. It lives in ``scripts/`` and is
imported only by ``canon_generate.py`` and its own test. It must never be imported from
``model.py``, ``observables.py``, ``recover.py``, ``losses/`` or ``eval/``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from gen_tg3 import stable_seed                                    # noqa: E402
from td_figures import PEAK_BIN_MIN                                # noqa: E402

# The generator's own class boundaries (scripts/gen_tg3.py::classify). Reproduced here so
# margins are measured against the SAME cuts that produced the stored labels; this module
# must not invent a second taxonomy.
PHI_SPOTS_MAX = 0.34
PHI_HOLES_MIN = 0.66
ANISO_STRIPES_MIN = 0.55

# Contrast floor. The generator rejects below 0.05 as "collapsed to homogeneous"; 0.30 is
# far above that, chosen so a canonical exemplar is unambiguous rather than merely legal.
CV_MIN = 0.30

# Periods-per-box for the 512 grid. Three properties pull against each other here, and the
# range is the compromise between them:
#
#   * k* PRECISION wants HIGH p. The RAPS bin width is 2*pi/L, so the relative half-bin
#     precision is 1/(2p).
#   * RESOLUTION wants LOW p at a fixed grid: px/wavelength = 512/p, against a measured
#     floor of 6.0 (D15).
#   * L-DECOUPLING wants a WIDE RELATIVE range. If every sample shares one p then
#     k* = p*2*pi/L inverts exactly and the domain size is the label again (D6). The
#     strength of the decoupling scales with the SPREAD of p, not its magnitude.
#
# MEASURED 2026-08-10, oracle best-fixed-p blind predictor over 4000 random 5-sample draws
# (the oracle picks the single p minimising median error AFTER seeing the answers):
#
#   p range   spread   median oracle error   % of draws below 15%   px/wl      k* floor
#    16-32      2.0x           9.1%                   95%          16.0-32.0  1.6-3.1%
#     8-40      5.0x          15.0%                   48%          12.8-64.0  1.2-6.2%
#     3-14      4.7x          20.0%                   27%          36.6-170.7 3.6-16.7%
#
# 16-32 was the first choice and is the WORST of these: 95% of draws leave the leak
# predictor inside 15%. 8-40 gives a 5.0x spread — wider than the legacy qvar range's 4.67x
# — while keeping every sample better than the legacy data on BOTH resolution (12.8 px/wl
# worst case against 16.0, both far above the 6.0 floor) and k* precision (6.2% worst case
# against 8.3%). So it dominates the legacy set outright and materially improves the
# decoupling. See docs/DECISIONS.md D-CANON-3.
#
# NOTE the honest limit: with only 5 samples per dataset NO range decouples strongly, because
# an oracle single p can always sit near the middle of five values. Even the legacy 3-14
# range only reaches 20% at n=5, against the 45.5% qvar achieves at n=34. Corpus-level
# `kstar_rel_err` medians are therefore not meaningful on these datasets at any range.
P_CHOICES = tuple(range(8, 41))

# Everything the generator's classifier can emit. `holes` is omitted deliberately: it is
# structurally unreachable, because species 0 is the self-activator in all six topologies
# and its field is bounded in [b/mu, (b+V)/mu] with the basal floor 1-3 decades below
# saturation, so the activator channel is positively skewed by construction while
# `holes` (phi > 0.66) needs strong negative skew. Measured class-mean skews: spots +1.235,
# labyrinth -0.145, stripes -0.058. The corpus contains zero `holes` samples in 413.
TAXONOMY_CLASSES = ("spots", "stripes", "labyrinth")

# What we actually ship as canonical datasets.
#
# `stripes` is EXCLUDED, on measurement rather than convenience (2026-08-10):
#   * 9/23 multiL systems change morphology label when only the box size changes, and every
#     single flip is labyrinth <-> stripes -- the uncalibrated `A > 0.55` boundary.
#   * All 5 gated qvar stripes candidates flip to labyrinth under the stability probe.
#     Zero stripes survive gates + stability anywhere in the re-simulatable corpus.
#   * corr(periods_per_box, anisotropy) = -0.312, and the stripes fraction falls
#     monotonically 26% -> 20% -> 16% -> 8% -> 0% across p bins. NO sample at p >= 11 is
#     ever labelled stripes; stripes sit at median p = 4.5 against 9.5 for everything else.
#
# The mechanism: a small box admits few orientations, forcing the pattern onto a single
# axis and raising anisotropy. Give it room and it relaxes into labyrinth. So `stripes`
# here is largely a small-box artefact rather than a distinct pattern type, and the
# canonical range p = 16..32 lies entirely above the regime where it has ever occurred.
# Shipping a "stripes" set would ship the artefact. See docs/DECISIONS.md D-CANON-2.
CANONICAL_CLASSES = ("spots", "labyrinth")


def class_margin(morphology, phi, anisotropy):
    """Signed distance from the nearest boundary that defines this class.

    Positive means "inside the class with room to spare"; negative means the sample is
    actually on the wrong side of a cut and its stored label is wrong.
    """
    phi = float(phi)
    a = float(anisotropy)
    if morphology == "spots":
        return PHI_SPOTS_MAX - phi
    if morphology == "holes":
        return phi - PHI_HOLES_MIN
    if morphology == "stripes":
        return min(a - ANISO_STRIPES_MIN, phi - PHI_SPOTS_MAX, PHI_HOLES_MIN - phi)
    if morphology == "labyrinth":
        return min(ANISO_STRIPES_MIN - a, phi - PHI_SPOTS_MAX, PHI_HOLES_MIN - phi)
    raise ValueError(f"unknown morphology {morphology!r}; expected one of "
                     f"spots/holes/stripes/labyrinth")


def passes_gates(row):
    """Admission gates, applied before any ranking."""
    return bool(row["peak_bin"] >= PEAK_BIN_MIN
                and row["cv"] >= CV_MIN
                and class_margin(row["morphology"], row["area_frac"], row["anisotropy"]) > 0)


# The leak bar. `oracle_leak_error` below is the median relative error of the best blind
# predictor `k = q*2pi/L` with q chosen AFTER seeing the answers — a deliberately generous
# adversary. 0.25 is comfortably above the 0.15 first used and is met by construction; the
# legacy three_gene sets score 0.000 and three_gene_qvar scores 0.455 at n=34.
LEAK_MIN_ORACLE_ERR = 0.25


def oracle_leak_error(periods):
    """How badly does the best image-blind predictor of k* do on this set of periods?

    A blind predictor guesses `k = q*2pi/L` for a single fixed integer q. Since the truth is
    `k = p*2pi/L`, its relative error on a sample is exactly ``|q - p| / p`` — L cancels.
    Higher is better: it means the domain size does not tell you the wavenumber.
    """
    ps = np.asarray(periods, float)
    return float(min(np.median(np.abs(q - ps) / ps) for q in range(1, 80)))


def draw_periods(dataset_id, n, seed):
    """``n`` DISTINCT periods-per-box for one dataset, spread GEOMETRICALLY.

    Distinctness alone is not enough. The leak predictor's error is ``|q - p| / p`` — a
    RELATIVE quantity — so what protects against it is the spread of p in log space, not in
    absolute terms. An i.i.d. draw from the allowed range clusters badly at n=5: the first
    version of this function drew {12,15,19,20,24} for ``turing_spots``, which an oracle
    fits to 12.5% median error. A geometric ladder across the range maximises the minimum
    relative gap and reaches ~50% instead.

    The ladder is shifted by a seeded sub-rung offset so the two datasets differ, and the
    result is CHECKED against ``LEAK_MIN_ORACLE_ERR`` rather than assumed — if a shift fails
    the bar it is rejected and the next one tried. Seeded via SHA-256 so the choice is
    process-independent.
    """
    lo, hi = float(min(P_CHOICES)), float(max(P_CHOICES))
    rng = np.random.default_rng(stable_seed(f"{seed}:{dataset_id}"))
    rungs = np.linspace(np.log(lo), np.log(hi), n)
    step = rungs[1] - rungs[0]
    for _ in range(500):
        shift = rng.uniform(-0.5, 0.5) * step
        vals = np.exp(rungs + shift)
        ps = sorted({int(round(float(np.clip(v, lo, hi)))) for v in vals})
        if len(ps) == n and oracle_leak_error(ps) >= LEAK_MIN_ORACLE_ERR:
            return ps
    raise ValueError(
        f"no geometric ladder of {n} periods in [{lo:.0f}, {hi:.0f}] clears the leak bar of "
        f"{LEAK_MIN_ORACLE_ERR:.0%}. Widen P_CHOICES rather than lowering the bar — a narrow "
        f"range is exactly what makes L informative about k*.")


# ======================================================================================
# the candidate table
# ======================================================================================
# Only these two datasets carry params_json AND sim_seed, which is what re-simulation at a
# new resolution requires. The 127 legacy three_gene_{train,val,test} samples carry
# neither — their generator lived in a gitignored tree and their kinetics are gone — and
# the *_classical_* families carry kinetics but no seed and are not 3-gene GRNs.
ELIGIBLE_DATASETS = ("three_gene_qvar", "three_gene_multiL")

CANDIDATE_FIELDS = ("source_dataset", "source_key", "system_id", "morphology", "area_frac",
                    "anisotropy", "cv", "peak_bin", "margin", "k_star", "L", "uid")


def row_uid(row):
    """Stable identity for a SYSTEM (kinetics), independent of which replicate we read."""
    return f"{row['source_dataset']}:{int(row['system_id'])}"


def candidate_table(datasets_root=None):
    """One row per distinct system across the eligible datasets.

    ``three_gene_multiL`` holds 23 systems x 4 domain sizes; we keep the first replicate as
    the representative and use the other three for the free label-stability check.
    """
    import td_figures as TD

    rows = []
    for ds in ELIGIBLE_DATASETS:
        seen = set()
        for s in TD.load_samples(ds, datasets_root):
            if not TD.is_resimulatable(s):
                raise ValueError(f"{ds}/{s['key']} is not re-simulatable but {ds} is listed "
                                 f"as eligible — the eligibility list is wrong")
            sid = int(s["attrs"]["system_id"])
            if sid in seen:
                continue
            seen.add(sid)
            v = TD.patterning_verdict(s["final_frame"][0], s["L"])
            r = {
                "source_dataset": ds,
                "source_key": s["key"],
                "system_id": sid,
                "morphology": str(s["morphology"]),
                "area_frac": float(s["attrs"]["area_frac"]),
                "anisotropy": float(s["attrs"]["anisotropy"]),
                "cv": float(v["cv"]),
                "peak_bin": int(v["peak_bin"]),
                "k_star": float(s["k_star"]),
                "L": float(s["L"]),
            }
            r["margin"] = class_margin(r["morphology"], r["area_frac"], r["anisotropy"])
            r["uid"] = row_uid(r)
            rows.append(r)
    return rows


# ======================================================================================
# label stability across domain size
# ======================================================================================
def native_periods(row):
    """Periods-per-box the sample was generated at: ``p = L*k*/(2*pi)``."""
    return int(round(float(row["L"]) * float(row["k_star"]) / (2 * np.pi)))


def stability_probe_p(row):
    """A second periods-per-box at which to re-simulate, as far from the native one as the
    system allows.

    Probing at the sample's own p would re-run an identical simulation and prove nothing.
    The candidate set is ``gen_tg3.feasible_periods(k*)`` — the periods for which
    ``L = p*(2pi/k*)`` lands inside the generator's [18, 220] acceptance window — because a
    p outside it makes ``simulate_and_classify`` raise. Picking the FURTHEST feasible p
    makes the probe the strongest available test of whether the label is a property of the
    system or of its box.
    """
    from gen_tg3 import feasible_periods

    native = native_periods(row)
    feas = [p for p in feasible_periods(float(row["k_star"])) if p != native]
    if not feas:
        raise ValueError(f"no feasible stability probe for native p={native} "
                         f"(k*={row['k_star']:.4f}); this system admits only one box size")
    return max(feas, key=lambda p: (abs(p - native), p))


def label_is_stable(row, probe_labels):
    """Does the stored class survive every probe?"""
    if not probe_labels:
        raise ValueError("no probe labels supplied; stability cannot be asserted")
    return all(str(x) == str(row["morphology"]) for x in probe_labels)


def multiL_labels(system_id, datasets_root=None):
    """Labels of all four replicates of one multiL system.

    Free stability evidence: multiL already simulated each system at p in {4,7,10,13}, so
    for those 23 systems the probe costs no compute at all.
    """
    import td_figures as TD
    return [str(s["morphology"]) for s in TD.load_samples("three_gene_multiL", datasets_root)
            if int(s["attrs"]["system_id"]) == int(system_id)]


# ======================================================================================
# selection
# ======================================================================================
# Systems that recovery experiments have already been run against (counted over every
# experiments/**/runs.jsonl). They are burned for tuning, so they may seed the TUNING half
# of a class but must never land in the held-out half.
PREVIOUSLY_RUN = {"three_gene_qvar:0", "three_gene_qvar:1", "three_gene_qvar:2",
                  "three_gene_qvar:3", "three_gene_qvar:4", "three_gene_multiL:13"}


def select(table, stability, per_class=5, n_tuning=2, seed=2026):
    """Choose ``per_class`` systems for each canonical class and assign split roles.

    Ranking is by margin, descending, after gates and stability. Previously-run systems are
    preferred for the TUNING slots — they are already burned, so spending them costs
    nothing — and forbidden from the held-out slots, where they would not be held out in
    any meaningful sense.
    """
    out = {"seed": seed, "per_class": per_class, "n_tuning": n_tuning, "datasets": {}}
    for cls in CANONICAL_CLASSES:
        pool = [r for r in table
                if r["morphology"] == cls and passes_gates(r) and stability.get(r["uid"], False)]
        if len(pool) < per_class:
            raise ValueError(
                f"class {cls!r}: only {len(pool)} admissible systems, need {per_class}. "
                f"Screen fresh candidates with scripts/gen_tg3.py rather than lowering a gate.")
        burned = sorted([r for r in pool if r["uid"] in PREVIOUSLY_RUN],
                        key=lambda r: (-r["margin"], r["uid"]))
        fresh = sorted([r for r in pool if r["uid"] not in PREVIOUSLY_RUN],
                       key=lambda r: (-r["margin"], r["uid"]))
        tuning = (burned + fresh)[:n_tuning]
        taken = {r["uid"] for r in tuning}
        held = [r for r in fresh if r["uid"] not in taken][:per_class - n_tuning]
        if len(held) < per_class - n_tuning:
            raise ValueError(f"class {cls!r}: not enough never-run systems for the held-out "
                             f"half ({len(held)} of {per_class - n_tuning})")
        chosen = tuning + held
        ds_id = f"turing_{cls}"
        periods = draw_periods(ds_id, per_class, seed)
        samples = []
        for r, p, role in zip(chosen, periods,
                              ["tuning"] * n_tuning + ["held_out"] * (per_class - n_tuning)):
            samples.append({**{k: r[k] for k in CANDIDATE_FIELDS},
                            "periods_per_box": int(p), "role": role})
        out["datasets"][ds_id] = {"morphology": cls, "samples": samples}
    return out


def compute_stability(table, datasets_root=None, cache_path=None, verbose=True):
    """Stability verdict per system.

    multiL systems are free — they were already simulated at p in {4,7,10,13}, so the probe
    is a lookup. qvar systems need ONE extra 96x96 simulation each (~20-40 s). Results are
    cached so the (slow) sweep is restartable.
    """
    import td_figures as TD

    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    qvar = {s["key"]: s for s in TD.load_samples("three_gene_qvar", datasets_root)}
    stability = {}
    for r in table:
        if r["uid"] in cache:
            stability[r["uid"]] = bool(cache[r["uid"]])
            continue
        if not passes_gates(r):
            stability[r["uid"]] = False          # never probed; cannot be selected anyway
            continue
        # Every gated system is probed, including classes we do not ship. Restricting the
        # probe to CANONICAL_CLASSES would leave the stripes evidence — the reason stripes
        # is not shipped — outside the record it is supposed to justify.
        if r["source_dataset"] == "three_gene_multiL":
            labels = multiL_labels(r["system_id"], datasets_root)
        else:
            import canon_generate as CG
            labels = [CG.probe_label(qvar[r["source_key"]], stability_probe_p(r))]
        ok = label_is_stable(r, labels)
        stability[r["uid"]] = ok
        cache[r["uid"]] = ok
        if verbose:
            print(f"  probe {r['uid']:24s} {r['morphology']:10s} -> "
                  f"{'STABLE' if ok else 'FLIPPED'}")
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2, sort_keys=True)
    return stability


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Freeze the canonical dataset selection.")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "canonical_selection.json"))
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data",
                                                    "canonical_stability_cache.json"))
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--n-tuning", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--datasets-root", default=None)
    a = ap.parse_args(argv)

    table = candidate_table(a.datasets_root)
    print(f"{len(table)} distinct systems in the eligible pool")
    stability = compute_stability(table, a.datasets_root, a.cache)
    sel = select(table, stability, a.per_class, a.n_tuning, a.seed)
    sel["excluded_classes"] = {
        "stripes": "every candidate flips to labyrinth when only the box size changes; "
                   "no sample at p >= 11 is ever labelled stripes (see DECISIONS D-CANON-2)",
        "holes": "structurally unreachable: species 0 is the self-activator in all six "
                 "topologies, so the observed channel is positively skewed by construction",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(sel, f, indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")
    for name, d in sel["datasets"].items():
        print(f"\n{name}:")
        for s in d["samples"]:
            print(f"   {s['uid']:24s} {s['source_key']:13s} p={s['periods_per_box']:3d} "
                  f"px/wl={512/s['periods_per_box']:5.1f} margin={s['margin']:.3f} {s['role']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
