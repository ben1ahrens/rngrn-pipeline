"""c1_analyse.py — report-driven analysis of the C1 tuning cells.

Every cell in this unit writes its run records into ONE shared `--runs-root`, so a script
that globs `runs/*` would pool cells that differ by an override. This one is driven by the
target-report JSON instead: each report names, in `run_ids`, exactly the runs that produced
it, so a cell's statistics can never absorb another cell's runs.

Three blocks, in PREREGISTRATION section 3 priority order:

  TABLE     the reported criteria per (cell, sample), with the K actually present.
  RESTARTS  pooled per-restart `sig_max_pos` (the C2 sensitive readout, scripts/c1_restarts.py):
            `target-report` scores only the ONE restart the loss-argmin returned, so when the
            Turing count floors near 0 it cannot rank two configs. Every run stores all
            `n_restarts` restarts, so pooling gives 64 x K samples instead of K.
            `n_pos > 0` with 0 scored Turing means the SELECTION rule is discarding Turing
            restarts; `n_pos == 0` means the regime is never reached. Different defects.
  REPRO     criterion 3.1 at sign-zero rtol 0.02 / 0.05 / 0.10 (the tolerance is flagged
            UNCALIBRATED in section 3.1) with the SIZE-MATCHED cross-target control that
            section 3.1 requires: one seed drawn from each of K distinct (cell, target)
            groups, modal fraction of that mixed group, averaged over draws.

Usage:  python scripts/c1_analyse.py [cell ...]      (default: every cell present)
"""
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

WT = Path(__file__).resolve().parent.parent
REPORTS = WT / "experiments" / "tune_comp_reports"
ROOTS = [WT / "experiments" / "tune_comp", WT / "experiments" / "tune_comp_legacy"]

RTOLS = (0.02, 0.05, 0.10)
N_CROSS_DRAWS = 4000

COLS = [
    ("turing_frac", "turing"),
    ("recovered_frac", "recov"),
    ("topology_consistency", "topo_cons"),
    ("mean_agreement", "mean_agr"),
    ("turing_volume_10pct_median", "tv10"),
    ("turing_volume_4p8pct_median", "tv4.8"),
    ("kstar_fft_rel_err_median", "kfft_err"),
    ("trivial_kstar_err_mean", "trivial"),
    ("morphology_match_frac", "morph"),
    ("plausibility_score_mean", "plaus"),
]


def run_dir(run_id):
    for r in ROOTS:
        d = r / "runs" / run_id
        if d.is_dir():
            return d
    return None


def load_cells(cells=None):
    """-> {(cell, sample): report dict}, only for reports that are non-empty."""
    out = {}
    if not REPORTS.is_dir():
        return out
    for cd in sorted(REPORTS.iterdir()):
        if not cd.is_dir():
            continue
        if cells and cd.name not in cells:
            continue
        for p in sorted(cd.glob("*.json")):
            if p.stat().st_size == 0:
                print(f"  !! ZERO-BYTE report ignored: {p}", file=sys.stderr)
                continue
            try:
                out[(cd.name, p.stem)] = json.load(open(p))
            except Exception as e:
                print(f"  !! unreadable {p}: {e}", file=sys.stderr)
    return out


def results_of(rep):
    """-> [(seed, train_results dict)] for the runs this report names."""
    got = []
    for seed, rid in sorted(json.loads(rep["run_ids"]).items(), key=lambda kv: int(kv[0])):
        d = run_dir(rid)
        if d is None:
            continue
        f = d / "results" / "train_results.json"
        if not f.exists() or f.stat().st_size == 0:
            print(f"  !! EMPTY results for {rid}", file=sys.stderr)
            continue
        got.append((int(seed), json.loads(f.read_text())))
    return got


def fmt(v):
    if v is None:
        return "     -"
    try:
        f = float(v)
    except Exception:
        return f"{str(v)[:6]:>6}"
    return "   nan" if math.isnan(f) else f"{f:6.3f}"


def block_table(reps):
    hdr = f"{'cell':<14}{'sample':<12}{'K':>3}" + "".join(f"{s:>10}" for _, s in COLS)
    print("\n=== TABLE — reported criteria, in PREREGISTRATION section 3 priority order ===")
    print(hdr)
    print("-" * len(hdr))
    for (cell, samp), r in sorted(reps.items()):
        k = len(json.loads(r["run_ids"]))
        row = f"{cell:<14}{samp:<12}{k:>3}" + "".join(f"{fmt(r.get(c)):>10}" for c, _ in COLS)
        print(row)
        ps = json.loads(r.get("per_seed", "[]"))
        tur = [p["seed"] for p in ps if p.get("recovered_turing")]
        bad = [p["seed"] for p in ps if p.get("status") != "ok"]
        print(f"{'':<29}turing_seeds={tur}  raised={bad}  n_seeds_requested={r.get('n_seeds_requested')}")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def block_restarts(reps):
    hdr = (f"{'cell':<14}{'sample':<12}{'K':>3}{'nRS':>6}{'n_pos':>7}{'rate':>8}"
           f"{'sig_p90':>9}{'sig_best':>10}{'win_sig_med':>12}{'Dlo/Dmid':>10}")
    print("\n=== RESTARTS — pooled per-restart sig_max_pos (sensitive readout) ===")
    print(hdr)
    print("-" * len(hdr))
    for (cell, samp), r in sorted(reps.items()):
        rows = results_of(r)
        sig = [x["sig_max_pos"] for _, d in rows for x in d.get("restarts", [])
               if x.get("sig_max_pos") is not None and x["sig_max_pos"] == x["sig_max_pos"]]
        npos = sum(1 for s in sig if s > 0)
        wins, dlm = [], []
        for _, d in rows:
            m = d.get("metric", {})
            if m.get("recovered_sig_max") is not None:
                wins.append(m["recovered_sig_max"])
            D = sorted(d.get("recovered", {}).get("D", []) or [])
            if len(D) == 3 and D[1] > 0:
                dlm.append(D[0] / D[1])
        med = lambda xs: pct(xs, 0.5) if xs else float("nan")
        print(f"{cell:<14}{samp:<12}{len(rows):>3}{len(sig):>6}{npos:>7}"
              f"{npos / max(1, len(sig)):>8.4f}{pct(sig, 0.90):>9.4f}"
              f"{(max(sig) if sig else float('nan')):>10.4f}{med(wins):>12.4f}{med(dlm):>10.4f}")


def sign_key(J, rtol):
    import numpy as np
    J = np.asarray(J, dtype=float)
    m = float(abs(J).max())
    if m <= 0:
        return tuple([0] * J.size)
    s = np.sign(J)
    s[abs(J) < rtol * m] = 0
    return tuple(int(v) for v in s.ravel())


def block_perm(reps):
    """DIAGNOSTIC ONLY. The modal fraction after quotienting by the node-label permutation.

    `topology_consistency` compares sign structures ENTRYWISE, so two seeds that recovered
    the SAME network with its three nodes in a different order score as two distinct
    structures. Nothing in the objective pins the node ordering -- the loss is a function of
    sigma(k), which is permutation-invariant -- so this symmetry is real and unbroken, and
    with N=3 there are 6 relabelings.

    THIS IS NOT A SUBSTITUTE FOR CRITERION 3.1 AND MUST NOT BE READ AS ONE. The
    pre-registered statistic is the raw modal fraction at rtol 0.05 and it stays the number
    reported against the 0.75 bar. This column exists to say HOW MUCH of a 3.1 failure is
    the relabeling symmetry and how much is a genuinely different network.
    """
    import itertools

    import numpy as np
    print("\n=== PERM — DIAGNOSTIC: modal fraction after quotienting by node relabeling ===")
    print("    (NOT criterion 3.1. 3.1 is the raw column in the REPRO block above.)")
    hdr = f"{'cell':<14}{'sample':<12}{'K':>3}" + "".join(
        f"{'raw@' + str(t):>10}{'canon@' + str(t):>12}" for t in RTOLS)
    print(hdr)
    print("-" * len(hdr))
    for (cell, samp), r in sorted(reps.items()):
        Js = [d.get("recovered", {}).get("J") for _, d in results_of(r)]
        Js = [np.asarray(J, dtype=float) for J in Js if J is not None]
        if not Js:
            continue
        row = f"{cell:<14}{samp:<12}{len(Js):>3}"
        for t in RTOLS:
            raw = Counter(sign_key(J, t) for J in Js)
            canon = Counter(
                min(sign_key(J[np.ix_(list(p), list(p))], t)
                    for p in itertools.permutations(range(J.shape[0])))
                for J in Js)
            row += (f"{raw.most_common(1)[0][1] / len(Js):>10.3f}"
                    f"{canon.most_common(1)[0][1] / len(Js):>12.3f}")
        print(row)


def target_of(rep, key):
    """The TARGET a group of seeds was fitted to: (dataset_id, sample_key).

    The group key is (cell, sample_key), and `sample_key` alone does NOT identify a target:
    three_gene_val/sample_0000 and three_gene_qvar/sample_0000 are different systems that
    share a stem. Excluding "the same sample_key" from the cross-target control would
    therefore silently drop a genuinely different target from the control, biasing the one
    number PREREGISTRATION section 3.1 calls the substantive half of the criterion. The
    dataset_id is carried in every target-report, so the target is keyed on the pair.
    """
    return (rep.get("dataset_id"), rep.get("sample_key", key[1]))


def block_repro(reps):
    groups = {}   # (cell, sample) -> [J, ...]
    tgt = {}      # (cell, sample) -> (dataset_id, sample_key)
    for key, r in sorted(reps.items()):
        Js = [d.get("recovered", {}).get("J") for _, d in results_of(r)]
        Js = [J for J in Js if J is not None]
        if Js:
            groups[key] = Js
            tgt[key] = target_of(r, key)
    print("\n=== REPRO — criterion 3.1, modal fraction vs the SIZE-MATCHED cross-target control ===")
    hdr = (f"{'cell':<14}{'sample':<12}{'K':>3}" +
           "".join(f"{'within@' + str(t):>13}{'cross@' + str(t):>12}{'gap':>7}" for t in RTOLS))
    print(hdr)
    print("-" * len(hdr))
    rng = random.Random(0)
    for key, Js in sorted(groups.items()):
        cell, samp = key
        row = f"{cell:<14}{samp:<12}{len(Js):>3}"
        for t in RTOLS:
            keys = [sign_key(J, t) for J in Js]
            within = Counter(keys).most_common(1)[0][1] / len(keys)
            # cross-target control: one seed from each of K DISTINCT other groups,
            # size-matched to the within-target group.
            others = [k for k in groups if tgt[k] != tgt[key]]
            if len(others) >= 2:
                vals = []
                for _ in range(N_CROSS_DRAWS):
                    pick = others if len(others) <= len(Js) else rng.sample(others, len(Js))
                    mixed = [sign_key(rng.choice(groups[o]), t) for o in pick]
                    vals.append(Counter(mixed).most_common(1)[0][1] / len(mixed))
                cross = sum(vals) / len(vals)
                row += f"{within:>13.3f}{cross:>12.3f}{within - cross:>7.3f}"
            else:
                row += f"{within:>13.3f}{'  -':>12}{'  -':>7}"
        print(row)
    if len(set(tgt.values())) < 2:
        print("  (cross-target control needs >= 2 distinct targets; not computable yet)")


def main():
    cells = sys.argv[1:] or None
    reps = load_cells(cells)
    if not reps:
        print("no non-empty reports found")
        return
    block_table(reps)
    block_restarts(reps)
    block_repro(reps)
    block_perm(reps)


if __name__ == "__main__":
    main()
