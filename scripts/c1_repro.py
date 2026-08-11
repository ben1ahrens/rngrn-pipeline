"""c2_repro.py — criterion 3.1 with its control and its sensitivity, from stored J.

`target-report` emits `topology_consistency` at ONE sign-zero tolerance (0.05) and does not
emit the cross-target control at all. PREREGISTRATION sec 3.1 requires both:

  * the modal fraction reported at rtol 0.02 / 0.05 / 0.10, because that tolerance is
    flagged UNCALIBRATED and the conclusion's sensitivity to it must be visible;
  * the CROSS-TARGET control -- the same statistic over seeds fitted to DIFFERENT targets.
    "If within-target is approximately cross-target, the model is reproducing itself, not
    the target, and the criterion fails REGARDLESS of the absolute number."

Both are computed here from the `recovered.J` stored in every run record, using the same
sign rule as `scoring/reproducibility.py`: entries with |J_ij| < rtol * max|J| collapse to 0.

The cross-target control is built to have the SAME group size as the within-target
statistic, so the two modal fractions are directly comparable: for each of many random
draws, one seed is taken from each of K distinct targets and the modal fraction of that
mixed group is computed. Comparing a K-of-one-target number against a
(K x n_targets)-of-all-targets number would understate the control by construction.

Usage:  python scripts/c2_repro.py experiments/<root> [...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RTOLS = (0.02, 0.05, 0.10)
N_CROSS_DRAWS = 2000


def sign_key(J, rtol):
    J = np.asarray(J, dtype=float)
    m = np.abs(J).max()
    if m <= 0:
        return tuple(np.zeros(J.size, dtype=int))
    s = np.sign(J)
    s[np.abs(J) < rtol * m] = 0
    return tuple(int(v) for v in s.ravel())


def load(root):
    smap = {}
    p = Path(root) / "runs.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                smap[r["run_id"]] = r["sample_key"]
    by = {}
    for rd in sorted(Path(root).glob("runs/*")):
        f = rd / "results" / "train_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        J = d.get("recovered", {}).get("J")
        if J is None:
            continue
        by.setdefault(smap.get(rd.name, "?"), []).append(J)
    return by


def modal_fraction(keys):
    if not keys:
        return float("nan")
    return Counter(keys).most_common(1)[0][1] / len(keys)


def main(roots):
    for root in roots:
        by = load(root)
        if not by:
            continue
        targets = sorted(by)
        K = max(len(v) for v in by.values())
        print(f"\n=== {Path(root).name}  ({len(targets)} targets, K up to {K}) ===")
        print(f"{'target':<12} " + " ".join(f"{'r=' + str(r):>8}" for r in RTOLS)
              + f" {'ndistinct@0.05':>15}")
        within = {r: [] for r in RTOLS}
        for t in targets:
            cells = []
            for r in RTOLS:
                ks = [sign_key(J, r) for J in by[t]]
                mf = modal_fraction(ks)
                within[r].append(mf)
                cells.append(mf)
            nd = len(set(sign_key(J, 0.05) for J in by[t]))
            print(f"{t:<12} " + " ".join(f"{c:>8.3f}" for c in cells) + f" {nd:>15}")
        print(f"{'MEAN within':<12} "
              + " ".join(f"{np.mean(within[r]):>8.3f}" for r in RTOLS))

        if len(targets) < 2:
            print("cross-target control: needs >= 2 targets in this root")
            continue
        # Group size is matched between the two statistics. Modal fraction rises with a
        # SMALLER group by chance alone, so comparing a K=8 within-target number against a
        # 5-member cross-target group would flatter the gap. Both are recomputed at the
        # common size n_pick = min(K, n_targets).
        rng = np.random.default_rng(0)
        n_pick = min(K, len(targets))
        cross, within_m = {}, {}
        for r in RTOLS:
            keys_by_t = {t: [sign_key(J, r) for J in by[t]] for t in targets}
            cv, wv = [], []
            for _ in range(N_CROSS_DRAWS):
                picked = rng.choice(len(targets), size=n_pick, replace=False)
                cv.append(modal_fraction(
                    [keys_by_t[targets[i]][rng.integers(len(keys_by_t[targets[i]]))]
                     for i in picked]))
                t = targets[rng.integers(len(targets))]
                ks = keys_by_t[t]
                idx = rng.choice(len(ks), size=min(n_pick, len(ks)), replace=False)
                wv.append(modal_fraction([ks[i] for i in idx]))
            cross[r] = float(np.mean(cv))
            within_m[r] = float(np.mean(wv))
        print(f"{'within@n=' + str(n_pick):<12} "
              + " ".join(f"{within_m[r]:>8.3f}" for r in RTOLS)
              + f"   (size-matched to the control)")
        print(f"{'CROSS ctrl':<12} " + " ".join(f"{cross[r]:>8.3f}" for r in RTOLS)
              + f"   (group size {n_pick}, {N_CROSS_DRAWS} draws)")
        print(f"{'GAP':<12} "
              + " ".join(f"{within_m[r] - cross[r]:>8.3f}" for r in RTOLS)
              + "   (3.1 needs within(K=8) >= 0.75 AND gap >= 0.25)")


if __name__ == "__main__":
    main(sys.argv[1:])
