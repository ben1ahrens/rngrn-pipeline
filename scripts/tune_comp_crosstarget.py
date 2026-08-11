"""tune_comp_crosstarget.py — the CONTROL half of pre-registration 3.1.

3.1 requires `topology_consistency` >= 0.75 AND a within-target minus CROSS-target gap
>= 0.25. `optim/target_report.py` computes only the within-target half; nothing in the
library computes the control. Without it a cell that returns the SAME network regardless
of input scores a perfect 1.0 and passes, which is exactly the false success 3.1 was
placed first to catch.

This reads `recovered.J` straight off the stored run records under a runs-root, groups
them by (cell, sample), and reports:

  within  = mean over samples of pairwise_sign_agreement(J of the K seeds on THAT sample)
  cross   = pairwise_sign_agreement over one J per sample, drawn from DIFFERENT samples
  gap     = within - cross

`mean_agreement` (mean pairwise) is used for BOTH halves rather than modal_fraction,
because a gap is only meaningful between two identically-defined statistics and
modal_fraction is not defined for a set of size 1 per group. The modal-fraction
`topology_consistency` is what the 0.75 bar is read against and comes from the
target-report; the 0.25 gap is read against this file. Both are reported at
sign_zero_rtol 0.02 / 0.05 / 0.10, per 3.1's "known uncalibrated input".

Only Turing-reaching runs are pooled: the sign structure of a network that does not
pattern is not the recovered topology of a Turing system.
"""
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rngrn.scoring.reproducibility import pairwise_sign_agreement  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTOLS = (0.02, 0.05, 0.10)


def _cell_key(fc):
    """The tuning axes, read off the run's OWN frozen config, never off the command line.

    train_results.json carries no config block, so cell identity comes from
    config/frozen_config.yaml, which is what the run actually executed.
    """
    loss, model, train = fc.get("loss", {}), fc.get("model", {}), fc.get("train", {})
    w = loss.get("weights", {}) or {}
    return (f"steps={train.get('adam_steps')}",
            f"detach={loss.get('detach_xstar')}",
            f"dinit={model.get('d_init_from_kstar')}",
            f"prior={w.get('param_prior')}",
            f"anchor={w.get('anchor')}",
            f"init={model.get('init')}",
            f"offfrac={loss.get('staging_off_frac')}",
            f"rampfrac={loss.get('staging_ramp_frac')}")


def load_runs(runs_root):
    """cell/sample/seed -> J, for Turing-reaching runs under one runs-root."""
    import yaml
    out = []
    base = os.path.join(ROOT, runs_root, "runs")
    for rid in sorted(os.listdir(base)):
        rp = os.path.join(base, rid, "results", "train_results.json")
        fp = os.path.join(base, rid, "config", "frozen_config.yaml")
        if not (os.path.exists(rp) and os.path.exists(fp)):
            continue
        try:
            d = json.load(open(rp))
            fc = yaml.safe_load(open(fp))
        except Exception:
            continue
        J = (d.get("recovered", {}) or {}).get("J")
        if J is None or not (d.get("metric", {}) or {}).get("recovered_turing"):
            continue
        out.append({
            "run_id": rid,
            "cell": _cell_key(fc),
            "sample": (fc.get("data", {}) or {}).get("sample_key"),
            "seed": (fc.get("train", {}) or {}).get("seed"),
            "J": np.asarray(J, dtype=float),
        })
    return out


def agreement(J_list, rtol):
    if len(J_list) < 2:
        return float("nan")
    return pairwise_sign_agreement(J_list, sign_zero_rtol=rtol)["mean_agreement"]


def report(runs, label):
    by_sample = defaultdict(list)
    for r in runs:
        by_sample[r["sample"]].append(r["J"])
    samples = sorted(k for k in by_sample if k)
    print(f"\n### {label}   samples={samples}  "
          f"n_turing={{{', '.join(f'{s}:{len(by_sample[s])}' for s in samples)}}}")
    if len(samples) < 2:
        print("  cross-target control NOT COMPUTABLE: needs >= 2 samples with Turing runs")
    print(f"{'rtol':>6}{'within':>10}{'cross':>10}{'gap':>10}   (bar: gap >= 0.25)")
    for rtol in RTOLS:
        wi = [agreement(by_sample[s], rtol) for s in samples]
        wi = [w for w in wi if not np.isnan(w)]
        within = float(np.mean(wi)) if wi else float("nan")
        # cross: every pair of J drawn from DIFFERENT samples, averaged.
        pairs = [agreement([a, b], rtol)
                 for s1, s2 in combinations(samples, 2)
                 for a in by_sample[s1] for b in by_sample[s2]]
        cross = float(np.mean(pairs)) if pairs else float("nan")
        print(f"{rtol:>6.2f}{within:>10.3f}{cross:>10.3f}{within - cross:>10.3f}")


if __name__ == "__main__":
    roots = sys.argv[1:] or ["experiments/tune_comp"]
    for rr in roots:
        runs = load_runs(rr)
        by_cell = defaultdict(list)
        for r in runs:
            by_cell[r["cell"]].append(r)
        for cell in sorted(by_cell):
            report(by_cell[cell], f"{rr}  {','.join(cell)}")
