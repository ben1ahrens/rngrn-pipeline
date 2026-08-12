"""diag_fft_d5_analysis.py — D5: zero-call and seed-spread measurement (PLAN §3).

Reads the ≥10-seed baseline `target-report` runs under experiments/diag_fft/d5/runs
(recovery OUTPUTS: checkpoints + train_results.json — no payload.h5, no answer key) and
measures the three D5 quantities:

1. Sign-structure reproducibility of the CURRENT baseline objective (the R1 statistic's
   baseline value): distinct sign structures and cluster sizes, under recover._topology's
   rule (gate>0.5 sign, absent if s <= 5%·max(s) — the UNCALIBRATED inherited rule).
2. Zero-call calibration input (SPEC §9.3): magnitude distributions (s/max(s)) of
   consistently-present vs consistently-absent edges across seeds. Non-separation is a
   finding, not a licence to pick a cut.
3. Parameter spread on TIME-SCALE-INVARIANT combinations (SPEC §9.4): per-quantity
   log10-space max pairwise spread, over all seeds / Turing seeds / largest sign cluster.

Usage: .venv/bin/python scripts/diag_fft_d5_analysis.py [--runs experiments/diag_fft/d5/runs]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.model import RNGRN            # noqa: E402
from rngrn.utils import d_ratio_of       # noqa: E402

ZERO_CALL_FRAC = 0.05      # recover._topology's inherited rule — UNCALIBRATED (SPEC §9.3)
PRESENT_CONSENSUS = 0.8


def load_run(run_dir: str) -> dict:
    ck = torch.load(os.path.join(run_dir, "checkpoints", "model.pt"),
                    map_location="cpu", weights_only=True)
    model = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    model.load_state_dict(ck["state_dict"])
    tr = json.load(open(os.path.join(run_dir, "results", "train_results.json")))
    s = model.s.detach().numpy()
    g = model.gate.detach().numpy()
    sign = np.where(g > 0.5, 1, -1)
    sign = np.where(s > ZERO_CALL_FRAC * s.max(), sign, 0)
    return dict(
        run=os.path.basename(run_dir),
        sign=sign, s_norm=s / s.max(),
        alpha=model.alpha.detach().numpy(), delta=model.delta.detach().numpy(),
        beta=model.beta.detach().numpy(), D=model.D.detach().numpy(),
        KA=model.KA.detach().numpy(), KR=model.KR.detach().numpy(),
        turing=bool(tr["metric"]["recovered_turing"]),
        kstar_model=float(tr["kstar_model"]), kstar_obs=float(tr["kstar_obs"]),
        restarts=tr.get("restarts", []),
    )


def log_spread(values: list[np.ndarray]) -> float:
    """Max pairwise |Δ log10| over runs, max over vector entries. NaN when < 2 runs —
    deliberate: a spread of one sample is undefined, not zero."""
    if len(values) < 2:
        return float("nan")
    logs = np.log10(np.stack([np.abs(v).ravel() + 1e-300 for v in values]))
    return float((logs.max(axis=0) - logs.min(axis=0)).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/diag_fft/d5/runs")
    ap.add_argument("--out", default="experiments/diag_fft/d5/analysis.json")
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.runs, "*")))
    runs = [load_run(d) for d in dirs if os.path.isdir(d)]
    if len(runs) < 10:
        raise SystemExit(f"expected >=10 runs, found {len(runs)} in {a.runs}")
    N = runs[0]["sign"].shape[0]
    report: dict = dict(n_runs=len(runs), N=N, zero_call_frac=ZERO_CALL_FRAC,
                        runs=[r["run"] for r in runs])

    # 1 — sign-structure clusters
    keys = ["".join(str(int(x)) for x in r["sign"].ravel()) for r in runs]
    clusters: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        clusters.setdefault(k, []).append(i)
    cl_sorted = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    report["n_distinct_structures"] = len(clusters)
    report["cluster_sizes"] = [len(v) for _, v in cl_sorted]
    report["clusters"] = {k: v for k, v in cl_sorted}
    report["turing_flags"] = [r["turing"] for r in runs]
    report["kstar_model"] = [r["kstar_model"] for r in runs]
    n_turing = sum(r["turing"] for r in runs)

    # 2 — zero-call input: consistent-edge magnitude distributions
    signs = np.stack([r["sign"] for r in runs])            # (R, N, N)
    mags = np.stack([r["s_norm"] for r in runs])           # (R, N, N)
    present_frac = (signs != 0).mean(axis=0)               # (N, N)
    edge_class = np.full((N, N), "mixed", dtype=object)
    consistent_sign = np.zeros((N, N), int)
    for i in range(N):
        for j in range(N):
            col = signs[:, i, j]
            nz = col[col != 0]
            if present_frac[i, j] >= PRESENT_CONSENSUS and len(set(nz.tolist())) == 1:
                edge_class[i, j] = "present"
                consistent_sign[i, j] = int(nz[0])
            elif present_frac[i, j] <= 1.0 - PRESENT_CONSENSUS:
                edge_class[i, j] = "absent"
    mag_present = mags[:, edge_class == "present"].ravel().tolist()
    mag_absent = mags[:, edge_class == "absent"].ravel().tolist()
    report["edge_class"] = edge_class.astype(str).tolist()
    report["present_frac"] = present_frac.tolist()
    report["consistent_sign"] = consistent_sign.tolist()
    report["mag_present"] = mag_present
    report["mag_absent"] = mag_absent
    separated = (len(mag_present) > 0 and len(mag_absent) > 0
                 and min(mag_present) > max(mag_absent))
    report["separated"] = bool(separated)
    report["separation_point"] = (
        float(np.sqrt(min(mag_present) * max(mag_absent))) if separated else None)

    # 3 — invariant-combination spreads (log10 max pairwise), per subset
    def combos(r):
        ks = r["kstar_obs"]
        return dict(d_ratio=np.array([d_ratio_of(r["D"])]),
                    alpha_over_delta=r["alpha"] / r["delta"][:, None],
                    beta_over_delta=r["beta"] / r["delta"],
                    KA=r["KA"], KR=r["KR"],
                    Dk2_over_delta=r["D"] * ks**2 / r["delta"])

    subsets = dict(all=list(range(len(runs))),
                   turing=[i for i, r in enumerate(runs) if r["turing"]],
                   largest_cluster=cl_sorted[0][1])
    spread = {}
    for sname, idx in subsets.items():
        cs = [combos(runs[i]) for i in idx]
        spread[sname] = {q: log_spread([c[q] for c in cs]) for q in
                         ("d_ratio", "alpha_over_delta", "beta_over_delta",
                          "KA", "KR", "Dk2_over_delta")}
        spread[sname]["n"] = len(idx)
    report["log10_spread"] = spread

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"D5 baseline ({len(runs)} seeds): {n_turing}/{len(runs)} Turing; "
          f"{len(clusters)} distinct sign structures, cluster sizes "
          f"{report['cluster_sizes']}")
    print(f"edge classes:\n{edge_class}")
    if separated:
        print(f"zero-call: present/absent magnitudes SEPARATE; midpoint "
              f"{report['separation_point']:.4f} (present min {min(mag_present):.4f}, "
              f"absent max {max(mag_absent):.4f})")
    else:
        pmin = min(mag_present) if mag_present else float("nan")
        amax = max(mag_absent) if mag_absent else float("nan")
        print(f"zero-call: distributions DO NOT separate (present min {pmin:.4f} vs "
              f"absent max {amax:.4f}) — a finding, not a licence to pick a cut")
    for sname, sp in spread.items():
        print(f"log10 spread [{sname}, n={sp['n']}]: " +
              ", ".join(f"{q} {sp[q]:.2f}" for q in sp if q != "n"))
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
