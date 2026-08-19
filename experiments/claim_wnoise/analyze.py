"""analyze.py — paper-wnoise unit (Unit A): tables + figure for the weight-noise curve.

Reads the four cells' target_reports.jsonl (headline aggregates) and runs.jsonl
(seed-level rows), writes:
  * summary_table.json / summary_table.csv  (tracked — the numbers behind the claim)
  * seed_level.csv                          (tracked — full seed-level distributions)
  * figures/wnoise_curves.png               (gitignored — regenerate with this script)

Pre-registered analysis: docs/DECISIONS.md::D-WNOISE-1. No pass/fail language — the
probe levels are UNCALIBRATED; this reports the measured curve against the sigma_w=0
control. Run from the worktree root:  .venv/bin/python experiments/claim_wnoise/analyze.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CELLS = [("0.0", "sigma_w_0p00"), ("0.048", "sigma_w_0p048"),
         ("0.10", "sigma_w_0p10"), ("0.20", "sigma_w_0p20")]
VOL_COLS = ("turing_volume_1pct", "turing_volume_4p8pct",
            "turing_volume_10pct", "turing_volume_20pct")


def _rows(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load():
    out = []
    for sigma_w, d in CELLS:
        cell = os.path.join(ROOT, d)
        tr_path = os.path.join(cell, "target_reports.jsonl")
        runs_path = os.path.join(cell, "runs.jsonl")
        if not (os.path.exists(tr_path) and os.path.exists(runs_path)):
            print(f"[skip] {d}: missing report/runs (cell not finished?)")
            continue
        trs = _rows(tr_path)
        if len(trs) != 1:
            raise RuntimeError(f"{d}: expected exactly 1 target report row, got {len(trs)}")
        out.append((sigma_w, d, trs[0], _rows(runs_path)))
    return out


def main():
    cells = load()
    if not cells:
        raise SystemExit("no finished cells found")

    summary = []
    for sigma_w, d, tr, runs in cells:
        row = dict(
            sigma_w=float(sigma_w), cell=d,
            n_seeds=int(tr["n_seeds_requested"]), n_recovered=int(tr["n_recovered"]),
            n_turing=int(tr["n_turing"]),
            kstar_fft_rel_err_mean=tr.get("kstar_fft_rel_err_mean"),
            kstar_fft_rel_err_median=tr.get("kstar_fft_rel_err_median"),
            morphology_match_frac=tr.get("morphology_match_frac"),
            morphology_n_compared=tr.get("morphology_n_compared"),
        )
        for c in VOL_COLS:
            row[f"{c}_mean"] = tr.get(f"{c}_mean")
            row[f"{c}_median"] = tr.get(f"{c}_median")
        summary.append(row)

    with open(os.path.join(ROOT, "summary_table.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(ROOT, "summary_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    seed_cols = ["seed", "recovered_turing", "kstar_fft_rel_err", "kstar_rel_err",
                 "morphology_match", "morphology_scored", "rollout_patterned",
                 "rollout_status"] + list(VOL_COLS)
    with open(os.path.join(ROOT, "seed_level.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_w", "run_id"] + seed_cols)
        for sigma_w, d, tr, runs in cells:
            for r in runs:
                w.writerow([sigma_w, r.get("run_id")] + [r.get(c) for c in seed_cols])

    # ---- console table ---------------------------------------------------------------
    hdr = ("sigma_w n_turing " +
           " ".join(f"vol{t}med/mean" for t in ("1", "4p8", "10", "20")) +
           " kstar_fft_err morph")
    print(hdr)
    for row in summary:
        vols = " ".join(
            f"{(row[f'{c}_median'] if row[f'{c}_median'] is not None else float('nan')):.3f}/"
            f"{(row[f'{c}_mean'] if row[f'{c}_mean'] is not None else float('nan')):.3f}"
            for c in VOL_COLS)
        ke = row["kstar_fft_rel_err_mean"]
        mm = row["morphology_match_frac"]
        print(f"{row['sigma_w']:<7} {row['n_turing']}/{row['n_seeds']}   {vols} "
              f"{(ke if ke is not None else float('nan')):.4f} "
              f"{(mm if mm is not None else float('nan'))}")

    # ---- figure (gitignored; regenerable by this script) -----------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    sig = [r["sigma_w"] for r in summary]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    for c, lab in zip(VOL_COLS, ("1%", "4.8%", "10%", "20%")):
        ax.plot(sig, [r[f"{c}_median"] for r in summary], "o-", label=f"median @{lab}")
    ax.set_xlabel("train weight-noise sigma_w"); ax.set_ylabel("local Turing volume")
    ax.set_title("median Turing volume vs sigma_w\n(per evaluation perturbation level)")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)

    ax = axes[1]
    # seed-level 20pct distribution: the column where the control is not at ceiling
    for i, (sigma_w, d, tr, runs) in enumerate(cells):
        vals = [r.get("turing_volume_20pct") for r in runs
                if r.get("turing_volume_20pct") is not None]
        ax.plot([float(sigma_w)] * len(vals), vals, "o", alpha=0.55, ms=5)
        med = sorted(vals)[len(vals) // 2] if vals else None
        if med is not None:
            ax.plot([float(sigma_w)], [med], "k_", ms=18)
    ax.set_xlabel("train weight-noise sigma_w"); ax.set_ylabel("turing_volume_20pct")
    ax.set_title("seed-level Turing volume @20%\n(dash = median)")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[2]
    ax.bar([str(s) for s in sig], [r["n_turing"] for r in summary], color="#88a")
    ax.set_xlabel("train weight-noise sigma_w")
    ax.set_ylabel(f"seeds recovered Turing (of {summary[0]['n_seeds']})")
    ax.set_title("recovered_turing count")

    fig.suptitle("Weight-noise curve — three_gene_qvar/sample_0001, 8 seeds/level "
                 "(UNCALIBRATED probe levels; D-WNOISE-1)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(figdir, "wnoise_curves.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
