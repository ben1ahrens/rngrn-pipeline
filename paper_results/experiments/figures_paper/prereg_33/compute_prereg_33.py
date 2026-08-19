"""compute_prereg_33.py — close PREREGISTRATION.md §3.3's third sub-condition.

WHAT THIS COMPUTES
-------------------
docs/PREREGISTRATION.md §3.3 ("Same pattern") reads:

    `morphology_match` true, and `morphology_distance` <= the 75th percentile of
    within-class distances in `three_gene_train`, and `kstar_fft_rel_err` <= 8.3%,
    and `trivial_kstar_err` reported in the same record.

The 75th-percentile threshold was never computed. This script computes it, exactly as
worded: the 75th percentile of the distribution of pairwise `morphology_distance`
between two DIFFERENT samples of the SAME class ("within-class"), over the labelled
samples of the `three_gene_train` dataset (n=88: 42 labyrinth, 42 spots, 4 stripes,
per src/rngrn/scoring/morphology.py's own recorded provenance for TRAIN_SCALE/
TRAIN_CLASS_MEANS, reproduced here from the same payload).

AMBIGUITY, MADE EXPLICIT
-------------------------
"the 75th percentile of within-class distances in three_gene_train" pools grammatically
to ONE number over ONE set — it does not say "per class". We compute and report BOTH:
  - POOLED  (primary, matches the literal wording): all within-class pairs from all
    three classes concatenated into one distribution, one 75th percentile.
  - PER-CLASS (context): the within-class pairwise distances computed separately per
    class label, each with its own 75th percentile. Useful because a run is judged
    against its OWN target's class in spirit, even though the prereg sentence does not
    condition the threshold on class.
The adjudication in step 2 uses the POOLED threshold as primary (matches the wording),
and reports the per-class threshold as a secondary check for context.

DATA / FIREWALL NOTE
---------------------
This reads data/datasets/three_gene_train/payload.h5 directly (ground-truth morphology
label + final_frame). That is legal here: this is a standalone, scoring-side, post-hoc
analysis script living under experiments/, not under src/rngrn/{losses,eval} or
scripts/ — it is not part of the recovery path and is not on the firewall's forbidden
list surface. It computes exactly what
src/rngrn/scoring/morphology.py's TRAIN_SCALE_PROVENANCE docstring already asserts was
computed from this same file (reproduced here for the percentile, not re-derived).

morphology_distance(a, b, scale=None) is called with its own default scale (TRAIN_SCALE,
itself measured from this exact 88-sample split) — the natural, already-established scale
for this computation.

USAGE
-----
    cd worktrees/lift-ladder
    .venv/bin/python experiments/figures_paper/prereg_33/compute_prereg_33.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import h5py
import numpy as np

import sys
# Collected into paper_results/ (branch feature/paper-results): the campaign run
# data lives under the pack root; the dataset payload and src/ come from the
# enclosing repo checkout. Original roots: the lift-ladder worktree
# (WORKTREE_ROOT) and the main checkout (MAIN_CHECKOUT).
_PACK = Path(__file__).resolve().parents[3]   # .../paper_results
_REPO = _PACK.parent                          # repo root: payload symlinks, src/
sys.path.insert(0, str(_REPO / "src"))
from rngrn.scoring.morphology import morphology_distance  # noqa: E402

HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = _PACK
MAIN_CHECKOUT = _REPO
DATASET_ID = "three_gene_train"
PAYLOAD = MAIN_CHECKOUT / "data" / "datasets" / DATASET_ID / "payload.h5"
CAMPAIGN_RUNS = _PACK / "experiments" / "c2_P_t8k8_consol" / "runs"
OUT_DIR = HERE


def load_labelled_fields(payload_path: Path) -> dict:
    """{label: [2-D field (channel 0 of final_frame), ...]} for every sample."""
    fields_by_label: dict = {}
    with h5py.File(payload_path, "r") as f:
        for key in f.keys():
            grp = f[key]
            label = str(grp.attrs["morphology"])
            field = np.asarray(grp["final_frame"][0], dtype=float)
            fields_by_label.setdefault(label, []).append(field)
    return fields_by_label


def within_class_distances(fields_by_label: dict) -> dict:
    """{label: [morphology_distance(a, b) for every unordered pair within the class]}"""
    out = {}
    for label, fields in fields_by_label.items():
        dists = [morphology_distance(a, b) for a, b in itertools.combinations(fields, 2)]
        out[label] = dists
    return out


def main() -> None:
    if not PAYLOAD.exists():
        raise FileNotFoundError(f"payload not found: {PAYLOAD}")

    fields_by_label = load_labelled_fields(PAYLOAD)
    n_per_class = {k: len(v) for k, v in fields_by_label.items()}
    n_total = sum(n_per_class.values())

    dists_by_class = within_class_distances(fields_by_label)
    n_pairs_per_class = {k: len(v) for k, v in dists_by_class.items()}

    pooled = [d for dists in dists_by_class.values() for d in dists]
    pooled_p75 = float(np.percentile(pooled, 75))

    per_class_p75 = {k: float(np.percentile(v, 75)) for k, v in dists_by_class.items()}
    per_class_summary = {
        k: {
            "n_samples": n_per_class[k],
            "n_pairs": n_pairs_per_class[k],
            "p75": per_class_p75[k],
            "min": float(np.min(v)),
            "median": float(np.median(v)),
            "max": float(np.max(v)),
        }
        for k, v in dists_by_class.items()
    }

    # --- Step 2: adjudicate the c2_P campaign's 16 runs against the pooled threshold ---
    run_dirs = sorted(p for p in CAMPAIGN_RUNS.iterdir() if p.is_dir())
    adjudication = []
    for run_dir in run_dirs:
        results_path = run_dir / "results" / "train_results.json"
        with open(results_path) as fh:
            results = json.load(fh)
        m = results["metric"]
        scored = m.get("morphology_scored")
        target_class = m.get("morphology_pred_target")
        dist = m.get("morphology_distance")  # absent when scored == "target_only"
        match = m.get("morphology_match")
        kstar_fft_rel_err = m.get("kstar_fft_rel_err")
        trivial_kstar_err = m.get("trivial_kstar_err")

        if dist is None:
            # No model-side field to compare (morphology_scored != "compared"): the
            # distance sub-condition cannot be evaluated, so it fails by construction.
            pass_distance_pooled = False
            pass_distance_per_class = False
        else:
            pass_distance_pooled = dist <= pooled_p75
            class_p75 = per_class_p75.get(target_class)
            pass_distance_per_class = (
                (dist <= class_p75) if class_p75 is not None else None
            )
        pass_full_3_3 = bool(
            match
            and pass_distance_pooled
            and kstar_fft_rel_err is not None
            and kstar_fft_rel_err <= 0.083
            and trivial_kstar_err is not None
        )

        adjudication.append({
            "run": run_dir.name,
            "morphology_scored": scored,
            "morphology_pred_target": target_class,
            "morphology_match": match,
            "morphology_distance": dist,
            "pass_distance_pooled_threshold": pass_distance_pooled,
            "pass_distance_per_class_threshold": pass_distance_per_class,
            "kstar_fft_rel_err": kstar_fft_rel_err,
            "kstar_fft_rel_err_le_8_3pct": (
                kstar_fft_rel_err is not None and kstar_fft_rel_err <= 0.083
            ),
            "trivial_kstar_err": trivial_kstar_err,
            "trivial_kstar_err_reported": trivial_kstar_err is not None,
            "pass_full_section_3_3": pass_full_3_3,
        })

    n_pass_distance_pooled = sum(a["pass_distance_pooled_threshold"] for a in adjudication)
    n_pass_full = sum(a["pass_full_section_3_3"] for a in adjudication)

    results_out = {
        "prereg_quote": (
            "`morphology_match` true, and `morphology_distance` <= the 75th percentile "
            "of within-class distances in `three_gene_train`, and `kstar_fft_rel_err` "
            "<= 8.3%, and `trivial_kstar_err` reported in the same record."
        ),
        "dataset_id": DATASET_ID,
        "payload_path": str(PAYLOAD),
        "n_samples_total": n_total,
        "n_per_class": n_per_class,
        "distance_scale": "morphology_distance default scale (TRAIN_SCALE, "
                          "data-derived from this same three_gene_train split)",
        "ambiguity_note": (
            "'the 75th percentile of within-class distances in three_gene_train' pools "
            "grammatically to one number over one set (not per-class). POOLED is "
            "reported as primary/adjudicating; PER-CLASS is context."
        ),
        "threshold_pooled_p75": pooled_p75,
        "n_pairs_pooled": len(pooled),
        "per_class": per_class_summary,
        "campaign": "c2_P_t8k8_consol",
        "n_runs": len(adjudication),
        "n_pass_distance_pooled_threshold": int(n_pass_distance_pooled),
        "n_pass_full_section_3_3": int(n_pass_full),
        "adjudication": adjudication,
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results_out, indent=2))

    print(f"pooled p75 threshold: {pooled_p75:.4f}  (n_pairs={len(pooled)})")
    for k, v in per_class_summary.items():
        print(f"  {k:10s} p75={v['p75']:.4f}  n_samples={v['n_samples']} n_pairs={v['n_pairs']}")
    print(f"distance sub-condition (pooled threshold): {n_pass_distance_pooled}/{len(adjudication)} pass")
    print(f"full §3.3 conjunction (all 4 sub-conditions): {n_pass_full}/{len(adjudication)} pass")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"labyrinth": "#4C72B0", "spots": "#DD8452", "stripes": "#55A868"}
    for label, dists in dists_by_class.items():
        ax.hist(dists, bins=max(4, min(15, len(dists))), alpha=0.5,
                label=f"{label} (n_pairs={len(dists)})",
                color=colors.get(label, "gray"), density=False)
    ax.axvline(pooled_p75, color="black", linestyle="--", linewidth=2,
               label=f"pooled p75 = {pooled_p75:.3f}")

    run_dists = [a["morphology_distance"] for a in adjudication if a["morphology_distance"] is not None]
    y_top = ax.get_ylim()[1]
    ax.scatter(run_dists, [y_top * 0.05] * len(run_dists), marker="x", color="red",
               zorder=5, label=f"c2_P runs (n={len(run_dists)})")

    ax.set_xlabel("morphology_distance (z-scored 4-D units)")
    ax.set_ylabel("count (within-class pairs)")
    ax.set_title("PREREG §3.3: within-class morphology distances, three_gene_train\n"
                 "75th-percentile threshold vs. c2_P recovered-model distances")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distance_distribution.png", dpi=150)
    print(f"wrote {OUT_DIR / 'distance_distribution.png'}")


if __name__ == "__main__":
    main()
