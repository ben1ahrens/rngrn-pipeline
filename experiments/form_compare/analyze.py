#!/usr/bin/env python
"""analyze.py — nc1 vs competitive robustness comparison numbers (paper form-robustness unit).

Reads persisted run records directly by absolute path -- no rngrn import needed, so this
script has no data-firewall exposure of its own (it reads recovered-model metrics, never
answer-key generating parameters).

Data sources (read-only):
  nc1 (MAIN checkout, committed):
    /home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/c2_P_t8k8_consol/
      runs/*/results/train_results.json          -- 16 runs, sample_0001 (x8) + sample_0004 (x8)
  competitive (this worktree, produced by experiments/form_compare/run_cells.sh):
    experiments/form_compare/comp_0001/runs/*/results/train_results.json
    experiments/form_compare/comp_0004/runs/*/results/train_results.json
  population baseline (MAIN checkout):
    /home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/exp11_robustness_baseline.csv
      -- 127 generator systems x 400 perturbation draws, frac_strict at sigma in
         {0.01, 0.048, 0.1, 0.2} = the 1%/4.8%/10%/20% perturbation levels

See docs/DECISIONS.md::D-FORMCOMP-1 for the full design and the disclosed
nc1-tuned-hyperparameter confound. This script computes numbers only -- it invents no
threshold and applies no pass/fail beyond reporting the pre-registered PREREGISTRATION.md
Sec 3.2 bars (median turing_volume_10pct >= 0.90, median turing_volume_4p8pct >= 0.95)
alongside each form's own numbers.
"""
from __future__ import annotations
import csv
import glob
import json
import statistics as st
import os

MAIN = "/home/benja/projects/personal/rngrn/rngrn-pipeline"
WT = "/home/benja/projects/personal/rngrn/worktrees/paper-formcomp"
BASELINE_CSV = os.path.join(MAIN, "experiments/exp11_robustness_baseline.csv")
NC1_ROOT = os.path.join(MAIN, "experiments/c2_P_t8k8_consol")
COMP_ROOT = os.path.join(WT, "experiments/form_compare")

TARGETS = ["sample_0001", "sample_0004"]
TV_KEYS = ["turing_volume_1pct", "turing_volume_4p8pct", "turing_volume_10pct", "turing_volume_20pct"]
PASS_TV10 = 0.90   # PREREGISTRATION.md Sec 3.2
PASS_TV48 = 0.95   # PREREGISTRATION.md Sec 3.2


def load_baseline():
    rows = list(csv.DictReader(open(BASELINE_CSV)))
    by_sigma = {s: [] for s in ["0.01", "0.048", "0.1", "0.2"]}
    for r in rows:
        s = r["sigma"]
        if s in by_sigma:
            by_sigma[s].append(float(r["frac_strict"]))
    return by_sigma


def _target_of_run_dir(d, form_prefix):
    """Recover which sample_key a run directory belongs to.

    nc1 c2_P_t8k8_consol interleaves sample_0001 and sample_0004 runs under one runs/
    directory distinguished only by run_name timestamp groups (8 seeds per target, two
    groups) -- read it back off the frozen config rather than assuming order.
    """
    cfg_path = os.path.join(d, "config", "frozen_config.yaml")
    if os.path.exists(cfg_path):
        for line in open(cfg_path):
            if "sample_key" in line:
                for t in TARGETS:
                    if t in line:
                        return t
    return None


def load_target_report(runs_root, target):
    """Read the authoritative per-target aggregate row (recovered_frac against the true
    n_seeds_requested denominator, seed_errors, topology_consistency) from
    <runs_root>/target_reports.jsonl. Returns None if no matching row exists yet."""
    path = os.path.join(runs_root, "target_reports.jsonl")
    if not os.path.exists(path):
        return None
    for line in open(path):
        d = json.loads(line)
        if d.get("sample_key") == target:
            return d
    return None


def load_runs(root, per_target_dirs=None):
    """Load all results/train_results.json under root/runs/*, or under
    per_target_dirs[target]/runs/* if given (competitive: one runs-root per target)."""
    out = {t: [] for t in TARGETS}
    if per_target_dirs:
        for t, d in per_target_dirs.items():
            for p in sorted(glob.glob(os.path.join(d, "runs/*/results/train_results.json"))):
                out[t].append(json.load(open(p)))
        return out
    for p in sorted(glob.glob(os.path.join(root, "runs/*/results/train_results.json"))):
        run_dir = os.path.dirname(os.path.dirname(p))
        t = _target_of_run_dir(run_dir, None)
        if t is None:
            raise RuntimeError(f"could not determine target for {run_dir}")
        out[t].append(json.load(open(p)))
    return out


def stats_block(runs):
    """runs: list of train_results.json dicts (one target's seeds). Returns a dict of
    per-metric stats plus the raw per-seed values, over all K seeds (not just Turing ones,
    except where a metric is only defined for Turing/compared seeds -- noted per key)."""
    n = len(runs)
    metrics = [r["metric"] for r in runs]
    turing_metrics = [m for m in metrics if m.get("recovered_turing")]
    n_turing = len(turing_metrics)
    out = {
        "n_seeds": n,
        # NOTE: `runs` here is the set of train_results.json files actually present under
        # runs/*/results/, i.e. seeds that produced a scored recovery. A seed that RAISED
        # in target-report never gets a results/ dir, so it is silently absent from this
        # count rather than appearing as a 0 -- recovered_frac is therefore NOT computed
        # here (would read as 1.0 by construction). Read recovered_frac and seed_errors
        # from the runs-root's target_reports.jsonl instead, which target-report writes
        # against the true denominator n_seeds_requested (see D-FORMCOMP-1 README note).
        "turing_frac": n_turing / n if n else float("nan"),
        "n_turing": n_turing,
    }
    # turing_volume_* is scored per PREREGISTRATION §3.2 over TURING-REACHING seeds only
    # (src/rngrn/optim/target_report.py::_robustness_block(turing_rows)) -- a non-Turing
    # seed's local perturbation-cloud volume is not a robustness measurement of a Turing
    # pattern and pooling it in would understate robustness for a form with a lower
    # turing_frac, which is exactly the axis this comparison must not bias.
    for k in TV_KEYS:
        vals = [m[k] for m in turing_metrics if k in m and m[k] is not None]
        out[k] = {
            "n": len(vals),
            "median": st.median(vals) if vals else None,
            "mean": st.mean(vals) if vals else None,
            "min": min(vals) if vals else None,
            "per_seed": vals,
        }
    for k in ["kstar_fft_rel_err", "kstar_rel_err", "trivial_kstar_err"]:
        vals = [m[k] for m in metrics if k in m and m[k] is not None]
        out[k] = {"median": st.median(vals) if vals else None,
                   "mean": st.mean(vals) if vals else None,
                   "per_seed": vals}
    morph_scored = [m for m in metrics if m.get("morphology_scored") == "compared"]
    out["morphology_match_frac"] = {
        "n_compared": len(morph_scored),
        "n_match": sum(1 for m in morph_scored if m.get("morphology_match")),
        "frac": (sum(1 for m in morph_scored if m.get("morphology_match")) / len(morph_scored)
                 if morph_scored else None),
    }
    dvals = [m["morphology_distance"] for m in morph_scored if m.get("morphology_distance") is not None]
    out["morphology_distance_mean"] = st.mean(dvals) if dvals else None
    plaus = [m["plausibility_score"] for m in metrics if m.get("plausibility_score") is not None]
    out["plausibility_score_mean"] = st.mean(plaus) if plaus else None
    return out


def pooled_stats_block(runs_by_target):
    all_runs = [r for t in TARGETS for r in runs_by_target[t]]
    return stats_block(all_runs)


def summarize(label, runs_by_target, runs_root_by_target):
    out = {"label": label, "per_target": {}, "pooled": None}
    for t in TARGETS:
        block = stats_block(runs_by_target[t])
        tr = load_target_report(runs_root_by_target[t], t)
        if tr is not None:
            block["recovered_frac"] = tr["recovered_frac"]
            block["n_seeds_requested"] = tr["n_seeds_requested"]
            block["seed_errors"] = json.loads(tr["seed_errors"])
            block["topology_consistency"] = tr["topology_consistency"]
        else:
            block["recovered_frac"] = None
            block["n_seeds_requested"] = None
            block["seed_errors"] = None
            block["topology_consistency"] = None
        out["per_target"][t] = block
    out["pooled"] = pooled_stats_block(runs_by_target)
    return out


def render_markdown(nc1, comp, baseline):
    lines = []
    lines.append("# form_compare numbers -- nc1 vs competitive robustness comparison\n")
    lines.append("Auto-generated by `experiments/form_compare/analyze.py`. "
                  "See `docs/DECISIONS.md::D-FORMCOMP-1` for design and the disclosed "
                  "nc1-tuned-hyperparameter confound. Population baseline: "
                  "`exp11_robustness_baseline.csv`, n=127 systems x 400 draws.\n")

    lines.append("## Population baseline (n=127 x 400 draws)\n")
    lines.append("| level | median | mean | worst |")
    lines.append("|---|---|---|---|")
    for s, lab in zip(["0.01", "0.048", "0.1", "0.2"], ["1%", "4.8%", "10%", "20%"]):
        vals = baseline[s]
        lines.append(f"| {lab} | {st.median(vals):.3f} | {st.mean(vals):.3f} | {min(vals):.3f} |")
    lines.append("")

    def fmt(x, spec=".4f"):
        return format(x, spec) if x is not None else "n/a"

    for block, label in [(nc1, "nc1 (c2_P_t8k8_consol, committed)"),
                          (comp, "competitive (form_compare, this unit)")]:
        lines.append(f"## {label}\n")
        for scope in TARGETS + ["pooled"]:
            d = block["per_target"][scope] if scope in TARGETS else block["pooled"]
            lines.append(f"### {scope}\n")
            lines.append(f"- n_seeds (scored)={d['n_seeds']}, turing_frac={d['turing_frac']:.3f} "
                         f"({d['n_turing']}/{d['n_seeds']})")
            if scope in TARGETS:
                lines.append(f"- recovered_frac (of n_seeds_requested)="
                             f"{fmt(d.get('recovered_frac'), '.3f')} "
                             f"(n_seeds_requested={d.get('n_seeds_requested')}, "
                             f"seed_errors={d.get('seed_errors')})")
                lines.append(f"- topology_consistency (rtol 0.05)="
                             f"{fmt(d.get('topology_consistency'), '.3f')}")
            lines.append("| metric | median | mean | min | n |")
            lines.append("|---|---|---|---|---|")
            for k, pretty in [("turing_volume_1pct", "tv 1%"),
                               ("turing_volume_4p8pct", "tv 4.8%"),
                               ("turing_volume_10pct", "tv 10%"),
                               ("turing_volume_20pct", "tv 20%")]:
                v = d[k]
                lines.append(f"| {pretty} | {fmt(v['median'])} | {fmt(v['mean'])} | "
                             f"{fmt(v['min'])} | {v['n']} |")
            for k, pretty in [("kstar_fft_rel_err", "kstar_fft_rel_err"),
                               ("kstar_rel_err", "kstar_rel_err (linear)"),
                               ("trivial_kstar_err", "trivial_kstar_err (control)")]:
                v = d[k]
                lines.append(f"| {pretty} | {fmt(v['median'])} | {fmt(v['mean'])} | -- | "
                             f"{len(v['per_seed'])} |")
            lines.append("")
            mm = d["morphology_match_frac"]
            if mm["frac"] is not None:
                lines.append(f"- morphology_match_frac: {mm['n_match']}/{mm['n_compared']} "
                             f"({mm['frac']:.3f})")
            else:
                lines.append("- morphology_match_frac: n/a (0 compared)")
            lines.append(f"- morphology_distance (mean, compared only): "
                         f"{fmt(d['morphology_distance_mean'])}")
            lines.append(f"- plausibility_score (mean): {fmt(d['plausibility_score_mean'])}")
            lines.append("")
        pt10 = block["pooled"]["turing_volume_10pct"]["median"]
        pt48 = block["pooled"]["turing_volume_4p8pct"]["median"]
        cmp10 = (">=" if pt10 is not None and pt10 >= PASS_TV10 else "<") if pt10 is not None else "n/a vs"
        cmp48 = (">=" if pt48 is not None and pt48 >= PASS_TV48 else "<") if pt48 is not None else "n/a vs"
        lines.append(f"**Sec 3.2 bars (pre-registered, reported per-form, not gated):** "
                     f"pooled median tv10={fmt(pt10)} {cmp10} {PASS_TV10}; "
                     f"pooled median tv4.8={fmt(pt48)} {cmp48} {PASS_TV48}\n")
    return "\n".join(lines)


def main():
    baseline = load_baseline()
    nc1_runs = load_runs(NC1_ROOT)
    comp_dirs = {"sample_0001": os.path.join(COMP_ROOT, "comp_0001"),
                 "sample_0004": os.path.join(COMP_ROOT, "comp_0004")}
    comp_runs = load_runs(None, per_target_dirs=comp_dirs)

    for t in TARGETS:
        assert len(nc1_runs[t]) == 8, f"nc1 {t}: expected 8 runs, got {len(nc1_runs[t])}"

    nc1_root_by_target = {t: NC1_ROOT for t in TARGETS}
    nc1 = summarize("nc1", nc1_runs, nc1_root_by_target)
    comp = summarize("competitive", comp_runs, comp_dirs)

    out_json = {"baseline_summary": {s: {"median": st.median(v), "mean": st.mean(v), "min": min(v)}
                                       for s, v in baseline.items()},
                "nc1": nc1, "competitive": comp}
    json_path = os.path.join(COMP_ROOT, "numbers.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"wrote {json_path}")

    md = render_markdown(nc1, comp, baseline)
    md_path = os.path.join(COMP_ROOT, "numbers.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"wrote {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
