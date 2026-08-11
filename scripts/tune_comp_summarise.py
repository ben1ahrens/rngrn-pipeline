"""tune_comp_summarise.py — collapse the tune_comp target-report JSONs into one table.

Reads experiments/tune_comp_reports/<cell>/<sample>.json and prints, per cell,
the pre-registered criteria in PRIORITY order (reproducibility, robustness, pattern,
viability) plus the rate. Reports NOTHING as pass/fail — the bars live in
docs/PREREGISTRATION.md and are applied by the reader.
"""
import json, sys, glob, os, math
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "experiments", "tune_comp_reports")

COLS = ["turing_frac", "recovered_frac", "topology_consistency", "mean_agreement",
        "turing_volume_10pct_median", "turing_volume_4p8pct_median",
        "kstar_fft_rel_err_median", "trivial_kstar_err_mean",
        "morphology_match_frac", "plausibility_score_mean", "kstar_spread"]


def load(cell):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, cell, "*.json"))):
        try:
            out[os.path.basename(p)[:-5]] = json.load(open(p))
        except Exception as e:
            print(f"  !! unreadable {p}: {e}", file=sys.stderr)
    return out


def fmt(v):
    if v is None:
        return "  -  "
    try:
        f = float(v)
    except Exception:
        return str(v)[:6]
    return "  nan" if math.isnan(f) else f"{f:6.3f}"


def main(cells):
    for cell in cells:
        reps = load(cell)
        if not reps:
            continue
        print(f"\n### {cell}   ({len(reps)} target(s))")
        hdr = f"{'sample':<14}" + "".join(f"{c[:13]:>15}" for c in COLS)
        print(hdr)
        acc = {c: [] for c in COLS}
        for s, r in reps.items():
            row = f"{s:<14}"
            for c in COLS:
                v = r.get(c)
                row += f"{fmt(v):>15}"
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        acc[c].append(fv)
                except Exception:
                    pass
            print(row)
            # per-seed turing detail
            ps = json.loads(r.get("per_seed", "[]"))
            tur = [p["seed"] for p in ps if p.get("recovered_turing")]
            fail = [p["seed"] for p in ps if p.get("status") != "ok"]
            print(f"{'':<14}  turing seeds={tur}  failed={fail}")
        print(f"{'MEAN':<14}" + "".join(f"{fmt(np.mean(acc[c]) if acc[c] else float('nan')):>15}"
                                        for c in COLS))


if __name__ == "__main__":
    cells = sys.argv[1:] or sorted(os.listdir(ROOT))
    main(cells)
