"""c2_summary.py — read the target_reports table of one or more C2 runs-roots and print
the pre-registered criteria per (cell, target), ranked the way PREREGISTRATION.md ranks
them: reproducibility (3.1), robustness (3.2), pattern (3.3), viability (3.4).

Reports RATES with K in the denominator, never a filtered denominator.
Usage:  python scripts/c2_summary.py experiments/<root> [experiments/<root> ...]
"""
import json
import sys
from pathlib import Path


def rows(root):
    p = Path(root) / "target_reports.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            r["_root"] = root
            out.append(r)
    return out


def fmt(v, n=3):
    if v is None:
        return "  -  "
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return " nan "
    return f"{f:.{n}f}"


def main(roots):
    hdr = (f"{'root':<26} {'target':<12} {'K':>2} {'rec':>4} {'tur':>4} "
           f"{'topo':>6} {'ndist':>5} {'tv10':>6} {'tv4.8':>6} "
           f"{'kfft':>6} {'triv':>6} {'morphN':>6} {'mmatch':>6} {'plaus':>6}")
    print(hdr)
    print("-" * len(hdr))
    for root in roots:
        for r in sorted(rows(root), key=lambda x: x["sample_key"]):
            print(f"{Path(root).name:<26} {r['sample_key']:<12} "
                  f"{r['n_seeds_requested']:>2} {r['n_recovered']:>4} {r['n_turing']:>4} "
                  f"{fmt(r.get('topology_consistency')):>6} "
                  f"{fmt(r.get('n_distinct_structures'), 0):>5} "
                  f"{fmt(r.get('turing_volume_10pct_median')):>6} "
                  f"{fmt(r.get('turing_volume_4p8pct_median')):>6} "
                  f"{fmt(r.get('kstar_fft_rel_err_median')):>6} "
                  f"{fmt(r.get('trivial_kstar_err_mean')):>6} "
                  f"{r.get('morphology_n_compared', 0):>6} "
                  f"{fmt(r.get('morphology_match_frac')):>6} "
                  f"{fmt(r.get('plausibility_score_mean')):>6}")
    print()
    for root in roots:
        rs = rows(root)
        if not rs:
            continue
        K = sum(r["n_seeds_requested"] for r in rs)
        tur = sum(r["n_turing"] for r in rs)
        rec = sum(r["n_recovered"] for r in rs)
        tgt_any = sum(1 for r in rs if r["n_turing"] > 0)
        print(f"{Path(root).name:<26} seeds={K:>3} recovered={rec:>3} "
              f"turing_seeds={tur:>3} ({tur / K:.3f})  targets_with_any_turing="
              f"{tgt_any}/{len(rs)}")


if __name__ == "__main__":
    main(sys.argv[1:])
