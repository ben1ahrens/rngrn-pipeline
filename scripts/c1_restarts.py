"""c2_restarts.py — the SENSITIVE readout for unit C2.

`target_reports.jsonl` scores the ONE restart recovery returned (lowest total loss). At
400 steps on nc1 that number is 0 Turing everywhere, which is a floor: it cannot rank two
configs that both fail. But every run stores all `n_restarts` restarts in
`results/train_results.json`, each with its own `sig_max_pos` (the max of sigma(k) over the
instability hinge's support -- exactly the quantity that must become positive for a Turing
regime). Pooling those gives 64 x K samples per (cell, target) instead of K, and turns a
binary floor into a continuous one.

Reported per (root, target):
  n_pos     restarts with sig_max_pos > 0, out of n_restarts * K   <- the reachability rate
  sig_p90   90th percentile of sig_max_pos over all restarts        <- how close the best get
  sig_best  the single best sig_max_pos seen
  win_sig   sig_max_pos of the loss-argmin winner, averaged over seeds
  Dlo/Dmid  the near-immobile-third-node ratio of the winner (BIO_VIABILITY sec 1.2b)

`n_pos > 0` while the target report says 0 Turing means the SELECTION rule (lowest total
loss) is discarding Turing restarts; `n_pos == 0` means recovery never reaches the regime
at all. Those are different defects with different fixes, and this script separates them.
"""
import json
import sys
from pathlib import Path
from statistics import median


def run_rows(root):
    out = []
    for rd in sorted(Path(root).glob("runs/*")):
        f = rd / "results" / "train_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        m = d.get("metric", {})
        D = sorted(d.get("recovered", {}).get("D", []) or [])
        out.append(dict(
            run=rd.name,
            sample=m.get("sample_key") or d.get("sample_key"),
            restarts=d.get("restarts", []),
            loss=d.get("loss"),
            sig=m.get("recovered_sig_max"),
            turing=m.get("recovered_turing"),
            patterned=m.get("rollout_patterned"),
            kfft=m.get("kstar_fft_rel_err"),
            triv=m.get("trivial_kstar_err"),
            plaus=m.get("plausibility_score"),
            dratio=m.get("plausibility_d_ratio_value"),
            draw=(D[-1] / D[0]) if len(D) == 3 and D[0] > 0 else None,
            dlo_dmid=(D[0] / D[1]) if len(D) == 3 and D[1] > 0 else None,
        ))
    return out


def index_sample(root):
    """sample_key per run_id, from the run index (train_results.json omits it)."""
    p = Path(root) / "runs.jsonl"
    m = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                m[r["run_id"]] = r["sample_key"]
    return m


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(q * (len(xs) - 1)))
    return xs[i]


def main(roots):
    hdr = (f"{'root':<24} {'target':<12} {'K':>2} {'nRS':>5} {'n_pos':>6} {'rate':>7} "
           f"{'sig_p90':>8} {'sig_best':>9} {'win_sig':>8} {'Dlo/Dmid':>9} {'D_raw':>8}")
    print(hdr)
    print("-" * len(hdr))
    for root in roots:
        smap = index_sample(root)
        rows = run_rows(root)
        by = {}
        for r in rows:
            key = r["sample"] or smap.get(r["run"], "?")
            by.setdefault(key, []).append(r)
        for tgt in sorted(by):
            rs = by[tgt]
            sig = [x["sig_max_pos"] for r in rs for x in r["restarts"]
                   if x.get("sig_max_pos") is not None and x["sig_max_pos"] == x["sig_max_pos"]]
            npos = sum(1 for s in sig if s > 0)
            wins = [r["sig"] for r in rs if r["sig"] is not None]
            dlm = [r["dlo_dmid"] for r in rs if r["dlo_dmid"] is not None]
            draw = [r["draw"] for r in rs if r["draw"] is not None]
            print(f"{Path(root).name:<24} {tgt:<12} {len(rs):>2} {len(sig):>5} {npos:>6} "
                  f"{npos / max(1, len(sig)):>7.4f} {pct(sig, 0.90):>8.4f} "
                  f"{(max(sig) if sig else float('nan')):>9.4f} "
                  f"{(median(wins) if wins else float('nan')):>8.4f} "
                  f"{(median(dlm) if dlm else float('nan')):>9.4f} "
                  f"{(median(draw) if draw else float('nan')):>8.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
