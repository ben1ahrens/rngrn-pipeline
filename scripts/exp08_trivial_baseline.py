"""EXP08 - The trivial-baseline control. Is recovered k* actually FROM THE IMAGE?

THE CONFOUND
  The generators set L = clip(6 * 2pi/k*_true, 18, 220), so for every unclipped sample
  k*_true == 6*2pi/L EXACTLY. L is a legal recovery input (it is geometry, not an answer
  key), which means a "recovery" that ignores the image entirely and returns 6*2pi/L
  scores 0.000% error. Any k* number we report is therefore uninterpretable until we
  separate image-derived skill from this arithmetic identity.

THE CONTROLS
  trivial   : predict k* = 6*2pi/L. Uses NO image information at all.
  fft_only  : predict k* = kstar_of(frame, L). Uses the image, but it is also the loss
              target, so it measures the anchor, not the recovery.
  recovery  : the full staged fit (what we are actually evaluating).
  shuffled_L: run recovery on the frame but with ANOTHER sample's L. If recovery is
              genuinely reading the image, mis-stating L should damage it in a way that
              tracks the image, not simply reproduce 6*2pi/L_wrong. If instead the
              recovered k* follows the WRONG L, recovery is just inverting the identity.

  The last one is the discriminating test, and it is the reason this script exists.
"""
import argparse, json
import numpy as np, torch, h5py
from rngrn import observables as obs
import _runlog
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "exp05", pathlib.Path(__file__).parent / "exp05_pixel_minibatch.py")
exp05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp05)


def rel(a, b):
    return abs(a - b) / b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/exp08_trivial_baseline.json")
    a = ap.parse_args()
    run = _runlog.start("exp08_trivial_baseline", vars(a))
    path = f"data/datasets/{a.dataset}/payload.h5"
    with h5py.File(path) as f:
        keys = list(f.keys())[:a.samples]
        data = {k: (f[k]["final_frame"][:], float(f[k].attrs["L"]),
                    float(f[k].attrs["k_star"])) for k in keys}
    Ls = [v[1] for v in data.values()]
    rows = []
    print(f"{'sample':12s} {'L':>7s} {'k*_true':>8s} {'trivial':>8s} {'fft':>8s} "
          f"{'recovery':>9s} {'shuffL':>8s}")
    for i, (k, (frame, L, kt)) in enumerate(data.items()):
        ft = torch.tensor(np.asarray(frame, dtype=float))
        triv = 6 * 2 * np.pi / L
        ko = obs.kstar_of(frame[0], L=L)
        L_wrong = Ls[(i + 1) % len(Ls)]                      # another sample's L
        ko_wrong = obs.kstar_of(frame[0], L=L_wrong)
        def best_of(Luse, kobs):
            out = []
            for s in range(a.seeds):
                r = exp05.fit(s, ft, Luse, kobs, a.steps, None, 0.0)
                if r and r["turing"]:
                    out.append(r["kstar"])
            return float(np.median(out)) if out else None
        rec = best_of(L, ko)
        shuf = best_of(L_wrong, ko_wrong)
        row = dict(sample=k, L=L, kstar_true=kt, trivial=triv, fft=ko,
                   recovery=rec, shuffled_L=L_wrong, recovery_shuffled=shuf,
                   trivial_err=rel(triv, kt), fft_err=rel(ko, kt),
                   recovery_err=rel(rec, kt) if rec else None,
                   shuffled_err=rel(shuf, kt) if shuf else None)
        rows.append(row)
        print(f"{k:12s} {L:7.1f} {kt:8.4f} {triv:8.4f} {ko:8.4f} "
              f"{rec if rec else float('nan'):9.4f} {shuf if shuf else float('nan'):8.4f}")
    def med(field):
        v = [r[field] for r in rows if r[field] is not None]
        return float(np.median(v)) if v else None
    summary = {f"{f}_median_pct": (100 * med(f) if med(f) is not None else None)
               for f in ["trivial_err", "fft_err", "recovery_err", "shuffled_err"]}
    print("\nMEDIAN RELATIVE ERROR vs k*_true:")
    for kk, vv in summary.items():
        print(f"   {kk:26s} {vv:.2f}%" if vv is not None else f"   {kk:26s} n/a")
    json.dump(dict(rows=rows, summary=summary), open(a.out, "w"), indent=1)
    _runlog.write_meta(a.out, run)
    row = dict(summary); row.update(dataset=a.dataset, samples=a.samples,
                                     seeds=a.seeds, steps=a.steps)
    _runlog.record("experiments", run, row)
