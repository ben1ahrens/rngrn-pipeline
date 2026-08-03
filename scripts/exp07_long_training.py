"""EXP07 - Iteration count as the axis. "Batch size introducing iterations."

THE OBSERVATION THIS TESTS
  Endres runs 200,000 Adam iterations on batches of 128 pixels. Every RNGRN experiment
  so far has run 400-1500. That is a 100-500x difference in optimisation budget, and it
  is not incidental: small batches make each step cheap AND inject gradient noise, so a
  long small-batch run is a fundamentally different optimiser (SGD-style exploration)
  from a short full-batch one (near-deterministic descent into the nearest basin).

  With the patterning region at ~0.1% of parameter space (Scholes et al.), escaping bad
  basins is the whole problem -- exactly what sustained gradient noise is for.

WHAT IS MEASURED
  Turing hit rate and k* error vs iteration budget, at fixed wall-clock-per-step, with
  and without a minibatched residual. If the hit rate climbs with iterations, the
  earlier "6/12 seeds" ceiling was a budget artefact, not a property of the method.

Parallelised across seeds over the local cores; reported as a seed DISTRIBUTION.
FIREWALL: frame, L, observed_idx only.
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch, h5py
import _runlog

torch.set_num_threads(1)   # one core per worker; parallelism is across seeds


def _work(args):
    seed, steps, batch, w_resid, path, sample = args
    import torch, numpy as np, h5py
    torch.set_num_threads(1)
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "exp05", pathlib.Path(__file__).parent / "exp05_pixel_minibatch.py")
    exp05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp05)
    from rngrn import observables as obs
    with h5py.File(path) as f:
        g = f[sample]; frame = g["final_frame"][:]; L = float(g.attrs["L"])
    ft = torch.tensor(np.asarray(frame, dtype=float))
    ko = obs.kstar_of(frame[0], L=L)
    t0 = time.time()
    r = exp05.fit(seed, ft, L, ko, steps, batch, w_resid)
    if r: r["wall_s"] = round(time.time() - t0, 1)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--sample", default="sample_0000")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--budgets", type=int, nargs="+", default=[400, 2000, 8000])
    ap.add_argument("--out", default="experiments/exp07_long_training.json")
    a = ap.parse_args()
    run = _runlog.start("exp07_long_training", vars(a))
    path = f"data/datasets/{a.dataset}/payload.h5"
    with h5py.File(path) as f:
        kstar_true = float(f[a.sample].attrs["k_star"])
    print(f"{a.sample}  k*_true={kstar_true:.4f}   seeds={a.seeds}  workers={a.workers}\n")
    print(f"{'steps':>7s} {'residual':>16s} {'conv':>7s} {'turing':>7s} {'k* median':>11s} {'wall/seed':>10s}")
    rows = []
    for steps in a.budgets:
        for label, batch, wr in [("none", None, 0.0), ("batch128 w=1", 128, 1.0)]:
            jobs = [(s, steps, batch, wr, path, a.sample) for s in range(a.seeds)]
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                res = list(ex.map(_work, jobs))
            ok = [r for r in res if r]; tur = [r for r in ok if r["turing"]]
            errs = [abs(r["kstar"] - kstar_true) / kstar_true for r in tur]
            med = float(np.median(errs)) if errs else None
            wall = float(np.median([r["wall_s"] for r in ok])) if ok else None
            rows.append(dict(steps=steps, residual=label, converged=len(ok),
                             turing=len(tur), seeds=a.seeds, kstar_rel_err_median=med,
                             wall_s_per_seed=wall))
            print(f"{steps:7d} {label:>16s} {len(ok):5d}/{a.seeds} {len(tur):7d} "
                  + (f"{100*med:10.1f}%" if med is not None else "       n/a")
                  + f" {wall if wall else 0:9.1f}s")
            json.dump(dict(rows=rows, kstar_true=kstar_true), open(a.out, "w"), indent=1)
    _runlog.write_meta(a.out, run)
    _runlog.record("experiments", run, dict(dataset=a.dataset, sample=a.sample,
                   seeds=a.seeds, n_rows=len(rows)))
