"""EXP06 - Joint sweep: pixel batch size x residual weight.

exp05 found a clear GRADIENT in the residual weight at fixed batch 128:
  w=0.3 -> k* median 23.9% ;  w=1.0 -> 3.4% ;  no residual -> 0.4%
w=1.0 was the largest weight tried, so it was the EDGE of the sweep, not a minimum.
This sweeps both axes to find whether a minibatched residual ever BEATS omitting it,
or whether the curve simply approaches the no-residual limit from above (which would
mean the term contributes nothing here and should default off).

Reported as a SEED DISTRIBUTION (median + hit rate), never the best seed: every exp05
config found a ~0.37% solution on some seed, so best-of is uninformative.
"""
import argparse, itertools, json, time
import numpy as np, torch, h5py
from rngrn import observables as obs
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "exp05", pathlib.Path(__file__).parent / "exp05_pixel_minibatch.py")
exp05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp05)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--sample", default="sample_0000")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="experiments/exp06_residual_sweep.json")
    a = ap.parse_args()
    with h5py.File(f"data/datasets/{a.dataset}/payload.h5") as f:
        g = f[a.sample]; frame = g["final_frame"][:]
        L = float(g.attrs["L"]); kstar_true = float(g.attrs["k_star"])
    frame_t = torch.tensor(np.asarray(frame, dtype=float))
    kstar_obs = obs.kstar_of(frame[0], L=L)
    batches = [64, 128, 512]
    weights = [1.0, 3.0, 10.0]
    rows = []
    print(f"baseline (no residual) reference: exp05 gave median 0.4%, 5/10 Turing\n")
    print(f"{'batch':>6s} {'w':>6s} {'conv':>6s} {'turing':>7s} {'k* median':>11s}")
    t0 = time.time()
    for b, w in itertools.product(batches, weights):
        res = [exp05.fit(s, frame_t, L, kstar_obs, a.steps, b, w) for s in range(a.seeds)]
        ok = [r for r in res if r]; tur = [r for r in ok if r["turing"]]
        errs = [abs(r["kstar"] - kstar_true) / kstar_true for r in tur]
        med = float(np.median(errs)) if errs else None
        rows.append(dict(batch=b, w=w, converged=len(ok), turing=len(tur),
                         kstar_rel_err_median=med))
        print(f"{b:6d} {w:6.1f} {len(ok):5d}/{a.seeds} {len(tur):7d} "
              + (f"{100*med:10.1f}%" if med is not None else "        n/a"))
    json.dump(dict(rows=rows, meta=dict(kstar_true=kstar_true, steps=a.steps,
              seeds=a.seeds, wall_s=round(time.time()-t0,1))), open(a.out,"w"), indent=1)
