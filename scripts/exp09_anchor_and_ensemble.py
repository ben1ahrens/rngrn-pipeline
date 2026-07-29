"""EXP09 - Two questions at the 4000-step budget found by exp07.

Q1  THE BAD-ANCHOR HYPOTHESIS. In exp08 (1200 steps) two of four samples never reached
    Turing, and they were exactly the two whose FFT anchor was worst (kstar_obs 0.2202
    vs k*_true 0.3419; 0.6900 vs 0.7529). Either the anchor quality gates recovery, or
    1200 steps was simply too few. exp07 showed the hit rate plateaus at 15/16 by 4000
    steps, so rerunning those samples at 4000 separates the two explanations.

Q2  PER-SEED vs ENSEMBLE ACCURACY. exp07 and exp08 report DIFFERENT statistics and I
    should not have compared them:
      exp07  median over seeds of |k*_seed - k*_true|/k*_true   -> 3.4%   (typical seed)
      exp08  |median over seeds of k*_seed - k*_true|/k*_true   -> 0.30%  (ensemble)
    The second is smaller only if seed errors straddle the truth and cancel. If that
    holds it is a genuinely useful result -- run many seeds, take the median k*, and the
    ensemble is an order of magnitude better than any typical seed -- and it matches
    Endres treating multiple parameter alternatives as a feature rather than noise.
    Both are computed here, side by side, so the distinction is explicit.

FIREWALL: inputs are the frame, its L and observed_idx. k*_true is scoring-only.
"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch, h5py
import _runlog

torch.set_num_threads(1)


def _work(args):
    sample, seed, steps, path = args
    import torch, numpy as np, h5py, importlib.util, pathlib
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "exp05", pathlib.Path(__file__).parent / "exp05_pixel_minibatch.py")
    exp05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp05)
    from rngrn import observables as obs
    with h5py.File(path) as f:
        g = f[sample]; frame = g["final_frame"][:]; L = float(g.attrs["L"])
    ft = torch.tensor(np.asarray(frame, dtype=float))
    ko = obs.kstar_of(frame[0], L=L)
    r = exp05.fit(seed, ft, L, ko, steps, None, 0.0)
    return dict(sample=sample, seed=seed, kstar=(r or {}).get("kstar"),
                turing=bool((r or {}).get("turing")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default="experiments/exp09_anchor_ensemble.json")
    a = ap.parse_args()
    run = _runlog.start("exp09_anchor_and_ensemble", vars(a))
    path = f"data/datasets/{a.dataset}/payload.h5"
    from rngrn import observables as obs
    with h5py.File(path) as f:
        keys = list(f.keys())[:a.samples]
        meta = {}
        for k in keys:
            g = f[k]
            meta[k] = dict(L=float(g.attrs["L"]), kstar_true=float(g.attrs["k_star"]),
                           kstar_obs=float(obs.kstar_of(g["final_frame"][:][0],
                                                        L=float(g.attrs["L"]))))
    jobs = [(k, s, a.steps, path) for k in keys for s in range(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_work, jobs))
    rows = []
    print(f"{'sample':12s} {'anchor err':>11s} {'turing':>8s} {'per-seed':>9s} {'ensemble':>9s}")
    for k in keys:
        m = meta[k]
        got = [r for r in res if r["sample"] == k and r["turing"] and r["kstar"]]
        anchor_err = abs(m["kstar_obs"] - m["kstar_true"]) / m["kstar_true"]
        if got:
            ks = np.array([r["kstar"] for r in got])
            per_seed = float(np.median(np.abs(ks - m["kstar_true"]) / m["kstar_true"]))
            ensemble = float(abs(np.median(ks) - m["kstar_true"]) / m["kstar_true"])
        else:
            per_seed = ensemble = None
        rows.append(dict(sample=k, **m, anchor_rel_err=anchor_err,
                         n_turing=len(got), n_seeds=a.seeds,
                         per_seed_median_err=per_seed, ensemble_err=ensemble))
        print(f"{k:12s} {100*anchor_err:10.1f}% {len(got):5d}/{a.seeds} "
              + (f"{100*per_seed:8.1f}% {100*ensemble:8.2f}%" if per_seed is not None
                 else "      n/a       n/a"))
    ok = [r for r in rows if r["per_seed_median_err"] is not None]
    summ = dict(
        n_samples=len(rows), n_with_turing=len(ok), steps=a.steps, seeds=a.seeds,
        per_seed_median_pct=100*float(np.median([r["per_seed_median_err"] for r in ok])) if ok else None,
        ensemble_median_pct=100*float(np.median([r["ensemble_err"] for r in ok])) if ok else None,
        anchor_median_pct=100*float(np.median([r["anchor_rel_err"] for r in rows])),
        corr_anchor_vs_hitrate=float(np.corrcoef([r["anchor_rel_err"] for r in rows],
                                                 [r["n_turing"] for r in rows])[0,1]) if len(rows)>2 else None,
        wall_s=round(time.time()-t0,1))
    print(f"\nacross samples: per-seed {summ['per_seed_median_pct']:.1f}%  "
          f"ENSEMBLE {summ['ensemble_median_pct']:.2f}%  anchor {summ['anchor_median_pct']:.1f}%")
    print(f"corr(anchor error, Turing hit count) = {summ['corr_anchor_vs_hitrate']:+.3f}"
          "   (negative => bad anchor hurts)")
    json.dump(dict(rows=rows, summary=summ), open(a.out, "w"), indent=1)
    _runlog.write_meta(a.out, run)
    row = dict(summ); row.update(dataset=a.dataset, samples=a.samples)
    _runlog.record("experiments", run, row)
