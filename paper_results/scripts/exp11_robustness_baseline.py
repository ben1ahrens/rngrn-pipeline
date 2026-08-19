"""EXP11 — Robustness baseline of the GENERATOR systems, and Tica's immobile node.

WHY: "more robust than Tica et al." is the project's stated end goal, and robustness
has never been measured here. Tica report a LOCAL parameter-volume figure: adding 1%
relative uncertainty around a Turing solution leaves 33% Turing; 5% leaves 5%. The
comparable quantity on our side is the fraction of a perturbation cloud around a known
Turing system that is STILL Turing. This script establishes that baseline on the
answer-key systems, so any recovered system has something to be compared against.

WHAT IT MEASURES (two things):

 1. Local Turing volume of all 127 three_gene answer keys, at four noise levels.
    Perturbation: independent log-normal multiplicative factors on every NONZERO
    Jacobian entry and on every diffusivity. Sign- and topology-preserving by
    construction (magnitudes move, signs and zeros do not). sigma = 0.048 is included
    because that is Tica's measured experimental parameter CV (4.8%).

 2. What happens when one node is made NON-DIFFUSIBLE (D_i := 0), which is what Tica's
    node C is. Tested separately for the slowest, middle and fastest diffuser.

CRITERION: "Turing" here is STRICT — max Re eig(J) < 0 (the homogeneous state is
linearly stable to ALL perturbations) AND an unstable mode exists at some k > 0. Note
this WAS stricter than eval/analysis.turing_ok, which tested tr(J) < 0 — CORRECTED
2026-08-04 (docs/DECISIONS.md D-EVID-11), so turing_ok now applies this same criterion
and the library no longer carries two definitions of "Turing". This note is kept because
it is where the discrepancy was first written down. On the
UNPERTURBED data the two agree 127/127, but under perturbation the loose criterion
overcounts by up to 73% of draws for a single sample, so the strict form is used and
the loose one is recorded alongside for comparison (frac_loose, frac_loose_only).

FIREWALL: this script reads the ANSWER KEY directly and deliberately so. It is a
scoring/characterisation tool, not part of recovery. Nothing here may be imported by
model.py, losses/, or recover.py.

Outputs: experiments/exp11_robustness_baseline.csv (per sample x sigma)
         experiments/exp11_immobile_node.csv      (per sample)
"""
import argparse, glob, hashlib, os, time
import h5py, numpy as np, pandas as pd
import _runlog

KG = np.concatenate([[0.0], np.linspace(1e-3, 4.0, 250)])
K2 = (KG ** 2)[:, None, None]
SIGMAS = (0.01, 0.048, 0.10, 0.20)


def stable_seed(*parts):
    """Deterministic seed from strings/numbers. NOT hash() -- Python salts string
    hashing per process (PYTHONHASHSEED), so hash() would give different clouds on
    every run and the CSV would not be reproducible."""
    h = hashlib.blake2b("|".join(map(repr, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % 2 ** 32


def turing_strict(J, D):
    """(is_turing, kstar, sigma_max) under the STRICT criterion."""
    if np.linalg.eigvals(J).real.max() >= 0:
        return False, np.nan, np.nan
    sig = np.linalg.eigvals(J[None, :, :] - K2 * np.diag(D)[None, :, :]).real.max(axis=1)
    i = int(np.argmax(sig[1:])) + 1
    return bool(sig[i] > 1e-9), KG[i], sig[i]


def cloud(J, D, sigma, rng, ndraw):
    """Fraction of a log-normal cloud around (J, D) that is still Turing.

    Perturbs nonzero J entries and all D entries multiplicatively. A zero diffusivity
    stays zero (an immobile node does not become mobile under parameter noise).
    """
    mask = (J != 0)
    fJ = np.ones((ndraw,) + J.shape)
    fJ[:, mask] = rng.lognormal(0.0, sigma, size=(ndraw, int(mask.sum())))
    Jp = J[None] * fJ
    Dp = D[None, :] * rng.lognormal(0.0, sigma, size=(ndraw, len(D)))
    Dp[:, D == 0] = 0.0
    strict_u = np.linalg.eigvals(Jp).real.max(axis=1) < 0
    loose_u = np.trace(Jp, axis1=1, axis2=2) < 0
    n = J.shape[0]
    Dm = np.zeros((ndraw, n, n)); Dm[:, range(n), range(n)] = Dp
    sig = np.linalg.eigvals(Jp[:, None, :, :] - K2[None] * Dm[:, None, :, :]).real.max(axis=2)
    ipos = sig[:, 1:].argmax(axis=1) + 1
    unst = sig[np.arange(ndraw), ipos] > 1e-9
    kst = KG[ipos]
    keep = strict_u & unst
    return dict(
        frac_strict=float(keep.mean()),
        frac_loose=float((loose_u & unst).mean()),
        frac_loose_only=float((loose_u & unst & ~strict_u).mean()),
        kstar_cv=float(kst[keep].std() / kst[keep].mean()) if keep.any() else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/datasets")
    ap.add_argument("--family", default="three_gene", help="prefix; 'classical' sets are skipped")
    ap.add_argument("--ndraw", type=int, default=400)
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args()
    run = _runlog.start("exp11_robustness_baseline", vars(args))

    cloud_rows, imm_rows = [], []
    t0 = time.time()
    for path in sorted(glob.glob(f"{args.datasets_root}/{args.family}_*/payload.h5")):
        name = os.path.basename(os.path.dirname(path))
        if "classical" in name:
            continue
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                J = np.array(g["jacobian"]); D = np.array(g["D"])
                slow, fast = int(np.argmin(D)), int(np.argmax(D))
                mid = [i for i in range(len(D)) if i not in (slow, fast)][0]
                base = dict(ds=name, key=key, topology=str(g.attrs.get("topology")),
                            morph=str(g.attrs.get("morphology")), L=float(g.attrs["L"]),
                            kstar_true=float(g.attrs["k_star"]), n_hill=float(g.attrs.get("n", np.nan)),
                            Dratio=float(D.max() / D.min()), slow_idx=slow)
                for s in SIGMAS:
                    row = dict(base); row["sigma"] = s
                    row.update(cloud(J, D, s, np.random.default_rng(
                        stable_seed(name, key, s)), args.ndraw))
                    cloud_rows.append(row)
                rec = dict(base)
                rec["base_rob_10"] = cloud(J, D, 0.10, np.random.default_rng(7), args.ndraw)["frac_strict"]
                for label, idx in (("slow", slow), ("mid", mid), ("fast", fast)):
                    D2 = D.copy(); D2[idx] = 0.0
                    ok, _, _ = turing_strict(J, D2)
                    rec[f"imm_{label}_ok"] = ok
                    rec[f"imm_{label}_rob"] = (
                        cloud(J, D2, 0.10, np.random.default_rng(7), args.ndraw)["frac_strict"]
                        if ok else 0.0)
                imm_rows.append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    cdf = pd.DataFrame(cloud_rows); idf = pd.DataFrame(imm_rows)
    cdf.to_csv(f"{args.out_dir}/exp11_robustness_baseline.csv", index=False)
    idf.to_csv(f"{args.out_dir}/exp11_immobile_node.csv", index=False)
    wall_s = round(time.time() - t0, 1)
    _runlog.write_meta(f"{args.out_dir}/exp11_robustness_baseline.csv", run)
    _runlog.record(args.out_dir, run, dict(family=args.family, ndraw=args.ndraw,
                   n_samples=len(idf), wall_s=wall_s))
    print(f"n samples {len(idf)} | wall {time.time() - t0:.0f}s")
    print(cdf.groupby("sigma").frac_strict.describe()[["count", "mean", "50%", "min"]].round(3).to_string())
    print("\nstill Turing with one node immobile:",
          {l: int(idf[f"imm_{l}_ok"].sum()) for l in ("slow", "mid", "fast")}, f"of {len(idf)}")


if __name__ == "__main__":
    main()
