"""c2_jacobian_probe.py — is nc1's failure in J, or in D?

Every nc1 baseline run comes back with sigma(k) monotone decreasing and everywhere
negative (BIO_VIABILITY sec 4.3 point 3 saw the same on competitive's three failing
targets). Two very different defects produce that:

  (a) the recovered REACTION Jacobian J admits no Turing instability at ANY diffusion
      matrix -- then D tuning, init tuning and more steps are all irrelevant, and the fix
      has to change which J recovery lands on;
  (b) J is Turing-capable but the recovered D is in the wrong place -- then the init, the
      D-prior and the step budget are exactly the right knobs.

This separates them. J is read from the stored run (it is recorded in
results/train_results.json), held FIXED, and D is scanned over a log grid. If any D on
the grid gives max_k>0 sigma(k) > 0 while max Re eig(J) < 0, the J is Turing-capable.

The strict test of docs/DECISIONS.md D-BIO-3 is used throughout: uniform stability by
`max Re eig(J) < 0`, never `tr(J) < 0` -- Stage 0 measured the trace test overcounting by
64x on 80,000 draws.

Usage:  python scripts/c2_jacobian_probe.py experiments/<root> [n_draws]
"""
import json
import sys
from pathlib import Path

import numpy as np

KGRID = np.concatenate([[0.0], np.logspace(-3, 3.5, 400)])


def _draw_D(scheme, N, rng):
    if scheme == "free":
        return 10.0 ** rng.uniform(-4.0, 3.0, size=N)
    # 'box': the two MOST MOBILE species carry a SCORED d_ratio inside bio_box's [1, 60]
    # row, and the third is free to be near-immobile below them. That is exactly
    # scoring/plausibility.py::d_ratio_of's construction and Stage 0's `immobile` scheme
    # (BIO_VIABILITY sec 1.2b), so every draw here scores plausibility d_ratio in-box.
    d_mid = 10.0 ** rng.uniform(-2.0, 2.0)
    d_hi = d_mid * 10.0 ** rng.uniform(0.0, np.log10(60.0))
    d_lo = d_mid * 10.0 ** rng.uniform(-5.0, 0.0)
    d = np.array([d_lo, d_mid, d_hi])
    rng.shuffle(d)
    return d


def turing_capable(J, n_draws, rng, scheme="free"):
    """Scan D with J fixed. Returns (n_turing, best_sigma, bestD)."""
    if np.max(np.real(np.linalg.eigvals(J))) >= 0:
        return None  # uniform state already unstable: not a Turing question
    best, bestD, n = -np.inf, None, 0
    for _ in range(n_draws):
        d = _draw_D(scheme, J.shape[0], rng)
        # sigma(k) = max Re eig(J - k^2 diag(d))
        sig = np.array([np.max(np.real(np.linalg.eigvals(J - (k ** 2) * np.diag(d))))
                        for k in KGRID])
        s = sig[1:].max()
        if s > 1e-9:
            n += 1
        if s > best:
            best, bestD = s, d
    return n, best, bestD


def main(root, n_draws=400):
    smap = {}
    p = Path(root) / "runs.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                smap[r["run_id"]] = r["sample_key"]
    print(f"{'run':<26} {'target':<12} {'maxReEig':>9} {'sig_obs':>9} "
          f"{'free nT':>8} {'freeSig':>8} {'box nT':>7} {'boxSig':>8} "
          f"{'boxDlo/mid':>10} {'boxScored':>9}")
    for rd in sorted(Path(root).glob("runs/*")):
        f = rd / "results" / "train_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        J = np.array(d["recovered"]["J"], dtype=float)
        m = d.get("metric", {})
        ev = float(np.max(np.real(np.linalg.eigvals(J))))
        name = rd.name[-14:]
        tgt = smap.get(rd.name, "?")
        obs = m.get("recovered_sig_max", float("nan"))
        outs = []
        for scheme in ("free", "box"):
            outs.append(turing_capable(J, n_draws, np.random.default_rng(0), scheme))
        if outs[0] is None:
            print(f"{name:<26} {tgt:<12} {ev:>9.4f} {obs:>9.4f} {'UNIF-UNSTABLE':>8}")
            continue
        (nf, bf, _), (nb, bb, bD) = outs
        s = np.sort(bD)
        print(f"{name:<26} {tgt:<12} {ev:>9.4f} {obs:>9.4f} "
              f"{f'{nf}/{n_draws}':>8} {bf:>8.4f} {f'{nb}/{n_draws}':>7} {bb:>8.4f} "
              f"{s[0] / s[1]:>10.4f} {s[2] / s[1]:>9.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
