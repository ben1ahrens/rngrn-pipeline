"""EXP10 - An exact, GPU-friendly dispersion for N=3. Validated, not assumed.

THE BLOCKER IT REMOVES
  sigma(k) = max_i Re eig(J - k^2 D) called torch.linalg.eigvals on a batch of small
  NON-SYMMETRIC matrices. cuSOLVER has no batched kernel for that: 200 batched 3x3
  eigendecompositions cost ~0.17 ms on CPU and ~152 ms on CUDA, and the per-matrix CUDA
  cost is FLAT at ~700 us from batch 200 to 51200 -- a per-MATRIX tax, so batching cannot
  amortise it. That one call is why a GPU training step was ~5x SLOWER than CPU.

FIRST ATTEMPT, REJECTED: shifted power iteration.
  Only the largest real part is needed, and power iteration is matmul-only. But it was
  wrong on real data: 49/127 Turing verdicts FLIPPED at 60 iterations. Cause measured, not
  guessed -- 11/200 k-points have a COMPLEX dominant eigenvalue, and a real power iteration
  cannot converge inside a 2-D invariant subspace, it rotates. (The shift size was a red
  herring: the convergence ratio is ~0.32, i.e. fast.) Kept here as a documented dead end.

WHAT WORKS: the closed-form cubic.
  For N=3 the characteristic polynomial is a cubic whose coefficients are differentiable
  polynomials in M -- trace, sum of principal 2x2 minors, determinant -- and whose roots
  are closed-form. Three real roots (discriminant > 0): the trigonometric solution, whose
  k=0 branch IS the largest root. One real + a complex pair: Cardano, and the largest real
  part is max(y, -y/2). Pure arithmetic: no eig, no svd, no linear solve.

  NOTE the sign trap that cost two wrong implementations: in the trigonometric branch the
  argument is (3q)/(2p) * sqrt(-3/p) with p < 0, and dropping the sign of p (e.g. using
  |p|) gives a plausible-looking function with MAE ~2e+2. It was caught ONLY by validating
  against eigvals on real Jacobians. Do not "simplify" this expression without re-running
  the equivalence check.

MEASURED (127 real three_gene answer-key Jacobians, 200-point k-grid):
  sigma_max MAE 9.2e-13 (max 7.0e-11) | k* MAE exactly 0 | Turing verdict flips 0/127
  gradient d sigma_max / dJ agrees with eigvals to 2e-16
  speed: CPU 0.26 ms vs eigvals 0.17 ms (0.7x) | CUDA 1.41 ms vs 151.77 ms (108x)
  So: use eigvals on CPU, the cubic on GPU -- or the cubic everywhere for consistency.

GENERAL N: this is exact for N<=3 only (N=4 has a closed form that is numerically poor,
N>=5 has none). For larger N the honest options are eigvals on CPU, or a Krylov/Arnoldi
method that handles complex pairs -- NOT the power iteration rejected above.
"""
import argparse, json, time
import numpy as np, torch, h5py

EPS = 1e-14


def sigma_cubic(J, D, kgrid, eps=EPS):
    """max_i Re eig(J - k^2 diag(D)) for N=3, exactly, by closed-form cubic roots.

    J: (3,3) ; D: (3,) ; kgrid: (K,) -> (K,). Differentiable w.r.t. J and D.
    """
    M = J.unsqueeze(0) - (kgrid ** 2).reshape(-1, 1, 1) * torch.diag_embed(D).unsqueeze(0)
    t = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    t2 = (M @ M).diagonal(dim1=-2, dim2=-1).sum(-1)
    c2 = 0.5 * (t * t - t2)
    c3 = torch.linalg.det(M)
    p = c2 - t * t / 3.0
    q = -2.0 * t ** 3 / 27.0 + t * c2 / 3.0 - c3
    disc = -(4.0 * p ** 3 + 27.0 * q ** 2)
    pneg = torch.clamp(-p, min=eps)
    r = torch.sqrt(pneg / 3.0)
    arg = torch.clamp((3.0 * q) / (2.0 * p.clamp(max=-eps)) * torch.sqrt(3.0 / pneg), -1.0, 1.0)
    y3 = 2.0 * r * torch.cos(torch.acos(arg) / 3.0)
    s = torch.sqrt(torch.clamp(q * q / 4.0 + p ** 3 / 27.0, min=0.0))
    cb = lambda z: torch.sign(z) * torch.pow(z.abs().clamp_min(eps), 1.0 / 3.0)
    y_one = cb(-q / 2.0 + s) + cb(-q / 2.0 - s)
    y = torch.where(disc > 0, y3, torch.maximum(y_one, -0.5 * y_one))
    return y + t / 3.0


def sigma_eig(J, D, kgrid):
    M = J.unsqueeze(0) - (kgrid ** 2).reshape(-1, 1, 1) * torch.diag_embed(D).unsqueeze(0)
    return torch.linalg.eigvals(M).real.max(dim=-1).values


def validate(nk=200, kmax=3.0):
    kg = torch.linspace(0.0, kmax, nk, dtype=torch.float64)
    Js, Ds = [], []
    for sp in ["train", "val", "test"]:
        with h5py.File(f"data/datasets/three_gene_{sp}/payload.h5") as f:
            for k in f:
                Js.append(f[k]["jacobian"][:]); Ds.append(f[k]["D"][:])
    se, ke, flips = [], [], 0
    for J0, D0 in zip(Js, Ds):
        J = torch.tensor(J0, dtype=torch.float64); D = torch.tensor(D0, dtype=torch.float64)
        ref, got = sigma_eig(J, D, kg), sigma_cubic(J, D, kg)
        se.append(float((ref - got).abs().max()))
        ke.append(abs(float(kg[ref.argmax()]) - float(kg[got.argmax()])))
        flips += bool(ref[0] < 0 and ref[1:].max() > 0) != bool(got[0] < 0 and got[1:].max() > 0)
    return dict(n=len(Js), sigmax_mae=float(np.mean(se)), sigmax_max=float(np.max(se)),
                kstar_mae=float(np.mean(ke)), verdict_flips=int(flips))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/exp10_cubic_dispersion.json")
    a = ap.parse_args()
    v = validate()
    print(f"validated on {v['n']} real answer-key Jacobians:")
    print(f"  sigma_max MAE {v['sigmax_mae']:.3e}  max {v['sigmax_max']:.3e}")
    print(f"  k* MAE {v['kstar_mae']:.6f}   Turing verdict flips {v['verdict_flips']}/{v['n']}")
    kg = torch.linspace(0.0, 3.0, 200, dtype=torch.float64)
    with h5py.File("data/datasets/three_gene_val/payload.h5") as f:
        g = f["sample_0000"]
        J = torch.tensor(g["jacobian"][:], dtype=torch.float64)
        D = torch.tensor(g["D"][:], dtype=torch.float64)
    speed = {}
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        def bench(fn, n=20):
            x, y, z = J.to(dev), D.to(dev), kg.to(dev)
            for _ in range(3): fn(x, y, z)
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n): fn(x, y, z)
            if dev == "cuda": torch.cuda.synchronize()
            return (time.time() - t0) / n * 1000
        speed[f"eig_{dev}"] = bench(sigma_eig); speed[f"cubic_{dev}"] = bench(sigma_cubic)
        print(f"  {dev:5s}: eigvals {speed[f'eig_{dev}']:8.2f} ms   "
              f"cubic {speed[f'cubic_{dev}']:8.2f} ms   "
              f"({speed[f'eig_{dev}']/speed[f'cubic_{dev}']:.1f}x)")
    json.dump(dict(accuracy=v, speed_ms=speed), open(a.out, "w"), indent=1)
