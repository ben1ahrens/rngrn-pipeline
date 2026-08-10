"""gen_tg3.py — tracked port of the 3-gene Turing dataset generator (unit 11).

WHY THIS FILE EXISTS
--------------------
The generator that produced every registered three_gene_* dataset lived at
``data/staging/tg3/generator.py``. ``data/staging/`` is gitignored, so the code that
made all our data was not in version control at all. This is that generator, ported
into ``scripts/`` so it is TRACKED, plus one behavioural change described below.

THE CHANGE: THE DOMAIN SIZE NO LONGER ENCODES THE ANSWER
--------------------------------------------------------
The original set ``L = clip(6 * lambda, 18, 220)`` with ``lambda = 2*pi/k_star``, i.e.
the box always held exactly SIX periods. Measured consequence on the existing data:
``k_star`` is identically ``6*2pi/L`` for 94.8% of the 287 registered samples, so an
image-blind predictor that reads only ``L`` scores 1.4e-14 % median error on
``kstar_rel_err``. There is also no L-axis to generalise across, because every sample
is the same dimensionless system at a different physical scale.

Here the periods-per-box ``p`` is a free integer drawn from ``PERIODS_CHOICES``
(default 3..14) instead of the constant 6, and candidates whose feasible ``p`` set is
empty are rejected rather than clipped — so no sample sits on an ``L`` bound where the
relation would collapse again.

MODES
-----
``qvar``   : one simulation per screened system, ``p`` drawn per system. Decouples L
             from k*.
``multiL`` : the SAME screened system simulated at ``len(MULTIL_PERIODS)`` different
             ``p`` (hence different L). Every sample carries ``system_id``, so the
             replicates of one system can be identified. This is the direct
             L-generalisation test set.
``smoke``  : a 3-sample qvar set for pipeline tests.

The physics (``screen_model``, ``simulate_and_classify``, ``classify``) is a verbatim
port; only the L policy, the payload writer and the CLI are new.

Output is written directly in the registry layout:
``<datasets-root>/<dataset_id>/payload.h5`` with one group per sample. Run
``rngrn scan-datasets`` afterwards to write the manifest and index row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy import ndimage
from scipy.optimize import fsolve

# ---------------- models: topology x reaction kinetics ----------------
TOPOLOGIES = {
    "double_inhibitor": [[1, -1, -1], [1, 0, 0], [1, 0, 0]],
    "relay_chain":      [[1, -1, 0], [0, 0, 1], [1, 0, 0]],
    "cross_repress":    [[1, -1, -1], [1, 0, 0], [1, -1, 0]],
    "selfinhib":        [[1, -1, 0], [1, -1, 0], [1, 0, -1]],
    "activator_ring":   [[1, 0, -1], [-1, 1, 0], [0, -1, 1]],
    "substrate_depl":   [[1, -1, 0], [1, 0, -1], [0, 1, 0]],
}

# L policy. The original generator hard-coded 6 periods per box; these are the
# replacements. L_MIN/L_MAX are the original clip bounds, now used as an ACCEPTANCE
# window (candidates outside it are rejected) instead of a clamp.
PERIODS_CHOICES = tuple(range(3, 15))      # {3..14}
MULTIL_PERIODS = (4, 7, 10, 13)            # the four L of one system in multiL mode
L_MIN, L_MAX = 18.0, 220.0


def make_reaction(M, b, V, mu, K, n, form):
    M = np.asarray(M, float)
    deg = np.maximum((M != 0).sum(1), 1)

    def R(x):
        x = np.clip(x, 0, None)
        out = np.empty(3)
        for i in range(3):
            gs = []
            for j in range(3):
                if M[i, j] > 0:
                    gs.append(x[j] ** n / (K[i] ** n + x[j] ** n))
                elif M[i, j] < 0:
                    gs.append(K[i] ** n / (K[i] ** n + x[j] ** n))
            if not gs:
                reg = 1.0
            elif form == "multiplicative":
                reg = np.prod(gs)
            else:
                reg = np.sum(gs) / deg[i]
            out[i] = b[i] + V[i] * reg - mu[i] * x[i]
        return out
    return R


def steady_states(R, tries=12, seed=0):
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(tries):
        s, info, ier, _ = fsolve(lambda x: R(x), rng.uniform(0.01, 5, 3), full_output=True)
        if ier == 1 and np.all(s > 1e-6) and np.max(np.abs(R(s))) < 1e-8:
            if not any(np.allclose(s, t, atol=1e-4) for t in sols):
                sols.append(s)
    return sols


def jac_fd(R, x, eps=1e-6):
    J = np.zeros((3, 3))
    f0 = R(x)
    for j in range(3):
        dx = np.zeros(3)
        dx[j] = eps
        J[:, j] = (R(x + dx) - f0) / eps
    return J


def turing_test(J, D, kmax=3.5, nk=350):
    stable = np.max(np.real(np.linalg.eigvals(J))) < -1e-9
    ks = np.linspace(1e-3, kmax, nk)
    Dm = np.diag(D)
    sig = np.array([np.max(np.real(np.linalg.eigvals(J - (k ** 2) * Dm))) for k in ks])
    i = int(np.argmax(sig))
    return {"stable_no_diff": bool(stable),
            "is_turing": bool(stable and sig[i] > 1e-6 and ks[i] > ks[1]),
            "k_star": float(ks[i]), "sigma_max": float(sig[i])}


def feasible_periods(k_star, choices=PERIODS_CHOICES):
    """Periods-per-box values p for which L = p * (2pi/k_star) lands inside [L_MIN, L_MAX].

    Rejecting infeasible candidates (rather than clipping L to a bound, as the original
    generator did) is what keeps L a free variable: at a clip bound L is constant and
    carries no information about k*, but it also makes the pattern under- or
    over-resolved, and the original mixed both regimes into one dataset.
    """
    lam = 2 * np.pi / k_star
    return tuple(p for p in choices if L_MIN <= p * lam <= L_MAX)


def screen_model(topo, form, n_samples, seed, want, periods_choices=PERIODS_CHOICES,
                 require_all=False):
    """Sample kinetic parameters, keep numerically verified Turing-unstable systems.

    Verbatim from the staging generator except for the ``feasible_periods`` filter:
    a candidate whose wavelength admits no p in ``periods_choices`` is dropped, so
    every surviving candidate can be simulated at an unclipped L. With
    ``require_all=True`` a candidate must be feasible at EVERY p in ``periods_choices``
    — what multiL mode needs, since a partial replicate group is not a usable
    L-generalisation test.
    """
    M = TOPOLOGIES[topo]
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(n_samples):
        b = 10 ** rng.uniform(-2, -0.3, 3)
        V = 10 ** rng.uniform(-0.3, 1, 3)
        mu = 10 ** rng.uniform(-0.4, 0.7, 3)
        K = 10 ** rng.uniform(-1, 0.6, 3)
        n = float(rng.choice([2., 3., 4.]))
        Dr = 10 ** rng.uniform(0.9, 2.4, 2)
        D = np.array([1., Dr[0], Dr[1]])
        R = make_reaction(M, b, V, mu, K, n, form)
        for s in steady_states(R, tries=10, seed=int(rng.integers(1e9))):
            t = turing_test(jac_fd(R, s), D)
            if not (t["is_turing"] and 0.08 < t["k_star"] < 3.0 and t["sigma_max"] > 0.01):
                continue
            feas = feasible_periods(t["k_star"], periods_choices)
            if not feas or (require_all and len(feas) != len(periods_choices)):
                continue
            hits.append({"topology": topo, "reaction": form, "b": b.tolist(),
                         "V": V.tolist(), "mu": mu.tolist(), "K": K.tolist(),
                         "n": n, "D": D.tolist(), "x_star": s.tolist(),
                         "feasible_periods": list(feas), **t})
        if len(hits) >= want:
            break
    return hits[:want]


# ---------------- simulation (spectral IMEX) + classification ----------------
def simulate_and_classify(p, grid=96, n_traj=6, Tmax=260.0, seed=1, cv_every=None):
    """Nonlinear spectral IMEX simulation + morphology classification.

    Verbatim from the staging generator except that the domain size comes from
    ``p["periods_per_box"]`` instead of the hard-coded 6, and is NOT clipped (the
    caller guarantees feasibility via ``feasible_periods``).

    ``cv_every`` (added 2026-08-10 for the canonical datasets): if set, record the
    species-0 spatial coefficient of variation every ``cv_every`` steps into
    ``cv_trace``/``cv_times``. This is a READ-ONLY observation — it never touches ``X`` —
    so results are bit-identical with and without it. It exists so a saturation gate can
    ask whether the pattern had actually stopped changing by ``Tmax``. That question
    matters more at large grids: ``Tmax = 260`` was chosen for 96x96 boxes, and a bigger
    box holds more pattern to organise, while recovery solves ``f(x*) = 0`` and never
    integrates time — so a transient frame is silently outside the model's assumptions.
    """
    M = np.array(p["_M"], float)
    b = np.array(p["b"]); V = np.array(p["V"]); mu = np.array(p["mu"])
    K = np.array(p["K"]); n = p["n"]; D = np.array(p["D"])
    xs = np.array(p["x_star"]); form = p["reaction"]
    deg = np.maximum((M != 0).sum(1), 1)
    ppb = int(p["periods_per_box"])
    lam = 2 * np.pi / p["k_star"]
    L = float(ppb * lam)
    if not (L_MIN <= L <= L_MAX):
        raise ValueError(f"periods_per_box={ppb} gives L={L:.2f} outside [{L_MIN}, {L_MAX}]; "
                         "screen_model should have rejected this candidate")
    dx = L / grid
    dt = 0.02 / max(mu.max(), 1.0)
    nsteps = int(Tmax / dt)
    kx = 2 * np.pi * np.fft.fftfreq(grid, d=dx)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    denom = [1.0 / (1.0 + dt * D[i] * k2) for i in range(3)]
    rng = np.random.default_rng(seed)
    X = xs[:, None, None] * np.ones((3, grid, grid)) \
        + 0.01 * xs[:, None, None] * rng.standard_normal((3, grid, grid))

    def react(X):
        Xc = np.clip(X, 0, None)
        out = np.empty_like(Xc)
        for i in range(3):
            gs = []
            for j in range(3):
                if M[i, j] > 0:
                    gs.append(Xc[j] ** n / (K[i] ** n + Xc[j] ** n))
                elif M[i, j] < 0:
                    gs.append(K[i] ** n / (K[i] ** n + Xc[j] ** n))
            reg = (np.prod(gs, 0) if form == "multiplicative"
                   else np.sum(gs, 0) / deg[i]) if gs else np.ones((grid, grid))
            out[i] = b[i] + V[i] * reg - mu[i] * Xc[i]
        return out

    save_at = set(np.linspace(int(nsteps * 0.5), nsteps - 1, n_traj).astype(int))
    frames = []
    times = []
    cv_trace = []
    cv_times = []
    for s in range(nsteps):
        Xr = X + dt * react(X)
        for i in range(3):
            X[i] = np.real(np.fft.ifft2(np.fft.fft2(Xr[i]) * denom[i]))
        np.clip(X, 0, None, out=X)
        if cv_every and (s % cv_every == 0 or s == nsteps - 1):
            cv_trace.append(float(X[0].std() / max(X[0].mean(), 1e-9)))
            cv_times.append(s * dt)
        if s in save_at:
            frames.append(X.astype(np.float32).copy())
            times.append(s * dt)
        if not np.all(np.isfinite(X)):
            return None
    final = X.astype(np.float32)
    cv0 = float(final[0].std() / max(final[0].mean(), 1e-9))
    if cv0 < 0.05:
        return None                     # collapsed to homogeneous
    cls = classify(final[0], L)
    out = {"final": final, "traj": np.array(frames), "times": np.array(times),
           "L": L, "dx": dx, "dt_sim": dt, "grid": grid, "cv0": cv0, **cls, "params": p}
    if cv_every:
        out["cv_trace"] = np.array(cv_trace, dtype=np.float64)
        out["cv_times"] = np.array(cv_times, dtype=np.float64)
    return out


def raps_dominant_k(field, L):
    n = field.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    F = np.fft.fftshift(np.fft.fft2((field - field.mean()) * win))
    P = np.abs(F) ** 2
    ki = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / n))
    KX, KY = np.meshgrid(ki, ki)
    Kp = np.sqrt(KX ** 2 + KY ** 2)
    kb = np.arange(0, n // 2 + 1)
    raps = np.zeros(len(kb) - 1)
    for i in range(len(kb) - 1):
        m = (Kp >= kb[i]) & (Kp < kb[i + 1])
        if m.sum():
            raps[i] = P[m].mean()
    kc = 0.5 * (kb[:-1] + kb[1:]) * (2 * np.pi / L)
    if raps[1:].max() <= 0:
        return 0.0
    return float(kc[1 + int(np.argmax(raps[1:]))])


def anisotropy(field):
    n = field.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    F = np.fft.fftshift(np.fft.fft2((field - field.mean()) * win))
    P = np.abs(F) ** 2
    ki = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / n))
    KX, KY = np.meshgrid(ki, ki)
    Kp = np.sqrt(KX ** 2 + KY ** 2)
    kd = Kp[np.unravel_index(np.argmax(P * (Kp > 0)), P.shape)]
    ring = (Kp > 0.6 * kd) & (Kp < 1.4 * kd)
    w = P[ring]
    th = np.arctan2(KY[ring], KX[ring])
    if w.sum() <= 0:
        return 0.0
    C = np.sum(w * np.cos(2 * th)) / w.sum()
    S = np.sum(w * np.sin(2 * th)) / w.sum()
    return float(np.sqrt(C ** 2 + S ** 2))


def classify(field, L):
    z = (field - field.mean()) / (field.std() + 1e-12)
    k = raps_dominant_k(field, L)
    wav = 2 * np.pi / k if k > 0 else np.inf
    hi = z > 0.4
    phi = float(hi.mean())
    lbl, _ = ndimage.label(hi)
    ncomp = int(lbl.max())
    A = anisotropy(field)
    if phi < 0.34:
        morph = "spots"
    elif phi > 0.66:
        morph = "holes"
    else:
        morph = "stripes" if A > 0.55 else "labyrinth"
    return {"morphology": morph, "wavelength": round(float(wav), 2),
            "k_star_fft": round(k, 3), "area_frac": round(phi, 3),
            "n_components": ncomp, "anisotropy": round(A, 3)}


def _worker(p):
    return simulate_and_classify(p, seed=int(p["sim_seed"]))


# ---------------- candidate construction ----------------
def stable_seed(text):
    """Process-independent integer seed from a string.

    The staging generator used ``abs(hash(topo))``. Python salts string hashing per
    process (PYTHONHASHSEED), so the recorded seed did NOT reproduce the screen — two
    runs of the original generator at the same seed drew different parameters. This is
    a real auditability defect in how the existing three_gene_* data was made; it is
    fixed here rather than reproduced.
    """
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def screen_candidates(topologies, form, want_per_topology, seed, n_screen,
                      periods_choices=PERIODS_CHOICES, require_all=False):
    cand = []
    for topo in topologies:
        h = screen_model(topo, form, n_samples=n_screen,
                         seed=(seed * 7919 + stable_seed(topo) % 99999) % (2 ** 31),
                         want=want_per_topology, periods_choices=periods_choices,
                         require_all=require_all)
        for x in h:
            x["_M"] = TOPOLOGIES[topo]
        cand += h
        print(f"screen {topo:16s}/{form:14s}: {len(h)} candidates", flush=True)
    return cand


def expand_qvar(cand, seed):
    """One simulation per system; periods-per-box drawn uniformly from its feasible set."""
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(cand):
        c = dict(c)
        c["system_id"] = i
        c["replicate"] = 0
        c["periods_per_box"] = int(rng.choice(c["feasible_periods"]))
        c["sim_seed"] = int(rng.integers(1, 2 ** 31))
        out.append(c)
    return out


def expand_multiL(cand, seed, periods=MULTIL_PERIODS):
    """The SAME system at every p in `periods` — but only systems feasible at ALL of them.

    Requiring full feasibility (rather than dropping the infeasible L of a system) keeps
    every system_id a complete, balanced replicate group: an incomplete group would make
    "the same network from each L" an unequal comparison across systems.
    """
    rng = np.random.default_rng(seed)
    out = []
    sid = 0
    for c in cand:
        feas = set(c["feasible_periods"])
        if not set(periods) <= feas:
            continue
        sim_seed = int(rng.integers(1, 2 ** 31))
        for r, p in enumerate(periods):
            d = dict(c)
            d["system_id"] = sid
            d["replicate"] = r
            d["periods_per_box"] = int(p)
            # SAME initial-condition seed across the group: the four samples differ ONLY
            # in domain size, so any recovery difference is attributable to L alone.
            d["sim_seed"] = sim_seed
            out.append(d)
        sid += 1
    return out


# ---------------- payload writing ----------------
PARAM_KEYS = ("b", "V", "mu", "K", "n", "D")


def write_payload(sims, out_path):
    """Write the registry-layout payload. Matches the three_gene_* HDF5 contract.

    Adds two things the existing three_gene family does NOT store:
      * ``params_json`` — the generating kinetics (b, V, mu, K, n, D, topology,
        reaction, interaction matrix). Without it the kinetics of a sample are
        unrecoverable from the payload, which is what happened to the 127 existing
        three_gene samples.
      * ``system_id`` / ``replicate`` / ``periods_per_box`` — so replicates of one
        system at different L can be identified.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for idx, s in enumerate(sims):
            p = s["params"]
            g = f.create_group(f"sample_{idx:04d}")
            g.create_dataset("final_frame", data=s["final"], compression="gzip")
            g.create_dataset("trajectory", data=s["traj"], compression="gzip")
            g.create_dataset("times", data=s["times"])
            g.create_dataset("x_star", data=np.array(p["x_star"]))
            g.create_dataset("D", data=np.array(p["D"]))
            g.create_dataset("interaction_matrix", data=np.array(p["_M"]))
            J = jac_fd(make_reaction(p["_M"], np.array(p["b"]), np.array(p["V"]),
                                     np.array(p["mu"]), np.array(p["K"]), p["n"],
                                     p["reaction"]), np.array(p["x_star"]))
            g.create_dataset("jacobian", data=J)
            for kk in ("topology", "reaction", "n", "k_star", "sigma_max"):
                g.attrs[kk] = p[kk]
            for kk in ("L", "dx", "grid", "dt_sim", "morphology", "wavelength",
                       "k_star_fft", "area_frac", "n_components", "anisotropy", "cv0"):
                g.attrs[kk] = s[kk]
            g.attrs["system_id"] = int(p["system_id"])
            g.attrs["replicate"] = int(p["replicate"])
            g.attrs["periods_per_box"] = int(p["periods_per_box"])
            g.attrs["sim_seed"] = int(p["sim_seed"])
            g.attrs["params_json"] = json.dumps(
                {**{k: p[k] for k in PARAM_KEYS},
                 "topology": p["topology"], "reaction": p["reaction"],
                 "interaction_matrix": p["_M"]})
    return out_path


def summarise(sims):
    from collections import Counter
    L = np.array([s["L"] for s in sims])
    k = np.array([s["params"]["k_star"] for s in sims])
    ppb = np.array([s["params"]["periods_per_box"] for s in sims])
    trivial = 6 * 2 * np.pi / L                       # the image-blind leak predictor
    rel = np.abs(trivial - k) / np.abs(k)
    return {"n_samples": len(sims),
            "L_range": [float(L.min()), float(L.max())],
            "k_star_range": [float(k.min()), float(k.max())],
            "periods_per_box": dict(Counter(int(v) for v in ppb)),
            "n_systems": len({s["params"]["system_id"] for s in sims}),
            "morphology": dict(Counter(s["morphology"] for s in sims)),
            "trivial_predictor_6_2pi_over_L_median_rel_err_pct": float(np.median(rel) * 100),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}


def audit_leak(datasets_root, dataset_id):
    """Measure how well an IMAGE-BLIND predictor using only L recovers k*.

    Two predictors are reported and they answer different questions:

    * ``fixed6``  — k_hat = 6*2pi/L. This is the specific leak the original generator
      created (it hard-coded six periods per box). It is the headline number and it is
      the one comparable across datasets.
    * ``bestp``   — k_hat = p*2pi/L with the single integer p that MINIMISES the median
      error on this dataset, chosen with the answers in hand. This is an ORACLE control,
      not a predictor anyone could fit blind; it is reported so that a dataset cannot
      look decoupled merely because the constant moved from 6 to something else.

    Neither number is a model score. They are properties of the DATA.
    """
    from rngrn.data import registry as reg
    with h5py.File(reg.payload_path(datasets_root, dataset_id), "r") as f:
        keys = sorted(f.keys())
        L = np.array([float(f[k].attrs["L"]) for k in keys])
        kstar = np.array([float(f[k].attrs["k_star"]) for k in keys])
    ppb = kstar * L / (2 * np.pi)
    rel6 = np.abs(6 * 2 * np.pi / L - kstar) / np.abs(kstar)
    cands = {p: float(np.median(np.abs(p * 2 * np.pi / L - kstar) / np.abs(kstar)))
             for p in range(1, 31)}
    best_p = min(cands, key=cands.get)
    return {"dataset_id": dataset_id, "n_samples": len(keys),
            "L_range": [float(L.min()), float(L.max())],
            "periods_per_box_range": [float(ppb.min()), float(ppb.max())],
            "periods_per_box_unique_rounded": sorted({round(float(v), 3) for v in ppb})[:20],
            "fixed6_median_rel_err_pct": float(np.median(rel6) * 100),
            "fixed6_frac_within_1pct": float((rel6 < 0.01).mean()),
            "oracle_best_p": int(best_p),
            "oracle_best_p_median_rel_err_pct": float(cands[best_p] * 100)}


MODE_DEFAULTS = {
    #        want_per_topology, n_screen, periods
    "qvar":   dict(want=8, n_screen=4000),
    "multiL": dict(want=4, n_screen=6000),
    "smoke":  dict(want=1, n_screen=1500),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("qvar", "multiL", "smoke", "audit"), required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--datasets-root", default="data/datasets")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--want-per-topology", type=int, default=None)
    ap.add_argument("--n-screen", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--reaction", default="multiplicative", choices=("multiplicative", "additive"))
    args = ap.parse_args(argv)

    if args.mode == "audit":
        out = audit_leak(args.datasets_root, args.dataset_id)
        print(json.dumps(out, indent=2))
        return out

    d = MODE_DEFAULTS[args.mode]
    want = args.want_per_topology if args.want_per_topology is not None else d["want"]
    n_screen = args.n_screen if args.n_screen is not None else d["n_screen"]

    t0 = time.time()
    topos = list(TOPOLOGIES) if args.mode != "smoke" else ["double_inhibitor", "relay_chain", "cross_repress"]
    if args.mode == "multiL":
        cand = screen_candidates(topos, args.reaction, want, args.seed, n_screen,
                                 periods_choices=MULTIL_PERIODS, require_all=True)
    else:
        cand = screen_candidates(topos, args.reaction, want, args.seed, n_screen)
    print(f"total candidates: {len(cand)} | screen wall {time.time()-t0:.0f}s", flush=True)
    if not cand:
        raise RuntimeError("screening found no Turing candidate — refusing to write an "
                           "empty dataset (raise --n-screen or widen the k* band)")

    if args.mode == "multiL":
        jobs = expand_multiL(cand, args.seed)
        if not jobs:
            raise RuntimeError("no screened system is feasible at all of "
                               f"{MULTIL_PERIODS} periods — nothing to simulate")
    else:
        jobs = expand_qvar(cand, args.seed)
    if args.max_samples is not None:
        jobs = jobs[:args.max_samples]
    print(f"simulating {len(jobs)} samples on {args.procs} procs", flush=True)

    t1 = time.time()
    with Pool(args.procs) as pool:
        sims = [r for r in pool.map(_worker, jobs) if r is not None]
    print(f"simulated -> {len(sims)} real patterns | sim wall {time.time()-t1:.0f}s", flush=True)
    if args.mode == "multiL":
        # A replicate group is only usable as an L-generalisation test if ALL of its L
        # survived. Simulations do fail (collapse to homogeneous / diverge), so drop any
        # partial group rather than shipping an unbalanced comparison.
        from collections import Counter
        n_per = Counter(s["params"]["system_id"] for s in sims)
        keep = {sid for sid, c in n_per.items() if c == len(MULTIL_PERIODS)}
        dropped = len(sims) - sum(n_per[s] for s in keep)
        sims = [s for s in sims if s["params"]["system_id"] in keep]
        print(f"multiL: kept {len(keep)} complete groups, dropped {dropped} orphan samples",
              flush=True)
    if not sims:
        raise RuntimeError("every simulation collapsed or diverged — nothing to write")

    out = os.path.join(args.datasets_root, args.dataset_id, "payload.h5")
    write_payload(sims, out)
    summary = summarise(sims)
    print("SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print(f"TOTAL WALL {time.time()-t0:.0f}s -> {out}", flush=True)
    return summary


if __name__ == "__main__":
    main()
