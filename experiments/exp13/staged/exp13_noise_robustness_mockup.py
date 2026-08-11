"""EXP13 — Weight-noise robustness: EXEMPLAR / SURROGATE deck for supervisor discussion.

    *** THIS SCRIPT DOES NOT TRAIN AN RNGRN. ***

WHY IT EXISTS. "Training becomes more robust by adding noise to the weights" is a stated
research goal that has NEVER been run in this repo. This script produces the figures that
a POSITIVE result would look like, so the claim, the controls and the failure modes can be
argued over before the training code is written. Every panel is either (i) real measured
data from exp11, or (ii) a surrogate computed on the real qvar answer-key Jacobians and
labelled ILLUSTRATIVE on the figure face. Nothing here is evidence about the RNGRN.

WHAT THE SURROGATE IS, precisely. Weight noise during training makes the optimiser descend
the NOISE-SMOOTHED objective

    L~_s(theta) = E_{eps ~ N(0, s^2 I)} [ L(theta + eps) ],

which penalises sharp minima. The antithetic evolution-strategies estimator IS the gradient
of that smoothed objective,

    grad L~_s = (1 / (2 m s^2)) * sum_j eps_j [ L(theta + eps_j) - L(theta - eps_j) ],

so running it at probe scale s = sigma_train reproduces the SELECTION EFFECT of weight noise
without needing the torch recovery loop. sigma_train = 0 is realised at a tiny probe scale
(1e-3), i.e. an ordinary finite-difference gradient of the UNsmoothed objective: the
deterministic control arm.

WHY THE SELECTION EFFECT IS THE WHOLE POINT. The objective's zero set is a MANIFOLD, not a
point: 3 constraints (k*-anchor, uniform stability, instability) on 10 free parameters
(nonzero |J| entries + D, signs frozen). Both arms reach loss ~ 0. They differ only in WHICH
point of that manifold they select — and local Turing volume is exactly a measure of how far
from that point the system stays Turing. So "wide minimum" and "robust parameterisation" are
the same geometric object here, which is why the mechanism is worth showing.

PARAMETERISATION. Free: log|J| on nonzero entries (signs and structural zeros FROZEN) and
log D. Perturbing in log space keeps the multiplicative noise sign-symmetric, avoiding the
raw-theta defects recorded in docs/ROBUSTNESS_MEASUREMENT.md §3.1-3.2.

OBJECTIVE. Split-support Turing hinges (docs: losses/terms.turing_hinges_split) + the
k*-anchor against k_star_fft (the IMAGE observable, per the 2026-07-29 validation-target
decision), on a per-sample k-grid 0..4*k_obs like recover._kgrid_for. Firewall note: the
anchor target is an image observable; the answer key is read only for SCORING, as in exp11.

CRITERION. exp11's STRICT criterion throughout: max Re eig(J) < 0 AND an unstable mode at
some k > 0. Leading-mode Turing-I vs Turing-Hopf is classified separately (via eigvals for
the imaginary part), the distinction Tica quantify.

DISPERSION. Exact closed-form cubic for N=3 (validated here against np.linalg.eigvals on the
34 real qvar Jacobians: MAE 6.8e-13, 0/34 k* flips, 0/34 verdict flips; 2.5e-9 on a
perturbed cloud where the complex-pair branch fires). SIGN TRAP, per mem/exp10: the
trigonometric argument is (3q)/(2p)*sqrt(-3/p) with p < 0 -- using |p| gives a
plausible-looking function with MAE ~ 2e+2.

DATASET. three_gene_qvar (34 systems, periods-per-box ~ U{3..14}) -- the PRIMARY set per
docs/PREREGISTRATION.md §1. The legacy three_gene_* splits are NOT used: their L is derived
from k* so an image-blind predictor scores 0.0% median k* error, and k* on legacy data
measures that leak rather than the model.

DETERMINISM. Seeds via blake2b over repr-joined parts, never hash() (Python salts string
hashing per process, so hash() would make the CSVs irreproducible across runs).

OUTPUTS
  experiments/exp13_noise_sweep_qvar.csv    per system x sigma_train x seed
  experiments/exp13_noise_eval_curves.csv   per system x sigma_train x seed x sigma_eval
  experiments/exp13_controls.csv            post-hoc-jitter and wide-init control arms
  experiments/exp13/figures/fig{1..6}_*.png  the deck (experiments/ is gitignored)

Usage:
    .venv/bin/python scripts/exp13_noise_robustness_mockup.py --ndraw 400 --seeds 12
"""

from __future__ import annotations
import argparse, hashlib, os, sys
import h5py
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- objective / evaluation constants -------------------------------------------------
NK_L        = 81            # loss-side k-grid points (per sample, 0 .. 4*k_obs)
SOFT_T      = 60.0          # soft-argmax temperature for k*
TAU_INSTAB  = 0.02          # required growth rate at the unstable mode
W_ANCHOR, W_UNIFORM, W_INSTAB = 1.0, 4.0, 4.0
N_PAIR, N_STEPS, LR, PROBE0   = 10, 200, 0.05, 1e-3
SCRAMBLE    = 1.0           # init spread around truth, in log space
SIGMA_TRAIN = (0.0, 0.05, 0.10, 0.20, 0.35)
SIGMA_EVAL  = (0.01, 0.048, 0.10, 0.20)          # 4.8% = Tica's measured experimental CV
SIG_EVAL_FINE = np.array([0.005,0.01,0.02,0.03,0.048,0.07,0.10,0.14,0.20,0.28])
KG_E = np.concatenate([[0.0], np.linspace(1e-3, 4.0, 250)])   # exp11's evaluation k-grid
K2_E = KG_E ** 2


def stable_seed(*parts):
    """Deterministic seed. NOT hash() -- Python salts string hashing per process."""
    h = hashlib.blake2b("|".join(map(repr, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % 2 ** 32


def load_family(ds, repo=REPO):
    """Answer keys for one registered dataset. SCORING-side read (see firewall note)."""
    out = []
    with h5py.File(os.path.join(repo, "data", "datasets", ds, "payload.h5"), "r") as f:
        for k in sorted(f.keys()):
            g = f[k]
            out.append(dict(ds=ds, key=k,
                            J=g["jacobian"][:].astype(float), D=g["D"][:].astype(float),
                            L=float(g.attrs["L"]), kstar=float(g.attrs["k_star"]),
                            kstar_fft=float(g.attrs["k_star_fft"]),
                            sigma_max=float(g.attrs["sigma_max"]),
                            morph=str(g.attrs["morphology"]), topo=str(g.attrs["topology"])))
    return out


# ======================================================================================
# validation -- the equivalence check that must pass before anything depends on the cubic
# ======================================================================================
def validate_cubic(systems, verbose=True):
    """Closed-form cubic vs np.linalg.eigvals on REAL Jacobians. Never trust the algebra
    without this: two earlier implementations of this formula were wrong (sign trap) and
    only the equivalence check caught them."""
    errs, kflip, vflip = [], 0, 0
    for s in systems:
        M = s["J"][None] - K2_E[:, None, None] * np.diag(s["D"])[None]
        ref = np.linalg.eigvals(M).real.max(axis=1)
        got = max_re_eig_cubic(M)
        errs.append(np.abs(ref - got).max())
        i_r = int(np.argmax(ref[1:])) + 1
        i_g = int(np.argmax(got[1:])) + 1
        kflip += (i_r != i_g)
        t_r = (np.linalg.eigvals(s["J"]).real.max() < 0) and (ref[i_r] > 1e-9)
        t_g = (max_re_eig_cubic(s["J"][None])[0] < 0) and (got[i_g] > 1e-9)
        vflip += (t_r != t_g)
    errs = np.array(errs)
    if verbose:
        print(f"[validate] n={len(systems)}  sigma_max MAE={errs.mean():.3e} max={errs.max():.3e}"
              f"  k*-flips={kflip}  verdict-flips={vflip}")
    assert errs.max() < 1e-8, "cubic disagrees with eigvals -- refusing to continue"
    assert kflip == 0 and vflip == 0, "cubic flips a k* or a Turing verdict"
    return dict(mae=float(errs.mean()), max=float(errs.max()), kflip=int(kflip), vflip=int(vflip))


def max_re_eig_cubic(M):
    """M: (...,3,3) real. Returns max Re eigenvalue, closed form. No eig/svd/solve."""
    a = M[..., 0, 0]; b = M[..., 0, 1]; c = M[..., 0, 2]
    d = M[..., 1, 0]; e = M[..., 1, 1]; f = M[..., 1, 2]
    g = M[..., 2, 0]; h = M[..., 2, 1]; i = M[..., 2, 2]
    tr = a + e + i
    m2 = (e*i - f*h) + (a*i - c*g) + (a*e - b*d)          # sum principal 2x2 minors
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    c2, c1, c0 = -tr, m2, -det
    s = c2 / 3.0
    p = c1 - c2*c2/3.0
    q = c0 - c1*c2/3.0 + 2.0*c2**3/27.0
    disc = -4.0*p**3 - 27.0*q**2          # >0 : three distinct real roots
    out = np.empty(np.shape(tr), dtype=float)
    three_real = (disc > 0) & (p < 0)
    # --- three real roots: trigonometric form. SIGN TRAP (mem_33180b663ebe):
    # the argument is (3q)/(2p)*sqrt(-3/p) with p<0 -- NOT |p|.
    if np.any(three_real):
        pr = np.where(three_real, p, -1.0)
        qr = np.where(three_real, q, 0.0)
        arg = (3.0*qr)/(2.0*pr) * np.sqrt(-3.0/pr)
        phi = np.arccos(np.clip(arg, -1.0, 1.0))
        # largest real root is the k=0 branch of 2*sqrt(-p/3)*cos(phi/3 - 2*pi*k/3)
        t = 2.0*np.sqrt(-pr/3.0)*np.cos(phi/3.0)
        out = np.where(three_real, t - s, out)
    # --- one real root (+ complex pair): Cardano; max Re = max(real root, -real/2)
    other = ~three_real
    if np.any(other):
        po = np.where(other, p, 1.0); qo = np.where(other, q, 0.0)
        R = np.sqrt(np.maximum(qo*qo/4.0 + po**3/27.0, 0.0))
        u = np.cbrt(-qo/2.0 + R); v = np.cbrt(-qo/2.0 - R)
        t1 = u + v                       # the real root
        re_pair = -0.5*t1                # real part of the complex pair
        out = np.where(other, np.maximum(t1, re_pair) - s, out)
    return out


def sigma_curve(logJ, signJ, logD, kg2):
    """sigma(k) for a BATCH of parameter vectors. logJ:(B,3,3) magnitudes (log), signJ:(3,3)."""
    J = signJ[None] * np.exp(logJ) * (signJ[None] != 0)
    D = np.exp(logD)                                            # (B,3)
    B = J.shape[0]; nk = len(kg2)
    M = np.broadcast_to(J[:, None], (B, nk, 3, 3)).copy()
    idx = np.arange(3)
    M[:, :, idx, idx] -= kg2[None, :, None]*D[:, None, :]
    return max_re_eig_cubic(M)                                   # (B, nk)


def kgrid_for(kobs):
    """Per-sample loss k-grid, matching recover._kgrid_for's shape: 0 .. 4*k_obs."""
    kg = np.linspace(0.0, 4.0*kobs, NK_L)
    return kg, kg**2, max(1, int(0.1*NK_L))     # k_min = 0.4*k_obs, safely below target


def loss_from_sigma(sig, kstar_obs, kg, i_min):
    s0 = sig[:, 0]; sp = sig[:, i_min:]
    s_pos = sp.max(axis=1)
    w = np.exp(SOFT_T*(sp - s_pos[:, None])/max(s_pos.std(), 1e-3)) if False else np.exp(SOFT_T*(sp - s_pos[:, None]))
    w /= w.sum(axis=1, keepdims=True)
    kstar = (w*kg[i_min:][None]).sum(axis=1)
    L = (W_ANCHOR*np.log(np.maximum(kstar, 1e-9)/kstar_obs)**2
         + W_UNIFORM*np.maximum(s0, 0.0)**2
         + W_INSTAB*np.maximum(TAU_INSTAB - s_pos, 0.0)**2)
    return L, kstar, s0, s_pos


def pack(sys):
    """Free parameters: log|J| on NONZERO entries (signs frozen) + log D."""
    J, D = sys["J"], sys["D"]
    mask = (J != 0)
    return mask, np.sign(J), np.log(np.abs(J[mask])), np.log(D)


def unpack(theta, mask, nnz):
    """theta:(B,nnz+3) -> logJ:(B,3,3), logD:(B,3)"""
    B = theta.shape[0]
    logJ = np.full((B, 3, 3), -50.0)
    logJ[:, mask] = theta[:, :nnz]
    return logJ, theta[:, nnz:]


def loss_of_theta(theta, mask, signJ, nnz, kobs, kg, k2, i_min):
    logJ, logD = unpack(theta, mask, nnz)
    sig = sigma_curve(logJ, signJ, logD, k2)
    return loss_from_sigma(sig, kobs, kg, i_min)[0]


def theta_to_JD(theta, mask, signJ, nnz):
    logJ, logD = unpack(theta, mask, nnz)
    J = signJ[None]*np.exp(logJ)*(signJ[None] != 0)
    return J, np.exp(logD)


def train_seeds(sys, sigma_train, n_seeds, seed0, init_scramble=0.6, n_steps=N_STEPS):
    mask, signJ, lJ0, lD0 = pack(sys)
    nnz = int(mask.sum()); dim = nnz+3
    kobs = sys["kstar_fft"]; kg, k2, i_min = kgrid_for(kobs)
    rng = np.random.default_rng(seed0)
    theta = np.concatenate([lJ0, lD0])[None] + rng.normal(0, init_scramble, (n_seeds, dim))
    theta_init = theta.copy()
    s_probe = sigma_train if sigma_train > 0 else PROBE0
    m1 = np.zeros_like(theta); m2 = np.zeros_like(theta)
    for t in range(1, n_steps+1):
        eps = rng.normal(0, s_probe, (n_seeds, N_PAIR, dim))
        cand = np.concatenate([theta[:, None]+eps, theta[:, None]-eps], axis=1)
        Lc = loss_of_theta(cand.reshape(-1, dim), mask, signJ, nnz, kobs, kg, k2, i_min).reshape(n_seeds, 2*N_PAIR)
        dL = Lc[:, :N_PAIR] - Lc[:, N_PAIR:]
        g = (eps*dL[:, :, None]).mean(axis=1)/(2.0*s_probe**2)
        m1 = 0.9*m1 + 0.1*g; m2 = 0.999*m2 + 0.001*g*g
        lr = LR*(1.0 - 0.9*t/n_steps)
        theta -= lr*(m1/(1-0.9**t))/(np.sqrt(m2/(1-0.999**t))+1e-8)
    return dict(theta=theta, theta_init=theta_init, mask=mask, signJ=signJ, nnz=nnz,
                kobs=kobs, kg=kg, k2=k2, i_min=i_min)


def local_turing_volume(J, D, sigma_eval, ndraw, seed):
    """exp11's STRICT criterion, cubic solver. Returns (frac_strict, kstar_cv, frac_hopf_leading)."""
    rng = np.random.default_rng(seed)
    m = (J != 0); nnz = int(m.sum())
    fJ = np.ones((ndraw,3,3)); fJ[:, m] = rng.lognormal(0, sigma_eval, (ndraw, nnz))
    Jp = J[None]*fJ
    Dp = D[None]*rng.lognormal(0, sigma_eval, (ndraw,3)); Dp[:, D == 0] = 0.0
    uniform_stable = max_re_eig_cubic(Jp) < 0
    idx = np.arange(3)
    M = np.broadcast_to(Jp[:, None], (ndraw, len(KG_E), 3, 3)).copy()
    M[:, :, idx, idx] -= K2_E[None, :, None]*Dp[:, None, :]
    sig = max_re_eig_cubic(M)
    ipos = sig[:, 1:].argmax(axis=1)+1
    unst = sig[np.arange(ndraw), ipos] > 1e-9
    ok = uniform_stable & unst
    ks = KG_E[ipos][ok]
    cv = float(ks.std()/ks.mean()) if ks.size > 2 else np.nan
    return float(ok.mean()), cv


def leading_mode_class(J, D):
    """Turing-I (real leading mode at k*) vs Turing-Hopf (complex). Uses eigvals for imag part."""
    M = J[None] - K2_E[:, None, None]*np.diag(D)[None]
    ev = np.linalg.eigvals(M)
    re = ev.real.max(axis=1)
    ip = int(np.argmax(re[1:]))+1
    j = int(np.argmax(ev[ip].real))
    return ("hopf" if abs(ev[ip][j].imag) > 1e-9 else "turing_I"), re[ip], KG_E[ip]


def basin_profile(theta_i, r, n_dir=64, tmax=0.8, nt=41, seed=0):
    """Along random unit directions u: fraction of directions still Turing at each |t|,
    and mean loss. This is the geometric width of the zero-loss / Turing basin."""
    rng = np.random.default_rng(seed)
    dim = theta_i.size
    U = rng.normal(size=(n_dir, dim)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    ts = np.linspace(0, tmax, nt)
    cand = theta_i[None, None] + ts[None, :, None]*U[:, None, :]      # (n_dir, nt, dim)
    flat = cand.reshape(-1, dim)
    Lv = loss_of_theta(flat, r["mask"], r["signJ"], r["nnz"], r["kobs"], r["kg"], r["k2"], r["i_min"])
    J_, D_ = theta_to_JD(flat, r["mask"], r["signJ"], r["nnz"])
    idx = np.arange(3)
    M = np.broadcast_to(J_[:, None], (flat.shape[0], len(KG_E), 3, 3)).copy()
    M[:, :, idx, idx] -= K2_E[None, :, None]*D_[:, None, :]
    sig = max_re_eig_cubic(M)
    ip = sig[:, 1:].argmax(axis=1)+1
    tur = (max_re_eig_cubic(J_) < 0) & (sig[np.arange(flat.shape[0]), ip] > 1e-9)
    return ts, tur.reshape(n_dir, nt).mean(axis=0), Lv.reshape(n_dir, nt).mean(axis=0)


def smoothed_loss(theta, r, s, npert=256, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(theta.shape[0]):
        cand = theta[i][None] + rng.normal(0, s, (npert, theta.shape[1]))
        out.append(loss_of_theta(cand, r["mask"], r["signJ"], r["nnz"], r["kobs"], r["kg"], r["k2"], r["i_min"]).mean())
    return np.array(out)


# ======================================================================================
# drivers
# ======================================================================================
def run_sweep(systems, sigma_train_list, n_seeds, ndraw, sigma_eval=SIGMA_EVAL, n_steps=N_STEPS):
    """Main arm table: one row per (system, sigma_train, seed)."""
    rows, sel = [], {}
    for sys_ in systems:
        for st in sigma_train_list:
            r = train_seeds(sys_, st, n_seeds, stable_seed(sys_["key"], st),
                            init_scramble=SCRAMBLE, n_steps=n_steps)
            J_, D_ = theta_to_JD(r["theta"], r["mask"], r["signJ"], r["nnz"])
            sel[(sys_["key"], st)] = (J_, D_, r)
            logJ, logD = unpack(r["theta"], r["mask"], r["nnz"])
            sig = sigma_curve(logJ, r["signJ"], logD, r["k2"])
            L, ks, s0, sp = loss_from_sigma(sig, r["kobs"], r["kg"], r["i_min"])
            for i in range(n_seeds):
                se_curve = max_re_eig_cubic(J_[i][None] - K2_E[:, None, None]*np.diag(D_[i])[None])
                ip = int(np.argmax(se_curve[1:])) + 1
                is_t = bool(max_re_eig_cubic(J_[i][None])[0] < 0 and se_curve[ip] > 1e-9)
                cls = leading_mode_class(J_[i], D_[i])[0] if is_t else None
                rec = dict(key=sys_["key"], morph=sys_["morph"], topo=sys_["topo"],
                           kstar_true=sys_["kstar"], kstar_fft=sys_["kstar_fft"],
                           sigma_train=st, seed=i, loss=float(L[i]), turing=is_t,
                           mode_class=cls, kstar_model=KG_E[ip],
                           sigma_max_model=se_curve[ip], s0=float(s0[i]),
                           Dratio=float(D_[i].max()/D_[i].min()),
                           kstar_rel_err=abs(KG_E[ip]-sys_["kstar"])/sys_["kstar"],
                           kstar_fft_rel_err=abs(KG_E[ip]-sys_["kstar_fft"])/sys_["kstar_fft"])
                if is_t:
                    for se_ in sigma_eval:
                        fr, cv = local_turing_volume(J_[i], D_[i], se_, ndraw,
                                                     stable_seed(sys_["key"], st, i, se_))
                        rec[f"rob_{se_}"] = fr
                        rec[f"kcv_{se_}"] = cv
                rows.append(rec)
    return pd.DataFrame(rows), sel


def run_eval_curves(sel, n_seeds, ndraw=300, grid=SIG_EVAL_FINE):
    """Survival curve vs evaluation noise, on the fine grid, for every selected point."""
    rows = []
    for (key, st), (J_, D_, r) in sel.items():
        for i in range(n_seeds):
            se_c = max_re_eig_cubic(J_[i][None] - K2_E[:, None, None]*np.diag(D_[i])[None])
            ip = int(np.argmax(se_c[1:])) + 1
            if not (max_re_eig_cubic(J_[i][None])[0] < 0 and se_c[ip] > 1e-9):
                continue
            cls = leading_mode_class(J_[i], D_[i])[0]
            for se_ in grid:
                fr, cv = local_turing_volume(J_[i], D_[i], se_, ndraw,
                                             stable_seed("fine", key, st, i, se_))
                rows.append(dict(key=key, sigma_train=st, seed=i, sigma_eval=se_,
                                 rob=fr, kcv=cv, mode_class=cls))
    return pd.DataFrame(rows)


def run_controls(systems, n_seeds, ndraw, sigma=0.20):
    """Two confound guards.

    posthoc_jitter    -- deterministic training, then ONE perturbation of size sigma applied
                         to the final point. Isolates "is it just stochasticity?" from "is it
                         the smoothed objective?".
    wide_init_nonoise -- deterministic training from 2x the init spread. Isolates "is it just
                         more exploration?".
    """
    rows = []
    for sys_ in systems:
        s0 = stable_seed(sys_["key"], "ctrl")
        r = train_seeds(sys_, 0.0, n_seeds, s0, init_scramble=SCRAMBLE)
        th_j = r["theta"] + np.random.default_rng(s0 + 1).normal(0, sigma, r["theta"].shape)
        rb = train_seeds(sys_, 0.0, n_seeds, s0 + 2, init_scramble=2.0 * SCRAMBLE)
        for arm, rr, th in (("posthoc_jitter", r, th_j), ("wide_init_nonoise", rb, rb["theta"])):
            J_, D_ = theta_to_JD(th, rr["mask"], rr["signJ"], rr["nnz"])
            for i in range(n_seeds):
                se_c = max_re_eig_cubic(J_[i][None] - K2_E[:, None, None]*np.diag(D_[i])[None])
                ip = int(np.argmax(se_c[1:])) + 1
                is_t = bool(max_re_eig_cubic(J_[i][None])[0] < 0 and se_c[ip] > 1e-9)
                rec = dict(arm=arm, key=sys_["key"], morph=sys_["morph"], seed=i, turing=is_t,
                           kstar_fft_rel_err=abs(KG_E[ip]-sys_["kstar_fft"])/sys_["kstar_fft"])
                if is_t:
                    for se_ in (0.10, 0.20):
                        rec[f"rob_{se_}"] = local_turing_volume(
                            J_[i], D_[i], se_, ndraw, stable_seed("ctrl", sys_["key"], arm, i, se_))[0]
                rows.append(rec)
    return pd.DataFrame(rows)


def matched_subset_table(df, sigma_train_list=SIGMA_TRAIN[1:], base=0.0, se_="rob_0.2"):
    """§1.2 comparable-conditions guard: compare k* ONLY on seeds that are Turing in BOTH
    arms. Without this, the pooled k* error mixes a population shift (noise arms convert
    previously-failing seeds) into what looks like an accuracy regression."""
    b = df[df.sigma_train == base].set_index(["key", "seed"])
    out = []
    for st in sigma_train_list:
        a = df[df.sigma_train == st].set_index(["key", "seed"])
        both = b.turing & a.turing
        new = (~b.turing) & a.turing
        out.append(dict(sigma_train=st, n_both=int(both.sum()), n_newly_turing=int(new.sum()),
                        kerr_base=b.loc[both, "kstar_fft_rel_err"].median(),
                        kerr_arm=a.loc[both, "kstar_fft_rel_err"].median(),
                        kerr_newly_turing=a.loc[new, "kstar_fft_rel_err"].median() if new.sum() else np.nan,
                        rob_base=b.loc[both, se_].median(), rob_arm=a.loc[both, se_].median()))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=REPO,
                    help="repo root holding data/datasets/<id>/payload.h5 (default: this script's tree)")
    ap.add_argument("--dataset", default="three_gene_qvar")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--ndraw", type=int, default=400)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--out", default=None, help="default: <repo>/experiments")
    # NOTE: default lands under experiments/, which is gitignored -- these outputs are
    # deliberately NOT tracked (see NOISE_ROBUSTNESS_PIPELINE.md; the script is the record).
    ap.add_argument("--fig-dir", default=None, help="default: <repo>/experiments/exp13/figures")
    ap.add_argument("--skip-figures", action="store_true")
    a = ap.parse_args(argv)
    a.out = a.out or os.path.join(a.repo, "experiments")
    a.fig_dir = a.fig_dir or os.path.join(a.repo, "experiments", "exp13", "figures")

    systems = load_family(a.dataset, repo=a.repo)
    print(f"[data] {a.dataset}: {len(systems)} systems")
    validate_cubic(systems)

    df, sel = run_sweep(systems, SIGMA_TRAIN, a.seeds, a.ndraw, n_steps=a.steps)
    dfine = run_eval_curves(sel, a.seeds)
    dctrl = run_controls(systems, a.seeds, a.ndraw)

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "exp13_noise_sweep_qvar.csv"), index=False)
    dfine.to_csv(os.path.join(a.out, "exp13_noise_eval_curves.csv"), index=False)
    dctrl.to_csv(os.path.join(a.out, "exp13_controls.csv"), index=False)

    tur = df[df.turing]
    print("\n[summary] by training-noise arm")
    print(df.groupby("sigma_train").agg(
        turing_rate=("turing", "mean"),
        rob_10=("rob_0.1", "median"), rob_20=("rob_0.2", "median"),
        kstar_fft_err=("kstar_fft_rel_err", "median")).round(4).to_string())
    print("\n[summary] matched subset (like-for-like k*)")
    print(matched_subset_table(df).round(4).to_string(index=False))

    piv = tur.pivot_table(index="key", columns="sigma_train", values="rob_0.2", aggfunc="median")
    p = piv.dropna(subset=[0.0, 0.20])
    w = stats.wilcoxon(p[0.20], p[0.0])
    d = p[0.20] - p[0.0]
    print(f"\n[paired] robustness@20%: n={len(p)} systems, median gain {d.median():+.4f}, "
          f"improved {int((d > 0.005).sum())}, worse {int((d < -0.005).sum())}, "
          f"Wilcoxon p={w.pvalue:.3e}")

    if not a.skip_figures:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from exp13_figures import draw_all
        except ImportError:
            print("[figures] exp13_figures.py not importable; skipping (data CSVs are written)")
        else:
            os.makedirs(a.fig_dir, exist_ok=True)
            base_csv = os.path.join(a.out, "exp11_robustness_baseline.csv")
            draw_all(df, dfine, dctrl, systems, a.fig_dir,
                     baseline_csv=base_csv if os.path.exists(base_csv) else None,
                     matched=matched_subset_table(df))
            print(f"[figures] written to {a.fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
