"""EXP03 - "Turing-first" (homotopy) recovery vs the current random-restart scheme.

LITERATURE GROUNDING
  Scholes et al. (Cell Systems 2019) and the Optimal-network-sizes study report that
  ~60% of 2-3 node TOPOLOGIES can pattern, but the PARAMETER volume that does is
  ~0.1% (this project's own training brief records 0.022% for the Scholes screen).
  So finding the Turing regime by random sampling is hopeless by construction -- our
  measured 0/300 random inits is the EXPECTED base rate, not a bug.
  Homotopy/continuation is the standard remedy for parameter identification whose
  objective has many poor local minima (Vyasarayani et al.; regularisation-homotopy
  for PDE parameter inversion): start from a point that already satisfies the hard
  constraint, then deform continuously toward the data-fitting objective.

STRATEGY UNDER TEST
  Stage A  reach the Turing manifold using a firewall-safe low-basal prior
           (recorded as "measured 82% Turing-unstable at init" -- WITHDRAWN 2026-08-04,
           docs/DECISIONS.md D-EVID-11: the strict figure is 0/398 = 0.0%, so Stage A
           never reached a Turing manifold at all. This script's rejection verdict
           stands, but the stated reason for expecting it to work does not).
  Stage B  homotopy: anneal the objective from "stay Turing" toward "match k*_obs",
           keeping the Turing hinges active so the path never leaves the manifold.

  Compared against the current scheme (default init + composite loss) on the SAME
  sample and the same step budget.

FIREWALL: the only data inputs are the frame, its L, and observed_idx. The observed
frame's mean intensity is an image observable, not an answer-key quantity.
"""
import argparse, json, time
import numpy as np, torch, h5py
from rngrn.model import RNGRN
from rngrn.losses.terms import steady_state, _softplus_hinge, kstar_anchor
from rngrn.eval.analysis import turing_ok
from rngrn import observables as obs


def turing_hinges_split(model, xstar, kgrid, margin=1e-3, k_min_frac=0.1):
    """Disjoint-support hinges (exp02): uniform stability from k=0, instability from
    k>0 only. The original shares sig.max() between both hinges, which at 60/60 inits
    IS sig0 -- the same scalar pulled in opposite directions."""
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)
    i_min = max(1, int(k_min_frac * len(kgrid)))
    return (_softplus_hinge(sig[0] + margin)
            + _softplus_hinge(-(sig[i_min:].max() - margin)))


def low_basal_init(seed, N=3):
    """FIREWALL-SAFE prior. Low basal + sub-saturated binding is what admits a positive
    Jacobian diagonal (self-activation): for the competitive form df_i/dx_i > 0 needs
    1 + KA*x^n < n. MEASURED 2026-08-04: 28.6% positive diagonal vs 0% for the default
    prior -- that is the real effect. The previously recorded "82% Turing-unstable" is
    withdrawn; the strict figure is 0% (docs/DECISIONS.md D-EVID-11)."""
    g = np.random.default_rng(seed)
    m = RNGRN(N=N, seed=seed)
    with torch.no_grad():
        m.theta_beta.copy_(torch.log(torch.expm1(torch.tensor(10**g.uniform(-4, -2, N)))))
        m.theta_s.copy_(torch.log(torch.expm1(torch.tensor(10**g.uniform(-2, -0.3, (N, N))))))
        m.theta_alpha.copy_(torch.log(torch.expm1(torch.tensor(10**g.uniform(0.3, 1.5, (N, N))))))
        m.theta_delta.copy_(torch.log(torch.expm1(torch.tensor(10**g.uniform(-1, 0.3, N)))))
        m.theta_g.copy_(torch.tensor(g.normal(0, 2.5, (N, N))))
        D = np.ones(N); D[1:] = np.sort(10**g.uniform(0.9, 2.4, N - 1))
        m.theta_D.copy_(torch.log(torch.tensor(D)))
    return m


def diagnose(m):
    xs, c = steady_state(m)
    if not c:
        return None
    J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
    D = m.D.detach().cpu().numpy()
    if not np.all(np.isfinite(J)):
        return None
    ok, info = turing_ok(J, D)
    return dict(turing=bool(ok), kstar=float(info["kstar"]), sig_max=float(info["sig_max"]))


def fit(mode, seed, frame, L, kstar_obs, steps, anchor_w=2.0, lr=0.05, nk=200):
    obs_scale = float(np.mean(frame))
    kg = torch.linspace(0.0, max(3.0, 4 * kstar_obs), nk, dtype=torch.float64)
    m = low_basal_init(seed) if mode == "turing_first" else RNGRN(N=3, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for it in range(steps):
        opt.zero_grad()
        xs, c = steady_state(m)
        if not c:
            break
        xsd = xs.detach()
        try:
            if mode == "turing_first":
                # homotopy: w goes 0 -> 1, handing weight from "be Turing" to "match k*"
                w = min(1.0, it / (0.4 * steps))
                Lk, _ = kstar_anchor(m, xsd, kg, kstar_obs)
                loss = turing_hinges_split(m, xsd, kg) + w * Lk
            else:
                Lk, _ = kstar_anchor(m, xsd, kg, kstar_obs)
                loss = turing_hinges_split(m, xsd, kg) + Lk
            loss = loss + anchor_w * ((np.log(obs_scale) - torch.log(xs.clamp_min(1e-6))) ** 2).mean()
        except Exception:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
    return diagnose(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--sample", default="sample_0000")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="experiments/exp03_turing_first.json")
    a = ap.parse_args()
    with h5py.File(f"data/datasets/{a.dataset}/payload.h5") as f:
        g = f[a.sample]
        frame = g["final_frame"][:]
        L = float(g.attrs["L"])
        kstar_true = float(g.attrs["k_star"])
    kstar_obs = obs.kstar_of(frame[0], L=L)
    print(f"{a.dataset}/{a.sample}  L={L:.2f}  k*_true={kstar_true:.4f}  k*_obs={kstar_obs:.4f}\n")
    out = {"_meta": dict(L=L, kstar_true=kstar_true, kstar_obs=kstar_obs, steps=a.steps)}
    t0 = time.time()
    for mode in ["baseline", "turing_first"]:
        res = [fit(mode, s, frame, L, kstar_obs, a.steps) for s in range(a.seeds)]
        ok = [r for r in res if r]
        tur = [r for r in ok if r["turing"]]
        errs = [abs(r["kstar"] - kstar_true) / kstar_true for r in tur]
        out[mode] = dict(converged=len(ok), turing=len(tur), seeds=a.seeds,
                         kstar_rel_err_median=float(np.median(errs)) if errs else None,
                         kstar_rel_err_min=float(np.min(errs)) if errs else None)
        print(f"{mode:14s} conv {len(ok):2d}/{a.seeds}  TURING {len(tur):2d}  "
              f"k*_rel_err median "
              f"{100*np.median(errs):.1f}%  best {100*np.min(errs):.1f}%" if errs
              else f"{mode:14s} conv {len(ok):2d}/{a.seeds}  TURING {len(tur):2d}  (no Turing solutions)")
    out["_meta"]["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
