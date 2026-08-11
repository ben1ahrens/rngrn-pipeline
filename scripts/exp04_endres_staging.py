"""EXP04 - Endres-style DATA-FIRST staging + the stationarity residual.

WHAT THE ENDRES/MATAS-GIL PAPER ACTUALLY DOES (arXiv 2309.06339, extracted from source):
  * Their "LS" method fits the PDE residual directly on the observed fields:
    L = ||b1*lap(u) + f(beta)||^2 + ||lap(v) + g(beta)||^2, assuming steady state ut=0.
    This is EXACTLY our terms.stationarity_residual -- and it is their most accurate
    method in the noiseless case. exp03 omitted it entirely.
  * Training is STAGED: the first 10-20k of 200k iterations minimise ONLY the
    approximation (data) loss; the PDE loss is introduced afterwards, each term
    weighted so the approximation is not destroyed. Data first, physics second.
  * They compare patterns in the FREQUENCY domain (RAPS), explicitly NOT pixel space,
    because position and shape are set by initial conditions while the MODEL sets the
    pattern type and wavelength.
  * Unknown intensity scale is handled with a free rescaling variable (our frame anchor).

DIFFERENCES FROM OUR SETTING (state them; do not paper over):
  * They assume the CANDIDATE MODEL IS THE TRUE MODEL. We do not -- the RNGRN gated-Hill
    form cannot represent the generator's kinetics exactly, so a zero residual is not
    attainable and the residual is a soft guide, not an exact constraint.
  * They observe ALL species (u AND v). We may observe m < N.
  * They use 50x50 zero-flux; our frames are 96x96 periodic.

TESTS: does adding the residual, and/or reversing the stage order, beat exp03's baseline
(10/16 Turing, k* median 5.7%) on the same sample?
FIREWALL: inputs are the frame, its L and observed_idx only.
"""
import argparse, json, time
import numpy as np, torch, h5py
from rngrn.model import RNGRN
from rngrn.losses import terms as T
from rngrn.eval.analysis import turing_ok
from rngrn import observables as obs


def split_hinges(model, xstar, kgrid, margin=1e-3, k_min_frac=0.1):
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)
    i0 = max(1, int(k_min_frac * len(kgrid)))
    return (T._softplus_hinge(sig[0] + margin)
            + T._softplus_hinge(-(sig[i0:].max() - margin)))


def fit(mode, seed, frame_t, L, kstar_obs, steps, w_resid, anchor_w=2.0, lr=0.05, nk=200):
    obs_scale = float(frame_t.mean())
    kg = torch.linspace(0.0, max(3.0, 4 * kstar_obs), nk, dtype=torch.float64)
    m = RNGRN(N=3, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    obsidx = list(range(frame_t.shape[0]))
    for it in range(steps):
        opt.zero_grad()
        xs, c = T.steady_state(m)
        if not c:
            break
        xsd = xs.detach()
        try:
            Lk, _ = T.kstar_anchor(m, xsd, kg, kstar_obs)
            scale = anchor_w * ((np.log(obs_scale) - torch.log(xs.clamp_min(1e-6))) ** 2).mean()
            if mode == "physics_first":          # exp03 baseline: hinges on from step 0
                loss = split_hinges(m, xsd, kg) + Lk + scale
            elif mode == "data_first":           # Endres ordering: data, then physics ramped in
                w = 0.0 if it < 0.25 * steps else min(1.0, (it - 0.25 * steps) / (0.25 * steps))
                loss = Lk + scale + w * split_hinges(m, xsd, kg)
            if w_resid > 0:
                try:
                    Lr = T.stationarity_residual(m, frame_t, L, obsidx, latent_fields=None)
                    Lr = Lr[0] if isinstance(Lr, tuple) else Lr
                    loss = loss + w_resid * Lr
                except Exception:
                    pass
        except Exception:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
    try:
        xs, c = T.steady_state(m)
        if not c:
            return None
        J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
        if not np.all(np.isfinite(J)):
            return None
        ok, info = turing_ok(J, m.D.detach().cpu().numpy())
        return dict(turing=bool(ok), kstar=float(info["kstar"]))
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--sample", default="sample_0000")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="experiments/exp04_endres_staging.json")
    a = ap.parse_args()
    with h5py.File(f"data/datasets/{a.dataset}/payload.h5") as f:
        g = f[a.sample]
        frame = g["final_frame"][:]
        L = float(g.attrs["L"]); kstar_true = float(g.attrs["k_star"])
    frame_t = torch.tensor(np.asarray(frame, dtype=float))
    kstar_obs = obs.kstar_of(frame[0], L=L)
    print(f"{a.sample}  L={L:.2f}  k*_true={kstar_true:.4f}  k*_obs={kstar_obs:.4f}\n")
    out = {"_meta": dict(L=L, kstar_true=kstar_true, kstar_obs=kstar_obs, steps=a.steps)}
    t0 = time.time()
    for mode in ["physics_first", "data_first"]:
        for wr in [0.0, 0.3]:
            key = f"{mode}_resid{wr}"
            res = [fit(mode, s, frame_t, L, kstar_obs, a.steps, wr) for s in range(a.seeds)]
            ok = [r for r in res if r]; tur = [r for r in ok if r["turing"]]
            errs = [abs(r["kstar"] - kstar_true) / kstar_true for r in tur]
            out[key] = dict(converged=len(ok), turing=len(tur), seeds=a.seeds,
                            kstar_rel_err_median=float(np.median(errs)) if errs else None,
                            kstar_rel_err_min=float(np.min(errs)) if errs else None)
            e = f"{100*np.median(errs):.1f}% / best {100*np.min(errs):.1f}%" if errs else "n/a"
            print(f"{key:26s} conv {len(ok):2d}/{a.seeds}  TURING {len(tur):2d}  k*err {e}")
    out["_meta"]["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
