"""EXP02 - Fix the self-defeating turing_hinges objective.

DIAGNOSIS (measured, exp02): at 100% of random inits argmax_k sigma(k) is the k~0 grid
point. turing_hinges then computes L_unstable from sig.max() -- which IS sig0 -- while
L_uniform pushes sig0 down. The same scalar receives opposing gradients, so the only
escape (an interior peak) is punished on approach rather than rewarded.

FIX: separate the two conditions onto disjoint parts of the k-grid.
  L_uniform  from sigma at k=0
  L_unstable from max over k >= k_min  (k_min > 0, excluding the uniform mode)
plus an optional 'band' term that rewards the interior peak directly.

FIREWALL: model-side only; no ground-truth quantity is used anywhere here.
"""
import argparse, json, time
import numpy as np, torch
from rngrn.model import RNGRN
from rngrn.losses.terms import steady_state, _softplus_hinge
from rngrn.eval.analysis import turing_ok


def turing_hinges_split(model, xstar, kgrid, margin=1e-3, k_min_frac=0.1):
    """Disjoint-support Turing hinges. kgrid[0] must be the ~0 mode."""
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)
    sig0 = sig[0]
    i_min = max(1, int(k_min_frac * len(kgrid)))
    sig_pos = sig[i_min:].max()                      # strictly k > 0
    L_uniform = _softplus_hinge(sig0 + margin)
    L_unstable = _softplus_hinge(-(sig_pos - margin))
    return L_uniform + L_unstable, dict(sig0=float(sig0.detach()),
                                        sig_max_pos=float(sig_pos.detach()))


def run(objective, seeds, steps, anchor_w, obs_scale, lr=0.05, N=3, nk=200, kmax=3.0):
    kg = torch.linspace(0.0, kmax, nk, dtype=torch.float64)
    conv = pos = tur = 0
    for seed in range(seeds):
        m = RNGRN(N=N, seed=seed)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            xs, c = steady_state(m)
            if not c:
                break
            try:
                L, _ = objective(m, xs.detach(), kg)
            except Exception:
                break
            if anchor_w > 0:
                L = L + anchor_w * ((np.log(obs_scale) - torch.log(xs.clamp_min(1e-6))) ** 2).mean()
            L.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
            opt.step()
        try:
            xs, c = steady_state(m)
            if not c:
                continue
            J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
            if not np.all(np.isfinite(J)):
                continue
            conv += 1
            pos += bool((np.diag(J) > 0).any())
            ok, _ = turing_ok(J, m.D.detach().cpu().numpy())
            tur += bool(ok)
        except Exception:
            pass
    return dict(converged=conv, self_activating=pos, turing=tur, seeds=seeds,
                turing_frac=tur / max(conv, 1))


if __name__ == "__main__":
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dataset", default="three_gene_val")
    ap.add_argument("--sample", default="sample_0000")
    ap.add_argument("--out", default="experiments/exp02_objective_fix.json")
    a = ap.parse_args()
    with h5py.File(f"data/datasets/{a.dataset}/payload.h5") as f:
        obs_scale = float(np.mean(f[a.sample]["final_frame"][:]))
    from rngrn.losses.terms import turing_hinges
    out = {}
    t0 = time.time()
    for name, obj in [("original", turing_hinges), ("split", turing_hinges_split)]:
        for aw in [0.0, 2.0]:
            key = f"{name}_anchor{aw}"
            out[key] = run(obj, a.seeds, a.steps, aw, obs_scale)
            print(f"{key:22s} conv {out[key]['converged']:3d}/{a.seeds}  "
                  f"self-act {out[key]['self_activating']:3d}  TURING {out[key]['turing']:3d} "
                  f"({100*out[key]['turing_frac']:.1f}%)")
    out["_meta"] = dict(obs_scale=obs_scale, steps=a.steps, wall_s=round(time.time()-t0, 1))
    json.dump(out, open(a.out, "w"), indent=1)
