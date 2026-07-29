"""EXP05 - Pixel minibatching: "tons of small batches taken from the same image".

WHAT ENDRES ACTUALLY DOES (arXiv 2309.06339, quoted from source):
  "we use batch training with batches of 128 elements for 200,000 iterations"
  "When training the approximation, we use information on the whole pattern, but when
   we switch to the LPDE, we only use interior points"
  Fig 3: "Schematic of choosing N randomly selected points (black) on the Turing
   pattern"; "a region of 3 x 3 pixels is sufficient to recover accurate enough
   parameters"; relative error falls with N as a power law with exponent ~ -1/2.

  So a "batch" is 128 SPATIAL PIXEL LOCATIONS drawn from the ONE image -- not 128
  images and not 128 samples. They train on tons of small pixel batches from a single
  pattern. My earlier reading (that no minibatch dimension existed) was wrong.

WHY IT MIGHT MATTER HERE, beyond copying them:
  The patterning region is ~0.1% of parameter space (Scholes et al.), so the landscape
  is dominated by poor local minima. Minibatch gradient NOISE is a standard escape
  mechanism -- full-batch gradients descend cleanly into the nearest bad basin.

WHICH RNGRN TERMS CAN BE MINIBATCHED (this is the crux):
  stationarity_residual  YES - a per-pixel PDE residual, exactly Endres' L_PDE.
  morphology             YES in principle (per-pixel statistics).
  kstar_anchor           NO  - depends on the Jacobian at the steady state and the
                               dispersion relation; it is a GLOBAL property of the
                               parameters, with no pixel index at all.
  turing_hinges          NO  - same reason.
  frame-scale anchor     NO  - but its target (mean intensity) can be a batch estimate,
                               which is itself a source of useful noise.
  So pixel batching enters through the residual. exp04 found the FULL-pixel residual
  degrades k* badly (33% vs 1.5%); the question here is whether a MINIBATCHED residual
  behaves differently, both as a regulariser and as a noise source.

FIREWALL: pixels come from the observed frame only.
"""
import argparse, json, time
import numpy as np, torch, h5py
from rngrn.model import RNGRN
from rngrn.losses import terms as T
from rngrn.eval.analysis import turing_ok
from rngrn import observables as obs
import _runlog


def split_hinges(model, xstar, kgrid, margin=1e-3, k_min_frac=0.1):
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)
    i0 = max(1, int(k_min_frac * len(kgrid)))
    return (T._softplus_hinge(sig[0] + margin)
            + T._softplus_hinge(-(sig[i0:].max() - margin)))


def residual_minibatch(model, full, lap, batch, gen):
    """Per-pixel PDE residual on a RANDOM SUBSET of pixels (Endres' L_PDE, batched).

    The Laplacian is precomputed spectrally on the WHOLE field (it is non-local, so it
    cannot be evaluated from a pixel subset), then the residual is scored on `batch`
    randomly chosen pixels. That mirrors Endres: whole pattern for the smooth part,
    sampled points for the PDE term.
    """
    N, H, W = full.shape
    P = H * W
    idx = torch.randint(0, P, (batch,), generator=gen)
    x = full.permute(1, 2, 0).reshape(P, N)[idx]              # (B, N)
    lapb = lap.permute(1, 2, 0).reshape(P, N)[idx]            # (B, N)
    fx = model.reaction(x)                                    # (B, N)
    rhs = model.D.reshape(1, N) * lapb + fx
    den = (model.D.reshape(1, N) * lapb).pow(2).mean() + fx.pow(2).mean() + 1e-9
    return rhs.pow(2).mean() / den


def fit(seed, frame_t, L, kstar_obs, steps, batch, w_resid, anchor_w=2.0, lr=0.05, nk=200):
    gen = torch.Generator().manual_seed(seed)
    obs_scale = float(frame_t.mean())
    kg = torch.linspace(0.0, max(3.0, 4 * kstar_obs), nk, dtype=torch.float64)
    m = RNGRN(N=3, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    full = frame_t.clone()
    lap = obs.laplacian_torch(full, L=L).detach()             # non-local: computed once, whole field
    for it in range(steps):
        opt.zero_grad()
        xs, c = T.steady_state(m)
        if not c:
            break
        xsd = xs.detach()
        try:
            Lk, _ = T.kstar_anchor(m, xsd, kg, kstar_obs)
            scale = anchor_w * ((np.log(obs_scale) - torch.log(xs.clamp_min(1e-6))) ** 2).mean()
            w = 0.0 if it < 0.25 * steps else min(1.0, (it - 0.25 * steps) / (0.25 * steps))
            loss = Lk + scale + w * split_hinges(m, xsd, kg)
            if w_resid > 0:
                if batch is None:
                    Lr = T.stationarity_residual(m, frame_t, L, list(range(frame_t.shape[0])))
                    Lr = Lr[0] if isinstance(Lr, tuple) else Lr
                else:
                    Lr = residual_minibatch(m, full, lap, batch, gen)
                loss = loss + w_resid * Lr
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
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="experiments/exp05_pixel_minibatch.json")
    a = ap.parse_args()
    run = _runlog.start("exp05_pixel_minibatch", vars(a))
    with h5py.File(f"data/datasets/{a.dataset}/payload.h5") as f:
        g = f[a.sample]
        frame = g["final_frame"][:]
        L = float(g.attrs["L"]); kstar_true = float(g.attrs["k_star"])
    frame_t = torch.tensor(np.asarray(frame, dtype=float))
    kstar_obs = obs.kstar_of(frame[0], L=L)
    print(f"{a.sample}  L={L:.2f}  k*_true={kstar_true:.4f}  pixels={frame.shape[1]*frame.shape[2]}\n")
    configs = [("no residual", None, 0.0),
               ("full-pixel residual", None, 0.3),
               ("batch 64", 64, 0.3), ("batch 128", 128, 0.3),
               ("batch 512", 512, 0.3), ("batch 128 w=1.0", 128, 1.0)]
    out = {"_meta": dict(L=L, kstar_true=kstar_true, kstar_obs=kstar_obs, steps=a.steps)}
    t0 = time.time()
    for name, b, wr in configs:
        res = [fit(s, frame_t, L, kstar_obs, a.steps, b, wr) for s in range(a.seeds)]
        ok = [r for r in res if r]; tur = [r for r in ok if r["turing"]]
        errs = [abs(r["kstar"] - kstar_true) / kstar_true for r in tur]
        out[name] = dict(converged=len(ok), turing=len(tur), seeds=a.seeds,
                         kstar_rel_err_median=float(np.median(errs)) if errs else None,
                         kstar_rel_err_min=float(np.min(errs)) if errs else None)
        e = f"{100*np.median(errs):5.1f}% / best {100*np.min(errs):5.2f}%" if errs else "n/a"
        print(f"{name:22s} conv {len(ok):2d}/{a.seeds}  TURING {len(tur):2d}  k*err {e}")
    out["_meta"]["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    _runlog.write_meta(a.out, run)
    _runlog.record("experiments", run, dict(dataset=a.dataset, sample=a.sample,
                   seeds=a.seeds, steps=a.steps, wall_s=out["_meta"]["wall_s"]))
