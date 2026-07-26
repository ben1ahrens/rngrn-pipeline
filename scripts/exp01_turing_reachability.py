"""EXP01 — Can optimisation REACH Turing instability from a non-Turing init?

The user's constraint: the model need NOT initialise Turing-unstable, but training
must RECOVER the instability. An earlier crude test reached 0/6, but it used a
subsampled k-grid and an ad-hoc objective whose two terms fought each other. This
re-tests with the project's own turing_hinges objective.

FIREWALL: uses no ground-truth quantity. Pure model-side reachability.
"""
import argparse, json, time
import numpy as np, torch
from rngrn.model import RNGRN
from rngrn.losses.terms import steady_state, turing_hinges
from rngrn.eval.analysis import turing_ok


def diagnose(m):
    xs, conv = steady_state(m)
    if not conv:
        return None
    J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
    D = m.D.detach().cpu().numpy()
    if not np.all(np.isfinite(J)):
        return None
    ok, info = turing_ok(J, D)
    return dict(turing=bool(ok), sig_max=float(info["sig_max"]),
                max_diag=float(np.diag(J).max()),
                uniform_stable=bool(np.linalg.eigvals(J).real.max() < 0),
                D_ratio=float(D.max() / D.min()))


def run(seed, steps, lr, N=3, device="cpu", kmax=3.0, nk=200):
    torch.manual_seed(seed)
    m = RNGRN(N=N, seed=seed).to(device)
    before = diagnose(m)
    kgrid = torch.linspace(1e-3, kmax, nk, dtype=torch.float64, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    hit_at = None
    for it in range(steps):
        opt.zero_grad()
        xs, conv = steady_state(m)
        xs = steady_state.__globals__["steady_state_diff"](m, xs) if False else xs.detach()
        try:
            L, parts = turing_hinges(m, xs, kgrid)
        except Exception:
            break
        L.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if hit_at is None and it % 25 == 0:
            d = diagnose(m)
            if d and d["turing"]:
                hit_at = it
    after = diagnose(m)
    return dict(seed=seed, before=before, after=after, hit_at=hit_at)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/exp01_reachability.json")
    a = ap.parse_args()
    t0 = time.time()
    res = [run(s, a.steps, a.lr, device=a.device) for s in range(a.seeds)]
    ok = [r for r in res if r["after"]]
    n_t = sum(r["after"]["turing"] for r in ok)
    summary = dict(n=len(res), converged=len(ok), reached_turing=n_t,
                   frac=n_t / max(len(ok), 1),
                   turing_at_init=sum(bool(r["before"] and r["before"]["turing"]) for r in res),
                   median_max_diag_after=float(np.median([r["after"]["max_diag"] for r in ok])) if ok else None,
                   steps=a.steps, lr=a.lr, wall_s=round(time.time() - t0, 1))
    json.dump(dict(summary=summary, runs=res), open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
