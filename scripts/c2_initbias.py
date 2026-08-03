"""c2_initbias.py — WHERE does criterion 3.1's agreement come from? (JOB B)

PREREGISTRATION §3.1 fails on nc1 in the way the criterion was written to catch: the
within-target modal fraction (0.250) equals the size-matched CROSS-target control (0.512
at n=2 group size, gap ~0.000 at every tolerance). That says the agreement present is not
the target's. It does NOT say what it IS. Three candidates, and they are separable:

  H1  THE INIT.   The random raw-parameter init already concentrates J's sign structure,
      and training never leaves that basin. Then cross-target agreement is the init's
      agreement, and it is visible with NO DATA AND NO TRAINING AT ALL.
  H2  THE PRIOR.  loss.weights.param_prior pulls every run toward one region.
  H3  THE OBJECTIVE. sigma(k) genuinely does not distinguish sign structures, so training
      moves J a long way and lands anywhere.

H1 is the only one measurable without spending a single training step, and it is the one
that would make the other two moot, so it is measured first. Two statistics:

  (A) THE INIT-ONLY BIAS FLOOR. Build untrained models exactly as recover() builds restart
      r of seed s -- RNGRN(N, form, seed=_restart_seed(model_seed, r), init=...) -- solve
      the model's own steady state and take its autodiff Jacobian. Then compute the SAME
      modal-fraction statistic scoring/reproducibility.py computes, over random groups of
      K, at rtol 0.02/0.05/0.10. No frame is read; nothing is trained. Whatever this
      number is, it is agreement that the data cannot be credited with.

  (B) HOW FAR TRAINING MOVES THE TOPOLOGY. For each stored run the winning restart is
      recoverable (argmin of restarts[].total), so the run's OWN init is reconstructible
      bit-for-bit. Compare sign(J_init) with sign(J_recovered) entrywise and as a whole
      structure. If they mostly agree, training is not choosing the topology -- the init
      is -- and H1 is confirmed against H3.

Usage:
  python scripts/c2_initbias.py floor [n_models] [form]
  python scripts/c2_initbias.py moved experiments/<root> [experiments/<root> ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rngrn.model import RNGRN
from rngrn.losses.terms import steady_state
from rngrn.recover import _restart_seed

RTOLS = (0.02, 0.05, 0.10)
N_GROUP_DRAWS = 2000


def sign_key(J, rtol):
    """The sign rule of scoring/reproducibility.py, reproduced so this script and
    c2_repro.py cannot drift: entries below rtol * max|J| collapse to 'no edge'."""
    J = np.asarray(J, dtype=float)
    m = np.abs(J).max()
    if m <= 0:
        return tuple(np.zeros(J.size, dtype=int))
    s = np.sign(J)
    s[np.abs(J) < rtol * m] = 0
    return tuple(int(v) for v in s.ravel())


def init_jacobian(seed, N=3, form="nc1", n_hill=2, init="default"):
    """J at the UNTRAINED init, at the model's own steady state. Returns None if the
    steady-state solve does not converge (recorded, never silently dropped)."""
    model = RNGRN(N=N, form=form, n_hill=n_hill, seed=int(seed),
                  dispersion_backend="cubic" if N == 3 else "eig", init=init)
    with torch.no_grad():
        pass
    xs, ok = steady_state(model)
    if not ok:
        return None
    J = model.jacobian(xs, create_graph=False).detach().cpu().numpy()
    return J


def canon_key(J, rtol):
    """sign_key quotiented by NODE RELABELING. Nothing in the objective pins node order:
    sigma(k) = max Re eig(J - k^2 D) is invariant under the simultaneous permutation
    J -> P J P^T, D -> P D P^T, so two seeds that recovered the SAME network with its
    nodes in a different order are scored as DIFFERENT structures by the raw statistic.
    The canonical representative is the lexicographic minimum over all N! relabelings.

    NOT a substitute for the pre-registered number. PREREGISTRATION §3.1 reads the RAW
    statistic at rtol 0.05 against the bar; this is reported beside it as an upper bound on
    how much of the shortfall is bookkeeping rather than disagreement."""
    import itertools
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    return min(sign_key(J[np.ix_(p, p)], rtol) for p in itertools.permutations(range(n)))


def modal_fraction(keys):
    c = Counter(keys)
    return c.most_common(1)[0][1] / len(keys)


def grouped_modal(keys, K, rng, n_draws=N_GROUP_DRAWS):
    """Mean modal fraction over random groups of size K drawn from `keys`.
    Matches the group size of the within-target statistic so the two are comparable."""
    keys = list(keys)
    if len(keys) < K:
        return float("nan")
    vals = []
    for _ in range(n_draws):
        idx = rng.choice(len(keys), size=K, replace=False)
        vals.append(modal_fraction([keys[i] for i in idx]))
    return float(np.mean(vals))


def cmd_floor(n_models=256, form="nc1", init="default"):
    """(A) the init-only bias floor."""
    # Use the SAME (model_seed, restart) derivation recovery uses, so these are draws from
    # the identical distribution -- not a re-implementation of "random init".
    seeds = []
    for ms in range(8):                       # the 8 train.seed values a cell uses
        for r in range(n_models // 8):
            seeds.append(_restart_seed(ms, r))
    Js, failed = [], 0
    for s in seeds:
        J = init_jacobian(s, form=form, init=init)
        if J is None:
            failed += 1
            continue
        Js.append(J)
    print(f"init={init} form={form}  n_models={len(Js)}  steady_state_failed={failed}")
    diag = np.array([np.diag(J) for J in Js])
    print(f"  Jacobian diagonal: any positive in {int((diag > 0).any(axis=1).sum())}"
          f"/{len(Js)} models   (STATE_OF_THE_SCIENCE §10: 88/88 real three_gene systems"
          f" have one; patterning needs self-activation)")
    rng = np.random.default_rng(0)
    print(f"  {'rtol':>6} {'pool modal':>11} {'K=8 modal':>10} {'K=2 modal':>10} "
          f"{'n_distinct':>10}")
    for rtol in RTOLS:
        keys = [sign_key(J, rtol) for J in Js]
        print(f"  {rtol:>6.2f} {modal_fraction(keys):>11.3f} "
              f"{grouped_modal(keys, 8, rng):>10.3f} {grouped_modal(keys, 2, rng):>10.3f} "
              f"{len(set(keys)):>10d}")
        top = Counter(keys).most_common(3)
        for k, n in top:
            print(f"         {''.join('+' if v > 0 else '-' if v < 0 else '0' for v in k)}"
                  f"  {n}/{len(keys)}")


def cmd_moved(roots):
    """(B) how far training moves the sign structure, per stored run."""
    print(f"{'root':<22} {'run':<16} {'target':<12} {'seed':>4} {'rwin':>4} "
          f"{'agree/9':>8} {'same struct':>11} {'|dJ|/|J|':>9}")
    tot = Counter()
    per_rtol_same = {r: [0, 0] for r in RTOLS}
    for root in roots:
        smap, seedmap = {}, {}
        p = Path(root) / "runs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    smap[r["run_id"]] = r.get("sample_key", "?")
                    seedmap[r["run_id"]] = r.get("seed", r.get("train_seed"))
        for rd in sorted(Path(root).glob("runs/*")):
            f = rd / "results" / "train_results.json"
            if not f.exists() or f.stat().st_size == 0:
                continue
            d = json.loads(f.read_text())
            Jr = np.array(d["recovered"]["J"], dtype=float)
            totals = [x["total"] for x in d["restarts"]]
            rwin = int(np.argmin(totals))
            # model_seed: cfg.model.seed is null in every C2 cell, so recovery used
            # train.seed. Read it from the run dir name rather than guessing.
            seed = seedmap.get(rd.name)
            if seed is None:
                seed = int(rd.name.split("seed")[-1])
            Ji = init_jacobian(_restart_seed(int(seed), rwin))
            if Ji is None:
                print(f"{Path(root).name:<22} {rd.name[-14:]:<16} "
                      f"{smap.get(rd.name, '?'):<12} {seed:>4} {rwin:>4} "
                      f"{'INIT-SS-FAILED':>8}")
                continue
            agree = int((np.sign(Ji) == np.sign(Jr)).sum())
            same05 = sign_key(Ji, 0.05) == sign_key(Jr, 0.05)
            rel = float(np.linalg.norm(Jr - Ji) / max(np.linalg.norm(Ji), 1e-30))
            for rtol in RTOLS:
                per_rtol_same[rtol][0] += int(sign_key(Ji, rtol) == sign_key(Jr, rtol))
                per_rtol_same[rtol][1] += 1
            tot["n"] += 1
            tot["agree"] += agree
            print(f"{Path(root).name:<22} {rd.name[-14:]:<16} "
                  f"{smap.get(rd.name, '?'):<12} {seed:>4} {rwin:>4} "
                  f"{agree:>8} {str(same05):>11} {rel:>9.3f}")
    if tot["n"]:
        print(f"\nmean raw-sign agreement init vs recovered: "
              f"{tot['agree'] / (9 * tot['n']):.3f}  over n={tot['n']} runs "
              f"(chance for 3 signs uniform = 0.333; for 2 signs = 0.500)")
        for rtol in RTOLS:
            a, b = per_rtol_same[rtol]
            print(f"  identical sign STRUCTURE at rtol {rtol:.2f}: {a}/{b} = {a / b:.3f}")


def _load_by_target(roots):
    by = {}
    for root in roots:
        smap = {}
        p = Path(root) / "runs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    smap[r["run_id"]] = r.get("sample_key", "?")
        for rd in sorted(Path(root).glob("runs/*")):
            f = rd / "results" / "train_results.json"
            if not f.exists() or f.stat().st_size == 0:
                continue
            J = json.loads(f.read_text()).get("recovered", {}).get("J")
            if J is not None:
                by.setdefault(smap.get(rd.name, "?"), []).append(np.array(J, dtype=float))
    return by


def cmd_compare(roots, K=8, n_models=256, key=sign_key, label="RAW (the pre-registered statistic)"):
    """The §3.1 statistic at the PRE-REGISTERED group size K=8, read against THREE
    references instead of one:

      * CROSS-TARGET, size-matched at the SAME K -- c2_repro.py is forced down to
        n_pick = min(K, n_targets) = 2 when a root holds two targets, and at group size 2
        the modal fraction saturates: it can only be 0.5 or 1.0, so its floor is 0.500 and
        it has almost no power. Drawing a K=8 group with a cap of ceil(K/n_targets) seeds
        per target keeps the group size matched AND keeps the group genuinely mixed.
      * INIT-ONLY -- the same statistic over UNTRAINED models. Agreement at or below this
        is agreement the data cannot be credited with.
      * ABSOLUTE FLOOR 1/K = 0.125 -- K mutually distinct structures.
    """
    by = _load_by_target(roots)
    targets = sorted(by)
    rng = np.random.default_rng(0)
    print(f"targets={len(targets)}  " + "  ".join(f"{t}:{len(by[t])}" for t in targets))
    init_keys = {}
    Js = []
    for ms in range(8):
        for r in range(n_models // 8):
            J = init_jacobian(_restart_seed(ms, r))
            if J is not None:
                Js.append(J)
    for rtol in RTOLS:
        init_keys[rtol] = [key(J, rtol) for J in Js]
    cap = -(-K // max(len(targets), 1))          # ceil
    print(f"\nstatistic: {label}")
    print(f"{'rtol':>6} {'within K=8':>11} {'cross K=8':>10} {'gap':>7} "
          f"{'init-only K=8':>14} {'floor':>6}")
    for rtol in RTOLS:
        keys_by_t = {t: [key(J, rtol) for J in by[t]] for t in targets}
        wv = [modal_fraction(keys_by_t[t]) for t in targets if len(keys_by_t[t]) >= K]
        cv = []
        for _ in range(N_GROUP_DRAWS):
            grp = []
            order = rng.permutation(len(targets))
            for i in order:
                ks = keys_by_t[targets[i]]
                take = min(cap, len(ks), K - len(grp))
                idx = rng.choice(len(ks), size=take, replace=False)
                grp += [ks[j] for j in idx]
                if len(grp) >= K:
                    break
            if len(grp) == K:
                cv.append(modal_fraction(grp))
        w = float(np.mean(wv)) if wv else float("nan")
        c = float(np.mean(cv)) if cv else float("nan")
        print(f"{rtol:>6.2f} {w:>11.3f} {c:>10.3f} {w - c:>7.3f} "
              f"{grouped_modal(init_keys[rtol], K, rng):>14.3f} {1 / K:>6.3f}")
    print("\n3.1 needs within(K=8) >= 0.75 AND (within - cross) >= 0.25.")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        cmd_compare(sys.argv[2:])
        cmd_compare(sys.argv[2:], key=canon_key,
                    label="PERMUTATION-QUOTIENTED (an upper bound on how much of the "
                          "shortfall is node-ordering bookkeeping; NOT the bar's number)")
    elif sys.argv[1] == "floor":
        cmd_floor(int(sys.argv[2]) if len(sys.argv) > 2 else 256,
                  sys.argv[3] if len(sys.argv) > 3 else "nc1",
                  sys.argv[4] if len(sys.argv) > 4 else "default")
    elif sys.argv[1] == "moved":
        cmd_moved(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
