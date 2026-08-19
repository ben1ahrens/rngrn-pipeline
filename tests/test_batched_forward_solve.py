"""test_batched_forward_solve.py — the BATCHED forward/adjoint path (rngrn.forward).

The serial twin is pinned by `tests/test_forward_solve.py`; nothing pinned the batched
one. What is asserted here, in the order the review (docs/REVIEW_gpu_optim_delta.md, C2)
ranks it:

1. `BatchedPatternSolve.backward` against finite differences — the same
   REGRESSION-TRIPWIRE role the serial `test_ift_gradient_matches_finite_differences`
   plays, not a re-acceptance of D1: a wrong adjoint, a dropped correction or a sign
   error gives O(1) error, not a tolerance-scale one.
2. `BatchedPatternSolve.backward` against the serial `PatternSolve.backward`, member by
   member, from an IDENTICAL warm start so the two paths differ only in the gradient
   machinery under test (their fresh relaxes differ by FFT backend and would confound it).
3. Gradient-scatter isolation: a loss that depends on member j alone must leave every
   other member's row of every (B, ...) parameter gradient EXACTLY zero. This is the
   failure that would not crash — a transposed or broadcast-collapsed scatter would train
   every member on a blend of the others and read as slow convergence.
4. `relax_to_pattern_torch_batched`'s snapshot-on-finish semantics and its per-member
   failure verdicts, against the serial `relax_to_pattern_torch`.
5. `BatchedPatternSolver.solve_subset`'s reason vocabulary, warm-state keying and warm
   clearing, against `PatternSolver`'s.

Scale is deliberately small — 32^2, B=3, N=3, competitive, CPU float64 — so the module
runs without a GPU.
"""
import pathlib

import numpy as np
import pytest
import torch

import rngrn.forward as fwd
from rngrn.forward import (THETA_NAMES, BatchedPatternSolver, PatternSolver,
                           relax_to_pattern_torch, relax_to_pattern_torch_batched)
from rngrn.losses.terms import steady_state_batched
from rngrn.model import RNGRN, BatchedRNGRN

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
N_GRID = 32
PERIODS = 4.0
SEEDS = (0, 1, 2)
B = len(SEEDS)
#: theta offset that turns the one tracked checkpoint into B DISTINCT Turing-unstable
#: members. The fixture verifies each member ignites and patterns on its own.
MEMBER_SPREAD = 2e-2
#: Central-difference step for the FD tripwire, and the number of random directions.
FD_EPS = 1e-4
FD_DIRECTIONS = 10
#: IFT-vs-FD regression tripwire. The VALUE IS MANDATED, not chosen here:
#: `docs/REDESIGN_rngrn.md` §4.2 "A/B discipline" (on branch `docs/redesign-rngrn`)
#: specifies a D1-style finite-difference check of "10 directions x the active loss
#: terms, tol 1e-4", and R3's hard gate in the same document is "FD-faithfulness at
#: tol 1e-4". FD_DIRECTIONS = 10 comes from that same sentence.
#:
#: Supporting evidence that the batched chain clears the mandated bar with room, not the
#: bar's origin: worst 2.16e-06, median 6.65e-07 over these 10 directions (2026-08-19,
#: 32^2, B=3, CPU float64) — a ~46x margin. It is tighter than the serial path's 5e-3
#: (test_forward_solve.py:181), which is a best-of-two-eps on ONE direction at 64^2, so
#: nothing is weakened by applying the mandated value here.
#:
#: Like the serial one this is a TRIPWIRE, not a re-acceptance of D1 — the failures it
#: guards (a wrong adjoint, a dropped correction, a sign error) produce O(1) error, not
#: tolerance-scale error. A permuted or member-blended scatter is caught by the two tests
#: that follow this one, not by this one.
FD_TOL = 1e-4
#: The serial solver's whole reason vocabulary (PatternSolver's docstring); the batched
#: solver's docstring claims parity with it, and this module checks that claim.
SERIAL_REASONS = {"ok", "solve_failed", "not_patterned"}


@pytest.fixture(scope="module", autouse=True)
def _single_threaded():
    """Pin torch to ONE CPU thread for this module, and restore afterwards.

    Not a style preference — measured on this machine at (3, 3, 32, 32) float64: 50
    batched ETDRK4 steps cost 0.04 s at 1 thread and 8.61 s at the default 14, a 215x
    thread-thrash penalty on FFTs this small (the serial numpy relax is unaffected,
    0.03 s either way). At the default the fixture's fresh batched relax takes ~7 minutes
    instead of ~6 seconds. This is a property of the batched CPU path, not of these
    tests; it is recorded here because it is what makes the module runnable.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# ------------------------------------------------------------------------ fixtures

def _load_member(spread: float, seed: int) -> RNGRN:
    """The tracked known-Turing checkpoint, optionally offset in theta to make a member."""
    assert CKPT.exists(), (
        f"tracked fixture checkpoint missing: {CKPT} — it is committed to git; "
        f"a missing file means a broken checkout, not a skippable test")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert not ck.get("nondim"), "fixture must be a dimensional run"
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    if spread:
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for nm in THETA_NAMES:
                p = getattr(m, nm)
                p.add_(spread * torch.from_numpy(rng.standard_normal(tuple(p.shape))))
    return m


def _members() -> list[RNGRN]:
    """B members: the checkpoint itself plus B-1 theta-offset copies of it."""
    return [_load_member(0.0 if b == 0 else MEMBER_SPREAD, 100 + b) for b in range(B)]


def _member_loss(u: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A fixed-target, TRANSLATION-INVARIANT scalar of one member's (N, n, n) field.

    |fft2|^2 of the mean-removed field is invariant to translation, so the loss cannot be
    contaminated by the grid-phase sensitivity F-D1-1 warns about, and it is dense over
    modes rather than a handful of summary scalars. Any fixed target is valid for a
    gradient check (the serial module makes the same point about its own closures); this
    one is deliberately NOT imported from losses/spectral.py so the test pins
    rngrn.forward alone.
    """
    f = u - u.mean(dim=(-2, -1), keepdim=True)
    P = torch.abs(torch.fft.fft2(f)) ** 2
    return ((torch.log(P + 1.0) - target) ** 2).sum()


def _targets(u_stack: torch.Tensor, seed: int = 7) -> list[torch.Tensor]:
    """Per-member targets perturbed off each member's own base spectrum, so the gradient
    at the base point is nonzero."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(u_stack.shape[0]):
        f = u_stack[j] - u_stack[j].mean(dim=(-2, -1), keepdim=True)
        base = torch.log(torch.abs(torch.fft.fft2(f)) ** 2 + 1.0)
        out.append(base + torch.from_numpy(
            0.05 * rng.standard_normal(tuple(base.shape))))
    return out


def _xstar(model) -> torch.Tensor:
    xs, conv = steady_state_batched(model)
    assert bool(conv.all()), f"batched steady state did not converge: {conv}"
    return xs


def _solve_warm(model, u_warm: torch.Tensor, L: float):
    """A warm Newton-ONLY batched solve of all B members starting from `u_warm`.

    Newton-only (the default warm_mode) is the D1 instrumentation contract the serial FD
    test uses: an ETDRK4 re-relax would drift the pinned phase and contaminate the check.
    """
    solver = BatchedPatternSolver(model, N_GRID, L, seeds=SEEDS)
    for m in range(B):
        solver._warm[m] = u_warm[m].detach().clone()
    u, ok_members, reasons = solver.solve_subset(list(range(B)), _xstar(model))
    return solver, u, ok_members, reasons


@pytest.fixture(scope="module")
def batch():
    """B ignited members solved to their patterned states once for the module."""
    models = _members()
    bm = BatchedRNGRN(models)
    xs = _xstar(bm)
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = bm.dispersion(xs, kg).detach()
    sig_max = sig.max(dim=1).values
    assert bool((sig_max > 0).all()), (
        f"every fixture member must be Turing-unstable, got sigma_max {sig_max}")
    L = PERIODS * 2.0 * np.pi / float(kg[int(sig[0].argmax())])

    solver = BatchedPatternSolver(bm, N_GRID, L, seeds=SEEDS)
    u, ok_members, reasons = solver.solve_subset(list(range(B)), xs)
    assert ok_members == list(range(B)), (
        f"fixture members did not all pattern: ok={ok_members} reasons={reasons}")
    u0 = u.detach().clone()
    return dict(models=models, L=L, u0=u0, targets=_targets(u0), xstar=xs)


# ---------------------------------------------------- 1. batched backward vs FD

def _flat_direction(model, seed: int) -> np.ndarray:
    sizes = [getattr(model, nm).numel() for nm in THETA_NAMES]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(sum(sizes))
    return v / np.linalg.norm(v)


def _perturbed_batch(models, v: np.ndarray, scale: float) -> BatchedRNGRN:
    """A fresh BatchedRNGRN with the whole (B, ...) parameter stack offset by scale*v.

    The direction spans every member's theta, so a backward that blended members would
    not cancel out of the directional derivative.
    """
    bm = BatchedRNGRN(models)
    off = 0
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(bm, nm)
            dv = torch.from_numpy(v[off:off + p.numel()]).reshape(p.shape)
            p.add_(scale * dv)
            off += p.numel()
    return bm


def test_batched_backward_matches_finite_differences(batch):
    """Regression tripwire on the batched IFT chain over FD_DIRECTIONS random directions.

    Central differences on the summed per-member loss, at FD points that are warm
    Newton-ONLY solves from the base pattern (the serial test's discipline, F-D1-1).
    """
    models, L, u0, targets = batch["models"], batch["L"], batch["u0"], batch["targets"]

    bm = BatchedRNGRN(models)
    _, u_b, ok_members, reasons = _solve_warm(bm, u0, L)
    assert ok_members == list(range(B)), (ok_members, reasons)
    loss = sum(_member_loss(u_b[j], targets[j]) for j in range(B))
    g = torch.autograd.grad(loss, [getattr(bm, nm) for nm in THETA_NAMES])
    g_flat = torch.cat([gi.reshape(-1) for gi in g]).numpy()

    rel_errs = []
    for d in range(FD_DIRECTIONS):
        v = _flat_direction(bm, 1000 + d)
        d_ift = float(g_flat @ v)
        vals = {}
        for sgn in (+1.0, -1.0):
            m2 = _perturbed_batch(models, v, sgn * FD_EPS)
            _, u2, ok2, why2 = _solve_warm(m2, u0, L)
            assert ok2 == list(range(B)), f"FD solve failed (dir {d}, sgn {sgn}): {why2}"
            drift = float((u2.detach() - u0).norm() / u0.norm())
            assert drift < 0.2, f"FD solve left the base branch (drift {drift:.2f})"
            vals[sgn] = float(sum(_member_loss(u2[j].detach(), targets[j])
                                  for j in range(B)))
        d_fd = (vals[+1.0] - vals[-1.0]) / (2.0 * FD_EPS)
        rel_errs.append(abs(d_fd - d_ift) / max(abs(d_fd), abs(d_ift), 1e-300))
    worst = max(rel_errs)
    print(f"\n[batched-fd] worst rel err {worst:.2e} over {FD_DIRECTIONS} directions "
          f"(median {float(np.median(rel_errs)):.2e})")
    assert worst <= FD_TOL, (
        f"batched IFT-vs-FD worst relative error {worst:.2e} exceeds the {FD_TOL:g} "
        f"regression tripwire (per-direction: {[f'{e:.1e}' for e in rel_errs]}) — the "
        f"batched adjoint has broken the chain the serial path verifies")


# ------------------------------------- 2. batched backward vs the serial per-member path

def test_batched_backward_matches_serial_pattern_solve_member_by_member(batch):
    """The equivalence mandate: member m's slice of the batched gradient must equal the
    gradient the serial `PatternSolve` produces for that member alone.

    Both sides are started from the SAME warm field and run Newton-only, so the two
    fresh-relax implementations (numpy vs batched torch ETDRK4) cannot confound the
    comparison — what is compared is the adjoint and the dF/dtheta scatter.
    """
    models, L, u0, targets = batch["models"], batch["L"], batch["u0"], batch["targets"]

    bm = BatchedRNGRN(models)
    _, u_b, ok_members, reasons = _solve_warm(bm, u0, L)
    assert ok_members == list(range(B)), (ok_members, reasons)
    loss = sum(_member_loss(u_b[j], targets[j]) for j in range(B))
    g_batched = dict(zip(THETA_NAMES,
                         torch.autograd.grad(loss,
                                             [getattr(bm, nm) for nm in THETA_NAMES])))

    worst = 0.0
    for m in range(B):
        serial = PatternSolver(models[m], N_GRID, L, SEEDS[m])
        serial._warm = u0[m].clone()
        u_s, reason = serial.solve()
        assert reason == "ok", f"serial solve for member {m}: {reason}"
        du = float((u_s.detach() - u_b[m].detach()).norm() / u_b[m].detach().norm())
        assert du < 1e-9, (
            f"member {m}: serial and batched Newton landed on different fields "
            f"(rel {du:.2e}) — the gradient comparison below would be meaningless")
        g_s = torch.autograd.grad(_member_loss(u_s, targets[m]),
                                  [getattr(models[m], nm) for nm in THETA_NAMES])
        for nm, gs in zip(THETA_NAMES, g_s):
            gb = g_batched[nm][m]
            rel = float((gb - gs).norm() / max(float(gs.norm()), 1e-300))
            worst = max(worst, rel)
    print(f"\n[batched-vs-serial] worst relative gradient difference {worst:.2e}")
    assert worst <= 1e-10, (
        f"batched and serial gradients differ by {worst:.2e} relative — the batched "
        f"backward is not the serial one with a member axis")


# ------------------------------------------------------- 3. gradient-scatter isolation

def test_batched_backward_scatters_to_exactly_one_member_row(batch):
    """A loss depending on member j ONLY must leave every other member's row of every
    (B, ...) parameter gradient EXACTLY zero, and member j's row nonzero.

    Run for every j, so a transposed, rolled or reversed scatter cannot survive by
    coincidence, and asserted with `count_nonzero`, not a tolerance: any nonzero at all
    in another member's row is a leak, however small.
    """
    models, L, u0, targets = batch["models"], batch["L"], batch["u0"], batch["targets"]

    for j in range(B):
        bm = BatchedRNGRN(models)
        _, u_b, ok_members, reasons = _solve_warm(bm, u0, L)
        assert ok_members == list(range(B)), (ok_members, reasons)
        g = torch.autograd.grad(_member_loss(u_b[j], targets[j]),
                                [getattr(bm, nm) for nm in THETA_NAMES])
        for nm, gi in zip(THETA_NAMES, g):
            assert gi.shape[0] == B, (nm, gi.shape)
            assert float(gi[j].abs().max()) > 0.0, (
                f"loss on member {j} produced no gradient in {nm}'s row {j} — the "
                f"scatter missed its own member")
            for other in range(B):
                if other == j:
                    continue
                leaked = int(torch.count_nonzero(gi[other]))
                assert leaked == 0, (
                    f"loss on member {j} leaked {leaked} nonzero gradient entries into "
                    f"{nm} row {other} (max |g| {float(gi[other].abs().max()):.3e}) — "
                    f"the dF/dtheta scatter is not member-isolated; every member would "
                    f"train on a blend of the others")


# ---------------------------- 4. relax_to_pattern_torch_batched vs the serial relax

def _relax_inputs(batch):
    bm = BatchedRNGRN(batch["models"])
    xs = batch["xstar"]
    J = bm.jacobian(xs, create_graph=False).detach().cpu().numpy()
    rates = np.abs(np.linalg.eigvals(J)).max(axis=-1)
    return bm, xs.detach().cpu().numpy(), 0.2 / rates


def test_batched_relax_snapshot_matches_the_serial_relax_per_member(batch):
    """SNAPSHOT-ON-FINISH: each member's returned field is the one IT held when ITS
    detector fired, which is exactly what the serial `relax_to_pattern_torch` returns.

    `flat_tol=1e-3` is loose enough that the members finish at DIFFERENT chunks, which is
    the only regime in which snapshot-on-finish and stop-on-last-member differ; the test
    asserts the number of distinct finishing chunks is > 1 so it cannot go vacuous.
    """
    bm, xs_np, dt = _relax_inputs(batch)
    L, dev = batch["L"], torch.device("cpu")
    idx = torch.arange(B)
    fields, why = relax_to_pattern_torch_batched(
        bm, idx, xs_np, N_GRID, L, dt, SEEDS, dev,
        chunk=200, max_chunks=60, flat_tol=1e-3)
    assert why == [None] * B, f"batched relax did not finish every member: {why}"

    finishes = []
    for m in range(B):
        serial_field = relax_to_pattern_torch(
            bm.member(m), xs_np[m], N_GRID, L, float(dt[m]), SEEDS[m], dev,
            chunk=200, max_chunks=60, flat_tol=1e-3)
        rel = float((fields[m] - serial_field).norm() / serial_field.norm())
        assert rel < 1e-12, (
            f"member {m}'s batched snapshot differs from the serial relax by {rel:.2e} "
            f"relative — the member is not being snapshotted when ITS detector fired")
        # how many chunks this member needed, from the serial path's own trajectory
        finishes.append(_serial_finish_chunk(bm.member(m), xs_np[m], L, float(dt[m]),
                                             SEEDS[m], dev))
    print(f"\n[batched-relax] serial finishing chunks per member: {finishes}")
    assert len(set(finishes)) > 1, (
        f"every member finished at the same chunk ({finishes}) — snapshot-on-finish and "
        f"stop-on-last-member are indistinguishable here, so this test is vacuous; "
        f"retune flat_tol or MEMBER_SPREAD")


def _serial_finish_chunk(model, xstar, L, dt, seed, dev, chunk=200, max_chunks=60,
                         flat_tol=1e-3) -> int:
    """The chunk index at which the serial detector fires for one member.

    Re-runs the serial relax with an increasing budget rather than instrumenting it, so
    the production function is untouched. Only used to prove the test above is not
    vacuous.
    """
    for mc in range(5, max_chunks + 1):
        try:
            relax_to_pattern_torch(model, xstar, N_GRID, L, dt, seed, dev, chunk=chunk,
                                   max_chunks=mc, flat_tol=flat_tol)
            return mc
        except RuntimeError:
            continue
    raise AssertionError("member never saturated within the budget")


def test_batched_relax_reports_blow_up_and_non_saturation_per_member(batch):
    """FAILURE IS PER MEMBER: one member going non-finite is that member's `blew_up` and
    does not reach the others, and a budget too short for the detector is
    `no_saturation` for everyone — the two verdicts the serial relax raises."""
    bm, xs_np, dt = _relax_inputs(batch)
    L, dev = batch["L"], torch.device("cpu")
    idx = torch.arange(B)

    # one poisoned member in an otherwise healthy stack
    X0 = batch["u0"].clone()
    X0[1] = torch.full_like(X0[1], 1e200)
    fields, why = relax_to_pattern_torch_batched(
        bm, idx, xs_np, N_GRID, L, dt, SEEDS, dev,
        chunk=200, max_chunks=20, flat_tol=1e-3, X0=X0)
    assert why[1] == "blew_up" and fields[1] is None, (why, fields[1] is None)
    assert why[0] is None and why[2] is None, (
        f"a blown member reached its neighbours: {why}")
    with pytest.raises(RuntimeError, match="blew up"):
        relax_to_pattern_torch(bm.member(1), xs_np[1], N_GRID, L, float(dt[1]), SEEDS[1],
                               dev, chunk=200, max_chunks=20, flat_tol=1e-3, X0=X0[1])

    # a budget too short for the 5-chunk flatness window
    fields, why = relax_to_pattern_torch_batched(
        bm, idx, xs_np, N_GRID, L, dt, SEEDS, dev, chunk=5, max_chunks=4, flat_tol=1e-12)
    assert why == ["no_saturation"] * B, why
    assert fields == [None] * B
    with pytest.raises(RuntimeError, match="no saturation"):
        relax_to_pattern_torch(bm.member(0), xs_np[0], N_GRID, L, float(dt[0]), SEEDS[0],
                               dev, chunk=5, max_chunks=4, flat_tol=1e-12)


# ------------------------ 5. solve_subset reason vocabulary, warm keying and clearing

@pytest.fixture()
def mocked(batch):
    """A batched solver whose relax and Newton are mocked per-test: the state machine's
    transitions and vocabulary are the contract here, not the numerics (contracts 1-3
    cover those). Mirrors `test_forward_solve.py`'s `machine` fixture."""
    bm = BatchedRNGRN(batch["models"])
    xs = batch["xstar"]
    flat = torch.from_numpy(np.broadcast_to(
        xs.detach().cpu().numpy()[:, :, None, None],
        (B, bm.N, N_GRID, N_GRID)).copy())
    solver = BatchedPatternSolver(bm, N_GRID, batch["L"], seeds=SEEDS)
    return solver, xs, flat, batch["u0"]


def test_solve_subset_reason_vocabulary_matches_the_serial_solver(mocked, monkeypatch):
    """"ok" / "not_patterned" / "solve_failed", the serial vocabulary verbatim, chosen
    per member from the same three situations the serial solver distinguishes."""
    solver, xs, flat, u0 = mocked
    monkeypatch.setattr(fwd, "newton_polish", lambda F_fn, u0_, *a, **k: (u0_, 1e-12))

    # member 0 patterns, member 1 converges homogeneous, member 2's relax fails
    def fake_relax(model, idx, *a, **k):
        out, why = [], []
        for m in idx.tolist():
            if m == 2:
                out.append(None); why.append("blew_up")
            else:
                out.append(u0[m].clone() if m == 0 else flat[m].clone()); why.append(None)
        return out, why

    monkeypatch.setattr(fwd, "relax_to_pattern_torch_batched", fake_relax)
    u, ok_members, reasons = solver.solve_subset([0, 1, 2], xs)
    assert ok_members == [0] and u is not None and u.shape[0] == 1
    assert reasons == {1: "not_patterned", 2: "solve_failed"}
    assert set(solver.last_reason) <= SERIAL_REASONS, solver.last_reason
    assert solver.last_reason == ["ok", "not_patterned", "solve_failed"]


def test_solve_subset_keys_warm_state_by_global_member_and_clears_it_on_failure(
        mocked, monkeypatch):
    """Warm state is keyed by GLOBAL member index: solving a subset touches only that
    subset's slots, a member that patterns keeps its field, and a member that fails or
    comes back homogeneous is cleared (a homogeneous warm start re-converges homogeneous
    forever — the serial solver's reason for clearing)."""
    solver, xs, flat, u0 = mocked
    monkeypatch.setattr(fwd, "newton_polish", lambda F_fn, u0_, *a, **k: (u0_, 1e-12))
    sentinel = torch.full_like(u0[1], 7.0)
    solver._warm[1] = sentinel

    def fake_relax(model, idx, *a, **k):
        return [u0[m].clone() for m in idx.tolist()], [None] * int(idx.shape[0])

    monkeypatch.setattr(fwd, "relax_to_pattern_torch_batched", fake_relax)
    _, ok_members, _ = solver.solve_subset([0, 2], xs)
    assert ok_members == [0, 2]
    assert solver._warm[0] is not None and solver._warm[2] is not None
    assert torch.equal(solver._warm[0], u0[0]) and torch.equal(solver._warm[2], u0[2])
    assert solver._warm[1] is sentinel, (
        "solving members [0, 2] must not touch member 1's warm slot")

    # CLEARING, arranged so the assertion is load-bearing. Member 0 is seeded with a
    # HOMOGENEOUS warm field and settled by the warm Newton pass itself, so its slot
    # holds a non-None value that only `solve_subset`'s clearing loop can remove; delete
    # that loop and the assertion below fails. (The obvious arrangement — reset _warm to
    # None and drive the homogeneous field in through the relax — is VACUOUS: the slot is
    # already None whatever the solver does, and the only code that could refill it sits
    # behind the `if not ok_members: return` this path takes.)
    def _no_relax(*a, **k):
        raise AssertionError("the warm Newton pass must settle member 0 without a relax")

    monkeypatch.setattr(fwd, "relax_to_pattern_torch_batched", _no_relax)
    solver._warm[0] = flat[0].clone()
    u, ok_members, reasons = solver.solve_subset([0], xs)
    assert u is None and ok_members == [] and reasons == {0: "not_patterned"}
    assert solver._warm[0] is None, (
        "a homogeneous member must not survive as a warm start — Newton from it "
        "re-converges homogeneous forever")
    assert solver._warm[2] is not None, (
        "solving member 0 must not touch member 2's warm slot")
