"""test_unrolled_grad.py — the truncated-unrolled gradient path (REDESIGN_rngrn.md §4.2).

The second of the redesign's two gradient paths. Where `forward.PatternSolve` differentiates
the patterned steady state through the IFT/adjoint (and therefore REQUIRES convergence), this
one runs the ETDRK4 relax as differentiable torch ops and backpropagates through a bounded
number of unrolled steps. It requires no convergence at all, which is the whole point.

Four contracts, in the order `docs/PLAN_redesign_R3.md` Task 12 states them:

1. **Gradients flow to theta** through the unrolled segment — every parameter tensor, D
   included (D enters only through the ETDRK4 linear-operator coefficients, so a gradient
   there is the test that those coefficients are built differentiably and not from a numpy
   copy).
2. **The warm start contributes EXACTLY zero.** §4.2's truncation is "detach the
   warm-started state and differentiate only the final saturated segment". Asserted the
   strong way: a warm state carrying a live graph to theta yields BITWISE the same theta
   gradient as the same state detached, and no gradient path back to the warm tensor at all.
3. **Checkpointed and non-checkpointed gradients agree** to float64 round-off. Gradient
   checkpointing is a memory transformation, not a numerical one; if the two disagree, one
   of them is wrong and the FD curve cannot tell which.
4. **Activation memory scales with SEGMENT length, not total steps.** Measured with
   `torch.autograd.graph.saved_tensors_hooks` over the forward — the bytes autograd actually
   retains — because the codebase has no memory-assertion helper and `torch.cuda.*_memory_*`
   is unavailable on the CPU path the suite runs on. Two halves: warm-up length must not move
   the number at all, and the number must grow linearly in the segment.

Everything here is CPU, float64, at a small grid: these are contracts on the gradient path,
not measurements of it. The measurement — unrolled-vs-FD relative error against segment
length at the n=96 commensurate box — is `scripts/r3_unrolled_segment.py` and its committed
result under `experiments/redesign_r3/unrolled_segment/`.
"""
import inspect
import pathlib

import numpy as np
import pytest
import torch

import rngrn.solve_box as sb
import rngrn.unrolled as unrolled
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN, THETA_NAMES
from rngrn.unrolled import unrolled_relax

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
N_GRID = 32          # contracts, not numerics: the smallest grid that still holds a pattern
SEED = 0


def _load_ckpt() -> RNGRN:
    assert CKPT.exists(), (
        f"tracked fixture checkpoint missing: {CKPT} — it is committed to git; "
        f"a missing file means a broken checkout, not a skippable test")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m


@pytest.fixture(scope="module")
def setup():
    """The known-Turing fixture on its own commensurate solve box, plus a noise IC.

    The IC is x* + noise rather than a saturated pattern: the four contracts below are about
    the gradient PATH (does it reach theta, is the warm start cut, does checkpointing change
    the value, what does it retain), none of which depends on the field being saturated. The
    saturated operating point is where the FD curve is measured, not here.
    """
    model = _load_ckpt()
    xs, ok = steady_state(model)
    assert ok, "steady state failed on the fixture checkpoint"
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    assert float(sig.max()) > 0, "fixture is not Turing-unstable"
    khat = float(kg[int(sig.argmax())])
    box = sb.geometry(khat, n=N_GRID)
    dt = 0.2 / float(torch.linalg.eigvals(J).abs().max())
    rng = np.random.default_rng(SEED)
    xstar = xs.detach().numpy()
    X0 = torch.from_numpy(
        xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, N_GRID, N_GRID)))
    return model, X0[None], int(box["n"]), float(box["L_solve"]), dt


def _loss(X: torch.Tensor) -> torch.Tensor:
    """A scalar on the relaxed field. Deliberately NOT a spectral loss: Task 12 owns the
    gradient path, Tasks 13/14 own what is scored on the box."""
    return (X[:, 0] ** 2).mean()


class _SavedBytes:
    """Bytes of distinct tensor storages autograd RETAINS for backward, via saved-tensor
    hooks. Keyed by storage pointer so an aliased view is not double counted; every counted
    storage is kept alive by the graph for the lifetime of the measurement, so a pointer
    cannot be recycled underneath it."""

    def __init__(self):
        self._by_ptr: dict[int, int] = {}

    def pack(self, t: torch.Tensor):
        st = t.untyped_storage()
        self._by_ptr[st.data_ptr()] = st.nbytes()
        return t

    def unpack(self, t: torch.Tensor):
        return t

    @property
    def total(self) -> int:
        return sum(self._by_ptr.values())


def _retained_bytes(model, X0, n, L, dt, **kw) -> int:
    counter = _SavedBytes()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        out = unrolled_relax(model, X0, n, L, dt, **kw)
        _loss(out)
    return counter.total


# --------------------------------------------------------------------------------------
# 1. gradients reach theta
# --------------------------------------------------------------------------------------
def test_gradients_flow_through_the_unrolled_segment_to_every_theta(setup):
    model, X0, n, L, dt = setup
    out = unrolled_relax(model, X0, n, L, dt, segment_steps=4, checkpoint_every=2)
    assert out.requires_grad, "the unrolled segment returned a field with no graph"
    grads = torch.autograd.grad(_loss(out), [getattr(model, nm) for nm in THETA_NAMES])
    for nm, g in zip(THETA_NAMES, grads):
        assert torch.isfinite(g).all(), f"non-finite gradient for {nm}"
        assert float(g.abs().max()) > 0.0, (
            f"zero gradient for {nm} — the unrolled path does not reach it "
            f"(theta_D reaches the field only through the ETDRK4 coefficients)")


# --------------------------------------------------------------------------------------
# 2. the warm start is cut
# --------------------------------------------------------------------------------------
def test_detached_warm_start_contributes_exactly_zero_gradient(setup):
    """§4.2's truncation, asserted bitwise: a warm state with a live graph to theta must
    give the SAME theta gradient as the same state detached, and carry no gradient itself."""
    model, X0, n, L, dt = setup
    params = [getattr(model, nm) for nm in THETA_NAMES]

    # a warm state that genuinely depends on theta: one differentiable ETDRK4 step. The two
    # runs below start from the SAME VALUES and differ only in whether that dependence is
    # still visible to autograd, so any difference in the theta gradient is the leak.
    warm_live = unrolled_relax(model, X0, n, L, dt, segment_steps=1)
    assert warm_live.requires_grad

    g_detached = torch.autograd.grad(
        _loss(unrolled_relax(model, warm_live.detach(), n, L, dt, segment_steps=3,
                             checkpoint_every=1)),
        params)

    out = unrolled_relax(model, warm_live, n, L, dt, segment_steps=3, checkpoint_every=1)
    g_live = torch.autograd.grad(_loss(out), params + [warm_live], allow_unused=True)

    assert g_live[-1] is None, (
        "the warm state received a gradient — the unrolled driver did not detach it, so the "
        "path is not truncated and the segment is not the only differentiated stretch")
    for nm, ga, gb in zip(THETA_NAMES, g_detached, g_live[:-1]):
        assert torch.equal(ga, gb), (
            f"{nm}: warm-start graph changed the gradient (max |delta| "
            f"{float((ga - gb).abs().max()):.3e}) — the warm contribution is not exactly zero")


# --------------------------------------------------------------------------------------
# 3. checkpointing is a memory transformation, not a numerical one
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("every", [1, 2, 3])
def test_checkpointed_gradients_match_the_plain_unrolled_gradients(setup, every):
    model, X0, n, L, dt = setup
    params = [getattr(model, nm) for nm in THETA_NAMES]
    g_plain = torch.autograd.grad(
        _loss(unrolled_relax(model, X0, n, L, dt, segment_steps=6)), params)
    g_ckpt = torch.autograd.grad(
        _loss(unrolled_relax(model, X0, n, L, dt, segment_steps=6,
                             checkpoint_every=every)), params)
    flat_p = torch.cat([g.reshape(-1) for g in g_plain])
    flat_c = torch.cat([g.reshape(-1) for g in g_ckpt])
    rel = float((flat_p - flat_c).norm() / flat_p.norm())
    assert rel <= 1e-14, (
        f"checkpoint_every={every} changed the gradient by relative {rel:.3e} — "
        f"checkpointing must only trade compute for memory")


# --------------------------------------------------------------------------------------
# 4. activation memory
# --------------------------------------------------------------------------------------
def test_activation_memory_is_independent_of_warmup_length(setup):
    """The half of §4.2 that makes the path affordable: the un-differentiated warm relax may
    be arbitrarily long and costs no activation memory at all."""
    model, X0, n, L, dt = setup
    short = _retained_bytes(model, X0, n, L, dt, segment_steps=2, warmup_steps=0,
                            checkpoint_every=1)
    long = _retained_bytes(model, X0, n, L, dt, segment_steps=2, warmup_steps=16,
                           checkpoint_every=1)
    assert short == long, (
        f"16 extra warm-up steps changed retained activation bytes {short} -> {long}: the "
        f"warm relax is being differentiated")


def test_activation_memory_grows_linearly_in_segment_length(setup):
    """...and the other half: what IS differentiated costs a fixed amount per step, so the
    segment length is the memory knob."""
    model, X0, n, L, dt = setup
    b = [_retained_bytes(model, X0, n, L, dt, segment_steps=s, checkpoint_every=1)
         for s in (2, 4, 6)]
    assert b[1] > b[0] and b[2] > b[1], f"retained bytes did not grow with the segment: {b}"
    d1, d2 = b[1] - b[0], b[2] - b[1]
    assert abs(d2 - d1) <= 0.02 * d1, (
        f"retained bytes are not linear in segment length: increments {d1}, {d2} from {b}")


def test_checkpointing_actually_saves_activation_memory(setup):
    """A tripwire on the MECHANISM: if `checkpoint_every` silently stopped checkpointing, the
    three tests above would all still pass — retained bytes would remain warm-up-independent
    and linear in the segment, just ~31x larger per step.

    Asserted on MARGINAL bytes per differentiated step, taken from two segment lengths, NOT on
    the totals. The segment-INDEPENDENT coefficient graph (12.59 MB at n=32, the contour
    integral retained by making the ETDRK4 coefficients differentiable in D) dominates both
    totals, so at S=6 they differ by only 1.35x — a floor asserted on totals would either be
    unmeetable or would pass with checkpointing switched off.

    MEASURED at n=32, N=3, float64: 773,184 B/step un-checkpointed against 24,576 B/step at
    `checkpoint_every=1` — exactly one (1, 3, 32, 32) float64 field — a **31.5x** saving.
    Asserted at >5x, well below the measured value, because the exact ratio is an artefact of
    ETDRK4's four reaction evaluations per step and would move if the scheme did; the property
    being guarded is that checkpointing is ON, not that it saves precisely 31.5x.
    """
    model, X0, n, L, dt = setup

    def marginal(every):
        b2 = _retained_bytes(model, X0, n, L, dt, segment_steps=2, checkpoint_every=every)
        b6 = _retained_bytes(model, X0, n, L, dt, segment_steps=6, checkpoint_every=every)
        return (b6 - b2) / 4.0

    ckpt, plain = marginal(1), marginal(None)
    assert ckpt > 0 and plain > 0, f"degenerate marginals: ckpt {ckpt}, plain {plain}"
    saving = plain / ckpt
    assert saving > 5.0, (
        f"gradient checkpointing saved only {saving:.2f}x per differentiated step "
        f"({plain:.0f} B/step plain vs {ckpt:.0f} B/step at checkpoint_every=1; 31.5x "
        f"measured) — checkpointing looks disabled, and the segment length is then a "
        f"memory knob ~31x coarser than §4.2 assumes")


# --------------------------------------------------------------------------------------
# fail loud
# --------------------------------------------------------------------------------------
def test_a_blown_up_segment_raises_rather_than_returning_a_nan_field(setup):
    """CLAUDE.md §4: a non-finite field is refused, not handed to the optimiser as a NaN
    gradient. Forced with a dt 1000x the stability limit, which measured finite (~5e60) at
    8 steps and non-finite by 32 — the scheme's own blow-up, not an injected NaN."""
    model, X0, n, L, dt = setup
    with pytest.raises(RuntimeError, match="blew up"):
        unrolled_relax(model, X0, n, L, 1e3 * dt, segment_steps=32, checkpoint_every=8)


def test_the_adopted_segment_length_is_the_calibrated_default():
    """D-R3-2's ruling, pinned. A DRIFT TRIPWIRE, not a behaviour test: 128 is calibrated in
    `experiments/redesign_r3/unrolled_segment/results/curve.json` on ONE fixture, box and
    seed, so a silent change would move what every unrolled gradient means without touching a
    single call site. Verified red by mutating the constant."""
    assert unrolled.SEGMENT_STEPS_DEFAULT == 128
    default = inspect.signature(unrolled.unrolled_relax).parameters["segment_steps"].default
    assert default == unrolled.SEGMENT_STEPS_DEFAULT, (
        "the signature default drifted from the calibrated constant, so callers relying on "
        "the default are no longer getting the adopted number")


def test_segment_steps_must_be_positive(setup):
    model, X0, n, L, dt = setup
    with pytest.raises(ValueError, match="segment_steps"):
        unrolled_relax(model, X0, n, L, dt, segment_steps=0)
