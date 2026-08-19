"""test_raps_torch_parity.py — `observables.raps_torch` against `observables.raps` (C3).

`observables.raps` is the CALIBRATED PRIMARY k* estimator: the sub-bin centroid k* and the
one-bin bar are calibrated on it, and a silent swap of the estimator underneath a consumer
is forbidden. `observables.raps_torch` is its declared parity port, and it was promoted to
the relax saturation detector on two call paths — `forward.relax_to_pattern_torch` and
`forward._kstar_of_torch_batched` — while its own docstring said "not PINNED by a test —
add one before any further caller relies on it". This module is that pin
(`docs/REVIEW_gpu_optim_delta.md` C3).

Two regimes are tested SEPARATELY, because they fail differently:

* the **saturated pattern**, where the port's docstring claimed agreement to 3e-16, and
* the **detector's own operating point** — x* + 1e-2 noise, at the grid sizes the detector
  sees on its first chunk — which is the ill-conditioned regime the same docstring recorded
  a 14 % bin / 0.24 % k* divergence in.

Scale is deliberately small (<= 96^2, CPU float64, one tracked checkpoint) so the module
runs without a GPU.
"""
import pathlib

import numpy as np
import pytest
import torch

import rngrn.observables as obs
from rngrn.forward import relax_to_pattern_torch
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
#: Domain = this many periods of the model's fastest-growing linear mode, as
#: `tests/test_batched_forward_solve.py` sizes its box.
PERIODS = 4.0
#: Grid sizes the relax detector is actually run at, plus the two the port's docstring
#: quoted its own measurements on (16^2 noise, 64^2 pattern).
DETECTOR_GRIDS = (16, 32, 64, 96)
#: Grid for the saturated-pattern regime — the size the port's 3e-16 claim was made at.
PATTERN_GRID = 64

#: Parity tolerances. MEASURED, not chosen: on 13 (n, L) geometries where the two binnings
#: provably assign every lattice point identically — so the only remaining difference is
#: FFT backend and summation order (numpy `bincount` vs torch `scatter_add_`) — the worst
#: per-bin relative difference over 78 field/geometry pairs (noise and multi-mode cosine,
#: n in 16..128) was **4.80e-14** and the worst k* relative difference **4.50e-16**
#: (2026-08-19, this worktree's .venv, CPU float64). The bars below sit ~20x above those
#: floors. The k* floor is what the port's docstring reported as "3e-16"; the per-bin floor
#: is two orders looser because a bin's power is an unordered sum of up to n^2 terms and
#: the two accumulators add them in different orders.
PARITY_TOL_BIN = 1e-12
PARITY_TOL_KSTAR = 1e-14


@pytest.fixture(scope="module", autouse=True)
def _single_threaded():
    """Pin torch to ONE CPU thread for this module, and restore afterwards.

    Same reason as `tests/test_batched_forward_solve.py`'s fixture: torch's CPU FFTs on
    grids this small lose ~200x to thread thrash at the default thread count, which is
    what makes the one ETDRK4 relax in this module cheap enough to keep.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def geometry():
    """The tracked known-Turing checkpoint, its x*, its ETDRK4 dt and its box size L."""
    assert CKPT.exists(), (
        f"tracked fixture checkpoint missing: {CKPT} — it is committed to git; "
        f"a missing file means a broken checkout, not a skippable test")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert not ck.get("nondim"), "fixture must be a dimensional run"
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    xs, ok = steady_state(m)
    assert ok, "fixture checkpoint's steady state did not converge"
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = m.dispersion(xs, kg).detach()
    assert float(sig.max()) > 0, "fixture checkpoint must be Turing-unstable"
    L = PERIODS * 2.0 * np.pi / float(kg[int(sig.argmax())])
    J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
    dt = 0.2 / float(np.abs(np.linalg.eigvals(J)).max())
    return dict(model=m, xstar=xs.detach().cpu().numpy(), L=L, dt=dt)


def _worst(field: np.ndarray, L: float) -> tuple[float, float]:
    """(worst per-bin relative difference, k* relative difference) raps vs raps_torch."""
    kc, p, ks = obs.raps(field, L)
    kct, pt, kst = obs.raps_torch(torch.from_numpy(np.ascontiguousarray(field)), L)
    pt = pt.detach().cpu().numpy()
    assert p.shape == pt.shape, (
        f"the two RAPS disagree on the NUMBER OF BINS: {p.shape} vs {pt.shape}")
    assert np.array_equal(kc, kct.detach().cpu().numpy()), (
        "the two RAPS disagree on the bin CENTRES, which makes any k* comparison "
        "meaningless")
    den = np.maximum(np.maximum(np.abs(p), np.abs(pt)), 1e-300)
    bin_rel = float((np.abs(p - pt) / den).max())
    return bin_rel, float(abs(ks - kst) / abs(ks))


def _on_edge_lattice_points(n: int, L: float) -> int:
    """How many of the n^2 lattice radii |k| land EXACTLY on a bin edge m*dk.

    That is the only place `floor(|k|/dk)` and `np.digitize(|k|, edges)` can disagree, so
    a geometry with none of them cannot detect a binning swap and the parity tests on it
    are vacuous.
    """
    kfreq = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    KX, KY = np.meshgrid(kfreq, kfreq)
    r = np.sqrt(KX ** 2 + KY ** 2).ravel() / (2 * np.pi / L)
    return int(((np.abs(r - np.rint(r)) < 1e-9) & (np.rint(r) > 0)).sum())


# ------------------------------------------------------------------ non-vacuity

def test_the_fixture_geometries_can_detect_a_binning_swap(geometry):
    """NON-VACUITY GUARD for everything below.

    `floor(|k|/dk)` and `digitize` differ only where a lattice radius sits exactly on a
    bin edge; on a geometry with no such radius the parity assertions pass whichever
    binning `raps_torch` uses, and pin nothing. Every grid this module tests must have
    at least one.
    """
    L = geometry["L"]
    counts = {n: _on_edge_lattice_points(n, L) for n in set(DETECTOR_GRIDS) | {PATTERN_GRID}}
    assert all(c > 0 for c in counts.values()), (
        f"no lattice radius lies on a bin edge for grids {counts} at L={L!r} — the parity "
        f"tests in this module cannot see a floor/digitize swap and are vacuous; pick a "
        f"different L or grid")


def test_raps_torch_bins_exactly_as_np_digitize(geometry):
    """The cached bin assignment must be numpy's, point for point.

    This is the sharp form of the parity claim: `raps` bins with
    `np.digitize(|k|, edges) - 1` clipped, and `_raps_torch_bins` must produce the SAME
    integer for all n^2 points. `floor(|k|/dk)` is equal only up to an ulp of the
    division, and disagrees exactly on the bin-edge radii the guard above counts.
    """
    L = geometry["L"]
    for n in sorted(set(DETECTOR_GRIDS) | {PATTERN_GRID}):
        kfreq = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
        KX, KY = np.meshgrid(kfreq, kfreq)
        KR = np.sqrt(KX ** 2 + KY ** 2)
        dk = 2 * np.pi / L
        kbins = np.arange(0, KR.max() + dk, dk)
        want = np.clip(np.digitize(KR.ravel(), kbins) - 1, 0, len(kbins) - 2)
        idx, _, _ = obs._raps_torch_bins(n, L, torch.device("cpu"), torch.float64)
        got = idx.detach().cpu().numpy()
        nbad = int((want != got).sum())
        assert nbad == 0, (
            f"n={n}: {nbad}/{n * n} lattice points are binned differently from "
            f"`raps`'s np.digitize — `raps_torch` is not a parity port of the primary "
            f"estimator on this geometry")


# ------------------------------------------------------------------ regime 1: pattern

def test_raps_torch_matches_raps_on_a_saturated_pattern(geometry):
    """REGIME 1 — the saturated pattern the port's docstring claimed 3e-16 on."""
    g = geometry
    field = relax_to_pattern_torch(
        g["model"], g["xstar"], PATTERN_GRID, g["L"], g["dt"], 0, torch.device("cpu"),
        chunk=500, max_chunks=120, flat_tol=1e-4)
    ch0 = field[0].detach().cpu().numpy()
    assert float(ch0.std()) > 1e-3, (
        f"the fixture relax returned a flat field (std {ch0.std():.2e}) — this test is "
        f"not measuring the patterned regime")
    bin_rel, ks_rel = _worst(ch0, g["L"])
    assert bin_rel <= PARITY_TOL_BIN, (
        f"saturated {PATTERN_GRID}^2 pattern: worst RAPS bin differs by {bin_rel:.3e} "
        f"relative, above the {PARITY_TOL_BIN:.0e} round-off bar")
    assert ks_rel <= PARITY_TOL_KSTAR, (
        f"saturated {PATTERN_GRID}^2 pattern: k* differs by {ks_rel:.3e} relative, above "
        f"the {PARITY_TOL_KSTAR:.0e} round-off bar")


# ------------------------------------------------ regime 2: the detector's own input

@pytest.mark.parametrize("n", DETECTOR_GRIDS)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_raps_torch_matches_raps_at_the_detector_operating_point(geometry, n, seed):
    """REGIME 2 — x* + 1e-2 noise, the field the saturation detector sees on chunk ONE.

    This is where the port's docstring recorded its 14 % bin / 0.24 % k* divergence and
    argued it did not matter because the detector runs in the patterned regime. It does
    not: `relax_to_pattern_torch` evaluates k* from the FIRST chunk.
    """
    g = geometry
    rng = np.random.default_rng(seed)
    X = g["xstar"][:, None, None] + 1e-2 * rng.standard_normal((g["model"].N, n, n))
    field = np.ascontiguousarray(X[0])
    bin_rel, ks_rel = _worst(field, g["L"])
    assert bin_rel <= PARITY_TOL_BIN, (
        f"n={n} seed={seed}: worst RAPS bin differs by {bin_rel:.3e} relative, above the "
        f"{PARITY_TOL_BIN:.0e} round-off bar")
    assert ks_rel <= PARITY_TOL_KSTAR, (
        f"n={n} seed={seed}: k* differs by {ks_rel:.3e} relative, above the "
        f"{PARITY_TOL_KSTAR:.0e} round-off bar")


def test_kstar_of_torch_is_the_same_number_as_kstar_of(geometry):
    """The public one-liners, not just the arrays: `kstar_of_torch` is what the detector
    calls, and `kstar_of` is what the numpy relax and `recover` call."""
    g = geometry
    rng = np.random.default_rng(3)
    for n in DETECTOR_GRIDS:
        X = g["xstar"][:, None, None] + 1e-2 * rng.standard_normal((g["model"].N, n, n))
        f = np.ascontiguousarray(X[0])
        a = obs.kstar_of(f, L=g["L"])
        b = obs.kstar_of_torch(torch.from_numpy(f), L=g["L"])
        rel = abs(a - b) / abs(a)
        assert rel <= PARITY_TOL_KSTAR, (
            f"n={n}: kstar_of={a!r} vs kstar_of_torch={b!r} — {rel:.3e} relative, above "
            f"the {PARITY_TOL_KSTAR:.0e} round-off bar")
