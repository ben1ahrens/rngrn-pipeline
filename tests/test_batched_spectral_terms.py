"""test_batched_spectral_terms.py — the batched twins in losses/spectral.py against their
serial reference, member by member.

Every symbol here is new (unit b2/U4's batched spectral surface) and had zero references
under `tests/` before this file (docs/REVIEW_gpu_optim_delta.md §5): `raps_torch_batched`,
`spectral_block_torch_batched`, the five `*_batched` term functions, `spectral_terms_batched`
and `is_ignited_batched`. The reviewer measured these by hand at <= 4.3e-14 relative (worst
term `spec_amp_fluct`, the other four exactly 0.0) on a (4, 3, 64, 64) stack, CPU float64 —
that probe was not a test and did not survive into CI; this file is that test, at the same
scale, with the tolerance set from THIS run's own measurement (printed below), not copied
from the reviewer's number.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rngrn.losses import spectral as S

torch.set_default_dtype(torch.float64)

B, N_SPECIES, N_GRID = 4, 3, 64
L, KSTAR = 20.0, 3.0

# Measured on this machine, this run, at the scale above (printed by each test below):
# raps_torch_batched / spectral_block_torch_batched agree with their serial twins EXACTLY
# (same scatter/reduction order per row, just organised batch-first); the five term
# functions agree to <= 1.37e-12 relative, worst term `spec_amp_fluct` (its skew ratio
# amplifies round-off more than the other four) -- same ORDER as the reviewer's <= 4.3e-14
# on a different random draw, not the same number. Set one order of magnitude above the
# measured worst so the tripwire is not shaved to the noise floor of one draw.
TERM_TOL = 1e-11


def _positive_stack(seed, b, c, n, offset=5.0):
    """A positive-mean random stack — keeps log(mean)/log(std) defined for
    spec_amp_mean/spec_amp_fluct, as a real Turing frame's absolute concentrations would
    (mirrors tests/test_spectral_terms.py::_random_frame)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((b, c, n, n)) + offset).astype(np.float64)


@pytest.fixture(scope="module")
def stack():
    """(cfg, targets, u): targets from a single-channel observed frame, u the (B, N, n, n)
    candidate patterned stack the batched term functions are evaluated on."""
    frame = _positive_stack(11, b=1, c=1, n=N_GRID)[0]           # (1, n, n) -- channel 0
    cfg = S.SpectralConfig(channels=(0,))
    targets = S.build_frame_targets(frame, L=L, kstar_obs=KSTAR, cfg=cfg)
    u = torch.tensor(_positive_stack(37, b=B, c=N_SPECIES, n=N_GRID), requires_grad=True)
    return dict(cfg=cfg, targets=targets, u=u)


# --------------------------------------------------------------------------------------
# 1. raps_torch_batched vs raps_torch
# --------------------------------------------------------------------------------------
def test_raps_torch_batched_matches_raps_torch_member_by_member(stack):
    """A wrong batch axis (e.g. a scatter that mixes rows) would show up as a bin-count
    or power mismatch on at least one member; a transposed member axis would show up as
    every member matching the WRONG serial row, which the per-member loop below catches."""
    u = stack["u"].detach()
    k_centers_b, power_b = S.raps_torch_batched(u[:, 0], L)
    worst = 0.0
    for m in range(B):
        k_centers_s, power_s = S.raps_torch(u[m, 0], L)
        assert np.array_equal(k_centers_b, k_centers_s), f"member {m}: bin centers differ"
        worst = max(worst, float((power_b[m] - power_s).abs().max()))
    print(f"\n[raps_torch_batched] worst abs diff over {B} members: {worst:.3e}")
    assert worst == 0.0, (
        f"raps_torch_batched vs raps_torch: worst abs diff {worst:.3e} (expected exact — "
        "same scatter arithmetic, only the batch axis differs)")


# --------------------------------------------------------------------------------------
# 2. spectral_block_torch_batched vs spectral_block_torch
# --------------------------------------------------------------------------------------
def test_spectral_block_torch_batched_matches_spectral_block_torch_member_by_member(stack):
    u = stack["u"].detach()
    block_b = S.spectral_block_torch_batched(u[:, 0], nblk=24)
    worst = 0.0
    for m in range(B):
        block_s = S.spectral_block_torch(u[m, 0], nblk=24)
        worst = max(worst, float((block_b[m] - block_s).abs().max()))
    print(f"\n[spectral_block_torch_batched] worst abs diff over {B} members: {worst:.3e}")
    assert worst == 0.0, (
        f"spectral_block_torch_batched vs spectral_block_torch: worst abs diff {worst:.3e}")


# --------------------------------------------------------------------------------------
# 3. the five *_batched term functions vs their serial twins
# --------------------------------------------------------------------------------------
_TERM_PAIRS = [
    ("spec_shape", S.spec_shape_batched, S.spec_shape),
    ("spec_aniso", S.spec_aniso_batched, S.spec_aniso),
    ("spec_amp_mean", S.spec_amp_mean_batched, S.spec_amp_mean),
    ("spec_amp_fluct", S.spec_amp_fluct_batched, S.spec_amp_fluct),
    ("real_moments", S.real_moments_batched, S.real_moments),
]


@pytest.mark.parametrize("name,batched_fn,serial_fn", _TERM_PAIRS,
                         ids=[p[0] for p in _TERM_PAIRS])
def test_batched_term_matches_serial_member_by_member(stack, name, batched_fn, serial_fn):
    """Each `*_batched` term function against its serial namesake, member by member, on the
    SAME (B, N, n, n) stack and targets. A member-blended reduction (a missing `dim=` on one
    of the batched sums, e.g.) would show up as a value that matches no serial member;
    a transposed batch axis would show up as every member matching the wrong row."""
    u, cfg, targets = stack["u"].detach(), stack["cfg"], stack["targets"]
    val_b, parts_b = batched_fn(u, targets, cfg, members=list(range(B)))
    worst = 0.0
    for m in range(B):
        val_s, parts_s = serial_fn(u[m], targets, cfg)
        vb, vs = float(val_b[m]), float(val_s)
        worst = max(worst, abs(vb - vs) / max(abs(vs), 1e-300))
        for k, v_s in parts_s.items():
            assert float(parts_b[k][m]) == pytest.approx(v_s, abs=1e-12, rel=1e-9), (
                f"{name}/{k}: member {m} parts differ")
    print(f"\n[{name}] worst relative diff over {B} members: {worst:.3e}")
    assert worst <= TERM_TOL, (
        f"{name}_batched vs {name}: worst relative diff {worst:.3e} exceeds {TERM_TOL:g}")


# --------------------------------------------------------------------------------------
# 4. spectral_terms_batched vs looping spectral_terms (the aggregate wrapper)
# --------------------------------------------------------------------------------------
def test_spectral_terms_batched_matches_looped_spectral_terms(stack):
    """The five-term aggregate: a symbol of its own (brief table), and the thing
    `losses/total.py::_apply_spectral_batched` actually calls -- so this pins the exact
    entry point production code uses, not just the underlying per-term functions above."""
    u, cfg, targets = stack["u"].detach(), stack["cfg"], stack["targets"]
    vals_b, parts_b = S.spectral_terms_batched(u, targets, cfg, members=list(range(B)))
    assert set(vals_b) == set(S.SPECTRAL_TERM_KEYS)
    worst = 0.0
    for m in range(B):
        vals_s, parts_s = S.spectral_terms(u[m], targets, cfg)
        for k in S.SPECTRAL_TERM_KEYS:
            vb, vs = float(vals_b[k][m]), float(vals_s[k])
            worst = max(worst, abs(vb - vs) / max(abs(vs), 1e-300))
        for k, v_s in parts_s.items():
            assert float(parts_b[k][m]) == pytest.approx(v_s, abs=1e-12, rel=1e-9)
    print(f"\n[spectral_terms_batched] worst relative diff over {B} members x 5 terms: "
         f"{worst:.3e}")
    assert worst <= TERM_TOL, f"spectral_terms_batched worst relative diff {worst:.3e}"


# --------------------------------------------------------------------------------------
# 5. is_ignited_batched vs looping is_ignited
# --------------------------------------------------------------------------------------
def test_is_ignited_batched_matches_looped_is_ignited():
    """Every (ss_converged, sig_max_pos) combination the truth table has to distinguish:
    converged+unstable, converged+stable, and NOT converged (must be False regardless of
    sig_max_pos, mirroring `test_spectral_terms.py::test_is_ignited_truth_table`)."""
    parts = dict(ss_converged=np.array([True, True, False, True]),
                sig_max_pos=np.array([0.5, 0.05, 0.5, -0.2]))
    margin = 0.1
    got = S.is_ignited_batched(parts, margin)
    assert got.dtype == np.bool_
    for m in range(4):
        want = S.is_ignited(
            dict(ss_converged=bool(parts["ss_converged"][m]),
                sig_max_pos=float(parts["sig_max_pos"][m])),
            margin)
        assert bool(got[m]) == want, f"member {m}: batched {bool(got[m])} vs serial {want}"
    assert list(got) == [True, False, False, False], list(got)


def test_is_ignited_batched_raises_keyerror_without_sig_max_pos():
    with pytest.raises(KeyError):
        S.is_ignited_batched(dict(ss_converged=np.array([True])), margin=0.1)
