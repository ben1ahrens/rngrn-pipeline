"""test_spectral_utils.py — unit tests for rngrn.utils.spectral_block (side-neutral port).

Unit U2 (Milestone M1): Port of scoring.morphology._spectral_block to side-neutral utils.py,
with drift tests binding the two against divergence. The scoring module cannot be imported
recovery-side (firewall), so the arithmetic must live where both sides can reach it.

Drift tests verify bit-identical output across seeded random fields and multiple n values.
Error-path parity tests ensure both implementations reject the same invalid inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from rngrn import utils
from rngrn.scoring import morphology as MORPH


# --------------------------------------------------------------------------------------
# Drift tests: bit-identical output of utils.spectral_block vs scoring.morphology._spectral_block
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape,n_values", [
    ((64, 64), (24, 16)),
    ((96, 96), (24, 16)),
    ((128, 128), (24, 16)),
])
def test_spectral_block_drift_matches_scoring_exactly(shape, n_values):
    """On seeded random float64 fields, utils.spectral_block produces EXACT byte-for-byte
    output matching scoring.morphology._spectral_block at all tested block sizes.

    Drift tests are mandatory: the two definitions must not diverge under maintenance.
    numpy.array_equal checks for exact equality (same dtype, shape, and bit content).
    """
    rng = np.random.default_rng(42)
    field = rng.standard_normal(shape).astype(np.float64)

    for n in n_values:
        utils_result = utils.spectral_block(field, n=n)
        morph_result = MORPH._spectral_block(field, n=n)

        # Exact bit-for-bit equality required
        assert np.array_equal(utils_result, morph_result), (
            f"Drift detected at shape={shape}, n={n}: "
            f"max diff = {np.max(np.abs(utils_result - morph_result))}")


def test_spectral_block_drift_on_multiple_random_seeds():
    """Drift test across multiple random seeds ensures the port is robust, not lucky."""
    shape, n = (96, 96), 24
    for seed in (0, 123, 999):
        rng = np.random.default_rng(seed)
        field = rng.standard_normal(shape).astype(np.float64)

        utils_result = utils.spectral_block(field, n=n)
        morph_result = MORPH._spectral_block(field, n=n)

        assert np.array_equal(utils_result, morph_result), f"Drift at seed {seed}"


# --------------------------------------------------------------------------------------
# Error-path parity: both implementations reject the same invalid inputs
# --------------------------------------------------------------------------------------
def test_constant_field_raises_in_both():
    """A constant field (all values identical) has zero power in the central block and
    should raise ValueError in BOTH implementations with the same substance."""
    field = np.full((64, 64), 5.0, dtype=np.float64)

    with pytest.raises(ValueError, match="no power|spatial mode"):
        utils.spectral_block(field, n=24)

    with pytest.raises(ValueError, match="no power|spatial mode"):
        MORPH._spectral_block(field, n=24)


def test_block_size_too_small_raises_in_both():
    """Block size n < 4 must be rejected in both implementations."""
    field = np.random.default_rng(42).standard_normal((64, 64)).astype(np.float64)

    for n in (1, 2, 3):
        with pytest.raises(ValueError, match="need 4 <="):
            utils.spectral_block(field, n=n)

        with pytest.raises(ValueError, match="need 4 <="):
            MORPH._spectral_block(field, n=n)


def test_block_size_too_large_raises_in_both():
    """Block size n > min(H, W) must be rejected in both implementations."""
    field = np.random.default_rng(42).standard_normal((64, 64)).astype(np.float64)

    for n in (65, 100):
        with pytest.raises(ValueError, match="need 4 <="):
            utils.spectral_block(field, n=n)

        with pytest.raises(ValueError, match="need 4 <="):
            MORPH._spectral_block(field, n=n)


# --------------------------------------------------------------------------------------
# Sanity checks on utils.spectral_block output properties
# --------------------------------------------------------------------------------------
def test_spectral_block_output_is_normalized():
    """The output should sum to approximately 1 (within floating-point precision).
    This is the post-compression normalization: block / total_log."""
    field = np.random.default_rng(42).standard_normal((96, 96)).astype(np.float64)
    result = utils.spectral_block(field, n=24)

    assert pytest.approx(result.sum(), rel=1e-14) == 1.0


def test_spectral_block_output_shape():
    """Output should be a 1-D array of size n*n."""
    for n in (16, 24, 32):
        field = np.random.default_rng(42).standard_normal((96, 96)).astype(np.float64)
        result = utils.spectral_block(field, n=n)

        assert result.shape == (n * n,)
        assert result.ndim == 1


def test_spectral_block_output_all_finite():
    """All output values should be finite (no NaN, no inf)."""
    field = np.random.default_rng(42).standard_normal((96, 96)).astype(np.float64)
    result = utils.spectral_block(field, n=24)

    assert np.all(np.isfinite(result))
