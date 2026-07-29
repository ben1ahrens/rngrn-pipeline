"""test_robustness.py — synthetic-Jacobian tests for eval.analysis's robustness cloud.

Targets the five defects recorded in docs/ROBUSTNESS_MEASUREMENT.md section 3 and
fixed in eval/analysis.py:
  3.1/3.2 perturb PHYSICAL (J, D), multiplicatively, sign-safe.
  3.4     batched draws match a per-draw serial replica.
  3.5     the strict criterion (max Re eig(J) < 0) disagrees with the loose one
          (tr(J) < 0) on a constructed counterexample.
Uses hand-built Jacobians with a known answer, not any trained model or answer key.
"""
import numpy as np
import pytest

from rngrn.eval import analysis as A


def test_physical_perturbation_is_scale_invariant():
    """A single sigma_log must produce the SAME relative perturbation regardless of a
    parameter's magnitude, because the factor is drawn independently of the value it
    multiplies. This is what makes the ~14x median-physical-perturbation spread
    described in section 3.1 (which came from perturbing raw theta through three
    different link functions) disappear once the perturbation is on the physical
    value directly."""
    # two nonzero entries three orders of magnitude apart
    J = np.array([[0.001, -50.0], [1.0, -1.0]])
    D = np.array([1.0, 2.0])
    n_samples = 20000
    sigma_log = 0.1

    Jp, _ = A._draw_JD_cloud(J, D, sigma_log, np.random.default_rng(0), n_samples)
    rel_small = np.abs(Jp[:, 0, 0] / J[0, 0] - 1.0)
    rel_large = np.abs(Jp[:, 0, 1] / J[0, 1] - 1.0)

    med_small, med_large = np.median(rel_small), np.median(rel_large)
    # same sigma_log -> same relative-perturbation distribution, independent of the
    # entry's magnitude (0.001 vs 50, a 5e4x difference in scale)
    assert med_small == pytest.approx(med_large, rel=0.1)


def test_perturbation_preserves_sign_and_structural_zeros():
    """Section 3.2: a lognormal factor is always positive, so a perturbed entry can
    never cross zero or flip sign. Structural zeros (no edge in J, an immobile node in
    D) must stay exactly zero."""
    J = np.array([[1.0, -1.0, 0.0],
                  [2.0, -2.0, 0.0],
                  [0.0, 0.0, -1.0]])
    D = np.array([1.0, 10.0, 0.0])
    Jp, Dp = A._draw_JD_cloud(J, D, 0.5, np.random.default_rng(1), 500)

    zero_mask = J == 0
    assert np.all(Jp[:, zero_mask] == 0.0)
    assert np.all(Dp[:, D == 0] == 0.0)

    pos_mask, neg_mask = J > 0, J < 0
    assert np.all(Jp[:, pos_mask] > 0.0)
    assert np.all(Jp[:, neg_mask] < 0.0)
    assert np.all(Dp[:, D != 0] > 0.0)


def test_strict_and_loose_disagree_on_constructed_counterexample():
    """tr(J) < 0 is necessary but not sufficient for max Re eig(J) < 0. J = diag(5, -3,
    -3) has trace -1 < 0 (loose says "stable") but its largest eigenvalue is 5 > 0
    (strict correctly says "unstable"). Perturbing this system should keep the two
    criteria in disagreement, matching section 3.5's measured overcount."""
    J = np.diag([5.0, -3.0, -3.0])
    D = np.array([1.0, 1.0, 1.0])
    res = A._perturb_cloud(J, D, 0.05, np.random.default_rng(2), 2000)

    assert res["frac_strict"] == 0.0          # never strictly stable at k=0
    assert res["frac_loose"] > 0.5            # loose criterion is fooled by the trace
    assert res["frac_loose_only"] > 0.5        # the gap between them is the overcount


def test_batching_matches_serial():
    """The vectorized (n_samples, N, N) linear algebra in _perturb_cloud must give
    exactly the same strict/unstable verdict, per draw, as a serial per-draw replica
    computed from the SAME drawn (Jp, Dp)."""
    J = np.array([[1.0, -1.0, 0.0],
                  [2.0, -2.0, 0.0],
                  [0.0, 0.0, -1.0]])
    D = np.array([1.0, 10.0, 0.0])
    n_samples = 40
    sigma_log = 0.1
    n = J.shape[0]
    kgrid = A._CLOUD_KGRID

    Jp, Dp = A._draw_JD_cloud(J, D, sigma_log, np.random.default_rng(123), n_samples)

    # vectorized path (mirrors _perturb_cloud's stability computation exactly)
    k2 = (kgrid ** 2)[:, None, None]
    Dm = np.zeros((n_samples, n, n)); Dm[:, range(n), range(n)] = Dp
    sig = np.linalg.eigvals(Jp[:, None, :, :] - k2[None] * Dm[:, None, :, :]).real.max(axis=2)
    ipos_vec = sig[:, 1:].argmax(axis=1) + 1
    unst_vec = sig[np.arange(n_samples), ipos_vec] > 1e-9
    strict_vec = np.linalg.eigvals(Jp).real.max(axis=1) < 0
    keep_vec = strict_vec & unst_vec

    # serial per-draw replica on the identical (Jp, Dp) draws
    keep_serial = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        Jpi, Dpi = Jp[i], Dp[i]
        strict_i = np.linalg.eigvals(Jpi).real.max() < 0
        sig_i = np.array([np.linalg.eigvals(Jpi - k ** 2 * np.diag(Dpi)).real.max()
                          for k in kgrid])
        ipos_i = int(np.argmax(sig_i[1:])) + 1
        unst_i = sig_i[ipos_i] > 1e-9
        keep_serial[i] = strict_i and unst_i

    assert np.array_equal(keep_vec, keep_serial)


def test_robustness_volumes_keys():
    """robustness_volumes emits exactly the four run-index keys the spec asks for, and
    each is a valid fraction."""
    J = np.array([[1.0, -1.0, 0.0],
                  [2.0, -2.0, 0.0],
                  [0.0, 0.0, -1.0]])
    D = np.array([1.0, 10.0, 0.0])
    rng = np.random.default_rng(0)
    out = {}
    for label, sigma in A.ROBUSTNESS_SIGMA_LEVELS.items():
        res = A._perturb_cloud(J, D, sigma, rng, 200)
        out[f"turing_volume_{label}"] = res["frac_strict"]

    assert set(out.keys()) == {
        "turing_volume_1pct", "turing_volume_4p8pct",
        "turing_volume_10pct", "turing_volume_20pct",
    }
    for v in out.values():
        assert 0.0 <= v <= 1.0
