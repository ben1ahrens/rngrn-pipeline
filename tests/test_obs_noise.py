"""test_obs_noise.py — the observation-noise knob at the firewall gate (claim 5 unit).

`gate.from_3gene_hdf5` (and its siblings) can now perturb the OBSERVED frame with
gaussian noise, relative to each channel's own clean std, before handing it to
recovery. This is legal at the gate -- gate.py sits astride the firewall (see its
module docstring) -- and the perturbation happens strictly after `_observe()` slices
the full frame down to the observed channels, so AnswerKey values are never touched.

Design fixed by the controller (docs/DECISIONS.md, claim-5 entry):
  * sigma is RELATIVE to the clean observed channel's own std (scale-free).
  * sigma=0 is the identity path: no RNG is constructed, bytes are unchanged.
  * sigma>0 with no seed RAISES (house style: no silent irreproducibility).
  * noise is per-channel, drawn from numpy.random.default_rng(seed).
"""
import numpy as np
import h5py
import pytest


def _write_sample(path, sample_key="sample_0000", N=3, H=16, L=57.0, seed=0):
    """Minimal synthetic 3-gene HDF5 sample in the from_3gene_hdf5 layout."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group(sample_key)
        # Use a non-trivial mean/scale per channel so relative-sigma noise is
        # distinguishable from channel to channel.
        base = rng.standard_normal((N, H, H)).astype("float64")
        scale = np.array([1.0, 3.0, 0.2])[:N].reshape(N, 1, 1)
        frame = base * scale + np.arange(N).reshape(N, 1, 1) * 5.0
        g.create_dataset("final_frame", data=frame)
        g.create_dataset("jacobian", data=rng.standard_normal((N, N)))
        g.create_dataset("x_star", data=rng.standard_normal(N))
        g.create_dataset("D", data=np.array([1.0, 40.0, 20.0])[:N])
        g.attrs["L"] = float(L)
        g.attrs["k_star"] = float(6.0 * 2.0 * np.pi / L)
        g.attrs["k_star_fft"] = float(1.08 * 6.0 * 2.0 * np.pi / L)


@pytest.fixture
def sample_path(tmp_path):
    path = tmp_path / "sample.h5"
    _write_sample(path)
    return str(path)


# --------------------------------------------------------------------------------------
# (i) sigma=0 -> bit-identical with/without the noise kwargs, no RNG constructed
# --------------------------------------------------------------------------------------
def test_zero_sigma_is_bit_identical_with_and_without_kwargs(sample_path):
    from rngrn.data import gate
    ri_default, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3,
                                         observed_idx=[0, 1, 2])
    ri_explicit_zero, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3,
                                               observed_idx=[0, 1, 2],
                                               obs_noise_sigma=0.0, obs_noise_seed=123)
    assert np.array_equal(ri_default.frame, ri_explicit_zero.frame)


def test_zero_sigma_does_not_construct_an_rng(sample_path, monkeypatch):
    """sigma=0 must not touch numpy's RNG machinery at all -- the identity path."""
    import numpy as _np
    from rngrn.data import gate

    def _boom(*a, **k):
        raise AssertionError("default_rng constructed on the sigma=0 path")

    monkeypatch.setattr(_np.random, "default_rng", _boom)
    ri, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                 obs_noise_sigma=0.0, obs_noise_seed=None)
    assert ri.frame is not None


# --------------------------------------------------------------------------------------
# (ii) same sigma+seed -> identical frames across two independent loads
# --------------------------------------------------------------------------------------
def test_same_sigma_and_seed_reproduces_identical_frames(sample_path):
    from rngrn.data import gate
    ri1, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                  obs_noise_sigma=0.05, obs_noise_seed=4201)
    ri2, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                  obs_noise_sigma=0.05, obs_noise_seed=4201)
    assert np.array_equal(ri1.frame, ri2.frame)


# --------------------------------------------------------------------------------------
# (iii) different seed -> different frames
# --------------------------------------------------------------------------------------
def test_different_seed_gives_different_frames(sample_path):
    from rngrn.data import gate
    ri1, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                  obs_noise_sigma=0.05, obs_noise_seed=4201)
    ri2, _ = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3, observed_idx=[0, 1, 2],
                                  obs_noise_sigma=0.05, obs_noise_seed=4205)
    assert not np.array_equal(ri1.frame, ri2.frame)


# --------------------------------------------------------------------------------------
# (iv) per-channel noise std ~ sigma * channel_std (statistical tolerance)
# --------------------------------------------------------------------------------------
def test_noise_std_matches_sigma_times_channel_std(tmp_path):
    """Use a large H so the added-noise std estimate is tight, and a clean (noiseless)
    frame so the measured std of (noisy - clean) is exactly the injected noise."""
    from rngrn.data import gate
    path = tmp_path / "big.h5"
    N, H, L = 3, 96, 57.0
    with h5py.File(path, "w") as f:
        g = f.create_group("sample_0000")
        # constant per-channel base -> channel std is driven entirely by injected noise,
        # so std(noisy - clean) is the injected noise's own std by construction.
        base = np.zeros((N, H, H))
        base[0] = 2.0
        base[1] = 10.0
        base[2] = -4.0
        g.create_dataset("final_frame", data=base)
        g.create_dataset("jacobian", data=np.eye(N))
        g.create_dataset("x_star", data=np.ones(N))
        g.create_dataset("D", data=np.array([1.0, 40.0, 20.0])[:N])
        g.attrs["L"] = float(L)
        g.attrs["k_star"] = float(6.0 * 2.0 * np.pi / L)

    ri_clean, _ = gate.from_3gene_hdf5(str(path), "sample_0000", N=3,
                                       observed_idx=[0, 1, 2])
    sigma = 0.2
    # channel std of a CONSTANT clean channel is 0, so noise magnitude must instead be
    # measured against a non-trivial reference channel: build one with a real std.
    with h5py.File(path, "a") as f:
        g = f["sample_0000"]
        del g["final_frame"]
        rng = np.random.default_rng(7)
        frame = rng.standard_normal((N, H, H)) * np.array([1.0, 3.0, 0.5])[:N].reshape(N, 1, 1)
        g.create_dataset("final_frame", data=frame)

    ri_clean, _ = gate.from_3gene_hdf5(str(path), "sample_0000", N=3, observed_idx=[0, 1, 2])
    ri_noisy, _ = gate.from_3gene_hdf5(str(path), "sample_0000", N=3, observed_idx=[0, 1, 2],
                                       obs_noise_sigma=sigma, obs_noise_seed=999)
    added = ri_noisy.frame - ri_clean.frame
    for c in range(N):
        expected_std = sigma * ri_clean.frame[c].std()
        measured_std = added[c].std()
        assert measured_std == pytest.approx(expected_std, rel=0.15), (
            c, measured_std, expected_std)


# --------------------------------------------------------------------------------------
# (v) sigma>0 + seed None RAISES -- fail loud, no silent irreproducibility
# --------------------------------------------------------------------------------------
def test_positive_sigma_without_seed_raises(sample_path):
    from rngrn.data import gate
    with pytest.raises(ValueError, match="obs_noise_seed"):
        gate.from_3gene_hdf5(sample_path, "sample_0000", N=3, observed_idx=[0, 1, 2],
                             obs_noise_sigma=0.05, obs_noise_seed=None)


# --------------------------------------------------------------------------------------
# (vi) noise applies to the OBSERVED frame only; AnswerKey values are untouched
# --------------------------------------------------------------------------------------
def test_answer_key_untouched_by_observation_noise(sample_path):
    from rngrn.data import gate
    ri_clean, ak_clean = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3,
                                              observed_idx=[0, 1, 2])
    ri_noisy, ak_noisy = gate.from_3gene_hdf5(sample_path, "sample_0000", N=3,
                                              observed_idx=[0, 1, 2],
                                              obs_noise_sigma=0.1, obs_noise_seed=4201)
    assert not np.array_equal(ri_clean.frame, ri_noisy.frame)
    np.testing.assert_array_equal(ak_clean.J, ak_noisy.J)
    np.testing.assert_array_equal(ak_clean.x_star, ak_noisy.x_star)
    np.testing.assert_array_equal(ak_clean.D, ak_noisy.D)
    assert ak_clean.kstar == ak_noisy.kstar
    assert ak_clean.kstar_fft == ak_noisy.kstar_fft


def test_train_threads_obs_noise_from_data_config(tmp_path, monkeypatch):
    """train._resolve_recovery_input must read cfg.data.obs_noise_sigma/seed and pass
    them through to the gate -- otherwise the config field is a no-op knob."""
    from rngrn.config import Config, DataConfig, ModelConfig
    from rngrn.train import _resolve_recovery_input
    from rngrn.data import gate as gate_mod

    path = tmp_path / "sample.h5"
    _write_sample(path)

    cfg = Config(
        data=DataConfig(source="hdf5_3gene", hdf5_path=str(path), sample_key="sample_0000",
                        obs_noise_sigma=0.05, obs_noise_seed=4201),
        model=ModelConfig(N=3, m=3),
    )
    ri_via_train, _ = _resolve_recovery_input(cfg)
    ri_direct, _ = gate_mod.from_3gene_hdf5(str(path), "sample_0000", N=3,
                                            observed_idx=[0, 1, 2],
                                            obs_noise_sigma=0.05, obs_noise_seed=4201)
    np.testing.assert_array_equal(ri_via_train.frame, ri_direct.frame)
