"""test_weight_noise.py — train-time weight noise (paper-wnoise unit, Unit A).

`recover()` can now perturb the model's raw parameters ONCE PER ADAM STEP before the
loss evaluation, restoring the clean parameters before `opt.step()` — the classic
weight-noise / smoothed-objective estimator: the loss (and hence the gradient) is
evaluated at the perturbed point, and the step is applied to the clean parameters.

Noise model (docs/DECISIONS.md D-WNOISE-1): LOGNORMAL MULTIPLICATIVE on the positive
physical parameters — for each family p in {s, alpha, delta, beta, D}, the perturbed
physical value is p * exp(sigma * z) with z ~ N(0, 1) elementwise. This is EXACT, not
approximate: theta_D is a log (D = exp(theta_D)) so it takes sigma*z additively, and the
softplus families are mapped through the exact softplus inverse. The gate logit theta_g
is NOT perturbed — the gate is a bounded (0,1) split of the binding budget, not a
positive scale, and leaving it clean preserves the perturbation's sign structure exactly
as `eval/analysis._draw_JD_cloud` (the evaluation perturbation model) does.

Design fixed by the controller:
  * sigma=0 is the identity path: no generator constructed, results bit-identical to a
    call that never mentions the knob.
  * sigma>0 with no seed RAISES (house style: no silent irreproducibility).
  * noise is resampled each Adam step from a torch.Generator seeded by
    weight_noise_seed, so a (sigma, seed) pair is exactly reproducible.
  * wired on BOTH the batched and the serial path.
"""
import numpy as np
import pytest
import torch

from rngrn import recover as R
from rngrn.model import RNGRN


N = 3
WEIGHTS = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0)


class _RI:
    """Minimal stand-in for data.gate.RecoveryInput — (frame, L, observed_idx, N) only."""
    def __init__(self, frame, L, N):
        self.frame = frame
        self.L = L
        self.N = N
        self.observed_idx = list(range(N))


def _frame(seed=0, H=24, W=24):
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                          torch.arange(W, dtype=torch.float64), indexing="ij")
    f = 1.0 + 0.3 * torch.sin(2 * np.pi * 4 * x / W) * torch.cos(2 * np.pi * 3 * y / H)
    return (f + 0.01 * torch.rand(H, W, generator=g, dtype=torch.float64)).unsqueeze(0)


def _ri():
    f = _frame()
    return _RI(np.repeat(f.numpy(), N, axis=0), 40.0, N)


def _recover(**kw):
    base = dict(form="competitive", n_restarts=2, adam_steps=5, lbfgs_steps=0, seed=7,
                staging_keys=(), weights=WEIGHTS)
    base.update(kw)
    return R.recover(_ri(), **base)


def _flat_params(res):
    return np.concatenate([np.asarray(res.params[k]).ravel()
                           for k in ("KA", "KR", "alpha", "delta", "beta", "D")])


# --------------------------------------------------------------------------------------
# (a) sigma=0 is the identity path, batched and serial
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("batched", [False, True])
def test_zero_sigma_is_bit_identical_with_and_without_kwargs(batched):
    ref = _recover(batched=batched)
    zero = _recover(batched=batched, weight_noise_sigma=0.0, weight_noise_seed=123)
    assert ref.loss == zero.loss
    np.testing.assert_array_equal(_flat_params(ref), _flat_params(zero))
    np.testing.assert_array_equal(ref.xstar, zero.xstar)


# --------------------------------------------------------------------------------------
# (b) sigma>0 changes the optimisation, batched and serial
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("batched", [False, True])
def test_positive_sigma_changes_the_result(batched):
    clean = _recover(batched=batched)
    noisy = _recover(batched=batched, weight_noise_sigma=0.1, weight_noise_seed=4242)
    # Same restart-init seeds, same steps: only the injected noise differs, so the
    # recovered parameters must move.
    assert float(np.abs(_flat_params(clean) - _flat_params(noisy)).max()) > 0.0
    assert clean.loss != noisy.loss


# --------------------------------------------------------------------------------------
# (c) same (sigma, weight_noise_seed) -> bit-identical; different seed -> different
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("batched", [False, True])
def test_same_noise_seed_reproduces_bit_identically(batched):
    a = _recover(batched=batched, weight_noise_sigma=0.1, weight_noise_seed=4242)
    b = _recover(batched=batched, weight_noise_sigma=0.1, weight_noise_seed=4242)
    assert a.loss == b.loss
    np.testing.assert_array_equal(_flat_params(a), _flat_params(b))


def test_different_noise_seed_diverges():
    a = _recover(batched=True, weight_noise_sigma=0.1, weight_noise_seed=4242)
    b = _recover(batched=True, weight_noise_sigma=0.1, weight_noise_seed=4243)
    assert float(np.abs(_flat_params(a) - _flat_params(b)).max()) > 0.0


# --------------------------------------------------------------------------------------
# (d) sigma>0 with no seed RAISES — fail loud, no silent irreproducibility
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("batched", [False, True])
def test_positive_sigma_without_seed_raises(batched):
    with pytest.raises(ValueError, match="weight_noise_seed"):
        _recover(batched=batched, weight_noise_sigma=0.1)


def test_negative_sigma_raises():
    with pytest.raises(ValueError, match="weight_noise_sigma"):
        _recover(batched=True, weight_noise_sigma=-0.1, weight_noise_seed=1)


# --------------------------------------------------------------------------------------
# (e) the perturbation itself: exact lognormal on the physical positives, gate untouched,
#     restore exact
# --------------------------------------------------------------------------------------
def test_perturbation_is_lognormal_on_physical_params_and_gate_is_untouched():
    torch.manual_seed(0)
    model = RNGRN(N=N, seed=11)
    sigma = 0.2
    gen = torch.Generator().manual_seed(99)
    clean = {name: getattr(model, name).detach().clone()
             for name in ("s", "gate", "alpha", "delta", "beta", "D")}
    log_ratios = []
    n_draws = 200
    for _ in range(n_draws):
        saved = R._weight_noise_perturb(model, sigma, gen)
        for fam in ("s", "alpha", "delta", "beta", "D"):
            pert = getattr(model, fam).detach()
            log_ratios.append((torch.log(pert) - torch.log(clean[fam])).ravel())
        # the gate is NOT perturbed: bit-identical while the noise is applied
        assert torch.equal(getattr(model, "gate").detach(), clean["gate"])
        R._weight_noise_restore(model, saved)
    lr = torch.cat(log_ratios)
    # exact lognormal: log-ratios are N(0, sigma^2) elementwise
    assert float(lr.mean().abs()) < 0.01
    assert float(lr.std()) == pytest.approx(sigma, rel=0.05)


def test_perturb_then_restore_is_bit_exact():
    model = RNGRN(N=N, seed=5)
    before = {name: getattr(model, name).detach().clone() for name in
              ("theta_s", "theta_g", "theta_alpha", "theta_delta", "theta_beta", "theta_D")}
    gen = torch.Generator().manual_seed(1)
    saved = R._weight_noise_perturb(model, 0.3, gen)
    R._weight_noise_restore(model, saved)
    for name, val in before.items():
        assert torch.equal(getattr(model, name).detach(), val), name


# --------------------------------------------------------------------------------------
# (f) config fields exist with inert defaults and fit() threads them into recover()
# --------------------------------------------------------------------------------------
def test_train_config_fields_default_off():
    from rngrn.config import TrainConfig
    cfg = TrainConfig()
    assert cfg.weight_noise_sigma == 0.0
    assert cfg.weight_noise_seed is None
    import inspect
    sig = inspect.signature(R.recover)
    assert sig.parameters["weight_noise_sigma"].default == 0.0
    assert sig.parameters["weight_noise_seed"].default is None


def test_fit_threads_weight_noise_from_train_config(tmp_path, monkeypatch):
    """fit() must pass cfg.train.weight_noise_sigma/seed to recover() — otherwise the
    config field is a no-op knob and frozen_config.yaml asserts noise the run never had."""
    from rngrn import train as TR
    from rngrn.config import Config, TrainConfig

    captured = {}

    class _Stop(Exception):
        pass

    def fake_recover(ri, **kw):
        captured.update(kw)
        raise _Stop()

    monkeypatch.setattr(TR.R, "recover", fake_recover)
    monkeypatch.setattr(TR, "_resolve_recovery_input", lambda cfg: (object(), object()))
    cfg = Config(train=TrainConfig(weight_noise_sigma=0.07, weight_noise_seed=5307))
    with pytest.raises(_Stop):
        TR.fit(cfg, runs_root=str(tmp_path))
    assert captured["weight_noise_sigma"] == 0.07
    assert captured["weight_noise_seed"] == 5307
