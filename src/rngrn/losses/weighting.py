"""weighting.py — adaptive loss-weighting strategies (Stage 3 outer loop).

A weighting strategy maps a dict of named UNWEIGHTED loss terms -> a scalar total,
choosing the per-term coefficients. All share the same interface so the
source-tagging and fail-loud guards in total.py survive regardless of strategy.

TEMPLATE STATUS: `fixed`, `scheduled`, and `ratio` are fully implemented. `ratio` is
the Matas-Gil & Endres ratio-based scheme (weight_k = loss_data / loss_k, recomputed
every `update_every` steps) — see RatioWeighting below. `gradnorm` and `ntk` are
registered names with NO estimator behind them; both raise NotImplementedError at
construction rather than silently running with fixed weights.
"""
from __future__ import annotations
import torch

from ..registry import Registry

WEIGHTING: "Registry[type]" = Registry("weighting")


class WeightingStrategy:
    """Base interface. `combine` returns (scalar_total, weights_used)."""

    # True when the per-term weights are a known function of `base` and `step` alone, so a
    # caller may read a weight of 0 as "this term can be skipped". Adaptive strategies that
    # move weights from gradient statistics set this False — for them a base weight of 0
    # says nothing about the weight at step 500.
    static_weights = True

    def __init__(self, base_weights: dict, **kw):
        self.base = dict(base_weights)
        self.kw = kw

    def combine(self, terms: dict, step: int, model=None) -> tuple:
        raise NotImplementedError


@WEIGHTING.register("fixed")
class FixedWeighting(WeightingStrategy):
    def combine(self, terms, step, model=None):
        w = self.base
        total = sum(w.get(k, 0.0) * v for k, v in terms.items())
        return total, dict(w)


@WEIGHTING.register("scheduled")
class ScheduledWeighting(WeightingStrategy):
    """Linearly ramp selected weights from 0 to base over `warmup` steps. Lets the
    k*-anchor and Turing hinges lead early, then phase in the residual."""
    def combine(self, terms, step, model=None):
        warmup = self.kw.get("warmup", 300)
        ramp_keys = self.kw.get("ramp", ["resid"])
        frac = min(1.0, step / max(1, warmup))
        w = {k: (v * frac if k in ramp_keys else v) for k, v in self.base.items()}
        total = sum(w.get(k, 0.0) * v for k, v in terms.items())
        return total, w


@WEIGHTING.register("gradnorm")
class GradNormWeighting(WeightingStrategy):
    """GradNorm-style balancing: scale each term so per-term gradient magnitudes match.

    NOT IMPLEMENTED. The interface is registered so config validation can name the
    strategy, but there is no gradient-statistics estimator behind it — running this
    silently with fixed weights would be a fail-loud violation, so it raises instead.
    """
    static_weights = False

    def __init__(self, base_weights, **kw):
        super().__init__(base_weights, **kw)
        raise NotImplementedError(
            "gradnorm weighting is not implemented (no per-term gradient-norm estimator "
            "exists). Use strategy: fixed, scheduled, or ratio.")


@WEIGHTING.register("ntk")
class NTKWeighting(WeightingStrategy):
    """NTK-PINN weighting: set weights from the diagonal of the neural-tangent-kernel /
    per-term convergence-rate estimate.

    NOT IMPLEMENTED. The interface is registered so config validation can name the
    strategy, but there is no NTK/convergence-rate estimator behind it — running this
    silently with fixed weights would be a fail-loud violation, so it raises instead.
    """
    static_weights = False

    def __init__(self, base_weights, **kw):
        super().__init__(base_weights, **kw)
        raise NotImplementedError(
            "ntk weighting is not implemented (no NTK/convergence-rate estimator "
            "exists). Use strategy: fixed, scheduled, or ratio.")


@WEIGHTING.register("ratio")
class RatioWeighting(WeightingStrategy):
    """Ratio-based loss balancing, as actually shipped by Matas-Gil & Endres
    (github.com/Endres-group/IPTP-paper-code, models/RBFPINNs_ChemPat.py lines 701-707) —
    simpler than GradNorm or NTK and the scheme that produced their results.

    Every `update_every` steps, recompute weight_k = loss_data / loss_k for every term
    k other than the data term, from the CURRENT unweighted term values. This renormalises
    each physics term on the fly to match the current data-loss magnitude, so a weighted
    physics term equals the current unweighted data loss. The data term itself keeps its
    FIXED base weight (Endres used 0.1) — it is never rescaled by this rule. No annealing,
    no gradient norms; weights are held constant between updates.
    """
    # This strategy MOVES weights, so a base weight of 0 licenses nothing and recover()
    # must not use it to prove the stationarity residual can be skipped. Set at the
    # 13-unit merge: unit 13 wrote this class before unit 1 introduced `static_weights`,
    # so it silently inherited the base class's True and would have qualified for the skip.
    static_weights = False

    def __init__(self, base_weights, **kw):
        super().__init__(base_weights, **kw)
        self.data_key = kw.get("data_key", "resid")
        self.update_every = max(1, int(kw.get("update_every", 50)))
        self.eps = kw.get("eps", 1e-12)
        self.w = dict(base_weights)

    def combine(self, terms, step, model=None):
        if step % self.update_every == 0:
            data_loss = float(terms[self.data_key].detach())
            w = dict(self.w)
            for k, v in terms.items():
                if k == self.data_key:
                    w[k] = self.base.get(self.data_key, 0.1)
                else:
                    w[k] = data_loss / max(float(v.detach()), self.eps)
            self.w = w
        total = sum(self.w.get(k, 0.0) * v for k, v in terms.items())
        return total, dict(self.w)


# --------------------------------------------------------------------------------------
# data-first staging (Endres-style): hold selected terms OFF, then ramp them in
# --------------------------------------------------------------------------------------
def staging_factor(step: int, total_steps: int, off_frac: float = 0.25,
                   ramp_frac: float = 0.25) -> float:
    """0 for the first `off_frac` of the budget, linear 0->1 over the next `ramp_frac`, then 1.

    PROVENANCE: scripts/exp05_pixel_minibatch.py line 87 —
        w = 0.0 if it < 0.25*steps else min(1.0, (it - 0.25*steps) / (0.25*steps))
    which is exactly this with off_frac = ramp_frac = 0.25. Those two fractions were used
    as-is in every experiment that reached the Turing regime and have never been swept.
    """
    if total_steps <= 0:
        raise ValueError(f"staging_factor: total_steps must be > 0, got {total_steps}")
    if not (0.0 <= off_frac <= 1.0):
        raise ValueError(f"staging_factor: off_frac must be in [0,1], got {off_frac}")
    if not (0.0 < ramp_frac <= 1.0):
        raise ValueError(f"staging_factor: ramp_frac must be in (0,1], got {ramp_frac}")
    off = off_frac * total_steps
    if step < off:
        return 0.0
    return min(1.0, (step - off) / (ramp_frac * total_steps))


class DataFirstStaging(WeightingStrategy):
    """WRAPPER that applies `staging_factor` to selected terms of ANY inner strategy.

    It delegates to `inner.combine` for the per-term weights (so it composes with fixed,
    scheduled, gradnorm and ntk alike), scales the staged keys by the schedule, and
    re-forms the total. `weights_used` reports the POST-staging weights, so a log of the
    weights actually applied stays truthful.

    Not registered in WEIGHTING: it takes an inner strategy, not a weights dict, so it
    cannot be constructed by `build_strategy`'s `cls(loss_cfg.weights)` protocol.
    recover.py wraps the strategy it is handed.
    """
    def __init__(self, inner: WeightingStrategy, total_steps: int,
                 keys=("turing",), off_frac: float = 0.25, ramp_frac: float = 0.25):
        super().__init__(inner.base)
        self.inner = inner
        self.total_steps = int(total_steps)
        self.keys = tuple(keys)
        self.off_frac = float(off_frac)
        self.ramp_frac = float(ramp_frac)
        # validate the schedule once, at construction, rather than on every step
        staging_factor(0, self.total_steps, self.off_frac, self.ramp_frac)

    @property
    def static_weights(self):
        """Staging is itself a static schedule, so this is exactly the inner strategy's."""
        return self.inner.static_weights

    def combine(self, terms, step, model=None):
        _, w = self.inner.combine(terms, step, model=model)
        f = staging_factor(step, self.total_steps, self.off_frac, self.ramp_frac)
        w = {k: (v * f if k in self.keys else v) for k, v in w.items()}
        total = sum(w.get(k, 0.0) * v for k, v in terms.items())
        return total, w


def build_strategy(loss_cfg) -> WeightingStrategy:
    cls = WEIGHTING.get(loss_cfg.strategy)
    kw = {}
    if loss_cfg.strategy == "ratio":
        kw["update_every"] = loss_cfg.ratio_update_every
    return cls(loss_cfg.weights, **kw)
