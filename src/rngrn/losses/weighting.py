"""weighting.py — adaptive loss-weighting strategies (Stage 3 outer loop).

A weighting strategy maps a dict of named UNWEIGHTED loss terms -> a scalar total,
choosing the per-term coefficients. Fixed/scheduled are baselines; GradNorm and
NTK are the adaptive families that rebalance terms by gradient statistics
(Cox-Matthews/Kassam-Trefethen is the solver side; on the weighting side these are
the GradNorm and NTK-PINN methods). All share the same interface so the
source-tagging and fail-loud guards in total.py survive regardless of strategy.

TEMPLATE STATUS: `fixed` and `scheduled` are fully implemented and exercised by
the dry run. `gradnorm` and `ntk` are structured stubs — the interface, state,
and update signature are in place and they RUN (falling back to current weights),
but the gradient-statistics update is left for Claude Code to implement and tune
on CUDA (see TUNING.md §Weighting). They are wired into the registry so switching
strategy is a config flag.
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

    STUB — runs with current (base) weights until Claude Code implements the update.
    The hook: compute ||grad_theta (w_k L_k)|| per term, nudge w_k toward the mean.
    """
    static_weights = False

    def __init__(self, base_weights, **kw):
        super().__init__(base_weights, **kw)
        self.w = dict(base_weights)

    def combine(self, terms, step, model=None):
        # TODO(claude-code): every `update_every` steps, backprop each term separately,
        # measure gradient norms, and rebalance self.w toward equal gradient magnitude.
        total = sum(self.w.get(k, 0.0) * v for k, v in terms.items())
        return total, dict(self.w)


@WEIGHTING.register("ntk")
class NTKWeighting(WeightingStrategy):
    """NTK-PINN weighting: set weights from the diagonal of the neural-tangent-kernel /
    per-term convergence-rate estimate (handles dissimilar term convergence, incl. the
    structural-parameter sensitivity relevant to inverse problems).

    STUB — runs with current (base) weights until Claude Code implements the estimator.
    """
    static_weights = False

    def __init__(self, base_weights, **kw):
        super().__init__(base_weights, **kw)
        self.w = dict(base_weights)

    def combine(self, terms, step, model=None):
        # TODO(claude-code): estimate per-term NTK trace (or residual-decay rate) and set
        # w_k inversely; recompute every `update_every` steps.
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
    return cls(loss_cfg.weights)
