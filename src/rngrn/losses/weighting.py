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
    def __init__(self, base_weights, **kw):
        super().__init__(base_weights, **kw)
        self.w = dict(base_weights)

    def combine(self, terms, step, model=None):
        # TODO(claude-code): estimate per-term NTK trace (or residual-decay rate) and set
        # w_k inversely; recompute every `update_every` steps.
        total = sum(self.w.get(k, 0.0) * v for k, v in terms.items())
        return total, dict(self.w)


def build_strategy(loss_cfg) -> WeightingStrategy:
    cls = WEIGHTING.get(loss_cfg.strategy)
    return cls(loss_cfg.weights)
