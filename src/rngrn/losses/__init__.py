"""losses/ — the composite objective (RECOVERY SIDE).

terms.py     : differentiable loss terms (k*-anchor, Turing hinges, full-RHS
               residual, anti-collapse, morphology). Repackaged from
               rngrn_objective.py (tracked as SCAFFOLD — needs validation).
weighting.py : fixed/scheduled/gradnorm/ntk weighting-strategy registry.
total.py     : assemble named terms -> weighting strategy -> scalar.

Nothing here imports the answer-key side. observables (image-only) is fine.
"""
from .total import total_loss, compute_terms          # noqa: F401
from .weighting import WEIGHTING, build_strategy       # noqa: F401
