"""RNGRN — inverse-Turing pipeline package.

A per-frame variational pipeline that recovers a gene regulatory network (GRN)
from a single converged image of a Turing pattern. The learnable object is a
differentiable reaction-diffusion model (the RNGRN) whose weights ARE the
recovered network.

The package is layered by rate-of-change (see nn-research-codebase-principles.md):
  * slow-changing library    : model, observables, regulation, dynamical, numerics, analysis
  * medium-changing orchestr.: losses/, train, optim/, eval/, validate
  * fast-changing experiments: configs/, scripts/

THE ONE NON-NEGOTIABLE RULE (the firewall): no analytic quantity of a
ground-truth system — its steady state x*, Jacobian J, dispersion sigma(k), or any
generating coefficient — may enter the recovery pipeline anywhere, including
parameter initialisation. Those live only in a separate answer-key object used
to SCORE recovery after the fact. Enforced by tests/test_firewall.py.
"""
from __future__ import annotations

__version__ = "0.1.0-template"

from .registry import Registry  # noqa: F401
