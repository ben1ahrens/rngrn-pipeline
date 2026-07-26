"""rd_models.py — reference reaction-diffusion systems (ANSWER-KEY SIDE).

These are the ground-truth generators. They live on the answer-key side of the
firewall: they produce frames AND the analytic answer key (x*, J, k*), but ONLY
the frame + L + observed_idx are ever handed to recovery. This module must never
be imported by any recovery-side module (enforced in tests/test_firewall.py).

Two verified 2-species reference systems (GM stripes, Schnakenberg spots); their
parameter values and steady states match rngrn_design_doc.md Section 10.2. The
3-node dataset is NOT generated here — its kinetics live in the external generator
that produced three_gene_dataset/*.h5, and those samples enter via
data.gate.from_3gene_hdf5 (final_frame as the observable; jacobian/x_star/
interaction_matrix quarantined as the answer key). To generate NEW 3-node systems
in-pipeline, Claude Code registers a 3-species ReferenceSystem here (see TUNING.md).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve

from ..registry import Registry

RD_SYSTEMS: "Registry[type]" = Registry("rd_systems")


def _jac_fd(f, x, eps=1e-6):
    x = np.asarray(x, float); n = x.size
    J = np.zeros((n, n)); f0 = f(x)
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        J[:, j] = (f(xp) - f0) / eps
    return J


def _dispersion_max(J, D, kgrid):
    sig = np.array([np.max(np.real(np.linalg.eigvals(J - k**2 * np.diag(D)))) for k in kgrid])
    return sig


class ReferenceSystem:
    """Base: a reaction f(x), diffusion D, domain L, and an analytic answer key."""
    name = "base"
    D = None
    L = 100.0
    n_species = 2

    def reaction(self, x):
        raise NotImplementedError

    def steady_state(self):
        raise NotImplementedError

    def answer_key(self) -> dict:
        """The quarantined truth: x*, J, k*, sigma_max, D, coefficients. VALIDATION ONLY."""
        xstar = np.asarray(self.steady_state(), float)
        J = _jac_fd(self.reaction, xstar)
        kg = np.linspace(1e-3, 60.0, 6000)
        sig = _dispersion_max(J, np.asarray(self.D, float), kg)
        i = int(np.argmax(sig))
        return dict(name=self.name, x_star=xstar.tolist(), J=J.tolist(),
                    D=list(map(float, self.D)), L=float(self.L),
                    kstar=float(kg[i]), sigma_max=float(sig[i]),
                    coefficients=self.coefficients())

    def coefficients(self) -> dict:
        return {}


@RD_SYSTEMS.register("gierer_meinhardt")
class GiererMeinhardt(ReferenceSystem):
    """Saturated Gierer-Meinhardt (stripes). f = a + u^n/(v(1+K u^n)) - b u ; g = u^n - c v."""
    name = "gierer_meinhardt"

    def __init__(self, a=0.5, b=1.0, c=6.1, K=0.003, n=2, Du=1.0, Dv=100.0, L=100.0):
        self.a, self.b, self.c, self.K, self.n = a, b, c, K, n
        self.D = (Du, Dv); self.L = L

    def reaction(self, x):
        u, v = x
        a, b, c, K, n = self.a, self.b, self.c, self.K, self.n
        f = a + u**n / (v * (1 + K * u**n)) - b * u
        g = u**n - c * v
        return np.array([f, g])

    def steady_state(self):
        n, c, K, a, b = self.n, self.c, self.K, self.a, self.b
        u = fsolve(lambda u: a + c / (1 + K * u**n) - b * u, 6.0)[0]
        v = u**n / c
        return np.array([u, v])

    def coefficients(self):
        return dict(a=self.a, b=self.b, c=self.c, K=self.K, n=self.n)


@RD_SYSTEMS.register("schnakenberg")
class Schnakenberg(ReferenceSystem):
    """Schnakenberg (spots). f = gamma(a - u + u^2 v) ; g = gamma(b - u^2 v)."""
    name = "schnakenberg"

    def __init__(self, a=0.1, b=0.9, gamma=100.0, Du=1.0, Dv=40.0, L=100.0):
        self.a, self.b, self.gamma = a, b, gamma
        self.D = (Du, Dv); self.L = L

    def reaction(self, x):
        u, v = x; a, b, g = self.a, self.b, self.gamma
        return np.array([g * (a - u + u**2 * v), g * (b - u**2 * v)])

    def steady_state(self):
        u = self.a + self.b
        v = self.b / (self.a + self.b) ** 2
        return np.array([u, v])

    def coefficients(self):
        return dict(a=self.a, b=self.b, gamma=self.gamma)


def build_system(name: str, **kwargs) -> ReferenceSystem:
    return RD_SYSTEMS.get(name)(**kwargs)
