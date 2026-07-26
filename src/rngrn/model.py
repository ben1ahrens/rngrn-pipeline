"""model.py — the RNGRN model core (VALIDATED science, repackaged as-is).

Gated-promoter differentiable reaction-diffusion model. Adapts Recurrent Neural
Chemical Reaction Networks (Dack 2024, arXiv:2406.03456) to spatial diffusion.

Design contract (rngrn_design_doc.md Sections 3-7, 12):
  * reaction f_i(x) is POINTWISE in space; the Laplacian is supplied externally.
  * all rates/weights/diffusivities strictly positive BY CONSTRUCTION.
  * gated binding: shared budget s and gate g in (0,1) -> KA=s*g, KR=s*(1-g).
  * two regulation forms (competitive Shea-Ackers, non-competitive NC-1),
    identical parameter shapes, selectable by name.
  * Jacobian by autodiff of the reaction (never hand-derived), graph retained so
    J is differentiable w.r.t. theta.
  * dispersion sigma(k)=max Re eig(J - k^2 D) via a GENERAL-N differentiable
    eigendecomposition; 2x2 closed form kept only as a self-test.

Nothing here reads any analytic quantity of a ground-truth system; init is random.
Parameter count: 3 N^2 + 3 N.

Reference validation (float64): KA+KR==s to 1e-6; autodiff J vs finite-diff to
1e-11; general-eig vs 2x2 dispersion agree to 0.01 away from the discriminant
crossing; gradients flow to theta_s and theta_D.

TUNING KNOBS for Claude Code (see TUNING.md): n_hill (Hill exponent, default 2),
the raw-parameter init scales in __init__ (they set where recovery starts).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .registry import Registry

MODELS: "Registry[type]" = Registry("models")

# Recovery is small (N=2 -> 18 params); float64 makes the autodiff Jacobian and the
# differentiable eigendecomposition numerically tight. Revisit for large N on GPU.
torch.set_default_dtype(torch.float64)


def _softplus(x):
    return torch.nn.functional.softplus(x)


@MODELS.register("gated_promoter")
class RNGRN(nn.Module):
    """Gene-regulatory reaction model with learnable, biologically-meaningful params.

    Parameters
    ----------
    N : int          number of genes (species).
    form : str       'competitive' or 'nc1'.
    n_hill : int     Hill exponent (default 2). Need not equal the truth's.
    seed : int|None  RNG seed for RANDOM raw-parameter init (no analytic leakage).
    """

    def __init__(self, N: int, form: str = "competitive", n_hill: int = 2,
                 seed: int | None = None):
        super().__init__()
        assert form in ("competitive", "nc1"), form
        self.N = int(N)
        self.form = form
        self.n_hill = int(n_hill)
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        def randn(*shape):
            return torch.randn(*shape, generator=g) if g is not None else torch.randn(*shape)

        # raw (unconstrained) parameters theta. Small random init keeps early KA*x^n moderate.
        # -- TUNING: these init scales set the recovery starting distribution.
        self.theta_s     = nn.Parameter(randn(N, N) * 0.5 - 1.0)   # binding budget (softplus)
        self.theta_g     = nn.Parameter(randn(N, N) * 0.5)         # gate logit (sigmoid)
        self.theta_alpha = nn.Parameter(randn(N, N) * 0.5)         # production weight (softplus)
        self.theta_delta = nn.Parameter(randn(N) * 0.3)            # degradation (softplus)
        self.theta_beta  = nn.Parameter(randn(N) * 0.3 - 1.0)      # basal (softplus)
        self.theta_D     = nn.Parameter(randn(N) * 0.5)            # diffusion (exp)

    # ---- device / dtype ------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Device the parameters live on. Callers that create tensors to feed this
        model (steady-state seeds, k-grids, latent fields) MUST place them here, or
        .to('cuda') silently breaks with a cross-device RuntimeError."""
        return self.theta_s.device

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype (float64 by default). The dispersion eigendecomposition and
        the Newton steady-state solve both rely on it; do not silently downcast."""
        return self.theta_s.dtype

    # ---- constrained physical parameters (all strictly positive) --------------------
    @property
    def s(self):     return _softplus(self.theta_s)
    @property
    def gate(self):  return torch.sigmoid(self.theta_g)
    @property
    def KA(self):    return self.s * self.gate
    @property
    def KR(self):    return self.s * (1.0 - self.gate)
    @property
    def alpha(self): return _softplus(self.theta_alpha)
    @property
    def delta(self): return _softplus(self.theta_delta)
    @property
    def beta(self):  return _softplus(self.theta_beta)
    @property
    def D(self):     return torch.exp(self.theta_D)

    # ---- the pointwise reaction f(x) -------------------------------------------------
    def reaction(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) for a batch of state vectors. x: (..., N) -> (..., N). Pointwise: no grid."""
        n = self.n_hill
        KA, KR, alpha, beta, delta = self.KA, self.KR, self.alpha, self.beta, self.delta
        xn = torch.clamp(x, min=0.0) ** n
        xj = xn.unsqueeze(-2)                                   # (..., 1, j)
        KAx = KA * xj
        KRx = KR * xj
        if self.form == "competitive":
            denom = 1.0 + (KAx + KRx).sum(dim=-1)              # (..., i)
            prod = (alpha * KAx).sum(dim=-1) / denom
        else:  # nc1: independent sites, additive activation * multiplicative repression veto
            thetaA = KAx / (1.0 + KAx)
            thetaR = KRx / (1.0 + KRx)
            activation = (alpha * thetaA).sum(dim=-1)
            veto = torch.prod(1.0 - thetaR, dim=-1)
            prod = activation * veto
        return beta + prod - delta * x

    def forward(self, x):
        return self.reaction(x)

    # ---- autodiff Jacobian at a single state x* -------------------------------------
    def jacobian(self, xstar: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """J_ij = d f_i / d x_j at xstar (shape (N,)). Form-agnostic (autodiff of the
        reaction). Retains graph so J is differentiable w.r.t. theta when create_graph."""
        xstar = xstar.reshape(self.N)
        J = torch.autograd.functional.jacobian(
            lambda xx: self.reaction(xx), xstar, create_graph=create_graph, vectorize=True)
        return J.reshape(self.N, self.N)

    # ---- differentiable dispersion relation (general N) -----------------------------
    def dispersion(self, xstar: torch.Tensor, kgrid: torch.Tensor,
                   J: torch.Tensor | None = None) -> torch.Tensor:
        """sigma(k) = max_i Re eig(J - k^2 diag(D)) over a grid of k. General-N,
        differentiable w.r.t. theta through both J and D."""
        if J is None:
            J = self.jacobian(xstar, create_graph=True)
        Dvec = self.D
        k2 = (kgrid ** 2).reshape(-1, 1, 1)
        M = J.unsqueeze(0) - k2 * torch.diag_embed(Dvec).unsqueeze(0)
        ev = torch.linalg.eigvals(M)
        return ev.real.max(dim=-1).values

    def dispersion_2x2(self, J: torch.Tensor, kgrid: torch.Tensor,
                       soft_sqrt_eps: float = 1e-8) -> torch.Tensor:
        """Closed-form 2-species sigma(k). SELF-TEST reference; smooth sqrt clamp keeps a
        usable gradient where the discriminant is negative (brief Section 12)."""
        assert self.N == 2
        Du, Dv = self.D[0], self.D[1]
        J11, J12, J21, J22 = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
        k2 = kgrid ** 2
        tr = (J11 + J22) - k2 * (Du + Dv)
        det = (J11 - k2 * Du) * (J22 - k2 * Dv) - J12 * J21
        disc = tr ** 2 - 4 * det
        disc_pos = torch.nn.functional.softplus(disc * 4.0) / 4.0
        return 0.5 * (tr + torch.sqrt(disc_pos + soft_sqrt_eps))


def build_model(cfg) -> RNGRN:
    """Construct a model from a ModelConfig (registry-dispatched by architecture)."""
    cls = MODELS.get(cfg.architecture)
    return cls(N=cfg.N, form=cfg.form, n_hill=cfg.n_hill, seed=cfg.seed)
