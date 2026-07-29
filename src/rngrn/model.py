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


def _low_basal_raw_params(N: int, g: "torch.Generator | None"):
    """Draw raw (unconstrained) theta for the low-basal init (ported from
    scripts/exp03_turing_first.py::low_basal_init, lines 45-59). Log-uniform ranges:
    beta in 1e-4..1e-2, s(binding budget) in 1e-2..10^-0.3, alpha in 10^0.3..10^1.5
    (~2..32), delta in 0.1..10^0.3, D ratio (species 1..N-1 vs species 0) in
    10^0.9..10^2.4. gate logit is a wide zero-mean normal (sub-saturated binding).
    Fixed numeric ranges only -- FIREWALL-safe, not derived from any target system.
    Draws from the given (possibly None) torch.Generator, matching RNGRN.__init__'s
    `randn` helper, so it is reproducible independently of global RNG state.

    Returns (theta_beta, theta_s, theta_alpha, theta_delta, theta_g, theta_D).
    """
    def loguniform(low, high, *shape):
        u = torch.empty(*shape).uniform_(low, high, generator=g)
        return 10.0 ** u

    def inv_softplus(x):
        return torch.log(torch.expm1(x))

    theta_beta = inv_softplus(loguniform(-4, -2, N))
    theta_s = inv_softplus(loguniform(-2, -0.3, N, N))
    theta_alpha = inv_softplus(loguniform(0.3, 1.5, N, N))
    theta_delta = inv_softplus(loguniform(-1, 0.3, N))
    theta_g = torch.empty(N, N).normal_(0.0, 2.5, generator=g)
    D = torch.ones(N)
    if N > 1:
        D[1:] = torch.sort(loguniform(0.9, 2.4, N - 1)).values
    theta_D = torch.log(D)
    return theta_beta, theta_s, theta_alpha, theta_delta, theta_g, theta_D


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
                 seed: int | None = None, dispersion_backend: str = "eig",
                 init: str = "default"):
        super().__init__()
        assert form in ("competitive", "nc1"), form
        assert dispersion_backend in ("eig", "cubic"), dispersion_backend
        assert init in ("default", "low_basal"), init
        self.N = int(N)
        self.form = form
        self.n_hill = int(n_hill)
        self.init = init
        # "eig": torch.linalg.eigvals, any N, the reference. "cubic": exact closed-form
        # roots, N=3 ONLY, 162x faster on CUDA (see _sigma_max_cubic). Default stays "eig"
        # so nothing silently changes; set "cubic" for GPU runs.
        # Rejected at CONSTRUCTION, not lazily at the first dispersion() call: a model that
        # can never evaluate its own dispersion is misconfigured the moment it is built, and
        # a run that only discovers that mid-optimisation has already wasted the budget.
        if dispersion_backend == "cubic" and int(N) != 3:
            raise ValueError(
                f"dispersion_backend='cubic' is exact for N=3 only (got N={N}). "
                "Use dispersion_backend='eig' for any other N.")
        self.dispersion_backend = dispersion_backend
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        def randn(*shape):
            return torch.randn(*shape, generator=g) if g is not None else torch.randn(*shape)

        if init == "low_basal":
            # FIREWALL-SAFE low-basal prior (docs/STATE_OF_THE_SCIENCE.md section 10):
            # measured 0/200 default-init Jacobian diagonals are positive vs 88/88 true
            # systems' diagonals, the root cause of 0/300 Turing-unstable inits. This
            # prior measured 82% Turing-unstable at init. Fixed numeric ranges only, not
            # derived from any target system. DEFAULT STAYS "default" (see config.py) --
            # adopting this changes which solutions recovery finds, so it must stay a
            # measured choice, not a silent bias.
            (self.theta_beta, self.theta_s, self.theta_alpha,
             self.theta_delta, self.theta_g, self.theta_D) = (
                nn.Parameter(t) for t in _low_basal_raw_params(N, g))
            return

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
        if self.dispersion_backend == "cubic":
            if self.N != 3:
                raise ValueError(
                    f"dispersion_backend='cubic' is exact for N=3 only (got N={self.N}). "
                    "N=4's closed form is numerically poor and N>=5 has none; use 'eig'.")
            return _sigma_max_cubic(M)
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


def _sigma_max_cubic(M: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    """max_i Re eig(M) for BATCHED 3x3 M, exactly, via closed-form cubic roots.

    Why this exists: torch.linalg.eigvals on batched small NON-SYMMETRIC matrices has no
    cuSOLVER batched kernel, costing ~700 us PER MATRIX on CUDA regardless of batch size
    (measured flat from batch 200 to 51200) versus ~1 us on CPU. That single call made a
    GPU training step ~5x slower than CPU. This routine is arithmetic only -- no eig, svd
    or linear solve -- so it maps onto GPU kernels: measured 162x faster than eigvals on
    CUDA, and numerically EXACT rather than approximate (validated against eigvals on 127
    real answer-key Jacobians: sigma_max MAE 9.2e-13, k* MAE 0, Turing verdict flips
    0/127, d sigma_max/dJ agreeing to 2e-16).

    The characteristic polynomial of a 3x3 is lam^3 - t lam^2 + c2 lam - c3, whose
    coefficients (trace, sum of principal 2x2 minors, determinant) are differentiable
    polynomials in M. Substituting lam = y + t/3 gives the depressed cubic y^3 + p y + q.
    Three real roots (discriminant > 0): the trigonometric solution, whose k=0 branch is
    the LARGEST root. One real root plus a complex-conjugate pair: Cardano, where the
    largest real part is max(y, -y/2) since the pair has real part -y/2.

    SIGN TRAP -- do not "simplify" this. In the trigonometric branch the argument is
    (3q)/(2p) * sqrt(-3/p) with p < 0. Substituting |p| for p yields a plausible-looking
    function that is wrong by MAE ~2e+2 and flips ~1 in 4 Turing verdicts. Two earlier
    implementations failed exactly this way; it was caught only by the equivalence test
    against eigvals (tests/test_dispersion_cubic.py). Re-run that test after any edit.

    A real shifted POWER ITERATION was tried first and rejected: 11/200 k-points on real
    data have a COMPLEX dominant eigenvalue, inside whose 2-D invariant subspace a real
    power iteration rotates instead of converging (49/127 verdicts flipped).
    """
    t = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    t2 = (M @ M).diagonal(dim1=-2, dim2=-1).sum(-1)
    c2 = 0.5 * (t * t - t2)
    c3 = torch.linalg.det(M)
    p = c2 - t * t / 3.0
    q = -2.0 * t ** 3 / 27.0 + t * c2 / 3.0 - c3
    disc = -(4.0 * p ** 3 + 27.0 * q ** 2)
    pneg = torch.clamp(-p, min=eps)
    r = torch.sqrt(pneg / 3.0)
    arg = torch.clamp((3.0 * q) / (2.0 * p.clamp(max=-eps)) * torch.sqrt(3.0 / pneg), -1.0, 1.0)
    y_three_real = 2.0 * r * torch.cos(torch.acos(arg) / 3.0)
    s = torch.sqrt(torch.clamp(q * q / 4.0 + p ** 3 / 27.0, min=0.0))
    cbrt = lambda z: torch.sign(z) * torch.pow(z.abs().clamp_min(eps), 1.0 / 3.0)
    y_one_real = cbrt(-q / 2.0 + s) + cbrt(-q / 2.0 - s)
    y = torch.where(disc > 0, y_three_real, torch.maximum(y_one_real, -0.5 * y_one_real))
    return y + t / 3.0


def build_model(cfg) -> RNGRN:
    """Construct a model from a ModelConfig (registry-dispatched by architecture)."""
    cls = MODELS.get(cfg.architecture)
    return cls(N=cfg.N, form=cfg.form, n_hill=cfg.n_hill, seed=cfg.seed,
               dispersion_backend=cfg.dispersion_backend,
               init=getattr(cfg, "init", "default"))
