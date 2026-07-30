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
import math
import torch
import torch.nn as nn

from .registry import Registry

MODELS: "Registry[type]" = Registry("models")

# Recovery is small (N=2 -> 18 params); float64 makes the autodiff Jacobian and the
# differentiable eigendecomposition numerically tight. Revisit for large N on GPU.
torch.set_default_dtype(torch.float64)


def _softplus(x):
    return torch.nn.functional.softplus(x)


def _reaction_raw(KA, KR, alpha, beta, delta, x, form: str, n_hill: int):
    """The pointwise reaction f(x), as a PURE function of the CONSTRAINED parameters.

    Shared by RNGRN.reaction and BatchedRNGRN.reaction so there is exactly ONE copy of the
    arithmetic (unit b2). The op sequence is unchanged from the pre-b2 RNGRN.reaction body,
    so serial values are bit-identical; tests/test_batched.py::test_reaction_shared_kernel
    pins that.

    Broadcasting contract: (KA, KR, alpha) are (..., N, N), (beta, delta) are (..., N) and
    x is (..., N), with IDENTICAL leading dims. The serial model uses no leading dim; the
    batched model uses exactly one (B,). x carrying EXTRA leading dims relative to the
    parameters (e.g. per-pixel states against batched parameters) is NOT supported here --
    the caller must broadcast the parameters itself.
    """
    xn = torch.clamp(x, min=0.0) ** n_hill
    xj = xn.unsqueeze(-2)                                   # (..., 1, j)
    KAx = KA * xj
    KRx = KR * xj
    if form == "competitive":
        denom = 1.0 + (KAx + KRx).sum(dim=-1)              # (..., i)
        prod = (alpha * KAx).sum(dim=-1) / denom
    else:  # nc1: independent sites, additive activation * multiplicative repression veto
        thetaA = KAx / (1.0 + KAx)
        thetaR = KRx / (1.0 + KRx)
        activation = (alpha * thetaA).sum(dim=-1)
        veto = torch.prod(1.0 - thetaR, dim=-1)
        prod = activation * veto
    return beta + prod - delta * x


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
    kstar_obs : float|None  unit B4 (defect 2): when given and init='default', shifts the
        D init so it starts at median 1/kstar_obs**2 instead of median 1.0 -- see recover.py
        module docstring. Must be a positive finite wavenumber, IN WHATEVER UNITS THE CALLER
        RUNS THE OBJECTIVE (rad/length dimensional, rad/box nondim); it is used as-is, never
        rescaled by anything here (that is what keeps the fix L-free). Ignored when None
        (default) or when init='low_basal', which has its own D-ratio scheme.
    """

    def __init__(self, N: int, form: str = "competitive", n_hill: int = 2,
                 seed: int | None = None, dispersion_backend: str = "eig",
                 init: str = "default", kstar_obs: float | None = None):
        super().__init__()
        assert form in ("competitive", "nc1"), form
        assert dispersion_backend in ("eig", "cubic"), dispersion_backend
        assert init in ("default", "low_basal"), init
        if kstar_obs is not None and not (math.isfinite(kstar_obs) and kstar_obs > 0):
            raise ValueError(
                f"kstar_obs must be a positive finite wavenumber, got {kstar_obs!r}")
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
        theta_D = randn(N) * 0.5
        if kstar_obs is not None:
            # unit B4 (defect 2): shift so D = exp(theta_D) starts at median 1/kstar_obs**2
            # instead of median 1.0. kstar_obs is already in the objective's own units (the
            # caller computed it there), so this is L-free by construction on both the
            # dimensional and non-dimensional paths -- no L is read here.
            theta_D = theta_D - 2.0 * math.log(kstar_obs)
        self.theta_D     = nn.Parameter(theta_D)                   # diffusion (exp)

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
        return _reaction_raw(self.KA, self.KR, self.alpha, self.beta, self.delta, x,
                             self.form, self.n_hill)

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
    # cfg.seed is None by default ('derive from train.seed'); RNGRN treats seed=None as
    # 'use global RNG', which is the right behaviour for a caller that did not choose one.
    return cls(N=cfg.N, form=cfg.form, n_hill=cfg.n_hill, seed=cfg.seed,
               dispersion_backend=cfg.dispersion_backend,
               init=getattr(cfg, "init", "default"))


# ======================================================================================
# unit b2 — BATCHED model: B independent members advanced by ONE set of tensor ops
# ======================================================================================
THETA_NAMES = ("theta_s", "theta_g", "theta_alpha", "theta_delta", "theta_beta", "theta_D")


class BatchedRNGRN(nn.Module):
    """B INDEPENDENT RNGRN members carried as one leading batch dimension.

    WHY THIS EXISTS. Serially the GPU LOSES, because each step is a long chain of tiny (3,3)
    kernels whose launch overhead dominates: measured at N=3, K=400, resid off, cubic
    backend, float64, one Adam step of ONE restart costs 4.99 ms on CPU and 20.01 ms on CUDA
    -- CUDA is 4.0x SLOWER. Nothing about the maths is unfavourable; the batch dimension is
    missing. With a leading B the same arithmetic goes B-wide and the fixed per-kernel cost
    amortises. Measured cost per RESTART-step, same config (5 reps after 3 warmup):

        B         1      8     32    128    512
        serial-CPU     4.99   4.72   4.76   4.83     -
        batched-CPU    5.96   1.07   0.69   0.61   0.59
        batched-CUDA  24.47   3.60   0.84   0.33   0.16

    So batching alone buys 8.0x on CPU at B=128, and CUDA overtakes batched-CPU only between
    B=48 (0.61 vs 0.57, CPU still ahead) and B=64 (0.53 vs 0.58) -- call the crossover B~56.
    At B=512 CUDA is 3.8x batched-CPU and 30x serial-CPU. Below B~32 the GPU is not worth
    using at all, and at B=1 it is 4.9x worse than serial CPU.

    THE 'eig' BACKEND MUST NOT BE USED ON CUDA. Measured batched-CUDA with eig: 908 ms/step
    at B=1 and 6529 ms/step at B=8 -- 816 ms per restart-step, i.e. FLAT in B, confirming
    there is no batched cuSOLVER kernel for small non-symmetric eigenproblems. That is 2500x
    worse than the cubic backend at B=8. batched-CPU with eig is fine (0.85 ms/restart-step
    at B=32 vs 0.69 for cubic).

    PLAIN LEADING-DIM BROADCASTING, not torch.vmap over modules, for the parameters and the
    forward pass. Justification: every expression in `_reaction_raw` and in `dispersion` is
    already written with `...` leading dims and reduces over trailing axes, so a leading B
    needs ZERO change to the arithmetic and the serial values come out bit-identical (the
    equivalence tests measure 0.0 difference on the reaction). vmap is used for exactly one
    thing -- the per-member Jacobian -- because there the map is over an autodiff transform,
    not over an elementwise expression, and `torch.func.vmap(torch.func.jacrev(...))` is the
    only way to get B Jacobians for the price of N backward passes instead of B*N.

    FLOAT32 WAS MEASURED AND IT LOSES; the default dtype is deliberately unchanged. On this
    GPU (sm_120) float32 was expected to be faster. It is not: batched-CUDA at B=128 costs
    12.89 ms per restart-step in float32 against 0.29 in float64 (45x WORSE), and 3.07 vs
    0.16 at B=512. Two caveats keep this from being the last word -- the float32 arm had to
    monkeypatch torch.linalg.det out of _sigma_max_cubic to run at all (see below), and the
    module-level torch.set_default_dtype(torch.float64) stays in force, so float32 tensors
    meet float64 defaults throughout. A clean float32 measurement would require changing that
    global, which would alter every recorded number. The accuracy cost is also real: the same
    cubic dispersion in float32 differs from float64 by max abs 7.1e-3 / MAE 8.2e-6 on sigma
    (64 x 400 random 3x3), against a Turing hinge margin of 1e-3. Measured, unattractive on
    both axes, NOT adopted.

    AN UPSTREAM BUG BLOCKS FLOAT32 ENTIRELY, diagnosed precisely: torch 2.13.0+cu130's
    backward for torch.linalg.det builds its LU factor at the GLOBAL DEFAULT dtype rather
    than the input's, so with set_default_dtype(float64) any float32 backward through det
    raises "linalg.lu_solve: Expected LU and B to have the same dtype, but found LU of type
    Double and B of type Float". Setting the default to float32 makes the same call succeed.
    _sigma_max_cubic reaches det on every step, so float32 recovery is unrunnable today.
    Related and worth recording: _sigma_max_cubic's docstring claims it is "arithmetic only
    -- no eig, svd or linear solve", but torch.linalg.det IS an LU factorisation. Replacing
    it with the explicit 3x3 cofactor determinant was measured -- 36.63 vs 38.32 ms/step at
    B=128 CUDA and 73.34 vs 82.10 at B=512, a 4-11% gain, with max abs deviation 5.5e-10
    (MAE 3.3e-14, 0/64 Turing verdict flips) -- and REJECTED here: a few percent does not
    justify editing a validated function, and 5.5e-10 is not nothing. Left as a marked,
    measured option rather than a silent change.

    INDEPENDENCE IS THE CONTRACT. Members share no tensor except the (read-only) k-grid and
    frame. Every loss term reduces per member, so a sum over members has gradient exactly
    equal to each member's own gradient. Adam is elementwise, so one Adam over the stacked
    parameters IS B independent Adams -- provided gradient clipping is also per member
    (recover._clip_grad_norm_per_member; torch's clip_grad_norm_ would clip the JOINT norm
    and silently couple the members).

    Construct from serial members (`from_seeds`, or the constructor's list) so a batched run
    starts from EXACTLY the inits the serial run would have used, and pull a member back out
    with `member(b)` for scoring/reporting on the serial reference path.
    """

    def __init__(self, models: "list[RNGRN]"):
        super().__init__()
        if not models:
            raise ValueError("BatchedRNGRN needs at least one member model")
        m0 = models[0]
        for i, m in enumerate(models[1:], start=1):
            if (m.N, m.form, m.n_hill, m.dispersion_backend) != (
                    m0.N, m0.form, m0.n_hill, m0.dispersion_backend):
                raise ValueError(
                    f"BatchedRNGRN members must share (N, form, n_hill, dispersion_backend); "
                    f"member {i} has ({m.N}, {m.form!r}, {m.n_hill}, {m.dispersion_backend!r}) "
                    f"vs member 0's ({m0.N}, {m0.form!r}, {m0.n_hill}, {m0.dispersion_backend!r})")
        self.B = len(models)
        self.N = m0.N
        self.form = m0.form
        self.n_hill = m0.n_hill
        self.init = m0.init
        self.dispersion_backend = m0.dispersion_backend
        for name in THETA_NAMES:
            stacked = torch.stack([getattr(m, name).detach().clone() for m in models], dim=0)
            self.register_parameter(name, nn.Parameter(stacked))

    @classmethod
    def from_seeds(cls, N: int, seeds, form: str = "competitive", n_hill: int = 2,
                   dispersion_backend: str = "eig", init: str = "default",
                   kstar_obs: float | None = None) -> "BatchedRNGRN":
        """One member per entry of `seeds`, in order.

        Takes the seeds EXPLICITLY rather than a base seed plus an offset rule. The rule
        belongs to the caller (recover._restart_seed, unit B1), so the batched and serial
        paths cannot drift apart on which inits they start from. An earlier version took
        `seed0` and used `seed0 + r`, which is precisely the sliding-window scheme B1
        removed — run seed s and s+1 then shared B-1 of their B inits.
        """
        return cls([RNGRN(N=N, form=form, n_hill=n_hill, seed=s,
                          dispersion_backend=dispersion_backend, init=init,
                          kstar_obs=kstar_obs)
                    for s in seeds])

    # ---- device / dtype ------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.theta_s.device

    @property
    def dtype(self) -> torch.dtype:
        return self.theta_s.dtype

    # ---- constrained physical parameters, all with a leading B ----------------------
    @property
    def s(self):     return _softplus(self.theta_s)          # (B,N,N)
    @property
    def gate(self):  return torch.sigmoid(self.theta_g)      # (B,N,N)
    @property
    def KA(self):    return self.s * self.gate
    @property
    def KR(self):    return self.s * (1.0 - self.gate)
    @property
    def alpha(self): return _softplus(self.theta_alpha)      # (B,N,N)
    @property
    def delta(self): return _softplus(self.theta_delta)      # (B,N)
    @property
    def beta(self):  return _softplus(self.theta_beta)       # (B,N)
    @property
    def D(self):     return torch.exp(self.theta_D)          # (B,N)

    # ---- forward -------------------------------------------------------------------
    def reaction(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) per member. x: (B, N) -> (B, N). ONE state vector per member.

        Deliberately NOT (B, P, N): a per-pixel batch would need the parameters broadcast to
        (B, 1, N, N), which the shared kernel does not do. The only term that wants per-pixel
        states is the stationarity residual, which the batched path does not support (its
        weight is 0 by default; losses/total.compute_terms_batched raises if asked)."""
        if tuple(x.shape) != (self.B, self.N):
            raise ValueError(f"BatchedRNGRN.reaction expects x of shape ({self.B}, {self.N}), "
                             f"got {tuple(x.shape)}")
        return _reaction_raw(self.KA, self.KR, self.alpha, self.beta, self.delta, x,
                             self.form, self.n_hill)

    def forward(self, x):
        return self.reaction(x)

    # ---- per-member autodiff Jacobian ----------------------------------------------
    def jacobian(self, xstar: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """J[b]_ij = d f_i / d x_j for member b at xstar[b]. (B,N) -> (B,N,N).

        AUTODIFF, never hand-derived (the standing model.py contract), via
        vmap(jacrev(reaction)). Cost is N vmapped backward passes for the WHOLE batch, where
        B calls to `RNGRN.jacobian` would be B*N. Measured bit-identical (max abs diff 0.0)
        to the serial `torch.autograd.functional.jacobian` result.

        create_graph=False evaluates under no_grad, so the returned J is detached and no
        graph is built (verified: torch.func transforms work under no_grad). That is the
        Newton-solve path; create_graph=True keeps J differentiable w.r.t. theta.
        """
        x = xstar.reshape(self.B, self.N)
        form, n_hill = self.form, self.n_hill

        def f(kA, kR, al, be, de, xx):
            return _reaction_raw(kA, kR, al, be, de, xx, form, n_hill)

        jac_fn = torch.func.vmap(torch.func.jacrev(f, argnums=5))
        args = (self.KA, self.KR, self.alpha, self.beta, self.delta, x)
        if create_graph:
            J = jac_fn(*args)
        else:
            with torch.no_grad():
                J = jac_fn(*args)
        return J.reshape(self.B, self.N, self.N)

    # ---- per-member dispersion ------------------------------------------------------
    def dispersion(self, xstar: torch.Tensor, kgrid: torch.Tensor,
                   J: torch.Tensor | None = None) -> torch.Tensor:
        """sigma_b(k) = max_i Re eig(J[b] - k^2 diag(D[b])). (B,N) x (K,) -> (B,K).

        The 'cubic' backend is pure arithmetic and already batch-shaped, so it takes the
        (B,K,N,N) stack unchanged -- that is the whole point of it on GPU. 'eig' falls back
        to torch.linalg.eigvals, which HAS no batched cuSOLVER kernel for small
        non-symmetric matrices and therefore does not speed up with B (measured ~700 us per
        matrix flat from batch 200 to 51200). Batched + eig on CUDA is expected to be slow;
        use dispersion_backend='cubic' for N=3 GPU runs.
        """
        if J is None:
            J = self.jacobian(xstar, create_graph=True)
        k2 = (kgrid ** 2).reshape(1, -1, 1, 1)
        M = J.unsqueeze(1) - k2 * torch.diag_embed(self.D).unsqueeze(1)   # (B,K,N,N)
        if self.dispersion_backend == "cubic":
            if self.N != 3:
                raise ValueError(
                    f"dispersion_backend='cubic' is exact for N=3 only (got N={self.N}). "
                    "N=4's closed form is numerically poor and N>=5 has none; use 'eig'.")
            return _sigma_max_cubic(M)
        ev = torch.linalg.eigvals(M)
        return ev.real.max(dim=-1).values

    # ---- extraction -----------------------------------------------------------------
    def member(self, b: int) -> RNGRN:
        """Member b as a standalone serial RNGRN (parameters DETACHED and copied).

        Used to hand the winning restart back to the existing scoring/reporting code, which
        is written against RNGRN, so nothing downstream has to know a batched run happened.
        """
        if not (0 <= b < self.B):
            raise IndexError(f"member index {b} out of range for B={self.B}")
        m = RNGRN(N=self.N, form=self.form, n_hill=self.n_hill, seed=0,
                  dispersion_backend=self.dispersion_backend, init=self.init)
        with torch.no_grad():
            for name in THETA_NAMES:
                getattr(m, name).copy_(getattr(self, name)[b].detach())
        return m.to(self.device)
