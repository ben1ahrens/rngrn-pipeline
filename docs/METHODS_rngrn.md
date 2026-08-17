# Methods — recovering a mechanistic Turing GRN from a static concentration frame

> **Status of this draft.** This document describes the **ratified redesign** of the
> `rngrn-pipeline` recovery method (`docs/REDESIGN_rngrn.md`, RATIFIED 2026-08-17;
> rulings D-LIFT-1, D-REDESIGN-1, D-REDESIGN-2 in `docs/DECISIONS.md`;
> preregistration amendment §3.7 in `docs/PREREGISTRATION.md`). **Implementation is in
> progress.** Components are marked below as *implemented* (verified against the named
> source file), *specified* (fixed by the ratified design but not yet in the source), or
> *planned*. **No recovery results exist and none are claimed anywhere in this document.**
> Every threshold that the design marks UNCALIBRATED is marked UNCALIBRATED here; every
> number carries its provenance (source file, document section, decision ID, or run
> directory). Numbers quoted from prior diagnostics are quoted as *measurements of the
> instrument*, never as recovery outcomes.

---

## 0. Overview of the pipeline

The task is an inverse problem. A ground-truth three-gene reaction–diffusion system is
simulated to a saturated, spatially patterned steady state on a periodic box; one (or a
few) frames of that state are handed to the recovery method; the method must return a
*mechanistic* gene-regulatory network — every learned parameter a biological quantity —
whose own dynamics reproduce the observed pattern. Verification is by forward simulation
of the recovered parameters, ultimately after undoing the quasi-steady-state (QSS)
reduction that the model itself is built on.

The pipeline has five stages:

1. **Generation** (§1) — screened Turing systems, simulated by a spectral IMEX scheme to
   saturation on a commensurate periodic box.
2. **Observation** (§4) — a strictly image-derived observable set: the radially averaged
   power spectrum (RAPS), its sub-bin peak $k^\ast_\mathrm{obs}$, real-space moments, and
   angular statistics.
3. **Training** (§§2, 3, 5, 6, 7) — a population of mechanistic models is *ignited* by
   linear theory (the dispersion relation), then refined against a *simulated* spectrum
   obtained by solving the model's own patterned steady state, with gradients taken
   through that solve.
4. **Verification** (§8) — the recovered model is lifted out of the QSS reduction into the
   explicit finite-timescale promoter system and gated there.
5. **Evaluation** (§9) — the deliverable is an ensemble statistic, not a point estimate.

Two invariants hold throughout. **The data firewall**: nothing on the recovery side may
read a generating parameter, the true steady state, the true Jacobian, or the true
dispersion relation; this is enforced as a static AST audit over a declared list of
recovery-side modules (`tests/test_firewall.py`). **Evidence discipline**: a threshold is
a placeholder until something measured it on this data, and is labelled as such.

---

## 1. Synthetic data generation

### 1.1 The ground-truth system

Ground truth is a three-species reaction–diffusion system on a periodic square of side
$L$,

$$\frac{\partial x_i}{\partial t} \;=\; D_i \nabla^2 x_i \;+\; R_i(\mathbf{x}),
\qquad i = 1,2,3 ,$$

with Hill kinetics (`scripts/gen_tg3.py::make_reaction`):

$$R_i(\mathbf{x}) \;=\; b_i \;+\; V_i\, g_i(\mathbf{x}) \;-\; \mu_i x_i .$$

Here $b_i$ is basal production, $V_i$ the maximal regulated production, $\mu_i$ the
first-order degradation rate, and $g_i \in [0,1]$ the regulatory input of gene $i$. Note
that $\mu$ in the generator is the *degradation rate*; it is unrelated to the timescale
ratio $\mu$ of §8, and corresponds to the model's $\delta$ (§2).

The regulatory input is built from a signed interaction matrix $M \in \{-1,0,+1\}^{3\times 3}$.
For each regulator $j$ with $M_{ij} \neq 0$ a single Hill factor is formed,

$$g_{ij} \;=\;
\begin{cases}
\dfrac{x_j^{\,n}}{K_i^{\,n} + x_j^{\,n}}, & M_{ij} > 0 \quad\text{(activation)},\\[2ex]
\dfrac{K_i^{\,n}}{K_i^{\,n} + x_j^{\,n}}, & M_{ij} < 0 \quad\text{(repression)},
\end{cases}$$

and the factors of row $i$ are combined in one of two ways, selected per dataset by the
`--reaction` flag:

$$g_i \;=\;
\begin{cases}
\displaystyle\prod_{j:\,M_{ij}\neq 0} g_{ij}, & \text{\texttt{multiplicative} (the default and the form used for every canonical set)},\\[2ex]
\displaystyle\frac{1}{\deg_i}\sum_{j:\,M_{ij}\neq 0} g_{ij}, & \text{\texttt{additive} (sum-normalized)},
\end{cases}$$

with $\deg_i = \max\!\big(\#\{j : M_{ij}\neq 0\},\,1\big)$ and $g_i \equiv 1$ for an
unregulated row. Two properties of this form matter downstream. The dissociation constant
$K_i$ is **per target gene**, not per edge — the six kinetic thresholds of a row are one
number — and the Hill exponent $n$ is **global**. Neither is true of the recovery model
(§2), so the recovery model is not a superset of the generator: the recovery problem is
mis-specified by construction, which is deliberate and is the honest analogue of fitting a
mechanistic model to data whose true mechanism it does not exactly contain.

### 1.2 Topology families

Six signed topologies are enumerated (`gen_tg3.TOPOLOGIES`), each with a self-activating
diagonal on gene 1 and a distinct wiring of the remaining edges:

| name | $M$ (rows = target $i$, cols = regulator $j$) |
| --- | --- |
| `double_inhibitor` | $[[1,-1,-1],\,[1,0,0],\,[1,0,0]]$ |
| `relay_chain` | $[[1,-1,0],\,[0,0,1],\,[1,0,0]]$ |
| `cross_repress` | $[[1,-1,-1],\,[1,0,0],\,[1,-1,0]]$ |
| `selfinhib` | $[[1,-1,0],\,[1,-1,0],\,[1,0,-1]]$ |
| `activator_ring` | $[[1,0,-1],\,[-1,1,0],\,[0,-1,1]]$ |
| `substrate_depl` | $[[1,-1,0],\,[1,0,-1],\,[0,1,0]]$ |

### 1.3 Parameter priors

Kinetic parameters are drawn log-uniformly, independently per species
(`gen_tg3.screen_model`), from the exact ranges

$$b_i \sim 10^{\,U(-2,\,-0.3)},\quad
V_i \sim 10^{\,U(-0.3,\,1)},\quad
\mu_i \sim 10^{\,U(-0.4,\,0.7)},\quad
K_i \sim 10^{\,U(-1,\,0.6)},$$

with the Hill exponent drawn uniformly from $n \in \{2,3,4\}$ and the diffusivities set as

$$D = \big(1,\; 10^{\,U(0.9,\,2.4)},\; 10^{\,U(0.9,\,2.4)}\big),$$

i.e. species 1 is the unit of diffusivity and the other two are drawn independently over
roughly $8$–$250$, so the median largest-to-unit ratio $D_{\max}/D_{\min}$ is $\approx 91$
while the median ratio of the **two most mobile** species — the quantity
`configs/bio_box.yaml`, `losses/terms.py::param_prior` and §9.2 all use — is $\approx 2.8$
(both from a $4\times10^{6}$-draw simulation of the stated priors, recorded as D-REDESIGN-3).

The $\approx 135$ figure carried by `configs/bio_box.yaml`'s header comment,
`docs/PREREGISTRATION.md` §3.4 (as $134.9$) and `docs/BIO_VIABILITY.md` **names no
measurement and does not follow from these priors**: it is quoted, not verified, and the
statistic it refers to is unidentified. It is recorded as such under D-REDESIGN-3 and is not
used as a number in this document. What *is* measured on the D5 row is that the recovered
$d$-ratio (two most mobile) is in-box $10/10$ at prior weight $0$
(`plausibility_d_ratio_in_box_frac` $= 1.0$, `experiments/diag_fft/d5/target_reports.jsonl`).

The comparison that motivates the prior is therefore between the two-most-mobile ratio the
generator produces and the literature-derived plausibility box used as a prior in recovery
(`configs/bio_box.yaml`: $D$-ratio $\in [1,60]$, centre $7.5$). Centring the prior on
biologically viable literature values rather than on the synthetic population is an explicit
recorded owner decision, and §5.4 explains why it is kept as a *soft* prior rather than a
hard box.

### 1.4 The Turing screen

Each candidate parameter draw is screened numerically, never analytically
(`gen_tg3.screen_model`, `steady_states`, `jac_fd`, `turing_test`):

1. **Steady states.** Up to 10 calls to `scipy.optimize.fsolve` on $R(\mathbf{x}) = 0$ from
   independent starts $\mathbf{x}_0 \sim U(0.01, 5)^3$. A root is accepted only if
   `fsolve` reports `ier == 1`, every component satisfies $x_i > 10^{-6}$, and the
   residual satisfies $\max_i |R_i(\mathbf{x})| < 10^{-8}$; duplicates are removed at
   absolute tolerance $10^{-4}$.
2. **Jacobian.** A forward finite difference with step $\varepsilon = 10^{-6}$,
   $J_{:,j} = [R(\mathbf{x} + \varepsilon e_j) - R(\mathbf{x})]/\varepsilon$.
3. **Strict dispersion test.** With $\sigma(k) = \max_i \operatorname{Re}
   \operatorname{eig}\!\big(J - k^2 \operatorname{diag}(D)\big)$ evaluated on
   $k \in \mathrm{linspace}(10^{-3},\, 3.5,\, 350)$, the candidate is Turing if
   $$\max_i \operatorname{Re}\operatorname{eig}(J) < -10^{-9}
   \quad\text{and}\quad
   \sigma(k^\ast) > 10^{-6}
   \quad\text{and}\quad k^\ast > k_1 ,$$
   where $k^\ast$ is the grid argmax and $k_1$ the second grid point. The final clause is
   what excludes a uniformly unstable system whose "peak" sits at the grid floor — the
   same defect that D-EVID-11 later removed from the recovery-side verdict (§3.2).
4. **Band acceptance.** Additionally $0.08 < k^\ast < 3.0$ and $\sigma(k^\ast) > 0.01$, so
   the screened population is patterned at a resolvable wavelength and unstable at a rate
   that saturates in a tractable horizon.

### 1.5 Box size, and why periods-per-box is *drawn*

The domain is chosen commensurate with the linear wavelength,

$$L \;=\; p \cdot \lambda^\ast, \qquad \lambda^\ast = \frac{2\pi}{k^\ast},$$

with $p$ an integer *drawn* from a feasible set rather than fixed. This is a correction to
the original generator and is the single most important property of the dataset. The
original set $L = \mathrm{clip}(6\lambda^\ast, 18, 220)$, i.e. exactly six periods per box.
The measured consequence, recorded in `gen_tg3.py`'s module docstring: $k^\ast$ equals
$6\cdot 2\pi/L$ identically for **94.8 % of the 287 registered samples**, so an
*image-blind* predictor that reads only $L$ achieves a median relative error of
$1.4\times10^{-14}\,\%$ on $k^\ast$. Any method trained on that data could score perfectly
on wavelength without looking at the image, and there is no $L$ axis to generalise across,
because every sample is the same dimensionless system at a different physical scale.

The fix is structural: $p$ is drawn per system from `PERIODS_CHOICES` $= \{3,\dots,14\}$
(legacy sets) restricted to those $p$ for which $L = p\lambda^\ast$ lands inside the
acceptance window, and candidates whose feasible set is empty are **rejected rather than
clipped** — a clipped $L$ sits on a bound where the relation collapses again. The
`multiL` mode simulates the *same* screened system at $p \in \{4,7,10,13\}$ with a shared
initial-condition seed, so that any recovery difference within a replicate group is
attributable to $L$ alone; incomplete groups are dropped.

### 1.6 Integration to the saturated steady state

Simulation is a **spectral IMEX** scheme on an $n \times n$ periodic grid
(`gen_tg3.simulate_and_classify`): explicit reaction, implicit (exactly solved) diffusion
in Fourier space. One step is

$$\hat{X}^{\,r}_i = \mathcal{F}\big[\,X_i + \Delta t\, R_i(X)\,\big], \qquad
X_i \leftarrow \max\!\Big(0,\;\mathcal{F}^{-1}\Big[\frac{\hat{X}^{\,r}_i}{1 + \Delta t\, D_i |k|^2}\Big]\Big),$$

with $|k|^2 = k_x^2 + k_y^2$ built from $k = 2\pi\,\mathrm{fftfreq}(n, \Delta x)$,
$\Delta x = L/n$, and $\Delta t = 0.02/\max(\max_i \mu_i, 1)$. There is no
finite-difference Laplacian anywhere in the generator. The initial condition is the
homogeneous state perturbed multiplicatively,
$X_i = x^\ast_i\,(1 + 0.01\,\xi)$ with $\xi \sim \mathcal{N}(0,1)$ i.i.d. per pixel. A run
that produces a non-finite field, or whose final species-0 spatial coefficient of variation
falls below $0.05$, is discarded as collapsed.

For the **canonical 512² sets** used by the redesign (`scripts/canon_generate.py`) three
things change and are recorded there: the grid is $512^2$; the horizon is
$T_{\max} = 500$ non-dimensional time units, escalating up to $2000$; and a **saturation
gate** is enforced — the species-0 coefficient of variation is traced every 200 steps and
the frame is rejected unless it has moved by less than $1\,\%$ over the final $20\,\%$ of
the run (`canon_generate.SATURATION_TOL = 0.01`). This gate exists because recovery solves
$f(\mathbf{x}^\ast) = 0$ and never integrates time, so a transient frame is silently outside
the model's assumptions. The $1\,\%$ carries the script's own tag: **UNCALIBRATED in the
strict sense** — it is a *convergence* tolerance, not a threshold separating two
populations, no control run has been made against it, and it is enforced fail-loud rather
than used to judge anything.
Resolution is guarded in pixels-per-wavelength, $12 \le n/p \le 64$, not by an absolute
$L$ window: $L$ enters the physics only as a unit. The redesign's target sample,
`turing_labyrinth/sample_0000`, is a competitive-form $N{=}3$ system at $512^2$ with
$p = 8$, $L = 185.006$ (the value in the payload attributes; $185.01$ where earlier
documents round it), spatial $\mathrm{cv} = 0.81$ (`docs/REDESIGN_rngrn.md` §1).

**Provenance of that sample is recorded as incomplete.** Its manifest records
`provenance.source` as *"manually placed; indexed by scan"*, so the generating invocation and
seed are **unrecorded** for it — the failure mode `CLAUDE.md` §6a exists to prevent. It is
therefore not established that it came through the `canon_generate.py` path described above;
indeed that script's own comment reports a measured canonical $L$ range of
$245.8$–$794.0$, which excludes $L = 185.006$. The frame's properties are measured from the
payload; its generating command is not on record and is not reconstructed here.

### 1.7 The redesign's data protocol

Ratified as register item 4 (`docs/REDESIGN_rngrn.md` §4.6, §8). *Specified; the
multi-realization emission is a generator-side change not yet implemented.*

**Training frames — few and large.** $N = 8$ independent *realization* frames of the same
target system: identical generating parameters, independent initial-condition noise seeds
(`gen_tg3` already draws a per-sample `sim_seed`), each integrated to saturation at $512^2$
with $p = 8$. The realizations are split **6 train / 2 held out**. All eight are
observation-side, so this is firewall-legal. The six training realizations supply three
things the single-frame protocol cannot: a direct per-bin variance estimate for the
spectral targets, hence inverse-variance weighting in the shape loss (§5.5); a
$\sqrt{R}$-tighter averaged target; and a matched population for the held-out pair. The
two held-out realizations are the **realization-consistency check** — the fitted model's
spectra must match frames it never saw as well as it matches the frames it fitted.

Spatial splits cannot do this job. A saturated Turing pattern is statistically homogeneous,
so two halves of one frame are two samples of the same realization and carry no independent
evidence; independent initial conditions do (`docs/SPEC_fourier_training.md` §1). Read
honestly, a pass here says the fit captured the *system's statistics* rather than one
realization's defect layout. It is not evidence of recovery. $N=8$ and the 6/2 split are
starting choices and are **UNCALIBRATED**; the measured per-bin variance calibrates both
(register item 14).

**Validation frames — many and small, held out entirely.** The same target system is
additionally generated at small boxes, $p \in \{2,3,4\}$, $S$ seeds each ($S$
**UNCALIBRATED**, starting at 8), at $64^2$–$128^2$ so that pixels-per-wavelength stays
above the measured $6\ \mathrm{px}/\lambda$ breakdown floor. These are **never trained on**,
for a reason that follows directly from the estimator geometry.

**Why small boxes are validation assets: mode quantisation and bin width.** On a periodic
square of side $L$ the admissible wavevectors form a lattice
$\mathbf{k} = (2\pi/L)(m_1, m_2)$, $m_i \in \mathbb{Z}$, so the realisable radial
wavenumbers are $|k| = (2\pi/L)\sqrt{m_1^2 + m_2^2}$. The RAPS bins the 2-D power onto
exactly this lattice spacing: `observables.raps` uses bin width $\Delta k = 2\pi/L$, the
fundamental. Substituting the commensurate box $L = p\,\lambda^\ast = 2\pi p/k^\ast$ gives

$$\Delta k \;=\; \frac{2\pi}{L} \;=\; \frac{k^\ast}{p} ,$$

so **the RAPS bin width is $k^\ast/p$ and the fundamental sits in bin $p$**. Two
consequences follow. First, the *fitted* band $B_\mathrm{train} = [0.60, 1.55]\,k^\ast$
(§4.4) spans $0.95\,k^\ast$, i.e. $0.95\,p$ bins — about $7.6$ bins at the nominal $p = 8$ but
only $1.9$ bins at $p = 2$ and $2.9$ at $p = 3$. (On the target itself D3 *measured* $8$ bins
carrying $\ge 1\,\%$ of the peak in $[0.60,1.55]\,k^\ast$, consistent with the $8.36$ periods
the box actually holds rather than the nominal $8$; `docs/DIAGNOSTICS_fft.md` D3.)
At $p \in \{2,3,4\}$ the fitted band holds
only two to four numbers, so there is almost nothing there to fit and a great deal there to
falsify. Second, the realised pattern must snap to the lattice, and at small $p$ that
lattice is coarse: which mode wins is then dominated by the box rather than by the
kinetics. The canonical-dataset work measured exactly this regime as artifact-dominated
(D-CANON-2: stripe morphology is largely a small-box artifact, vanishing at $p \ge 11$).

Box dominance is a liability for training and an asset for validation, twice over:

1. **Cross-$L$ transfer.** The *recovered* model is simulated at the same small boxes with
   no refit, and its physical $k^\ast$ invariance and periods-per-box slope are compared
   with the generator's own held-out small-box realizations. Because the training k-grid is
   continuous (§3.3) while the box lattice is discrete, this is a genuine stress test
   rather than a restatement of the fit — the argument `docs/PREREGISTRATION.md` §3.5a
   relies on.
2. **Mode-selection distribution** (reported, never gated). Across $S$ seeds at small $p$
   the generator produces a distribution over which lattice mode wins; the recovered model,
   run identically, should reproduce it. This is sensitive to both the shape of the
   dispersion peak and to nonlinear saturation, but it is a **novel, unvalidated
   estimator** and therefore enters report-only.

---

## 2. The RNGRN model

*Implemented: `src/rngrn/model.py` (the mechanistic core, unchanged by the redesign) —
including the $(s,g)$ parameterization of §2.3 (`model.py:203–210`). §2.4 is analysis of the
implemented model, not a component. Specified: only §2.5 (fixed-point pinning), plus §2.6's
box-sigmoid reparameterization.*

**Antecedent.** The differentiable gated-promoter core is not new to this work: `model.py`
records at its head that it **adapts Recurrent Neural Chemical Reaction Networks** (Dack 2024,
arXiv:2406.03456) *to spatial diffusion*. The reaction forms of §2.2 and the parameterization
of §2.3 are carried over from that antecedent; what is added here is the diffusive part, the
inverse problem on a static frame, and the constraints of §2.5–§2.6.

### 2.1 The reaction–diffusion system

Recovery fits the same class of object it is asked to explain:

$$\frac{\partial \mathbf{x}}{\partial t} \;=\; \mathbf{D}\,\nabla^2 \mathbf{x} \;+\; \mathbf{f}(\mathbf{x}),
\qquad \mathbf{D} = \operatorname{diag}(D_1,\dots,D_N).$$

The reaction $\mathbf{f}$ is **pointwise in space**; the spatial operator never appears in
the training objective except through its Fourier eigenvalues (§3). The default dtype is
`float64` throughout, so that the autodiff Jacobian and the eigendecomposition are tight at
the small $N$ this model runs at.

### 2.2 The gated-promoter reaction, both forms

Let $u_{ij} = K^{A}_{ij}\,x_j^{\,n}$ and $v_{ij} = K^{R}_{ij}\,x_j^{\,n}$, where $n$ is the
Hill exponent (default $n = 2$; it need not equal the generator's). Then
(`model.py::_reaction_raw`), for the **competitive** (Shea–Ackers, shared-occupancy) form

$$f_i(\mathbf{x}) \;=\; \beta_i \;+\; \frac{\displaystyle\sum_{j} \alpha_{ij}\,K^{A}_{ij}\,x_j^{\,n}}
{\,1 + \displaystyle\sum_{j}\big(K^{A}_{ij} + K^{R}_{ij}\big) x_j^{\,n}\,}
\;-\; \delta_i x_i ,$$

and for the **nc1** (independent sites; additive activation, multiplicative repressive
veto) form

$$f_i(\mathbf{x}) \;=\; \beta_i \;+\;
\underbrace{\left(\sum_{j} \alpha_{ij}\,\frac{u_{ij}}{1 + u_{ij}}\right)}_{\text{activation}}
\cdot
\underbrace{\prod_{j}\left(1 - \frac{v_{ij}}{1 + v_{ij}}\right)}_{\text{veto}}
\;-\; \delta_i x_i ,$$

where the veto factor simplifies to $\prod_j (1 + v_{ij})^{-1}$. In both forms
$x_j$ is clamped at $0$ before exponentiation. The competitive denominator is a single
shared partition function per promoter — all regulators of gene $i$ compete for one binding
site — whereas nc1 gives each regulator its own independent site. The two forms have
identical parameter shapes and are selected by name, so a run can be repeated on either
without touching anything else.

The mechanistic reading of the two blocks is what makes every parameter a biological
quantity: $\beta_i$ is basal (leak) production, $\alpha_{ij}$ the transcriptional
contribution of a bound activator, $K^{A}_{ij}$ and $K^{R}_{ij}$ association constants for
activating and repressing binding, $\delta_i$ first-order degradation, $D_i$ diffusivity.

### 2.3 Parameterization, and the shared binding budget

Every physical parameter is a link function of an unconstrained tensor, so positivity holds
by construction and no constrained optimizer is needed:

$$s = \operatorname{softplus}(\theta_s),\qquad
g = \operatorname{sigmoid}(\theta_g),\qquad
K^{A} = s\odot g,\qquad
K^{R} = s\odot(1-g),$$

$$\alpha = \operatorname{softplus}(\theta_\alpha),\qquad
\delta = \operatorname{softplus}(\theta_\delta),\qquad
\beta = \operatorname{softplus}(\theta_\beta),\qquad
D = \exp(\theta_D).$$

The $(s, g)$ pair is the design's central modelling choice for the sign structure. Because
$K^{A}_{ij} + K^{R}_{ij} = s_{ij}$ **exactly**, $s_{ij}$ is the *total binding budget* of
edge $(i,j)$ and $g_{ij} \in (0,1)$ is the fraction of it spent on activation. The sign of
an edge is then $\operatorname{sign}(g_{ij} - \tfrac12)$ and its strength is $s_{ij}$:
sign and magnitude are decoupled, an edge can move continuously between activation and
repression without passing through a discontinuity, and a weak edge is a small $s$ rather
than a cancellation of two large numbers. This is what makes it possible to keep the sign
structure *learnable* — a standing owner constraint on this project, so no edge is masked
and no topology enumeration is performed.

Parameter count: $\theta_s, \theta_g, \theta_\alpha$ are $N \times N$ and
$\theta_\delta, \theta_\beta, \theta_D$ are length $N$, giving

$$\#\theta \;=\; 3N^2 + 3N \;=\; 36 \quad\text{at } N = 3 .$$

Against this, a single frame supplies essentially two scalars ($k^\ast_\mathrm{obs}$ and
the frame mean) plus priors — the data-starvation diagnosis of
`docs/SPEC_fourier_training.md` §2, and the reason the redesign moves as many constraints
as possible from the loss into the construction.

### 2.4 The gauge group, and why it is fixed by construction

The objective a $\sigma(k)$-only method minimises is a function of the dispersion relation
alone, and $\sigma(k)$ is exactly invariant under a group $G$ generated by:

- **transpose**, $J \mapsto J^{\mathsf T}$, since $\operatorname{spec}(M^{\mathsf T}) = \operatorname{spec}(M)$
  and $\operatorname{diag}(D)$ is symmetric;
- **diagonal similarity** (the concentration gauge), $J \mapsto S J S^{-1}$ with
  $S = \operatorname{diag}(s_1,\dots,s_N)$, which commutes with $\operatorname{diag}(D)$
  and therefore leaves $J - k^2\operatorname{diag}(D)$ conjugate to itself for every $k$.
  Overall scale acts trivially, so this is an $(N-1)$-parameter continuous family —
  **2 parameters at $N=3$**;
- **node permutation**, $J \mapsto P J P^{\mathsf T}$ with $D \mapsto P D$;
- **time scaling**, $(\mathbf f, \mathbf D) \mapsto (c\,\mathbf f, c\,\mathbf D)$ for
  $c > 0$, under which $\sigma(k) \mapsto c\,\sigma(k)$: the *sign* of $\sigma$ and the
  location of its argmax $k^\ast$ are unchanged, so a static frame cannot see $c$ at all.

`docs/DECISIONS.md` **D-C1-GAUGE** records both the derivation and a decisive negative
result: quotienting the reproducibility statistic by this entire group after the fact
rescues it only from $0.125$ to $0.375$ against a $0.75$ bar. The measured sign-structure
multiplicity is therefore **not** gauge-orbit multiplicity — independent seeds found
genuinely different networks. Post-hoc canonicalization cannot fix that, which is why the
redesign removes the gauges at the point where they enter the model.

### 2.5 Fixed-point pinning: solving $\beta$ out

*Specified (`docs/REDESIGN_rngrn.md` §3.2, register item 7); not yet implemented — `model.py`
still carries $\beta = \operatorname{softplus}(\theta_\beta)$ as a free parameter.*

The homogeneous steady state of the model satisfies $f_i(\mathbf{x}^\ast) = 0$, i.e.

$$\beta_i \;+\; \mathrm{prod}_i(\mathbf{x}^\ast) \;-\; \delta_i x^\ast_i \;=\; 0
\qquad\Longleftrightarrow\qquad
\boxed{\;\beta_i \;=\; \delta_i\,x^\ast_i \;-\; \mathrm{prod}_i(\mathbf{x}^\ast)\;}$$

where $\mathrm{prod}_i$ is the regulated production block of §2.2 evaluated at
$\mathbf{x}^\ast$. Rather than learning $\beta$ and then *solving* for $\mathbf{x}^\ast$,
the redesign **pins $\mathbf{x}^\ast$ to the data** — $x^\ast_i \equiv$ the mean of observed
channel $i$; $x^\ast_i \equiv 1$ by convention for any hidden channel — and reads $\beta$
off the equation above. Mechanistically $\beta$ is a basal production rate and cannot be
negative, so the constraint $\beta_i \ge 0$ is imposed as a smooth hinge (§5.3) and
violators are rejected at gate time. This is a mechanistic constraint, not a tuning knob.

Four things happen at once, and they are the reason for the change:

1. the continuous concentration gauge is gone — all $N$ scales are pinned, not just their
   ratios;
2. the amplitude observable becomes a **hard constraint** rather than the soft
   `frame_scale_anchor` loss term it currently is;
3. $N$ parameters are eliminated, $36 \to 33$ at $N=3$;
4. the per-step damped Newton solve for $\mathbf{x}^\ast$ leaves the training loop
   entirely. That solve is the dominant per-step cost, a convergence-fragility surface, and
   — because a flatter $\sigma$ makes it easier — part of the escape route the degenerate
   $k^\ast$ minimum exploits (§5.2).

**What this can and cannot claim is bounded by a measurement.** The estimator
$\hat{x}^\ast_0 = \text{frame.mean()}$ tracks the true $x^\ast_0$ at Spearman $\rho = 0.95$
but with ratio median $0.921$ (IQR $0.796$–$1.036$, $n = 127$ `three_gene` samples; exp12,
`docs/STATE_OF_THE_SCIENCE.md` §2.8), which `TUNING.md` records as making it "a scale
regulariser, **not a fit**". Pinning to it therefore imports a systematic $\sim 8\,\%$
scale error into the model's fixed point, measured on a *different dataset family* than the
target. The estimator's bias on `turing_labyrinth/sample_0000` is **UNMEASURED and must be
measured before the pinning is used**, or the $\beta$-hinge feasibility test and every
downstream rate inherit an uncharacterised offset.

### 2.6 Box-sigmoid reparameterization, and the remaining symmetries

*Specified (`docs/REDESIGN_rngrn.md` §3.3, register item 3).*

**$\alpha$ and $\delta$: hard box.** The softplus link is replaced by a box sigmoid,

$$\theta \;\longmapsto\; \ell + (h - \ell)\,\operatorname{sigmoid}(\theta),$$

with $(\ell, h)$ from `configs/bio_box.yaml`: $\alpha \in [0.5, 10]$, $\delta \in [0.4, 5]$,
whose provenance is `docs/GOAL_tica_equivalent.md` §2.2 — the project's own recorded working
box, not an external wet-lab measurement, and the document must keep saying so. The
consequence must be stated in the same breath as the change: **plausibility for $\alpha$
and $\delta$ becomes $1.0$ by construction, is therefore not a result, and may never be
reported against the baseline's measured $0/10$.** Those columns are reported as
`structural`. This converts a preregistered measured outcome (`docs/PREREGISTRATION.md`
§3.4) into a structural truth, and is recorded as such.

**Time gauge.** Fixed by convention, $\operatorname{geomean}(\delta) \equiv 1$, and all
reported rates are restricted to time-scale-invariant combinations
(`docs/SPEC_fourier_training.md` §1). The invariance is not assumed: it was verified
numerically at $1.7\times10^{-12}$ analytically and $2.5\times10^{-13}$ dynamically, with the
invariant combinations unchanged to $\le 2\times10^{-16}$ (`docs/DIAGNOSTICS_fft.md` D6,
Check B). **The caveat recorded at that measurement is carried here rather than dropped:**
scaling $\mathbf f$ by $c$ *is not exactly expressible inside `RNGRN`'s softplus
parameterisation*, so the analytic check builds $c\,(J - k^2 D)$ directly; the $D$-half of the
map, which *is* expressible (`theta_D += log c`), matches to $3.2\times10^{-12}$. The verified
statement is therefore about the dispersion relation under that construction, not about a
parameter-space orbit the model can traverse. The only data that could pin absolute rates is a *transient* pair — one frame at
roughly half saturation plus the final frame, whose band-power growth estimates
$\sigma(k^\ast)$ in absolute units. Two *steady-state* frames $\mathrm{d}t$ apart carry zero
rate information. That protocol is designed and deliberately deferred; no current gate needs
absolute rates.

**Permutation.** Observed channels carry their data identity, so no permutation freedom
remains among them; in later hidden-species arms the unobserved species are ordered by
descending $D$.

**Transpose.** This is *not* a symmetry of the mechanistic model — there is in general no
parameter map realising $J^{\mathsf T}$ — only of $\sigma(k)$-based objectives. It is
therefore broken **objective-side**, by loss terms that depend on eigenvectors and on
nonlinear saturation, i.e. the simulated-spectrum terms of §5.5. Any run trained on
$\sigma$-only losses must report its transpose partner as an explicit unresolved two-fold
ambiguity.

---

## 3. Linear theory: dispersion and the Turing conditions

*Implemented: `model.py::dispersion`, `model.py::_sigma_max_cubic`,
`eval/analysis.py::turing_ok`, `recover._kgrid_for`.*

### 3.1 The dispersion relation

Linearising about the homogeneous fixed point and Fourier-transforming the Laplacian,
$\nabla^2 \to -|k|^2$, the growth rate of the mode of wavenumber $k$ is the leading
eigenvalue of the shifted Jacobian:

$$\sigma(k) \;=\; \max_i \operatorname{Re}\operatorname{eig}\!\Big(J - k^2 \operatorname{diag}(D)\Big),
\qquad J_{ij} = \frac{\partial f_i}{\partial x_j}\bigg|_{\mathbf{x}^\ast}.$$

$J$ is obtained by **autodiff of the reaction**, never hand-derived
(`model.jacobian`, `torch.autograd.functional.jacobian` with `create_graph=True`), so
$\sigma(k)$ is differentiable with respect to $\theta$ through both $J$ and $D$ and remains
correct if the reaction form changes. In the batched model the same object is obtained as
`vmap(jacrev(...))`, costing $N$ vmapped backward passes for the whole population instead of
$B\!\cdot\!N$. **Neither batched quantity is pinned to bit-identity**, and the two bars differ:
the batched *reaction* is held to round-off against the serial kernel (worst difference
$< 10^{-13}$) while the batched *Jacobian* is held to $< 10^{-12}$ against serial autodiff
(`tests/test_batched.py`).

**The $-k^2 D$ term *is* the Laplacian.** Training never discretises space and never
simulates a field to evaluate $\sigma$; there is no stencil, no grid, and therefore no
spatial discretisation error in this part of the objective.

### 3.2 The strict Turing conditions

A model is called Turing-unstable when both conditions hold strictly
(`eval/analysis.py::turing_ok`, corrected 2026-08-04 under D-EVID-11):

$$\textbf{(i) uniformly stable:}\quad \max_i \operatorname{Re}\operatorname{eig}(J) < 0,
\qquad\qquad
\textbf{(ii) structurally unstable:}\quad \max_{k>0}\sigma(k) > \mathrm{tol} = 10^{-9}.$$

Condition (i) was previously $\operatorname{tr}(J) < 0$, and the replacement is not
cosmetic. The trace is the *sum* of the eigenvalues, so one eigenvalue may be positive while
the sum is negative; and because the default scan grid begins at $k = 10^{-3}$, where
$\sigma(k) \approx \sigma(0)$, a merely uniformly unstable system also satisfied condition
(ii) *with its own uniform instability*, returning "Turing" with $k^\ast$ pinned to the grid
floor. Measured: of 398 converged low-basal draws at $N=3$, **206 (51.8 %) passed the loose
test and 0 (0.0 %) pass the strict one**, all 206 with $k^\ast$ exactly at the grid floor.
The loose verdict is retained alongside, explicitly labelled, so rows recorded under the old
criterion stay interpretable.

Turing-unstable and *patterns* remain different claims throughout this document, and are
never used as proxies for one another (§8, §9).

### 3.3 The training $k$-grid and mode quantisation

The dispersion is evaluated on a grid anchored to the observed wavenumber
(`recover._kgrid_for`):

$$k_{\min} = \frac{k^\ast_\mathrm{obs}}{50} + 10^{-3},
\qquad
k_{\max} = \max\!\big(8\,k^\ast_\mathrm{obs},\; 2 k_{\min}\big),
\qquad
K = 400 \text{ points, linearly spaced.}$$

The anchoring is firewall-legal: $k^\ast_\mathrm{obs}$ comes from an FFT of the observed
frame (§4.1) and from nothing else. The relative floor replaced an absolute $2.0$
rad/length floor which was not scale-free and silently dominated whenever
$k^\ast_\mathrm{obs} < 0.25$ — 11 of 287 samples, $3.8\,\%$ — pinning the grid to the wrong
band.

**The training grid is continuous; the box is not.** A periodic domain of size $L$ admits
only $|k| = (2\pi/L)\sqrt{m_1^2+m_2^2}$, whereas training hinges on 400 continuous
wavenumbers, including ones the domain cannot support. A model can therefore be
Turing-unstable *in training* and still fail to pattern in a rollout, because the realised
mode must snap to the admissible lattice. This is milder in 2-D than the 1-D picture
suggests — the set of $\sqrt{m_1^2+m_2^2}$ is denser than the set of $|m|$ — but it is
genuinely coarse at small $p$ (§1.7), and it is precisely why the verification stage
simulates rather than inspecting $\sigma$ (§8) and why $L$-transfer is a non-vacuous test.

$L$ itself enters only as a unit. On the non-dimensional path $L_\mathrm{model} = 1$, and
since $k \propto 1/L$ and the Laplacian $\propto 1/L^2$, the change of variables is
**exact**, not an approximation; only the units of $D$ and $k$ change. That is exactly why
the non-dimensional path may not claim credit for $L$-invariance
(`docs/PREREGISTRATION.md` §3.5b): it is $L$-invariant by construction.

### 3.4 The closed-form cubic backend at $N=3$

Evaluating $\sigma$ by `torch.linalg.eigvals` is the reference path but is unusable on GPU:
small non-symmetric eigenproblems have no batched cuSOLVER kernel, so the cost is
$\approx 700\,\mu\mathrm{s}$ **per matrix**, measured flat from batch 200 to 51 200
(`model.py::_sigma_max_cubic` docstring). For $N = 3$ the roots are available in closed
form, as arithmetic that maps onto GPU kernels — with one recorded exception: the determinant
is `torch.linalg.det` (`model.py:304`), which **is** an LU factorisation, so the routine is
not literally eig/solve-free. An explicit $3\times3$ cofactor determinant was measured
($4$–$11\,\%$ faster at $B \in \{128, 512\}$ on CUDA, max absolute deviation
$5.5\times10^{-10}$) and **deliberately not adopted**, on the recorded grounds that a few
percent does not justify editing a validated function (`model.py:389–393`).

The characteristic polynomial of $M = J - k^2\operatorname{diag}(D)$ is
$\lambda^3 - t\lambda^2 + c_2\lambda - c_3$ with

$$t = \operatorname{tr} M, \qquad
c_2 = \tfrac12\big(t^2 - \operatorname{tr}(M^2)\big), \qquad
c_3 = \det M ,$$

all differentiable polynomials in $M$. The substitution $\lambda = y + t/3$ depresses it to
$y^3 + py + q$ with

$$p = c_2 - \frac{t^2}{3}, \qquad q = -\frac{2t^3}{27} + \frac{t\,c_2}{3} - c_3,
\qquad \Delta = -\big(4p^3 + 27q^2\big).$$

For $\Delta > 0$ (three real roots) the trigonometric solution is used, whose $k=0$ branch
is the largest root,

$$y \;=\; 2\sqrt{-p/3}\;\cos\!\left(\frac{1}{3}\arccos\!\left[\frac{3q}{2p}\sqrt{\frac{-3}{p}}\right]\right),
\qquad p < 0 ;$$

for $\Delta \le 0$ (one real root plus a conjugate pair) Cardano's formula gives the real
root $y_1$, and since the conjugate pair has real part $-y_1/2$ the leading real part is
$\max(y_1, -y_1/2)$. In both branches $\sigma = y + t/3$.

The sign of $p$ in the trigonometric argument is load-bearing and is documented as a trap in
the source: substituting $|p|$ for $p$ yields a plausible function that is wrong by MAE
$\sim 2\times10^{2}$ and flips about one Turing verdict in four. Validation against
`eigvals` on 127 real Jacobians: $\sigma_{\max}$ MAE $9.2\times10^{-13}$, $k^\ast$ MAE $0$,
$0/127$ verdict flips, $\partial\sigma_{\max}/\partial J$ agreeing to $2\times10^{-16}$;
measured $162\times$ faster than `eigvals` on CUDA. A real shifted power iteration was tried
first and rejected: $11/200$ $k$-points on real data have a complex dominant eigenvalue,
inside whose 2-D invariant subspace a real power iteration rotates rather than converging
($49/127$ verdicts flipped). The backend is rejected at **construction** for $N \ne 3$, not
lazily at first use.

---

## 4. Observables and spectral estimators

*Implemented: `src/rngrn/observables.py` (measurement side, numpy),
`src/rngrn/losses/spectral.py` (differentiable, torch).*

Everything in this section is computed from the observed frame alone. No analytic quantity
of any ground-truth system is read.

### 4.1 The radially averaged power spectrum and the sub-bin $k^\ast$

For a single channel $u$ of an $n\times n$ frame on a periodic square of side $L$
(`observables.raps`):

$$\tilde{u} = u - \langle u\rangle, \qquad
P(\mathbf{k}) = \big|\mathcal{F}[\tilde u](\mathbf{k})\big|^2,\qquad
\mathbf{k} = 2\pi\,\mathrm{fftfreq}(n,\, L/n)\ \text{per axis},$$

$$\mathcal{P}(k_b) \;=\; \frac{1}{|\{\mathbf{k} \in b\}|}\sum_{\mathbf{k}\in b} P(\mathbf{k}),
\qquad
b = \big[\,b\,\Delta k,\ (b{+}1)\Delta k\,\big),\quad \Delta k = \frac{2\pi}{L},$$

with the DC bin zeroed. The bin width is the fundamental $2\pi/L$, which is what makes
$\Delta k = k^\ast/p$ on a commensurate box (§1.7). The peak wavenumber is a **power-weighted
centroid over the peak bin and its two neighbours on each side**, giving sub-bin precision:

$$b^\ast = \arg\max_b \mathcal{P}(b), \qquad
k^\ast_\mathrm{obs} \;=\;
\frac{\sum_{b \in [b^\ast-2,\, b^\ast+2]} k_b\,\mathcal{P}(b)}
{\sum_{b \in [b^\ast-2,\, b^\ast+2]} \mathcal{P}(b) + 10^{-12}},$$

clipped to exclude the DC bin, where $k_b$ is the **bin centre**
$\tfrac12\big(b\,\Delta k + (b{+}1)\Delta k\big)$ — `observables.py:49` builds
`kcent = 0.5 * (kbins[:-1] + kbins[1:])`, not the lower edge.

**What is and is not known about this estimator.** `observables.py`'s docstring records it as
reliable to $\sim 10$–$15\,\%$ and biased **high** off onset (the selected wavelength is not the
linear one), but that line is unchanged since the template commit and cites an
`observables_spec.md` that **does not exist in this repo**: it is therefore **UNCALIBRATED** and
corroborates only the *direction*. The one measurement on this corpus
(`docs/HANDOFF_training_data_plots.md` §3, 2026-08-10) gives median relative deviation
$8.3\,\%$ (90th percentile $22.7\,\%$), signed median $+4.6\,\%$, with $64\,\%$ of samples above
the generator's $k^\ast$ — a tendency high, not a uniform offset. That measurement compares
against **generating parameters** and may therefore not be used to calibrate any tolerance that
judges recovery.

### 4.2 Morphology features

Auxiliary, and report-only throughout this design (`observables.morphology`): intensity
skewness and excess kurtosis of the frame, plus two angular statistics of the dominant
Fourier ring. The ring is $\big|\,|\mathbf{k}|_{\text{px}} - k_d\big| < \max(1.5,\,0.15 k_d)$
with $k_d = k^\ast/2\pi$ in cycles-across-box; the power on the ring is histogrammed into
$n_\theta = 18$ bins over $\theta \in [0,\pi)$ and normalised to $\hat p$, giving

$$\text{ang\_conc} = \max_\theta \hat p_\theta,
\qquad
\text{ang\_entropy} = -\frac{1}{\log n_\theta}\sum_\theta \hat p_\theta \log \hat p_\theta .$$

A frame with parallel stripes concentrates the ring at one angle (high concentration, low
entropy); a labyrinth spreads it (low concentration, high entropy).

### 4.3 The differentiable twin, and the estimator-pair rule

Every spectral loss is a comparison between an *observed* and a *model-side* spectrum. The
governing rule is: **it is an estimator pair, or it is a bug.** Any estimator choice —
windowing, binning, normalisation — is applied identically on both sides, so its bias
cancels. `losses/spectral.py::raps_torch` is the differentiable twin of
`observables.raps`, with the binning geometry computed in numpy (it carries no gradient) and
only the power carrying gradient; its bin edges are built as `arange(n_edges) * dk` rather
than by a floating-point step, a deviation from the numpy reference recorded as finding
F-D6-1. `spectral_block_torch` extracts the central $n_\mathrm{blk} \times n_\mathrm{blk}$
block of the shifted 2-D power ($n_\mathrm{blk} = 24$) and normalises it as

$$B \;=\; \frac{\log(1 + P_\mathrm{blk}/\textstyle\sum P_\mathrm{blk})}{\sum \log(1 + P_\mathrm{blk}/\sum P_\mathrm{blk})},$$

normalising by the *block's own* power rather than the whole spectrum's, so that it matches
the scoring-side comparator exactly.

### 4.4 Bands: fitted and held out

All bands are defined **relative to $k^\ast_\mathrm{obs}$**, never in absolute units
(`docs/SPEC_fourier_training.md` §4; `losses/spectral.py::band_mask`):

$$B_\mathrm{train} = \big[0.60,\ 1.55\big]\cdot k^\ast_\mathrm{obs},
\qquad
B_\mathrm{low} = \big(0,\ 0.60\big)\cdot k^\ast_\mathrm{obs}\ \text{(DC excluded)},
\qquad
B_\mathrm{harm} = \big(1.55,\ 3.0\big]\cdot k^\ast_\mathrm{obs}.$$

The bracket convention matters and is stated because `band_mask` is **inclusive at both ends**:
written closed on both sides, $B_\mathrm{train}$ and $B_\mathrm{harm}$ would both contain a bin
centre falling exactly on $1.55\,k^\ast_\mathrm{obs}$, and the same at $0.60$. The held-out
bands are therefore written half-open above, as the convention this document uses; the shared
endpoints belong to $B_\mathrm{train}$.

The training band's edges are **closed by measurement**, not chosen: D-FFT-9 closure 1 fixed
$[0.60, 1.55]$ as the contiguous RAPS support of the target frame's channel 0 above $1\,\%$
of its peak, replacing the provisional $[0.5, 1.5]$ that the diagnostic tested. Spectra are
normalised *within* the training band,
$S(k) = \mathcal{P}(k)/\sum_{k'\in B_\mathrm{train}}\mathcal{P}(k')$, never by total power
and never by any statistic a held-out band could influence.

**Specified; not implemented.** `losses/spectral.py` carries only the *generic* `band_mask(k_centers,
kstar_obs, lo, hi)` and `SpectralConfig.b_lo = 0.60` / `b_hi = 1.55` (mirrored as
`LossConfig.spectral_b_lo`/`spectral_b_hi`). There are **no** $B_\mathrm{low}$/$B_\mathrm{harm}$
constants and **no gate-time consumer** of the held-out bands anywhere in `src/` today; both live
in `scripts/diag_fft_d3.py`, `scripts/diag_fft_d4.py` and `docs/SPEC_fourier_training.md` §9.5.

As specified, $B_\mathrm{low}\cup B_\mathrm{harm}$ are **held out**: never trained on, never used
for selection, consulted exactly once by scoring code at gate time. They carry the validation
content, and the reason is a measurement. D4 constructed alternative wirings that match the
fundamental $k^\ast$ and compared their spectra to the target's: the held-out-band
`spec_shape` distance separates them $47 \to 149$ (a factor $3.2$), and $19.6 \to 107.8$ on
$B_\mathrm{harm}$ alone (a factor $5.5$), against the $\sim 31\,\%$-per-bin estimation floor
that D3 measured from within-frame quadrant variance. In other words, matching the
fundamental is cheap and many mechanisms do it; matching the *harmonic content* of the
saturated pattern is what discriminates, and it is measurable above the noise floor at
$512^2$, $p=8$. This is the empirical basis for making simulated-spectrum terms primary
(§5.5).

### 4.5 The windowed estimator path

*Specified, report-only (`docs/REDESIGN_rngrn.md` §4.6; register item 15).*

The **unwindowed full-frame RAPS with the sub-bin centroid $k^\ast$ stays primary** for the
synthetic campaign. The generator's fields are periodic by construction, so the full-frame
FFT has zero leakage; windowing a periodic frame only convolves the spectrum with the window
transform — about a two-bin main lobe for a Hann window — buying nothing and costing peak
resolution. A **Hann–Welch** path (2-D Hann window, $\sim 50\,\%$-overlapping patches,
window-power normalised) is added *alongside* it with three jobs:

1. **Per-bin variance from within a single frame.** Cropped patches are not periodic, so
   unwindowed patch spectra leak — D3's $\sim 31\,\%$/bin floor was measured on plain
   quadrants. Windowed overlapping patches are *expected* to give roughly twice the effective
   number of independent spectral samples per unit frame area, with leakage suppressed,
   tightening the inverse-variance weights of §5.5. That factor is standard Welch-overlap
   accounting, **unmeasured here**, and is itself one of the windowed path's own calibration
   deliverables.
2. **Within-frame versus across-realization variance.** Comparing windowed-patch variance
   against the 6/2 realization split's across-frame variance answers whether realization
   noise is patch noise writ large (in which case fewer generated frames suffice) or whether
   there is genuine realization-level structure. This comparison is the calibration
   measurement for $N$.
3. **Real-data readiness.** Experimental images are cropped, non-periodic and
   boundary-contaminated; there, assuming periodicity is simply wrong and windowing is
   mandatory. Validating the windowed estimator against the unwindowed one on synthetic
   frames, where both are computable, is the on-ramp.

Because a windowed estimator changes what $k^\ast_\mathrm{obs}$ and every spectral target
*mean* — the sub-bin centroid and the one-bin bar are calibrated on the unwindowed
estimator — the windowed path is **report-only until calibrated against the unwindowed
baseline on synthetic frames**, and never silently replaces the primary.

---

## 5. The training objective

The objective has two regimes and an explicit switch between them. **Linear theory ignites**
— cheap, closed-form terms that drive an arbitrary initialisation into the Turing regime —
and **the simulated spectrum discriminates**, once a member is in that regime. The switch is
a measured event, not a schedule.

Throughout, the smooth one-sided hinge is
$h(v) = \operatorname{softplus}(\beta_h v)/\beta_h$ with $\beta_h = 10$
(`losses/terms.py::_softplus_hinge`); $v>0$ means the condition is violated.

### 5.1 Notation and the current baseline objective

The implemented objective (the control arm A0) is a weighted sum of unweighted term values,
with default weights `kstar` 1.0, `turing` 1.0, `resid` 0.0, `anticollapse` 0.5, `anchor`
2.0, `param_prior` 0.0, and all five spectral weights 0.0
(`losses/terms.py::DEFAULT_WEIGHTS`), run **with `lbfgs_steps=50` (LBFGS polish active, as in
all ten D5 frozen configs; §7.4, Appendix A1)**. Two of those zeros are settled, not merely
untuned: `resid` is **measured harmful** — exp06 swept pixel batch $\{64,128,512\}\times$ weight
$\{1,3,10\}$ at 8 seeds per cell and *all nine cells* collapsed to $1/8$ Turing seeds at
best-median $k^\ast$ error $11.8\,\%$, against $5/10$ Turing seeds and $0.4\,\%$ with the
term off; note the differing denominators (8 seeds per cell with the term on, 10 in the control),
which makes the two rates indicative rather than directly comparable. The
spectral weights are zero so that no default configuration triggers the expensive forward
solve.

### 5.2 The $k^\ast$ anchor, its degenerate minimum, and the scale-invariant replacement

**Implemented (`losses/terms.py::kstar_anchor`), retained for the A0 control arm.** With
$\operatorname{lse}$ the log-sum-exp over the $K$-point grid,

$$\mathcal{L}_{k^\ast} \;=\; \max\!\left(0,\;
\underbrace{\frac{1}{T}\log\sum_{i=1}^{K} e^{T\sigma(k_i)}}_{\text{smooth}\ \max_k \sigma}
\;-\; \sigma(k^\ast_\mathrm{obs})\right),
\qquad T = 60 ,$$

with $\sigma(k^\ast_\mathrm{obs})$ obtained by linear interpolation between the two
bracketing grid points. The term is non-negative and vanishes exactly when
$k^\ast_\mathrm{obs}$ is the argmax of $\sigma$. The temperature $T = 60$ was inherited and
has **never been swept**.

**The degenerate minimum.** Before a model is Turing-unstable, $\sigma$ peaks at $k = 0$, so
$\operatorname{lse}(T\sigma)/T \approx \sigma(0)$ and the loss is
$\approx \sigma(0) - \sigma(k^\ast_\mathrm{obs}) \ge 0$. That difference can be reduced *by
flattening $\sigma$* — shrinking $\|J\|$ and the $D$-separation until the whole curve is
level — instead of by relocating its peak. Flattening is a strictly easier descent direction
than moving the argmax, and it destroys precisely the structure the term is supposed to
build. This is the measured failure recorded as exp09 in
`docs/STATE_OF_THE_SCIENCE.md` §2.4: held-out generalisation of $2/6$ samples ever
succeeding, $0/8$ seeds on the other four, and the best-anchored *failing* sample still
failing $0/8$.

**`kstar_si` — the scale-invariant replacement.** *Specified
(`docs/REDESIGN_rngrn.md` §4.4, register item 11); **not present in the source** — there is
no `kstar_si` anywhere in `losses/terms.py` today.* Normalise the dispersion by its own
range before anchoring:

$$\bar\sigma(k) \;=\; \frac{\sigma(k)}{\max_k \sigma - \min_k \sigma + \varepsilon},
\qquad\quad
\mathcal{L}_{k^\ast\text{-si}} \;=\; \frac{1}{T}\log\sum_i e^{T\bar\sigma(k_i)} \;-\; \bar\sigma(k^\ast_\mathrm{obs}).$$

At $\varepsilon = 0$ the map $\sigma \mapsto c\,\sigma$ ($c>0$) leaves $\bar\sigma$ pointwise
unchanged, hence leaves $\mathcal{L}_{k^\ast\text{-si}}$ **exactly invariant**; at
$\varepsilon > 0$ it is invariant up to the regulariser. Uniform flattening therefore buys
nothing, and only relocating the argmax reduces the loss. The residual degenerate direction
is the opposite one — driving the *range* to zero, where $\bar\sigma \to 0$ uniformly and the
loss $\to 0$ through the $\varepsilon$ term — and it is closed by the split Turing hinges
below, which floor the range at twice their margin whenever they are active. Both
$\varepsilon$ and $T$ are **UNCALIBRATED**: $T$ starts at the inherited, never-swept $60$ and
$\varepsilon$ at the smallest value keeping the gradient finite while the hinges are
inactive; both are swept before any campaign. The claim that flattening is never profitable
is a *design argument to be verified numerically*, not a measured result.

### 5.3 Turing hinges on disjoint $k$-support, and the $\beta \ge 0$ hinge

**Implemented (`losses/terms.py::turing_hinges_split`), promoted default.** The two Turing
conditions are enforced as hinges on **disjoint slices of the $k$-grid**, so that no single
$k$ is asked to be both stable and unstable:

$$i_{\min} = \max\big(1,\ \lfloor \phi\,K \rfloor\big), \qquad \phi = 0.1,\ K = 400
\;\Rightarrow\; i_{\min} = 40,$$

$$\mathcal{L}_\mathrm{stab} = h\big(\sigma(k_1) + m\big),
\qquad
\mathcal{L}_\mathrm{unst} = h\big(m - \max_{i \ge i_{\min}} \sigma(k_i)\big),
\qquad
\mathcal{L}_\mathrm{turing} = \mathcal{L}_\mathrm{stab} + \mathcal{L}_\mathrm{unst},$$

with margin $m = 10^{-3}$. The first hinge pushes the low-$k$ end below $-m$; the second
pushes the maximum over the *upper* slice above $+m$. Two properties of this
implementation are recorded at the site and matter for interpretation. First, $k_1$ is not
$k = 0$: the grid starts at $k^\ast_\mathrm{obs}/50 + 10^{-3}$, so
$\mathcal{L}_\mathrm{stab}$ constrains $\sigma(k^\ast_\mathrm{obs}/50)$, a proxy for the
uniform mode rather than the uniform mode itself. Second, **$\phi$ is a fraction of the grid
index, not of $k^\ast$**, so the physical location of the instability floor depends on the
grid span: on the current grid it lands at $0.822\,k^\ast_\mathrm{obs}$ where the exp05 grid
put the same nominal setting at $0.698\,k^\ast_\mathrm{obs}$. Redefining $\phi$ as
$k^\ast$-relative is a ratified change (register item 12) that **changes what the term
means** and therefore requires re-measurement; the A0 control arm keeps the grid-relative
form.

The quantity $\max_{i\ge i_{\min}}\sigma(k_i)$ is exported as `sig_max_pos` and is the
ignition signal used by §5.5.

**The $\beta \ge 0$ hinge.** *Specified; arrives with fixed-point pinning (§2.5).* With
$\beta$ solved out rather than learned, non-negativity becomes a constraint on the remaining
parameters, imposed as $\sum_i h(-\beta_i)$ with
$\beta_i = \delta_i x^\ast_i - \mathrm{prod}_i(\mathbf{x}^\ast)$ and enforced hard at gate
time. This is a mechanistic statement — basal production cannot be negative — not a tuning
knob, and it carries no calibrated threshold of its own.

### 5.4 The biological prior

**Implemented (`losses/terms.py::param_prior`), default weight $0$, promoted to non-zero in
the redesign arm.** Two parts. A log-normal prior on the diffusivity ratio of the **two most
mobile species**,

$$\mathcal{L}_{d} \;=\; \frac{\big(\log D_{(N)} - \log D_{(N-1)} - \log r_0\big)^2}{2 s_0^2},
\qquad r_0 = 7.5,\quad s_0 = 1.0\ \text{(natural-log units)},$$

where $D_{(N)} \ge D_{(N-1)}$ are the largest and second-largest diffusivities, so a
near-immobile species never enters the ratio. The spread $s_0$ is **[TUNE] /
UNCALIBRATED** — `configs/bio_box.yaml` tags it as a chosen width, "not itself independently
calibrated"; only the centre $r_0$ is cited. And one-sided box hinges on $\alpha$ and
$\delta$ only,

$$\mathcal{L}_\mathrm{box} = \sum_{\theta \in \{\alpha,\delta\}} \sum_{\text{entries}}
\Big[\,h(\ell_\theta - \theta) + h(\theta - h_\theta)\,\Big].$$

$\beta$ is **never hinged**: `configs/bio_box.yaml` marks its row `UNCITED`, and the loader
returns `None` bounds rather than fabricating a range. This is the general rule — a row that
cannot be cited is reported but never scored in-or-out of box and never hinged.

The centre $r_0 = 7.5$ is the measured Nodal/Lefty ratio in live zebrafish, and the box
$[1, 60]$ has $1$ from Turing's own requirement that the inhibitor diffuse faster than the
activator, and $60$ as an explicitly flagged *reasoned proxy* rather than a measured ratio
bound. **The recorded tension must be stated with the prior, not after it**, and it must be
stated in the statistic the prior actually acts on. The generator's *two-most-mobile* ratio has
median $\approx 2.8$ under the stated priors and its $D_{\max}/D_{\min}$ median is
$\approx 91$ (§1.3, D-REDESIGN-3); the $\approx 135$ figure older documents attach to this
comparison names no measurement and is not used here. Centring the prior on biologically viable
literature values rather than on the synthetic population is a recorded owner decision, and a
hard box would risk making the synthetic target unrepresentable, which is why $D$-ratio is kept
as a **soft** prior while $\alpha$ and $\delta$ move into the construction (§2.6). And the prior
is not fixing a measured failure: at weight $0$ the D5 baseline measured $\alpha$ in-box $0/10$
and $\delta$ in-box $0/10$ but $d$-ratio in-box $10/10$. The row the prior acts on already
passes without it; the prior's purpose — per `docs/REDESIGN_rngrn.md` §2/§3.3 as corrected
2026-08-17 — is to hold the recovered ratio near the literature value against whatever the
generator produces, making the viability-versus-recovery tension *measurable*.

`anticollapse` — $h(\text{jac\_floor} - \|J\|_F)$ with floor $1.0$, weight $0.5$ — is
retired in the redesign arm on the structural argument that box-confined rates cannot let
$\|J\|$ collapse. What is and is not measured must be said plainly: the measured-rejected
result is *strengthening* the term to $2.0$ ($6/6 \to 2/6$), not removing it. Removal is an
untested change to the objective the baseline row was measured under, and the A0 arm keeps
it at $0.5$ as the control.

### 5.5 Spectral terms: the simulated spectrum

**Implemented in form (`losses/spectral.py`); the batched-across-members implementation is
new work.** Let $u^\ast(\theta)$ be the model's *patterned* steady state, obtained by the
forward solve of §6, and let targets be built once per run from the observed frame. All five
terms are unweighted scalars; all are computed on observed channels only, with Stage 0
fitting channel 0. One asymmetry in the implementation is worth stating: `spec_shape` and
`spec_aniso` are hard-wired to `u_star[0]` regardless of `cfg.channels`, while
`spec_amp_mean`, `spec_amp_fluct` and `real_moments` loop over `cfg.channels`
(`losses/spectral.py`) — so widening the channel set widens the amplitude and moment terms
only.

$$
\begin{aligned}
\textbf{spec\_shape:}&&
S(k) &= \frac{\mathcal{P}(k)}{\sum_{k'\in B_\mathrm{train}}\mathcal{P}(k')},
& \mathcal{L}_\mathrm{shape} &= \sum_{k\in B_\mathrm{train}}
\Big[\log S(k) - \log S^\mathrm{obs}(k)\Big]^2 ,\\[1ex]
\textbf{spec\_aniso:}&&
&& \mathcal{L}_\mathrm{aniso} &= \frac{1}{n_\mathrm{blk}^2}\sum_{\text{block}}
\big(B - B^\mathrm{obs}\big)^2 ,\\[1ex]
\textbf{spec\_amp\_mean:}&&
&& \mathcal{L}_{\mu} &= \sum_{c}\Big[\log \langle u^\ast_c\rangle - \log \langle u_c\rangle_\mathrm{obs}\Big]^2 ,\\[1ex]
\textbf{spec\_amp\_fluct:}&&
&& \mathcal{L}_{s} &= \sum_{c}\Big[\log \operatorname{std}(u^\ast_c) - \log \operatorname{std}(u_c)_\mathrm{obs}\Big]^2 ,\\[1ex]
\textbf{real\_moments:}&&
\gamma_c &= \frac{\langle v_c^3\rangle}{\max(\langle v_c^2\rangle,\,10^{-300})^{3/2}},
& \mathcal{L}_\gamma &= \sum_c \big(\gamma_c - \gamma_c^\mathrm{obs}\big)^2 ,
\end{aligned}
$$

with $v_c = u^\ast_c - \langle u^\ast_c\rangle$ and $B$ the normalised central log-power
block of §4.3. `spec_shape` is **gain-invariant by construction**: normalising within
$B_\mathrm{train}$ means $\mathcal{P}\to c\,\mathcal{P}$ leaves $S$ unchanged, so the term
constrains the *shape* of the band and the amplitude terms constrain its level, separately
and without double counting. Note that `spec_shape` reduces by a **sum** over band bins while
`spec_aniso` reduces by a **mean** over block pixels; their natural scales therefore differ
and their weights are not comparable to one another.

**Ignition gating: omitted, never zeroed.** A member's spectral terms are computed only when

$$\texttt{is\_ignited} \;\equiv\; \big(\text{steady-state solve converged}\big)
\;\wedge\; \big(\texttt{sig\_max\_pos} > m_\mathrm{ign}\big),
\qquad m_\mathrm{ign} = 10^{-3}\ \textbf{(UNCALIBRATED)} ,$$

i.e. on **detected** patterning-capability, never on a configuration flag. When the test
fails, or the forward solve fails or returns an unpatterned field, the keys are **omitted
from the term dictionary entirely** and a `spectral_skipped` reason is recorded — they are
never computed and multiplied by zero. The distinction is not cosmetic: a zeroed term enters
any weighting strategy that reads term values, whereas an omitted one cannot. Absence of the
`sig_max_pos` diagnostic raises rather than being coerced to "not ignited". Note also that
`is_ignited` is a *linear-theory* gate; the forward solve applies its own real-space
amplitude floor $\text{pattern\_floor} = \max(10^{-3},\,0.02\,|x^\ast_0|)$, closed by
measurement in D-FFT-9 closure 2 (Turing saturated amplitude $0.170$–$0.172$ against
non-Turing decay $\le 6.7\times10^{-17}$ for the fixture, a separation of $2.6\times10^{15}$
and a margin of $\ge 17\times$). Turing-unstable and *patterns* are different claims and are
gated separately.

**Primary after ignition; guard-rail demotion.** *Specified (register item 5).* Once a member
ignites, the spectral terms become **primary** and the ignition terms drop to a guard-rail
weight, nominally $\sim 10\,\%$ of their ignition weights — **UNCALIBRATED**, swept at the
first campaign. A member that later de-ignites falls back to the ignition weights. This
amends the earlier specification decision that every term stays on unchanged for the whole
run, and the rationale is empirical: the linear-theory objective as *primary carrier* is the
measured baseline failure, whereas the mechanism-discriminating information demonstrably
lives in the simulated spectrum (§4.4, D4: held-out-band separation $47\to149$, and
$19.6\to107.8$ on $B_\mathrm{harm}$ alone, against D3's $\sim 31\,\%$/bin floor).

**Inverse-variance per-bin weights.** *Specified, contingent on the multi-realization
protocol of §1.7.* With $R = 6$ training realizations the per-bin variance of the log-RAPS
target is estimated directly, and the shape term becomes

$$\mathcal{L}_\mathrm{shape} \;=\; \sum_{k \in B_\mathrm{train}} w_k
\Big[\log S(k) - \overline{\log S^\mathrm{obs}}(k)\Big]^2,
\qquad
w_k \;\propto\; \Big(\widehat{\operatorname{Var}}_R\big[\log S^\mathrm{obs}(k)\big]\Big)^{-1},$$

replacing the single-frame patch-quadrant floor with a measured, per-bin, realization-level
uncertainty, and replacing the single target by an $R$-frame average that is $\sqrt{R}$
tighter. The held-out bands are never trained on and never used for selection.

**The transpose note.** Only the spectral terms break the transpose degeneracy of §2.4,
because only they see eigenvectors and nonlinear saturation. A member that never ignites is
reported with its transpose partner as an unresolved two-fold ambiguity.

---

## 6. The forward map and its gradients

*Implemented: `src/rngrn/forward.py`, `src/rngrn/eval/numerics.py`,
`src/rngrn/etdrk4_torch.py`. Specified: the adaptive commensurate solve box and the
truncated-unrolled gradient path.*

### 6.1 The patterned steady state as a root-finding problem

The model-side spectrum is not a linear-theory prediction; it is the spectrum of an actual
saturated pattern. That pattern is defined as a root of the spectral reaction–diffusion
operator (`forward.py::make_spatial_F`). For a field $u \in \mathbb{R}^{N\times n \times n}$
on a periodic square of side $L_\mathrm{solve}$,

$$F(u,\theta) \;=\; \mathbf{D}\odot \mathcal{F}^{-1}\!\big[-|\mathbf{k}|^2\,\mathcal{F}[u]\big]
\;+\; \mathbf{f}(u) \;=\; 0 ,$$

with $|\mathbf{k}|^2$ built from $2\pi\,\mathrm{fftfreq}$ on the first axis and
$2\pi\,\mathrm{rfftfreq}$ on the second (matching the `rfft2` layout). The Laplacian is
again spectral and exact; **no finite-difference stencil appears anywhere in the forward
map**. The closure is differentiable in both $u$ and $\theta$, which is what allows the same
object to serve as the Newton operator, the adjoint operator, and the $\partial F/\partial
\theta$ vector–Jacobian product.

### 6.2 ETDRK4 relax

The root is approached by integrating the corresponding PDE to saturation with an
**exponential time-differencing fourth-order Runge–Kutta** scheme (Cox & Matthews 2002;
Kassam & Trefethen 2005), operator-split into a Fourier-diagonal linear part
$L_i(\mathbf{k}) = -D_i|\mathbf{k}|^2$ treated exactly, and the full reaction treated
explicitly. Writing $v = \mathcal{F}[u]$ and $\mathcal{N}(v) = \mathcal{F}[\mathbf{f}(\mathcal{F}^{-1}v)]$,
one step is

$$
\begin{aligned}
a &= E_2 v + Q\,\mathcal{N}(v), \\
b &= E_2 v + Q\,\mathcal{N}(a), \\
c &= E_2 a + Q\,\big(2\mathcal{N}(b) - \mathcal{N}(v)\big), \\
v^{+} &= E\,v + \mathcal{N}(v)\,f_1 + 2\big(\mathcal{N}(a) + \mathcal{N}(b)\big) f_2 + \mathcal{N}(c)\, f_3 ,
\end{aligned}
$$

with $E = e^{\Delta t L}$, $E_2 = e^{\Delta t L/2}$ exact, and the $\varphi$-functions
$Q, f_1, f_2, f_3$ — all of which are $0/0$ as $\Delta t L \to 0$ — evaluated stably by the
Kassam–Trefethen **contour-integral** average over $M = 32$ points on a circle around each
$\Delta t L$:

$$Q = \Delta t\,\Big\langle \tfrac{e^{z/2}-1}{z}\Big\rangle,\quad
f_1 = \Delta t\,\Big\langle \tfrac{-4-z+e^{z}(4-3z+z^2)}{z^3}\Big\rangle,\quad
f_2 = \Delta t\,\Big\langle \tfrac{2+z+e^{z}(z-2)}{z^3}\Big\rangle,\quad
f_3 = \Delta t\,\Big\langle \tfrac{-4-3z-z^2+e^{z}(4-z)}{z^3}\Big\rangle,$$

$z = \Delta t L + r_m$, $r_m = e^{i\pi(m-\tfrac12)/M}$. The timestep is set from the local
reaction rate, $\Delta t = 0.2/\rho(J)$ with $\rho(J) = \max_i|\lambda_i(J)|$ at the
homogeneous state, recomputed each solve. The relax is run in chunks of 500 steps and
declared **saturated** when the last five chunk-boundary values of *both* the channel-0
amplitude and $k^\ast$ are flat to relative tolerance $10^{-4}$; failure to saturate within
the chunk budget, or a non-finite field, raises rather than returning a value. The half-
spectrum (`rfft2`) implementation is not an approximation — it agrees with the full-spectrum
path to $\le 6.7\times10^{-16}$ after 100 steps — and the torch port used on GPU is
bit-equivalent to the numpy reference at $1.1\times10^{-13}$ over 100 steps.

### 6.3 Newton polish and the translation-mode correction

The relaxed field is polished by a damped, matrix-free Newton iteration on
$\|F(u)\|/\|u_0\|$ (`forward.py::newton_polish`). The linearisation $A = \partial F/\partial u$
is never assembled: $A x$ is a forward-mode `jvp` through the $F$ closure and $A^{\mathsf T}x$
a reverse-mode `vjp`. Each step solves

$$A\,s \;=\; -F(u)$$

in the **minimal-norm least-squares** sense by right-preconditioned LSMR with the symmetric
Fourier-diagonal preconditioner $M = (\gamma + D|\mathbf{k}|^2)^{-1}$, $\gamma = \rho(J)$:
substituting $s = M y$ gives the operator $AM$ whose transpose $MA^{\mathsf T}$ is likewise
matrix-free. The solve is wrapped in up to six rounds of iterative refinement **on the true,
unpreconditioned residual**, stopping on tolerance or on failure to reduce the residual by
more than $10\,\%$ (stall detection). The step is then damped by up to six halvings, accepting
the first $\lambda$ for which $\|F(u+\lambda s)\|$ decreases; if none does, the iteration
stops and reports its residual rather than pretending to converge. At most 30 Newton
iterations run; the internal early exit is at relative residual $10^{-11}$ and the
externally enforced bar is

$$\frac{\|F(u^\ast)\|}{\|u_0\|} \;\le\; \texttt{CONVERGENCE\_TOL} = 10^{-9}.$$

**The translation-mode correction (finding F-D1-2).** A periodic patterned steady state is
translation-invariant in the continuum, so $\partial u^\ast/\partial x$ and
$\partial u^\ast/\partial y$ are exact zero modes of $A$. On a grid they are *near*-null
rather than null ($\|At\|/\|Av\| = 3.6\times10^{-4}$ at $96^2$), and a $\theta$-perturbed
fixed point sits at a slightly different phase. Left alone, the Newton solve stalls at
$\sim 2.4\times10^{-8}$ instead of reaching $\le 10^{-11}$. The fix solves the phase part
explicitly in the two-dimensional mode subspace. With
$t_x = \mathcal{F}^{-1}[\mathrm{i}k_x \hat u]$, $t_y = \mathcal{F}^{-1}[\mathrm{i}k_y\hat u]$
orthonormalised by Gram–Schmidt to $\{q_1,q_2\}$, and $r = F(u) + A s$ the residual left by
the bulk step,

$$G_{ij} = \langle q_i,\, A q_j\rangle, \qquad
\rho_i = \langle q_i,\, r\rangle, \qquad
c = G^{-1}\rho, \qquad
s \;\leftarrow\; s - \big(c_1 q_1 + c_2 q_2\big),$$

applied only when both modes survive orthogonalisation and $\operatorname{cond}(G) < 10^8$.

### 6.4 The adaptive commensurate solve box (resolution of F-D1-5)

*Specified (`docs/REDESIGN_rngrn.md` §4.3, register item 2).*

**The problem, measured.** The premise of §6.5's exact gradient is strict convergence
$F(u^\ast) = 0$. On the *data box* — $L = 185.006$, about $8.36$ periods of the observed
pattern but only $\sim 5.2$ periods of the fixture model's own pattern — the Newton polish
**wall-caps** at relative residuals of $2.9\times10^{-5}$, $4.7\times10^{-5}$,
$3.4\times10^{-5}$ and $1.5\times10^{-5}$ at $n \in \{96,128,192,256\}$ respectively, against
the $10^{-9}$ bar; no grid reaches it (F-D1-5, NEGATIVE, 2026-08-13). The unifying pattern —
recorded there **as a hypothesis, and treated as one here** — is that every convergent solve
to date used a box commensurate with the model's own pattern, so that the pattern can tile
the domain without strain. Because $\theta$ varies continuously during training,
incommensurability is the *generic* condition, not an accident.

**The resolution.** Training solves run on a box made commensurate with the model's *current*
pattern:

$$L_\mathrm{solve}(\theta) \;=\; p\,\frac{2\pi}{\hat k(\theta)},
\qquad p = 8, \qquad n = 96\ (12\ \mathrm{px}/\lambda),$$

where $\hat k(\theta) = \arg\max_k \sigma(k;\theta)$ is the current dispersion peak, taken
**detached** — solve-box geometry is not differentiated. Comparison with the observed
spectrum is made **scale-free**: the observed log-RAPS targets are interpolated onto the
solve box's bin centres expressed in $k/k^\ast_\mathrm{obs}$ units, once per re-tile, and the
band masks remain defined relative to $k^\ast_\mathrm{obs}$. Re-tiling is **hysteretic** —
triggered only when $\hat k$ drifts by more than half a solve-box bin,
$|\Delta \hat k|/\hat k > 1/(2p)$ — and on re-tile the $k$-grid is rebuilt, warm starts are
cleared, and targets are re-interpolated.

Three honest gaps are attached to this and must be closed before any spectral gradient is
believed. (i) **Training-grid spectral fidelity is UNMEASURED**: D2 measured band-limited
spectral quantities agreeing with $512^2$ at $256^2$ but *not* at $96^2$, where one-bin
$k^\ast$ agreement fails $2/6$; training therefore compares a $96^2$ model spectrum to a
$512^2$ observed spectrum with the model-side estimator's fidelity unmeasured at that grid.
The $96^2$-versus-$512^2$ RAPS band distance is to be measured on the fixture, and if it does
not clear D3's $\sim31\,\%$/bin estimation floor, $n$ rises to the coarsest grid that does.
(ii) **Interpolation error is UNMEASURED** and is expected far below that floor; it is
measured once per re-tile geometry. (iii) Commensurability is proven **neither necessary nor
sufficient** — an independent stall channel exists — which is exactly why the finite-difference
check of §6.6 is a *gate*, not an assumption.

A per-run **stall-rate counter** is kept. If more than $\sim 20\,\%$ of ignited members'
solves miss the $10^{-9}$ bar on the commensurate box (**UNCALIBRATED** — no baseline exists;
the fraction is to be calibrated from the measured stall-rate distribution against the
measured gradient-error difference between the two paths, not chosen for convenience), those
members switch to the unrolled gradient path rather than loosening the convergence bar.
Loosening the bar touches a pre-registered-class threshold and is off the table.

**Validation never uses the adaptive box.** All gate evaluations run at $512^2$ on the data
box; no cheaper grid is licensed, because morphology class flips on $2/3$ seeds at $256^2$
(D-FFT-9 closure 3).

### 6.5 Implicit differentiation: the adjoint

The primary gradient path *is* backpropagation through the simulated network, taken at its
patterned steady state by the implicit function theorem. Since $F(u^\ast(\theta),\theta) = 0$
identically in $\theta$,

$$\frac{\partial F}{\partial u}\frac{\mathrm{d}u^\ast}{\mathrm{d}\theta}
+ \frac{\partial F}{\partial \theta} = 0
\qquad\Longrightarrow\qquad
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta}
= \frac{\partial \mathcal{L}}{\partial u^\ast}\frac{\mathrm{d}u^\ast}{\mathrm{d}\theta}
= -\,\lambda^{\mathsf T}\frac{\partial F}{\partial \theta},
\qquad\text{where}\quad
\Big(\frac{\partial F}{\partial u}\Big)^{\!\mathsf T}\lambda = \frac{\partial \mathcal{L}}{\partial u^\ast}.$$

This is implemented as a custom autograd node (`forward.py::PatternSolve`) whose forward pass
returns the already-solved $u^\ast$ and whose backward pass performs exactly **one** adjoint
solve — autograd pre-accumulates every consuming term's $\partial\mathcal{L}/\partial u^\ast$
at that node — followed by a single `vjp` through $F(u^\ast;\theta)$ with cotangent
$-\lambda$ to produce $\mathrm{d}\mathcal{L}/\mathrm{d}\theta$. Memory is $O(1)$ in the number
of relax steps and the result is exact given a converged $u^\ast$.

**Why minimal-norm, and why this is not a matter of taste (D-FFT-10, BINDING).** $A^{\mathsf T}$
is near-singular on the same two-dimensional translation subspace as $A$. A Krylov method
whose search space is *projected off* those modes converges quickly to a solution of the
projected system and reports a small **projected** residual — measured $10^{-13}$ — while its
**true** residual stalls at $5.5\times10^{-4}$–$5.7\times10^{-3}$. The resulting gradient bias
was measured at $10^{-5}$ to $5.5\times10^{-2}$ **relative**, i.e. up to an order of magnitude
*above* the solver's own true residual, constant across finite-difference step size and hence
systematic rather than noise. The adjoint is therefore required to be the **minimal-norm
least-squares** solution, obtained by the same right-preconditioned LSMR with true-residual
refinement (tolerance $10^{-10}$), never by a translation-projected Krylov method. With the
minimal-norm scheme true residuals are $\le 3.6\times10^{-12}$ ($64^2$ probe) and
$\le 6.1\times10^{-12}$ at the committed $96^2$ record.

Because a silently biased gradient is the most dangerous failure available here, a
**tripwire** is enforced in the backward pass: if the adjoint's true relative residual exceeds
$10^{-8}$ the backward pass *raises* rather than returning a gradient. That threshold sits
about four orders above the measured healthy regime ($\sim10^{-12}$) and about four orders
below the measured biased regime ($\sim5.5\times10^{-4}$).

### 6.6 The truncated-unrolled alternative, and the A/B discipline

*Specified (register item 8: **adjoint primary, truncated-unrolled as stall fallback**).*

The structural fallback is to run the ETDRK4 relax as differentiable torch operations and
backpropagate through the unrolled steps with **gradient checkpointing**. Its decisive
property is that it **requires no convergence at all**: the loss is defined on the relaxed
field after finite time, so the F-D1-5 stall class cannot touch it. Two costs are stated
plainly. Checkpointed activation memory grows with segment length $\times$ fields (tractable
at the $96^2$ training grid). And gradients taken through the exponential-growth phase of the
instability can explode — mitigated by **truncation**: the warm-started state is detached and
only the final *saturated* segment is differentiated. The segment length is **UNCALIBRATED**
and is to be set from a measured gradient-error-versus-length curve.

**The A/B protocol.** Before either path feeds a training run at the operating point, a
finite-difference check runs on **both** paths at the same $\theta$: 10 random directions
$\times$ the active loss terms, central differences, tolerance $10^{-4}$. This is the same
protocol under which the adjoint path was accepted, with worst relative error
$6.4\times10^{-8}$ over the spectral terms and $8.1\times10^{-8}$ including `real_moments`, at
$96^2$. During training the adjoint is used wherever the Newton polish reaches the $10^{-9}$
bar and the truncated-unrolled path wherever it stalls. Promotion of the unrolled path to
primary is pre-specified so that no post-hoc choice exists: it may become primary **only if**
the A/B shows it finite-difference-faithful at tolerance $10^{-4}$ on both converged and
stalled members **and** its measured cost does not exceed the adjoint path's.

### 6.7 Warm starts

The solver is stateful across optimizer steps. In the training mode the warm state seeds a
*short* re-relax (budget-capped, **UNCALIBRATED**) followed by Newton polish; the alternative
Newton-only warm start was measured pathological at Adam-scale $\theta$ displacement
($5030\,\mathrm{s}$ warm versus $938\,\mathrm{s}$ fresh at $96^2$ on CPU) and is retained only
for finite-difference checking, where re-relaxing would drift the pinned phase and contaminate
the check. The warm state is **cleared** whenever a solve fails *or* returns an unpatterned
field — a homogeneous warm start would re-converge homogeneous forever.

CPU forward solves are disqualified for training outright: a fresh solve costs
$938$–$1374\,\mathrm{s}$ at $96^2$. The GPU integrator measures $3.25\,\mathrm{ms}$ per
ETDRK4 step at $512^2$ in float64, and the batched CUDA path measures $11.6\times$ over serial
CPU at $96^2$ with $B=32$.

---

## 7. Optimization

*Specified (`docs/REDESIGN_rngrn.md` §4.5). The batched ignition machinery is implemented
(`model.py::BatchedRNGRN`, `losses/total.py::compute_terms_batched`); the batched spectral and
forward paths are new work and are currently **refused by name** rather than silently
approximated.*

### 7.1 Why a population, and why batched

The recovered object is not a point estimate (§9), and the optimization is organised around
that. A population of $B$ independent members is carried as one leading batch dimension, so
that one set of tensor operations advances all of them. Independence is a contract: members
share no tensor except the read-only $k$-grid and frame, every loss term reduces per member,
and Adam is elementwise — so one Adam step over the stacked parameters *is* $B$ independent
Adam steps, **provided gradient clipping is also per member**. Torch's standard
`clip_grad_norm_` would clip the joint norm over all members and silently couple them; the
per-member rule
$\mathrm{coef}_b = \min\!\big(1,\ \text{max\_norm}/(\|g_b\| + 10^{-6})\big)$ is applied
independently to each member.

The measured motivation for batching is stark. Serially the GPU *loses*, because a step is a
long chain of tiny $(3,3)$ kernels whose launch overhead dominates: one Adam step of one
restart costs $4.99\,\mathrm{ms}$ on CPU against $20.01\,\mathrm{ms}$ on CUDA. With a leading
batch dimension the same arithmetic amortises the fixed per-kernel cost; measured cost per
restart-step at $N=3$, $K=400$, cubic backend, float64:

| $B$ | 1 | 8 | 32 | 128 | 512 |
| --- | --- | --- | --- | --- | --- |
| serial CPU (ms) | 4.99 | 4.72 | 4.76 | 4.83 | — |
| batched CPU (ms) | 5.96 | 1.07 | 0.69 | 0.61 | 0.59 |
| batched CUDA (ms) | 24.47 | 3.60 | 0.84 | 0.33 | 0.16 |

CUDA overtakes batched CPU at $B \approx 56$; at $B=512$ it is $3.8\times$ batched CPU and
$30\times$ serial CPU. The `eig` dispersion backend **must not** be used on CUDA (measured
flat in $B$, $2500\times$ worse than the cubic backend at $B=8$), which is what makes §3.4's
closed-form cubic a prerequisite rather than an optimisation. float32 was measured and loses
on both speed and accuracy at this size and is not adopted.

### 7.2 Phase I — ignition

$B$ members are optimised with **ignition terms only** on the closed-form cubic dispersion
backend, for $\sim1500$ Adam steps at learning rate $0.05$ with per-member gradient clipping.
$B = 512$ is a starting point and is **UNCALIBRATED**: the largest measured batch to date is
$B = 32$ at $96^2$, and $B$ is to be set from a measured throughput-and-RSS curve under the
project's host-memory guard. The per-member ignition probability is **not known** and is a
deliverable of this phase: the baseline's $4/10$ Turing rate is a per-*seed*, best-of-four
rate under a different objective and bounds neither quantity. Width is preferred to depth
because the hit rate was measured to saturate by $\sim 4000$ steps — measured on a different
sample family, with the source flagging its cross-budget $k^\ast$ comparison as not
like-for-like.

### 7.3 Cull — diversity-aware top-$K$

Ignited members are ranked by ignition loss and **at most one member per distinct recovered
sign structure** is kept, best-first. The sign structure of a member is read off the gate
(`recover._topology`):

$$\mathrm{sign}_{ij} =
\begin{cases}
0, & s_{ij} \le r_0 \max_{kl} s_{kl},\\
+1, & g_{ij} > 1/2,\\
-1, & g_{ij} \le 1/2,
\end{cases}
\qquad r_0 = 0.05\ \textbf{(UNCALIBRATED)} .$$

The zero-call tolerance $r_0$ is explicitly open: the baseline's ten seeds produced ten
distinct sign structures with no consistent edges to calibrate against, which is a finding
rather than a licence to pick a cut.

**Two different $0.05$ thresholds exist and must not be conflated.** The cull's rule above is
`recover.py:146`'s $s_{ij} > 0.05\,\max_{kl} s_{kl}$ — a cut on the model's own *binding
budget* $s$, deciding when an edge counts as absent. The reproducibility scorer's
`DEFAULT_SIGN_ZERO_RTOL = 0.05` (`scoring/reproducibility.py:100`) is a different object: a cut
on $|J_{ij}|$ relative to the matrix's own largest $|J|$ entry, deciding when a *Jacobian* entry
is called sign-zero when comparing two recovered networks. It is the **latter** that the
sensitivity sweep and the reporting convention belong to: the reproducibility statistic wanders
between $0.125$ and $0.500$ as that rtol sweeps $0.005 \to 0.5$, and it is reported at $0.02$,
$0.05$ and $0.10$. Neither threshold calibrates the other.
$K = 16$ is **UNCALIBRATED** and is to be set from the measured distinct-structure count — the
cull cannot keep more members than there are distinct structures.

**The culled runners-up are a candidate *pool*, not the null ensemble.** The binding
specification requires each alternative sign structure to be *refit* to $B_\mathrm{train}$
with gates frozen to that structure under an identical budget, plus single-edge-flip variants
of the winner, at least six members in total. Substituting unrefit pre-cull members would
weaken a binding gate; the design implements the specification as written and merely *seeds*
the refits from the candidate pool.

### 7.4 Phase II — spectral refinement, and Phase III

The $K$ survivors train with the spectral terms primary and the ignition terms at guard-rail
weight (§5.5), forward relaxes batched across members on GPU, warm-started, with Newton polish
and the §6.5 adjoint. Budget $\sim2000$ steps at learning rate $0.02$; **both UNCALIBRATED**
and swept. A de-ignited member falls back to guard-rail terms.

Phase III is an LBFGS polish of the winner with the ignition gate frozen, aborting on
de-ignition. It is an **ablation arm** — and it is *not* free-standing: the D5 baseline row was
produced **with `lbfgs_steps=50`** (all ten tracked frozen configs; `recover.recover`'s own
default is $50$), so the A0 control arm keeps $50$ for comparability. What does not exist is an
A/B isolating the polish's contribution; that is what this arm measures.
(`docs/REDESIGN_rngrn.md` §4.5 was corrected on 2026-08-17; Appendix A1.)

**Selection.** The winner and the gate-passers are chosen on fitted-band loss alone. The
held-out bands are consulted exactly once, by scoring code, at gate time.

### 7.5 The weight-noise arm

*Specified as an **arm**, never a default (register item 16; D-REDESIGN-2, adopting D-FFT-14's
pre-registered rules verbatim).*

Parameter noise during training is evaluated for two separable payoffs: **optimization
annealing** — if good mechanisms occupy wider basins, noise concentrates seeds into them,
which is an empirical claim about *this* landscape and not a given — and a
**mechanism-robustness prior**, since noise penalises sharp minima and so selects $\theta$
whose Turing instability survives perturbation. The placement constraints bind verbatim:
**noise never touches the $\theta$ the forward solve sees**. It is injected into the gradient
or into the cheap linear-theory evaluations only, and every solve runs at the clean current
$\theta$. The reasons are measured: warm-start cost is sharply sensitive to per-step $\theta$
displacement, and noise near the ignition margin would toggle the gate, producing ignition
chatter and wasted solves. Gate integrity binds too — the arm is judged on held-out-band
distances and channel co-gates *alongside* seed agreement, never on agreement alone, because
noise that merely smooths the landscape can raise agreement without raising truthfulness.
Schedule and magnitude are **UNCALIBRATED** and must be pre-registered before the arm runs,
with a matched no-noise control at identical seeds and budget. Making noise a default is
explicitly rejected: it would change what every recorded number means.

---

## 8. Verification: the dynamical lift

*Implemented: `src/rngrn/eval/lifted.py` (algebra, Jacobians, verdicts, `rescale_mu`,
`mu_critical`, `robustness_vs_mu`, `simulate_lifted`). **Unvalidated in space and at finite
$\mu$**; it gates nothing until the ladder of §8.6 passes (D-LIFT-1, preregistration §3.7).*

### 8.1 What the QSS reduction assumes, and what undoing it costs

The model's reaction (§2.2) is **quasi-steady-state algebra**: promoter occupancy is assumed
to equilibrate instantaneously with transcription-factor concentration, so the gates never
appear as state variables. Real TF–promoter binding is fast but finite. The lift makes the
gates explicit fast variables with dimensionless timescale $\mu$:

$$\textbf{competitive:}\quad
\mathrm{free}_i = 1 - \sum_j GA_{ij} - \sum_j GR_{ij},\qquad
\begin{aligned}
\frac{\mathrm{d}GA_{ij}}{\mathrm{d}t} &= \frac{K^{A}_{ij}x_j^{\,n}\,\mathrm{free}_i - GA_{ij}}{\mu},\\
\frac{\mathrm{d}GR_{ij}}{\mathrm{d}t} &= \frac{K^{R}_{ij}x_j^{\,n}\,\mathrm{free}_i - GR_{ij}}{\mu},
\end{aligned}
\qquad \mathrm{prod}_i = \sum_j \alpha_{ij}GA_{ij},$$

$$\textbf{nc1:}\quad
\frac{\mathrm{d}GA_{ij}}{\mathrm{d}t} = \frac{K^{A}_{ij}x_j^{\,n}(1-GA_{ij}) - GA_{ij}}{\mu},
\quad
\frac{\mathrm{d}GR_{ij}}{\mathrm{d}t} = \frac{K^{R}_{ij}x_j^{\,n}(1-GR_{ij}) - GR_{ij}}{\mu},
\quad
\mathrm{prod}_i = \Big(\sum_j \alpha_{ij}GA_{ij}\Big)\prod_j\big(1-GR_{ij}\big),$$

$$\textbf{both:}\qquad
\frac{\partial x_i}{\partial t} = D_i\nabla^2 x_i + \beta_i + \mathrm{prod}_i - \delta_i x_i .$$

The state is $z = [\,x\,(N)\ |\ GA\ (N^2)\ |\ GR\ (N^2)\,]$, of dimension
$N + 2N^2 = \mathbf{21}$ at $N=3$. **Only $x$ diffuses** — promoter states are DNA-bound — so
the lifted diffusion matrix is $\operatorname{diag}(D_1,\dots,D_N,0,\dots,0)$.

**The fixed point is $\mu$-independent, in two lines.** Setting $\mathrm{d}GA/\mathrm{d}t =
\mathrm{d}GR/\mathrm{d}t = 0$ multiplies out the $1/\mu$ and leaves
$GA_{ij} = K^{A}_{ij}x_j^n\,\mathrm{free}_i$ (competitive) or
$GA_{ij} = u_{ij}/(1+u_{ij})$ (nc1) — which are *exactly* the algebraic occupancies that §2.2
substituted away; solving the competitive constraint for $\mathrm{free}_i$ gives
$\mathrm{free}_i = [1 + \sum_j(K^{A}_{ij}+K^{R}_{ij})x_j^n]^{-1}$, reproducing the
Shea–Ackers denominator. Substituting back into $\mathrm{d}x/\mathrm{d}t = 0$ therefore
recovers $f_i(\mathbf{x}^\ast) = 0$ identically, for **every** $\mu$. Hence the lifted fixed
point is $z^\ast = (\mathbf{x}^\ast, GA_\mathrm{qss}(\mathbf{x}^\ast),
GR_\mathrm{qss}(\mathbf{x}^\ast))$ and $\mu$ moves *stability and dynamics, never the steady
state*. This is what makes every finite-$\mu$ comparison apples-to-apples, and it is pinned by
tests at $\max|f_\mathrm{lift}(z^\ast)| < 10^{-7}$ (`tests/test_lifted.py`, both forms, $N=3$).
A tighter figure ($1.28\times10^{-8}$ over 8 systems $\times$ 7 values of $\mu$ spanning
$10^{-6}$–$10^{3}$) exists only in the parked `TIMESCALE_MU` artefact, which its own record
marks **do-not-cite** — the value appears nowhere in `src/` or `tests/` — so it is not relied on
here, and §8.6's V0 *measures* the achievable floor instead.

### 8.2 Where biology sits on the $\mu$ axis

$\mu$ is the ratio of the TF–promoter binding timescale to the protein
production–degradation timescale. Both are times, so $\mu$ is dimensionless and the model's
own time unit cancels. The derivation (`eval/lifted.py`, `docs/TIMESCALE_MU.md` §2) is:

- **Numerator** — TF residence time on specific DNA, from live-cell single-molecule tracking.
  Chen J. *et al.* (2014), *Cell* **156**:1274–1285 measure Sox2 dwelling $12.0$–$14.6\,$s on
  specific target DNA and $0.75$–$0.9\,$s on non-specific chromatin. Tens of seconds is the
  standard order across factors; $1$–$30\,$s is taken as the defensible span.
- **Denominator** — protein mean lifetime in patterning tissue. Müller P. *et al.* (2012),
  *Science* **336**:721–724 measure clearance half-lives of Cyclops, Squint, Lefty1 and Lefty2
  in live zebrafish embryos spanning $95$–$218\,$min, i.e. mean lifetimes
  $t_{1/2}/\ln 2 = 8.2\times10^{3}$–$1.9\times10^{4}\,$s. (Schwanhäusser *et al.* 2011 give a
  median protein half-life of $46\,$h in cultured NIH3T3, an order of magnitude longer, which
  would push $\mu$ *down*; the developmental anchor is therefore the conservative, larger-$\mu$
  choice.)
- **Model time unit** — the cited $\delta$ row is $[0.4, 5.0]$, so one model time unit is
  $0.4$–$5$ protein lifetimes and $\mu$ in model units carries that factor.

$$\mu_\mathrm{bio} = \frac{t_\mathrm{residence}}{\delta_\mathrm{model}\,t_\mathrm{lifetime}}
= \left[\frac{1}{5.0\times1.9\times10^{4}},\ \frac{30}{0.4\times8.2\times10^{3}}\right]
= \big[\,1.1\times10^{-5},\ 9.2\times10^{-3}\,\big],$$

with central value $10/(1\times1.39\times10^{4}) = 7.2\times10^{-4}$. This is **cited, not
UNCALIBRATED**, but it is an order-of-magnitude band three decades wide, and its width must
stay visible on every figure.

### 8.3 The lifted dispersion and the strict stationary verdict

The lifted Jacobian is obtained by autodiff of the lifted right-hand side — never hand-derived
— and the lifted dispersion is the leading eigenvalue of a $21\times21$ problem per
wavenumber,

$$\sigma_\mathrm{full}(k,\mu) = \max \operatorname{Re}\operatorname{eig}
\Big(J_\mathrm{full}(\mu) - k^2 \operatorname{diag}(D_1,\dots,D_N,0,\dots,0)\Big),
\qquad
\omega(k) = \big|\operatorname{Im}\lambda_\mathrm{lead}\big| ,$$

on a log-spaced grid of 500 points over $10^{-3}$–$10^{3.5}$ with $k=0$ prepended and excluded
from the $k>0$ search (a near-immobile node's branch only turns over at $k \sim 10^{3}$).

**The `rescale_mu` trick makes a whole $\mu$ axis nearly free, and it is exact.** $\mu$ enters
the lifted right-hand side only as an overall $1/\mu$ on the $2N^2$ gate equations; the $x$
equations do not contain it at all. Every *gate row* of the Jacobian therefore carries exactly
one factor $1/\mu$ and the $x$ rows carry none:

$$J_\mathrm{full}(\mu)\big|_{\text{rows} \ge N} = \frac{1}{\mu}\,J_\mathrm{full}(1)\big|_{\text{rows}\ge N},
\qquad
J_\mathrm{full}(\mu)\big|_{\text{rows} < N} = J_\mathrm{full}(1)\big|_{\text{rows}<N}.$$

Verified against a fresh autodiff Jacobian to relative $<10^{-12}$ at $\mu = 10^{-3}, 0.1, 7.3$.
One autodiff pass therefore buys an entire $\mu$ scan.

**The verdict is strict and stationary.** A candidate passes only when

$$\max\operatorname{Re}\operatorname{eig}\big(J_\mathrm{full}\big) < 0
\quad\wedge\quad
\max_{k>0}\sigma_\mathrm{full}(k) > 10^{-9}
\quad\wedge\quad
\omega(k^\ast_\mathrm{lift}) \le 10^{-9} .$$

**The trace test must not be used here, and the reason is structural.** $\operatorname{tr}
J_\mathrm{full}$ is dominated by the $-1/\mu$ gate diagonal and stays negative at *any* $\mu$,
including after the uniform state has already gone unstable — so a trace criterion would
report "stable" for a system that has gone Hopf. The trace criterion was independently measured
overcounting by $64\times$ on the reduced system. The **Hopf exclusion** is likewise
non-negotiable: a Turing-unstable mode with non-zero imaginary part is a travelling wave, not
the stationary pattern this work claims, and it is counted separately everywhere.

**The measured $\mu$-scan, cited honestly.** One scan is on record — harvest survivor
`nc1/mobile3` draw 0, $n = 1$, the only one — and it shows a **re-entrant** Turing band:
strictly Turing for $\mu \le 10^{-2}$, lost over $\mu \sim 10^{-1}$–$10^{1}$ via an
oscillatory (Hopf) route, and strictly Turing again for $\mu \ge 10^{2}$, where the frozen
gates themselves act as slow non-diffusing species. On that single draw the biological band
sits entirely inside the first Turing window, which is weak evidence *for* safety at
biological $\mu$, not against it. What the scan does establish, and what governs the design,
is that **the verdict is non-monotone in $\mu$**: single-point extrapolation is unsound, a
plain bisection is wrong, and the band must be scanned. Whether QSS-Turing implies
lifted-Turing at biological $\mu$ *across a population* is **UNMEASURED**.

### 8.4 The spatial integrator: Strang splitting with an exact gate substep

The gates relax at rate $1/\mu$, so an explicit scheme would need $\Delta t \ll \mu$. Instead,
**at frozen $x$ the gate block is linear in $(GA, GR)$** and is integrated in closed form
(`lifted.gate_step_exact`). One Strang step is: gates for $\Delta t/2$ (exact, frozen $x$) →
$x$ over $\Delta t$ by ETDRK4 with the gates frozen → gates for $\Delta t/2$.

For **nc1** the block is diagonal: with $u = K x^n$,
$\mathrm{d}G/\mathrm{d}t = (u - (1+u)G)/\mu$, so

$$G \;\leftarrow\; G_\infty + (G - G_\infty)\,e^{-(1+u)\Delta t/\mu},
\qquad G_\infty = \frac{u}{1+u}.$$

For **competitive**, per target row $i$ the $2N$ gates $w = [GA_i \,|\, GR_i]$ obey
$\mathrm{d}w/\mathrm{d}t = (a - (I + a\mathbf{1}^{\mathsf T})w)/\mu$ with
$a = [K^{A}_i x^n \,|\, K^{R}_i x^n]$. The matrix $A = I + a\mathbf{1}^{\mathsf T}$ is a
**rank-one update of the identity** — eigenvalue $1$ on the $\mathbf 1$-orthogonal subspace
and $1 + S$ along $a$, $S = \sum a$ — so its exponential is closed form:

$$e^{-A\tau} = e^{-\tau}\left(I + \frac{e^{-S\tau}-1}{S}\,a\mathbf{1}^{\mathsf T}\right),
\qquad
w_\infty = A^{-1}a = \frac{a}{1+S}, \qquad \tau = \frac{\Delta t}{\mu},$$

and $w_\infty$ *is* the QSS occupancy. The substep therefore needs no linear solve and no
matrix-exponential routine; the $(e^{-S\tau}-1)/S$ factor is taken to its limit $-\tau$ as
$S\to0$. The scheme is unconditionally stable at any $\Delta t/\mu$, exact in $G$, and reduces
*exactly* to the QSS scheme as $\Delta t/\mu \to \infty$; the $-1/\mu$ stiffness never touches
the ETDRK4 $x$-substep. Measured against a $2\times10^{5}$-substep explicit Euler of the same
ODE: $<10^{-5}$.

**Exactness of the substep is not accuracy of the coupling — the trap, stated loudly.** At
$\mu_\mathrm{gate} = 10^{-3}$ (owner-set, D-REDESIGN-5) under the inherited policy
$\Delta t = 0.2/\rho(J)$, the ratio $\Delta t/\mu \sim 20$–$200$: each gate substep relaxes
*fully* to quasi-steady state, silently re-imposing QSS within a step. A genuine finite-$\mu$
oscillatory route could then be smoothed into a stationary-looking pattern — the mirror image
of the artefact the earlier deferral decision feared. Two mitigations are mandatory: the linear
verdict L1 (exact in $\mu$, no integrator involved) is a **co-gate**; and gated rollouts use

$$\Delta t = \min\!\Big(\frac{0.2}{\rho(J)},\ \frac{\mu}{2}\Big)$$

with a $\Delta t$-halving convergence check and at least one anchor run per gated model at
$\Delta t$ refined below $\mu$. The policy constant and the halving protocol are
**UNCALIBRATED** (register item 14).

Cost is estimated and unmeasured: per step $\approx 2$–$3\times$ the QSS ETDRK4 ($N$ diffusing
fields plus $2N^2$ elementwise gate fields, no extra FFTs). At the measured
$3.25\,\mathrm{ms}$/step $512^2$ CUDA scaling that projects to $\sim11$–$16\,$min per field at
$10^5$ steps — a projection for a GPU port that **does not exist** (the spatial lift is numpy
today); the port's own measurement replaces this estimate, and the port is a precondition.

### 8.5 The QSS rollout as the $\mu \to 0$ control

The existing QSS rollout (`eval/rollout.py::simulate`) is retained, **demoted from sole gate
to $\mu\to0$ control** and co-gate. It uses the same growth-rate-aware policy —
$\Delta t = 0.2/\rho(J)$, horizon $T = 40/|\sigma_{\max}|$ growth times, `max_steps` cap — and
the same amplitude floor, so lifted and QSS runs of the same model use the same timestep and
horizon and are directly comparable. `stopped_reason` is recorded for every field: a run that
ends on `step_budget` was **truncated** and its field is not a statement about the attractor.

### 8.6 The validation ladder V0–V4

Strictly ordered; each rung licenses the next, and **nothing gates until the ladder
completes**.

- **V0 — algebraic invariants.** Fixed-point residual of the lifted system at or below the
  bar the existing tests hold, $10^{-7}$; the achievable *floor* is **UNCALIBRATED** — the only
  number on record ($1.28\times10^{-8}$ over 8 systems $\times$ 7 $\mu$) sits in a parked
  artefact whose own documentation marks it do-not-cite — so V0 **measures** the floor and
  records it rather than asserting one. Also: `rescale_mu` equals a fresh autodiff Jacobian to
  round-off, and gates-at-QSS reproduce the reduced reaction exactly. Extended to $\ge 20$
  generator draws.
- **V1 — linear: eigenvalue continuation, all 21 branches.** At
  $\mu \in \{10^{-7},10^{-6},10^{-5}\}$ against the QSS dispersion on the same $k$-grid: the
  $N$ slow branches converge to the QSS eigenvalues at empirical order $\approx 1$ in $\mu$
  (slope check across the triplet); the $2N^2$ fast branches behave as $\approx -(1+u)/\mu$
  with all $\operatorname{Re} < 0$; and $k^\ast$ of the lifted dispersion lies within one grid
  point of the QSS $k^\ast$. Branches are tracked by **eigenvector overlap** between adjacent
  $\mu$, never by sorting real parts — branch crossings are exactly what the re-entrant band is
  made of. Absolute error constants: **UNCALIBRATED**, measured and recorded.
- **V2 — temporal, 0-D.** The 21-dimensional well-mixed lifted ODE against a trusted stiff
  reference (Radau, rtol $10^{-10}$): trajectories converge to the 3-dimensional QSS ODE as
  $\mu\to0$; and the Strang stepper matches the reference at
  $\mu \in \{10^{-5}, 7.2\times10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ under $\Delta t$-halving, with
  observed order $\approx 2$ where $\Delta t \lesssim \mu$ and sup-norm error at the horizon
  $\le 0.1 \times \text{pattern\_floor}$ — derived, not invented: integration error must sit an
  order below the smallest amplitude ever called a pattern.
- **V3 — spatial.** (a) Lifted versus QSS rollout at $N=3$, both forms,
  $\mu\in\{10^{-4},10^{-5},10^{-6}\}$, same seed, timestep and horizon: patterned flag equal,
  morphology class equal (at $512^2$ only), $k^\ast$ within one radial bin, and field relative
  $L_2$ difference decreasing with $\mu$ (absolute bound **UNCALIBRATED** — the measured curve
  becomes the calibration). (b) Then at $\mu_\mathrm{gate}$ with the §8.4 timestep policy
  and a $\Delta t$-halving pair — the first run in territory where the lift can say something
  new, and report-only until the ladder completes.
- **V4 — re-entrant-band survey.** `mu_critical` (with re-entrance detection) and
  `robustness_vs_mu` over every Turing-positive system on hand at single-target scope.
  Deliverables: the distribution of the first Turing-loss edge against the biological band, the
  fraction re-entrant, and the decision-relevant number
  $P(\text{lifted-Turing across } [1.1\times10^{-5},\,9.2\times10^{-3}] \mid \text{QSS-Turing})$.
  **V4 is a measurement, not a gate: no pass threshold, by design.**

`mu_critical` itself is built for non-monotonicity: it *scans* a log grid (8 points per decade)
upward from the low end, finds the first non-Turing point, and bisects that bracket in
$\log_{10}\mu$; it records whether the verdict returns at larger $\mu$ (`reentrant`) and which
condition failed (`uniform`, i.e. a Hopf/delay route if $\omega \ne 0$, versus `band`, the
$k>0$ band closing). A plain bisection seeded on the full range would see Turing at both ends
and report "never loses it", which is false.

### 8.7 The gate

Evaluated per gate-passing candidate at $512^2$ on the data box.

- **L1 — lifted linear verdict at $\mu_\mathrm{gate} = 10^{-3}$** (owner-set 2026-08-17, D-REDESIGN-5; inside the biological band, equal to `TIMESCALE_MU.md`'s headline estimate; the literature-central $7.2\times10^{-4}$ is a reported point in the band sweep)**:** strict Turing on
  the full 21-branch lifted Jacobian **and** classified `stationary` (§8.3).
- **L2 — lifted rollout patterns at $\mu_\mathrm{gate}$:** amplitude above the existing
  `pattern_floor`, with `stopped_reason == "horizon"` (a step-budget truncation is not
  evidence), under the §8.4 timestep policy and a passing $\Delta t$-halving check.
- **L3 — wavelength:** $k^\ast$ of the lifted rollout within **one radial bin** of
  $k^\ast_\mathrm{obs}$. **The formula is the bar:**
  $$|k^\ast_\mathrm{lift} - k^\ast_\mathrm{obs}| \;\le\; \Delta k \;=\; \frac{2\pi}{L} .$$
  On this target that is a **measured $\approx 12.0\,\%$** of $k^\ast_\mathrm{obs}$
  ($L = 185.006$, $k^\ast_\mathrm{obs} = 0.283878$, $\Delta k = 0.033962$, i.e. $8.36$ periods
  in the box — $0.033962/0.283878 = 11.96\,\%$). The familiar $1/p = 12.5\,\%$ is the value at
  the *nominal* $p = 8$ and is **not** the number the gate applies; quoting it as an
  equivalence is arithmetically false here, because the box holds $8.36$ measured periods, not
  $8$. The bar is one *whole* bin: the $8.3\,\%$ figure in older documents is *half* a bin,
  derived on the legacy `three_gene` sets, and its import here was explicitly rejected as
  sub-resolution at $p=8$ (D-FFT-3). See `docs/PREREGISTRATION.md` §3.7 and
  `docs/REDESIGN_rngrn.md` §4.3/§5.3/§5.4, corrected formula-first 2026-08-17, and Appendix A7.

  **Margin over the trivial predictor, both figures.** On *this target* the tracked control is
  `trivial_kstar_fft_err_mean` $= 0.294$ (`experiments/diag_fft/d5/target_reports.jsonl`), i.e.
  $\approx 2.4\times$ one bin. The $33$–$37\,\%$ figure quoted elsewhere is the
  **canonical-set** number, not this sample's, and gives $\approx 3\times$. Both are stated
  because the gate is applied to this target and the canonical figure is not its control.

**Reported, never gated:** the L1 verdict at 9 log-spaced $\mu$ across the band plus both
endpoints (near-free via `rescale_mu`); `mu_critical` with the re-entrance flag; the drawn-$\mu$
robustness row; the morphology class of the lifted field against the target; and the
QSS-versus-lifted field difference. Morphology classification is **report-only throughout**:
the classifier was measured at $57.1\,\%$ stripe recall in-bank and $33.3\,\%$ on held-out
$n=3$, which is too weak to gate.

**Gate at $\mu_\mathrm{gate}$, report across the band.** $\mu$ is a three-decade
*uncertainty band* on a literature-derived ratio, not a swept parameter. Gating band-wide would
turn one uncertain quantity into a conjunction of claims biology does not license, and the
measured re-entrance means an edge point can flip on the least-certain digit. Band-wide gating
is the materially stronger claim that remains available and is revisited after V4. **The QSS
criteria remain binding co-gates**: the lift must agree with QSS as $\mu\to0$ anyway, so they
cost nothing and preserve comparability with the baseline.

**Null and numerics controls.** L1–L3 run identically on the null ensemble of §7.3. The
candidate must pass; the null pass-fraction is *reported* — and a high null pass-rate is itself
the finding that the lifted gate does not discriminate mechanism. Per gated model: the
$\Delta t$-halving check (mandatory), 5-seed initial-condition replication of the lifted
rollout, and one independent stiff cross-check field by a BDF1/Newton–Krylov integrator whose
current **silent fallback** to ETDRK4 is removed and made loud, unconditionally.

### 8.8 Training–verification coupling

A periodic **non-differentiable** lifted-linear audit runs inside training: one lifted Jacobian
at $\mu=1$, `rescale_mu` to $\mu_\mathrm{gate}$, and an eigenscan of the $21\times21$ across
a coarse $k$-grid — cheap enough to run every $M$ steps (cost to be measured, not asserted). It
is used for **checkpoint selection** — prefer lifted-Turing checkpoints, reject
QSS-Turing-but-lift-dead candidates before expensive verification — and **not** as a loss term.
The reason is numerical: non-symmetric eigendecomposition gradients are ill-conditioned exactly
at the eigenvalue crossings that define this system's $\mu$-structure, so a differentiable
penalty there is fragile machinery bolted onto the degenerate points that matter. The
escalation rule is tied to V4: if the survey shows $P(\text{lifted-Turing} \mid
\text{QSS-Turing})$ high (working notion $\gtrsim 0.95$, **UNCALIBRATED**, set from V4 itself)
the audit stays diagnostic; if materially lower it becomes a hard model-selection filter; a
differentiable proxy is last resort and would be an objective change.

---

## 9. Evaluation design

### 9.1 What is gated and what is reported

| | gated | reported only |
| --- | --- | --- |
| **Linear (QSS)** | strict Turing verdict (§3.2) | $\sigma$ curve, robustness cloud |
| **Forward (QSS)** | patterns above the amplitude floor; $k^\ast$ within one radial bin | morphology class |
| **Spectral** | held-out-band `spec_shape` distance beating **every** member of the $\ge6$-member null ensemble | train-band distances, per-term traces |
| **Channels** | per-channel amplitude-ratio log-error (channels 1–2 against channel 0) beating **every** null-ensemble member (`docs/SPEC_fourier_training.md` §9.6, **BINDING at Stage 0**) | the raw log-errors; and the recorded caveat that channels 1–2 of this sample are near-flat (cv $\approx 0.075$) though D3 measures a $k^\ast$ on both within one radial bin |
| **Lifted** | L1, L2, L3 at $\mu_\mathrm{gate} = 10^{-3}$ (after the ladder; D-REDESIGN-5) | band-wide L1, `mu_critical`, robustness-vs-$\mu$, lifted morphology |
| **Cross-$L$** | — (see the disclosure below) | small-box transfer, mode-selection distribution |
| **Plausibility** | — | $\alpha,\delta$ reported as `structural`; $D$-ratio and $\beta$ as measured/unscored |

**Disclosure on the QSS $k^\ast$ row.** That gate uses the **one-bin** bar of D-FFT-3 /
`docs/SPEC_fourier_training.md` §9.1, which is **looser** than
`docs/PREREGISTRATION.md` §3.3's pre-registered $\le 8.3\,\%$ (half a bin, derived on the legacy
`three_gene` sets). The change is an owner decision on record (D-FFT-3, 2026-08-11);
`PREREGISTRATION.md` §3.3 now carries a cross-reference note (2026-08-17) recording which bar
applies to which criterion. Any report quoting either must say which it used.

**Disclosure on the cross-$L$ row.** `docs/PREREGISTRATION.md` §3.5a is a **PRIMARY**
pre-registered criterion with numeric bars, and this table reports rather than gates it. That is
a change to a pre-registered pass condition's *binding status*, and it is **escalated to the
owner, not settled here**. There is also a set difference that no calibration covers: §3.5a
specifies $L \in \{0.5, 1, 2, 4\}\times L_\mathrm{train}$ with ground truth on
`three_gene_multiL`, whereas the small-box protocol of §1.7 spans $p \in \{2,3,4\}$, i.e.
$\{0.25, 0.375, 0.5\}\times L_\mathrm{train}$, against *new realizations of the target*. Neither
the bars nor their calibration have been re-derived for that set.

Selection uses fitted-band loss only. The held-out bands are consulted exactly once, by scoring
code, at gate time.

### 9.2 The deliverable is an ensemble

The recovered object is not a point estimate. It is two things.

**The claim** is the **modal recovered sign structure**, scored over *the same population the
baseline used* — every seed that produced a scored recovery, ignited or not — so that the
number is read against the baseline's $0.1$. A gate-passers-only variant is reported alongside,
labelled as the conditional statistic it is. Neither is a preregistration §3.1 verdict, because
§3.1 requires independent seeds and a cross-target control while the diversity cull makes a
Phase-II population non-independent *by construction*.

**The uncertainty** is the distribution — median and decade spread — of a named set of
**time-scale-invariant combinations** over the gate-passing members. These are exactly the
quantities that survive the time gauge of §2.6, and each is defined here:

| combination | definition | why invariant |
| --- | --- | --- |
| $d\text{-ratio}$ | $D_{(N)}/D_{(N-1)}$, the two most mobile species | ratio of two $D$'s; the common factor $c$ cancels |
| $\alpha/\delta$ | entrywise $\alpha_{ij}/\delta_i$ | both scale as $c$ |
| $\beta/\delta$ | $\beta_i/\delta_i$ ($\beta$ now derived, §2.5) | both scale as $c$ |
| $D k^{\ast 2}/\delta$ | $D_i (k^\ast)^2/\delta_i$ | a diffusive rate over a decay rate |
| per-edge occupancy | $\theta_{ij} = K^{A}_{ij}x_j^{\ast n}\big/\big(1 + \sum_l (K^{A}_{il}+K^{R}_{il})x_l^{\ast n}\big) \in [0,1]$ (competitive; the independent-site analogue for nc1) | dimensionless promoter occupancy at the operating point; $K$'s carry no time dimension |

This invariance is not assumed: it was measured, with the combinations unchanged to
$\le2\times10^{-16}$ under $(f,D)\to(cf,cD)$ — carrying the caveat recorded at the measurement
and restated in §2.6, that scaling $\mathbf f$ by $c$ is not exactly expressible inside the
softplus parameterisation, so the analytic check builds $c\,(J - k^2 D)$ directly while the
expressible $D$-half (`theta_D += log c`) matches to $3.2\times10^{-12}$
(`docs/DIAGNOSTICS_fft.md` D6, Check B).

**$K^{A}$ and $K^{R}$ themselves are still reported in raw decades**, alongside the bounded
occupancy. They were the two worst baseline spreads ($8.0$ and $10.9$ decades) and nothing in
the redesign touches them; substituting occupancy alone would remove exactly the worst numbers
from view.

### 9.3 Reproducibility: the unit of independence

**Ruled (register item 13): the unit of independence is the full pipeline replicate.**
Reproducibility statistics are computed across the winners of $K_\mathrm{rep} = 5$ independent
Phase I → cull → Phase II replicates run from independent master seeds. Members inside one
replicate's population are **never** counted as independent seeds. The rejected alternative —
treating culled members as seeds — is biased in exactly the direction of the statistic being
measured, because the cull selects members *for* sign-structure distinctness.

### 9.4 The data firewall

Recovery never sees ground truth, and this is enforced mechanically rather than by convention:
`tests/test_firewall.py` is a **static AST audit** that parses each recovery-side module and
asserts that a list of forbidden names does not appear among its imports — the generator, the
solver, the dataset registry and cache, the answer-key type, and every payload-reading script.
A completeness test additionally asserts that every module under the loss and evaluation
packages is *classified* as recovery-side, scoring-side or side-neutral, so a new module is
never unaudited by default. $k^\ast_\mathrm{obs}$ from an FFT of the observed frame is legal;
anything derived from a generating parameter is not. The answer key's contents are read in
exactly one place, the scoring function, and the training entry point holds the key only to
classify the experiment arm. Every new training-side component introduced by this design —
solve-box logic, unrolled gradients, the lifted audit — is recovery-side and touches only
observation-derived quantities; the lifted *gate evaluation* joins the scoring side.

### 9.5 Overfitting axes: realization consistency and cross-$L$

Two checks probe different failure modes and neither is a recovery claim.

**Realization consistency** (§1.7): the fitted model's spectra must match the two held-out
realizations as well as they match the six trained ones. Because the realizations differ only
in initial-condition noise, a pass says the fit captured the *system's statistics* rather than
one realization's defect layout.

**Small-box cross-$L$** (§1.7): the recovered model is simulated at $p\in\{2,3,4\}$ with no
refit and compared against the generator's own held-out small-box realizations on physical
$k^\ast$ invariance and the periods-per-box slope. The pre-registered form of this test
(`docs/PREREGISTRATION.md` §3.5a) requires $k^\ast$ coefficient of variation $\le 0.10$ across
box sizes, a fitted periods-per-box-versus-$L$ slope within $15\,\%$ of
$k^\ast_\mathrm{phys}/2\pi$, and morphology class preserved on $\ge 0.75$ of domains — the last
set at $3/4$ rather than $1$ precisely because the morphology classifier is weak on held-out
stripes.

**Two disclosures belong with it, and neither is settled in this document.** First, §3.5a is a
**PRIMARY** pre-registered criterion with those numeric bars, while the first campaign *reports*
rather than gates it (§9.1); that is a change to a pre-registered pass condition's binding
status and is **escalated to the owner, not decided here**. Second, the box sets differ: §3.5a
specifies $L \in \{0.5, 1, 2, 4\}\times L_\mathrm{train}$ with ground truth supplied by
`three_gene_multiL` (the same systems simulated at four real domain sizes), whereas the protocol
above spans $p \in \{2,3,4\}$, i.e. $\{0.25, 0.375, 0.5\}\times L_\mathrm{train}$, against new
realizations of the *target* system. The three bars were derived for the former set; **neither
the bars nor their calibration have been re-derived for the latter**, so a pass or failure under
this protocol is not a §3.5a verdict.

### 9.6 The comparison row

The redesign arm is read against the **D5 baseline** and against the null ensemble, never
against zero. The baseline is
`experiments/diag_fft/d5/target_reports.jsonl`, git_sha `1b9ecda`, 2026-08-12, 10 seeds on
`turing_labyrinth/sample_0000` under the current default objective **with `lbfgs_steps=50`
(LBFGS polish active, as in all ten D5 frozen configs)**, with: $4/10$ seeds
Turing-unstable; $10/10$ **distinct** recovered sign structures ($0/45$ pairwise agreements,
`mean_agreement` $0.499 \approx$ the $0.5$ expected under independent $\pm1$ signs given the
row's $5\,\%$ zero-call rate); `kstar_fft_rel_err` median $0.977$; `topology_consistency`
$0.1$; recovered $\alpha$ and $\delta$ inside the plausibility box in $0/10$ seeds; recovered
$d$-ratio in-box in $10/10$.

**What that $k^\ast$ median is a median of.** The population is bimodal, not spread: the four
Turing seeds each land at `kstar_fft_rel_err` $\approx 0.014$, while the six non-Turing seeds sit
at the identical value $0.9769$ — a grid floor, not a measurement of error. The median $0.977$ is
therefore *dominated by the six non-Turing seeds pinned at that floor*
(`docs/DIAGNOSTICS_fft.md`:208) and must not be read as a typical error of a recovered model.

**Comparability is stated up front, and is narrower than the row.** Three columns are
comparable and are the ones the redesign is read against: Turing fraction, number of distinct
sign structures, and $k^\ast$ error. **Not comparable, and never to be reported as an
improvement over the baseline:** the $\alpha/\delta$ plausibility columns (structural by §2.6),
any spectral-target-derived quantity (single frame versus the multi-realization average of
§1.7), and `topology_consistency` unless computed over the same population as the baseline's.

Finally, and bindingly: the first campaign runs on a sample already burned for tuning.
**Work on that sample supports no recovery claim and no generalisation claim.** The licensed
sentence class is *"the redesign's objective changes the measured baseline statistics on the
tuning sample, in this direction, by this much, against the D5 row and the null ensemble."* A
recovery claim requires a separate, held-out campaign under a frozen configuration.

---

## Appendix A — Discrepancies found while verifying this draft

Recorded rather than silently resolved, per the project's evidence rules. Each names what the
source says and what the document says.

**A1. `lbfgs_steps` — RESOLVED 2026-08-17. A design-document claim contradicted by the tracked
run records; `docs/REDESIGN_rngrn.md` §4.5 has since been corrected.** The evidence, retained:
§4.5 previously stated that "**No measurement of `lbfgs_steps` exists in this repo** (every
recorded run sets it to 0)". Against the source: `recover.recover`'s own default
is `lbfgs_steps=50` (`src/rngrn/recover.py:289`) and LBFGS runs whenever the value is non-zero
(`recover.py:600–601`). Against the records: of the 11 tracked `frozen_config.yaml` files under
`experiments/`, **10 set `lbfgs_steps: 50`** and one sets 0 — including all ten D5 baseline
seeds, whose frozen config carries `lbfgs_steps: 50` with `batched: false`. The lone 0 is a
`tune_comp` run with `batched: true`, which `recover.py:504–506` *requires* to be 0. The D5 row
was therefore produced *with* an LBFGS polish. This does not change the D5 numbers, but it does
change what "the current default objective" means and it invalidated the premise that nothing
rests on the knob. §4.5 now records the corrected position; this draft states the LBFGS setting
explicitly wherever the baseline row or the A0 objective is quoted (§5.1, §7.4, §9.6).

**A2. Two different central $\mu$ values from the same two citations.**
`docs/TIMESCALE_MU.md` §2 names $\mu = 10^{-3}$ (headline) and $10^{-2}$ (stress), from
$10\,\mathrm{s}/2\times10^{4}\,\mathrm{s} \approx 5\times10^{-4}$ "rounded up one notch".
`src/rngrn/eval/lifted.py` (MU_BIO_*) gives $\mu_\mathrm{central} = 7.2\times10^{-4}$ and the
band $[1.1\times10^{-5},\,9.2\times10^{-3}]$, from the same Chen 2014 and Müller 2012 sources
but with the model-time-unit factor $\delta_\mathrm{model} \in [0.4,5]$ carried explicitly. The
band is the reported sweep; the gate point is $\mu_\mathrm{gate} = 10^{-3}$, set by the
owner (D-REDESIGN-5, 2026-08-17) — coinciding with TIMESCALE_MU's headline — and this
draft follows that record. The two are reconcilable (they differ by the $\delta$ factor and a rounding), but they
are **two separate calibration passes reported as one number** in places, and the older
headline $10^{-3}$ should not be quoted alongside the band without saying so.

*Addendum — a third discrepancy in the same artefact.* `docs/TIMESCALE_MU.md`:50 converts the
Müller half-lives of $95$–$218\,$min to "mean lifetimes $\sim 1.4$–$1.9\times10^{4}\,$s". The
upper end is right, the lower end is not: $t_{1/2}/\ln 2$ gives $8224\,$s and $18873\,$s, so the
span is $8.2\times10^{3}$–$1.9\times10^{4}\,$s. The code carries the correct pair
(`eval/lifted.py`'s `MU_BIO_*` band derives from $8.2\times10^{3}$), and §8.2 of this draft
follows the code, not the doc line.

**A3. `docs/TIMESCALE_MU.md` asserts a confirmation whose tables are unfilled.** §1 of that
document states "the measurement CONFIRMS it", but §4, §5, §6 and §8 contain only unexpanded
template placeholders (`<!--RESULTS_TABLE-->`, `<!--LADDER_TABLE-->`, `<!--DTCONV_TABLE-->`,
`<!--OSC_LINE-->`, `<!--DISSENT-->`, `<!--CONTEXT-->`). No populated results table exists in the
tracked file. This is consistent with the redesign's own characterisation of the artefact as
parked and do-not-cite, and it is why §8.6's V0 *measures* the residual floor rather than
inheriting $1.28\times10^{-8}$. **No finite-$\mu$ robustness result is cited in this draft.**

**A4. Components described by the design that are not in the source.** Verified absent:
`kstar_si` (no such term anywhere in `losses/terms.py`); fixed-point pinning and the derived
$\beta$ (`model.py` still has $\beta=\operatorname{softplus}(\theta_\beta)$ as a free
parameter); the box-sigmoid reparameterization of $\alpha,\delta$; per-bin inverse-variance
weights in `spec_shape` (no weight array, no hook, `.sum()` with uniform implicit weight); the
$B_\mathrm{low}$/$B_\mathrm{harm}$ machinery — precisely, `losses/spectral.py` carries the
*generic* `band_mask(k_centers, kstar_obs, lo, hi)` and `SpectralConfig.b_lo`/`b_hi` (mirrored as
`LossConfig.spectral_b_lo`/`spectral_b_hi`), so the band *mechanism* exists while the named
$B_\mathrm{low}$/$B_\mathrm{harm}$ constants and any gate-time consumer of the held-out bands do
**not** exist anywhere in `src/`; both live in `scripts/diag_fft_d3.py`,
`scripts/diag_fft_d4.py` and `docs/SPEC_fourier_training.md` §9.5; the adaptive commensurate solve box; the
truncated-unrolled gradient path; and a batched form of the spectral terms or the forward solve
(`compute_terms_batched` **raises** on them rather than approximating). Each is marked
*specified* at its point of use above.

**A5. `kstar_anchor` accepts a `tau` argument it never uses.** `losses/terms.py::kstar_anchor`
takes `tau=0.12` and the configuration exposes `loss.tau` (documented as the "$k^\ast$
tolerance band"), but `tau` appears nowhere in the function body — only `temp` enters the
formula. Any sweep of `loss.tau` against this term is a no-op. Flagged, not changed.

**A6. Band-edge drift between diagnostics.** `scripts/diag_fft_d3.py` uses the *provisional*
$B_\mathrm{train} = [0.5, 1.5]$ and $B_\mathrm{harm} = [1.5, 3.0]$, while
`scripts/diag_fft_d4.py` and `losses/spectral.py` use the **closed** $[0.60, 1.55]$ (D-FFT-9
closure 1), with $B_\mathrm{harm} = [1.55, 3.0]$. The $\sim31\,\%$/bin estimation floor quoted
throughout was therefore measured on the **provisional** $[0.5,1.5]$ band. It is used here as an
order-of-magnitude floor, which is how the design uses it, but the two band conventions should
not be conflated when the floor is quoted quantitatively.

**A7. The nominal $p=8$ and the measured $8.36$ — RESOLVED 2026-08-17, formula-first.** The
target's box holds $8.36$ periods of the *observed* $k^\ast$ ($L = 185.006$,
$k^\ast_\mathrm{obs} = 0.283878$, $\Delta k = 0.033962$), so one radial bin is
$0.033962/0.283878 = 11.96 \approx 12.0\,\%$ of $k^\ast_\mathrm{obs}$. Earlier versions of this
draft and of several repo docs stated the bar as "$\le 2\pi/L \Longleftrightarrow 1/p = 12.5\,\%$
at $p=8$", and that equivalence is **arithmetically false on this target**: $12.5\,\%$ is the
value at the *nominal* integer $p$, not at the $8.36$ periods the box actually holds.

**The bar is the formula $|\Delta k^\ast| \le 2\pi/L$**, with $\approx 12.0\,\%$ as its measured
value here and $12.5\,\%$ as a nominal-$p$ approximation that the gate does not apply. The
$8.3\,\%$ figure is the legacy half-bin on the `three_gene` sets and its import was rejected
(D-FFT-3). `docs/PREREGISTRATION.md` §3.7, `docs/REDESIGN_rngrn.md` §4.3/§5.3/§5.4 and D-LIFT-1
were all corrected formula-first on 2026-08-17 and now state it this way.

*Residual drift, controller-ledgered and owner-facing.* `docs/SPEC_fourier_training.md`
§9.1:383–384 and D-FFT-3 (`docs/DECISIONS.md`:2606) still present the bar as
"rel-err $\le 1/p$ — $12.5\,\%$ at $p=8$" *without* noting that this target holds $8.36$ measured
periods, so the number they quote is the nominal one. Both are formula-first in form; what they
lack is the measured-$p$ correction. They are not edited here.

The $0.5$-percentage-point gap is worth keeping visible for a second reason: it is the *same*
incommensurability that F-D1-5 identifies as the cause of the Newton stall (§6.4), and it should
not be rounded away in either place.

**A8. Two `morphology` weight defaults.** `losses/terms.py::DEFAULT_WEIGHTS` sets
`morphology=0.0` while `config.py::LossConfig.weights` sets `morphology=0.1`, and the D5 frozen
configs carry $0.1$. The term is inert either way — `morphology_consistency` is a
non-differentiable numpy diagnostic and is not in the differentiable sum — so no number changes,
but the redesign's removal of the key "for clarity, no behaviour change" should remove **both**.

**A9. Bin-edge construction differs between the numpy and torch RAPS.**
`observables.raps` builds bin edges as `np.arange(0, KR.max()+dk, dk)` (floating-point step)
whereas `losses/spectral.py::raps_torch` builds them as `arange(n_edges) * dk` — the integer
form adopted under finding F-D6-1 after a 1-ulp lattice-edge mismatch on 63 of 364 edges. The
two were then measured equivalent bin-for-bin to $\le6.1\times10^{-16}$ in power and
$1.8\times10^{-15}$ in $k^\ast$, so this is benign today. It is recorded because the
estimator-pair rule (§4.3) makes any divergence between the measurement-side and
differentiable-side estimators a correctness issue by construction, not a style issue.

