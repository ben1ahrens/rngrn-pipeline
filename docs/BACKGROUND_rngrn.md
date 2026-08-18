# Background theory

**Status: draft for the paper.** Theory section, following the Introduction and preceding
Methods: reaction–diffusion and the Turing instability, mode quantisation, Hill/Shea–Ackers
kinetics, the quasi-steady-state reduction, and the identifiability structure of the inverse
problem. Established results are derived; this project's own measurements are labelled as such.
No results claims, no motivation or contributions framing (those belong to the Introduction),
no methods detail.

---

## 1. Reaction–diffusion systems and the Turing instability

Take $N$ species with concentrations $x(r,t) \in \mathbb{R}^N$ on a periodic domain
$\Omega = [0,L)^2$, evolving under

$$\frac{\partial x}{\partial t} \;=\; D\,\nabla^2 x \;+\; f(x), \qquad D = \mathrm{diag}(D_1,\dots,D_N),$$

with $f$ the pointwise reaction and $D$ diagonal and positive: species diffuse, they do not
co-diffuse. A *homogeneous steady state* is a constant $x^\*$ with $f(x^\*)=0$. Turing's (1952)
observation is that such a state can be stable to uniform perturbations and unstable to
spatially structured ones. Write $x = x^\* + u$ and keep first order:

$$\frac{\partial u}{\partial t} \;=\; D\,\nabla^2 u \;+\; J u \;+\; O(\lVert u\rVert^2), \qquad J = \left.\frac{\partial f}{\partial x}\right|_{x^\*}.$$

The linearised operator has constant coefficients and the domain is periodic, so Fourier modes
diagonalise the spatial part: substituting $u(r,t) = \hat u_k(t)\,e^{i k\cdot r}$ and using
$\nabla^2 e^{ik\cdot r} = -\lvert k\rvert^2 e^{ik\cdot r}$ decouples the modes into
$N$-dimensional linear ODEs,

$$\frac{d\hat u_k}{dt} \;=\; \bigl(J - k^2 D\bigr)\hat u_k, \qquad k \equiv \lvert k\rvert,$$

whose growth rate is set by the rightmost eigenvalue. Defining the **dispersion relation**

$$\sigma(k) \;=\; \max_i \operatorname{Re}\,\lambda_i\!\left(J - k^2 D\right)$$

the two strict **Turing conditions** are

$$\sigma(0) < 0 \quad\text{(the uniform state is stable)}, \qquad \exists\,k>0:\ \sigma(k) > 0 \quad\text{(some finite mode grows)} .$$

Diffusion, dissipative alone, has *destabilised* the system, by short-range activation with
long-range inhibition (Gierer & Meinhardt 1972): a self-reinforcing activator amplifies a local
fluctuation, an inhibitor it drives suppresses the surroundings, and how much further the
inhibitor reaches sets the length scale. That "further" is a strict requirement, not a picture.
For $N=2$ with
$J = \begin{psmallmatrix} a & b \\ c & d\end{psmallmatrix}$, stability at $k=0$ means
$a+d<0$ and $\det J>0$. Since $\operatorname{tr}(J-k^2D) = a+d-k^2(D_1+D_2) < 0$ for all $k$,
instability requires $\det(J - k^2 D) < 0$ for some $k$, i.e.

$$D_1 D_2 k^4 \;-\; \bigl(D_2 a + D_1 d\bigr)k^2 \;+\; \det J \;<\; 0 ,$$

which, the quartic being positive at $k=0$ and as $k\to\infty$, needs $D_2 a + D_1 d > 0$ and
$(D_2 a + D_1 d)^2 > 4 D_1 D_2 \det J$. Given $a+d<0$, the first forces opposite signs (say
$a>0>d$: species 1 self-activates, 2 self-inhibits) and then $D_2 a > D_1\lvert d\rvert$, so
$D_2/D_1 > \lvert d\rvert/a > 1$, since $a+d<0$ already gives $\lvert d\rvert > a$.
**Differential diffusivity is necessary**, and the inhibitor must be the faster diffuser. For
$N\ge3$ the bound weakens — indirect inhibition through a third species relaxes it — but a
D-ratio above one remains the mechanism's structural signature.

## 2. Finite domains and mode quantisation

The derivation above treats $k$ as continuous; a finite box does not. On $[0,L)^2$ with
periodic boundaries the admissible wavevectors form a lattice,

$$k = \frac{2\pi}{L}\,(m_1, m_2), \quad m \in \mathbb{Z}^2, \qquad \lvert k\rvert = \frac{2\pi \lvert m\rvert}{L},$$

and the realised pattern must snap to it. A model can be Turing-unstable at a $k$ the domain
cannot support; in the linear picture the realised wavelength is then the admissible mode nearest
the linear optimum, though which mode actually wins is a nonlinear question, and predicted and
realised wavelength diverge as the number of periods per box falls. Worse,
linear theory ranks growth rates at $t=0^+$ and says nothing about which competing unstable mode
ultimately wins, nor where growth saturates: **selection and amplitude are nonlinear phenomena**.
"Turing-unstable" and "patterns" are therefore distinct claims, and only simulation of the full
nonlinear system decides the second.

## 3. Mechanistic GRN models: Hill kinetics and promoter occupancy

To make every learned quantity a biological one, $f$ is built from gene-regulatory kinetics
rather than a generic polynomial. For a transcription factor $X$ binding a promoter site $S$
with cooperativity $n$, $S + nX \rightleftharpoons SX_n$, binding equilibrium with association
constant $K$ gives

$$[SX_n] = K\,[S]\,x^n \quad\Longrightarrow\quad \theta \;=\; \frac{[SX_n]}{[S]+[SX_n]} \;=\; \frac{K x^n}{1 + K x^n},$$

the **Hill function** — occupancy saturating at 1, half-occupied at $x = K^{-1/n}$, steepness
set by $n$.

With several regulators the bookkeeping is the statistical-thermodynamic one of Shea & Ackers
(1985): enumerate promoter microstates, weight each by its Boltzmann factor, normalise by the
partition function. Regulators competing for one shared site give mutually exclusive microstates
and hence a shared denominator,

$$P_i(x) \;=\; \frac{\sum_j \alpha_{ij} K^A_{ij} x_j^{\,n}}{1 + \sum_j \bigl(K^A_{ij} + K^R_{ij}\bigr) x_j^{\,n}},$$

so any bound regulator, activating or repressing, excludes the others. If instead the sites are
independent, each carries its own Hill occupancy and contributions combine additively across
activators and multiplicatively across repressors,

$$P_i(x) \;=\; \Bigl(\textstyle\sum_j \alpha_{ij}\,\theta^A_{ij}\Bigr)\prod_j \bigl(1 - \theta^R_{ij}\bigr), \qquad \theta^{A/R}_{ij} = \frac{K^{A/R}_{ij} x_j^{\,n}}{1 + K^{A/R}_{ij} x_j^{\,n}} .$$

The two forms encode genuinely different promoter architectures — competitive occlusion versus
independent-site logic — and are not reparameterisations of one another. Closing the model with
basal production, first-order turnover and diffusion,

$$\frac{\partial x_i}{\partial t} \;=\; D_i \nabla^2 x_i \;+\; \beta_i \;+\; P_i(x) \;-\; \delta_i x_i ,$$

every parameter is a measurable biological quantity: binding strengths $K^A, K^R$, maximal
production rates $\alpha$, basal rates $\beta$, degradation rates $\delta$, diffusivities $D$.
That is what makes a recovered parameter set a circuit rather than a fit, and why the sign
structure of the regulatory matrix is the object of scientific interest.

The differentiable form of this construction is not new here: it adapts **Recurrent Neural
Chemical Reaction Networks** (Dack 2024, arXiv:2406.03456), whose gated-promoter reaction is
carried over essentially unchanged; what this work adds is the spatial (diffusive) part and the
inverse problem posed on a static frame (`src/rngrn/model.py` records the antecedent at its head).

## 4. Quasi-steady-state reduction and timescale separation

Writing $P_i$ as an algebraic function of $x$ already assumes that promoter occupancy
equilibrates instantaneously relative to protein turnover. Transcription-factor residence on DNA
is measured in tens of seconds and morphogen lifetimes in patterning tissue in hours, so the
separation is real but finite. Making the gates explicit restores a fast subsystem and gives the
generic two-timescale form, written here **schematically**,

$$\frac{\partial x}{\partial t} = D\nabla^2 x + f(x,g), \qquad \mu\,\frac{\partial g}{\partial t} = g_\infty(x) - g,$$

with $\mu$ the ratio of the two characteristic times. The schematic form suppresses an
occupancy-dependent relaxation rate that the implemented lift carries explicitly: each gate
relaxes at $(1+u_{ij})/\mu$ in the independent-site form and at $\{1,\,1+S_i\}/\mu$ (the
eigenvalues of a rank-one update of the identity, $S_i$ the row's total binding weight) in the
competitive form, rather than at a uniform $1/\mu$ (`src/rngrn/eval/lifted.py`). The singular
limit $\mu\to0$ collapses the
fast variables onto the slow manifold $g = g_\infty(x)$, leaving
$\partial_t x = D\nabla^2 x + f(x, g_\infty(x))$ — exactly the algebraic model of §3.
Tikhonov–Fenichel theory licenses this on finite time intervals provided the fast subsystem's
equilibrium is hyperbolically stable.

The caveat is that the reduction constrains trajectories, not the *stability classification* of
the full system. Reduced and full systems share their steady states, but the full Jacobian is
larger — the gate directions add coordinates — and eigenvalues living there are absent from the
reduced spectrum. They can cross the imaginary axis at finite $\mu$, typically as a Hopf
bifurcation turning a stationary pattern into a travelling wave, and nothing in the reduction
forbids it. Turing instability established under the QSS algebra is therefore established only
in the limit; establishing it under the *lifted* dynamics at finite, biologically defensible
$\mu$ is a strictly stronger statement about the same circuit.

## 5. The inverse problem and its identifiability structure

The forward map, parameters to pattern, is a simulation; its inverse is not, and what it can
resolve is bounded before any estimator is chosen. Matas-Gil & Endres (*iScience*, 2024;
arXiv:2309.06339), fitting Schnakenberg, FitzHugh–Nagumo and Brusselator systems by
least-squares and by physics-informed neural networks, establish the structural point:
**position and shape are set by the initial condition while wavelength and pattern type are set
by the model**, so comparison belongs in the frequency domain — they use the radially averaged
power spectrum (RAPS), not pixel space, with an unknown image intensity scale absorbed into a
free rescaling variable. Two claims often attached to that paper's PINN are not PINN results,
corrected here per this project's audited record (`docs/STATE_OF_THE_SCIENCE.md` §2.7). The
*"<10 % parameter error"* figure and the treatment of the inverse map's stochasticity as a
*feature* yielding parameter alternatives belong to Kho et al. (*Design of Turing Systems with
Physics-Informed Neural Networks*, IEEE SSCI 2022, doi:10.1109/SSCI51031.2022.10022026,
arXiv:2211.13464); Matas-Gil & Endres quote no percentage PINN error at all.
The *"3×3 pixel region suffices"* claim and the $N^{-1/2}$ error scaling **are** theirs, but are
least-squares results — ordinary $\sqrt{N}$ averaging of a linear estimator, licensing no
inference about PINN data efficiency — and the paper is self-inconsistent on the pixel count
(3×3 in the Fig. 3 caption, 4×4 in the body; the real minimum pixel count is a rank condition).
Both works assume the candidate model is the true model
and, in their synthetic benchmarks, that all species are observed; neither holds for a GRN of
unknown wiring seen through a subset of its species.

What a static frame can determine is then bounded exactly. **The degeneracies below are this
work's own measurements on its own objective, not literature results.** An objective seeing the
parameters only through $\sigma(k)$
is invariant under transposition, since $\operatorname{spec}(M^\top)=\operatorname{spec}(M)$ and
$(J-k^2D)^\top = J^\top - k^2 D$ for diagonal $D$; under diagonal similarity $J\mapsto SJS^{-1}$,
which commutes with $D$ and so leaves $\operatorname{spec}(J-k^2D)$ untouched, a two-parameter
continuous gauge at $N=3$; and under node relabelling. Measured on real recovered $(J,D)$ at
$\max\lvert\Delta\sigma\rvert \approx 10^{-15}$, a dimension count indicating these exhaust the
blindness rather than sample it (`docs/DECISIONS.md` D-C1-GAUGE). Separately, a static frame
carries no time unit, so $(f,D)\mapsto(cf,cD)$ sends $\sigma(k)\mapsto c\,\sigma(k)$ with the
argmax unmoved: invisible in the data, confirmed analytically and dynamically, with only
time-scale-invariant combinations ($D$-ratios, $\alpha/\delta$, $\beta/\delta$, binding
constants, $Dk^{*2}/\delta$) surviving (`docs/DIAGNOSTICS_fft.md` D6). Quotienting by the whole
group was measured *not* to reconcile independently recovered networks (D-C1-GAUGE); separately,
distinct circuits were measured to match the fundamental $k^\ast$ while differing strongly in
held-out spectral bands (`docs/DIAGNOSTICS_fft.md` D4), so the multiplicity is neither a gauge
artefact nor an equivalence of dispersion relations.

Underdetermination compounds the gauges: when the objective sees only a few summary statistics —
a peak wavenumber, a spectral shape — the parameter count far exceeds the constraint count, and
the set consistent with the data is a manifold rather than a point. Identifiability is therefore
bounded twice over: exactly, by the invariance group of a $\sigma(k)$-only objective, and
statistically, by the number of independent constraints the observables carry.

---

*Prose word count, excluding the status header, section headers, display equations, inline math
and this line: **1353** (measured, not estimated; was 1172 before the 2026-08-17 evidence-audit
corrections, which added the Kho/Matas-Gil attribution split, the antecedent credit in §3, the
lifted-gate relaxation-rate caveat in §4, and the held-out-band clause in §5). Target was ~1000;
the overage sits in §5, whose literature paragraph carries the attribution corrections, and in
§3–§4, whose additions are provenance rather than exposition. If the Introduction already covers
the prior art, §5's first paragraph can be cut to its structural claim (frequency-domain
comparison, because position and shape are initial-condition-determined) and the section loses
roughly 150 words.*
