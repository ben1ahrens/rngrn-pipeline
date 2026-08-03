# PREREGISTRATION — the pass conditions, fixed before the runs that are judged by them

**Status: binding.** Written and committed *before* any convergence run. Every threshold
below is stated with the measured baseline it was calibrated against and the control it is
read against. Nothing here may be revised after seeing a convergence result; if the
pipeline falls short, the gap gets reported and this file stays as written.

**Authority.** The repository's standing rule is that choosing a metric, threshold, pass
condition, estimator or control is a stop-and-ask decision (`docs/CODE_REALITY.md` §10).
On 2026-07-29 the owner explicitly delegated threshold-setting to me for this work, and
asked that everything be auditable for a later paper. These bars are therefore mine, not
the owner's, and are recorded as such in `docs/DECISIONS.md`.

**Why pre-register at all.** A threshold chosen after seeing the numbers is
indistinguishable, to a reader, from one chosen *because* of them. The git history of this
file is the evidence that these were fixed in advance.

---

## 0. What is being claimed

That the RNGRN **robustly and reproducibly recovers a gene regulatory network and dynamics
that produce the same Turing pattern as a target system**, for **both** the competitive
(Shea–Ackers) and non-competitive (`nc1`) regulation forms, with **biologically viable**
parameters, and that the result **generalises across spatial domain sizes**.

Note what is deliberately *not* claimed: that the recovered parameters match the
generator's. The owner is explicit — "we do not want to obtain the same D values as the
training data necessarily… the model is supposed to be an engine so we find values that are
biologically plausible." Parameter agreement is not scored anywhere below.

---

## 1. Datasets, and which claims each may support

| dataset | n | role |
|---|---|---|
| `three_gene_qvar` | 34 systems, periods-per-box ~ U{3..14} | **PRIMARY.** All headline claims. |
| `three_gene_multiL` | 23 systems × 4 domain sizes | **L-generalisation only.** |
| `three_gene_val` (legacy) | 19, periods-per-box ≡ 6.000 | **Secondary/continuity only. May not support any k\* claim.** |

**The legacy exclusion is not optional.** Every legacy generator set
`L = clip(6·2π/k*, 18, 220)`, so an image-blind predictor using `L` alone scores **0.0 %
median error** on all 127 samples — better than recovery. On `three_gene_qvar` that same
predictor costs **45.5 %**, and an *oracle* best-fixed-p chosen after seeing the answers
still costs **28.6 %**. k\* on legacy data measures the leak, not the model.

**Held-out discipline.** Any sample used while tuning is burned; final numbers come from
samples untouched during tuning. `three_gene_multiL` is grouped by `system_id`, and a
system never straddles a split.

### 1a. The `three_gene_qvar` split — declared 2026-07-29, before any convergence run

`three_gene_qvar` was registered with `splits: {}` (unit 11 deliberately left splitting
open). It therefore has no train/val/test structure, and choosing which samples are
"held out" *after* seeing results would destroy the meaning of the word. The split is
fixed here instead, by position, before the first run:

| role | samples | n |
|---|---|---|
| **TUNING (burned)** | `sample_0000` … `sample_0007` | 8 |
| **HELD OUT** | `sample_0008` … `sample_0033` | 26 |

The six held-out targets carrying the §3 claim are pre-named so they cannot be selected
after the fact: **`sample_0008`, `sample_0012`, `sample_0016`, `sample_0020`,
`sample_0024`, `sample_0028`** — every fourth sample from the start of the held-out range.
No other rule was tried; this one was chosen for being mechanical rather than favourable.

`sample_0000`–`0007` may be used freely for tuning, and any number from them is reported
as a tuning number, never as evidence for §3.

*Decided by me under the delegated authority recorded in §0, and logged in
`docs/DECISIONS.md`. This is a split definition, not a weakening of any pass condition —
the §3 thresholds are untouched.*

---

## 2. Seeds

**K = 8 independent seeds per target**, and *independent* is load-bearing. Before phase-B
unit B1, `recover()` built restart *r* as `RNGRN(seed=seed+r)`, so seeds *s* and *s+1*
shared R−1 of R restart inits — measured bit-identical results across seed pairs, meaning
8 seeds were ~4 independent draws. **Every seed count in this document refers to
post-B1 independent draws.** Pre-B1 seed counts elsewhere in the record are not comparable
and must not be quoted alongside these.

A seed that raises counts as a **failure**, never as a silent drop from the denominator.

---

## 3. The pass conditions

A target **passes** only if all **five** criteria below hold. (A sixth, §3.6, was added and
withdrawn on 2026-08-03 without any measurement being made against it; it is retained there
as future work and binds nothing.) A **form** (competitive / `nc1`)
passes only if ≥ **4 of 6** held-out targets pass. The overall claim requires **both** forms
to pass.

### 3.1 Reproducibility — the primary criterion

> `topology_consistency` ≥ **0.75** (≥ 6 of 8 independent seeds share the modal J sign
> structure), **and** the within-target minus cross-target gap ≥ **0.25**.

*Calibration.* 0.75 is the level at which the consensus topology survives dropping any
single seed — with 8 seeds, 6 agreeing means no one seed is load-bearing.

*The control, which is the substantive half.* Cross-target agreement: the same statistic
computed over seeds fitted to **different** targets. That measures how much agreement comes
from the model's inductive bias rather than from the data. If within-target ≈ cross-target,
the model is reproducing itself, not the target, and the criterion fails **regardless of the
absolute number**. This is the criterion most likely to expose a false success and it is
deliberately placed first.

*Known uncalibrated input.* `topology_consistency` collapses Jacobian entries below
`DEFAULT_SIGN_ZERO_RTOL = 0.05` (relative to each matrix's own max|J|) to "no edge", and is
defined as the modal fraction rather than mean pairwise agreement. Both were flagged
UNCALIBRATED by unit 3. Results will be reported at 0.02 / 0.05 / 0.10 so the conclusion's
sensitivity to that choice is visible rather than hidden.

### 3.2 Robustness

> Median `turing_volume_10pct` over Turing-reaching seeds ≥ **0.90**, and median
> `turing_volume_4p8pct` ≥ **0.95**.

*Calibration.* Measured over all 127 generator systems, 400 draws each — these are the
systems we are trying to be as good as:

| noise SD | mean | median | worst sample |
|---|---|---|---|
| 1 % | 0.993 | 1.000 | 0.538 |
| **4.8 %** | 0.954 | **1.000** | 0.385 |
| 10 % | 0.879 | **0.935** | 0.232 |
| 20 % | 0.746 | 0.755 | 0.220 |

0.90 at 10 % sits between the generator population's mean (0.879) and median (0.935): the
recovered system must be *within the spread of real Turing systems*, not merely nonzero.
4.8 % is Tica et al.'s **measured experimental parameter CV**, which is what makes this
number comparable to theirs.

### 3.3 Same pattern

> `morphology_match` true, **and** `morphology_distance` ≤ the 75th percentile of
> within-class distances in `three_gene_train`, **and** `kstar_fft_rel_err` ≤ **8.3 %**,
> **and** `trivial_kstar_err` reported in the same record.

*Calibration.* 8.3 % is half an FFT bin. One bin is 16.7 % of k\*, and stored `k_star_fft`
sits within 0.011 bins of a half-integer for all 128 measured samples, so ±8.33 % is the
estimator's own resolution floor. Demanding better than the estimator can resolve would be
meaningless. (Note this makes the configured `loss.tau = 0.12` *below* the resolution floor
— recorded as a live [TUNE] issue, not silently relied on.)

*Why the distance term.* A class label alone is too weak: the morphology classifier scores
92.1 % pooled / 81.8 % balanced leave-one-out, but only **33.3 % on held-out stripes**
(n=3). Requiring the distance as well as the label prevents a lucky label from carrying a
pass. Stripes calls are reported as hypotheses, per the scorer's own documented limitation.

*Never quoted alone.* A k\* number without `trivial_kstar_err` beside it is not reportable
under this pre-registration.

### 3.4 Biological viability

> `plausibility_score` = **1.0** — every *cited* box row satisfied — on the seeds that pass
> 3.1–3.3.

*Calibration.* The box lives in `configs/bio_box.yaml` with a citation per row. `beta` is
marked UNCITED and is therefore **not scored**; scoring an uncited row would manufacture
authority the source does not have. The D-ratio prior is centred at **7.5** (measured
Nodal/Lefty), which is the owner's explicit decision and *reverses* an earlier choice of
~100.

*Expected tension, recorded in advance so it cannot be presented as a surprise.* The
generators sample D-ratio as 10^U(0.9,2.4) ≈ **7.9–251**, population median **134.9**. A
prior centred at 7.5 sits below the low tail and pulls against most of the population. A
lower raw hit rate is the **expected** consequence, not a defect. If viability and recovery
rate trade off, that trade-off is a result and gets reported as one; the centre does not
move to make the bar easier.

### 3.5 Generalisation across domain size

**Amended 2026-07-30, before any convergence result existed.** The owner clarified the
intent: *"The post training evaluation should run on a different domain size. That is part
of the idea. Shows that the learned grn generalises across domain sizes."* The original
3.5 (below, retained as 3.5b) compared **separately recovered** models at different L. That
is a weaker test, and in one specific way a defective one: it cannot distinguish a network
that generalises from a network that merely refits whatever periodicity each box wants,
because each L gets its own fit. The primary test is therefore now a **transfer** test —
one recovery, evaluated on domains it never saw.

Amending this *adds* a criterion and tightens the claim; it does not weaken a threshold, and
no convergence numbers existed when it was written. The original wording is kept verbatim
so the change is visible rather than silent.

#### 3.5a PRIMARY — cross-L transfer of a single recovered GRN

Recover on one domain size `L_train`, then simulate the recovered model at
`L ∈ {0.5, 1, 2, 4} × L_train` without refitting anything:

> `kstar_phys_cv` ≤ **0.10** across those L, **and** the fitted slope of
> `periods_per_box` against `L` within **15 %** of `kstar_phys/2π`, **and**
> `morphology_class_preserved` ≥ **0.75**.

*Why these are the right quantities.* Turing theory fixes k\* from the reaction and
diffusion parameters — it is a property of the network, not of the box. So the physical
wavelength must be **invariant** in L while the number of periods across the box scales
**linearly** with L. The failure mode this catches is the interesting one: a network that
memorised a box produces a constant `periods_per_box` with k\* drifting as 1/L, which
inverts both signatures at once. Reporting the CV and the slope together makes the two
outcomes unambiguous, which the original 3.5 could not do.

*Calibration.* 0.10 on the CV is one FFT bin's worth of scatter at the resolutions in use
(one bin is 16.7 % of k\*, so ±8.3 % is the estimator's own floor — a CV below that would be
claiming precision the measurement does not have). 15 % on the slope is the same floor
carried through the fit. 0.75 on class preservation is 3 of 4 domain sizes, and is
deliberately not 1.0 because the morphology classifier scores only 33.3 % on held-out
stripes (n=3) and would otherwise fail this criterion on classifier noise rather than on
physics.

*Ground truth is available and must be used.* `three_gene_multiL` contains the **same 23
systems each simulated at 4 real domain sizes**, so a transferred prediction at L₂ can be
compared against the actual generator sample at L₂ — not merely checked for
self-consistency.

*Resolution is a real confound, not a detail.* Changing L at fixed grid n changes dx, so
large L means fewer pixels per wavelength and k\* estimation degrades until it is
meaningless. `pixels_per_wavelength` is reported for every L and evaluation refuses below a
stated floor (~6 px/wavelength, where the generator itself already sat at its coarsest).
A criterion met only by unresolved fields is not met.

*What is trivial here and what is not — stated plainly, because the obvious objection is
correct.* The dispersion relation σ(k) = max Re eig(J − k²D) is a function of **J and D
alone**. The *linear* k\* is therefore L-independent by construction, for any model
whatsoever, on either the dimensional or the non-dimensional path. A reviewer asking "isn't
k\*-invariance guaranteed?" is right about the linear prediction, and this criterion would
be vacuous if that were all it measured.

It is not. Three things make the transfer test substantive, and all three are nonlinear or
discrete:

1. **Mode quantisation.** On a periodic box of size L only wavenumbers k = 2πm/L exist, so
   the realised pattern must snap to the nearest admissible mode. At small L (few periods)
   that lattice is coarse and the realised k\* can depart materially from the linear
   prediction. Invariance across L is a claim that selection is robust to the lattice, not
   a restatement of the eigenvalue problem.
2. **Nonlinear selection and saturation.** k\* is the fastest-growing *linear* mode; the
   saturated pattern's dominant mode need not equal it, and which mode wins can depend on
   the domain through the competition between admissible modes. `morphology_class_preserved`
   and `amplitude` are properties of the saturated state and are not fixed by (J, D) at all.
3. **Reachability.** A recovered network that is only marginally Turing-unstable may pattern
   at one L and fail to pattern at another (`patterned` False), which no linear argument
   rules out.

And one further correction, made here rather than left for a referee: `periods_per_box` is
*defined* as `L·k*/2π`, so the slope criterion and the CV criterion are **two views of one
measurement**, not independent evidence. If k\* is invariant then q ∝ L follows
algebraically. They are both retained because together they make the failure direction
legible at a glance — constant q with k\* ∝ 1/L is a visibly different table from constant
k\* with q ∝ L — but they must be counted as **one** criterion, and the earlier claim that
the slope independently "falsifies box-memorisation" was wrong: in a transfer test nothing
is refitted, so that failure mode cannot arise the way it would in 3.5b.

**Therefore the load-bearing number in 3.5a is the comparison against ground truth**: does
the model recovered at L_train reproduce the k\* and morphology of the *real*
`three_gene_multiL` sample at L₂, L₃, L₄? That is a test against an independently generated
system, not a self-consistency check, and it is the only part of 3.5a that cannot be
satisfied by structure alone. Class preservation and patterning-at-every-L rank next, since
both are properties of the saturated state that (J, D) does not fix. `kstar_phys_cv` and the
slope rank last and are reported for completeness.

Recorded this way deliberately: a criterion whose triviality is discovered by a reviewer
costs more than one whose limits were stated by its author.

#### 3.5b SECONDARY — agreement between independent recoveries at different L

Retained from the original pre-registration, now reported as a secondary consistency check
rather than the headline: on `three_gene_multiL`, for one system across its 4 domain sizes,
recovered J sign-structure agreement ≥ **0.75**, and physical D-ratio spread (CV) ≤ **0.25**.

*The control.* Cross-*system* agreement at fixed L. As in 3.1, the gap between within-system
and cross-system is the substantive number; a high absolute value with no gap means the
model returns the same network regardless of input.

*Reported on both paths.* Dimensional (`nondim=False`) and non-dimensional
(`nondim=True`), because the non-dimensional path is the one that is L-invariant *by
construction* and therefore cannot be allowed to claim credit for it. Endres et al. never
separated L from the parameter vector — their `x̂ = x·√(c₂/D_u)` folds length and diffusion
into one variable, and their code hard-codes L via `dx` — so there is no prior result to
inherit here.

### 3.6 Robustness under finite timescale separation — added AND WITHDRAWN 2026-08-03

> **WITHDRAWN THE SAME DAY IT WAS ADDED, AT THE OWNER'S DIRECTION, BEFORE ANY MEASUREMENT
> EXISTED AGAINST IT. This is NOT a pass condition. It is future work.**
>
> The owner first raised finite-`mu` robustness to the essential criterion, then scoped it
> out: *"For now, let's stick to finite mu. Separation of time scales is a whole nother
> problem. My previous work leads me to believe that training under the QSS is sufficient to
> discover circuits that pattern under finite mu."* and *"Just note work on mu for future
> work if we have time."*
>
> **Nothing was measured against this section before it was withdrawn** — the unit building
> the machinery was stopped mid-run and its code is parked, unvalidated, on branch
> `feature/rngrn-c-mu` (commit `17e9ad2`). No run, no number, and no decision anywhere in
> this repository was judged by §3.6.
>
> **Why the text is retained rather than deleted.** A pre-registration whose withdrawn
> sections vanish is not a pre-registration. Keeping it visible means a reader can see
> exactly what was considered, when, and on whose authority it was set aside — which is the
> whole function of this document. It is retained *verbatim as originally written* below.
>
> **Scope note, so the record is not misread.** §3.2 continues to bind and is unaffected. It
> is measured on the QSS Jacobian, and this section is the standing statement of what that
> does *not* cover: the lifted finite-`mu` system's robustness is **unmeasured in this
> project**. Any claim of robustness made under §3.2 is a claim about the QSS reduction, and
> should be worded that way in the paper.

---

*Original text, as written before withdrawal:*

**Added at the owner's explicit direction, before any finite-`mu` number existed.** The
owner's words: *"what I really care about is the robustness of the circuit when finite mu is
introduced. That is the most essential thing that needs to hold."*

Amending this way **adds** a criterion and **tightens** the claim; it weakens no threshold
and moves no bound. §3.2 is retained verbatim above. This section is dated so that the git
history shows it was fixed before the runs it judges, exactly as §3.5a was.

#### What this measures, and why §3.2 alone does not

The RNGRN trains on a **quasi-steady-state** reaction (`model.py::_reaction_raw`): promoter
occupancy is assumed to equilibrate *instantaneously* with transcription-factor
concentration, so the gates never appear as state variables. Real gene regulation does not
do that — TF–promoter binding is fast but finite, and `mu` parameterises the separation
between that fast binding and the slow protein production/degradation the model resolves.

`eval/dynamical.py::lift_check` makes the gates explicit fast variables and has existed
since the scaffold, but it is called from exactly one place — `tests/test_science.py:71`, a
2-species model at `mu = 1e-4` — and it only checks that the *production algebra* converges
to the QSS expression as `mu → 0`. **The lift has never been simulated in space, never run
at finite `mu`, never coupled to diffusion, and never asked whether a pattern forms or
survives perturbation.** Every `turing_volume_*` number in this project is therefore
computed on the **QSS** Jacobian, and says nothing about the system the biology actually
runs.

The lifted state is `(x, G_A, G_R)` of dimension `N + 2N²` (21 at N=3). Only `x` diffuses —
promoter states are bound to DNA — so the diffusion matrix is `diag(D_x, 0, …, 0)`. The
lifted system's **fixed points are identical to the QSS fixed points at every `mu`**, since
setting `dG/dt = 0` recovers the QSS algebraic relations exactly. So `mu` changes stability
and dynamics, never the steady state, and the QSS and finite-`mu` measurements are
comparable at the same `x*` by construction.

#### The criterion

> §3.2's bars, recomputed on the **lifted finite-`mu` Jacobian** rather than the QSS one:
> median `turing_volume_10pct` ≥ **0.90** and median `turing_volume_4p8pct` ≥ **0.95**,
> holding at **every `mu` across the cited biological band**, not at a point estimate.
>
> A draw counts as Turing only if the leading unstable mode is **stationary**. The
> oscillatory fraction is reported separately and never folded in.

*Calibration — inherited, not invented.* The two numbers are §3.2's, which are calibrated
against all 127 generator systems at 400 draws each (population mean 0.879 / median 0.935 at
10 %; 4.8 % is Tica et al.'s measured experimental parameter CV). No new threshold is
introduced here: the bar is the same height, moved onto the harder system. That is
deliberate — inventing a softer number for the finite-`mu` case would be the exact move §5
forbids.

*Why "across the band" and not "at `mu_bio`".* `mu_bio` is a ratio of literature timescales
(TF–DNA residence seconds against protein turnover tens of minutes to hours) and will not be
pinned better than an order of magnitude. A criterion evaluated at a point estimate of a
quantity known only to a decade is a criterion evaluated at an arbitrary point. Requiring it
to hold across the whole defensible band is what makes the claim mean something, and the
band's width is set by the literature rather than chosen here. **The band itself is an input
to this criterion and is marked UNCALIBRATED until Stage 0b's citations land**; the band is
recorded in `docs/DECISIONS.md` with its sources before any pass/fail is read against it.

*The shape matters as much as the value.* `turing_volume` against `mu` is reported as a
**curve**, per form, with the 0.90 and 0.95 lines and the biological band drawn on it. A
circuit that clears the bar at `mu_bio` but cliff-edges just above it is not robust in the
sense being claimed, and a point value cannot show that. Graceful degradation and a cliff
are different results and are reported as such.

*Two failure routes, distinguished.* Explicit slow gates classically introduce delay-driven
oscillation, so the uniform state can go unstable via a Hopf bifurcation *before* the `k > 0`
band closes. Which of the two eats the volume is reported, because they carry different
biological meanings — a Hopf route says the circuit oscillates rather than patterning; a
closing band says it relaxes to uniformity.

*`mu` is itself uncertain.* Perturbing `(alpha, delta, D, K)` at fixed `mu` understates the
real exposure. Both are reported: the standard parameter cloud at each fixed `mu`, and a
cloud in which `mu` is drawn from its own band alongside the parameters. **The second is the
one this criterion is read against.**

*Strict test only.* `max Re eig(J) < 0` for uniform stability, never `tr(J) < 0`. Stage 0
measured the trace test overcounting by 64× on 80,000 box-constrained draws, with all 1,196
extra acceptances being uniform instabilities rather than Turing patterns.

*Integrator honesty, stated in advance because it is the dangerous failure mode.* The gates
relax at rate `1/mu`, so a stiff integrator that numerically damps the instability would
manufacture a "pattern died at finite `mu`" result as an **artefact**. Any finite-`mu`
simulation supporting this criterion must show `dt`-convergence and must reproduce the QSS
field at `mu ≤ 1e-4` before its finite-`mu` runs are quoted.

*Applies to both forms and to the recovered networks, not only synthetic draws.* The
circuits carrying the claim — currently `three_gene_qvar` `sample_0003` and `sample_0004`,
prior-ON, the two that are simultaneously `plausibility_score = 1.0` and patterned — are
evaluated at full depth. If this criterion fails, it fails; §5 applies unchanged and the
shortfall is reported against the bar as written.

---

## 4. What is reported regardless of outcome

- The **rate**: how many of K seeds produced a scored recovery, and how many reached the
  Turing regime. The starting point for comparison is the measured library benchmark —
  40 seeds × 400 steps giving 38/40 converged, 14 Turing, `turing_frac 0.3684` — and, on
  legacy held-out data, Turing seeds on only **2 of 6** samples.
- Every criterion's value, including the ones that fail.
- Both forms, including `nc1` if it remains training-unstable. At the time of writing,
  `nc1` is numerically correct at init (Newton 100/100 seeds; autodiff J vs FD 2.06e-8;
  cubic dispersion vs eigvals 3.55e-14) but diverges partway through Adam training
  (observed steps 26–49, 4/4 restarts) where competitive with identical seeds completes.
  If that is not fixed, "`nc1` cannot be trained with this objective because X" is a
  legitimate reported outcome.
- Negative and superseded results stay in the record with the reason they were superseded.

## 5. Stopping rule

Convergence work stops when either every criterion in §3 is met for both forms, or the
compute budget is exhausted, whichever comes first. **In the second case the bars stay as
written and the shortfall is reported against them.** Tuning continues only against the
metrics defined here; a new metric invented mid-run to rescue a claim would void this
pre-registration, and if one turns out to be genuinely needed it goes in a *new* dated
section that says plainly it was added after seeing results.

---

## 6. Threats to validity, acknowledged in advance

- **The legacy 127 samples cannot be regenerated from their recorded seed.** The generator
  seeded from `abs(hash(topo))`, which Python salts per process. Fixed for the new
  datasets; permanent for the legacy ones.
- **Hidden-channel (m<N) recovery has no known-good objective.** The latent fields enter
  only through the stationarity residual, whose gradient is exactly 0.0 at weight 0 — and
  exp06 measured that residual as harmful (9/9 swept cells collapsed). Experiment A is
  therefore out of scope for these claims, and `recover()` refuses such runs rather than
  returning a dressed-up random init.
- **Two evidence-integrity defects were active during earlier work**: `cli.py -o` silently
  kept only the last override, so some runs did not use the budget their command line
  asked for; and the pre-push hook printed "skipping tests" and exited 0. Both fixed at
  the phase-A merge. Numbers from before that merge are read with this in mind.
- **The objective changed at the phase-A merge.** k\*/loss numbers produced before it came
  from a different objective and are not comparable to numbers produced after it.
- **`morphology_scored` was `"target_only"` on every run before phase A**, so no
  model-side morphology comparison exists in the historical record at all.
