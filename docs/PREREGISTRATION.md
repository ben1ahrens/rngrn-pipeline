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

**Held-out discipline.** Targets are drawn from the val/test portions. Any sample used
while tuning is burned and is reported as such; final numbers come from samples untouched
during tuning. `three_gene_multiL` is grouped by `system_id`, and a system never straddles
a split.

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

A target **passes** only if all five criteria below hold. A **form** (competitive / `nc1`)
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

> On `three_gene_multiL`: for one system across its 4 domain sizes, recovered J
> sign-structure agreement ≥ **0.75**, and physical D-ratio spread (CV) ≤ **0.25**.

*The control.* Cross-*system* agreement at fixed L. As in 3.1, the gap between within-system
and cross-system is the substantive number; a high absolute value with no gap means the
model returns the same network regardless of input.

*Reported on both paths.* Dimensional (`nondim=False`) and non-dimensional
(`nondim=True`), because the non-dimensional path is the one that is L-invariant *by
construction* and therefore cannot be allowed to claim credit for it. Endres et al. never
separated L from the parameter vector — their `x̂ = x·√(c₂/D_u)` folds length and diffusion
into one variable, and their code hard-codes L via `dx` — so there is no prior result to
inherit here.

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
