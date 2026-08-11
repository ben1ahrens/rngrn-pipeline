# The target: a Tica-equivalent circuit, more robust

**Status: information, not instruction.** This document describes what the final goal
*is* and what would count as reaching it. It does not prescribe how. Every number
attributed to this repo was measured; every number attributed to the literature is
cited. Where a decision is still open, it says so rather than picking.

---

## 1. What Tica et al. actually built

Tica J, Oliver Huidobro M, Zhu T, Wachter GKA, *et al.* (12 authors) — *A three-node
Turing gene circuit forms periodic spatial patterns in bacteria*.

- **Version of record:** Cell Systems **15**(12):1123–1132.e3, 18 Dec 2024,
  [10.1016/j.cels.2024.11.002](https://doi.org/10.1016/j.cels.2024.11.002)
  (PMID 39626670). Citation fields verified against the PubMed record; the journal
  article itself was **not** retrieved for this document.
- **Source actually read here:** the bioRxiv preprint,
  [10.1101/2023.10.19.563112](https://doi.org/10.1101/2023.10.19.563112), posted
  19 Oct 2023, whose own footer states it was not certified by peer review.

**Every quantity quoted below is from the preprint.** Cite the version of record, but
re-check any number against it before publishing — figure numbering, supplementary
section numbers, and fitted values may have changed between the preprint and the final
article.

### 1.1 The circuit

Three nodes, chosen not invented. The topology is the three-node network that formed
Turing patterns for the largest number of parameter combinations in the Scholes et al.
network-atlas screen — the paper names networks **#3954** and **#1754** as the basis,
and Fig. 2b works with #1754.

Structurally:

- **A ↔ B** is a classical self-activation + lateral-inhibition motif, except that the
  positive feedback on A is implemented as *repressible constitutive activity* rather
  than direct autoactivation.
- **C is a third node that does not diffuse.** Its stated purposes are (i) to enlarge
  the set of parameters that pattern and (ii) **to relax the differential-diffusion
  requirement**.
- A and B are carried by orthogonal quorum-sensing diffusers pC-HSL and 3OHC14-HSL;
  C is an orthogonal cIλ variant with dual activation–repression behaviour. LacI, TetR
  and cI implement the repressions; a DAPG-inducible AiiA lactonase gives tuneable
  diffuser degradation. GFP and mCherry report nodes B and C.
- Six genes in the built circuit; the PDE model has **six equations** (two diffusible
  HSLs plus four intracellular TFs), non-dimensionalised, each equation carrying a
  basal leak term, a regulated Hill term, linear degradation and diffusion. Receptors
  are held at constant concentration; QSS is assumed for mRNA and for diffuser
  production.

### 1.2 The pattern

Stationary, periodic, **concentric stripes** in a growing *E. coli* colony —
out-of-phase green/red rings added sequentially at the growing edge. Three stripes
where outgrowth was strongest, two where growth was slower, one where slowest; the
**number** of stripes tracked field size while the **wavelength stayed constant**,
which is the Turing signature the paper leans on.

### 1.3 The numbers that matter to us

| quantity | value | why it matters here |
|---|---|---|
| Turing fraction of sampled parameter space | **0.022 %** (linear stability + Latin hypercube) | This is the same order as the <0.1 % Scholes figure. It is why random search cannot find the regime and an optimiser must. |
| Hopf | 4 % of samples | A competing instability class our scoring currently cannot name. |
| Turing–Hopf | 0.004 % | Also produces regular stationary patterns — i.e. "stationary and periodic" does not uniquely imply Turing I. |
| local enrichment near a Turing solution | 1 % relative uncertainty → **33 %** Turing; 5 % → **5 %** Turing | The paper's own robustness metric. It is a *local* parameter-volume fraction, exactly the quantity `eval/analysis.robustness_cloud` was built to estimate. |
| measured experimental parameter CV | **4.8 %** (Vm and Km across biological repeats) | The natural noise scale to evaluate robustness at. Our figure marks it. |
| transfer-function matching | ~**30-fold** increase in Turing probability | Robustness was engineered by matching component sensitivities, not by changing topology. |
| ATC | high ATC dramatically raises Turing probability; DAPG and IPTG do not | Robustness has an *operating point*, not just a parameter set. |

Note the shape of their claim: *0.022 % globally, but 33 % within 1 % of a found
solution.* Their argument for feasibility is that global rarity and local density
coexist. Any robustness claim we make should be stated on the same footing — as a
local parameter-volume fraction at a named noise level, not as a global hit rate.

### 1.4 What their model is *not*

Three differences shape what "equivalent" can mean:

1. **Growth.** Their solver couples the PDE to a stochastic cellular automaton for the
   colony, computing reaction+diffusion inside cells and diffusion-only outside; ADI
   time stepping; reflecting or absorbing outer boundaries depending on dish size.
   Our generators are **static 96×96 periodic domains** integrated to an attractor.
   Concentric rings are a growth phenomenon; on a periodic box the equivalent object
   is a stationary periodic pattern with the same wavelength, not rings.
2. **Species count.** Six equations, of which two diffuse. Our RNGRN is N species with
   one diffusivity each and no distinction between "TF" and "diffuser".
3. **Fitting target.** They fit to *dose–response curves from liquid culture*
   (subcircuits #1 and #3) and then check whether the fitted multivariate Gaussian
   contains Turing solutions. We fit to **one converged image**. Different inverse
   problem: they have per-component transfer functions, we have a pattern.

---

## 2. The goal has three components, not one

An earlier draft of this document treated "more robust" as the whole goal. It is not.
The end goal has **three** components, and they are separate targets with separate
evidence requirements:

1. **Robustness** — a larger local Turing parameter volume than the reference. §2.1
   below and `ROBUSTNESS_MEASUREMENT.md`.
2. **Biologically realistic parameter values** — the recovered parameters must be
   plausible, as a *constraint* rather than a fitting target. §2.2.
3. **Inferring a 3-gene circuit from data generated by an N=2 system** — i.e. the
   over-parameterisation direction. §2.3.

Component 3 is the one most easily lost, because it inverts the usual framing: the
question is not "can we recover the hidden third gene" (that is Experiment A, N=3
truth) but "**given a pattern that a two-gene system made, what does a three-node model
do with it?**" Those are opposite experiments and they need opposite success criteria.

### 2.1 Robustness — four incompatible readings

Four readings are possible. They are not equivalent and they have different evidence
requirements. The project has not chosen between them; this section lays them out so
the choice is explicit when it happens.

**Reading A — same topology class, better parameter volume.**
The RNGRN recovers a 3-node network whose *sign structure* matches the
#1754/#3954 family (self-activation on A via repressible constitutive activity,
lateral inhibition A↔B, a non-diffusible modulatory C), and whose recovered parameter
point has a **larger local Turing volume** than the fitted point Tica reports (33 % at
1 % uncertainty; 5 % at 5 %). Evidence needed: sign-structure comparison + a
robustness cloud at matched noise levels.

**Reading B — same observable pattern, better parameter volume.**
Ignore topology. The RNGRN produces a system whose dominant spatial mode and
morphology match the target pattern, and whose local Turing volume beats the
reference. This is the reading most consistent with the project's stated success
criterion (§3) and with the "engine for biologically plausible values" framing.

**Reading C — same circuit, relaxed differential diffusion.**
Tica's stated reason for node C was to *relax* the differential-diffusion
requirement. A robustness win of this kind would be: patterns at a **lower D-ratio**
than the classical two-node requirement, at equal or better local volume. This is
measurable and is the one reading where the "more robust" axis is not parameter noise
at all. Section 4.3 below reports what our own data says about this — and it is not
what one would assume.

**Reading D — robust to the thing that actually broke the experiment.**
Their pattern was destroyed by: added DAPG (raising diffuser degradation), 30 °C
instead of 37 °C (rate constants shift), a permeable outer boundary, and thicker agar
(diffuser dilution into the substrate). These are *structured*, not isotropic,
perturbations. A circuit robust to isotropic log-normal noise is not necessarily
robust to a coordinated shift in all degradation rates. If robustness is to mean
anything experimentally, the perturbation model has to be argued, not defaulted.

**The open decision:** which of A–D is the target, and at what noise level the claim
is stated. Everything downstream — what to score, what counts as a control, whether
sign structure matters — follows from it. Under the project's autonomy rule this is a
science decision, not a mechanical one.

### 2.2 Biologically realistic parameter values

The second component. Parameter agreement with the generator is **not** a target
(§3), but plausibility **is** a constraint — the model is meant to be an engine that
returns biologically credible parameter sets, of which there may be several.

**What the constraint can be checked against.** The classical families store their
generating parameters in a `params_json` attr, so the generator's own sampling box is
recoverable. Measured across all samples that carry it:

| family | n | parameter ranges (min / median / max) |
|---|---|---|
| `two_gene_classical` | 75 (39 Gierer–Meinhardt, 36 Schnakenberg) | `a` 0.011/0.085/0.467 · `b` 0.509/1.298/2.766 · GM `c` 0.756/2.261/9.912, `K` 0.001/0.013/0.293, `nh` 2–3 · Schnak `g` 2.14/8.30/55.9 |
| `three_gene_classical` | 85 (4 sub-families: `gm_relay` 31, `schnak_relay` 19, `schnak_cross` 18, `gm_two_inhibitor` 17) | `a` 0.010/0.076/0.489 · `b` 0.503/1.018/2.352 · `c` 0.717/3.736/9.787 · `e` 0.292/1.662/3.140 · `K` 0.001/0.014/0.331 · `lam` 0.105/0.481/1.803 · `g` 2.06/10.86/42.83 · `h` 0.104/0.408/1.488 · `d` 0.032/0.178/0.493 · `f` 0.125/0.664/1.822 · `kappa` 0.102/0.326/1.501 · `nh` 2–3 |

**Important caveat: the `three_gene` family — the 127 samples all the recovery work uses
— does NOT store `params_json`.** It stores `interaction_matrix`, `topology`, `reaction`
and `n` instead. So for the family that matters most, the generating kinetic parameters
are **not recoverable from the payload**; only `J`, `x*`, `D` and the topology are. Any
plausibility check on that family has to be made against *stated priors* or literature,
not against the generator's own draws.

The generator priors recorded earlier in the project (α 0.5–10, δ 0.4–5, D-ratio ~8–250)
are the working plausibility box for the RNGRN's own parameters. Note the D-ratio tension
in `STATE_OF_THE_SCIENCE.md` §11: the measured biological ratio (Nodal/Lefty ~7.5) sits
an order of magnitude below the generators' 108–140, and Tica's own argument is that a
third node relaxes the requirement.

**What is not yet defined:** there is no implemented plausibility *score*. Nothing in
`validate.score_recovery` checks whether recovered parameters lie in a plausible box, and
no such column reaches the run index. "Biologically realistic" is currently a stated
intent, not a measured quantity — which makes it the least instrumented of the three goal
components after robustness.

### 2.3 Inferring a 3-gene circuit from N=2 data

The third component, and the one whose success criterion is most counter-intuitive: the
target pattern was generated by a **two**-species system, and the model is given **N=3**.

This is already built as **Experiment B** — `configs/expB_overparam.yaml` (N=3 model on
`two_gene_classical_val`, `observed_idx: [0,1]`, species 2 is spare capacity) with the
matched control `expB_control_matched.yaml` (correctly-sized N=2 model, same data).
`scoring/overparam.py` implements the metrics. The 2×96×96 data is already in registry
format, so no new generation is required.

**Success is NOT a "correct" 3×3 Jacobian.** It is two things jointly:

1. the **spare species stays inert** — no strong edge to or from species 2; and
2. the **observed 2×2 sub-block matches the truth**.

A confident third gene is a **false positive** — that is the failure mode this experiment
exists to detect. This inverts the usual reading of "infer a 3-gene circuit": the win is
that the extra capacity is *not* used, not that it is filled in.

Contrast with **Experiment A** (`expA_hidden_channel.yaml`): N=3 truth, one gene
genuinely hidden, and there success *is* recovering the 3×3 up to relabelling. A and B
feed the model the same shape of input (2 observed channels, 3×3 output) and mean
opposite things by success. See `docs/IDENTIFIABILITY_EXPERIMENTS.md` and
`STATE_OF_THE_SCIENCE.md` §9.

**What stands in the way, specifically for B:**

- The inertness threshold (`DEFAULT_COUPLING_THRESHOLD = 0.05`, relative) is explicitly
  **uncalibrated**. It must be calibrated from the distribution of `max_abs_coupling_*`
  in the `expB_control_matched` arm, where no spare species exists by construction.
- `sign_match_frac` is deliberately NaN in the over-parameterised arm;
  `subblock_sign_match_mean` is the only column comparable across every arm.
- Both configs pin `sample_key: sample_0000`, so as written each is a single-sample run.
- **No expB result has been produced yet.** The harness runs; the experiment has not been
  executed on the current objective.

**And a tension with the primary criterion.** §3's criterion is the dominant spatial mode
and morphology. Experiment B's criterion is a statement about the *recovered network's
structure* (inertness + sub-block). B therefore cannot be graded by the primary criterion
alone — it needs the structural metrics, which means the "parameter values are not a
target" rule has a carve-out here: for B, the recovered coupling *magnitudes* are the
measurement.

---

## 3. The project's own success criterion, and the tension with the goal

Recorded verbatim from the user:

> "the only thing I really care about is that the model recovers the same dominant
> spatial mode, pattern morphology."

and:

> "we do not want to obtain the same D values as the training data necessarily. Same
> goes for the production and degradation rates. The model is supposed to be an engine
> so we find values that are biologically plausible."

So parameter agreement is **explicitly not** a target; biological plausibility is a
**constraint** on recovered parameters. This overrides older docs in this repo that
emphasise J sign structure or parameter recovery as goals.

**The tension to be aware of:** Reading A above requires sign-structure agreement,
which the stated criterion demotes. Reading B is consistent with the criterion but
makes "the equivalent gene circuit" a claim about behaviour rather than wiring — and
"equivalent circuit" in the goal statement reads more like A. Worth resolving
explicitly rather than by drift.

---

## 4. Where the project stands relative to this goal

### 4.1 The forward direction works; the inverse direction generalises poorly

The current best recovery recipe (data-first staging, split-support Turing hinges,
frame-scale anchor; 4000 Adam steps) reaches a Turing-unstable recovered system on
**2 of 6** held-out `three_gene_val` samples, with 7/8 seeds succeeding on each of
those two and 0/8 on the other four (exp09). On the two that work, recovered k* lands
~3 % from the linear-theory k*, beating its own FFT anchor (6.6 %). See
`STATE_OF_THE_SCIENCE.md` §2 for the full ledger and the diagnosed root cause.

That is the honest gap: **the pipeline cannot yet reliably recover any single target
pattern**, let alone recover one and then demonstrate it is more robust than a
published circuit.

### 4.2 Two of the three goal components are uninstrumented

**Robustness.** The quantity Tica reports — local Turing parameter volume — is the
natural axis, and this repo has a function for it (`eval/analysis.robustness_cloud`). It
has never been run on a recovery result, never validated, and does not reach the run
index. It also has four measured defects that make its current output non-comparable to
Tica's number. See `ROBUSTNESS_MEASUREMENT.md` §3.

**Biological plausibility.** No score, no column, no check (§2.2).

**N=2 → N=3 inference.** Harness complete, never run (§2.3).

Of the three components only the first has even a baseline; see §4.3.

### 4.3 The training data's own robustness, measured

Measured for this handoff on all 127 `three_gene` samples (see
`ROBUSTNESS_MEASUREMENT.md` for method and `robustness_baseline.csv` for the rows).
Perturbation: independent log-normal multiplicative noise on every nonzero Jacobian
entry and on every diffusivity; "Turing" = strict (max Re eig J < 0) plus an unstable
mode at k > 0.

| noise SD | median fraction still Turing | worst sample |
|---|---|---|
| 1 % | 1.000 | 0.538 |
| **4.8 %** (Tica's measured CV) | **1.000** | 0.385 |
| 10 % | 0.935 | 0.232 |
| 20 % | 0.755 | 0.220 |

Two consequences:

1. **The generator systems are already robust at the experimentally relevant noise
   scale** — median 100 % survival at 4.8 %, versus the ~5 % Tica reports at 5 %
   uncertainty around their fitted point. The two are not measured the same way (ours
   perturbs J and D of an already-Turing system; theirs perturbs the full kinetic
   parameter vector of a six-equation model and re-derives everything), so this is
   **not** a like-for-like win. But it means "beat Tica on robustness" cannot be
   demonstrated by comparing to these numbers directly — the perturbation model must
   be matched first, and probably means perturbing RNGRN θ, not J.
2. **Robustness varies 1.45× to 4.30× between samples inside a single topology**
   (`substrate_depl`: 0.232 to 1.000 at 10 %). Topology alone does not determine
   robustness. That is good news for the "engine" framing: there is real headroom for
   an optimiser to find a *more* robust parameterisation of the same wiring.

### 4.4 Tica's immobile third node, tested on our data

Their node C does not diffuse. Setting one diffusivity to exactly zero in each of the
127 systems and re-testing:

| node made non-diffusible | still strictly Turing | median robustness at 10 % noise |
|---|---|---|
| the **slowest** diffuser | **127 / 127** | 1.000 (up from 0.960) |
| the middle diffuser | 81 / 127 | 0.968 |
| the **fastest** diffuser | 38 / 127 | 0.884 |

In every one of the 127 samples the slowest diffuser is species index 0, and the
generators fix `D[0] = 1` with the others 10–250× larger.

The finding: **immobilising a node is only free when it is the slow node** — and when
it is, it *improves* local robustness (1.000 vs 0.960), consistent with Tica's stated
rationale for C. Immobilising a fast node destroys patterning in 70 % of systems.
So "add a non-diffusible third node" is not a topology trick that works generically;
it works because the immobile node is the *activator-side* slow species. Anything
built to mimic C should place it on the slow side.

---

## 5. Gap list — what stands between here and the goal

Ordered by what blocks what, not by effort. None of these are instructions; they are
the gaps as they currently read.

1. **Recovery does not generalise** (2/6 samples). Root cause diagnosed: the k*-anchor
   has a degenerate minimum reachable by flattening σ(k) instead of relocating its
   peak. Two fixes tried and rejected. This blocks everything else, because a
   robustness claim needs a recovered system to make it about.
2. **The D-ratio prior is designed but unwired.** Failing runs collapse the D-ratio
   toward ~8 while the generators sit at 108–140. The literature-vs-generator tension
   (measured Nodal/Lefty ratio ~7.5 in zebrafish vs our generators' ~100) is
   unresolved, and Tica's own point is that a third node *relaxes* the requirement.
   Note §4.4: our data says the third node helps only in the immobile-slow-node
   configuration.
3. **Robustness is uninstrumented.** No robustness number reaches the run index;
   `robustness_cloud` perturbs raw θ (not physical parameters), can flip signs, ignores
   `dispersion_backend`, and costs ~59 ms/draw serially.
4. **The dataset's L defect.** `L = clip(6·2π/k*, 18, 220)` makes k*_true ≡ 6·2π/L for
   94.8 % of samples, so the domain size carries the answer. Only regeneration fixes
   it. Any k* result on the current data is confounded by this.
5. **No Turing-vs-Hopf-vs-Turing-Hopf classification.** `turing_ok` tests `tr J < 0`,
   which is weaker than uniform stability, and never inspects whether the leading mode
   at k* is complex. On the current data this happens not to matter (127/127 agree
   under both criteria, 0/127 have a complex leading mode at k*), but under
   perturbation the loose criterion overcounts by up to 70 % of draws for a single
   sample. Tica's 4 % Hopf / 0.004 % Turing–Hopf figures are exactly the classes this
   cannot name.
6. **No growth, no rings.** The target pattern is a growing-colony phenomenon. On a
   static periodic box the achievable claim is "same wavelength and morphology", not
   "same rings". Whether the goal requires a growth-coupled generator is open.
7. **No experimental-perturbation model.** Reading D above. DAPG, temperature and agar
   thickness are structured perturbations; isotropic log-normal noise is a proxy whose
   adequacy has not been argued.
8. **Biological plausibility is uninstrumented** (goal component 2, §2.2). No plausibility
   score exists, nothing checks recovered parameters against a box, and the `three_gene`
   family — the one all recovery work uses — does not store `params_json`, so the
   generator's own draws are not available for that family.
9. **Experiment B has never been run** (goal component 3, §2.3). The harness and metrics
   exist; the inertness threshold is uncalibrated; both configs are single-sample as
   written. Until it runs there is no evidence either way on whether a 3-node model
   invents a gene from 2-gene data.
10. **The validation target changed on 2026-07-29** to `k_star_fft` from the analytic
    `k_star`. Every k\* number recorded before that date was measured against the old
    target. See `STATE_OF_THE_SCIENCE.md` §8.

---

## 6. Reading order for the rest of this handoff

- `STATE_OF_THE_SCIENCE.md` — every measured result, every rejected approach, and the
  numbers that superseded earlier numbers. Read this before running anything. §2.7 covers
  the Endres/Matas-Gil method the current recipe derives from (200,000 iterations from a
  single image, and what their "batch of 128" actually means); §2.8 covers what the power
  spectrum's amplitude encodes; §8 covers the k\* validation-target reversal.
- `ROBUSTNESS_MEASUREMENT.md` — how the §4.3/§4.4 numbers were produced, the three
  defects in `robustness_cloud`, and what a comparable-to-Tica measurement requires.
- `DATA_INTO_MODEL.md` (already in this directory) — the firewall audit: exactly what
  the model sees, what it must never see, and where the current guard is weaker than
  the docs claim.
- `../CLAUDE.md` on branch `docs/agent-conventions` — the shared working contract.
