# BIO_VIABILITY.md — is the biological box Turing-EMPTY?

**Status: a FEASIBILITY PRECONDITION, not a pre-registered criterion.** Nothing here can
pass or fail `docs/PREREGISTRATION.md` §3, and per §5 nothing here moves a bound: an empty
box would be a headline finding, not a reason to widen a bound. What this unit answers is
the question nobody had asked — whether the region §3.4 demands (`plausibility_score` =
1.0, every *cited* row of `configs/bio_box.yaml` satisfied) contains any system that also
satisfies §3.1–3.3's pattern criteria at all.

**The one-line answer: NO, the box is not Turing-empty — and the co-existence is not merely
theoretical.** Two of the five real `three_gene_qvar` targets recovered with the biological
prior active come out **`plausibility_score = 1.0` AND patterned at the same time**, with a
visible labyrinthine field (§4, `f4_part4_prior_recovery.png` rows 4–5). §3.4 and §3.1–3.3
are jointly satisfiable, on real data, today.

**Three qualifications, all of which matter more than the headline.**

1. **Under a uniform prior the box is THIN.** The strictly-Turing acceptance rate inside it
   is **18 / 40 000 (4.5e-4) for `competitive`** and **1 / 40 000 (2.5e-5) for `nc1`** —
   `nc1` is ~18× harder and is therefore the binding form for §3.4. (§1. But see §1.4: most
   of that thinness is the width of the *unscored* binding prior, not the box.)
2. **The near-immobile third node — the escape valve `d_ratio_of` was written around — does
   the work.** It raises the `competitive` rate **17-fold**, and both of the real targets
   that pattern under the prior land in that regime (`D_lo/D_mid` = 0.026 and 0.039)
   *without being told to*. Parts 1 and 4 agree on the mechanism.
3. **The biological prior was a NO-OP and had to be fixed before part 4 could run at all.**
   `loss.weights.param_prior` was never read by the assembler `recover.py` calls, so a run
   configured with the prior on trained without it (§4.0, `docs/DECISIONS.md` D-EVID-5). No
   published number is affected — it was caught before it produced evidence.

Every figure below is committed under `experiments/figures_report/stage0/`, and the arrays
behind every panel under `experiments/figures_report/stage0/arrays/`. Decisions taken here
are registered as `D-BIO-1`…`D-BIO-7` and `D-EVID-5` in `docs/DECISIONS.md`.

---

## 0. What `mu` is, and one useful fact about it

The owner asked whether the model patterns "under the biological conditions (i.e. when we
introduce mu)". `scripts/gen_tg3.py:93` writes the generator's reaction as

```
out[i] = b[i] + V[i]*reg - mu[i]*x[i]
```

and `src/rngrn/model.py::_reaction_raw`'s last line writes the RNGRN's as

```
beta + prod - delta * x
```

so **the owner's `mu` IS the RNGRN's `delta`**, the degradation rate, and it is a CITED row
of `configs/bio_box.yaml` with bounds `[0.4, 5.0]`.

**The useful fact.** The generator draws `mu = 10**U(-0.4, 0.7)` = **0.398 … 5.01**
(`gen_tg3.py:159`). The box row is **[0.4, 5.0]**. These are the same range to two
significant figures. So unlike the D-ratio axis — where the generators sample 10^U(0.9,2.4)
≈ 7.9–251 (median ~135) against a box of [1, 60], the tension `bio_box.yaml` and
`STATE_OF_THE_SCIENCE.md` §11 both record in advance — **the degradation axis is one where
the box does not fight the data at all.** Anything that goes wrong on the `mu` axis is not
a generator/box mismatch.

`delta` moves BOTH the steady state `x*` (through `x* ≈ (beta + prod)/delta`) and the
Jacobian diagonal, so its effect on the Turing region is not obvious a priori. That is why
§2 sweeps it rather than asserting it.

---

## 1. Is the box Turing-empty? — the acceptance rate, with its denominator

**Figure:** `f1_box_acceptance_generator.png` · **Arrays:**
`arrays/part1_box_draws_generator.npz`, `arrays/part1_summary_generator.json`

**Method.** Draw the CONSTRAINED parameters log-uniformly *inside* the cited box rows —
`alpha ∈ [0.5, 10]` (9 entries), `delta ∈ [0.4, 5]` (3 entries), `d_ratio ∈ [1, 60]` — then
invert `model.py`'s link functions (`softplus`/`exp`/`sigmoid`, lines 196–215) to set
`theta`. Every draw therefore scores `plausibility_score = 1.0` with
`plausibility_n_scored = 3` **by construction** (asserted in the script against
`scoring/plausibility.py::plausibility_report`, not assumed). `beta` and the binding
parameters are unscored — see §1.4.

*The round trip is exact, and was checked rather than assumed.* Rebuilding a model from a
draw and reading back its constrained properties: `alpha` max abs error **4.4e-16**,
`delta` **0.0**, `D` max rel error **4.8e-16**, and the gated-budget invariant
`KA + KR == s` to **3.8e-11** (4 draws, N=3, float64). Serial `RNGRN` and `BatchedRNGRN`
member 0 agree to **0.0** on both `alpha` and `D`.

Steady state from `losses/terms.py::steady_state` (multistart Newton). **A draw whose
steady state does not converge is a FAILED draw**: it stays in the denominator with
`converged = False` and is never silently dropped.

**Verdict = the STRICT test** (`D-BIO-3`): `max Re eig(J) < 0` AND `max_{k>0} σ(k) > 1e-9`,
on `[0] ++ logspace(-3, 3.5, 500)`.

### 1.1 The table

`N = 3`, `n_hill = 2`, `seed = 0`, `beta_mode = generator`, **20 000 draws per
(form, D-scheme)**, 80 000 total. Wilson 95 % intervals (the normal interval is meaningless
at k of order 1).

| form | D-scheme | n | converged | **strict** | **rate** | Wilson 95 % | loose | loose rate |
|---|---|---|---|---|---|---|---|---|
| competitive | mobile3 | 20 000 | 19 944 | 1 | 0.00005 | [0.000009, 0.000283] | 270 | 0.01350 |
| competitive | immobile | 20 000 | 19 944 | **17** | **0.00085** | [0.000531, 0.001361] | 286 | 0.01430 |
| competitive | **POOLED** | 40 000 | 39 888 | **18** | **0.00045** | [0.000285, 0.000711] | 556 | 0.01390 |
| nc1 | mobile3 | 20 000 | 19 883 | **0** | 0.00000 | [0.000000, 0.000192] | 329 | 0.01645 |
| nc1 | immobile | 20 000 | 19 883 | 1 | 0.00005 | [0.000009, 0.000283] | 330 | 0.01650 |
| nc1 | **POOLED** | 40 000 | 39 766 | **1** | **0.00003** | [0.000004, 0.000142] | 659 | 0.01648 |

### 1.2 Independent replication — and one statistical caveat about it

A second pass at seed offset **+10 000** (an independent draw stream, not a re-analysis)
draws until it has 6 survivors per cell or hits a 300 000-draw cap:

| cell | survivors | draws | rate | part-1 rate | part-1 Wilson 95 % |
|---|---|---|---|---|---|
| competitive / mobile3 | 6 | 22 500 | 2.7e-4 | 5.0e-5 | [9e-6, 2.8e-4] |
| competitive / immobile | 6 | 2 500 | 2.4e-3 | 8.5e-4 | [5.3e-4, 1.4e-3] |
| nc1 / mobile3 | **5** | **300 000** | **1.7e-5** | 0 | [0, 1.9e-4] |
| nc1 / immobile | 6 | 27 500 | 2.2e-4 | 5.0e-5 | [9e-6, 2.8e-4] |

**Read these as a consistency check, NOT as a second measurement, because the stopping rule
biases them UPWARD.** A pass that stops as soon as it has `k` hits stops preferentially on a
lucky stretch, so `k/n` is a biased-high estimator (this is a negative-binomial design, not
a binomial one). `competitive/immobile` at 2.4e-3 — above part 1's Wilson interval — is that
bias, and it is expected: that cell stopped after its very first 2 500-draw batch. **Part
1's fixed-`n` table is the measurement; nothing in this document quotes a harvest rate as
the rate.**

The one row that IS unbiased is `nc1/mobile3`, because it hit the cap rather than the target:
**5 survivors in 300 000 draws, 1.7e-5**, a fixed-`n` estimate. That is the tightest number
this unit has for the hardest cell, and it converts part 1's uninformative `0/20 000` into a
real rate. It also confirms the direction of everything above: `nc1` with three mobile
species is the worst cell in the study by more than an order of magnitude.

### 1.2b The three things the part-1 table says

1. **NOT EMPTY, for both forms.** `competitive` 18/40 000, `nc1` 1/40 000. §3.4 and
   §3.1–3.3 are not mutually unsatisfiable. The ~30 GPU-hours are not being spent on the
   empty set.

2. **The near-immobile node is doing the work, exactly as designed.** `competitive`
   17 / 20 000 with a near-immobile third node against 1 / 20 000 without — a **17×**
   difference, and the two arms are a PAIRED comparison (same rng stream, differing only in
   the slowest D; `D-BIO-2`). For `nc1` it is the difference between 1 and 0. This is
   `scoring/plausibility.py::d_ratio_of`'s design intent measured directly: it takes the
   ratio of the two MOST MOBILE species so a near-immobile third node never enters the
   scored ratio, precisely so a ~7.5 D-ratio can be Turing-viable
   (`docs/ROBUSTNESS_MEASUREMENT.md` §4.4, where 127/127 generator systems stayed strictly
   Turing under immobilisation of the slowest diffuser). **Practical consequence:** a
   recovery run that is expected to land in the box should not be pushed toward three
   comparably-mobile species.

3. **`nc1` is ~18× harder than `competitive` inside this box** (4.5e-4 vs 2.5e-5 pooled).
   `PREREGISTRATION.md` §3 requires **both** forms to pass. On this evidence `nc1` is the
   binding form for criterion 3.4, and it is where tuning effort should go.

### 1.3 The trace test overcounts by ~64× here — do not use `turing_ok`'s number

On the same 80 000 draws the loose (`tr(J) < 0`) criterion accepts **1215** and the strict
criterion **19**. `eval/analysis.py::turing_ok` uses the loose test.

**All 1196 loose-only draws have `max Re eig(J) > 0`** (measured: 0 of 1196 had
`max Re eig(J) ≤ 0`). They are **uniformly unstable** states whose dispersion peaks at
k → 0 — a homogeneous instability, not a Turing pattern. This is `f1` panel (c): almost
every draw sits on the diagonal `σ_max = max Re eig(J)`, and the Turing corner is the small
upper-left population off that diagonal.

This is consistent with, and sharper than, `docs/ROBUSTNESS_MEASUREMENT.md` §3's finding
that the trace test overcounts by up to 70 % of draws.

**The disagreement is entirely one-directional, and that was checked.** Re-running
`eval/analysis.py::turing_ok` verbatim — its own `linspace(1e-3, 50, 4000)` grid and all —
on all **19 strict survivors** returns `True` on **19/19**, with `k*` agreeing to within
1.3 % on every one (largest gap 17.12 vs 16.90). So `turing_ok` never *misses* a genuine
Turing system here; it only adds 1196 uniform instabilities. Reported for continuity, and
because it also validates the log k-grid of `D-BIO-3` against the library's own grid on
the population where both are applicable.

> **⚠ CORRECTED 2026-08-14 — `turing_ok`'s default changed under this section.** See
> `src/rngrn/eval/analysis.py:30-35,72` and `docs/DECISIONS.md` D-EVID-11 (2026-08-04, after
> this section was written). `turing_ok`'s primary test (`stable_uniform`, feeding `ok`) is
> now the **strict** `max Re eig(J) < 0` — the same test this section argues for — not the
> `tr(J) < 0` test the section header and "`eval/analysis.py::turing_ok` uses the loose test"
> (above) describe. The old trace test now survives only as the separately-reported
> `turing_loose` / `stable_uniform_loose`. So the header's "do not use `turing_ok`'s number"
> is no longer the right caution: today `turing_ok`'s own `ok` already reflects the strict
> criterion this section measured at 19/80 000. The measurement itself (1215 loose vs. 19
> strict on these 80 000 draws) is untouched, as is everything below this box.

### 1.4 The unscored directions, and what the rate is a rate *of*

`beta` is `source: UNCITED` and is never scored (`D-BIO-1`). The default arm draws it from
the generator's own basal distribution `10**U(-2, -0.3)`. `KA`/`KR` have **no box row at
all** (`D-BIO-7`): the binding budget `s` is drawn `10**U(-2, 2)` and the gate logit
`N(0, 2.5)`.

**Where the survivors actually live** (`f1` panel (d), 18 competitive + 1 nc1 survivor):

| quantity | min | median | max | prior support |
|---|---|---|---|---|
| `d_ratio` | 1.11 | 10.16 | 56.43 | [1, 60] — the whole box row |
| geometric-mean `s` | 0.312 | 1.166 | 5.656 | 1e-2 … 1e2 (4 decades) |
| `delta` (all entries) | 0.40 | 1.33 | 4.91 | [0.4, 5] — the whole box row |
| `alpha` (all entries) | 0.51 | 2.57 | 9.95 | [0.5, 10] — the whole box row |

So the Turing region spans the **entire** cited box on every scored axis, and is narrow only
in the **unscored** binding direction: survivors occupy about **1.26 of the prior's 4
decades** in `s`. That is the honest reading of the low acceptance rate — **it is largely a
statement about the width of the unscored prior, not about the box being nearly empty.**
A recovery that is already in a sensible binding regime faces far better odds than 4.5e-4.

**What this rate is NOT.** It is not the probability that recovery lands in the box: recovery
is an optimisation with a Turing hinge in its objective, not a uniform draw. It is the
volume fraction under a stated prior, and its job is to answer empty / not-empty and to rank
the two forms and the two D-schemes against each other.

---

## 2. The `mu` (= `delta`) axis, swept

**Figures:** `f2_mu_boundary_generator.png` (uniform rescale, both forms, several base
systems), `f2b_mu_per_species_generator.png` (one `delta_i` at a time) ·
**Arrays:** `arrays/part2_mu_sweep_generator.npz`

**Method** (`D-BIO-5`). Take strictly-Turing base systems from the harvest, hold everything
else fixed, and sweep `delta` over **41 log-spaced points on [0.04, 50]** — the cited box
row `[0.4, 5.0]` plus one decade either side, so the boundary is *located* relative to the
box edges rather than asserted to be outside them. Two parameterisations, because the box
bounds every species and a uniform rescale is the easy case:

* **uniform** — rescale the base draw's per-species `delta` pattern so its geometric mean
  equals the swept value. `all_delta_in_box` is recorded per point, because an individual
  `delta_i` can leave `[0.4, 5]` while the mean is inside it.
* **per_species** — move ONE `delta_i`, hold the rest at the base draw.

### 2.1 The answer: the patterning window lies ENTIRELY INSIDE the box, on 12 of 12 systems

12 base systems (6 per form, **round-robined 3 `mobile3` / 3 `immobile`** so the band is not
one scheme's anecdote), 42 swept points each, 504 evaluations. 32 of the 504 did not converge
to a steady state and are recorded as such (they appear as breaks in the curves, not as
zeros).

| | competitive | nc1 |
|---|---|---|
| base systems with a strictly-Turing window | 6/6 | 6/6 |
| narrowest window | 1.681 – 1.690 | 1.402 – 1.690 |
| widest window | **0.693 – 2.885** | **1.690 – 2.885** |
| window entirely inside [0.4, 5.0] | **6/6** | **6/6** |

Pooled over all 12: `delta_lo` spans 0.58–3.07 (median 1.40), `delta_hi` spans 0.59–3.07
(median 1.69). **No window has a single strictly-Turing point below 0.4 or above 5.0, and
none reaches either end of the swept range [0.04, 50].**

So, to the question the sweep was built to answer — *does patterning die inside the box or
outside it?* — **it dies INSIDE the box, at both ends, with room to spare.** The cited
`delta` row [0.4, 5.0] is **wider** than the Turing window it contains, on every system
measured. **`delta` is not the binding constraint.** Combined with §0's observation that the
box row and the generator's own `mu` draw are the same range to two significant figures,
the degradation axis is the one axis of `bio_box.yaml` where nothing is in tension with
anything.

### 2.2 A trap in reading the figure: the high-`delta` excursions are NOT patterning

Several curves in `f2` rise back above σ = 0 at large `delta` (competitive base1 at
δ ≈ 4.9–7.0, base5 at δ ≈ 7.0–8.4, nc1 base5 at δ ≈ 3.5). **None of those points carry a
dot, because none is Turing.** Checked directly: their `max Re eig(J)` is +1.155, +0.802 and
+0.508 respectively — the uniform state has gone unstable, so this is a HOMOGENEOUS
instability, not a pattern-forming one. Over all 504 swept points, 61 have
`max_{k>0} σ(k) > 0` and **19 of those (31 %) are uniform instabilities**; only 42 are
strictly Turing. This is the same trap §1.3 measures on the box draws, and it is why the
dots (not the curve's sign) are the readout.

### 2.3 Per-species: the window is narrow in every single `delta_i` too

Moving ONE `delta_i` and holding the other two at the base draw (`f2b`, base0 of each form):

| form | `delta_0` | `delta_1` | `delta_2` |
|---|---|---|---|
| competitive | 4.48 – 5.89 | 0.674 – 0.693 | 1.573 only |
| nc1 | 1.405 – 1.414 | 2.010 – 2.020 | 0.693 – 0.990 |

Five of the six per-species windows lie inside [0.4, 5.0]; the exception is competitive
`delta_0`, whose window **4.48–5.89 straddles the box's upper edge of 5.0**. So the uniform
rescale really was the easy case: a single species' degradation rate can carry the system
across the box boundary even when the geometric mean cannot. **[UNCALIBRATED]** — this is
one base system per form, and it identifies a direction worth watching rather than
establishing a rate.

---

## 3. The patterns, at 64×64

**Figures:** `f3_patterns_competitive_generator.png`, `f3_patterns_nc1_generator.png` ·
**Arrays:** `arrays/part3_rollouts_generator.npz` (the FIELDS themselves),
`arrays/part3_rollouts_generator.json`

**Method** (`D-BIO-6`). `eval/rollout.py::simulate` with `n = 64`,
`integrator = "etdrk4_rfft"`, `early_stop = True`, `L = 6 × 2π/k*` — six full wavelengths of
the fastest-growing linear mode across the box, which is the generator's own original choice
(`gen_tg3.py:66`) and inside its replacement acceptance window `{3..14}`.

Each panel is labelled with `patterned`, `k*_lin` (linear prediction) vs `k*_field`
(measured from the field), amplitude, morphology class, `D`-ratio, the `delta` range, and
**`stopped_reason`** — a run that ended on `step_budget` was TRUNCATED and its field is not
a statement about the attractor, so it is marked as such rather than presented as a pattern.
`sig_max` is the SIGNED dispersion maximum: negative means linearly stable and nothing can
grow from noise. Each form's gallery includes a **non-patterning control** drawn from the
part-1 population (red title), because a figure that only shows successes is not evidence.

### 3.1 The headline: 9 of 12 box-constrained strictly-Turing systems form a visible pattern

| | count |
|---|---|
| rollouts | 14 (12 strictly-Turing + 2 non-Turing controls) |
| strictly Turing **and** patterned | **9 / 12** |
| strictly Turing, no pattern | 3 / 12 |
| non-Turing controls patterned | **0 / 2** — the controls behave |

Morphology of the 9: **8 labyrinth, 1 spots** (baked-in centroid bank, so labels are
hypotheses — see §6.6).

`f3_patterns_competitive_generator.png` and `f3_patterns_nc1_generator.png` are the answer
to the owner's request: real spot and labyrinth fields, on a 64×64 grid, from systems whose
`plausibility_score` is 1.0 by construction.

### 3.2 A DEFECT IN OUR OWN READOUT, found and fixed here: `patterned` reads channel 0 only

The first pass reported **4/12** patterned, not 9/12. The gap was ours, not the model's.
`eval/rollout.py::simulate` defines `amplitude` and `patterned` on **channel 0 only** —
entirely correct for a real dataset, where channel 0 is the observed channel. But
`draw_batch` **shuffles the species order** (nothing distinguishes the species except which
one got the small D), so channel 0 is arbitrary here. Measured on one competitive/`mobile3`
survivor: channel-0 std **2.0e-4**, channel-1 std **0.76**. The library called it
unpatterned while a large labyrinth sat one channel over.

Both verdicts are now recorded and neither overrides the other: `patterned` is the library's
channel-0 number, untouched, and `patterned_any` is true when ANY channel clears its own
`max(1e-3, 0.02·|mean|)` floor. **Every panel shows the channel with the largest spatial std
and names it in the title**, with all three channel stds listed. The 9/12 above is
`patterned_any`; the library's channel-0 count is 4/12 and is in the arrays.

### 3.3 The early-stop rule was checked, not trusted

Eight panels stopped on `saturated` with a small channel-0 amplitude — which is EXACTLY the
false positive `_saturated`'s own docstring warns about ("a too-loose tol stopping during the
induction period and reporting an unpatterned field ... shows up as
`stopped_reason='saturated'` together with `patterned=False`"). So all 12 strictly-Turing
systems were re-run with `early_stop=False` at the full 40-growth-time horizon (up to 17 753
steps): **0 / 12 verdicts changed**, and the final amplitudes agree to 4 significant figures.
The early stop is not manufacturing negatives here.

No panel ended on `step_budget`, so no field shown is truncated.

### 3.4 Where the realised k\* sits relative to the linear prediction

On the 9 patterned systems, `k*_field / k*_lin` has median **0.87** and range **0.27–1.05**.
Agreement is good (0.77–1.05) for the seven systems with `k*_lin < 10`, and poor for the two
with `k*_lin` of 27.7 and 34.1, where the realised wavenumber is 0.27× and 0.52× the linear
one. Those two run at L = 1.36 and 1.10 with only ~10.7 pixels per predicted wavelength, so
whether this is genuine mode selection or a 64×64 resolution effect is **not established
here**; they are flagged rather than explained. The seven well-resolved systems are the ones
to read.

---

## 4. Five real targets, recovered WITH the biological prior

**Figure:** `f4_part4_prior_recovery.png` · **Arrays:** `arrays/part4_prior_recovery.json`,
plus the full tracked run records under `experiments/stage0_bioviab/`

### 4.0 A defect had to be fixed first: the prior was never in the objective

`loss.weights.param_prior` was a **NO-OP** on the path `recover.py` runs. `losses/total.py`
— the assembler `recover.py` calls — never referenced `losses/terms.py::param_prior`; its
only caller was `terms.composite_loss`, the standalone reference form. `train.py` also never
passed `cfg.loss.dratio_centre` / `dratio_spread` / `bio_box_path` through to `recover()`.
A run configured with the prior on would have trained **without** it and recorded
`w_param_prior: 1.0` on its index row. Full write-up, and the reason no published number is
affected, in `docs/DECISIONS.md` `D-EVID-5`.

Fixed in this branch, guarded so the default path is byte-identical (`param_prior_kw=None`
omits the key from `term_vals` entirely), with a batched twin `param_prior_batched` pinned
equal to the serial term member by member, and 4 new tests.

### 4.1 Method

`three_gene_qvar` `sample_0000` … `sample_0004` — five, the owner's stated sufficiency, and
all inside `PREREGISTRATION.md` §1a's **TUNING** range, so nothing held out is burned.
`sample_0008` … `sample_0033` untouched.

Two arms per target, **prior ON** (`loss.weights.param_prior = 1.0`: D-ratio log-normal
centred at the literature 7.5, plus the soft `alpha`/`delta` box hinges) and **prior OFF**
(0.0) as the control — without the control, "the prior made this viable" has nothing to be
measured against. 400 Adam steps, 64 batched restarts, CUDA, `dispersion_backend = cubic`,
`lbfgs_steps = 0`, float64, one seed each. **This is a VISUAL check, indicative only — it is
not a measurement of recovery quality; that is Stage 1's job.**

### 4.2 The table — all ten runs

`three_gene_qvar`, N=3, m=3, 400 Adam steps, 64 batched restarts, seed 0, CUDA float64.
`D_lo/D_mid` is the ratio of the SLOWEST to the middle diffusivity — *not* the scored
`d_ratio`, which by construction excludes the slowest species. It is shown because §1
identified that direction as the one that decides Turing viability.

| target | arm | plausibility | scored `d_ratio` | `D_lo/D_mid` | **patterned** | `sig_max` | `kstar_fft_rel_err` | `trivial_kstar_err` |
|---|---|---|---|---|---|---|---|---|
| sample_0000 | prior OFF | 0.33 | 1.45 | 0.801 | False | −0.0267 | 0.979 | 1.000 |
| sample_0000 | **prior ON** | **1.00** | 7.50 | 0.171 | False | −0.0419 | 0.979 | 1.000 |
| sample_0001 | prior OFF | 0.67 | 3.88 | 0.931 | False | −0.0252 | 0.977 | 0.250 |
| sample_0001 | **prior ON** | **1.00** | 7.59 | 0.044 | False | −0.0647 | 0.285 | 0.250 |
| sample_0002 | prior OFF | 0.33 | 2.84 | 0.114 | False | −0.0541 | 0.979 | 0.400 |
| sample_0002 | **prior ON** | **1.00** | 7.50 | 0.090 | False | −0.0986 | 0.979 | 0.400 |
| sample_0003 | prior OFF | 0.33 | **421.21** | 0.640 | **True** | +0.2367 | 0.017 | 0.143 |
| sample_0003 | **prior ON** | **1.00** | 7.69 | 0.026 | **True** | +0.0261 | 0.087 | 0.143 |
| sample_0004 | prior OFF | 0.33 | **261.07** | 0.672 | **True** | +0.2564 | 0.046 | 0.000 |
| sample_0004 | **prior ON** | **1.00** | 7.65 | 0.039 | **True** | +0.0195 | 0.116 | 0.000 |

### 4.3 Five things this says, in order of importance

**1. §3.4 AND patterning CO-OCCUR on real data. This is the answer to the owner's
question.** `sample_0003` and `sample_0004` with the prior ON are simultaneously
`plausibility_score = 1.0` and `patterned = True`, with `sig_max` strictly positive and a
visible labyrinthine field (`f4` rows 4 and 5). The model **does** form Turing patterns
under biological conditions — not as a synthetic construction, but recovered from a real
target with every cited box row satisfied.

**2. The prior attains `plausibility_score = 1.0` on 5/5, reliably and cheaply.** It pins
the scored `d_ratio` to 7.50–7.69 — its own centre — every time, and pulls `alpha` and
`delta` inside their rows. The control confirms this is the prior's doing: without it the
score is 0.33 on four targets and 0.67 on one. **Criterion 3.4 is not the hard part.**

**3. The prior does not decide WHICH targets pattern.** 2/5 in the prior-ON arm and the
SAME 2/5 in the prior-OFF arm. Whatever stops `sample_0000/1/2` from patterning at 400
steps, it is not the biological prior — that is exactly what the control was run to
establish, and it is the single most useful thing in this part. `f5` shows why those runs
fail: their σ(k) is monotone decreasing and everywhere negative, peaking at k → 0. They are
linearly stable, so no integration time can produce a pattern.

**4. The prior stops the D-ratio escape, and the escape was real.** On precisely the two
targets that pattern, the unconstrained arm runs off to `d_ratio` **421** and **261** — far
above the box's ceiling of 60, and squarely in the generator population `bio_box.yaml` and
`STATE_OF_THE_SCIENCE.md` §11 record as the expected tension (10^U(0.9,2.4) ≈ 7.9–251,
median ~135). The prior holds it at 7.5 **and still patterns**.

**5. Parts 1 and 4 agree on the mechanism, and this was not designed in.** The two
prior-ON runs that pattern have `D_lo/D_mid` = **0.026** and **0.039** — a slowest species
26–38× less mobile than the middle one, i.e. squarely in §1's `immobile` regime, the one
measured there as 17× more Turing-viable. The prior-OFF runs that pattern sit at 0.64 and
0.67, comfortably `mobile3`. Nothing in `param_prior` mentions the slowest D — `d_ratio_of`
excludes it by construction — so the prior reaches the escape valve on its own, by
squeezing the two *mobile* species together and leaving recovery no other way to build an
instability. That is the clearest possible confirmation that `d_ratio_of`'s design choice
is load-bearing rather than cosmetic.

### 4.4 The cost, stated plainly

The prior is not free. On the two targets where a k\* comparison is meaningful:

| target | `kstar_fft_rel_err` prior OFF | prior ON | ~~`trivial_kstar_err`~~ | **`trivial_kstar_fft_err`** |
|---|---|---|---|---|
| sample_0003 | 0.017 | **0.087** | ~~0.143~~ | **0.1997** |
| sample_0004 | 0.046 | **0.116** | ~~0.000~~ | **0.0754** |

and `sig_max` drops by an order of magnitude (+0.237 → +0.026, +0.256 → +0.020), i.e. the
box-constrained solutions are much closer to marginal.

> **⚠ CORRECTED 2026-08-04 — the control column was the WRONG ONE, and a conclusion drawn
> from it has changed.** See `docs/DECISIONS.md` D-EVID-7. `trivial_kstar_err` is
> normalised by the LINEAR reference `answer_key.kstar`, while `kstar_fft_rel_err` is
> measured against `answer_key.kstar_fft` — pairing them compares errors against two
> different denominators. The honest control is `trivial_kstar_fft_err`, recomputed above
> from these same rows' stored `kstar_true` / `kstar_fft_true`.
>
> **What changes:**
> - **`sample_0004` prior OFF (0.046) DOES beat its honest baseline (0.0754).** The old
>   text said that row "carries no information about the model either way" because the
>   trivial error was 0.000. That was an artefact of the wrong denominator — the image-blind
>   predictor is exact against the *linear* k\*, not against the FFT k\* the claim is gated
>   on. It is in fact the strongest single k\* number in this table.
> - **`sample_0004` prior ON (0.116) LOSES to that baseline (0.0754)** — a cost the old
>   framing hid entirely, and one that *strengthens* §4.4's thesis rather than weakening it.
> - `sample_0003` is unchanged in direction: both arms still beat their control (now
>   0.1997), prior-ON still sits above §3.3's 8.3 % bar, prior-OFF still comfortably under.
>
> **Also note the resolution floor.** These rows carry `kstar_fft_bin_width` 0.143 and
> 0.167 — half-bin floors of 7.1 % and 8.3 % — so the pre-registered 8.3 % bar sits at or
> below the estimator's own resolution on both (D-EVID-8). That is the owner's to settle.

**Read the k\* numbers with the control computed against the SAME reference.**
`PREREGISTRATION.md` §3.3 still names `trivial_kstar_err`; D-EVID-7 records that this is now
the wrong column for the headline, and amending a pre-registered condition is owner-only.

This is precisely the trade-off §3.4 anticipated in writing — *"If viability and recovery
rate trade off, that trade-off is a result and gets reported as one; the centre does not
move to make the bar easier."* It is reported here as one. **One seed, 400 steps, two
usable targets: this is indicative, not a measurement.** Whether the gap survives a real
seed budget is Stage 1's job.

---

## 5. What this means for §3.4 co-existing with §3.1–3.3

**1. §3.4 is satisfiable, and it is not the binding criterion.** The box is non-empty for
both forms (§1), the entire `delta` patterning window lies inside the cited row (§2), 9 of 12
box-constrained systems form a visible pattern (§3), and — the decisive evidence — **two of
five real targets recovered with the prior active are simultaneously `plausibility_score` =
1.0 and patterned** (§4). The prior reaches 1.0 on 5/5 targets. Nothing here suggests §3.4
is the criterion that will fail.

**2. The binding criteria are §3.1–3.3.** In part 4 the same 2/5 targets pattern with the
prior ON and with it OFF; the other three produce a σ(k) that is monotone decreasing and
everywhere negative in both arms (`f5`). Recovery at 400 steps / 64 restarts simply does not
find a Turing regime for them, and the biological prior is not why. **That is where the ~30
GPU-hours should go.**

**3. `nc1` is the binding FORM, by roughly an order of magnitude.** Pooled acceptance
2.5e-5 against `competitive`'s 4.5e-4, and the harvest's unbiased `nc1/mobile3` cell is
1.7e-5 in 300 000 draws. `PREREGISTRATION.md` §3 requires BOTH forms to pass. Whatever
budget is allocated between the forms should not be equal.

**4. The near-immobile third node is the mechanism, and it should be allowed to operate.**
It raises the `competitive` in-box acceptance 17-fold (§1.2b), and in part 4 the prior drives
recovery into that regime *on its own* (`D_lo/D_mid` = 0.026 and 0.039 on the two targets
that pattern, against 0.64 and 0.67 without the prior) despite `param_prior` never mentioning
the slowest D. Any future change that penalises a small D — a floor on diffusivity, a
max/min D-ratio definition — would close the door this result walks through, and
`d_ratio_of`'s exclusion of the smallest D should be treated as load-bearing, not incidental.

**5. There is a real, measured cost, and it is on the k\* axis.** `kstar_fft_rel_err` 0.017 →
0.087 and 0.046 → 0.116 when the prior is switched on, and 0.087 sits just above §3.3's
pre-registered 8.3 % bar. One seed, 400 steps, two usable targets — indicative only. But
§3.4 anticipated this trade-off in writing and said it would be reported rather than
engineered away, so it is reported. **No bound was moved.**

**6. One thing that must be fixed before any tuning wave runs.** `loss.weights.param_prior`
was a no-op (§4.0, `D-EVID-5`). It is fixed on this branch. Any run launched from an older
commit with the prior "on" trained without it.

---

## 6. Limitations, stated

1. **The acceptance rate is a rate under a stated prior**, and the prior on the unscored
   binding directions is four decades wide and **[UNCALIBRATED]** (`D-BIO-7`). A narrower,
   better-justified binding prior would raise every rate in §1. What is prior-independent is
   the emptiness verdict, the ranking of the two D-schemes, and the ranking of the two forms.
2. **`n_hill` is fixed at 2** throughout. The generator draws `n ∈ {2, 3, 4}`
   (`gen_tg3.py:161`). The Turing region's dependence on the Hill exponent is not measured
   here.
3. **`steady_state` returns ONE root.** A model with multiple positive steady states is
   judged at whichever one the multistart Newton reaches. Systems that are Turing at a root
   the solver does not find are counted as non-Turing, so §1's rates are **lower bounds**.
4. **`nc1`'s numbers rest on few survivors.** One survivor in 40 000 part-1 draws; the
   harvest adds 11 more (5 `mobile3` in 300 000 draws, 6 `immobile` in 27 500). Everything
   said about `nc1` in §2 and §3 rests on those 11 and should be read as such.
7. **The `k*` disagreement on the two highest-`k*` panels is unexplained** (§3.4:
   `k*_field/k*_lin` = 0.27 and 0.52 at L = 1.36 and 1.10). Genuine mode selection and a
   64×64 resolution effect are both consistent with what was measured; neither was tested.
8. **The 3 strictly-Turing systems that never pattern are not explained either.** Their
   fields go to std = 0 exactly in all three channels. A linear instability that does not
   produce a nonlinear pattern is an ordinary outcome (subcritical branch, or relaxation to a
   different homogeneous root — see limitation 3), but which one applies here was not
   determined.
5. **Part 4 is one seed per target per arm at 400 steps.** It cannot support a claim about
   recovery quality, only about whether the prior is reachable and what it does to the
   plausibility score and the rolled-out field.
6. **The morphology class in §3 uses the baked-in centroid bank**
   (`morphology.default_reference_bank()`, 82.1 % accuracy vs 89.7 % for a real-sample bank),
   because these systems are not dataset samples. Labels are hypotheses, per the scorer's own
   documented limitation.
