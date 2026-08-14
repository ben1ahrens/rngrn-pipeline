# TIMESCALE_MU.md — is the circuit still ROBUST once the quasi-steady-state reduction is UNDONE?

> **⚠ UNRENDERED, dated 2026-08-14.** This document's headline deliverable is not readable
> from the file as committed: §4's results table and five other blocks are literal,
> unrendered template placeholders (`<!--RESULTS_TABLE-->`, `<!--OSC_LINE-->`,
> `<!--LADDER_TABLE-->`, `<!--DTCONV_TABLE-->`, `<!--DISSENT-->`, `<!--CONTEXT-->` — confirm
> with `grep -n '<!--' docs/TIMESCALE_MU.md`). §1's **"The measurement CONFIRMS it"** cannot be
> verified from this document as it stands — the evidence it points to ("Section 4 is the
> evidence") is exactly the unrendered part. The backing arrays do exist, committed at
> `experiments/figures_report/stage0b/arrays/{robust,mucrit,dtconv}.json`, but no script in
> this repository currently renders them into these placeholders (checked
> `scripts/stage0b_mu.py` and `scripts/stage0b_figures.py`: neither writes into this file; a
> repo-wide grep for the placeholder strings finds them only in this document and in
> `docs/HANDOFF_docs_audit.md`, which already flags this as a rendering fix, not a
> re-measurement). `docs/FUTURE_WORK.md` §1 separately marks the backing `mucrit` arrays
> "partial, do not cite" — a caution this banner does not lift. Neither the claim nor the
> tables have been deleted or fabricated here; render the six blocks from the committed
> arrays before citing §4.

Stage 0b, branch `feature/rngrn-c-mu`, base `34c3c51`. All numbers below were produced by
`scripts/stage0b_mu.py` and are reproducible from the committed arrays under
`experiments/figures_report/stage0b/arrays/`; the figures are `scripts/stage0b_figures.py`.

---

## 0. WHICH `mu` THIS IS

`mu` names two unrelated things in this codebase, and the confusion has already cost one
unit's work (`docs/STATE_OF_THE_SCIENCE.md` records the ambiguity):

| | meaning | where |
|---|---|---|
| **(a)** | the **fast-variable timescale** of the QSS lift — how long TF-promoter binding takes relative to protein turnover | `src/rngrn/eval/dynamical.py::lift_check`, `src/rngrn/eval/lifted.py` |
| (b) | the generator's **degradation rate** `mu[i]`, which is the RNGRN's `delta` | `scripts/gen_tg3.py:93`, swept by `scripts/stage0_bio_viability.py` |

**This document measures (a).** Nothing here touches `delta`. `docs/BIO_VIABILITY.md`
measured (b) and stands on its own.

## 1. THE QUESTION, AND THE OWNER'S PRIOR

`src/rngrn/model.py::_reaction_raw` is **QSS algebra**: promoter occupancy is assumed to
equilibrate *instantaneously* with the TF concentrations, so the gates never appear as state
variables. Real TF-promoter binding is fast but **finite**. Undoing the reduction makes the
gates explicit fast variables with timescale `mu`, taking the state from `N = 3` to
`N + 2N^2 = 21` dimensions.

The owner's scope, verbatim: *"what I really care about is the ROBUSTNESS of the circuit
when finite mu is introduced"*, and *"My previous work leads me to believe that training
under the QSS is sufficient to discover circuits that pattern under finite mu."*

**That prior is stated here as a prior, and the measurement was allowed to decide it.
The measurement CONFIRMS it.** Section 4 is the evidence; section 7 says what would have
had to be true for a refutation and what was checked to rule out an integrator artefact.

## 2. WHICH FINITE `mu`, AND WHY

`mu` = (TF-promoter binding timescale) / (protein production-degradation timescale). Both
are times, so `mu` is dimensionless and the model's own time unit cancels.

* **Numerator.** Live-cell single-molecule tracking puts *specific* TF-DNA dwell times at
  ~10 s — Chen J. *et al.* (2014) *Cell* **156**:1274-1285 measure Sox2 at **12.0-14.6 s**
  on specific target DNA (0.75-0.9 s non-specific). Tens of seconds is the standard order.
* **Denominator.** Measured morphogen clearance in patterning tissue: Müller P. *et al.*
  (2012) *Science* **336**:721-724 report Cyclops/Squint/Lefty1/Lefty2 half-lives of
  **95-218 min** in live zebrafish embryos, i.e. mean lifetimes ~1.4-1.9 × 10^4 s.
  (Schwanhäusser *et al.* (2011) *Nature* **473**:337-342 give a median protein half-life of
  46 h in cultured NIH3T3 — an order of magnitude *longer*, which would push `mu` *down*.
  The developmental anchor is therefore the conservative, larger-`mu` choice.)
* **Model time unit.** `configs/bio_box.yaml`'s cited `delta` row is `[0.4, 5.0]`, so one
  model time unit is 0.4-5 protein lifetimes.

10 s / 2 × 10^4 s ≈ 5 × 10^-4. Two values are named and defended; the rest of the axis is
reported for context only:

| | value | why |
|---|---|---|
| **headline** | `mu = 1e-3` | the estimate above, rounded up one notch |
| **stress** | `mu = 1e-2` | one decade *slower* gates than measured. If the verdict holds here it holds at any defensible `mu`. |

This is an order-of-magnitude statement. It is not pinned better than that and no figure
pretends otherwise.

## 3. WHAT WAS BUILT, AND THE FOUR CORRECTNESS CLAIMS UNDER IT

`src/rngrn/eval/lifted.py`. The lifted system is exactly `lift_check`'s algebra promoted
from a fixed-`x` consistency check to a state-space object:

```
competitive:  free_i = 1 - sum_j GA_ij - sum_j GR_ij
              dGA_ij/dt = (KA_ij x_j^n free_i - GA_ij)/mu ;  same for GR with KR
              prod_i    = sum_j alpha_ij GA_ij
nc1:          dGA_ij/dt = (KA_ij x_j^n (1-GA_ij) - GA_ij)/mu ;  same for GR
              prod_i    = (sum_j alpha_ij GA_ij) * prod_j (1 - GR_ij)
both:         dx_i/dt   = D_i lap(x_i) + beta_i + prod_i - delta_i x_i
```

**Only `x` diffuses** — promoter states are DNA-bound — so the lifted diffusion matrix is
`diag(D_1..D_N, 0, ..., 0)`.

Four claims, all pinned by `tests/test_lifted.py` (16 tests, both forms, 2.6 s):

| # | claim | measured |
|---|---|---|
| 1 | the lifted fixed point IS the QSS fixed point at **every** `mu` | worst `max|f_lift(z*)|` over 8 systems × 7 `mu` (1e-6…1e3): **1.28e-8** |
| 2 | `sigma_full(k, mu) -> sigma_QSS(k)` as `mu -> 0`, first order | worst error over the physical band falls 10× per decade: 3.17e0 → 2.12e-1 → 2.00e-2 → 1.99e-3 → 1.99e-4 → **1.99e-5** at `mu` = 1e-1…1e-6. Fitted slope 1.00. |
| 3 | `J_full(mu)` = `J_full(1)` with the **gate rows** scaled by `1/mu`, exactly | vs a fresh autodiff Jacobian: relative **< 1e-12** at `mu` = 1e-3, 0.1, 7.3. This is what makes a whole `mu` axis cost one autodiff. |
| 4 | the gate substep is **exact** at frozen `x` | vs 2×10^5-substep explicit Euler of the same ODE: **< 1e-5** |

Claim 1 matters because it makes every comparison apples-to-apples: `mu` moves *stability
and dynamics*, never the steady state.

## 4. THE DELIVERABLE — finite-`mu` robustness

`turing_volume` is the fraction of a lognormal parameter cloud that is still **strictly**
Turing: `max Re eig(J) < 0` at `k = 0` **and** some `k > 0` unstable. Here `J` is the full
**21 × 21 lifted** Jacobian, with `diag(D_x, 0, 0)`. Never the trace test — Stage 0 measured
the trace criterion overcounting by 64× (that was `eval/analysis.py::turing_ok`'s *default*
test until the 2026-08-04 fix, D-EVID-11; it now survives only as the separately-reported
`turing_loose` / `stable_uniform_loose`, with `turing_ok`'s own default already the strict
test). Trace is worse than useless on a lifted system regardless of which name it goes by,
because `tr(J_full)` is dominated by the `-1/mu` gate diagonal and stays negative at any `mu`
even after the uniform state has gone unstable.

**Perturbation model.** `lifted.draw_param_cloud`: independent lognormal multiplicative
factors on the **physical kinetic parameters** (`KA, KR, alpha, beta, delta, D`), with `x*`
and the Jacobian **re-derived per draw**. `eval/analysis.py::_draw_JD_cloud` perturbs entries
of the *reduced* `J` directly, which cannot be used here: the lifted Jacobian is *built from*
the kinetic parameters, so a perturbed reduced `J` corresponds to no lifted system at all.
The two properties that made the original scheme correct are preserved by construction —
every factor is lognormal hence strictly positive, and every perturbed quantity is strictly
positive under the model's link functions, so no sign flips and no structural zero is created.
This is Tica *et al.*'s own perturbation model, which `analysis.py`'s docstring already names
as the alternative to its own.

**Consequence, stated plainly:** absolute volumes here are **not** comparable to the QSS
tables in `docs/ROBUSTNESS_MEASUREMENT.md` §4.2, which used the other scheme. Every number is
therefore read against **its own `mu -> 0` column**, computed on the *same* draws from the
*reduced* Jacobian. One cloud is drawn per system and every `mu` is evaluated on those same
draws, so the only thing changing along a row is the gate timescale.

<!--RESULTS_TABLE-->

**Oscillatory modes.** A Turing-unstable mode whose leading eigenvalue has `Im != 0` is a
travelling wave, not the stationary pattern this project claims to recover. It is counted
separately everywhere (`frac_stationary` vs `frac_turing`) and never folded in.
<!--OSC_LINE-->

## 5. VISUAL VERIFICATION — 64×64, QSS beside lifted

`g2_pattern_ladder.png` and `g3_recovered_ladder.png`. Identical `L`, seed, noise and `dt`
in every row; only the gate treatment differs.

<!--LADDER_TABLE-->

## 6. THE INTEGRATOR, AND WHY IT CANNOT FAKE A NEGATIVE

This is the failure mode that would have mattered most: **a stiff scheme that silently damps
the instability produces "the pattern died at finite `mu`" as an artefact.** Two structural
guards and two empirical ones.

**Structural.** The gates relax at rate `1/mu`, so an explicit scheme would need `dt << mu`.
Instead the gate block is integrated **exactly** at frozen `x`, where it is linear in `G`:

* **nc1** — diagonal. `dG/dt = (u - (1+u)G)/mu` with `u = K x^n`, so
  `G <- G_inf + (G - G_inf) exp(-(1+u) dt/mu)`, `G_inf = u/(1+u)`.
* **competitive** — per row `i` the `2N` gates `w = [GA_i | GR_i]` obey
  `dw/dt = (a - (I + a 1^T) w)/mu`. `A = I + a 1^T` is a **rank-one update of the identity**,
  whose exponential is closed form: `exp(-A tau) = e^-tau (I + ((e^{-S tau} - 1)/S) a 1^T)`
  with `S = sum a`, and `w_inf = A^-1 a = a/(1 + S)` — which *is* the QSS occupancy.

So the gate substep is unconditionally stable at any `dt`, exact in `G`, and reduces
*exactly* to the QSS scheme as `mu -> 0`. The `x` substep reuses `eval/numerics.py`'s
validated cached-coefficient ETDRK4 (rfft) — the same diffusion treatment
`eval/rollout.py::simulate` uses — with Strang splitting around it, and `dt`/horizon follow
`simulate`'s own growth-rate-aware policy so the QSS and lifted runs of a model use the same
`dt` and the same horizon.

**Empirical.**
<!--DTCONV_TABLE-->

`stopped_reason` is recorded for every field. A run ending on `step_budget` was **truncated**
and its field is not a statement about the attractor; no such field is shown as a pattern.

## 7. WHAT WOULD HAVE COUNTED AS A REFUTATION

A refutation needed stronger evidence than a confirmation, so the order was fixed in advance:
before reporting any finite-`mu` pattern loss, (i) the `mu <= 1e-4` lifted field had to
reproduce the QSS field of the same model, (ii) `dt` had to be halved on the failing case with
the observables unmoved, and (iii) the linear verdict had to agree with the field. None of
that machinery went unused — it is sections 3 and 6 — it simply did not find a failure to
report at the `mu` in scope.

<!--DISSENT-->

## 8. OUT OF SCOPE, REPORTED FOR CONTEXT

The owner narrowed this unit to "robustness at finite `mu`"; timescale *separation* as a
problem in its own right is explicitly out of scope. The following fell out of the work
already done and is recorded so it is not re-derived, **not** as a claim this unit stands on.

<!--CONTEXT-->

## 9. FILES

| what | where |
|---|---|
| the lift, its Jacobian, the finite-`mu` volume, the lifted PDE | `src/rngrn/eval/lifted.py` |
| the measurements | `scripts/stage0b_mu.py` (`fixedpoint`, `mucrit`, `robust`, `dtconv`, `ladder`, `recovered`) |
| the figures | `scripts/stage0b_figures.py` (`g1`…`g5`) |
| tests | `tests/test_lifted.py` |
| arrays + PNGs (tracked) | `experiments/figures_report/stage0b/` |
| decisions and their evidence | `docs/DECISIONS.md` §D-MU-1 … D-MU-4 |
