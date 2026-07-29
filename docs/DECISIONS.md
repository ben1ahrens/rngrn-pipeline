# DECISIONS.md — the science decision register

**Status: information, not instruction.** This is an assembly, not new analysis. Every
entry below is traceable to something already in the repo (a merged PR body, a
docstring, a `[TUNE]`/`[IMPL]`/`[VALIDATE]` marker, or another doc) — nothing here was
invented or re-decided while writing this file. Where a source itself records an open
tension, that tension is preserved, not resolved.

This project is headed for a paper. The point of this register is that every number
in a future write-up can be traced back to the decision that shaped it, who made that
decision, and what evidence it was checked against.

**Format per entry:** ID · date · status (DECIDED / OPEN / SUPERSEDED) · the decision
in one sentence · who decided it · the evidence · what was rejected and why · where it
lives in code.

**Status legend**
- **DECIDED** — a choice has been made and is live in the code/config as described.
- **OPEN** — laid out, not yet chosen; picking it would bias what later numbers mean.
- **SUPERSEDED** — a later decision reversed this one; kept for the record because
  numbers measured under the old decision are not comparable to numbers measured
  after it.

---

## Part 1 — Evidence-integrity defects (read this before trusting any pre-2026-07-29 number)

These are not science decisions; they are defects in how evidence was recorded. They
are listed first, prominently, because they change how earlier numbers should be read.

### D-EVID-1 — `cli.py -o` silently kept only the last override

**Date found:** 2026-07-29 (unit 10, PR 10). **Fixed:** 2026-07-29 (merge `e662a15`).
**Status:** SUPERSEDED (defect; now fixed).

`sp.add_argument("-o", "--override", nargs="*", ...)` without `action="extend"` meant
repeated `-o key1=val1 -o key2=val2` flags on the CLI **replaced** `args.override` on
each occurrence instead of accumulating — so only the last `-o` took effect. Any prior
run invoked with multiple `-o` flags silently ran with the config's *default* values
for every override before the last one, not the budget the command line actually
asked for.

**Found by:** unit 10, while trying to reproduce a stage-2 e2e run
(`docs/CODE_REALITY.md` §"NOT fixed" list on PR 10). Confirmed by comparing
restart/step progress logs against the actually-requested override values.

**Fix:** `src/rngrn/cli.py:149` — `nargs="*", action="extend"`, with a comment
recording why it is load-bearing.

**Consequence for earlier results:** any run whose invocation used more than one `-o`
flag (rather than one `-o` with space-separated `key=val key=val`) needs its actual
effective config checked against `frozen_config.yaml`/`train_results.json`, not
assumed from the command line that launched it.

### D-EVID-2 — the pre-push hook printed "skipping tests" and exited 0 for all 13 phase-A units

**Date found / fixed:** 2026-07-29 (merge `e662a15`, "Integrate the 13-unit wave:
repair merge damage and close three wiring gaps"). **Status:** SUPERSEDED (defect;
now fixed).

The pre-push hook (`.githooks/pre-push`), meant to be the authoritative local test
gate, silently took the "no python interpreter found — skipping tests" branch and
exited 0 — every one of the 13 phase-A units pushed with a **green pre-push that ran
no tests at all**. PR 11's own body independently reports the same symptom
(`git push printed "pre-push: no python interpreter found; skipping tests."`),
corroborating this from the unit-11 side.

**Fix:** `.githooks/pre-push:31` now `echo "pre-push: no python interpreter found —
REFUSING to push untested." >&2` and fails the push instead of skipping silently. The
fix's own comment (`.githooks/pre-push:19`) records the defect it replaces almost
verbatim.

**Consequence for earlier results:** none of the 13 phase-A PRs' "tests pass" claims
were enforced by the push gate — each PR's reported pytest counts were produced by
manually running `pytest -q` inside the session (visible in each PR body above), so
the counts themselves are still evidence, but nothing mechanically prevented an
untested push from having landed. `main` @ `4509632` is the first commit with the real
gate; anything before it relied entirely on the acting agent's self-report.

**Where it lives:** `.githooks/pre-push`, `docs/CODE_REALITY.md` §8 ("The
authoritative test run is local").

---

## Part 2 — Decisions

### D1 — D-ratio prior centred at 7.5 (literature Nodal/Lefty), not ~100 (generator median)

**Date:** 2026-07-29 (unit 5 session). **Status:** DECIDED, with a deliberately
unresolved tension recorded alongside it. **Decided by:** the user, per
`configs/bio_box.yaml`'s header comment ("USER DECISION ON RECORD (see the unit 5
task brief)") — **reversing an earlier in-session choice of ~100** (matching the
generator's own draws).

**The decision:** the biological-plausibility box's D-ratio prior centre is **7.5**,
the measured Nodal/Lefty differential-diffusion ratio in live zebrafish (canonical
literature value, `docs/STATE_OF_THE_SCIENCE.md` §11), not ~100–135, which is where
the `three_gene` generator's own sampling distribution sits.

**Evidence:**
- `three_gene` generator D-ratio: min 16.6 / **median 134.9** / max 249.3 over 127
  samples (`docs/STATE_OF_THE_SCIENCE.md` §7, table). Sampling range quoted elsewhere
  as `10**U(0.9, 2.4)` ≈ 7.9–251.
- Literature anchor: Nodal diffuses ~7.5× slower than Lefty (zebrafish, cited in
  `docs/STATE_OF_THE_SCIENCE.md` §11) — a **~15× gap** from the generator median.
- The tension is explicitly *not* treated as a contradiction: the same literature
  reports that in extended networks (more nodes) differential diffusivity is not
  required at all for patterning, and this project's own data corroborates a version
  of that — `docs/ROBUSTNESS_MEASUREMENT.md` §4.4 measures that immobilising the
  slowest diffuser (making D-ratio effectively infinite in the classical sense) keeps
  **127/127** systems strictly Turing and *improves* median robustness (1.000 vs
  0.960 baseline).

**What was rejected and why:** centring on ~100 (the generator's own median) was the
earlier in-session choice, reversed because it would calibrate a "biological
plausibility" prior against the project's own synthetic sampling box rather than
against literature — precisely the kind of firewall-adjacent circularity
`docs/STATE_OF_THE_SCIENCE.md` §1 warns about for any new prior: "Centring a D-ratio
prior on the measured median true D-ratio... routes `AnswerKey.D` into recovery
through judgement rather than through an import."

**Deliberately left as an unresolved tension, not settled here:** the ~15× gap
between 7.5 and the generator's ~135 is recorded as intentional and unresolved in
`docs/STATE_OF_THE_SCIENCE.md` §11 ("Open design choice: centre the prior on ~7.5
... or ~100 ... a wide spread spanning both keeps the prior informative but honest").
The *bio_box* prior settles on 7.5; the wider open question of whether training itself
should ever be pointed at ~100 is not closed by that choice.

**Where it lives:** `configs/bio_box.yaml` (`d_ratio.centre: 7.5`, `spread: 1.0`
natural-log units, bounds `[1.0, 60.0]`); `src/rngrn/scoring/plausibility.py`;
`src/rngrn/losses/terms.py::param_prior` (`loss.dratio_centre` default `7.5`,
`loss.dratio_spread`); `docs/STATE_OF_THE_SCIENCE.md` §11.

### D2 — `d_ratio` is defined as the ratio of the two MOST-MOBILE species, excluding an immobile node

**Date:** 2026-07-29 (PR 6, unit 5). **Status:** DECIDED (a design choice, flagged as
`[IMPL]`, not independently validated against alternatives). **Decided by:** the
implementing agent under delegated authority, reasoned from measured robustness data.

**The decision:** both `scoring/plausibility.py::d_ratio_of` and
`losses/terms.py::param_prior`'s D-ratio use `largest / second-largest` diffusivity
(the two most-mobile species), never global `max/min`. For N≥3 this excludes the
single smallest D from the ratio by construction.

**Evidence:** `docs/ROBUSTNESS_MEASUREMENT.md` §4.4 — immobilising the slowest
diffuser keeps **127/127** `three_gene` systems strictly Turing (vs 81/127 for the
middle diffuser, 38/127 for the fastest). A near-immobile third node is therefore the
*mechanism*, not a defect, and a max/min ratio definition would penalise exactly the
node Tica et al.'s design intentionally makes near-zero.

**What was rejected and why:** plain global `max/min` over all N species — rejected
because it would drive the D-ratio to near-infinity (or force a floor/clamp) whenever
a near-immobile modulatory node is present, penalising a configuration the project's
own robustness baseline shows is *beneficial*, not aberrant.

**Not independently validated:** the module docstring and the task itself flag this as
`[IMPL]`, tested for the specific behaviour claimed
(`test_d_ratio_of_ignores_a_near_immobile_node`,
`test_param_prior_does_not_penalise_an_immobile_node`) but not compared empirically
against the max/min alternative on real recoveries.

**Where it lives:** `src/rngrn/scoring/plausibility.py::d_ratio_of`;
`src/rngrn/losses/terms.py::param_prior`.

### D3 — `bio_box` D-ratio upper bound of 60.0 is a reasoned proxy, not a measured bound

**Date:** 2026-07-29 (PR 6). **Status:** DECIDED, explicitly flagged as unmeasured.
**Decided by:** the implementing agent under delegated authority.

**The decision:** the plausibility box's D-ratio range is `[1.0, 60.0]`. The lower
bound (1.0) is Turing's own 1952 requirement that the inhibitor diffuse faster than
the activator. The upper bound (60.0) is **not** a measured D-ratio bound from any
source — it is derived from the same `STATE_OF_THE_SCIENCE.md` §11 paragraph's cited
*absolute* diffusivity spread across measurement regimes (~60 µm²/s FCS-local to
~1–2 µm²/s FRAP-global, roughly two orders of magnitude), reused as a ceiling on
physiologically plausible differential diffusion.

**What was rejected and why:** no direct literature citation for a D-ratio ceiling
was found; rather than leave the field uncited (as `beta` is, see below) or invent an
independent number, the two-orders-of-magnitude absolute-diffusivity spread was
repurposed as a proxy ceiling and labelled as such in the source comment.

**Where it lives:** `configs/bio_box.yaml` (`d_ratio.high: 60.0`, with the `source:`
field stating the reasoning above verbatim).

### D4 — `beta` (basal production) left UNCITED in the plausibility box

**Date:** 2026-07-29 (PR 6). **Status:** DECIDED not to invent a bound. **Decided by:**
the implementing agent, following the project's evidence-discipline rule.

**The decision:** `configs/bio_box.yaml`'s `beta` row has `low: null, high: null,
source: UNCITED`. An UNCITED row is reported (its in/out-of-box verdict is computed)
but never scored in/out-of-box and never hinged in the prior loss.

**Evidence rejected as insufficiently grounded:** `docs/STATE_OF_THE_SCIENCE.md` §10
records a `beta ~ 1e-4..1e-2` range, but that range came from an **init-distribution
search** (the low-basal init sweep, see D8 below) explicitly marked "a modelling
assumption rather than a neutral prior" and "deliberately not adopted" for training —
repurposing it as a biological-plausibility claim would misrepresent its provenance.

**Where it lives:** `configs/bio_box.yaml` (`beta:` block).

### D5 — `k_star_fft` (image-derived) is the headline validation target, not analytic `k_star`

**Date:** decided 2026-07-29, reversing the 2026-07-26 decision. **Status:** DECIDED
(current), SUPERSEDES the 2026-07-26 choice. **Decided by:** the user.

**The decision:** validation and every scoring table now head on `kstar_fft_rel_err`
(FFT-measured k\* from the image) as HEADLINE, with `kstar_rel_err` (analytic k\* from
the true J, D) as SECONDARY. This reverses the 2026-07-26 decision recorded at
`docs/STATE_OF_THE_SCIENCE.md` line 499, which had made analytic `k_star` the headline
and `k_star_fft` the secondary diagnostic.

**Evidence for the reversal:** analytic `k_star` is `argmax_k σ(k)` from the *true* J
and D — a property of the generating equations no real experiment can observe.
`k_star_fft` is a property of the image, which is what an actual inverse problem has
access to. For a pipeline whose goal involves experimentally obtained patterns, the
observable target is the defensible one (`docs/STATE_OF_THE_SCIENCE.md` §8.1).

**Circularity checked and found real but limited:** `docs/STATE_OF_THE_SCIENCE.md`
§8.2 measured `k_star_fft` (the stored attr) against `kstar_obs` (the live loss
anchor, `observables.kstar_of`) on 128 samples across 4 splits: median relative
difference 1.56%–3.67% by split, max up to 39.3%, **exact matches 0/128**, within 1%
22/128, correlation r=0.998. So validating against `k_star_fft` is not literally
circular (it is an independently computed FFT estimate) but §8.3 notes both
estimators share the same bias structure, so a model that reproduces the bias is
still rewarded to some degree.

**What was rejected and why:** keeping analytic `k_star` as headline was rejected
because it grades agreement with a latent quantity the model was never given access
to, rather than agreement with something measurable.

**Consequence for earlier numbers:** every k\* number recorded before 2026-07-29 was
reported against the analytic target and is **not directly comparable** to numbers
reported after the reversal (`docs/CODE_REALITY.md` §4, `docs/STATE_OF_THE_SCIENCE.md`
§8.1).

**Where it lives:** `src/rngrn/validate.py` (module docstring + `score_recovery`
inline comment); `src/rngrn/optim/benchmark.py` (`COLUMNS`, `DEGRADATION_COLUMNS`,
both docstrings, FFT column ordered first); `src/rngrn/data/gate.py` (`AnswerKey`,
`from_registry` docstrings). PR 2.

### D6 — the domain-size leak: vary L with random periods-per-box, decorrelating L from k\*

**Date:** 2026-07-29 (unit 11, "the L-generalisation task", executed as unit 12/PR 12,
`three_gene_ldata`). **Status:** DECIDED — settles open decision #2 in
`docs/CODE_REALITY.md` §11. **Decided by:** the user's requirement ("the model must
generalise across domain sizes") forecloses option (a); settled by unit 11/12.

**The decision:** every prior generator set `L = clip(6·2π/k*, 18, 220)`, holding
exactly 6 wavelengths in every box, so `L` was an algebraic function of `k*_true`. The
new policy is `L = p · (2π/k*)` with `p` a free integer drawn per-system from
`U{3..14}`; candidates admitting no feasible `p` in `[18, 220]` are rejected rather
than clipped (clipping recreates the same collapse at the clip bound). Two new
datasets were generated under this policy: `three_gene_qvar` (34 systems, one L each)
and `three_gene_multiL` (23 systems × 4 domain sizes each, same kinetics/`k_star`/
init seed per group, differing only in L).

**Evidence — the leak, measured:** on the original 127 `three_gene_{train,val,test}`
samples, the image-blind predictor `k_hat = 6·2π/L` scores **0.0% median relative
error** (100% of samples within 1%) — periods-per-box is exactly 6.000 for every one.
On `three_gene_qvar` (34 systems, `p ~ U{3..14}`) the same blind predictor scores
**45.5% median error**. On `three_gene_multiL` (92 samples, `p` in {4,7,10,13}) it
scores **45.0%**. An **oracle** best-fixed-`p` (chosen after seeing the answers, the
single strongest possible fixed-period predictor) still costs 28.6% on `qvar` and
29.2% on `multiL` — no fixed periods-per-box explains the new data.

**What was rejected and why** (all three options were measured and put to the user,
per `docs/STATE_OF_THE_SCIENCE.md` §7.1):
- **(a) fixed L** — matches the user's originally literal statement ("I wanted all of
  my patterns to have the same domain size") but makes L carry zero information and
  forecloses any domain-size-generalisation claim, and caps k\* to a 5.3× span at
  grid=96 versus the generators' 0.08–3.0 screening band. Rejected once the user
  clarified the model must generalise across domain sizes — a requirement fixed-L
  makes untestable.
- **(c) fixed L with grid raised to 192/256** — widens the band but at 4–7× simulation
  cost; not chosen (not depended on to close the leak).
- Chose **(b)**, vary L with periods-per-box randomised.

**Old data not deleted:** the original 127 `three_gene_{train,val,test}` samples
remain the comparison baseline (`p=6` exactly, by construction) precisely because they
now demonstrate what the leak looked like.

**A related permanent limitation recorded alongside this decision:** the staging
generator seeded each screen with `abs(hash(topo))`. Python salts string hashing per
process, so the recorded seed **never reproduced the screen** — confirmed directly
(two runs of the smoke mode at the same nominal seed screened a different number of
candidates). Fixed going forward with a SHA-256-derived seed (`scripts/gen_tg3.py`),
and both new datasets were regenerated after the fix — but **the existing 127
`three_gene` samples cannot be regenerated from their recorded seed.** This is a
permanent limitation of the legacy data, not something this decision (or any later
one) can repair; it does not affect the *validity* of those 127 samples, only the
ability to re-derive them byte-for-byte from the recorded provenance.

**Also fixed alongside this decision:** the generator (`data/staging/tg3/generator.py`)
lived only in a gitignored `data/staging/` tree and was never in version control.
Tracked now as `scripts/gen_tg3.py`.

**What is still OPEN, not settled by this decision:** whether `three_gene_multiL`
should be split by `system_id` so no system straddles a train/val/test split — both
new datasets ship with `splits: {}` (unsplit), left to the consumer. See D13 below.

**Where it lives:** `scripts/gen_tg3.py`; `docs/DATASETS_L.md`; `data/datasets/
three_gene_qvar/`, `data/datasets/three_gene_multiL/`; `docs/CODE_REALITY.md` §11
item 2 (struck through / marked SETTLED).

### D7 — `loss.weights.resid` defaulted to 0.0 — settled OFF

**Date:** decided in exp06 (pre-phase-A), promoted into the library defaults
2026-07-29 (PR 11, unit 1). **Status:** DECIDED. **Decided by:** exp06's measured
sweep; promoted into the library by the unit-1 agent under delegated authority
(a mechanical promotion of an already-decided value, not a new decision).

**The decision:** the PDE-residual loss term (`resid`) defaults to weight **0.0** in
`losses/total.py::compute_terms` / `LossConfig`. The term is kept in the code (not
removed) for future use, but is off by default and — as an optimisation — is *omitted
from computation entirely*, not computed and multiplied by zero, when the active
strategy's weights are static and its base weight is 0 (`compute_terms
(compute_resid=False)`, measured at 45% of a forward+backward step: 9.39ms with vs
5.15ms without, 96×96, N=3, 40 reps). `parts["resid_skipped"] = True` is set so no
downstream reader can mistake "not computed" for "computed as zero".

**Evidence:** exp06 swept pixel batch size {64, 128, 512} × residual weight {1, 3, 10}
— **9 cells × 8 seeds** — and **all nine cells collapsed to 1/8 Turing seeds**, with
best median k\* error 11.8% against 0.4% with the residual off.

**What was rejected and why:** any nonzero residual weight in the {1, 3, 10} × {64,
128, 512} grid tried — rejected because every cell measurably collapsed Turing
recovery relative to the residual-off control.

**Known consequence, recorded as an open problem (not a bug):** hidden-channel
(m < N) recovery has **no known-good objective** at `resid = 0`, because the latent
fields enter the objective through `stationarity_residual` and nothing else — their
gradient is exactly 0.0 at that weight (measured, N=3, m=2). `recover()` now refuses
such runs outright rather than silently returning the random init as a "recovered"
latent field, per PR 11.

**Where it lives:** `src/rngrn/config.py::LossConfig` (`weights.resid` default 0.0);
`src/rngrn/losses/total.py::compute_terms`; `TUNING.md` Stage 2, "`loss.weights.resid
= 0.0` is SETTLED, not untuned".

### D8 — split hinges + frame-scale anchor promoted into the library, taking library Turing fraction from 0% to 36.8%

**Date:** 2026-07-29 (PR 11, unit 1). **Status:** DECIDED. **Decided by:** the
implementing agent under delegated authority, promoting an already-validated
experiment-script objective (exp05) into the library — a mechanical promotion of
already-tested arithmetic, verified bit-identical before landing.

**The decision:** three terms proven in `scripts/exp05_pixel_minibatch.py::fit`
replace the library `losses/total.py`'s prior objective (which had never itself
produced a measured Turing recovery):
- `terms.turing_hinges_split` — the two Turing conditions evaluated on disjoint
  k-support (`i_min = max(1, int(0.1·len(kgrid)))`) — replaces the single-direction
  hinge, now kept only as the documented control arm.
- `terms.frame_scale_anchor` — `mean((log(frame.mean()) − log(x*))²)`, firewall-legal
  since `frame.mean()` is an image observable.
- `weighting.staging_factor` / `DataFirstStaging` — the exp05 schedule (off for the
  first 25% of steps, ramped to 1.0 over the next 25%), as a composable wrapper.

**Evidence:**
- **Bit-identical promotion, not reimplementation:** `turing_hinges_split` vs
  `exp02::turing_hinges_split` vs `exp05::split_hinges`, and `frame_scale_anchor` vs
  the exp02/exp05 expression, over 20 random-init seeds — **max |difference| =
  0.000e+00**.
- **The library-vs-experiment gap this closes:** the pre-promotion library objective
  scored **0% Turing** on the reference used at that time (`docs/CODE_REALITY.md`
  §1); `scripts/exp05_pixel_minibatch.py`, running the same terms outside the
  library, reaches **36.8%** — reconfirmed post-promotion at **38/40 converged, 14
  Turing, turing_frac 0.3684** over 40 seeds × 400 steps (per the phase-B unit brief's
  "what phase A established").
- End-to-end regression test: at the same config/seeds/budget, the pre-promotion
  objective fails all 4 restarts' steady-state solve; the promoted objective
  completes and recovers Turing (`recovered_turing = True`, kstar_rel_err 0.0506).

**What was rejected and why:** two design choices were identified but **deliberately
left unmade**, because settling either would bias what "36.8%" means and is a science
decision, not a mechanical one:
- **`detach_xstar`** — exp05 detaches x\* from the dispersion-term gradient; the
  library differentiates through it via `steady_state_diff`. Default kept `False`
  (library behaviour), but the 36.8% measurement was made with `True` — the two
  settings have never been A/B'd.
- **The hinge k-floor being grid-relative rather than k\*-relative** — measured on
  `sample_0000` (k\*_obs = 0.4320): the promoted library grid puts `k_min` at
  **0.822×k\*_obs** vs exp05's **0.698×k\*_obs**. Defining the floor relative to
  `kstar_obs` would change what the term means; not changed here.

**A defect found and fixed during this promotion (not itself the decision, but load-
bearing for it):** `recover()`'s final post-training scoring pass was previously
unguarded against `SteadyStateError` — one restart landing on a bad steady state
aborted the *entire* recovery and discarded every other restart's result. Fixed to log
`failed_at="final_eval"` and skip that restart only. This fired on restart 0 of the
verifying e2e run; without the fix, that run would have produced nothing.

**Where it lives:** `src/rngrn/losses/terms.py` (`turing_hinges_split`,
`frame_scale_anchor`); `src/rngrn/losses/weighting.py` (`staging_factor`,
`DataFirstStaging`); `src/rngrn/config.py::LossConfig` (`split_hinges`,
`hinge_k_min_frac`, `staging_keys`, `staging_off_frac`, `staging_ramp_frac`,
`detach_xstar`); `docs/CODE_REALITY.md` §11 item 6 (settles the open decision);
`TUNING.md` "Promoted from the experiments (unit 1)".

### D9 — low-basal init: implemented, available via `model.init="low_basal"`, but DEFAULT OFF

**Date:** 2026-07-29 (PR 9, unit 9). **Status:** DECIDED — default stays off; the
knob exists but is not wired into the CLI/`train.fit()` path. **Decided by:** per the
task brief's instruction to leave the default off; the implementing agent measured
the training-time consequence and reported it rather than reversing the default.

**The decision:** `RNGRN(init="default"|"low_basal")` — a firewall-safe alternate raw-
parameter init (`beta` 1e-4..1e-2, `s` 1e-2..10^-0.3, `alpha` 10^0.3..10^1.5, `delta`
0.1..10^0.3, gate logit ~N(0, 2.5), D-ratio 10^0.9..10^2.4), ported unmodified from
`scripts/exp03_turing_first.py::low_basal_init`. `ModelConfig.init` defaults to
`"default"` everywhere; `train.py`'s `cfg.model.init` is **not** threaded into
`recover()` (out of the unit's file scope), so today the field round-trips into
`frozen_config.yaml` but has no effect via the CLI path.

**Evidence — at init, low_basal is dramatically more Turing-unstable:** 400 seeds per
setting (Newton steady state + Jacobian-sign check only, no fit) —
**0/400 (0%)** Turing-unstable-at-init for `default` vs **206/255 converged (80.8%)**
for `low_basal`, consistent with the ~82% figure already documented in
`docs/STATE_OF_THE_SCIENCE.md` §10 for this same beta upper bound.

**Evidence — but it fails under actual training, which is why the default stays off:**
direct `recover(ri, init="low_basal", adam_steps=200, lbfgs_steps=0)` on
`three_gene_val/sample_0000` **failed all 40/40 restarts** (8 restarts × 5 seed
offsets) to converge to a valid steady state during training, where the matched
`init="default"` call (same restart/step budget) succeeded (loss=0.517). The raw
init-only measurement above shows low_basal's Newton solve *itself* converges less
often even before any objective is added (255/400 vs 400/400) — the training-time
failure compounds that.

**What was rejected and why:** flipping the default to `low_basal` was rejected/not
attempted, because the training-time measurement above shows it changes *which*
solutions recovery finds and currently fails outright on the one real dataset tested
— exactly the "biases what results mean" criterion that puts a decision outside
mechanical scope, and `docs/STATE_OF_THE_SCIENCE.md` §10 already recorded it as
"deliberately not adopted" prior to this unit.

**Where it lives:** `src/rngrn/model.py` (`RNGRN.__init__`, `_low_basal_raw_params`);
`src/rngrn/config.py::ModelConfig.init`; `src/rngrn/recover.py` (`init` kwarg, unwired
from `train.py`); `docs/CODE_REALITY.md` §11 item 5; `docs/STATE_OF_THE_SCIENCE.md`
§10.

### D10 — `DEFAULT_SIGN_ZERO_RTOL = 0.05` and `topology_consistency := modal_fraction`, both flagged UNCALIBRATED

**Date:** 2026-07-29 (PR 7, unit 3). **Status:** OPEN / UNCALIBRATED — implemented as
the working definition, explicitly not validated against the alternative.
**Decided by:** the implementing agent under delegated authority, as a metric-
definition choice flagged rather than made silently.

**The decisions (two, bundled in the same scorer):**
1. **Near-zero entry dead zone:** a recovered Jacobian entry is treated as
   structurally zero if it is below 5% of that matrix's own max |J| —
   `DEFAULT_SIGN_ZERO_RTOL = 0.05`. Same magnitude as (and for the same reason as)
   `scoring.overparam.DEFAULT_COUPLING_THRESHOLD` (also 0.05, also uncalibrated):
   J's overall scale is not pinned by the objective, so only a per-matrix relative
   cut is meaningful. Deliberately coarser than the 1e-9 "zero to floating-point
   round-off" cut used elsewhere — the two thresholds answer different questions.
2. **`topology_consistency` is defined as the modal fraction** (the fraction of seeds
   producing the single most common exact sign structure), not mean pairwise
   agreement. Both are computed and returned (`mean_agreement` is also in the
   report), but only `modal_fraction` is the headline scalar.

**Evidence for the choice, as reasoned (not measured against real data):**
`modal_fraction` is the strictest, most literal reading of the user's stated primary
success metric — "does the model consistently learn the same topology across seeds,
for one target". `mean_agreement` rewards partial pairwise closeness even when no two
seeds are identical, which is a softer question than the one asked.

**What was rejected and why:** `mean_agreement` as the headline — rejected as
answering a different (softer) question than the user's literal framing, though it is
still computed and reported alongside the headline so it is not lost.

**Explicitly UNCALIBRATED — not settled by this entry:** neither the 0.05 rtol nor
the modal-vs-mean choice has been validated against real cross-seed recovery data;
PR 7 could not obtain a second successful CLI recovery on the same target within a
reasonable seed budget (CLI hit `SteadyStateError` on ~30/31 seeds tried), so the
cross-run aggregation is verified only on synthetic matrices with known sign
patterns, not on live data.

**Where it lives:** `src/rngrn/scoring/reproducibility.py` (module docstring, "THE
topology_consistency DESIGN CHOICE" section); `src/rngrn/validate.py::score_recovery`.

### D11 — `gradnorm`/`ntk` loss-weighting strategies now RAISE rather than silently running fixed weights

**Date:** 2026-07-29 (PR 5, unit 13). **Status:** DECIDED. **Decided by:** the
implementing agent, correcting a fail-loud violation per the project's standing rule.

**The decision:** `GradNormWeighting.combine` and `NTKWeighting.combine`
(`losses/weighting.py`) were stubs that silently ran with fixed weights whenever
selected — a run configured for gradient-magnitude balancing or NTK-based weighting
would appear to execute normally while doing neither. Both now `raise
NotImplementedError` at construction. A working `ratio` strategy (Matas-Gil & Endres'
actual published technique — `weight_k = loss_data / loss_k`, recomputed every
`update_every` steps) was implemented and adopted in their place as the strategy this
project actually ships.

**Evidence this was a real defect, not a hypothetical:** a grep-verifiable code state
(the PR body records finding the stubs by reading `losses/weighting.py` directly) —
any prior config setting `loss.strategy: gradnorm` or `loss.strategy: ntk` would have
silently produced fixed-weight results while claiming otherwise.

**What was rejected and why:** leaving the stubs silently running fixed weights was
rejected outright as a fail-loud violation (project convention #2). Implementing
GradNorm/NTK properly was out of scope for this unit; raising is the correct interim
state until someone implements them for real.

**Where it lives:** `src/rngrn/losses/weighting.py` (`GradNormWeighting.combine`,
`NTKWeighting.combine`, both raising; `ratio` strategy new); `src/rngrn/config.py`
(`LossConfig.ratio_update_every`, default 50); `tests/test_weighting.py`.

### D12 — Domain-size design settled by decorrelating L from k\* (see D6)

Cross-reference: this is the same decision as D6 above (`docs/CODE_REALITY.md` open
decision #2). Recorded once, at D6, to avoid a duplicate entry with different
evidence excerpts drifting apart.

### D13 — Both new datasets left unsplit; whether `three_gene_multiL` should split by `system_id` — OPEN

**Date raised:** 2026-07-29 (PR 12, unit 11). **Status:** OPEN. **Decided by:** no one
yet — explicitly left to the consumer.

**The decision not yet made:** `three_gene_qvar` and `three_gene_multiL` both ship
with `splits: {}` (unsplit), matching the existing registered sets' convention. For
`three_gene_multiL` specifically — 23 systems × 4 domain sizes each, sharing identical
kinetics/`k_star`/init seed within a group — whether a train/val/test split must keep
every system's 4 replicates on the same side (so no system straddles a split) is
unresolved.

**Why it matters:** if a system's replicates are allowed to straddle a split, a model
could see one L of a system in training and be tested on another L of the *same*
system — which would inflate any cross-L generalisation claim, since the model would
have partial information about that exact system's kinetics rather than facing a
genuinely unseen one.

**What is NOT decided here:** this register does not pick an answer. Flagged as an
explicit `OPEN` entry per the task's instruction not to close a decision that has not
actually been made.

**Where it lives:** `data/datasets/three_gene_multiL/manifest.json` (`splits: {}`);
`docs/DATASETS_L.md`.

### D14 — Generator seeded from `abs(hash(topo))`: fixed going forward, permanent limitation for legacy data

Cross-reference: recorded in full under D6 above (same PR, same decision context).
Restated here as its own line because the brief calls it out separately: the existing
127 `three_gene` samples **cannot be regenerated from their recorded seed**, because
Python salts string hashing per process. This is a **permanent limitation of the
legacy data** — not something any later decision can retroactively fix — recorded so
a future reader does not attempt to re-derive those 127 samples byte-for-byte from
their nominal seed and conclude the pipeline is non-deterministic in general; the
non-determinism was specific to `abs(hash(str))` and is fixed in
`scripts/gen_tg3.py` (SHA-256-derived seeding) for every dataset generated after it.

---

## Part 3 — Reconciling `docs/CODE_REALITY.md` §11 ("open decisions currently blocking progress")

`docs/CODE_REALITY.md` §11 lists six open decisions as of the phase-A merge. Reconciled
against Part 2 above:

1. **"Which reading of 'more robust'"** (`GOAL_tica_equivalent.md` §2.1, four
   incompatible readings A–D). **Still OPEN.** No entry in this register settles it;
   `GOAL_tica_equivalent.md` §2.1 itself states "The project has not chosen between
   them" and "Under the project's autonomy rule this is a science decision, not a
   mechanical one." Nothing found in the 13 merged PRs picks one.

2. **"Domain-size design for regenerated data"** — **SETTLED.** By unit 11/12
   (D6 above): option (b), vary L with random periods-per-box. Evidence: the
   image-blind predictor going from 0.0% median error on the old 127 samples to
   45.5%/45.0% on the two new datasets. `docs/CODE_REALITY.md` §11 item 2 already
   marks this struck through as settled; this register's D6 is the fuller record.

3. **"The morphology pass condition"** — stripes classifies at 33.3% on held-out data
   (3 samples), and the options (balanced accuracy excluding stripes / continuous
   `morphology_distance` / generate more stripes first) are unchanged. **Still OPEN.**
   Nothing in the 13 merged PRs picks a pass condition; PR 13 (unit 7) made the
   morphology *rollout* affordable and wired it into every run
   (`morphology_scored: "compared"` now reachable), which makes the metric
   *measurable* but does not decide what counts as a pass. PR 13 separately raised its
   own OPEN METRIC DECISION (whether an unpatterned recovery should record
   `morphology_match=False` or stay `"target_only"`) — related but not identical to
   this item; also unresolved.

4. **"D-ratio prior centre — ~7.5 vs ~100"** — **DECIDED for the bio_box prior; the
   underlying tension is explicitly left open.** See D1 above: the user decided 7.5
   for `configs/bio_box.yaml`, reversing an earlier in-session ~100 choice. But
   `docs/STATE_OF_THE_SCIENCE.md` §11 still frames this as an "open design choice"
   at the document level, because 7.5 is a decision about the *plausibility box*
   specifically, not a resolution of whether training itself should ever target ~100.
   Recorded as decided-with-a-caveat rather than fully closed, per the task's
   instruction not to overclose a tension the source material calls deliberate.

5. **"Whether to adopt the low-basal init"** — **Still largely OPEN, but now with
   training-time evidence against adopting it as the default.** See D9: PR 9 measured
   that low_basal fails 40/40 restarts under actual training on the one dataset
   tested, versus `default` succeeding under the same budget. The *default* stays
   `"default"` and this register treats that as effectively answered by the evidence
   (do not flip the default) even though no one has stated "we will never adopt this."
   The knob exists, is unwired from the CLI path, and adopting it would require first
   explaining or fixing the training-time failure mode.

6. **"Whether `split_hinges` and the frame-scale anchor are promoted into the
   library"** — **SETTLED.** By unit 1/PR 11 (D8 above): promoted, bit-identical to
   the validated originals (max diff 0.000e+00 over 20 seeds), library Turing fraction
   confirmed at 38/40 converged, 14 Turing, turing_frac 0.3684 over 40 seeds × 400
   steps.

**Summary:** of the six, **items 2 and 6 are settled**; **item 4 is decided for its
narrow scope (the bio_box prior) but its wider tension is deliberately still open**;
**items 1, 3, and 5 remain open** — 5 now has training-time evidence pointing away
from adopting the alternative, but no one has made that a formal decision.

---

## Sources consulted

- `gh pr view` for PRs 1–13 (`ben1ahrens/rngrn-pipeline`, all merged 2026-07-29).
- `docs/CODE_REALITY.md` (§4, §11, and the evidence-integrity items cross-checked
  against `.githooks/pre-push` and `src/rngrn/cli.py` directly).
- `docs/STATE_OF_THE_SCIENCE.md` §1, §7, §8, §10, §11.
- `docs/GOAL_tica_equivalent.md` §2.1, §2.2.
- `docs/ROBUSTNESS_MEASUREMENT.md` §4.
- `TUNING.md` (Stage 2 and the "Promoted from the experiments" / "Morphology rollout"
  sections).
- `configs/bio_box.yaml` (per-parameter `source:` fields, read directly).
- `src/rngrn/losses/weighting.py`, `src/rngrn/cli.py`, `.githooks/pre-push` (read
  directly to confirm the two evidence-integrity fixes are actually in the code, not
  only claimed in a PR body).
