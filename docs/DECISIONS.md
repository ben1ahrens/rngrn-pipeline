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

### D-EVID-3 — `rngrn evaluate` re-ran every archived model at the WRONG domain size

**Date found / fixed:** 2026-07-30 (cross-domain-size evaluation unit). **Status:**
SUPERSEDED (defect; now fixed).

`cli.py::cmd_evaluate` called `simulate(model, L=cfg.data.L, ...)`. `cfg.data.L` is a
**generator** parameter that file-backed configs deliberately do not set — `config.py`
`DataConfig.L` says so in its own comment ("Do not add `L:` to a file-backed config
expecting it to apply"), because each registry sample carries its own `L`. So for every
registry config `cmd_evaluate` used the `base.yaml` default **100.0** while the model had
been recovered on the sample's real domain size. Re-running an archived model therefore
reproduced nothing: k\*, the pattern and the morphology were all computed on a box of the
wrong size.

**Measured, on the one archived run in this worktree carrying both files**
(`m3_registry_20260729_221743`, `three_gene_qvar/sample_0000`):
`config/frozen_config.yaml` → `data.L = 100.0`; `checkpoints/model.pt` → `L =
78.01357861389891`. A **28.2 % error in the length scale**, silent.

**Fix:** `cmd_evaluate` now defaults `L` to the **checkpoint's stored `L`**
(`io.save_checkpoint` has recorded it since unit 12) and warns at `RuntimeWarning` when
`cfg.data.L` disagrees. It also routes the model through
`eval.lgen_eval.physical_model_from_checkpoint` first, so a non-dimensional checkpoint's
`theta_D` is converted before anything is integrated (D16).

**Consequence for earlier results:** no number in the run index came from `cmd_evaluate` —
the `rollout_*` columns in `runs.jsonl` are written by `train._morphology_rollout`, which
has always used the target sample's own `ri.L` and is unaffected. The damage is limited to
any `rngrn evaluate` output quoted by hand from a terminal; such a figure is not
reproducible and must be re-measured.

**Where it lives:** `src/rngrn/cli.py` (`cmd_evaluate`).

### D-EVID-4 — `train.seed` does not change what recovery does: every "seed replicate" is the SAME draw

**Date found:** 2026-07-30 (unit P1, while validating the new `seed` / `model_seed` run-index
columns). **Status:** **OPEN — FOUND, NOT FIXED.** Reported rather than repaired because
changing the seed plumbing would change what every previously recorded "seed replicate"
number means, and that is the owner's call, not the implementing agent's.

**What was found.** `train.fit` calls
`R.recover(..., seed=cfg.train.seed, model_seed=cfg.model.seed)`. Inside `recover`,
`model_seed = seed if model_seed is None else model_seed` — and `cfg.model.seed` is `0`, never
None, so **`model_seed` is 0 on every run regardless of `train.seed`**. Restart inits come
from `_restart_seed(model_seed, r)` and a per-restart `torch.Generator`, never from the global
RNG that `seed_everything(cfg.train.seed)` seeds. `train.seed` therefore reaches only the
post-training rollout's noise field (`eval.rollout.simulate(seed=...)`), not the optimisation.

**The measurement.** A 12-run sweep over `train.adam_steps` ∈ {60, 120, 240} ×
`model.form` ∈ {competitive, nc1} × `train.seed` ∈ {0, 1} on
`three_gene_val/sample_0000`: the six pairs differing **only** in `train.seed` have
**bit-identical** loss (e.g. competitive/60: 0.223712 and 0.223712; nc1/240: 0.073764 and
0.073764) and identical `kstar_fft_rel_err`. The defect was invisible before unit P1 because
neither `seed` nor `model_seed` was on the run row.

**What it plausibly affects, stated as a question and not as a conclusion.**
`optim/target_report.py` builds its K per-seed jobs with `apply_overrides(..., ["train.seed=<seed>"])`
and nothing else, so on the reading above a K-seed target report runs K identical recoveries
and its cross-seed reproducibility statistics would be measuring nothing. **This has not been
verified end to end here** — it follows from the code path and from the sweep evidence above,
but no target report was re-run to confirm it, and any existing reproducibility number should
be re-checked against `model_seed` before it is quoted.

**Where it lives:** `src/rngrn/train.py` (the `model_seed=cfg.model.seed` argument);
`src/rngrn/recover.py` (`_restart_seed`, the `model_seed` default);
`src/rngrn/optim/target_report.py` (`run_target_report`'s per-seed overrides). The new
`seed` / `model_seed` columns on the run row are what make it visible.

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

### D15 — cross-L rollouts hold **dx** fixed (n scales with L), with a MEASURED floor of 6 pixels per wavelength

**Date:** 2026-07-30. **Status:** DECIDED. **Decided by:** the implementing agent, as an
explicitly delegated technical choice.

**The decision:** `eval/lgen_eval.evaluate_across_L` simulates the recovered model at each
domain size on a grid `n(L) = nearest even int to n_grid * L / L_train`, i.e. it holds the
grid spacing `dx = L/n` fixed rather than the grid size. Every row additionally records
`pixels_per_wavelength`, and a row that produces a pattern below **6.0** px/wavelength
**raises** instead of reporting the number.

**Why this way round:** at fixed `n`, a 4× box has a quarter of the pixels per wavelength,
so the k\* estimate degrades with L — and the degradation would land in `kstar_phys_cv`,
the headline invariance statistic, looking exactly like physics. Holding `dx` fixed makes
`pixels_per_wavelength` constant across L to within the even-rounding, so the comparison
across L is a comparison of the *pattern* and not of the discretisation.

**The floor is measured, not asserted.** Scanning `n` at fixed L on the Turing fixture of
`tests/test_rollout.py` (L = 60, q ≈ 6.2, `etdrk4_rfft`, seed 0), k\* referenced to the
n = 192 run:

| n | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|
| px/wavelength | 2.76 | 3.94 | 5.12 | 7.64 | 10.10 | 14.98 | 20.32 | 30.79 |
| \|Δk\*\|/k\* | 6.9 % | 2.4 % | 0.3 % | 0.8 % | 1.6 % | 2.8 % | 1.0 % | — |
| morphology class | spots | laby | laby | laby | laby | laby | laby | laby |

k\* survives to 3.94 px/wavelength within ~3 %; at 2.76 it moves 6.9 % **and** the class
call flips. 6.0 sits above that breakdown with margin and coincides with the coarsest
resolution the generators ever produced (`three_gene_qvar`/`multiL` draw up to p = 14
periods on a 96 grid → 96/14 = 6.86 px/wavelength), so the floor refuses exactly the
resolutions the data never contained. The ±3 % grid-to-grid spread in that table is also
the **noise floor of `kstar_phys_cv`**: no CV below it is meaningful.

**What was rejected and why:** (a) holding `n` fixed and merely reporting
`pixels_per_wavelength` — allowed by the brief, rejected because it puts a resolution
artefact into the headline statistic and relies on the reader to subtract it; (b) *clamping*
`n` at an `n_max` — rejected as a silent change to the one quantity the rule exists to hold
fixed, so `grid_for_L` raises at both ends instead; (c) tuning the floor down to the
measured 4 px/wavelength — rejected as over-fitting one fixture.

**A second resolution limit that this rule does NOT fix, recorded so it is not mistaken for
one:** `observables.raps` bins on the fundamental `dk = 2π/L`, so k\* is quantised at a
*relative* width of `1/q`. A small box holds few periods and its k\* is coarsely measured
however fine the grid (at q = 3, the bin is 31 % wide). Every row therefore records
`kstar_bin_width` and `kstar_bin_width_rel`, and `kstar_phys_cv` must be read next to the
largest of them. No q threshold was invented on top of the px/wavelength floor.

**Where it lives:** `src/rngrn/eval/lgen_eval.py` (`grid_for_L`, `PPW_FLOOR`,
`PPW_FLOOR_PROVENANCE`); `tests/test_lgen_eval.py`.

### D16 — the cross-L statistics: population CV of k\*, an origin-fixed q-vs-L slope, and NO pass threshold

**Date:** 2026-07-30. **Status:** DECIDED (the statistics); the **thresholds remain OPEN by
design**. **Decided by:** the implementing agent for the definitions; the thresholds are
deliberately left to `docs/PREREGISTRATION.md` and the caller.

**The decision, and the theory it tests.** k\* is the argmax of `σ(k) = eig(J − k²D)`;
neither J nor D contains the domain size, so for one recovered network simulated at several
L, Turing theory predicts an invariant **physical** k\*, a periods-per-box count
`q(L) = L·k*/2π` **linear in L**, and a preserved morphology class. The failure mode being
excluded is the opposite: q pinned to the box while k\* moves as 1/L, which is a model that
memorised a box. `summarise_across_L` reports:

- **`kstar_phys_cv`** — *population* (ddof = 0) SD of `kstar_phys` over patterned rows,
  divided by the mean. Population, not sample, SD so the number does not inflate at the 3–4
  L values this is ever used over.
- **`periods_slope_rel_err`** — q fitted against L **through the origin**
  (`slope = Σ L·q / Σ L²`), compared to the theory slope `mean(k*)/2π`. Because
  `q_i ≡ L_i k_i/2π` identically, the fit is the L²-weighted mean of `k_i/2π` and the
  prediction is the unweighted one: they agree exactly under invariance and diverge when k\*
  trends with L. It is therefore a differently-weighted **second view** of the same
  invariance, not an independent measurement, and both slopes are reported so that is
  visible.
- **`morphology_class_preserved`** — fraction of patterned non-reference rows matching the
  class at L_train, always alongside `n_class_compared` (rows that did not pattern have no
  class and are excluded rather than scored as mismatches; excluding them flatters the
  fraction).

**Separation, measured — the metric can fail.** Positive control (the Turing fixture at
L/L_train = 0.5, 1, 2, 4): `kstar_phys_cv` **0.021**, `periods_slope_rel_err` **0.017**.
Negative control (a synthetic frame carrying a fixed 6.5 periods per box at every L, pushed
through the same row and summary arithmetic): **0.715** and **0.624** — 33× and 37× larger.

**Morphology is explicitly the weakest of the three.** On the positive control the two
*intensity* features are invariant (skew 0.43–0.51, kurtosis −0.74 to −0.81 across an 8×
span of L) while the two *angular* features drift monotonically with q (ang_conc
0.190 → 0.074), because they are computed on a ring of radius q in pixel-frequency units —
a property of the measurement, not of the pattern. The class consequently reads
labyrinth → spots and `morphology_class_preserved` = 1/3, on borderline margins (0.17–0.62),
against a centroid bank fitted at q ≈ 6. And it fails the other way too: the box-pinned
negative control preserves its class **perfectly (1.0)** while k\* moves as 1/L. Class
agreement alone is evidence of nothing; the raw `morph_vector` is recorded on every row so a
reader can see which features moved.

**NO PASS/FAIL LIVES IN THE MODULE.** `summarise_across_L` returns a `verdict_components`
dict of raw quantities and no boolean — `tests/test_lgen_eval.py` asserts that, so a
threshold cannot be smuggled in later without the test noticing. This follows
`scoring/lgen.py`'s decision (3) ("NO PASS THRESHOLD is defined") rather than re-opening it.

**The nondim trap, closed.** `recover()` reports `params["D"]` physically on both paths, but
the **checkpoint** keeps `theta_D` in the units the objective ran in — `D/L_train²` when
`nondim=True` (`recover.py` lines 518–528 convert the *reported* params only). Simulating a
reloaded nondim checkpoint without converting integrates the wrong diffusivity and yields a
k\* wrong by a factor of L_train, silently. `physical_model_from_checkpoint` rewrites
`theta_D` on a deep copy, verifies `model.D` afterwards, and **raises** on a checkpoint
lacking the `nondim`/`L` extras: the two paths are indistinguishable after the fact, so an
older checkpoint is refused rather than assumed dimensional.

**Where it lives:** `src/rngrn/eval/lgen_eval.py` (`evaluate_across_L`,
`summarise_across_L`, `physical_model_from_checkpoint`); `src/rngrn/cli.py`
(`--eval-L` / `--l-factors`); `tests/test_lgen_eval.py`.

---

### D-PLOT-1 — the plottable ARRAYS are persisted per run, at float32 for fields and float64 for k axes, and are TRACKED in git

**Date:** 2026-07-30. **Status:** DECIDED. **Decided by:** the implementing agent, under the
owner's requirement "all data need to be stored in a way that I can make plots on this stage
for my research".

**The problem.** A run recorded scalars (`runs.jsonl`, `target_reports.jsonl`), the recovered
parameters and Jacobian as text, and a checkpoint. That audits a number and re-runs a model.
It cannot draw a figure: **you cannot plot a pattern, a dispersion relation or a spectrum
from a scalar**, and every array was discarded when the process exited. The canonical Turing
figure — recovered σ(k) overlaid on the answer key's σ(k) — was impossible to draw at all.

**The decision.** `train.fit` writes one compressed `<run_dir>/arrays/plot_arrays.npz`
(`solver.save_plot_arrays`, default **ON**) carrying `target_frame`, `model_field` (only when
the rollout patterned), `dispersion_k` + `dispersion_sigma` + `dispersion_sigma_true` on ONE
shared k grid, `raps_k` + `raps_target` + `raps_model`, the two 4-D morphology vectors with
their `morph_feature_order`, and a self-describing JSON `meta` (run_id, git sha, dataset_id,
sample_key, form, L, and every k\*).

**dtype, split deliberately.** *Fields and spectra* → **float32**: they are plotted, never
differentiated, and float32's ~7 significant digits are far beyond what a colour map or a
log-y axis resolves; it halves the file. *k axes and σ(k)* → **float64**: these are read as
numbers (k\* is their argmax, and σ near onset is ~1e−3 against a J of order 1), so they keep
the precision they were computed in.

**MEASURED size, and why it is tracked.** Per-run npz, real runs on `three_gene_multiL`
(96×96, N=3): **110 KB** for a short serial run with no history and no model field;
**118 KB** batched, 8 restarts, 60 steps; and **645 KB** for the full phase-C shape — the
tracked `m3_registry_20260730_013119` (64 restarts, 400 steps, `history_every=10`, both fields
present, `plot_arrays_bytes = 644886`). Of its 806 KB uncompressed content the training
trajectory is **570 KB** and the two fields **221 KB**; float32 mantissas barely compress, so
this is not an estimate that will improve. Projected for a phase-C wave of ~96 runs (8 seeds ×
6 targets × 2 forms): **≈62 MB**, plus **0.82 MB per cross-L evaluation** (5 L × 3 × 128 × 128
float32, measured on the tracked `m3_registry_20260730_005701` evaluation).

That is tens of MB, not GB, and no single run is multi-MB — so `.gitignore` now **tracks**
`experiments/**/arrays/*.npz` (it previously ignored `experiments/**/*.npz` wholesale, which
cost nothing only because nothing wrote one). Two levers exist if a wave needs to be cheaper,
both explicit rather than silent: `solver.save_plot_arrays=false` (no npz at all) and
`train.history_every` (the trajectory is 70% of the file; stride 20 halves it). The
per-run byte count goes on the run row as `plot_arrays_bytes`, so the cost stays auditable.

**Figures are NOT tracked** (`experiments/**/figures/`, `*.png` stay ignored): they are
redrawn from the arrays in seconds by `scripts/make_figures.py`. The arrays are the artefact.

**The one guard that matters, carried over.** A rollout that did not pattern writes **no**
`model_field` key, and `meta.rollout_status` says why. The project already recorded one false
`morphology_match` from classifying decayed float noise (see `train._morphology_rollout`);
saving that noise as a field would re-open the same hole from the plotting side.
`tests/test_plot_arrays.py::test_unpatterned_rollout_gets_no_model_field` pins it.

**Firewall.** `plotdata.py` reads the answer key's (J, D) — passed in as plain numpy by
`train.fit` — so it is SCORING side. It imports nothing truth-side, and
`tests/test_plot_arrays.py::test_recovery_side_does_not_import_the_scoring_side_writers`
audits statically that no recovery-side module imports it, mirroring `tests/test_firewall.py`.

**Where it lives:** `src/rngrn/plotdata.py`; `src/rngrn/train.py` (`_save_run_arrays`);
`src/rngrn/config.py` (`solver.save_plot_arrays`); `.gitignore`;
`scripts/make_figures.py`; `tests/test_plot_arrays.py`.

### D-PLOT-2 — the training trajectory is recorded at stride 10, for ALL members, at float32

**Date:** 2026-07-30. **Status:** DECIDED, with the stride flagged as a SIZE tradeoff rather
than a scientific one. **Decided by:** the implementing agent; the default stride of 10 was
the owner's suggestion and is adopted with its cost measured.

**The problem.** `recover()` kept one summary row per RESTART and nothing per step. A run
that reached its loss by step 100 and one that crawled there by step 1900 left identical
records, the weights actually in force were never written down (and `DataFirstStaging` ramps
`turing` from 0 across the first 50% of the budget, so a loss curve read without them is
misleading), and the D trajectory — whether recovery reaches biologically plausible
diffusivities is the live question behind D1/D3 — was invisible.

**The decision.** `history.TrainingHistory` records, per recorded step and per **member**:
every loss term (`L_*`) and the total, every weight in force (`w_*`), the **constrained
physical** parameters KA/KR/α/δ/β/D (36 columns at N=3 — *not* raw θ, which is a link-function
artefact), the derived D-ratio, the per-step diagnostics (`kstar_model`, `sig_max`,
`sig_max_pos`, `rel_err`), and `ss_converged`. Both the serial and the batched path are
instrumented, since phase C runs batched on CUDA.

**Stride 10, with the cost measured.** One record per step × 64 restarts × 36 parameters is
~921k floats per 400-step run (**≈3.7 MB** at float32) — not affordable across a 96-run wave.
Stride 10 measures **≈0.51 MB** per run (S=41, B=64, Q=17, P=36, float32, measured by
`savez_compressed`), and still puts 21 recorded points across the staging ramp at
`adam_steps=400`. **Step 0, the last training step, and the post-training evaluation step are
recorded unconditionally whatever the stride**, so no curve's endpoints are interpolated.
`train.history_every=0` disables the recorder entirely.

**ALL members are kept, thinned — never member 0 relabelled as "the run".** On the batched
path B members are optimised jointly and the reported result is the **best** member, whose
index is only known after the last step; recording one lane during training would either be
the wrong lane or require guessing. `meta.history_best_member` records which lane won, so a
figure can plot the winner without the trace having pre-committed to it. The offered
alternative (full trace for the winner plus min/median/max across the rest) was **not**
taken: it costs a second pass to identify the winner and would have discarded the
restart-to-restart spread, which is itself the answer to "is this recovery reproducible".

**A member that dies is an EVENT, not a gap.** A restart whose steady state diverges is
abandoned; its lane becomes NaN from that step on, and `hist_death_step` records the step it
died at. Lanes are never reindexed, so the seed-to-lane mapping stays auditable.

**float32, and one measured performance note.** The trajectory is plotted, not
differentiated. On CUDA the first draft read parameters per (member, parameter), which is
6·B synchronising device→host copies per recorded step (15 744 for a 400-step, 64-restart run
at stride 10); `history._param_block` reads each parameter once for the whole batch instead —
6 copies per recorded step. The loss/diagnostic entries of a batched `parts` dict are already
numpy (`losses/terms._np`), so they cost no extra sync.

**Where it lives:** `src/rngrn/history.py`; `src/rngrn/recover.py` (both loops);
`src/rngrn/config.py` (`train.history_every`); `src/rngrn/export.py` (`tidy_history`);
`tests/test_plot_arrays.py`.

### D-PLOT-3 — the cross-L table is persisted through the run-index machinery; exports are TIDY/long

**Date:** 2026-07-30. **Status:** DECIDED. **Decided by:** the implementing agent.

**The problem, twice over.** (1) `eval.lgen_eval.evaluate_across_L` returns a rich per-L table
and `cli.cmd_evaluate` **printed it to stdout and stored nothing** — so the measurement in
`docs/LGEN_TRANSFER_FIRST_RESULT.md` could not be re-plotted without re-running a 9-minute
recovery. (2) The indexes are one row per run with ~120 columns, which is a record, not a
plotting frame: drawing "outcome vs hyperparameter" from a wide row requires knowing which
columns are identifiers and reshaping first.

**The decisions.**

- **Cross-L results go through the SAME machinery** `optim/target_report.py` uses —
  `index.open_index(runs_root, <table>, backend)` — as **flat scalar** rows built with
  `dict.update()`: one row per (run_id, L) in table `lgen_eval`, one summary row per
  evaluation in `lgen_summary`. Both carry `run_id` and `git_sha` so a row joins back to the
  run that produced the model and the code that evaluated it. Two tables rather than one
  mixed table with a `row_kind` discriminator, because a tidy export of a mixed table would
  have to filter before it could be plotted. The per-L **fields** go to
  `<run_dir>/arrays/lgen_fields.npz` (`evaluate_across_L(keep_fields=True)`), one key per L
  because under `grid_rule='constant_dx'` the grids differ between L and cannot be stacked.
  Verified by re-running the tracked `m3_registry_20260730_005701` evaluation: every number in
  `docs/LGEN_TRANSFER_FIRST_RESULT.md`'s transfer table reproduced exactly
  (`kstar_phys_cv` 0.047636, `periods_slope_rel_err` 0.014345,
  `morphology_class_preserved` 1.0, `min_pixels_per_wavelength` 11.055).
- **`rngrn export` writes LONG/TIDY csv** — `runs_tidy.csv`, `target_reports_tidy.csv`,
  `lgen_tidy.csv`, `history_tidy.csv` — as `<identifier columns…>, variable, value,
  value_num`: **one observation per row**. `value` keeps the verbatim text (so a categorical
  observation like `morphology_pred` survives) and `value_num` the numeric reading, empty when
  there is none; booleans are numeric 0/1 *and* text. The **hyperparameters stay as identifier
  columns** (`export.RUN_ID_COLS`), which is exactly what makes an outcome-vs-hyperparameter
  plot a group-by rather than a reshape; a sweep already writes one run row per cell
  (`optim/sweep.py` calls `fit()` per point), so a sweep axis is a column like any other.
  Stdlib `csv` only — a tidy export must not be the reason a plotting environment needs
  pandas. `tests/test_plot_arrays.py` asserts the tidiness directly: every record has exactly
  one variable/value pair and (identifiers, variable) is unique.
- **Missing hyperparameter columns were added to the run row**: `seed` and `model_seed` (their
  absence made a seed-replicate sweep — this project's standard design — impossible to
  disaggregate from the index alone), `adam_lr`, `lbfgs_steps`, `grad_clip`, `tau`,
  `jac_floor`, `dratio_centre`, `dratio_spread`, `ratio_update_every`, the remaining loss
  weights, `n_grid`, `history_every`, `git_sha`, `plot_arrays_bytes`.
- **`history_tidy.csv` defaults to the winning member only** (`--history-members best`), which
  is what a learning curve "for this run" means and keeps the file at ~S·(Q+P) rows instead of
  ~112k for one 400-step, 64-restart run. `--history-members all` exports every recorded
  restart. Nothing is dropped from the npz either way — the arrays always hold every member;
  the flag chooses what the CSV projects, and it is documented in `--help`.

**Where it lives:** `src/rngrn/plotdata.py` (`lgen_rows`, `save_lgen_fields`);
`src/rngrn/cli.py` (`_persist_lgen`, `cmd_export`); `src/rngrn/export.py`;
`src/rngrn/eval/lgen_eval.py` (`keep_fields`); `tests/test_plot_arrays.py`.

---

## Part 2b — Stage 0: the biological-viability precondition (unit c-bioviab, 2026-08-03)

Decisions taken while measuring whether `configs/bio_box.yaml` is Turing-EMPTY — the
feasibility precondition behind `docs/PREREGISTRATION.md` §3.4. Results in
`docs/BIO_VIABILITY.md`; the D-EVID-5 entry below belongs with Part 1's evidence-integrity
defects and is placed here only to keep the unit's entries together.

---

### D-EVID-5 — `loss.weights.param_prior` was a NO-OP: the biological prior was never in the objective

**Date found:** 2026-08-03 (unit c-bioviab). **Fixed:** 2026-08-03, same unit.
**Status:** SUPERSEDED (defect; now fixed).

`losses/terms.py::param_prior` existed, was tested (`tests/test_plausibility.py`), had a
weight in `config.py` (`loss.weights.param_prior`, default 0.0), had its own knobs
(`loss.dratio_centre`, `loss.dratio_spread`, `loss.bio_box_path`) and had a column on
every run-index row (`w_param_prior`, `dratio_centre`, `dratio_spread`, written by
`train.py`) — **and no caller on the path recovery runs.** `losses/total.py`'s
`compute_terms` / `compute_terms_batched` — the assemblers `recover.py` actually calls —
never referenced it. Its only caller was `losses/terms.py::composite_loss`, which the
module's own docstring labels "the standalone reference form" and which `recover.py` does
not use. `train.py` also never passed `cfg.loss.dratio_centre` / `dratio_spread` /
`bio_box_path` to `recover()`.

Consequence: a run configured with `loss.weights.param_prior = 1.0` would have trained
with **no** biological prior, recorded `w_param_prior: 1.0` on its index row, and said
nothing. Any future claim of the form "recovery under the biological prior does X" made
before this fix would have been measuring the prior-OFF arm. No such run exists in the
index yet (`w_param_prior` is 0.0 on every recorded row), so no published number is
affected — this was caught before it produced evidence, not after.

A second, smaller defect found in the same read and **deliberately left alone**:
`composite_loss` computes `loss` twice, and its second assignment drops the
`w['anchor']*L_s` term that the first included. It is a standalone reference form with no
caller in `src/`, so fixing it is out of this unit's scope; recorded here so it is not
rediscovered as new.

**Fix:** `compute_terms` / `compute_terms_batched` take `param_prior_kw`; `None` (the
default) omits the key from `term_vals` entirely, so `strategy.combine`'s
`sum(weight * term)` never sees it and every previously recorded number is reproduced
bit-for-bit (`tests/test_plausibility.py::test_param_prior_kw_none_leaves_the_objective_byte_identical`).
`recover.py` builds the kwargs from the strategy's base weight, exactly as it already
decides `compute_resid`, and **raises** if the prior is asked for with an adaptive
(non-static) strategy, which would re-weight the prior by its own magnitude. A batched
twin `terms.param_prior_batched` was added (the serial one reads only `model.D/.alpha/
.delta`, all of which carry the leading `B`) and is pinned equal to the serial term member
by member. `configs/base.yaml`'s `weights` dict gained a `param_prior: 0.0` row — value
unchanged, but without it `apply_overrides` fails loud on an unknown key and
`-o loss.weights.param_prior=...` was unreachable from the CLI.

**Where it lives:** `src/rngrn/losses/total.py`, `src/rngrn/losses/terms.py`
(`param_prior_batched`), `src/rngrn/recover.py`, `src/rngrn/train.py`,
`configs/base.yaml`, `tests/test_plausibility.py` (4 new tests).

---

### D-EVID-6 — the pre-push hook tested a DIFFERENT worktree's source

**Date found:** 2026-08-03 (unit c-bioviab). **Fixed:** 2026-08-03, same unit.
**Status:** SUPERSEDED (defect; now fixed).

`.githooks/pre-push` falls back to `../turing-training/.venv/bin/python` when the current
worktree has no `.venv` of its own — which is every worktree. That venv was created by an
**editable install**, which writes `turing-training`'s absolute `src` path into a `.pth`
file. So from any other worktree the hook ran `pytest` against
`../turing-training/src/rngrn`, **not against the source being pushed.**

**MEASURED, not inferred.** The four tests added in this unit for the `param_prior` wiring
FAILED under the hook and PASS under `PYTHONPATH=$PWD/src` — because the hook was exercising
a tree that does not contain the change. Directly:

```
hook would import: .../worktrees/turing-training/src/rngrn
with PYTHONPATH  : .../worktrees/c-bioviab/src/rngrn
```

The failure mode is worse than a false red: a worktree whose change BREAKS something that
`turing-training/src` does not contain would have been green-lit. This is the same class of
defect as D-EVID-2 — a hook that reads as protection while protecting nothing — and it was
active for every worktree push since the sibling-venv fallback was added.

**Fix:** the hook now exports `PYTHONPATH="$repo_root/src"` before running pytest
(PYTHONPATH takes precedence over `.pth` entries) and PRINTS the resolved `rngrn` path, so
which tree was tested is visible in the push output instead of having to be inferred.

**Where it lives:** `.githooks/pre-push`.

---

### D-BIO-1 — `beta` for the box-viability sampler is the GENERATOR's own basal draw, and is never scored

**Date:** 2026-08-03. **Status:** DECIDED (scope: the Stage-0 sampler only).

`configs/bio_box.yaml` marks `beta` `source: UNCITED`, and `plausibility.py` never scores
an UNCITED row. But a value is still needed to *build* a model, so one had to be picked.
Picked: `beta ~ 10**U(-2, -0.3)` (0.01–0.5), which is `scripts/gen_tg3.py:157`'s own basal
draw `b = 10**rng.uniform(-2, -0.3, 3)` — i.e. calibrated against the data-generating
process this pipeline actually fits, not invented. **Rejected:** inventing a bound and
putting it in `bio_box.yaml`; and `docs/STATE_OF_THE_SCIENCE.md` §10's beta ~1e-4–1e-2
range, which that document explicitly marks as an init-distribution search result "not a
neutral prior" and "deliberately not adopted".

Sensitivity is measurable rather than assumed: `--beta-mode {generator,low,high}` reruns
the whole part-1 measurement at fixed beta = 1e-3 and beta ≈ 0.5.

**Where it lives:** `scripts/stage0_bio_viability.py::BETA_MODES`; `docs/BIO_VIABILITY.md`.

---

### D-BIO-2 — the D vector is sampled under TWO paired schemes, because the near-immobile node is an INTENDED escape valve

**Date:** 2026-08-03. **Status:** DECIDED (scope: the Stage-0 sampler only).

`scoring/plausibility.py::d_ratio_of` takes the ratio of the two MOST MOBILE species
deliberately, so a near-immobile third node never enters the scored `d_ratio` — the
mechanism `docs/ROBUSTNESS_MEASUREMENT.md` §4.4 measures as keeping 127/127 generator
systems strictly Turing when the slowest diffuser is immobilised. A sampler that ignored
that would answer a different question from the one the box asks. So D is drawn under two
schemes, both satisfying the `d_ratio` row by construction and reported separately:

* `mobile3` — all three species genuinely mobile: `D_lo ∈ 10**U(-1,0) × D_mid`.
* `immobile` — a near-immobile third node: `D_lo ∈ 10**U(-4,-2) × D_mid`.

Both draw from the SAME rng stream in the same order under a common seed, so they differ
in the slowest D and in nothing else: a PAIRED comparison. The absolute D scale is fixed
at `D_max = 1` because it is a pure length rescale (`k* ∝ 1/√D`) that part 3 absorbs into
its choice of L.

**[UNCALIBRATED]** the two decade windows (−1..0 and −4..−2) are a reasoned separation of
"mobile" from "near-immobile", not a measured boundary. What they support is the
*comparison* between the arms, not an absolute rate.

**Where it lives:** `scripts/stage0_bio_viability.py::D_SCHEMES`, `draw_batch`.

---

### D-BIO-3 — the Turing verdict is the STRICT test, on a LOG k-grid to 10^3.5

**Date:** 2026-08-03. **Status:** DECIDED.

Stability is `max Re eig(J) < 0`, not `tr(J) < 0`. `docs/ROBUSTNESS_MEASUREMENT.md` §3
measured the trace test overcounting by up to 70 % of draws, and the robustness fix already
moved to strict for that reason; `eval/analysis.py::turing_ok` still uses the trace test, so
its verdict is reported here only as `turing_loose`, never led with. **Measured in this
unit:** on 80 000 in-box draws the loose criterion accepts 1215 and the strict criterion 19
— the trace test overcounts by a factor of ~64 on this population, because every one of its
extra "successes" is a UNIFORMLY unstable state (`max Re eig(J) > 0`) whose dispersion peaks
at k → 0. That is not a Turing pattern; it is a homogeneous instability.

The k-grid is `[0] ++ logspace(-3, 3.5, 500)`, not `eval/analysis.py`'s
`linspace(1e-3, 50, 4000)`: with a near-immobile node (D ~ 1e-6 × D_max) the branch that
node dominates only turns over at `k ~ √(|J|/D_lo) ~ 10^3`, so a grid stopping at k = 50
cannot see whether it grows there. k = 0 is prepended and EXCLUDED from the instability
search so the uniform mode can never be reported as the growing one.

A draw whose steady state does not converge is a FAILED draw: it stays in the denominator
with `converged=False` and is never silently dropped.

**Where it lives:** `scripts/stage0_bio_viability.py::KGRID`, `turing_verdict`,
`turing_verdict_batch`.

---

### D-BIO-4 — the box-constrained sampler is BATCHED, and pinned against the serial reference

**Date:** 2026-08-03. **Status:** DECIDED.

The serial path costs ~55 ms per draw (model construction + `losses/terms.steady_state`),
so 80 000 draws would be 75 min. Batched through `BatchedRNGRN` + `steady_state_batched` +
batched numpy `eigvals` it is ~130 draws/s. Because a fast path that quietly disagrees with
the reference would invalidate every rate in this unit, part 1 re-runs 32 draws through the
serial `RNGRN` + `steady_state` + serial verdict and compares. **Measured:** 32/32 verdicts
agree; worst `|Δ max Re eig J|` = 4.27e-06, which comes from the two solvers legitimately
stopping at different residuals (the serial relaxation fallback accepts `|f| < 1e-4`, the
batched Newton reaches ~1e-10 on the same root). A verdict disagreement, or a deviation
above 1e-4, is fatal and aborts the run.

**A note on the artefact's size, since it was weighed rather than ignored.**
`part1_box_draws_generator.npz` is **25 MB** — 80 000 draws x 39 float64 parameters. That is
the single largest tracked artefact in the repo. It is kept whole because §1's rates are the
headline claim of this unit and D-PLOT-1's standing rule is that the arrays behind a claim
are versioned with it; 25 MB also sits inside the 10-62 MB per phase-C wave that rule already
accepts. Two cheaper options were considered and rejected: storing the parameters at float32
(halves it, but then the stored draws are not the draws that were evaluated), and storing
parameters only for the 19 survivors (the rest are exactly regenerable from the recorded
seed, but every consumer would then need an index-remapping layer for a saving of 23 MB).
Recorded so the choice is visible; if the repo later needs the space, regenerating from
`seed=0, n=20000, beta_mode=generator` reproduces the file exactly.

**Where it lives:** `scripts/stage0_bio_viability.py::part1` (the equivalence block).

---

### D-BIO-5 — the mu sweep varies the GEOMETRIC MEAN of delta, and separately one species at a time

**Date:** 2026-08-03. **Status:** DECIDED.

`delta` is a vector but the box row bounds every species, so there is no single "the delta".
Two parameterisations, both reported:

* **uniform** — hold the base draw's per-species delta *pattern* and rescale it so its
  geometric mean equals the swept value `d`. Individual `delta_i` can then sit outside
  [0.4, 5] even when `d` is inside it, so `all_delta_in_box` is recorded per point.
* **per_species** — move ONE `delta_i` and hold the others at the base draw.

Range: 41 log-spaced points over [0.04, 50], i.e. the cited box row [0.4, 5.0] plus one
decade either side, so the boundary is located relative to the box edges rather than
asserted to be outside them.

**Useful fact, worth stating explicitly:** the generator draws `mu = 10**U(-0.4, 0.7)`
= 0.398…5.01 (`scripts/gen_tg3.py:159`) and the box row is [0.4, 5.0]. These are the SAME
range to two significant figures, so unlike the D-ratio axis (generator ~8–250 vs box
[1, 60]) the degradation axis is one where the box does NOT fight the data.

**Where it lives:** `scripts/stage0_bio_viability.py::SWEEP_D`, `_sweep_one`.

---

### D-BIO-6 — the rollout domain is L = 6 × 2π/k* on a 64×64 grid

**Date:** 2026-08-03. **Status:** DECIDED.

Six full wavelengths of the fastest-growing linear mode across the box. Six is the
generator's OWN original choice — `scripts/gen_tg3.py:66` records "The original generator
hard-coded 6 periods per box" — and sits inside its replacement acceptance window
`PERIODS_CHOICES = {3..14}`, so these figures use the same domain size the training data was
built at. 64×64 is the owner's "small grid" and costs 0.54 s a field (`eval/rollout.py`,
measured, `etdrk4_rfft`, `OMP_NUM_THREADS=1`, N=3, 609 steps).

`stopped_reason` is recorded and printed on every panel: a run that ended on `step_budget`
was TRUNCATED and its field is not a statement about the attractor, so it is labelled as
such rather than shown as a pattern. `sig_max` is the SIGNED dispersion maximum.

**Where it lives:** `scripts/stage0_bio_viability.py::PERIODS`, `GRID_N`, `part3`.

---

### D-BIO-8 — part 3 shows the channel with the LARGEST spatial std, and records BOTH pattern verdicts

**Date:** 2026-08-03. **Status:** DECIDED.

`eval/rollout.py::simulate` defines `amplitude` and `patterned` on **channel 0 only**. That
is right for a real dataset, where channel 0 is the observed channel. It is wrong for the
Stage-0 sampler, and by the sampler's own doing: `draw_batch` shuffles the species order
(nothing distinguishes the species except which one drew the small D), so channel 0 is
arbitrary. **MEASURED:** a competitive/`mobile3` survivor has channel-0 std **2.0e-4** and
channel-1 std **0.76** — the library calls it unpatterned while a large labyrinth sits one
channel over. Reading channel 0 gave 4/12 patterned; reading any channel gives **9/12**.

Decided: each panel shows the channel with the largest spatial std and NAMES it in the
title, with all three channel stds listed beside it. Both verdicts are recorded and neither
overrides the other — `patterned` stays the library's channel-0 number, untouched, and
`patterned_any` is the new any-channel one. **Rejected:** changing `eval/rollout.py`'s
definition, which is correct for its actual callers; and quietly reporting the any-channel
number under the library's name.

**Where it lives:** `scripts/stage0_bio_viability.py::part3`,
`scripts/stage0_figures.py::fig_pattern_gallery`.

---

### D-BIO-9 — the sweep grid always contains the base draw's own delta, and the harvest is round-robined across D-schemes

**Date:** 2026-08-03. **Status:** DECIDED. Both are corrections to defects found by
dry-running the pipeline before the real data landed.

1. **The base point is inserted into the sweep grid.** The 41-point log grid does not
   generally contain the base draw's own `delta`, so a system harvested BECAUSE it is
   strictly Turing could report `0/41` strictly Turing — a curve missing the one point it is
   known to pass through. `_sweep_one` now adds the base value and marks it (`is_base`,
   drawn as an open diamond).
2. **Survivors are round-robined across the D-schemes.** `survivors(mobile3)[:n] +
   survivors(immobile)[:n]` truncated to `n` yields `n` mobile3 and ZERO immobile whenever
   mobile3 has `n` of its own — which is exactly what happened on the first part-2 run, and
   would have made every curve and panel one scheme's anecdote, the precise thing the two
   schemes exist to prevent. `balanced_picks` alternates and falls through when one pool
   empties. The re-run gives 3 `mobile3` / 3 `immobile` per form.

**Where it lives:** `scripts/stage0_bio_viability.py::_sweep_one`, `balanced_picks`.

---

### D-BIO-7 — the binding parameters (KA/KR) are the FREE directions and are sampled over four decades

**Date:** 2026-08-03. **Status:** DECIDED, marked **[UNCALIBRATED]**.

`configs/bio_box.yaml` has no row for the binding budget `s` or the gate `g` at all, so
`plausibility_score` is blind to them: they are the directions in which the Turing region is
free to live. The sampler draws `s ~ 10**U(-2, 2)` per entry and the gate logit
`~ N(0, 2.5)` (the wide, sub-saturated split of `model.py::_low_basal_raw_params`). The
four-decade window is chosen to bracket the generator's own Hill thresholds — `gen_tg3.py:160`
draws `K = 10**U(-1, 0.6)` with `n ∈ {2,3,4}`, so the comparable binding scale `1/K^n` spans
roughly 1e-2…1e2 — but the mapping between a Hill threshold and this model's `KA` is not an
identity, so this is a bracketing argument, not a calibration. Marked UNCALIBRATED rather
than presented as derived.

This matters for how the acceptance rate is read: it is the acceptance rate **under a stated
prior**, and a wide prior on an unscored direction lowers it. Figure `f1` panel (d) reports
where the survivors actually sit in `s`, which is the part of this that is a measurement.

**Where it lives:** `scripts/stage0_bio_viability.py::S_LOG_RANGE`, `GATE_LOGIT_SD`.

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

---

## Part 2c — Stage 0b: robustness at FINITE mu, i.e. with the QSS reduction UNDONE (unit c-mu, 2026-08-03)

Full write-up: `docs/TIMESCALE_MU.md`. Code: `src/rngrn/eval/lifted.py`,
`scripts/stage0b_mu.py`, `scripts/stage0b_figures.py`, `tests/test_lifted.py`.
Arrays and figures: `experiments/figures_report/stage0b/`.

### D-MU-0 — `mu` here is the QSS lift's FAST-VARIABLE TIMESCALE, not the generator's degradation rate

**Date:** 2026-08-03. **Status:** DECIDED (terminology, project-wide).

`mu` names two unrelated quantities in this project and the ambiguity has already cost one
unit's work. `docs/BIO_VIABILITY.md` / `scripts/stage0_bio_viability.py` swept the
GENERATOR's `mu[i]` (`scripts/gen_tg3.py:93`), which is the RNGRN's `delta`. This unit
measures the OTHER one: the promoter-gate relaxation timescale of
`eval/dynamical.py::lift_check`. Both readings are legitimate and both are now measured;
what is no longer acceptable is writing `mu` without saying which. Every docstring in
`src/rngrn/eval/lifted.py` and `scripts/stage0b_mu.py` opens by stating it.

**Where it lives:** `src/rngrn/eval/lifted.py` module docstring.

---

### D-MU-1 — the finite-mu cloud perturbs PHYSICAL KINETIC PARAMETERS, not entries of the reduced Jacobian

**Date:** 2026-08-03. **Status:** DECIDED (forced, with a stated cost).

`eval/analysis.py::_draw_JD_cloud` — the basis of every `turing_volume` number to date —
perturbs entries of the REDUCED Jacobian `J` directly. That cannot be used on a lifted
system: the 21×21 `J_full` is *built from* `(KA, KR, alpha, beta, delta)` and `x*`, so a
perturbed reduced `J` corresponds to no lifted system at all and cannot be lifted. The
perturbation therefore acts one level down — independent lognormal factors on `KA, KR,
alpha, beta, delta, D`, with `x*` re-solved and the Jacobian re-derived per draw. This is
Tica et al.'s own perturbation model, which `analysis.py`'s docstring already names as the
alternative to its own.

What is preserved by construction: every factor is lognormal hence strictly positive, and
every perturbed quantity is strictly positive under the model's link functions, so no sign
flips and no structural zero is created — the two properties that made the original scheme
correct.

**THE COST, STATED:** absolute volumes under this scheme are NOT comparable to the QSS
tables in `docs/ROBUSTNESS_MEASUREMENT.md` §4.2 or to the §3.2 calibration table. Measured
on the two recovered networks at the same 400 draws / seed 0: the repo's reduced-J scheme
gives `turing_volume_4p8pct` 0.657 / 0.655 and `_10pct` 0.390 / 0.450 for sample_0003 /
sample_0004, where this scheme gives 0.265 / 0.215 and 0.150 / 0.095. Perturbing kinetic
parameters moves `x*` as well as `J`, so it is strictly harsher. **Every finite-mu number is
therefore read against its OWN mu→0 column, computed on the SAME draws from the reduced
Jacobian** — along a curve the perturbation model is held fixed and only `mu` varies, so the
mu-dependence is isolated even where the absolute level is not comparable.

**Where it lives:** `src/rngrn/eval/lifted.py::draw_param_cloud`, `robustness_vs_mu`.
**Evidence:** `experiments/figures_report/stage0b/arrays/robust.json`,
`qss_repo_scheme_volumes.json`.

---

### D-MU-2 — the strict test, on the FULL 21×21 lifted Jacobian; the trace test is not merely loose here, it is broken

**Date:** 2026-08-03. **Status:** DECIDED.

Verdict: `max Re eig(J_full) < 0` at `k = 0` AND `max_{k>0} sigma_full(k) > 1e-9`, with
diffusion `diag(D_1..D_N, 0, ..., 0)` — only `x` diffuses, promoter states are DNA-bound.

`eval/analysis.py::turing_ok` uses `trace(J) < 0`, which Stage 0 measured overcounting by
64×. On a LIFTED system it is worse than loose: `tr(J_full)` is dominated by the `-1/mu`
gate diagonal and stays negative at ANY `mu`, so it would report "uniform state stable" for
a system that has already lost stability. It is never used here.

Separately, a Turing-unstable mode whose leading eigenvalue has `Im ≠ 0` is a travelling
wave, not the stationary pattern this project claims to recover. `frac_stationary` is
reported beside `frac_turing` everywhere and the two are never merged. **Measured: 0 of 8000
draws was oscillatory at any `mu` on the swept axis** — the distinction did not bind here,
but it is instrumented so it cannot silently start binding.

**Where it lives:** `src/rngrn/eval/lifted.py::verdicts_from_J`, `turing_verdict_lifted`.

---

### D-MU-3 — the gate substep is solved EXACTLY at frozen x, so the integrator structurally cannot fake a dead pattern

**Date:** 2026-08-03. **Status:** DECIDED.

The gates relax at rate `1/mu`, so an explicit scheme needs `dt << mu` and a `mu = 1e-3` run
costs ~1000× the QSS one. A stiff integrator that silently damped the instability would
produce "the pattern died at finite `mu`" as an ARTEFACT — the most dangerous failure mode
available in this unit. It is removed structurally rather than tested away: at frozen `x`
the gate block is LINEAR in `G` and is integrated in closed form.

* **nc1** — diagonal: `G ← G_inf + (G − G_inf) exp(−(1+u) dt/mu)`, `G_inf = u/(1+u)`.
* **competitive** — per row the `2N` gates obey `dw/dt = (a − (I + a 1ᵀ)w)/mu`, and
  `A = I + a 1ᵀ` is a RANK-ONE update of the identity whose exponential is closed form:
  `exp(−Aτ) = e^{−τ}(I + ((e^{−Sτ}−1)/S) a 1ᵀ)`, `S = Σa`, with `w_inf = a/(1+S)` — which
  IS the QSS occupancy. No linear solve, no `expm`, unconditionally stable, and it reduces
  EXACTLY to the QSS scheme as `mu → 0`.

The `x` substep reuses `eval/numerics.py`'s validated cached-coefficient ETDRK4 (rfft) under
Strang splitting, and `dt`/horizon follow `eval/rollout.py::simulate`'s own growth-rate-aware
policy, so the QSS and lifted runs of a model use the SAME `dt` and horizon and are directly
comparable.

**Evidence:** exact substep vs 2×10⁵-substep explicit Euler of the same ODE, both forms:
max abs error **< 1e-5** (`tests/test_lifted.py::test_gate_step_is_exact`). dt-halving and
the `mu → 0` QSS cross-check: `experiments/figures_report/stage0b/arrays/dtconv.json`,
figure `g5_dt_convergence.png`.

---

### D-MU-4 — `J_full(mu)` is obtained by scaling the GATE ROWS of `J_full(1)` by `1/mu`, exactly

**Date:** 2026-08-03. **Status:** DECIDED (implementation, measured exact).

`mu` enters the lifted RHS only as an overall `1/mu` on the `2N²` gate equations; the `x`
equations do not contain it. So every gate ROW of the Jacobian carries exactly one factor
`1/mu` and the `x` rows carry none. A whole `mu` axis therefore costs ONE vmapped autodiff
Jacobian per draw plus one eigen-scan per `mu` point, instead of one autodiff per point.

**Evidence:** vs a fresh autodiff Jacobian at `mu` = 1e-3, 0.1, 7.3, both forms: relative
deviation **< 1e-12** (`tests/test_lifted.py::test_rescale_mu_matches_autodiff`).

A companion fact, also measured and also load-bearing: the lifted FIXED POINT is the QSS one
at EVERY `mu` (worst `max|f_lift(z*)|` over 8 systems × 7 `mu` from 1e-6 to 1e3: **1.28e-8**),
so `mu` moves stability and dynamics and never the steady state, and every comparison across
`mu` is apples-to-apples.

---

### D-MU-5 — ONE finite `mu` (1e-3) plus a one-decade stress point (1e-2); the rest of the axis is context, not a claim

**Date:** 2026-08-03. **Status:** DECIDED (scope, set by the owner).

The owner scoped this unit to robustness AT finite `mu`, with timescale *separation* as a
problem explicitly out of scope. Two values are named and defended and nothing else is
claimed on:

* `mu = 1e-3` (**headline**). `mu` = (TF-promoter binding time)/(protein turnover time),
  dimensionless. Live-cell single-molecule tracking gives specific TF-DNA dwell times ~10 s
  — Chen J. et al. (2014) *Cell* 156:1274-1285 measure Sox2 at 12.0-14.6 s specific,
  0.75-0.9 s non-specific. Measured morphogen clearance in patterning tissue gives mean
  lifetimes ~1.4-1.9×10⁴ s — Müller P. et al. (2012) *Science* 336:721-724 report
  Cyclops/Squint/Lefty1/Lefty2 half-lives of 95-218 min. 10/2×10⁴ ≈ 5×10⁻⁴, rounded up one
  notch. `configs/bio_box.yaml`'s cited `delta` row `[0.4, 5.0]` puts one model time unit at
  0.4-5 protein lifetimes, which is inside the rounding.
* `mu = 1e-2` (**stress**). One decade slower gates than measured; if the verdict holds here
  it holds at any defensible `mu`.

**[ORDER OF MAGNITUDE]** this is not pinned better than a decade and no figure pretends
otherwise. Schwanhäusser et al. (2011) *Nature* 473:337-342's median protein half-life of
46 h in cultured NIH3T3 is an order of magnitude LONGER than the developmental anchor and
would push `mu` DOWN, so the value used is the conservative (larger-`mu`) choice.

**Where it lives:** `scripts/stage0b_mu.py::MU_FINITE`.

---
