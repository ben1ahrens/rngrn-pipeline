# HANDOFF — visual inspection of the training data

> **ARCHIVED 2026-08-14.** This handoff was picked up and completed. Every deliverable it
> promises exists on disk: `notebooks/training_data_simulations.ipynb`,
> `scripts/td_figures.py`, `tests/test_td_figures.py`, and
> `experiments/figures_report/training_data/` (62 files). `D-TDPLOT-1` is recorded in
> `docs/DECISIONS.md`. It is superseded by its own deliverables — the notebook and
> `experiments/figures_report/training_data/` are the current reference for the visual
> inspection this handoff performed; nothing else in the repo carries this content forward.

---

**Branch:** `feature/training-data-plots`, off `feature/fft-training` (`7c8bf37`).
**Date:** 2026-08-10.
**Written for a zero-context reader.**

---

## 1. Why this branch exists

The owner asked for a notebook that runs the training-data simulations, saves plots with a
colour bar, axis labels and tick marks, and confirms that the data actually shows patterning.

The underlying question is load-bearing: the whole inverse problem assumes the registered
datasets carry Turing patterns, and nobody had checked that across the corpus. The one
existing pattern gallery (`experiments/figures_report/stage0/f3_patterns_*.png`) covers
Stage-0 biological-viability rollouts, not the registered datasets, and its panels have
colorbars but **no axis labels and no tick marks**, so the physical scale of a pattern
cannot be read off the image.

## 2. What was added

| path | what it is |
|---|---|
| `notebooks/training_data_simulations.ipynb` | the deliverable, executed with outputs committed |
| `scripts/td_figures.py` | figure + measurement helpers (535 lines) |
| `tests/test_td_figures.py` | 31 tests; the figure contract as assertions |
| `experiments/figures_report/training_data/` | 57 PNGs, 23 MB — **will be tracked on commit**; `.gitignore` un-ignores `experiments/figures_report/**/*.png` |
| `tests/test_firewall.py` | added `td_figures`, `gen_tg3` to `FORBIDDEN` (see §6) |
| `docs/DECISIONS.md` → `D-TDPLOT-1` | the one science decision taken here |
| `pyproject.toml` | new `notebook` optional-dependency group |

Figures are tracked because `.gitignore` already un-ignores
`experiments/figures_report/**/*.png` — that exception exists precisely for the case where
"the IMAGE is the deliverable rather than a regenerable view of an array", which is this
case. Nothing else in the repo changed; no recovery-side module was touched.

**To re-run:** `pip install -e ".[notebook]"`, then
`cd notebooks && python -m nbconvert --to notebook --execute --inplace training_data_simulations.ipynb`.
~5 min on CPU. The re-simulation section dominates: 19.8 s/sample mean over the six
(11.1–23.5 s), ~2 min for the section. No trainer is launched, so `scripts/guarded_run.sh`
(CLAUDE.md §7a) does not apply; peak RSS was not measured, so no claim is made about it.

## 3. What is VERIFIED

All 413 samples across the **11 datasets registered as of 2026-08-10 14:00** — which was
every one that existed when this ran. It is no longer "all": `turing_spots` and
`turing_labyrinth` were generated six hours later the same day and are **not** among these
413. Their patterning is verified separately and differently — see
`docs/archive/HANDOFF_canonical_datasets.md` §2.

- **Every field panel carries a colorbar, both axes labelled, and tick marks in physical
  length units** (`extent=(0, L, 0, L)`), so a wavelength is readable straight off the
  figure. These are enforced by tests, not left to convention.
Provenance for every number below: `notebooks/training_data_simulations.ipynb`, cells 3/5/8/16,
executed 2026-08-10; reproducible with the nbconvert line in §2.

- **Payload integrity.** The cv recomputed from every stored final frame reproduces the
  generator's stored `cv0` attribute to **≤ 2.2e-07** across all 413 samples.
- **413/413 samples clear `cv >= 0.05`** (corpus cv min 0.0633, median 0.688, max 2.882) —
  but read the caveat below before citing this. It is **circular**.
- **The corpus is periodic on the spectral screen**, with exactly one exception (§4).
- **The wavenumber in each image tracks the generating system's** `k*`: median relative
  deviation 8.3 %, 90th percentile 22.7 %. Signed: median **+4.6 %**, with **64 %** of
  samples above the generator's `k*` — a *tendency* high, not a uniform offset, and 8.3 %
  is the median absolute deviation, not the bias.

  As far as this branch can determine, that is the **first measurement of this bias on this
  corpus**. `src/rngrn/observables.py:11-13` asserts the same direction ("biased HIGH off
  onset") and a "~10-15 %" reliability, but that docstring is **unchanged since the initial
  template commit (`7f33264`)** and cites `observables_spec.md`, which **does not exist
  anywhere in this repo**. So it corroborates the direction and calibrates nothing — it is
  the same template-era class of claim as the `~15 %` k\* tolerance CLAUDE.md §8 already
  names. An earlier draft of this handoff cited it as if it settled the question.
- **`three_gene_qvar` and `three_gene_multiL` reproduce BIT-EXACTLY.** Six samples
  re-simulated from `params_json` + the stored `sim_seed` through `scripts/gen_tg3.py`:
  relative L2 error `0.000e+00` on all six, morphology class preserved on all six. This is
  the concrete payoff of unit 11's switch from the process-salted `abs(hash(...))` seed to
  SHA-256 — those two datasets are exactly regenerable from version-controlled code.
- Morphology across the corpus: spots 225, labyrinth 139, stripes 49.

## 3a. The claim in §3 that is CIRCULAR, and must not be cited as evidence

"413/413 samples clear the generator's own no-pattern rule" is **very nearly a tautology.**
Every generator in the corpus applies the same `cv0 < 0.05 → reject` rule — `gen_tg3.py:241`,
and `gen_tg3.py:183-188` records that its simulator is "verbatim from the staging generator"
apart from the domain-size policy, so the rule produced the classical and staging families
too. A corpus filtered at 0.05 having a minimum of 0.0633 is what that filter does; it is
not evidence that the data is patterned.

The first draft of this branch reported it under "VERIFIED". That was wrong. What the check
actually establishes is the payload-integrity result above. **The informative screen is
`periodic` / `peak_bin`.**

## 4. The one real finding

**`three_gene_qvar/sample_0032` is not periodic.** In a domain of L = 177.8: **~95 % of
pixels sit within 1 % of the field's dynamic range above its minimum** (90.4 % lie within
1 % of the minimum *value* — an earlier wording conflated the two), species 1 and 2 are flat
to `cv = 0.002` (ranges 0.312–0.319 and 0.0311–0.0318), and its power spectrum decays
monotonically with **no interior peak** (`peak_bin = 1`, against a corpus minimum of 3
elsewhere).

Visually it reads as 3 blobs. The payload's stored `n_components = 5` counts wrap-around
fragments across the periodic boundary separately, and **nothing in this branch recomputes a
component count** — so "3 blobs" is an eyeball reading, not a measured quantity.

It is stored with `morphology = 'spots'` because `gen_tg3.classify` assigns `spots` on
`area_fraction < 0.34` **alone**, with no lower bound on area fraction or component count —
a condition an almost-empty field satisfies trivially.

Scale: 1/413 of the corpus (0.24 %), 1/34 of `three_gene_qvar` (2.9 %). Figure:
`experiments/figures_report/training_data/x_NOT_PERIODIC__three_gene_qvar__sample_0032.png`.

**Not acted on.** Excluding it would change what every `three_gene_qvar` number means; that
is the owner's call. See `D-TDPLOT-1`.

## 5. What is NOT verified — read this before citing anything here

- **No recovery result is produced or implied.** This inspects training *inputs* only. No
  model is fitted, no recovery metric is computed, and no number here is comparable to any
  recovery number.
- **"Patterned" is not "Turing-unstable."** This measures the former from the image. The
  latter is a statement about `sigma(k)` of the generating Jacobian and is not evaluated
  anywhere in this branch. CLAUDE.md §7c and `.claude/rules/reporting-numbers.md` require
  these be kept apart; they are.
- **7 of the 11 datasets were NOT re-simulated, and cannot be.**
  - `three_gene_train/test/val` (127 samples) store no `params_json`; their generator lived
    in gitignored `data/staging/` and their kinetics are unrecoverable from the payload.
  - `*_classical_*` (160 samples) store kinetics but **no seed**, and their sub-family
    generators (`schnak_cross`, `gm_relay`, `gm_two_inhibitor`, `schnak_relay`) are not in
    `rngrn.data.rd_models` — only 2-species `gierer_meinhardt` and `schnakenberg` are.

    For these families only the **stored** frames are shown. That is what training consumes,
    so the patterning claim holds; the *reproducibility* claim does not extend to them.
- **The periodicity screen is coarse.** It catches a spectrum with no interior peak, not a
  marginal pattern. `three_gene_classical_val/sample_0007` passes it (peak bin 4) while
  holding only 4 spots in the box; whether so few repeats is adequate for training is a
  separate question this branch does not answer.
- **The RAPS-vs-generator `k*` agreement is a consistency check, not a calibration.** It
  compares a measurement against ground truth for inspection purposes only.

  > **Do NOT calibrate the recovery `k*` tolerance from this figure.** `TUNING.md` carries
  > an uncalibrated `~15 %` k\* tolerance and this is exactly the number someone would reach
  > for to fix it. Deriving a tolerance that *judges recovery* from a comparison against the
  > **generating parameters** would be a circular threshold — recovery graded against ground
  > truth it is not allowed to see. Calibrate that tolerance from a control arm instead.

- **The periodicity threshold is descriptive, not calibrated against a control.** Its
  decision boundary has been exercised on exactly one real sample; `PEAK_BIN_MIN = 2` would
  partition this corpus identically. The known-answer controls are synthetic and live in the
  tests. See `D-TDPLOT-1` for the full list of limits.

- **`has_contrast` is not `patterned`.** `eval/rollout.py:272` owns `patterned` for a
  different quantity (rollout amplitude vs `max(1e-3, 0.02*|x*_0|)`), and that one is
  pre-registered. No number from this notebook may be read against a rollout `patterned`
  rate. The key here is named `has_contrast` precisely to keep them apart, and a test holds
  the separation.

## 6. Firewall position

`scripts/td_figures.py` is **data-side by construction** and must stay there. It reads
`payload.h5` — the generating kinetics via `params_json`, `x_star`, `D`, and the generator's
`k_star`/`sigma_max` — which is exactly what CLAUDE.md §5 forbids to recovery-side code. (It
does *not* read the stored `jacobian` or `interaction_matrix` datasets; an earlier draft of
this file said it did.) It is imported only by the notebook and by its own test. It is not
under `losses/` or `eval/`, and nothing in `model.py`, `observables.py`, `recover.py`,
`losses/` or `eval/` imports it — verified transitively with an AST walk over the full
import closure, not by grep.

**A latent structural gap this branch opened, and closed.** `td_figures` is the first
ground-truth-reading module that is (a) outside `src/rngrn/`, hence invisible to
`test_firewall.py`'s completeness glob, and (b) importable by **bare top-level name**,
because the notebook and `tests/test_td_figures.py` put `scripts/` on `sys.path` for the
whole pytest session. A recovery-side `import td_figures` would have resolved, reached
`params_json`/`x_star`/`D`/`k_star` in two lines, and passed every firewall test — because
the audit matches on names, and neither name was in the list. `td_figures` and `gen_tg3` are
therefore now in `FORBIDDEN` (`tests/test_firewall.py`). Nothing imports them today; the
entries make sure that stays true rather than relying on it.

The single thing it borrows from the package is `rngrn.observables.raps`, the image-only
spectrum estimator — the measurement side — so the `k*` drawn on a spectrum panel is
computed the same way recovery computes it.

## 7. Where to start if you pick this up

1. Open `notebooks/training_data_simulations.ipynb` — it runs top to bottom and its markdown
   states what each section does and does not establish.
2. Look at `experiments/figures_report/training_data/s0_corpus_patterning_summary.png` first;
   it is the whole corpus in three panels.
3. The prefixes: `g_` galleries (one per dataset), `d_` detail (species + spectrum),
   `t_` trajectories, `r_` re-simulation comparisons, `x_` flagged anomalies.
4. If you want to extend re-simulation to the classical families, the blocker is §5 above:
   their generators need porting into `rngrn.data.rd_models` and their seeds are gone.
