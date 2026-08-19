# PAPER CLAIM — form-robustness comparison (`nc1` vs `competitive`)

**Written for the paper, form-robustness unit (branch `feature/paper-form-robustness`,
based on `feature/paper-claim5-noise` @ `4515ea1`).** Structured the same way as
`worktrees/paper-pack/docs/PAPER_RESULTS_PACK.md`'s per-claim sections: headline numbers,
the run directory behind every number, figure file, and the caveats that must survive into
the prose. Every number here names a real run (CLAUDE.md §8); nothing is a dry-run figure.

**Why this comparison matters.** `docs/PREREGISTRATION.md` §0 states the claim under test
as holding "for **both** the competitive (Shea–Ackers) and non-competitive (`nc1`)
regulation forms". Every other robustness/pattern number in this paper's pack (claims 1, 4,
5) is measured on `nc1` only. This unit is the first and only measurement of the
`competitive` form's robustness against the same population baseline, same targets, and
same pre-registered §3.2 bars.

**Read `docs/DECISIONS.md::D-FORMCOMP-1` first.** It carries the full design and, most
importantly, the disclosed confound: the hyperparameters used for the `competitive` cells
here (`loss.weights.turing=8.0`, `loss.weights.kstar=8.0`) were tuned **for `nc1`**, not for
`competitive`. `competitive`'s own tuning campaign (`docs/C1_COMPETITIVE_TUNING.md`) never
swept a `kstar` axis and its own best cell used `turing=8` alone on different targets. The
result below happens to be favourable to `competitive`, which makes the confound *easier*
to misread as irrelevant — it is not: a fair form-vs-form claim still requires
`competitive`'s own `kstar` sweep, not run here.

---

## Headline

**At `nc1`-tuned `t8k8` hyperparameters, `competitive` reaches Turing instability and
clears both pre-registered §3.2 robustness bars on these two targets, at close to `nc1`'s
own level — with a real but modest gap that opens up only at the least-tuned, most
discriminating 20% perturbation level.**

- **Turing rate and recovery: identical, 16/16 both forms.** `turing_frac` = 1.000 (16/16)
  and `recovered_frac` = 1.000 (16/16) for both `nc1` and `competitive`, pooled across
  `sample_0001` + `sample_0004`. Zero seed errors on either side.
- **§3.2 bars: both forms pass, on both targets, pooled.**
  - `nc1` pooled: median `turing_volume_10pct` = **1.0000** (≥ 0.90 ✓), median
    `turing_volume_4p8pct` = **1.0000** (≥ 0.95 ✓).
  - `competitive` pooled: median `turing_volume_10pct` = **0.9800** (≥ 0.90 ✓), median
    `turing_volume_4p8pct` = **1.0000** (≥ 0.95 ✓).
- **The gap is concentrated at 20% perturbation, the only level with headroom to
  discriminate** (1%/4.8%/10% are all near a 1.000 ceiling for both forms and the
  population itself is at or near ceiling at 1%/4.8%). At 20%: `nc1` pooled median
  **0.9350**, `competitive` pooled median **0.8325** — `competitive` roughly a third of the
  way from `nc1`'s value down to the population's own median (0.755) at that level, still
  clearly above it.
- **k\*, morphology, plausibility: at parity.** Pooled median `kstar_fft_rel_err`: `nc1`
  0.0296, `competitive` 0.0305 (both well inside the 8.3% §3.3 bar, not scored here as a
  pass condition — see caveats). `morphology_match_frac`: `nc1` 15/15 (1.000, one seed not
  compared), `competitive` 16/16 (1.000). `plausibility_score` mean: 1.0000 for both.
- **`topology_consistency` (rtol 0.05): both far below the 0.75 §3.1 bar on both targets,
  and `competitive` is numerically higher, not lower** — `nc1` 0.125/0.125
  (`sample_0001`/`sample_0004`), `competitive` 0.250/0.250. Neither result should be read as
  informative about §3.1: both sit at the low end of the K=8 range documented as
  near-floor in `docs/C1_COMPETITIVE_TUNING.md` §7.2 / `docs/C2_NC1_TUNING.md` §5, and this
  unit ran no cross-target reproducibility control (that control needs a size-matched
  cross-target draw, out of scope here — see `docs/C1_COMPETITIVE_TUNING.md` §10.1 for how
  it is actually computed).

**So the honest headline is not "`competitive` underperforms `nc1`" — under this
hyperparameter recipe it does not, on the criteria that matter for §3.2. The honest
headline is "the two forms are close, with a real but small robustness gap at the least
tuned perturbation level, measured under hyperparameters that were never tuned for
`competitive`."** Whether that gap would close, widen, or reverse under a `competitive`-own
`kstar` sweep is not measured here.

## Figure

`experiments/form_compare/figures/1_form_robustness_comparison.png` (regenerate with
`.venv/bin/python experiments/form_compare/make_figures.py`; gitignored per repo convention
— the tracked deliverable is the script plus `experiments/form_compare/numbers.json`/
`numbers.md`) — population local-Turing-volume distributions at 1%/4.8%/10%/20%
perturbation, with both forms' pooled recovered models overlaid, and the §3.2 bars at
4.8%/10%. The 20% panel is the one that visually separates the three populations
(population < competitive < nc1); the other three panels show all three bunched near 1.0.

## Numbers

Population: `/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/exp11_robustness_baseline.csv`
(n=127 systems × 400 draws). `nc1`: `/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/c2_P_t8k8_consol/runs/*/results/train_results.json`
(n=16, committed, not re-run). `competitive`: `experiments/form_compare/comp_{0001,0004}/runs/*/results/train_results.json`
(n=16, this unit, produced 2026-08-19 13:02–14:23 BST).

| perturbation | population median (n=127) | nc1 pooled median (n=16) | competitive pooled median (n=16) |
|---|---|---|---|
| 1% | 1.000 | 1.0000 | 1.0000 |
| 4.8% | 1.000 | 1.0000 | 1.0000 |
| 10% | 0.935 | 1.0000 | 0.9800 |
| 20% | 0.755 | 0.9350 | 0.8325 |

| | nc1 sample_0001 / sample_0004 / pooled | competitive sample_0001 / sample_0004 / pooled |
|---|---|---|
| `turing_frac` | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| `recovered_frac` | 1.000 / 1.000 / — | 1.000 / 1.000 / — |
| `topology_consistency` (rtol 0.05) | 0.125 / 0.125 / — | 0.250 / 0.250 / — |
| `kstar_fft_rel_err` (median) | 0.0305 / 0.0202 / 0.0296 | 0.0404 / 0.0255 / 0.0305 |
| `morphology_match_frac` | 7/7 / 8/8 / 15/15 | 8/8 / 8/8 / 16/16 |
| `plausibility_score` (mean) | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |

(`topology_consistency` has no pooled value — it is a within-target statistic, computed
separately per target, not poolable across targets.)

Full per-seed values and both per-target and pooled breakdowns:
`experiments/form_compare/numbers.md` / `numbers.json`.

## Context, not a gate: the prior-difficulty asymmetry

Stage 0 measured the biological-box strictly-Turing acceptance rate at **2.5e-5** for `nc1`
against **4.5e-4** for `competitive`, over 80,000 box-constrained draws
(`docs/C2_NC1_TUNING.md` header) — an ~18× *harder* box for `nc1`. This is reported for
context, not used to adjust or excuse any number here: it says the two forms start from
differently-sized biologically-plausible regions, in `competitive`'s favour, which is the
opposite direction from the hyperparameter confound above (which was tuned for `nc1`). Read
together with the headline above, both a harder prior box for `nc1` *and* `nc1`-tuned
hyperparameters were in play, and `nc1` still comes out at or ahead of `competitive` at
every perturbation level measured (never behind). The two effects are not netted against
each other quantitatively anywhere in this document — there is no measurement that
isolates one from the other.

## What is and is NOT supportable from this unit

- **Supportable:** at the specific `t8k8` hyperparameter point (tuned for `nc1`), on these
  two targets, `competitive` reaches the Turing regime as reliably as `nc1` (16/16 both),
  clears both pre-registered §3.2 bars pooled, and matches `nc1` closely on k\*, morphology
  and plausibility. A real robustness gap opens at the 20% perturbation level, with
  `competitive` roughly midway between `nc1` and the population median.
- **NOT supportable:** that this establishes `competitive` as a robustness-equivalent
  regulation form in general. The comparison is confounded by hyperparameters tuned for the
  other form (D-FORMCOMP-1) and by only two targets, both from the tuning half
  (PREREGISTRATION §1a) and both already used by `nc1`'s `c2_P_t8k8_consol` — not a
  held-out or general-population claim for either form. A `competitive`-own hyperparameter
  sweep could move the 20% gap in either direction and was not run here.
- **NOT supportable:** any claim about criterion 3.1 (reproducibility) from the
  `topology_consistency` numbers above. Both forms are far below the 0.75 bar and this unit
  ran no cross-target control, which is the substantive half of 3.1
  (`docs/C1_COMPETITIVE_TUNING.md` §10.1).
- **NOT supportable:** `sample_0001` as an independent data point from
  `turing_labyrinth/sample_0000` used elsewhere in the paper — see caveats below. Only
  `sample_0004` is a genuinely independent second target, for both forms.

## Caveats to carry into the prose

- **The hyperparameter confound (D-FORMCOMP-1).** `t8k8` was tuned for `nc1`;
  `competitive`'s own campaign never swept `kstar`. The favourable result for `competitive`
  does not resolve this confound — it says this hyperparameter point happens to transfer
  reasonably well, not that it is optimal (or even good) for `competitive`.
- **Two independent targets, not three** — `sample_0001` is the same underlying system as
  `turing_labyrinth/sample_0000` at a different grid resolution (byte-identical Jacobian /
  `interaction_matrix` / `params_json` / `sim_seed`); only `sample_0004` is genuinely
  independent (`worktrees/paper-pack/docs/PAPER_RESULTS_PACK.md`, fact 1). This applies
  identically to both forms in this comparison.
- **Tuning-half data for both forms.** Neither side of this comparison is held-out evidence
  for a PREREGISTRATION §3 pass condition.
- **"Turing-unstable" and "patterns" are different claims** (CLAUDE.md §7c) —
  `turing_volume_*`/`turing_frac` are dispersion claims; `morphology_match` is a rollout
  claim. Both are reported above; neither substitutes for the other.
- **The §3.2 bars are reported per form, not newly gated.** They are pre-registered for the
  pipeline as a whole; this unit does not reinterpret PREREGISTRATION.md locally
  (CLAUDE.md §10) by inventing a per-form pass condition — it reports each form's numbers
  beside the existing bars.
- **`kstar_fft_rel_err` beside the §3.3 8.3% bar is reported for completeness, not scored
  as a pass/fail here** — this unit's design (D-FORMCOMP-1) commits only to the §3.2
  numbers as pre-registered reference points; both forms' medians (0.0296 / 0.0305) are
  well inside 8.3% regardless.

## Suite / provenance

Full suite run unsandboxed on this branch before push: **582 passed, 1 skipped**
(`KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`, ~4 min). No
`src/` changes were made in this unit — only `experiments/form_compare/` and `docs/`. Run
commands, exact seeds, and the config-equivalence verification (`m3_registry.yaml` vs
`nc1_m3_registry.yaml -o model.form=competitive`) are in
`experiments/form_compare/README.md`.
