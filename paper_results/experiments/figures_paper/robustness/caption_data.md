# Caption data — Robustness benchmark (claim 4) + Lift-invariance (claim 2)

Every number that appears in `figures/1_robustness_benchmark.png` and
`figures/2_lift_invariance.png`, with its source path and the git commit that produced it.
Regenerate both figures with:

```
.venv/bin/python experiments/figures_paper/robustness/make_robustness_figures.py
```

---

## Figure 1 — `1_robustness_benchmark.png` (claim 4: ROBUSTNESS BENCHMARK)

### Sources

- **Population baseline** — `/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/exp11_robustness_baseline.csv`
  (main checkout, commit `5467f07`). 127 `three_gene` generator answer-key systems, 400
  independent log-normal multiplicative perturbation draws on every nonzero Jacobian entry
  and every diffusivity, per system, per noise level. Criterion: strict (`max Re eig(J) < 0`
  and an unstable mode at some `k > 0` on a 251-point k-grid over `[0, 4]`). Column read:
  `frac_strict`, grouped by `sigma`. Documented in
  `/home/benja/projects/personal/rngrn/rngrn-pipeline/docs/ROBUSTNESS_MEASUREMENT.md`
  (commit `4527c6a`) §4.2.
- **Recovered models** — `/home/benja/projects/personal/rngrn/rngrn-pipeline/experiments/c2_P_t8k8_consol/runs/*/results/train_results.json`
  (main checkout, commit `9f72e5d`), 16 runs (`nc1_m3_registry_20260803_{190648,192351}_seed{0..7}`).
  Fields read: `metric.turing_volume_10pct`, `metric.turing_volume_4p8pct`.
- **Pass bars** — `docs/PREREGISTRATION.md` §3.2 (this worktree, commit `52ef7dd`):
  median `turing_volume_10pct` ≥ 0.90, median `turing_volume_4p8pct` ≥ 0.95.

### Numbers

Population baseline, n = 127 systems × 400 draws, `frac_strict` (measured directly from
the CSV by this script; matches ROBUSTNESS_MEASUREMENT.md §4.2 to within CSV float rounding):

| perturbation level | mean | median | worst sample |
|---|---|---|---|
| 1 % | 0.993 | **1.000** | 0.537 |
| 4.8 % | 0.954 | **1.000** | 0.385 |
| 10 % | 0.879 | **0.935** | 0.233 |
| 20 % | 0.746 | **0.755** | 0.220 |

Recovered c2_P models, n = 16:

| metric | median | mean |
|---|---|---|
| `turing_volume_4p8pct` | **1.0000** | 1.0000 |
| `turing_volume_10pct` | **1.0000** | 0.9947 |

**Headline (panel b):** recovered median `turing_volume_10pct` = **1.000** ≥ population's own
median at 10 % noise = **0.935**. Both the population median (0.935) and the recovered
median (1.000) clear the §3.2 pass bar (0.90); the recovered models sit at the ceiling of
the population's own spread rather than merely above the bar.

**Pre-registered claim 4 pass condition:** median `turing_volume_10pct` (1.000) ≥ 0.90 ✓,
median `turing_volume_4p8pct` (1.000) ≥ 0.95 ✓, both measured over all 16 Turing-reaching
c2_P seeds (all 16 recovered a strict Turing model — see `recovered_turing: true` in every
`train_results.json`).

---

## Figure 2 — `2_lift_invariance.png` (claim 2: LIFT-INVARIANCE)

### Sources

- `experiments/lift_ladder/v4/results/v4.json` (this worktree, `feature/lift-ladder`,
  commit `fa7380f`, data unchanged through current HEAD `52ef7dd`; reviewed in T6 round 1,
  `docs/HANDOFF_lift_ladder.md` §V4 row). V4 is the band survey rung of the V0–V4
  validation ladder (`docs/DIAGNOSTICS_lift.md`, `docs/DECISIONS.md` D-LIFT-2): 27 models
  drawn (23 harvested generator-population systems + 4 learned `d5/seed{1,3,5,6}` models),
  each evaluated at 9 log-spaced `mu` points across the biological band plus the QSS
  (`mu`→0) point.
- **Report-only status:** `docs/PREREGISTRATION.md` §3.7 (D-LIFT-1, owner-delegated
  2026-08-17): the robustness-vs-`mu` row is *"Reported, never gated."* §3.6, which would
  have gated §3.2's bars on the lifted Jacobian at every `mu`, was proposed then withdrawn
  by the owner the same day, before any number existed against it — this figure is
  informational under §3.7, not a pass/fail claim.
- **Biological `mu` band** `[1.1e-5, 9.2e-3]`: `v4.json["mu_band"]`, set in D-LIFT-1
  (Chen et al. 2014 / Müller et al. 2012 timescale ratio), ratified in D-REDESIGN-1.

### The four learned models (`d5/seed1`, `d5/seed3`, `d5/seed5`, `d5/seed6`), n = 4

Perturbation-cloud Turing volume (`frac_turing`), QSS baseline → top-of-band (`mu` = 9.2e-3):

| model | QSS `frac_turing` | top-of-band `frac_turing` | absolute drop | relative drop |
|---|---|---|---|---|
| `d5/seed1` | 0.430 | 0.310 | 12 pp | 27.9 % |
| `d5/seed3` | 0.365 | 0.335 | 3 pp | 8.2 % |
| `d5/seed5` | 0.355 | 0.335 | 2 pp | 5.6 % |
| `d5/seed6` | 0.315 | 0.285 | 3 pp | 9.5 % |

**"lift-INVARIANT (volume cost 2–12 percentage points across the band)"** — the absolute
(percentage-point) drop from QSS to the top of the biological band ranges 2–12 pp across
the four models; in relative terms this is 5.6–27.9 %. The minimum `frac_turing` reached
anywhere within the band (not necessarily at the top edge — `d5/seed3` dips to 0.330 at
`mu` = 3.97e-3 before recovering slightly to 0.335 at the band top) is at most 3.5 pp below
its QSS value for every model except `d5/seed1`, whose monotone 12 pp drop is the largest
seen anywhere in the band for any of the four.

**These absolute volumes (0.285–0.430) sit below the §3.2 robustness bar (0.90).** The claim
made here is invariance of the local Turing volume under the finite-`mu` lift, **not**
absolute robustness — §3.2's bar is a QSS-Jacobian criterion calibrated against the 127-system
generator population (Figure 1) and is shown only for visual reference; it is not applied to
these lifted numbers per §3.7.

`kstar_mean` (surviving draws), QSS → top-of-band — essentially flat:

| model | QSS `kstar_mean` | top-of-band `kstar_mean` | relative change |
|---|---|---|---|
| `d5/seed1` | 0.2784 | 0.2731 | 1.9 % |
| `d5/seed3` | 0.2990 | 0.2938 | 1.7 % |
| `d5/seed5` | 0.2855 | 0.2838 | 0.6 % |
| `d5/seed6` | 0.2950 | 0.2885 | 2.2 % |

### Aggregate across all 27 QSS-Turing models in the V4 run

`v4.json["n_qss_turing"] = 27`, `v4.json["n_lifted_turing_band_all"] = 27`:

> **P(lifted-Turing across the band | QSS-Turing) = 27/27 = 1.000, Wilson 95 % lower bound
> 0.8754.**

Computed with the standard Wilson score interval (x=27, n=27, z=1.959964):
`center = (p̂ + z²/2n) / (1 + z²/n)`, `margin = z·sqrt(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)`,
`lower = center - margin`. This aggregate is over the **full 27-model V4 population**
(23 harvested generator-population models + the 4 `d5` learned models plotted here), not the
4 learned models alone — labelled explicitly on the figure to avoid conflating the two `n`s.
`v4.json["p_lifted_given_qss"] = 1.0` (point estimate, unrounded).

### Provenance

`v4.json["provenance"]`: git commit at run time, host, torch version, CUDA availability,
device name, date — carried in the JSON, not reproduced here as it does not appear on the
figure.
