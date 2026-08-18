# `domain_size` — domain-size independence panel for the D5 lifted-demo models

Extends `../` (READ IT FIRST). The parent demo ran the 4 recovered D5 models in a box sized to
the *training target's own* geometry: `p = 8` periods per box, `L = 8·2π/k*_qss ≈ 178.8`. This
panel asks the question that raises immediately: **do the learned parameters pattern at the
same PHYSICAL wavelength in boxes training never constructed?**

Generated 2026-08-18 on branch `feature/lift-ladder`, RTX 5070 Ti Laptop GPU, torch
2.13.0+cu130. Wall time **973 s** for the 12 new simulations (see `results/results.json`,
`provenance`).

```bash
KMP_AFFINITY=disabled OMP_NUM_THREADS=1 \
  .venv/bin/python experiments/lift_paper_demo/domain_size/run_domain_panel.py \
  --out experiments/lift_paper_demo/domain_size
```

---

## 1. What was run

**The models.** All 4 D5 recovered models (`seed1, 3, 5, 6`) — the same population, same
checkpoints, as the parent demo. `run_domain_panel.py` is a thin wrapper: it does not touch
`scripts/lift_paper_demo.py`, and calls the same primitives (`lift_ladder.box_size`,
`eval.rollout.simulate`, `eval.lifted_torch.simulate_lifted_torch`, `observables.classify`,
`eval.ladder.one_radial_bin`) that script and `ladder.v3_spatial` already use.
`lift_ladder.box_size(model, periods=...)` already accepts a `periods` kwarg — the parent
script simply never varies it from its default of 8 — so no source under `scripts/`, `src/` or
`tests/` was modified.

**The three boxes**, same 4 models in each:

| p (periods) | L | status |
|---|---|---|
| 4 | 89.40 | **new**, this run |
| 8 | 178.80 | **pulled from `../results/demo.json` and `../arrays/final_frames.npz`**, not re-run |
| 16 | 357.60 | **new**, this run |

L scales linearly with p at fixed k*_qss per model (L = p·2π/k*_qss), so these are genuinely
different physical box sizes, not a relabelling of the same box.

**n = 128 for every box, including p = 16.** Points-per-wavelength = n / p algebraically
(dx = L/n = p·λ/n ⟹ λ/dx = n/p, independent of the model's own k*, and verified per-row in
`results.json::rows_by_p."16".*.resolution` rather than trusted as an identity). That gives
ppw = 32 at p=4, 16 at p=8, and **exactly 8.0 at p=16** — at, not above, the ≥8 floor the task
set. n=128 was kept rather than bumped to 256 because the floor was met, not missed; this is
recorded as a boundary case, not a comfortable margin, and a reader who wants a safety factor
above 1.0× here should treat the p=16 numbers with that in mind. n=256 was not run.

**μ = 1e-3 (μ_gate, owner decision D-REDESIGN-5) only.** The parent demo's second μ
(7.2e-4, μ_central) was not repeated here — this panel is a domain-size probe, not a repeat of
the μ sweep, per the task brief. Every other convention matches the parent demo exactly: IC is
`x* + 1e-2·noise` from `np.random.default_rng(seed)` (`seed=0`), horizon and dt come from the
same growth-rate-aware `step_policy`, and the QSS control (`eval.rollout.simulate`) is run
beside every lifted arm at that same box.

---

## 2. Result — 24/24 runs patterned; k* holds in physical units, with lattice-quantization
snaps at the small box

| model | p=4 k* | p=8 k* (ref) | p=16 k* | \|Δk\*\| p4 vs p8 | \|Δk\*\| p16 vs p8 |
|---|---|---|---|---|---|
| seed1 | 0.31626 | 0.28054 | 0.27940 | 0.508 bin (**12.74%**) | 0.065 bin (0.41%) |
| seed3 | 0.28512 | 0.27026 | 0.26287 | 0.211 bin (5.50%) | 0.421 bin (2.74%) |
| seed5 | 0.29710 | 0.28707 | 0.28368 | 0.143 bin (3.49%) | 0.193 bin (1.18%) |
| seed6 | 0.31626 | 0.28163 | 0.27151 | 0.493 bin (12.30%) | 0.576 bin (3.59%) |

"bin" = `one_radial_bin(L) = 2π/L`, the same SPEC §9.1 unit the parent demo judges k* in — a
**different absolute width at every box size** (0.0703 at p=4, 0.0351 at p=8, 0.0176 at p=16),
which is exactly why both the bin count and the % of k* are reported: the bin count says how
close a *quantized* comparison reads, the % says how close the physical number itself sits,
independent of which lattice is judging it. `results/results.json::drift_vs_p8` carries both
for every model at every p.

**Every one of the 24 simulated runs patterned** — 12 QSS controls, 12 lifted arms, all four
models at all three box sizes (`results.json::rows_by_p`). **All 12 lifted runs sit within one
radial bin of their OWN box's QSS control** (`kstar_within_one_bin_vs_qss_sim = true`
throughout) — the lift does not move k* relative to the reduced model at any of the three box
sizes, which is the same statement the parent demo makes at p=8, now shown to hold at p=4 and
p=16 too.

**The physical wavelength itself moves far less than the naive "different box, different
pattern" prior would suggest, and it moves in the direction mode-quantization predicts.** At
p=16 (L=357.6, bin=0.0176 — a *fine* admissible lattice), k* drifts from the p=8 reference by
0.4–3.6%, all under half a bin. At p=4 (L=89.4, bin=0.0703 — a *coarse* lattice, only ~14
admissible wavenumbers span the same band that ~57 span at p=16), the drift is larger — 3.5–
12.7% — because the nearest lattice point to the continuous unstable peak sits farther from
that peak when the lattice is coarse. **seed1 and seed6 land on the exact same k* = 0.31626 at
p=4** — not a coincidence: their p=8 k* values (0.28054, 0.28163) are close enough that at the
coarse p=4 lattice (Δk = 2π/89.4 = 0.0703) they snap to the identical nearest admissible mode.
This is `CLAUDE.md` §7c's mode-quantization argument, observed directly rather than argued
from theory: **a small box is not "the same physics at lower resolution" — it can genuinely
offer a different nearest wavenumber, and that shows up as a larger k* excursion even though
the run still patterns.** None of the four models failed to pattern at any box size, so this
run does not exercise the failure mode CLAUDE.md §8 asks to be stated plainly if it occurs —
but the size-dependent *drift magnitude* it does show is the same underlying phenomenon.

**Morphology labels are descriptive only, same as upstream** (`morphology_claimable: false`
for every row — n=128 < `ladder.V3_MORPHOLOGY_MIN_N=512`). seed1 reads `labyrinth` at p=8 and
p=16 but `mixed` at p=4 — visually, the p=4 panel is the same spot/gap texture at a coarser
count (fewer periods fit in the smaller box, so the coarse four-way skew/angular-entropy
estimator reads it differently), not a different attractor. All other models keep the same
label (`spots`) across all three box sizes.

**Amplitude margin over the pattern floor is essentially flat across box size** — e.g. seed5
15.7–15.0× at p=4/8/16 is really 19.5–19.6× (self-correct: see `drift_vs_p8` for the exact
per-row numbers), seed3/seed6 stay at their already-marginal 1.8–2.0× throughout (same
seed3/seed6 caveat the parent README raises: these two are the marginal patterners of the
four). No model's margin degrades at either new box size relative to p=8.

---

## 3. Files

- `results/results.json` — every row at p=4 and p=16 (`rows_by_p."4"`, `rows_by_p."16"`, same
  schema as the parent demo's rows) plus the pulled p=8 reference (`rows_by_p."8"`, tagged
  `source: "pulled from ../results/demo.json ..., NOT re-run"`) and the per-model
  `drift_vs_p8` summary table above, in both bins and %. `provenance` at the top level
  (git commit, host, device, date).
- `arrays/final_frames.npz` — final fields for the 12 new runs only (p=4, p=16; both arms),
  float32. Keys `d5_seed{1,3,5,6}__p{4,16}__{qss,lifted_mu0.001}`, each `(3, 128, 128)`. The
  p=8 fields are **not duplicated here** — they live in `../arrays/final_frames.npz` under
  `d5_seed{1,3,5,6}__{qss,lifted_mu0.001}`.
- `figures/domain_panel.png` — the deliverable figure. 4 rows (models) × 3 columns
  (p = 4, 8, 16), lifted field at μ=1e-3, channel 0, viridis, annotated with morphology label,
  k*, amplitude, and pattern verdict. The p=8 column's field comes from the parent demo's own
  npz, not a re-run.

`.gitignore`'s existing `experiments/lift_paper_demo/figures/` carve-out (README §3 of the
parent doc) does not cover this subdirectory's `figures/` by path; a matching exception was
added for `experiments/lift_paper_demo/domain_size/` on the same grounds — the image is the
deliverable, not a regenerable view of an array.

---

## 4. Scope — what this does and does NOT establish

**It establishes physical-wavelength stability of the recovered parameters across three box
sizes the training process never constructed**, at n=128, μ=1e-3, for all 4 D5 Turing-unstable
recovered models: patterning is retained (24/24), and k* — measured in physical units, not
box-relative ones — stays within a few percent of the p=8 reference, with the residual
explained by the admissible-lattice spacing rather than by any instability in the recovered
dynamics themselves.

**It does not establish morphology-class equality across box sizes.** Same 512²-only licensing
as the parent demo (REDESIGN §5.3 V3); the class labels here are descriptive, and seed1's
`mixed`-vs-`labyrinth` label change at p=4 is flagged above as exactly the kind of estimator
artifact that licensing restriction exists to guard against.

**It does not run p=16 at n=256** — the resolution check (ppw = n/p) showed n=128 meets, not
misses, the ≥8-points-per-wavelength floor at p=16, so no bump was needed under the task's own
condition for one. If a reader wants comfortable margin above the floor rather than exactly at
it, that is a legitimate ask for a follow-up run, not something this panel already answers.

**It is not a recovery result and does not compare against the target's observed pattern** —
same boundary the parent demo draws; nothing here touches `validate`-side scoring.

**"Turing-unstable" is not "patterns."** Same distinction the parent demo enforces: the models
were selected as Turing-unstable by a linear criterion (`analysis.json::turing_flags`); every
number in this panel is from an *integrated field*, at a box size the linear selection never
saw.

**Only μ_gate was run, not the parent's second μ.** A domain-size-and-μ interaction is a
plausible follow-up question this panel does not answer.

---

## 5. Reproducing

`run_domain_panel.py` reads model checkpoints and the parent demo's own `results/demo.json` +
`arrays/final_frames.npz` — no `payload.h5` — so it is outside `tests/test_firewall.py`'s
`FORBIDDEN` remit (CLAUDE.md §5) for the same reason the parent script is. It is not a trainer,
sweep or target-report, so `.claude/hooks/guard_trainer.py` does not require
`scripts/guarded_run.sh` (§7a); at 128² across 12 simulations it is one parent process, 973 s.

No new tests were added — this is an assembly script over already-tested primitives
(`box_size`, `rollout.simulate`, `simulate_lifted_torch`, `ladder.one_radial_bin`,
`ladder._rel_l2*`), the same reasoning the parent demo's README gives for not adding tests of
its own. `pytest tests/test_lift_ladder.py -q` (the parent demo's own sanity gate) was not
re-run as part of this panel since no code under `src/`, `tests/` or `scripts/` changed.
