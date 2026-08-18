# `lift_paper_demo` — a TRAINED model's recovered parameters pattern under the dynamical lift

The paper's centrepiece figure. Four **recovered** models — parameters that came out of
training, not out of a generator — are integrated as the **full lifted spatial system** and
form Turing patterns. Each is shown beside its matched QSS control.

Generated 2026-08-18 on branch `feature/lift-ladder`, code commit `52ef7dd`,
RTX 5070 Ti Laptop GPU, torch 2.13.0+cu130. Wall time **600 s** for the whole thing.

```bash
KMP_AFFINITY=disabled OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/lift_paper_demo.py --out experiments/lift_paper_demo
```

Defaults used: `--n 128 --seed 0 --mus 1e-3 7.2e-4 --seeds 1 3 5 6 --device cuda --dpi 220`.

---

## 1. What was run

**The models.** The 4 Turing-unstable D5 **recovered** models — seeds 1, 3, 5, 6 of
`experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed*`, the four `True` entries of
`analysis.json::turing_flags` — rebuilt from their tracked `checkpoints/model.pt` by
`scripts/lift_ladder.py::d5_models`. All four are `form=competitive`, N=3. The full run
directory and checkpoint path of each is recorded per row in `results/demo.json` under
`rows[i].provenance`.

**The box.** `L = 8 · 2π / k*_qss = 178.802` for all four (they share a k*), via
`lift_ladder.box_size`. p = 8 periods per box is the target `turing_labyrinth/sample_0000`'s
own geometry, at which SPEC §9.1's binding one-radial-bin tolerance `2π/L = 0.03514` is
exactly **12.50 % of k\*** — so k\* here is judged at the same relative precision as the
target's, not at one the box was chosen to make easy.

**The two arms**, matched structurally rather than by arrangement:

| | code | integrator |
|---|---|---|
| lifted | `eval/lifted_torch.simulate_lifted_torch` | Strang-split full lift, torch/CUDA |
| QSS control | `eval/rollout.simulate` | the same numpy ETDRK4 rollout every other rollout number in this project came from |

`lifted.step_policy` **is** `rollout.simulate`'s own growth-rate-aware policy evaluated on
the QSS Jacobian, and both sides draw the initial field from
`np.random.default_rng(0).standard_normal((N,n,n))` around the same x\*, with `noise=1e-2`.
The two runs therefore take the same dt, the same step count, and start from the
**bit-identical field**; the only difference between them is how the gates are handled. That
is what makes the field difference attributable to the lift.

**The two μ.** μ_gate = 1e-3 (owner decision D-REDESIGN-5) and μ_central = 7.2e-4
(`lifted.MU_BIO_CENTRAL`). Both are inside the biological band
[`MU_BIO_LO`, `MU_BIO_HI`] = [1.1e-5, 9.2e-3]; the script refuses an out-of-band μ rather
than running it silently.

`scripts/lift_paper_demo.py` is an **assembly script**: it adds no arithmetic. It mirrors
`ladder.v3_spatial`'s call pattern line for line and keeps the final fields, which
`v3_spatial` does not return. Its derived quantities come from the same primitives
`v3_spatial` calls — `observables.classify`, `observables.kstar_of` (inside the simulators),
`ladder.one_radial_bin`, `ladder._rel_l2`, `ladder._rel_l2_dev`.

---

## 2. Result — all 8 lifted runs patterned

Every run stopped at `horizon` (none truncated at `step_budget`, none blew up), so every
final field is a statement about the attractor rather than about a step budget.

| model | arm | patterned | class | k\* | \|Δk\*\| vs QSS sim | amplitude |
|---|---|---|---|---|---|---|
| seed1 | QSS | yes | labyrinth | 0.28060 | — | 0.1589 |
| seed1 | μ=1e-3 | **yes** | labyrinth | 0.28054 | 0.00006 (0.2 % of a bin) | 0.1587 |
| seed1 | μ=7.2e-4 | **yes** | labyrinth | 0.28054 | 0.00006 (0.2 %) | 0.1587 |
| seed3 | QSS | yes | spots | 0.27004 | — | 0.0198 |
| seed3 | μ=1e-3 | **yes** | spots | 0.27026 | 0.00022 (0.6 %) | 0.0198 |
| seed3 | μ=7.2e-4 | **yes** | spots | 0.27028 | 0.00024 (0.7 %) | 0.0198 |
| seed5 | QSS | yes | spots | 0.28708 | — | 0.1985 |
| seed5 | μ=1e-3 | **yes** | spots | 0.28707 | 0.00000 (0.0 %) | 0.1984 |
| seed5 | μ=7.2e-4 | **yes** | spots | 0.28711 | 0.00003 (0.1 %) | 0.1984 |
| seed6 | QSS | yes | spots | 0.28137 | — | 0.0206 |
| seed6 | μ=1e-3 | **yes** | spots | 0.28163 | 0.00026 (0.7 %) | 0.0206 |
| seed6 | μ=7.2e-4 | **yes** | spots | 0.28163 | 0.00027 (0.8 %) | 0.0206 |

Totals: **patterned 8/8**, **k\* within one radial bin 8/8** (against the simulated QSS
control *and* against the linear k\*_qss), **morphology label equal to the control's 8/8**.

Relative field difference `‖X_lift − X_qss‖_F / ‖X_qss‖_F` = 1.09e-2 … 5.58e-2. Under the
pattern-amplitude normalisation (`_rel_l2_dev`, the denominator that removes the uniform x\*
offset) the same difference is **5.10e-2 … 2.39e-1** — reported here because it is the
larger and less flattering of the two, and because §5.3 V3 marks the absolute bound
UNCALIBRATED either way. Neither is a bar; both are in `results/demo.json`.

`dt/μ` ranges 3.90–10.03. This is V3(a)'s **matched-dt** convention (dt from `step_policy`
with `dt=None`), not V3(b)'s `dt = min(0.2/jac_rate, μ/2)` policy — the same convention the
V3(a) campaign numbers were taken under, and the reason the two arms are directly comparable.

---

## 3. Files

- `results/demo.json` — every number above, per model per μ, with `provenance` (run dir,
  seed, checkpoint path) per row and git commit / host / device at the top level.
- `arrays/final_frames.npz` — the final fields of **both** arms, all channels, float32.
  Keys `d5_seed{1,3,5,6}__qss` and `d5_seed{1,3,5,6}__lifted_mu{0.001,0.00072}`, each
  `(3, 128, 128)`.
- `figures/demo_d5_seed{1,3,5,6}.png` — per model: QSS control | lifted μ=1e-3 |
  lifted μ=7.2e-4, channel 0, viridis, annotated with class, k\*, amplitude and verdict.
- `figures/demo_panel_all.png` — the combined 4-model panel (row 1 lifted at μ_gate, row 2
  QSS control). This is the figure for the paper.

`.gitignore` ignores `experiments/**/figures/` and `experiments/**/*.png`, so a scoped
exception was added for this one directory — the third such carve-out, alongside
`experiments/figures_report/` and `experiments/figures_paper/`, and on the same stated
grounds: the image is the deliverable, not a regenerable view of an array. 1.6 MB, 5 PNGs.
The `arrays/*.npz` was already tracked by the existing negation on line 48.

---

## 4. Scope — what this does and does NOT establish

**It is a pattern-formation demonstration at 128², and that is the claim.** What n = 128
supports is the amplitude verdict (the field cleared the pattern floor
`max(1e-3, 0.02·|x*₀|)`) and the k\* comparison against the control within one radial bin.

**The morphology-CLASS equality claim is licensed at 512² only** (REDESIGN §5.3 V3). Every
row here reports `morphology_claimable: false` by construction. The class labels in the table
and on the figures are **descriptive**; the class-equality evidence is the 512² campaign —
`experiments/lift_ladder/v3` (`d['full_rows']`, `morphology_agree` 4/4 on the
`morphology_claimable` subset), written up in `docs/DIAGNOSTICS_lift.md` §V3.

**"Turing-unstable" is not "patterns."** The first is a linear verdict on the dispersion
relation; the second is a property of an integrated field. The four models were selected on
the first (`analysis.json::turing_flags`) and this run measures the second. They are kept
distinct in every label here.

**This is not a recovery result.** It shows that these recovered parameters pattern under the
lift and that the lift does not move the pattern relative to the QSS reduction. It does *not*
show that the pattern matches the **target's** pattern — that is a `validate`-side comparison
against the observed frame, and nothing here touches it.

**Two of the four patterns are marginal.** seed3 (amp 0.0198) and seed6 (0.0206) clear their
pattern floors (0.01035, 0.01051) by only ≈1.9–2.0×, against ≈16× for seed1 and ≈20× for
seed5. The verdict is the same for all four; the margin is not, and a reader should see the
seed3/seed6 colourbars (spanning ~0.09) beside seed1/seed5's (~0.6).

**The morphology estimator is the coarse one.** `observables.classify`, the four-way
skew/angular-entropy rule — not `scoring/morphology.py::classify_morphology`, which needs a
labelled reference bank that this recovery-side code may not import. It has known limits:
seed1's field is labelled `labyrinth` while reading visually as inverted hexagonal spots.
That does not affect the statement actually being made, which is that **both arms receive the
same label**.

**The D5 population had never been through V3 before.** `docs/DIAGNOSTICS_lift.md` §0 records
the D5 four as appearing in **V1 and V4 only** — both *linear* rungs. The ladder's spatial
evidence (V3) is over the 23 `harvest` generator draws. This run is the first spatial,
dynamical measurement on the *recovered* population, which is exactly why it is the paper's
figure rather than a repeat of V3.

**k\*_qss is grid-dependent, and three numbers for it are in circulation.** All three are
argmaxes of the same continuous dispersion peak on different grids, and all agree to well
inside one radial bin:

| source | grid | k\*_qss | vs `box_size`'s value |
|---|---|---|---|
| `ladder.qss_verdict` (used by `box_size`, and so by this run) | `lifted.KGRID`, log-spaced | **0.281124** | — |
| `ladder.v1_continuation` → `DIAGNOSTICS_lift` §V1 | `default_kgrid`, linspace(1e-3, 50, 4000) | 0.28857 | 0.00745 = 21.2 % of a bin |
| `experiments/diag_fft/d5/analysis.json::kstar_model` | the training k-grid | 0.284755 | 0.00363 = 10.3 % of a bin |

`results/demo.json` records the value actually used as `rows[i].kstar_qss_linear`, and both
`kstar_abs_diff_vs_qss_linear` and `kstar_abs_diff_vs_qss_sim` are reported so neither
comparison has to be inferred.

---

## 5. Reproducing

`scripts/lift_paper_demo.py` reads model checkpoints only — **no `payload.h5`** — so it is
outside `tests/test_firewall.py::FORBIDDEN`'s remit (CLAUDE.md §5), and it is not a trainer,
sweep or target-report, so `.claude/hooks/guard_trainer.py` does not require
`scripts/guarded_run.sh` for it (§7a). At 128² the whole campaign is 600 s and one parent
process; if it is ever re-run at 512² that ceases to be true and it should go through the
guard.

Sanity gate for this branch: `pytest tests/test_lift_ladder.py -q` → **40 passed** (134 s,
unsandboxed, in-tree `.venv`).

---

## 6. Extension — domain-size independence

`domain_size/` (read its own README) answers the question this demo raises but does not
answer: does the recovered model pattern at the **same physical wavelength** in a box the
training process never constructed? p = 8 above is the training target's own geometry; that
subdirectory reruns the same 4 models at p = 4 and p = 16 (μ = 1e-3 only) and pulls this run's
p = 8 fields as the middle reference column rather than re-simulating them.

Headline: **24/24 runs patterned** (4 models × 3 box sizes × {QSS, lifted}), and every lifted
run stays within one radial bin of its own box's QSS control. k*, measured in physical units,
drifts 0.4–3.6% from the p=8 reference at the fine-lattice p=16 box and 3.5–12.7% at the
coarse-lattice p=4 box — the larger drift at p=4 is the mode-quantization effect CLAUDE.md §7c
describes (a coarser admissible-wavenumber lattice sits farther from the continuous unstable
peak), observed directly rather than assumed. No model failed to pattern at either new box
size.
