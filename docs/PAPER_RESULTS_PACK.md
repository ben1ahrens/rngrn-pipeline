# PAPER RESULTS PACK — per-claim data, figures, and caveats

**Written overnight 2026-08-19 for the ~22:00 submission deadline.** The owner writes the
prose; this pack is the structured evidence behind each of the six claims: headline
numbers, the run directory behind every number, figure files, and the caveats that must
survive into the prose. Every number here names a real run (CLAUDE.md §8); nothing is a
dry-run figure.

**Provenance map — where each path lives:**

| prefix in this doc | branch @ commit | checkout |
|---|---|---|
| `LL/` | `feature/lift-ladder` @ `2f50fff` | `worktrees/lift-ladder` |
| `MAIN/` | `main` @ `48441e4` | `rngrn-pipeline` |
| `C3/` | `feature/paper-claim3-hidden-channel` | `worktrees/paper-claim3` |
| `C5/` | `feature/paper-claim5-noise` | `worktrees/paper-claim5` |

Figures are committed image files — the deliverable is the `.png` itself (scoped
`.gitignore` carve-outs, see `LL/.gitignore`).

**Read this first — three facts that constrain all six claims:**

1. **Two independent targets, not three.** `turing_labyrinth/sample_0000` (used by the D5
   diagnostic and the lift demo) and `three_gene_qvar/sample_0001` (c2_P target A) are the
   **same underlying system** — byte-identical Jacobian, `interaction_matrix`,
   `params_json`, `sim_seed` (1072191045); they differ only in simulation grid (512² vs
   96²). Evidence: `LL/experiments/figures_paper/topology/README.md:57-61` and its
   `caption_data.md` caveat 3. Only c2_P target B (`sample_0004`) is a genuinely
   independent second target. Do not count three systems anywhere in the prose.
2. **The c2_P campaign is tuning-half data.** Its targets sit in PREREG §1a's burned/tuning
   range (`sample_0000`–`sample_0007`). Any c2_P number is a tuning number, not held-out
   evidence for a §3 pass condition. The paper may report it as such; it may not present it
   as held-out.
3. **"Turing-unstable" and "patterns" are different claims** (dispersion vs rollout;
   CLAUDE.md §7c). `recovered_turing` / Turing-volume numbers are dispersion claims;
   `morphology_match` / "patterned" numbers are rollout claims. The prose must not use one
   as a proxy for the other.

---

## Claim 1 — Robust Turing patterns recovered from an N=3 training example

**Headline:** on the primary target, recovery succeeds on every seed — 8/8 runs
Turing-unstable with k\* matched to the observed spectrum to ~3% median relative error and
morphology matching the target on all 7 scored runs; across both c2_P targets, 16/16 runs
are Turing-unstable with all 16 under the §3.3 8.3% k\* error line.

**Figures** (`LL/experiments/figures_paper/recovery/figures/`):
- `1_representative_panel.png` — target frame vs recovered rollout (shared colour scale),
  RAPS overlay with k\* marked, dispersion σ(k) vs answer key. Run
  `nc1_m3_registry_20260803_190648_seed4`.
- `2_campaign_summary.png` — all 16 c2_P runs: per-run `kstar_fft_rel_err` against the
  §3.3 8.3% line; `turing_volume_10pct`/`turing_volume_4p8pct` against §3.2's 0.90/0.95
  floors.
- `3_d5_contrast.png` — D5 diagnostic: 10 seeds on one fixed target; 4 land on interior
  k\* (success, rel err 0.0136), 6 on the k-grid floor (failure, rel err 0.977). This is
  the honest picture of restart-selection variance and the reason recovery is reported
  best-of-64.

**Numbers** (sources: `LL/experiments/figures_paper/recovery/caption_data.md`,
`MAIN/experiments/c2_P_t8k8_consol/{runs.jsonl,target_reports.jsonl}`):
- Primary target (`three_gene_qvar/sample_0001`, 8 seeds): `recovered_turing` 8/8;
  `kstar_fft_rel_err` mean 0.0352 / median 0.0305; `morphology_match` 7/7 compared (seed3
  scored `target_only` — no model rollout scored); `plausibility_score` 1.0 on 8/8.
- Both targets (16 runs): median `kstar_fft_rel_err` **0.0296** (range 0.0053–0.0700),
  16/16 ≤ 0.083; `recovered_turing` 16/16; median `turing_volume_10pct` 1.000 (min
  0.940), median `turing_volume_4p8pct` 1.000 (all 16 exactly 1.000).
- `trivial_kstar_err` (PREREG §3.3: a k\* number without this beside it is not
  reportable): **0.250** for every target-A run, **0.000** for every target-B run
  (`LL/experiments/figures_paper/prereg_33/results.json`). On target B the trivial
  estimator hits the true k\* *exactly*, so the ≤ 8.3% clause there carries no evidential
  weight — the informative k\* evidence is target A's.
- D5 restart-variance contrast (10 seeds, one target, `kstar_fft_true` 0.28868): 4
  successes at `kstar_fft_rel_err` 0.0136; 6 failures pinned at the k-grid floor
  (`kstar_model` 0.00668, rel err 0.977).

**Run directories:** `MAIN/experiments/c2_P_t8k8_consol/` (16 runs; records committed
`9f72e5d`, finalized `4c176a3`; read at MAIN @ `48441e4`; the runs themselves executed at
code commit `4a61201` per their `train_results.json` provenance);
`LL/experiments/diag_fft/d5/` (10 runs).

**Tempering fact (must appear in the prose):** PREREG §3.3's full conjunction — morphology
match AND morphology distance ≤ the calibrated threshold AND k\* error ≤ 8.3% — passes
**7/16**, not 16/16. The threshold (pooled 75th percentile of within-class distances in
`three_gene_train` = **2.1072**, D-EVID-18) was computed 2026-08-18; all 8 runs of the
second target (`sample_0004`) fail the distance sub-condition *despite*
`morphology_match = True` (distances 2.74–3.80, 1.3–1.8× threshold). The split cleaves
along the two targets except for one target-A run (`..._190648_seed3`), which fails by
construction — no model-side field to compare (`morphology_scored = "target_only"`,
D-EVID-18) — making the tally 7 pass / 9 fail, not 8/8; every run with a *measured*
distance passes on target A and fails on target B. Source:
`LL/experiments/figures_paper/prereg_33/` and `LL/docs/DECISIONS.md` D-EVID-18.

**Caveats to carry:** tuning-half data (fact 2 above); best-of-64 restarts with non-uniform
restart selection (fig 3); the two targets are the tuning slots of a superseded primary
dataset (`three_gene_qvar`; canonical datasets superseded it per D-CANON-4 — not
withdrawn, but no new headline claims from it).

---

## Claim 2 — Learned GRN topology + parameters robust to perturbation under dynamical lift

**Headline:** all four recovered D5 models stay Turing-unstable across the whole
biological band under the finite-μ lift, and their perturbation-cloud Turing volume is
nearly lift-invariant: it changes by only 2–12 pp from QSS to top-of-band (relative drops
5.6–27.9%), surviving k\* is flat to 0.6–2.2%, and across all 27 QSS-Turing systems
surveyed, P(lifted-Turing | QSS-Turing) = **27/27 = 1.000** (Wilson 95% lower bound
**0.8754**, n=27). The supported claim is *invariance of robustness under the lift*, not
absolute robustness of the perturbation cloud — see the disclosure below.

**⚠ MANDATORY DISCLOSURE — the §3.2 bar is reinterpreted, by owner decision (D-PAPER-1,
`LL/docs/DECISIONS.md`).** Side by side:
- *Original pre-registered bar* (PREREG §3.2, verbatim): "Median `turing_volume_10pct`
  over Turing-reaching seeds ≥ 0.90, and median `turing_volume_4p8pct` ≥ 0.95."
- *Owner reinterpretation* (2026-08-18): for this claim, "robust" = **remaining in the
  Turing regime after perturbation, under the lift** (lift-invariance), NOT clearance of
  the absolute 0.90/0.95 bars by the lifted systems. PREREGISTRATION.md was not edited.
- *Measured values*: absolute perturbation-cloud Turing volumes are **0.285–0.430** —
  well below the 0.90 bar — **and the QSS volumes themselves (0.315–0.430) are already
  below it; the shortfall is not caused by the lift.** Per model, QSS → top-of-band
  (μ=9.2e-3): seed1 0.430→0.310, seed3 0.365→0.335, seed5 0.355→0.335, seed6
  0.315→0.285.
- *Consistency*: the standing prereg never applies §3.2's bars to lifted systems — §3.7
  (D-LIFT-1) makes the robustness-vs-μ row "Reported, never gated", and §3.6 (which would
  have applied them) was added and withdrawn on 2026-08-03 before any measurement. The
  absolute bars *are* passed where they were designed to apply (claim 4, unlifted).

**Figures:**
- `LL/experiments/lift_paper_demo/figures/demo_panel_all.png` — **the centrepiece**: 4
  recovered models, lifted (μ=1e-3) vs QSS-control patterned fields.
- `LL/experiments/lift_paper_demo/figures/demo_d5_seed{1,3,5,6}.png` — per-model: QSS |
  lifted μ=1e-3 | lifted μ=7.2e-4.
- `LL/experiments/figures_paper/robustness/figures/2_lift_invariance.png` —
  perturbation-cloud Turing volume + surviving k\* vs μ across the biological band
  [1.1e-5, 9.2e-3], §3.2 bars drawn for visual reference only.
- `LL/experiments/lift_paper_demo/domain_size/figures/domain_panel.png` — 4 models × box
  sizes p∈{4,8,16}, lifted fields.

**Demo numbers** (`LL/experiments/lift_paper_demo/results/demo.json`): all **8/8** lifted
runs patterned (forward from x\*+1e-2 noise at T=0 to horizon); k\* within one radial bin
of both QSS-sim and QSS-linear **8/8**; morphology label = QSS control's label **8/8**.
Relative field difference ‖X_lift−X_qss‖_F/‖X_qss‖_F = 1.09e-2–5.58e-2
(amplitude-normalised 5.10e-2–2.39e-1 — the less flattering number, report it too).

**Domain-size numbers** (`LL/experiments/lift_paper_demo/domain_size/results/results.json`):
**24/24** runs patterned (4 models × p∈{4,8,16} × {QSS, lifted}); all 12 lifted runs
within one radial bin of their own box's QSS control; k\* drift vs p=8: 3.5–12.7% at p=4
(mode quantization — the admissible-wavenumber lattice is coarse at small L), 0.4–3.6% at
p=16. The learned dynamics hold their physical wavelength across boxes the training never
saw.

**Survey numbers** (`LL/experiments/lift_ladder/v4/results/v4.json`): 27/27 conditional
(23 harvested generator systems + 4 learned models); Wilson 95% LB **0.8754** — use this
value; 0.874 appears in an older task report and is a documented arithmetic error
(`LL/docs/DIAGNOSTICS_lift.md` §4, "matches no standard z"); 2/27 systems are re-entrant
in μ; only 3/27 show a band edge at all in the scanned range (at 6.5×, 6.5× — the two
re-entrant systems — and 19.5× above the band top); the remaining 24 have `mu_crit = ∞`
(no edge found).

**Caveats to carry:** seed3 and seed6 are **marginal patterners** — amplitudes clear the
pattern floor by only ≈1.9–2.0× (vs ≈16–20× for seed1/seed5); morphology-CLASS equality is
licensed at 512² only — at n=128 every row has `morphology_claimable: false`, labels are
descriptive; the demo/lift morphology numbers use the coarse `observables.classify`
estimator and are **not comparable** to any `validate.morphology_match` number (firewall
consequence, D-LIFT-2); three k\*_qss grid conventions circulate (0.281124 demo /
0.28857 V1 / 0.284755 training), all within one radial bin — name which is in use;
domain-size panel runs μ_gate=1e-3 only, and p=16's points-per-wavelength sits exactly at
the ≥8 floor.

---

## Claim 3 — Recovery of a 3N GRN from partial observations that patterns robustly

**Headline (run overnight 2026-08-19; branch `C3` @ `f9d1514`, D-PAPER-2):** with the
**slow channel hidden** (species 0 of 3 — unambiguously the slow one: D = [1.0, 190.3,
133.2], ~130–190× slowest diffuser; μ = [0.776, 1.321, 2.595], slowest-reacting),
recovery still succeeds on the primary target: **8/8 seeds recover, 7/8 are
Turing-unstable, and morphology matches the target on all 7 runs where a rollout
comparison was possible**, with morphology distances 0.35–0.93 — all well under the §3.3
threshold (2.1072). The cost of hiding the channel, read against the matched
full-observation control (same objective, same seeds, same restarts): k\*-FFT error
roughly doubles (mean 0.0743 vs 0.0527, median 0.0814 vs 0.0503) and one seed (6) loses
Turing instability. All hidden-arm k\* errors (0.015–0.129) remain far below the 0.250
trivial baseline for this target.

**Design** (full rationale + tables in `C3/docs/DECISIONS.md` § D-PAPER-2 and
`C3/experiments/claim3_hidden_slow/README.md`): target `three_gene_qvar/sample_0001`,
seeds 0–7, three arms —
- (a) `replica_fullobs`: the exact c2_P winning invocation on current code (batched CUDA,
  64 restarts, resid=0). **Code-drift check: bit-identical to the `4a61201` baseline**
  (`kstar_fft_rel_err_mean = 0.03522518377119767` in both).
- (b) `ctrl_fullobs_resid` / (c) `hidden_slow`: a **matched serial pair** (the batched
  path refuses m<N) — CPU serial, 32 restarts, `resid=0.3`, differing ONLY in
  observation (m=3 vs m=2, `observed_idx=[1,2]`).

| | (a) replica | (b) full-obs control | (c) hidden slow |
|---|---|---|---|
| n_recovered / n_turing | 8/8 / 8/8 | 8/8 / 8/8 | 8/8 / **7/8** |
| k\*_fft rel err mean / median | 0.0352 / 0.0305 | 0.0527 / 0.0503 | 0.0743 / 0.0814 |
| k\* rel err mean | 0.0365 | 0.0287 | 0.0340 |
| morphology compared / match | 7 / 7 | 8 / 7 | 7 / 7 |

Run dirs: `C3/experiments/claim3_hidden_slow/{replica_fullobs,ctrl_fullobs_resid,hidden_slow}`.
Suite on the branch: 574 passed / 1 skipped.

**Verdict wording for the prose: supported but not cleanly attributable.** Two confounds
are inseparable in this design and must be disclosed:
1. `resid=0.3` is **UNCALIBRATED** and mandatory for m<N — the stationarity residual is
   the *only* loss term the latent fields enter (`recover.py` raises otherwise, noting
   "hidden-channel recovery currently has no known-good objective"), and exp06 previously
   measured this residual as harmful to Turing recovery. Arm (b) vs (a) quantifies part
   of that cost at full observation (0.0527 vs 0.0352 mean k\*_fft err).
2. Species 0 is also the **sole topological hub** (`interaction_matrix` row
   `[+,−,−]` — self-activating, inhibited by both others): "hidden slow channel" and
   "hidden hub" cannot be separated on this dataset.
Also disclose: 32 restarts in the pair vs 64 in the c2_P baseline (compute-bound, serial
path); `recovered_turing` (dispersion) and `morphology_match` (rollout) disagree within
arm (b) itself on seed 6 — keep the two claims distinct.

**Incidental finding (repo-level):** `main` @ `48441e4` fails
`tests/test_firewall.py::test_every_loss_and_eval_module_is_classified` (`eval/lifted.py`
unclassified, from the turing-training merge) — fixed on `C3` by classifying it
RECOVERY_SIDE; any branch off this base needs the same one-line fix to pass the pre-push
gate.

---

## Claim 4 — Benchmark model robustness

**Headline:** on tuning-half data (fact 2), the 16 recovered c2_P models sit at the
**ceiling** of the generator population's own robustness distribution: recovered median `turing_volume_10pct` =
**1.000** (mean 0.995) vs population median 0.935 at the same 10% perturbation, and
median `turing_volume_4p8pct` = **1.000** vs population median 1.000 (mean 0.954). Both
pre-registered §3.2 bars pass as written: 1.000 ≥ 0.90 ✓ and 1.000 ≥ 0.95 ✓ over all 16
Turing-reaching seeds.

**Figure:** `LL/experiments/figures_paper/robustness/figures/1_robustness_benchmark.png` —
(a) population local-Turing-volume distributions at 1%/4.8%/10%/20% with the 16 recovered
models overlaid and both §3.2 bars; (b) headline comparison at 10%.

**Numbers** (population: `MAIN/experiments/exp11_robustness_baseline.csv`, n=127 systems ×
400 draws, documented in `MAIN/docs/ROBUSTNESS_MEASUREMENT.md`; recovered:
`MAIN/experiments/c2_P_t8k8_consol/runs/*/results/train_results.json`):

| perturbation | population mean / median / worst | recovered (n=16) median |
|---|---|---|
| 1% | 0.993 / 1.000 / 0.537 | — |
| 4.8% | 0.954 / 1.000 / 0.385 | **1.0000** |
| 10% | 0.879 / 0.935 / 0.233 | **1.0000** |
| 20% | 0.746 / 0.755 / 0.220 | — |

(4.8% is Tica et al.'s measured experimental parameter CV — the bar's calibration basis,
PREREG §3.2.)

**Caveats to carry:** the reference class is the generator population itself ("systems we
are trying to be as good as"); recovered models come from the tuning half (fact 2);
recovered n=16 spans **2 independent targets** (fact 1).

---

## Claim 5 — Robustness of training to noisy training data

**STATUS: experiment running overnight.** Design (D-PAPER-C5 entry on
`C5/docs/DECISIONS.md` when it lands): an observation-noise knob added at the data
firewall gate (TDD'd `src/` change on `C5`): gaussian noise per observed channel at
relative level σ ∈ {0 (control), 0.01, 0.05, 0.2} × per-channel clean std, seeded and
recorded in `frozen_config.yaml` so every noisy frame is byte-identically regenerable
from the tracked dataset checksum + σ + seed. Same target (`three_gene_qvar/sample_0001`),
same seeds 0–7, same winning c2_P config per level. **No calibrated threshold exists; the
deliverable is the measured curve, level by level against the 0-noise control — no
pass/fail language.** Note the noise also perturbs `kstar_obs` (the k-grid anchor is an
FFT of the observed frame) — deliberate; that *is* noisy training data.

*(Section completed from the four cells' target_reports.jsonl before hand-off; a
degradation-to-failure at some level is reported as the curve, not as a failed bar.)*

---

## Claim 6 — GRN network-topology plots: edge weights + sign structure

**Headline (equivalence-class framing, owner-approved):** the objective recovers the
*phenotype*, not the *wiring diagram*. Across 26 fully-observed runs, **26/26 distinct
Jacobian sign structures** (structural rtol 0.05); sign agreement with the true network
sits **at chance** (pooled median `sign_match_frac` = 0.3333; uniform chance 0.3333;
label-shuffled null mean 0.3886; a constant all-repressing guess scores 0.444 — *above*
the observed median) — yet **19/20** Turing-converged runs pattern, with median k\* errors
of 3.1–6.5% per group. The learned models form an equivalence class that reproduces the
pattern while disagreeing with the true topology and with each other.

**Figures** (`LL/experiments/figures_paper/topology/figures/`):
- `fig4_nonidentifiability.png` — **headline**: per-target sign heatmaps + score-vs-chance
  + phenotype contrast.
- `fig1_d5_turing_topologies.png`, `fig2_c2P_topologies.png` — learned network diagrams
  (4 D5 Turing seeds; all 16 c2_P seeds).
- `fig3_learned_vs_true_target{A,B}.png`, `fig3b_sign_matrices_target{A,B}.png` — true
  network beside worst/median/best learned; the 3×3 sign matrices actually compared.
- Data: `LL/experiments/figures_paper/topology/data/topology_{runs,stats}.json`.

**Numbers** (verified against `topology_stats.json`): 26 runs = D5 10/10 + c2_P A 8/8 +
c2_P B 8/8 distinct sign structures at rtol 0.05; pooled median pinned `sign_match_frac`
0.3333 (per-group structural medians: D5 0.3333, A 0.2778, B 0.2778); 19/20 pattern given
Turing (D5 4/4, A 7/8, B 8/8); true topology `double_inhibitor` with 4 non-zero edges,
every learned network visibly denser. True sign vectors, at structural rtol 0.05:
`[+,−,−,+,−,0,+,0,−]` (targets A/D5), `[+,0,−,+,−,0,+,0,−]` (B — at rtol 1e-9 B's second
entry reads `−`, which is why the tolerance tag matters). The k\* medians here must be
read with their trivial baselines (PREREG §3.3, see claim 1): 0.250 on target A's grid
but **0.000 on target B's** — B's 6.5% median k\* error is *worse than the trivial
estimator*, which is exact there.

**Caveats to carry:** headline score is always the **pinned** score — the 6-permutation
score is an upper bound (fully-observed runs admit exactly the identity permutation;
quoting the 6-perm median would inflate 0.333 → ~0.67); two zero-tolerances circulate and
the looser (rtol 1e-9) flatters — use 0.05 for structure claims; this is an observation,
not a controlled identifiability experiment (no claim about *why*); every run here is
fully observed — the easiest case — and it already fails to identify topology; edge
magnitudes and truth-membership of the equivalence class were not measured; **2
independent targets, not 3** (fact 1).

---

## Suite state at pack time

`feature/lift-ladder`: 657 passed / 23 skipped recorded in the session ledger at the last
figure commit (topology unit's run); the branch's own tracked docs record the scoped
40-test lift-ladder gate for the demo/domain-size commits. `main` @ 48441e4 untouched by
this pack. Claim 3/5 branch suite counts land with their sections.

## NOT-SUPPORTABLE statements (final list lands with the overnight sections)

- **Cannot claim three independent recovery targets** — two of the three systems in use
  are the same system at different resolution (fact 1).
- **Cannot claim §3.3 full-conjunction success for the campaign** — 7/16 (claim 1
  tempering fact); morphology *label* match is 15/15 scored, but the pre-registered
  distance clause fails on the whole second target.
- **Cannot claim absolute §3.2-bar robustness under the lift** — lifted volumes are
  0.285–0.430 vs the 0.90 bar; the supportable lift claim is invariance (claim 2
  disclosure, D-PAPER-1).
- **Cannot claim topology/wiring recovery** — sign agreement at/below chance (claim 6);
  the supportable claim is the equivalence-class/phenotype one.
- *(placeholders resolved before hand-off: claim 3 outcome; claim 5 curve.)*
