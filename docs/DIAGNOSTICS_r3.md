# DIAGNOSTICS — R3 gradient machinery (Phase-D measurements, collated)

Written 2026-09-01 (R3 Task 21) on `docs/r3-docs-pass`, collating the Phase-D
measurement tasks 14–17 of `PLAN_redesign_R3.md` (branch `docs/redesign-rngrn`) as run
on `feature/r3-integration` and its unit branches. House pattern of
`DIAGNOSTICS_fft.md`: per-measurement method, measured numbers with run paths,
PASS/FAIL/finding, and an explicit "what this does NOT establish" section. Every number
below traces to a tracked artifact under `experiments/redesign_r3/` or to the decision
entry cited beside it; the task-level detail (methods in full, fix rounds, self-reviews)
lives in the SDD reports on the `redesign-doc` worktree and in `docs/DECISIONS.md`
D-R3-1..8. **Nothing here is a training result** — these are gradient-machinery
diagnostics; recovery quality on real data is R4's question.

## FD A/B on both gradient paths (Task 14)  [status: **DONE** — unrolled PASSES both arms; adjoint PASSES converged, FAILS the forced-stall arm at O(1)]

**Method.** D1-style finite-difference check (10 directions × active spectral terms,
tol 1e-4) on both gradient paths at the same θ, two arms: a converged member (Newton
3.447e-13, bar 1e-9) and a FORCED stalled member (Newton capped at 1, achieved residual
1.786e-03 — inside D-FFT-10's measured stall band [1e-4, 1e-2]). A natural stall was
hunted and not found at this operating point (`hunt.found = false`).
Artifact: `experiments/redesign_r3/fd_ab/results/fd_ab.json` (+ `scripts/r3_fd_ab.py`).

| Arm | Path | Worst rel. err | Gate (tol 1e-4) |
| --- | --- | --- | --- |
| Converged | adjoint | 1.70e-06 (spec_shape) | PASS |
| Converged | unrolled | 1.92e-08 (spec_aniso) | PASS |
| Stalled (forced) | adjoint | **1.93** (every term O(0.2–1.9)) | **FAIL** |
| Stalled (forced) | unrolled | 1.44e-08 (spec_aniso) | PASS |

**Cost** (fresh spawned process, threads=1, reps=3): B=1 unrolled 1.544 s/member vs
adjoint 4.562 — **2.95× cheaper**, the load-bearing row. The B=8 rows (unrolled 2.084
s/member vs adjoint 104.3) carry two caveats stated in the report — partial CPU load
overlap and a shared-warm-state harness bias that penalises only the adjoint — so the
50× figure is not to be quoted standalone. Memory is the trade: +433–516 MB RSS for the
checkpointed segment where the adjoint pays ~nothing.

**Findings.** (1) The two-path design's prediction held exactly: the adjoint's premise
is F(u*)=0 and handing it F(u*)≈2e-3 produced O(1)-wrong gradients on every term, while
the unrolled path — which needs no convergence — stayed FD-faithful at 1.44e-08 from
the same displaced state. (2) Stated asymmetry: the stalled row tests the adjoint at
1.786e-03 but the unrolled path at warm residual 46.77 (it relaxes the displacement
away itself); it is not a like-for-like "at 1.8e-03" comparison. (3) The forced stall carries the report's own construction caveats in full: the
displaced state has the wrong residual *shape* relative to a natural stall, and the
unrolled PASS on that row is a re-measurement from a displaced warm state — it must not
be read as a survived stall. (4) The unrolled path's FD PASS rows (1.92e-08, 1.44e-08)
are near-circular by construction: FD differences the same unrolled computation the
analytic path backpropagates through, so they certify internal consistency more than
independent correctness. (5) This measurement plus the cost row is the evidence
D-R3-5's owner-ruled promotion reads (unrolled → primary/default; adjoint retained as
the A/B verification path).

## 96²-vs-512² RAPS band fidelity (Task 15)  [status: **DONE** — with the paired-IC design, 96² clears the D3 floor on every seed of every fixture; the unpaired headline is SUPERSEDED (D-R3-4)]

**Method.** Radially averaged power spectra compared across grids n ∈ {96, 128, 256}
against the same model's 512² reference, %-per-bin in the training band. The PAIRED
design (D-R3-4) starts coarse and fine rollouts from the same downsampled IC, removing
the pattern-selection confound the original unpaired sweep measured. 3 fixtures ×
8 seeds × 4 grids = 96 rollouts.
Artifact: `experiments/redesign_r3/raps_fidelity/results/raps_fidelity_paired.json`
(unpaired sweep retained in `raps_fidelity.json`).

| fixture | n=96 paired (mean/worst of 8 seeds) | n=96 unpaired (contrast) |
| --- | --- | --- |
| `tune_comp_seed3` | 13.37 / 16.78 % per bin | 34.74 / 51.74 |
| `d5_seed3` | 4.04 / 9.38 | 43.51 / 68.29 |
| `d5_seed5` | 0.05 / 0.08 | 44.99 / 64.08 |

**Findings.** (1) Against D3's ~31 %/bin floor, every grid down to 96² clears on every
seed of every fixture (worst single reading anywhere at n=96: 16.78 % — 1.85× margin);
the paired grid trend is monotone in every fixture, which it was in none unpaired.
(2) The IC confound dominated the unpaired sweep by 21.4/39.5/44.9 percentage points at
n=96 — pattern-selection variance, not discretisation. Both readings stand as answers
to different questions: grid fidelity at 96² is fine; independent solves of the same
model land 34–45 %/bin apart. (3) The training band holds the same 7 bins at every n
(`n_band_bins_by_grid` = [7] throughout): band resolution is a `p` question, not an `n`
question. n=96 was subsequently ruled to stand for Phase II (D-R3-6, under the owner's
delegation). (4) The IC-variance finding is a design lever: averaging over ICs, not
raising n, is what reduces it. (5) Caveats that travel with every number here: this is
a model-side self-consistency test — the fixture's own 512² rollout plays "the observed
frame" and `k_lin` plays `k*_obs`, with no observed data anywhere in the loop — and the
fixture base is 3 models in 2 `k_lin` clusters from one N=3 config family.

## Population stall-rate distribution (Task 16)  [status: **DONE** — pooled 25.7 % off-checkpoint, per-member 0–100 %; `stall_switch_fraction` RETIRED as a threshold (D-R3-7)]

**Method.** 12 members (2 per perturbation scale × {0.1, 0.3, 0.6, 1.0, 1.5, 2.0}
around the known-Turing checkpoint), up to 6 training-condition spectral solves each;
count Newton polishes missing the 1e-9 bar.
Artifact: `experiments/redesign_r3/stall_rate/results/stall_rate.json`
(+ `scripts/r3_stall_survey.py`).

**Findings.** (1) 35 ignited-solve attempts, 9 stalled — pooled 25.7 %; per-member
rates are strongly heterogeneous (min 0.0, median 0.167, p90 0.72, max 1.0 over the 8
members with ≥1 solved step) and rise with distance from the checkpoint (0 % at scale
0.1 → 60 % at 2.0; small n at the top scale). A mean misrepresents this — the spec's
distribution framing was right. (2) Real stalls are NEAR-misses: residuals
[1.22e-07, 1.36e-05] (median 8.4e-07) — 2–4 orders above the bar, versus Task 14's
forced stall at 1.786e-03 and converged solves at ~5e-13. The two populations are
cleanly separated; nothing was observed near the bar itself. (3) Consequence, ruled:
`stall_switch_fraction` is retired as a threshold — it never gated anything, and the
promotion answered the routing question by moving every member to the unrolled path
rather than picking a rate; the field survives as a recorded diagnostic only (D-R3-7). (4) The 25.7 % is NOT
bit-comparable to a production run's `n_stalled/n_ignited`: D-R3-7's "Two divergences"
records a more-permissive ignition gate and a different denominator in this survey; and
the survey ran at the n=96 grid, so the rates' transfer to any other grid is
unmeasured.

## Phase-II B/K cost curve, forward solve in the loop (Task 17)  [status: **DONE** — wall-clock, not memory, bounds B; evidence RETURNED, B/K deliberately un-ruled (D-R3-8 PROPOSED)]

**Method.** One full Adam-step cost per (arm, B) cell — batched ETDRK4 relax + per-member
Newton + backward — fresh spawned process per cell, threads=1, one step, no warm-up
(a real R3 step is fresh-relax by design), incremental save, 3600 s per-cell wall-clock
valve, §7a predictive+empirical memory guard.
Artifact: `experiments/redesign_r3/batch_curve/results/batch_curve.json`
(+ `scripts/r3_batch_curve.py`).

| B (unrolled) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| member-steps/s | 0.130 | 0.100 | 0.093 | 0.083 | 0.045 | 0.037 | 0.028 | timeout |
| s/step | 7.67 | 20.05 | 43.22 | 96.86 | 356.7 | 867.9 | 2298.5 | >3600 |

**Findings.** (1) The binding constraint is WALL-CLOCK, not §7a memory: the guard never
fired (`refused` empty), MemAvailable never fell below 13,008 MB against the 8192 MB
floor, peak RSS topped out at 3257 MB at B=64 — while an Adam step at B=64 already
costs 38.3 minutes at 1 thread and B=128 exceeded the 3600 s valve. Largest measured
B = 64. (2) Per-member cost is SUPERLINEAR in B (0.130 → 0.028 member-steps/s across
B=1→64, ~2.2–2.6× per doubling — noisy: the single B=8→16 doubling measured 3.68×),
because the unrolled backward leg is serial per member
— which also means raising B buys less than proportional throughput, strengthening the
IC-diversity-per-step reading of what B is for (framed under D-R3-4's variance finding;
not ruled). (3) Adjoint cross-check at B ∈ {1, 8}: unrolled is 4.9× / 4.5× faster per
member-step with the forward solve in the loop — consistent in direction with D-R3-5's
2.95× per-member-step estimator figure (which its rider 1 marks as not a whole-step
number — the retained polish was unmeasured there); the 4.9×/4.5× here is the first
full-step measurement of the gap. (4) B=512 / K=16 remain UNCALIBRATED (register item 14): the
evidence is returned to the controller/owner un-ruled; B=512 unrolled at 1 thread
projects (projection, not measurement) to hours per step on this host.

## What this does NOT establish

- **Adjoint gradient error at a MILD miss.** Task 14 measured O(1) failure at a forced
  residual ~1.8e-03; Task 16's real stalls sit at ~1e-6. Whether the adjoint is also
  wrong, or degrades gracefully, at the near-miss regime is UNMEASURED — a scoped R4
  gap, on the record in D-R3-5's riders.
- **B and K.** Returned, not ruled (D-R3-8 PROPOSED). Nothing here licenses B=512 or
  K=16; the B/K ruling goes to the owner with Task 22's roll-up.
- **K's distinct-structure ceiling under R3's own objective.** The only counts in
  evidence (3 collapsed structures; 44–62 in the legacy control) are R2's
  ignition-only run, cited-not-remeasured — no R3 population has been run to
  convergence.
- **A batched spectral run under the promoted default.** `gradient_path='unrolled'` is
  serial-only; `batched=True` + non-zero spectral weight REFUSES until a batched
  unrolled twin exists (T12's known gap). A batched spectral run today must select
  `gradient_path='adjoint'` explicitly, accepting the demoted estimator, and say so.
- **Generality of the no-blow-up finding.** D-R3-3's contraction argument is one
  fixture, one commensurate box, one seed, two placeholder losses — not a claim the
  unrolled path cannot explode elsewhere.
- **The D3 floor's own basis.** The ~31 %/bin floor is a spatial-subsampling estimate
  from quadrant patches of one 256² frame; whether it is the right bar for band
  fidelity remains open (Task 15's report, unresolved by the paired design).
- **Single-measurement noise.** T17 is one step per cell with no error bars (its B=8→16
  doubling is the single noisy point restated inline above, and the B=32/64 RSS jump above the
  linear model is flagged OPEN); T14's adjoint B=8 cost row is confounded and not
  quotable standalone.
- **Any recovery-quality claim.** No task here scored a recovery against a control on
  real data; the harness runs, the gradients are characterised, and that is all.
