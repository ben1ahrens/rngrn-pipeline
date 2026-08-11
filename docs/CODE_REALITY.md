# Code reality map — what is wired, what is a stub, what the docs get wrong

**Status: information, not instruction.** A map of the codebase as it actually is on
`feature/turing-training` @ `b4376a1`, written for someone who has not read it. Where a
docstring or a doc disagrees with the source, **the source is recorded here**.

> **This file is a SNAPSHOT and it has drifted before.** It was pinned to `d8070ca` while
> the branch moved on ~40 commits, and §1 ended up contradicting §11 of this same file.
> Re-pinned and corrected 2026-08-04. If you are reading it against a later HEAD, verify
> before believing it — `CLAUDE.md` and `docs/DECISIONS.md` are the maintained sources.

---

## 1. The single most important structural fact

**The library objective and the experiment objective are different code.**

**Corrected 2026-08-04.** The previous version of this section said the library "still
encodes the configuration that produced 0 % Turing" and that "nothing learned in exp02–exp10
has been promoted". **Unit 1 promoted it, and §11 item 6 of this same file already said so
250 lines below.** The source wins; both are now reconciled here.

- `src/rngrn/recover.py::recover` + `src/rngrn/losses/total.py` — the *library* path.
  Wired, tested, and now carrying the promoted objective: `turing_hinges_split`
  (`terms.py:186`, default `split_hinges=True`), the frame-scale anchor
  (`terms.py:237`, default weight 2.0), data-first staging (`staging_keys=["turing"]`),
  and the residual **off** (`resid=0.0`, settled by exp06, not merely untuned). Defaults
  live at `config.py:99-105`.
- `scripts/exp0[4-9]*.py::fit` — assembles its loss **inline**, and is what produced every
  result in `STATE_OF_THE_SCIENCE.md`.

What remains true, and is the actual reason this section exists:

1. **The scripts still assemble their loss inline.** A change to `losses/total.py` alone
   changes no number *in the scripts*, because they do not call it. The two paths agree on
   the objective today, but nothing enforces that they stay in agreement.
2. Consequently, a script result and a library result are comparable only if someone has
   checked the two assemblies still match. That check is manual.

---

## 2. Stubs and inert code

| thing | file | reality |
|---|---|---|
| `GradNormWeighting` | `losses/weighting.py:69` | **Corrected 2026-08-04:** raises `NotImplementedError` in `__init__`, naming the missing per-term gradient-norm estimator. It does *not* silently fall back, and it defines no `combine` method |
| `NTKWeighting` | `losses/weighting.py:87` | same — raises in `__init__`, names the missing NTK/convergence-rate estimator |
| `RatioWeighting` | `losses/weighting.py:94` | **implemented** (`strategy: ratio`), and mentioned by no other doc |
| `integrate_bdf1_newton_krylov` | `eval/numerics.py` | **stub** — delegates to ETDRK4, so the "independent stiff cross-check" does not exist |
| `morphology_consistency` | `losses/terms.py` | non-differentiable numpy diagnostic, **not in `compute_terms`** — the `loss.weights.morphology` knob is inert |
| `robustness_cloud` | `eval/analysis.py` | runs, but never validated, never run on a recovery, output never reaches the run index (see `ROBUSTNESS_MEASUREMENT.md` §3 for four measured defects) |
| in-pipeline 3-node reference systems | `data/rd_models.py` | absent — N=3 enters only via registry/HDF5, so new N=3 systems cannot be generated from this repo |
| a biological-plausibility score | `scoring/plausibility.py` | **Corrected 2026-08-04: implemented.** Box in `configs/bio_box.yaml`; run-index columns `plausibility_score` + `plausibility_*_in_box` (`validate.py:329`); aggregated as VIABILITY (`target_report.py:328-333`); matching training-time prior `terms.py::param_prior` (opt-in, `loss.weights.param_prior=0.0`). **Remaining caveat:** the box is centred on literature values, not the generators' — see `DECISIONS.md` D3 |
| Experiment B (N=2 truth → N=3 model) | `configs/expB_*.yaml`, `scoring/overparam.py` | **Corrected 2026-08-04:** the B arm (`N=3, m=2`) now *cannot* run under the default `resid: 0.0` — `recover.py:376` raises `ValueError`. It needs `-o loss.weights.resid=<nonzero>`, for which there is no known-good value (`TUNING.md:102`). Threshold uncalibrated; both configs pin `sample_key: sample_0000` |
| `integrate_etdrk4` fully-coupled variant | `eval/numerics.py` | diagonal linear operator only (`−D_i k²`), correct for diagonal diffusion; the `(J − k²D)` matrix-exponential form is not implemented |

---

## 3. Where the answer key is and is not allowed

`data/gate.py` is the boundary. It returns `(RecoveryInput, AnswerKey)`.

**What recovery legitimately sees:** `final_frame[observed_idx]` (m, 96, 96), the sample's
`L` attr, `observed_idx`; plus config (`N`, `m`, `form`, `n_hill`, `seed`, steps, lr,
weights) and its own random init.

**Derived inside recovery, from the image only:** `kstar_obs = observables.kstar_of(frame[0], L)`
(the anchor target), `obs_scale = frame.mean()` (the frame-scale anchor target),
`observables.laplacian_torch(frame, L)` (the PDE residual, when enabled).

**Quarantined in `AnswerKey`, consumed only by `validate.score_recovery`:**
`x_star`, `jacobian`, `D`, `k_star`, `k_star_fft`, `sigma_max`, `interaction_matrix`,
`trajectory`.

**One legitimate exception to know about:** `train.fit` holds the answer key and reads
`n_species_true` from it for **arm classification** (fully_observed / hidden_channel /
overparameterised / underparameterised). So "the answer key is read in exactly one
place" — as an older doc puts it — is imprecise: `score_recovery` reads its *contents*,
`fit` touches `n_species_true`, and `_resolve_recovery_input` is the actual firewall
boundary.

**The guard is weaker than the docs imply.** See `STATE_OF_THE_SCIENCE.md` §1 for the
empirical probe of what `tests/test_firewall.py` catches and misses, and for the `L`
exposure that is the real leak.

---

## 4. Sample data layout

`data/datasets/<dataset_id>/{payload.h5, manifest.json}`. The **directory name is the
dataset_id**. `payload.h5` is gitignored; manifests and `datasets.jsonl` are tracked.

A fresh worktree therefore *looks* provisioned — `rngrn list-datasets` lists all nine —
but the payloads are absent and the first load fails inside h5py. Symlink them from the
main checkout (~84 KB of links instead of ~91 MB of copies):

```bash
MAIN=/home/benja/projects/personal/rngrn/rngrn-pipeline
for d in "$MAIN"/data/datasets/*/; do n=$(basename "$d")
  [ -f "$d/payload.h5" ] || continue
  mkdir -p "data/datasets/$n" && ln -sfn "$d/payload.h5" "data/datasets/$n/payload.h5"
done
```

**Caution:** dataset manifests are *tracked*, so checking out a branch that predates
them **deletes them from disk**. `payload.h5` survives; regenerate with
`rngrn scan-datasets`.

Per-sample HDF5 members and attrs (`three_gene`; the two classical families lack
`interaction_matrix` and `trajectory`):

```
datasets: final_frame (3,96,96) · jacobian (3,3) · x_star (3,) · D (3,)
          interaction_matrix (3,3) · trajectory (6,3,96,96) · times (6,)
attrs:    L · k_star · k_star_fft · sigma_max · wavelength · morphology · topology
          reaction · n · grid · dx · dt_sim · anisotropy · area_frac · cv0 · n_components
```

Note `k_star` (linear theory) and `k_star_fft` (FFT of the image) are **different
attrs**. As of 2026-07-29 **`k_star_fft` is the validation target** and the analytic
`k_star` is the secondary diagnostic — this reverses the 2026-07-26 decision, so k\*
numbers recorded before that date are not directly comparable. Both already reach the run
index (`kstar_fft_rel_err`, `kstar_rel_err`); only which is the headline changed. See
`STATE_OF_THE_SCIENCE.md` §8.

---

## 5. Run index and scoring plumbing — two traps

**Rows are built by `dict.update()` merge, never `**`-expansion.** Scorers return keys
(`observed_idx`, `loss`, `n_true`) that collide with run-identity keys and raise
TypeError. Run identity wins on collision; non-scalar values are stringified because
rows must stay flat for the sqlite/jsonl index. New scorer keys flow through
automatically — do not re-add explicit kwargs for them.

**`np.asarray(None)` is a 0-d object array, not `None`** — so an `is None` guard silently
passes it through. `validate.py`'s missing-true-J guard therefore checks `ndim`/`size`
rather than identity, routing to `scoring_mode='no_true_J'`. Worth checking anywhere an
optional array field is guarded.

**Metric columns widen over time.** `kstar_fft_rel_err_mean` was added to `COLUMNS` and
`DEGRADATION_COLUMNS` once already. Any robustness metric would be the next addition.

**Spectral scoring is amplitude-blind, deliberately.** `observables.raps()` and
`scoring/morphology._spectral_block()` both subtract the field mean (annihilating the DC
bin, which by Parseval *is* the mean concentration) and `_spectral_block` then normalises
by the block sum. A 3× rescale of a field gives a self-distance of ~1e-11. The
concentration scale is instead used by the **frame-scale anchor**, where `frame.mean()`
is a ρ = 0.95 estimator of the true x\*₀. So amplitude is discarded for *mode* comparison
and exploited for *scale* anchoring — two different terms, by design.
`STATE_OF_THE_SCIENCE.md` §2.8.

---

## 6. Sweep axes: arm vs tuning

`optim/sweep.py` distinguishes experiment-**arm** axes from tuning axes.

- **Legal arm axes** (`_ARM_AXES`): `data.sample_key`, `data.dataset_id`,
  `model.observed_idx`, `model.m`, `model.N`.
- **Forbidden** (data *generation* params): `data.system`, `data.L`, `data.resolution`,
  `data.T_max`, `data.dt`, `data.seed`, `data.cache_root`, `data.datasets_root`,
  `data.source`. Varying these compares *different problems* rather than different
  methods.

`DataConfig.L_override` exists as the explicit, sweep-forbidden cross-check knob (added
when the hardcoded `L=100.0` bug was fixed).

---

## 7. Cost model — what is cheap and what is not

| operation | cost |
|---|---|
| one training step, N=3 | CPU **138 ms** / CUDA **418 ms** (GPU 3× slower per seed — launch-bound Newton) |
| `terms.steady_state` (damped Newton, ≤100 sequential 3×3 solves) | CPU 2.5 ms / CUDA 8.8 ms |
| one recovery seed at the 4000-step plateau | ~**26 s** |
| cubic dispersion, batch 60,800 (a val-split sweep) | CPU 14.1 ms / CUDA 1.0 ms → **14×** |
| `robustness_cloud` draw | ~**59 ms**, serial |
| `eval.rollout.simulate` at 96×96 | ~4.2 ms/step × ~128k steps = ~**9 minutes** per field (14 min at the 200k clip) |
| exp11 robustness baseline, 127 samples × 4 σ × 400 draws | ~60 s (vectorised numpy) |
| test suite | 557 passed + 1 skipped, ~2 min 55 s via the pre-push hook (re-measured 2026-08-11, sandbox disabled). Was 420 on 2026-08-04 — re-measure rather than trusting this cell |

The rollout figure is the one that bites: an earlier brief assumed ~1.9 s, off by three
orders of magnitude. Never roll out inside a per-run scoring path.

---

## 8. Environment and git mechanics

- **Use `rngrn-dev`** (the repo `.venv`) for all coding: torch 2.13.0+cu130, CUDA True,
  RTX 5070 Ti Laptop (sm_120, 12 GiB). `rngrn` (CPU-only torch 2.12.1 + a KMP_AFFINITY
  abort) is deprecated.
- **Each worktree needs its own `.venv`.** An editable install writes an
  `__editable__*.pth` holding the **absolute** path of the tree it was installed from, so
  reusing another tree's venv imports that tree's `src/` while you edit this one — tests
  silently exercise the wrong code. Verify:
  `.venv/bin/python -c "import rngrn,os;print(os.path.dirname(rngrn.__file__))"` must
  print a path inside the worktree. ~48 MB each with
  `python -m venv --system-site-packages .venv`.
- **Worktrees are siblings**, `../worktrees/<slug>`, never nested. The main checkout
  stays parked on `main`, which is what makes the new-branch-new-worktree rule
  self-enforcing: git refuses to create a worktree for a branch another worktree holds.
- **The authoritative test run is local**, via `.githooks/pre-push` (`git config
  core.hooksPath .githooks`, **once per repository** — local config is shared across
  worktrees since a worktree's `.git` is a pointer file). `git hook run pre-push` runs
  the suite without pushing.
- **GitHub Actions is unavailable** for this repo — the account's billing has lapsed, so
  runs are skipped with a check-run annotation and produce no logs. Do not diagnose those
  as code failures. `.github/workflows/tests.yml` is kept as a definition only.
- If `import torch` aborts with `OMP: Error #179` (containers), set
  `KMP_AFFINITY=disabled OMP_NUM_THREADS=1`. `tests/conftest.py` does this for the suite.

---

## 9. Branch state

**Rewritten 2026-08-11. The table that stood here was wholly obsolete** — it described `main`
as a bare template at `4509632` and routed the reader to five worktrees, *all five of which
have since been deleted*. Following it today produces `No such file or directory` on every
row. It is replaced by the one fact that now matters:

> **`main` is the trunk and contains everything.** It was fast-forwarded 137 commits to
> `e5d222a` on 2026-08-11. All the science that used to live on `feature/turing-training`,
> `feature/spatial-mode-recovery`, `feature/identifiability-experiments` and the rest is on
> `main`. **There are no worktrees** — `git worktree list` shows the single checkout.

Of 41 branches, **35 are fully merged into `main`** and are kept only as history; every one is
pushed to `origin` as an off-machine backup. Six carry commits `main` does not have, and all
six are deliberate:

| branch | why it is not on `main` |
|---|---|
| `feature/rngrn-c-mu` | PARKED — finite-mu machinery, explicitly incomplete and unvalidated |
| `chore/legacy-experiments` | stale run artefacts (a plumbing check + a noise-robustness **mockup**) parked off `main` on purpose |
| `chore/claude-md-reflow` | a WIP `CLAUDE.md` reflow based on a superseded version; merging it would revert real corrections |
| `chore/claude-harness` | holds the redundant commit dropped when `main` was fast-forwarded |
| `docs/agent-conventions`, `docs/hooks-config` | the original `CLAUDE.md` lineage; its content reached `main` by another path |

`CLAUDE.md` originated on `docs/agent-conventions` and now lives at the repo root on `main`.
That root copy — not the `docs/agent-conventions` one — is authoritative for environment, git,
testing, house style, the firewall, datasets, compute reality, evidence discipline, run
locations, the autonomy rule, and the subagent/workflow rules.

---

## 10. The two conventions most likely to be violated by accident

**Run outputs go to `experiments/<purpose>/`**, passed as `--runs-root` — e.g.
`experiments/dryrun/`, `experiments/tuning/`, `experiments/identifiability/`. The whole
`experiments/` tree is gitignored. Name the subdirectory for its purpose so a plumbing
check is never later mistaken for a result.

**The autonomy rule — SUPERSEDED 2026-07-29. See `CLAUDE.md` §10, which is authoritative.**
This section used to say "stop and ask whenever a science decision appears". The owner
withdrew that: science decisions are now **decided by the agent and recorded**, with evidence,
in `docs/DECISIONS.md`. Only two things still escalate — weakening a pre-registered pass
condition in `docs/PREREGISTRATION.md`, and anything outside the repo's technical scope.

The policy is deliberately **not** restated here. Duplicating it is exactly what let this copy
diverge from `CLAUDE.md` for a month.

Because the rule changed, the improvements this section used to list as blocked are no longer
blocked, and most have landed: `split_hinges` is promoted (`terms.py:186`, default on) and the
D-ratio prior is implemented (`terms.py:415`, opt-in). Only the **low-basal init** remains an
open decision — see `DECISIONS.md` D9 and item 5 below.

---

## 11. Open decisions currently blocking progress

Collected from across the docs so they are in one place. Struck-through entries have
been settled; the rest have not. **Full evidence and who decided each item, including
what was rejected, is in `docs/DECISIONS.md`** — this section stays a short index.

1. **Which reading of "more robust"** — `GOAL_tica_equivalent.md` §2, four incompatible
   options. **Still OPEN** — `DECISIONS.md` Part 3 item 1.
2. ~~**Domain-size design for regenerated data**~~ — **SETTLED (unit 11): option (b),
   vary L with random periods-per-box.** The user's later clarification that the model
   must generalise across domain sizes forecloses option (a) (fixed L), which makes L
   carry zero information and the generalisation claim untestable. Evidence and the two
   new datasets (`three_gene_qvar`, `three_gene_multiL`): `DATASETS_L.md`. Measured: the
   image-blind predictor `6·2π/L` goes from 0.0 % median error on the old 127 samples to
   45.5 % on `three_gene_qvar` (34 samples). The old sets are NOT deleted; they remain the
   comparison baseline. Full record: `DECISIONS.md` D6.
3. **The morphology pass condition** — stripes classifies at 33.3 % on held-out data
   (3 samples). Balanced accuracy excluding stripes / continuous `morphology_distance` /
   generate more stripes first. **Still OPEN** — `DECISIONS.md` Part 3 item 3.
4. ~~**D-ratio prior centre**~~ — **DECIDED for the bio_box prior: 7.5** (measured
   Nodal/Lefty), reversing an earlier same-session choice of ~100 (matches the
   generators). The ~15x gap between 7.5 and the generators' own ~135 median is
   recorded as a deliberate, unresolved tension, not a contradiction —
   `STATE_OF_THE_SCIENCE.md` §11, `configs/bio_box.yaml`. Full record: `DECISIONS.md`
   D1 and Part 3 item 4.
5. **Whether to adopt the low-basal init** (~~0 % → 82 % Turing-reachable at init~~ —
   **withdrawn 2026-08-04, `DECISIONS.md` D-EVID-11: the strict figure is 0 %, and what
   low_basal actually buys is the positive Jacobian diagonal, 28.6 % vs 0 %**) given
   that it biases which solutions are found. **Still OPEN as a default-flip decision**,
   but now with training-time evidence against it: PR 9 measured 0/40 restarts
   converging under actual training with `low_basal` vs a matched `default` run
   succeeding. Default stays `"default"`, unwired from the CLI path. `DECISIONS.md` D9
   and Part 3 item 5.
6. ~~**Whether `split_hinges` and the frame-scale anchor are promoted into the
   library**~~ — **SETTLED (unit 1): promoted.** Bit-identical to the validated
   originals (max diff 0.000e+00 over 20 seeds); library Turing fraction confirmed at
   38/40 converged, 14 Turing, turing_frac 0.3684 over 40 seeds x 400 steps. Full
   record: `DECISIONS.md` D8 and Part 3 item 6.
