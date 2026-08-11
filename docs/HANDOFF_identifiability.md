# Handoff — identifiability experiments branch

**Branch:** `feature/identifiability-experiments` · **Base:** `main` (merged in).
For an agent or collaborator picking this up cold. Read
[IDENTIFIABILITY_EXPERIMENTS.md](IDENTIFIABILITY_EXPERIMENTS.md) first for the *science*;
this file is the *engineering* state: what changed, what is verified, what is not, and
where to start.

## Status in one line

The harness for both experiments is built and wired. **Corrected 2026-08-04: it no longer
dry-runs for all four arms.** `expA_hidden_channel` and `expB_overparam` are both `N=3, m=2`,
and `recover.py:376` now raises `ValueError` when `m < N` and the stationarity residual has
weight 0 — which is the default (`configs/base.yaml:33`, `resid: 0.0`). Only the two control
arms run as written; see the corrected recipe below. (557 passed, 1 skipped, re-measured
2026-08-11; was 420 when this was written.) **Nothing is tuned and no scientific
result has been produced** — the dry run uses 6 Adam steps and recovers nothing meaningful.
That is expected and correct for this stage; do not present dry-run numbers as findings.

### Update 2026-07-26 — first real-data dry run

The repo moved to `~/projects/personal/rngrn/rngrn-pipeline` (a parent directory that will
hold sibling worktrees). `main` was merged into this branch, so the local dev setup — the
`.githooks/pre-push` test hook, the tracked dataset manifests, `configs/m3_registry.yaml` —
is present here too.

Three things were verified or fixed:

- **Experiment B could not run at all.** Both B configs referenced `dataset_id:
  two_gene_val`, which is not registered; `gate.from_registry` raised `FileNotFoundError`.
  The registered id is `two_gene_classical_val` (a dataset's directory name *is* its id).
  Fixed in both configs and in `docs/LOCAL_DATA_SETUP.md`, which had the same wrong name.
- **All four arms now load real data and route to the right scorer.** Verified:
  A-control and A-hidden → `permutation_aligned` with a 3×3 true J; B-control →
  `permutation_aligned` 2×2; B-overparam → `overparameterised`, `n_true=2, n_model=3`,
  `sign_match_frac=NaN` by design, spare-species metrics emitted. `subblock_sign_match` is
  populated in every arm, as intended.
- **`n_permutations_searched == 1` for Experiment A, as caveat 1 predicts.** Confirmed on
  real data, not just in the unit test.

**~~Open defect — `kstar_rel_err` is NaN in every arm.~~ RESOLVED 2026-08-04 (D-EVID-7,
D-EVID-11..15). Corrected here 2026-08-11.**

The defect was real and is fixed. The attribute name was the bug: payloads store `k_star`,
not `kstar`, and `gate.from_registry` now reads `_require_attr(attrs, "k_star", where)` —
`gate.py:226-229` carries a comment specifically warning not to "simplify" the name back.
Both quantities are now read verbatim from the sample and both are scored:
`kstar_fft_rel_err` against the FFT-measured k\* is the **headline** (owner decision), and
`kstar_rel_err` against the linear answer-key k\* is **secondary** — see `validate.py:15-39`.
Neither is NaN.

The science decision the rest of this section says must be made **has been made**: the
headline is the FFT-measured reference, with the linear one retained as a control, plus
`trivial_kstar_err` as an explicit image-blind leak instrument (because every generator sets
`L = clip(6·2π/k*, 18, 220)`, so `kstar` is partly recoverable from `L` alone). Do not
re-litigate it from this document; read `validate.py`'s module docstring and D-EVID-7.

The three-way distinction below remains an accurate and useful explanation of *why* it
mattered — kept for that reason, with the "status" column now historical:

| quantity | source | status |
|---|---|---|
| `kstar_model` | argmax_k of σ(k) from the **recovered** J, D | computed |
| `kstar_obs` | `observables.kstar_of(field)` — measured from the image | computed, drives the loss |
| `kstar_true` | argmax_k of σ(k) from the **true** J, D | **missing** — this is the NaN |

`kstar_true` is computable from the answer key with the machinery already in
`eval/analysis.turing_ok`, and that is the like-for-like comparison against `kstar_model`.
Scoring against `kstar_obs` instead compares a dispersion-relation prediction to a Fourier
measurement of a finite noisy frame — a different claim.

**How it was settled:** both are computed and reported, with the FFT-measured comparison as
the headline precisely because the linear reference is contaminated by the `L` leak above.
The weaker-claim concern was answered by shipping `trivial_kstar_err` alongside, so a reader
can see at a glance whether a headline number beats the image-blind baseline.

## What the two experiments are

Both feed the model **2 observed channels** and ask for a **3×3 GRN**. They test opposite
things, so they need different metrics.

- **A (`expA_hidden_channel`)** — N=3 truth, one gene genuinely hidden. *Can it be recovered?*
- **B (`expB_overparam`)** — N=2 truth, model given N=3. **No third gene exists.** *Does the
  spare capacity stay inert, or does the model invent a gene?*

Each has a control (`expA_control_full` m=3; `expB_control_matched` N=2). **Always read an
arm against its control, never against zero.**

## What changed on this branch

| file | change |
|---|---|
| `src/rngrn/scoring/` | **new package** — permutation.py, overparam.py, import-free `__init__.py` |
| `src/rngrn/validate.py` | routes scoring by arm; records `scoring_mode`, `n_true`, `n_model` |
| `src/rngrn/train.py` | classifies `arm`; records observed/hidden idx; row built by merge |
| `src/rngrn/optim/sweep.py` | explicit experiment-**arm** axes (distinct from tuning axes) |
| `src/rngrn/optim/benchmark.py` | `degradation_table()` + `degradation_markdown()` |
| `src/rngrn/cli.py` | `rngrn benchmark --degradation` |
| `configs/exp{A,B}_*.yaml` | 4 configs: 2 experiments + 2 controls |
| `tests/test_{permutation,overparam}_scoring.py`, `test_experiment_arms.py` | 46 new tests |

Test inventory: firewall 12 · science 7 · smoke 5 · registry 7 · permutation 23 ·
overparam 18 · experiment-arms 5 = **77**.

## Scoring API (all scoring-side; never imported by recovery)

```python
from rngrn.scoring import permutation as P, overparam as O

P.align_permutation(J_rec, J_true, observed_idx, *, free_observed=False) -> AlignmentScore
P.permuted_sign_match(J_rec, J_true, observed_idx, *, free_observed=False) -> dict
P.observed_subblock_score(J_rec, J_true, observed_idx) -> dict   # valid across N mismatch
P.latent_field_quality(latent_fields, true_hidden_channel) -> dict

O.spurious_species_metrics(J_rec, D_rec, observed_idx, n_true, coupling_threshold=0.05, ...)
O.observed_block_agreement(J_rec, J_true, observed_idx, sign_zero_tol=1e-9) -> dict
O.overparam_report(result, answer_key, observed_idx, ...) -> dict  # flat, run-index ready
```

`validate.score_recovery(result, answer_key, observed_idx=None)` dispatches:

| condition | `scoring_mode` | what you get |
|---|---|---|
| true J same size as recovered | `permutation_aligned` | aligned + identity sign match, best perm |
| `n_model > n_true` | `overparameterised` | spare-species metrics; `sign_match_frac` = NaN **by design** |
| no usable true J | `no_true_J` | subblock only |

`subblock_sign_match` is emitted in **every** arm — it is the only cross-arm-comparable column.

## Read these before changing anything

**1. The permutation search is degenerate for the configured Experiment A.** With N=3 and
2 observed channels there is exactly one unobserved species, so there is nothing to permute:
`n_permutations_searched == 1` and the aligned score *equals* the identity score. Alignment
only does work at ≥2 hidden species (N=4, m=2 → 2 candidates). A test asserts this so it
cannot be silently misread as "alignment helped". **The honest identifiability read for A is
`latent_field_quality`** — use `|r|`, since sign/scale is a gauge freedom on the latent field.
Do not report an "aligned" score for A as if alignment did something.

**2. The inertness threshold is uncalibrated.** `coupling_threshold=0.05` (relative) is a
placeholder. Calibrate from the distribution of `max_abs_coupling_*` in the
`expB_control_matched` arm, where no spare species exists by construction. Until then,
`spare_species_inert` is a provisional verdict.

**3. NaN in the degradation table usually means "not applicable to this arm"**, by
construction — not a failure. `sign_match_aligned_mean` is undefined for the
over-parameterised arm; `spare_inert_frac` is undefined everywhere else.

**4. The firewall is enforced by a test, not convention.** No recovery-side module may import
`data.rd_models`, `data.solver`, `data.cache`, `AnswerKey`, or `rngrn.scoring`.
`tests/test_firewall.py` is a static ast audit; both scoring modules carry their own audits.
Keep them green.

**5. House style: fail loud.** Raise rather than returning a fallback, a zero, or a silent
NaN. The original bug this branch fixes was exactly a silent NaN. Docstrings must not claim
more than the code does — a prior audit caught overstated provenance in this repo.

## Bugs found during integration (already fixed — don't reintroduce)

- **Duplicate-key `TypeError`s.** Scorers return keys (`observed_idx`, `loss`) that collided
  with run-identity keys in the run-index row. The row is now built by `dict.update()` merge,
  not `**`-expansion into a `dict()` literal, with run identity winning. If you add scorer
  keys, they flow through automatically — don't re-add explicit kwargs for them.
- **`np.asarray(None)` is a 0-d array, not `None`.** The missing-J guard checks `ndim`/`size`,
  not identity. Caught by `test_no_true_J_is_explicit_not_silent`.
- **Non-scalar metric values.** Run-index rows must be flat scalars (sqlite/jsonl); `train.py`
  stringifies anything else. Keep scorer outputs flat.

## Run it

```bash
pip install -e ".[dev]"
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1   # only if torch aborts with OMP Error #179
pytest -q     # expect 557 passed, 1 skipped (2026-08-11). Run with the SANDBOX DISABLED:
              # payload.h5 is on its read-deny list and a sandboxed run fakes ~15 failures.

# datasets are local and gitignored — see docs/LOCAL_DATA_SETUP.md
rngrn scan-datasets

# Only the two CONTROL arms run as written — see the note below. Trainers go through the
# memory guard (CLAUDE.md §7a).
for c in expA_control_full expB_control_matched; do
  bash scripts/guarded_run.sh rngrn --runs-root experiments train --config configs/$c.yaml
done
rngrn --runs-root experiments benchmark --degradation
```

> **Corrected 2026-08-04.** `expA_hidden_channel` and `expB_overparam` (`N=3, m=2`) raise
> `ValueError` at `recover.py:376`: `m < N` with the stationarity residual at weight 0, which
> is the default. Pass `-o loss.weights.resid=<nonzero>` to run them — there is no known-good
> value, so picking one is a science decision under CLAUDE.md §10. Full explanation in
> `docs/IDENTIFIABILITY_EXPERIMENTS.md` §"Running them".

Datasets needed: whatever `configs/expA_*.yaml:6` and `configs/expB_*.yaml:7` pin — today
`three_gene_val` (A) and `two_gene_classical_val` (B). **Read the ids from the configs, do not
restate them here:** this line said `two_gene_val` until 2026-08-04, which is the exact bug
lines 25-30 of this file document being fixed once already. Both already match the registry
layout — no conversion.

## Where to start (suggested order)

1. **Calibrate the inertness threshold** from the `expB_control_matched` arm. Cheap, and every
   Experiment-B verdict depends on it.
2. **Tune recovery on the controls first** (`expA_control_full`), where the problem is easiest.
   An experiment arm is only interpretable once its control recovers sensibly. Knobs: TUNING.md.
3. **Then run the experiment arms** with several seeds. NOTE: `kstar_identifiability_std`
   was NaN on every row until 2026-08-04 — `build_table` grouped on `config_id`, which
   hashes `train.seed`, so seeds never aggregated (`DECISIONS.md` D-EVID-13). Fixed, but
   runs recorded before that date cannot be regrouped and must be re-run to read it. Read
   `kstar_identifiability_std`
   (spread across in-regime seeds) as seriously as the means — this is a degenerate inverse
   problem.
4. **Consider ≥2 hidden species** (N=4, m=2) if you want the permutation machinery to be
   exercised for real; the current N=3/m=2 setup cannot.

Unrelated stubs still open elsewhere: `eval/numerics.bdf1_newton_krylov`, GradNorm/NTK
(which now **raise `NotImplementedError`** rather than falling back to fixed weights —
`weighting.py:69`, `:87`; corrected 2026-08-04)
weighting in `losses/weighting.py`, coupled-matrix ETDRK4. See TUNING.md.
