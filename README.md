# rngrn-pipeline

Per-frame variational recovery of a gene regulatory network (GRN) from a single
converged Turing-pattern image — the **inverse Turing problem**. The learnable
object is a differentiable reaction–diffusion model (the RNGRN) whose weights
*are* the recovered network.

> **Status: TUNING — past template, short of a result.** Rewritten 2026-08-11; it had said
> "TEMPLATE … the next stage, on a CUDA machine", which stopped being true weeks ago.
>
> **What is done.** The six-stage pipeline is wired and runs (**measure the pass count
> yourself** — `pytest -q` with the sandbox disabled; it has gone stale within days every
> time it was recorded here, see CLAUDE.md §3 — all eleven CLI subcommands work). Real
> tuning campaigns have run on real data: **133 recorded runs** across
> `experiments/`, with run records, frozen configs, checkpoints and plottable arrays tracked
> in git so every number is traceable to the run behind it. **39 dated decisions** are
> recorded in **[docs/DECISIONS.md](docs/DECISIONS.md)**, including the C1/C2 tuning
> campaigns, canonical dataset generation, morphology scoring and the D-ratio prior.
>
> **What is NOT done, stated as plainly as the above.** No pre-registered criterion in
> **[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)** §3 has been declared met, and
> criterion 3.1 (`topology_consistency`) has failed three independent ways. Almost all 133 runs
> are against `three_gene_qvar` or older sets. The exception is the D5 diagnostic cell
> (`experiments/diag_fft/d5/`): 10 committed recovery runs and a K=10 target report against
> `turing_labyrinth/sample_0000` — the tuning-role sample, not a split violation. **No
> held-out sample has been touched**: `sample_0004` (`turing_labyrinth`'s held-out split) is
> still untouched, and `turing_spots` has no runs at all. So: the machine runs and is
> measured, the science is not yet proven.
>
> Read **[docs/STATE_OF_THE_SCIENCE.md](docs/STATE_OF_THE_SCIENCE.md)** for the evidence
> ledger and **[TUNING.md](TUNING.md)** for the knobs, stubs and unproven science.

## The one non-negotiable rule — the firewall

No analytic quantity of a ground-truth system (steady state x\*, Jacobian J,
dispersion σ(k), or any generating coefficient) may enter recovery anywhere,
including initialisation. They live only in a separate **answer-key** object used to
*score* after the fact. The only truth-derived inputs to recovery are
`(frame, L, observed_idx)`. This is enforced by `tests/test_firewall.py` (a static
import audit), not by comments.

## Layout (layered by rate-of-change)

```
src/rngrn/
  registry.py            name -> implementation registries
  config.py              typed, hashable, YAML-composed config-as-experiment
  utils.py  io.py  tracking.py   seeding/provenance · run dirs + run index · Tracker
  index.py               metadata-index backends: jsonl (default) | sqlite
  model.py               the RNGRN core (VALIDATED science)         [recovery side]
  observables.py         image-only k*, morphology, Laplacians       [recovery side]
  forward.py             PatternSolver: forward pattern solve on detected instability
  etdrk4_torch.py        torch ETDRK4 step used by forward.py
  history.py             per-step optimisation history capture
  plotdata.py            plottable-array export for recorded runs
  export.py              run index -> tidy/long CSV
  data/
    rd_models.py         reference RD systems + analytic answer keys [ANSWER-KEY side]
    solver.py            forward RD generator                        [ANSWER-KEY side]
    spec.py  cache.py    dataset spec/hash · content-addressed cache [ANSWER-KEY side]
    registry.py          dataset STORE for downloaded/generated sets  [ANSWER-KEY side]
    gate.py              THE FIREWALL: -> (RecoveryInput, AnswerKey)  [boundary]
  losses/
    terms.py             differentiable loss terms
    spectral.py          spectral loss terms (spec_shape/aniso/amp_mean/amp_fluct, real_moments)
    weighting.py         fixed/scheduled/gradnorm/ntk strategies
    total.py             assemble terms -> weighting -> scalar (fail-loud)
  recover.py             per-frame variational recovery              [recovery side]
  train.py               fit(): config -> gate -> recover -> score -> run index
  validate.py            score a recovery vs the answer key (priority order)
  eval/
    numerics.py          ETDRK4 / IMEX / BDF1 stiff integrators
    dynamical.py         QSS->lift + mu->0 consistency
    rollout.py           lift-and-simulate (the real success test)
    analysis.py          linear stability, Turing conditions, robustness cloud
    topology.py          learned-GRN diagram
    lifted.py            lifted-field helpers for rollout/scoring
    lgen_eval.py         L-generalisation evaluation helpers
  scoring/
    lgen.py              cross-L modal-agreement metrics
    morphology.py        morphology matching/distance
    overparam.py         spare-species inertness scoring
    permutation.py       hidden-channel permutation alignment
    plausibility.py      biological-plausibility box scoring
    reproducibility.py   seed-reproducibility scoring
  optim/
    sweep.py             outer loop over fit() -> run index
    benchmark.py         cross-run comparison table + identifiability metric
  cli.py                 rngrn {generate-data,register-data,scan-datasets,list-datasets,
                                train,evaluate,analyze,sweep,benchmark,
                                target-report,export}
configs/                 15 configs — see configs/ for the full list
tests/                   firewall audit · science anchors · end-to-end smoke
```

## Install & run

```bash
pip install -e .            # torch, numpy, scipy, h5py, matplotlib, pyyaml, networkx, pandas
pytest -q                   # measure the pass count yourself: firewall + science anchors + scorers + smoke
                            # RUN THIS WITH THE SANDBOX DISABLED. payload.h5 is on the sandbox
                            # read-deny list, so a sandboxed run fakes ~15 PermissionError
                            # failures that look exactly like code faults. See CLAUDE.md §3.

# the six-stage CLI (dry-run scale shown; drop the overrides for real runs).
# NOTE: launch trainers through the memory guard — see CLAUDE.md §7a.
#
# generate-data was BROKEN until 2026-08-11 (D-EVID-17): the solver capped dt by the
# reaction Jacobian only, never by diffusion, so both milestone1_* frames diverged.
# Fixed by solving diffusion exactly in Fourier space; ~32 s (gierer_meinhardt) and
# ~16 min (schnakenberg) per 128x128 frame.
#   CAVEAT: the output is stable but NOT dt-converged — refining dt shifts the pattern
#   contrast ~35 %, against a 2.6 % seed control. Fine for plumbing and morphology; do
#   NOT quote a k* or contrast value from these frames as a reference number. D-EVID-17.
rngrn generate-data --config configs/milestone1_schnak.yaml
bash scripts/guarded_run.sh rngrn train --config configs/milestone1_schnak.yaml
rngrn evaluate      --config configs/milestone1_schnak.yaml --run-id <run_id>
rngrn analyze       --config configs/milestone1_schnak.yaml --run-id <run_id>
bash scripts/guarded_run.sh rngrn sweep --sweep configs/sweep_example.yaml
rngrn benchmark     --format markdown

# the auditable per-target instrument the pre-registration is judged on
bash scripts/guarded_run.sh rngrn target-report --config configs/m3_registry.yaml \
    --dataset-id three_gene_val --sample-key sample_0000 --form competitive --seeds 0 1 2
rngrn export        --out-dir exports     # indexes -> tidy/long CSV
```

Config is composed from `base.yaml` via `_base_:` and overridden on the CLI with
`-o dotted.key=value` (e.g. `-o loss.weights.resid=0.6 train.adam_lr=0.03`).

## Identifiability experiments

Two validation experiments feed the model 2 observed channels and ask for a 3x3 GRN:
**A** recovers a genuinely hidden gene (N=3 truth); **B** tests whether spare capacity
invents one (N=2 truth, N=3 model). They need different metrics and are read against their
own controls — see [docs/IDENTIFIABILITY_EXPERIMENTS.md](docs/IDENTIFIABILITY_EXPERIMENTS.md).
`rngrn benchmark --degradation` prints the per-arm comparison. Agent/collaborator handoff:
[docs/HANDOFF_identifiability.md](docs/HANDOFF_identifiability.md).

## Handoff: the target goal and the state of the science

Four documents written for a zero-context reader picking this up (information, not
instructions — every number is either measured here or cited):

| doc | contents |
|---|---|
| [docs/GOAL_tica_equivalent.md](docs/GOAL_tica_equivalent.md) | What Tica et al. built, what "equivalent but more robust" could mean (four incompatible readings), and the measured gap between here and there |
| [docs/STATE_OF_THE_SCIENCE.md](docs/STATE_OF_THE_SCIENCE.md) | Every measured result, every rejected approach, and the numbers that superseded earlier over-optimistic ones. **Read before running anything.** |
| [docs/ROBUSTNESS_MEASUREMENT.md](docs/ROBUSTNESS_MEASUREMENT.md) | The robustness baseline of the 127 generator systems, four measured defects in `robustness_cloud`, and what a Tica-comparable claim needs |
| [docs/CODE_REALITY.md](docs/CODE_REALITY.md) | What is wired vs stubbed, where docs disagree with source, the cost model, and the open decisions |

The shared working contract (**[CLAUDE.md](CLAUDE.md)**) is tracked at the repo root on this
branch (brought over in `c53ebf6`). It originated on `docs/agent-conventions`; that branch is
no longer the copy to read.

## The three milestones

| milestone | config | what it isolates |
|---|---|---|
| M1 | `milestone1_{gm,schnak}.yaml` | N=m=2 fully observed — tune the objective here |
| M2 | `milestone2_gm_partial.yaml` | N=2, m=1 — latent v field, identifiability degradation |
| M3 | `m3_registry.yaml` | N=3 from a registered dataset — defaults to the canonical `turing_spots` |

## Where datasets come from — four sources, one firewall

`data.source` selects how a frame is obtained; all four flow through the same gate,
so `final_frame` is the observable and `jacobian`/`x_star`/`D`/`kstar` are quarantined
as the answer key:

| `data.source` | use | keyed by |
|---|---|---|
| `reference` | regenerate a reference system on the fly | spec hash (content cache) |
| `cache` | reload a previously generated cache entry | `dataset_hash` |
| **`registry`** | **load a stored/downloaded dataset by name** | **`dataset_id`** |
| `hdf5_3gene` | one raw external HDF5 by path | `hdf5_path` + `sample_key` |

**The dataset registry** (`data/registry.py`) is the store for datasets you generated
elsewhere and downloaded — the ones you can't (or don't want to) regenerate from a
spec in this codebase. Register once, then load by name from any config:

```bash
# either: let the CLI copy the file into the store
rngrn register-data --dataset-id three_gene_v1 --payload /path/to/val.h5 \
     --provenance "3-gene val split"

# or: drop it in yourself, then index it in place (nothing is copied)
mkdir -p data/datasets/three_gene_v1
cp /path/to/val.h5 data/datasets/three_gene_v1/payload.h5
rngrn scan-datasets

rngrn list-datasets
rngrn train --config configs/m3_registry.yaml         # source: registry, dataset_id: turing_spots
```

**One-time per machine:** see [docs/LOCAL_DATA_SETUP.md](docs/LOCAL_DATA_SETUP.md)
for which artifacts to download, where to drop them, and which need conversion.

`payload.h5` files are gitignored; manifests and the index are tracked, so the repo
records which datasets an experiment used without carrying the data itself. A fresh
clone has no datasets — re-drop them and run `scan-datasets`.

Each dataset lives at `data/datasets/<dataset_id>/{payload.h5, manifest.json}`; the
manifest and a `datasets` index hold **metadata only** (provenance, shape, splits,
checksum) — never field arrays.

**Metadata index backend** (`tracking.index_backend`): both the run index and the
dataset registry index use it. `jsonl` (default) is zero-setup and diff-friendly;
`sqlite` gives real SQL (`SELECT … WHERE recovered_turing AND kstar_rel_err < 0.15
GROUP BY config_id`) once runs pile up. Same rows either way — switch freely.

## Tests: run them locally, before you push

The authoritative test run is **local**, via a `pre-push` hook. Enable it once per
repository — the setting is shared by every worktree, so you do not repeat it when you
add one:

```bash
git config core.hooksPath .githooks
```

From then on `git push` runs `pytest -q` first (using the `.venv` of whichever worktree
you push from, so the tests exercise the same torch build you develop against) and aborts
the push on failure. Bypass deliberately with `git push --no-verify`. To run the suite
through the hook without pushing: `git hook run pre-push`.

The GitHub Actions workflow in `.github/workflows/tests.yml` is kept as a definition
but should not be relied on: Actions minutes for private repos are metered, and runs
are skipped entirely when the account's billing is unavailable — the job then reports
a *failure* that has nothing to do with the code. The hook costs ~8 s and needs no
account state.

## Sandbox note

If `import torch` aborts with `OMP: Error #179` (seen in some containers), set
`KMP_AFFINITY=disabled OMP_NUM_THREADS=1` before running. `tests/conftest.py` does
this for the suite. Harmless on a normal workstation/GPU box.

## Where the science is proven vs. not

- **VALIDATED and repackaged as-is:** `model.py`, `observables.py` (autodiff Jacobian,
  general-N dispersion, RAPS k\*, morphology — all with prior numerical checks).
- **Repackaged from SCAFFOLD (needs validation):** `losses/terms.py`, `eval/*`,
  `validate.py`, `data/*`. The wiring is correct and dry-runs; the *recovery science*
  is unproven. TUNING.md marks each `[VALIDATE]` item.
