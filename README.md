# rngrn-pipeline

Per-frame variational recovery of a gene regulatory network (GRN) from a single
converged Turing-pattern image — the **inverse Turing problem**. The learnable
object is a differentiable reaction–diffusion model (the RNGRN) whose weights
*are* the recovered network.

> **Status: TEMPLATE.** Every component of the six-stage pipeline is present, wired
> together, and dry-runs end-to-end on CPU (24 tests pass; all six CLI subcommands
> run). It does **not** yet *recover* correctly — tuning the objective and proving the
> milestones is the next stage, on a CUDA machine. See **[TUNING.md](TUNING.md)** for
> the explicit list of knobs, stubs, and unproven science to work on.

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
  data/
    rd_models.py         reference RD systems + analytic answer keys [ANSWER-KEY side]
    solver.py            forward RD generator                        [ANSWER-KEY side]
    spec.py  cache.py    dataset spec/hash · content-addressed cache [ANSWER-KEY side]
    registry.py          dataset STORE for downloaded/generated sets  [ANSWER-KEY side]
    gate.py              THE FIREWALL: -> (RecoveryInput, AnswerKey)  [boundary]
  losses/
    terms.py             differentiable loss terms
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
  optim/
    sweep.py             outer loop over fit() -> run index
    benchmark.py         cross-run comparison table + identifiability metric
  cli.py                 rngrn {generate-data,register-data,list-datasets,
                                train,evaluate,analyze,sweep,benchmark}
configs/                 base.yaml + milestone1/2/3 + sweep_example.yaml
tests/                   firewall audit · science anchors · end-to-end smoke
```

## Install & run

```bash
pip install -e .            # torch, numpy, scipy, h5py, matplotlib, pyyaml, networkx
pytest -q                   # 24 tests: firewall + science anchors + smoke dry-run

# the six-stage CLI (dry-run scale shown; drop the overrides for real runs)
rngrn generate-data --config configs/milestone1_schnak.yaml
rngrn train         --config configs/milestone1_schnak.yaml
rngrn evaluate      --config configs/milestone1_schnak.yaml --run-id <run_id>
rngrn analyze       --config configs/milestone1_schnak.yaml --run-id <run_id>
rngrn sweep         --sweep  configs/sweep_example.yaml
rngrn benchmark     --format markdown
```

Config is composed from `base.yaml` via `_base_:` and overridden on the CLI with
`-o dotted.key=value` (e.g. `-o loss.weights.resid=0.6 train.adam_lr=0.03`).

## The three milestones

| milestone | config | what it isolates |
|---|---|---|
| M1 | `milestone1_{gm,schnak}.yaml` | N=m=2 fully observed — tune the objective here |
| M2 | `milestone2_gm_partial.yaml` | N=2, m=1 — latent v field, identifiability degradation |
| M3 | `milestone3_3gene.yaml` | N=3 from the 3-gene HDF5 dataset |

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
rngrn register-data --dataset-id three_gene_v1 --payload /path/to/val.h5 \
     --provenance "3-gene val split"
rngrn list-datasets
rngrn train --config configs/registry_example.yaml    # source: registry, dataset_id: three_gene_v1
```

Each dataset lives at `data/datasets/<dataset_id>/{payload.h5, manifest.json}`; the
manifest and a `datasets` index hold **metadata only** (provenance, shape, splits,
checksum) — never field arrays.

**Metadata index backend** (`tracking.index_backend`): both the run index and the
dataset registry index use it. `jsonl` (default) is zero-setup and diff-friendly;
`sqlite` gives real SQL (`SELECT … WHERE recovered_turing AND kstar_rel_err < 0.15
GROUP BY config_id`) once runs pile up. Same rows either way — switch freely.

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
