# rngrn-pipeline README.md

Per-frame variational recovery of a gene regulatory network (GRN) from a single
converged Turing-pattern image: the **inverse Turing problem**. The learnable
object is a differentiable reaction–diffusion model (the RNGRN) whose weights
*are* the recovered network.


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
  cli.py                 rngrn {generate-data,register-data,scan-datasets,list-datasets,
                                train,evaluate,analyze,sweep,benchmark,
                                target-report,export}
configs/                 base.yaml + milestone1/2/3 + sweep_example.yaml
tests/                   firewall audit · science anchors · end-to-end smoke
```

## Install & run

```bash
pip install -e .            # torch, numpy, scipy, h5py, matplotlib, pyyaml, networkx
pytest -q                   # 557 passed, 1 skipped: firewall + science anchors + scorers + smoke
                            # RUN THIS WITH THE SANDBOX DISABLED. payload.h5 is on the sandbox
                            # read-deny list, so a sandboxed run fakes ~15 PermissionError
                            
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
own controls


| `data.source` | use | keyed by |
|---|---|---|
| `reference` | regenerate a reference system on the fly | spec hash (content cache) |
| `cache` | reload a previously generated cache entry | `dataset_hash` |
| **`registry`** | **load a stored/downloaded dataset by name** | **`dataset_id`** |
| `hdf5_3gene` | one raw external HDF5 by path | `hdf5_path` + `sample_key` |

**The dataset registry** (`data/registry.py`) is the store for datasets you generated
elsewhere and downloaded. Register once, then load by name from any config:

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

