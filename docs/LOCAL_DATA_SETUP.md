# Local dataset setup (one-time, per machine)

Datasets live on your machine, not in git. This is deliberate: `payload.h5` files are
gitignored, while manifests and the dataset index ARE tracked — so the repo records
*which* dataset (by id and checksum) an experiment used, without carrying the bytes.
A fresh clone has no data; do this once per machine.

## 1. Download from the Claude Science project

Grab the artifacts you want (the ones below are the useful ones for the milestones).
Sizes are the compressed tarballs.

| artifact | size | what's inside | good for |
|---|---|---|---|
| `three_gene_dataset.tar.gz` | 80 MB | `train/val/test.h5` (127 samples, 3×96×96), `summary.json`, `metadata.csv`, `generator.py` | **M3** N=3, and both N=3 identifiability experiments |
| `three_gene_classical.tar.gz` | 8 MB | `train/val/test.h5`, 3×96×96 | smaller N=3 set for fast iteration |
| `two_gene_classical.tar.gz` | 4.6 MB | `train/val/test.h5`, 2×96×96 | **M1/M2**, and the N=2-truth → N=3-model experiment |

## 2. Unpack and drop in

Each `*.h5` split becomes one registered dataset. Name the file `payload.h5`:

```bash
cd /path/to/rngrn-pipeline
mkdir -p /tmp/tg && tar xzf ~/Downloads/three_gene_dataset.tar.gz -C /tmp/tg

mkdir -p data/datasets/three_gene_train && cp /tmp/tg/train.h5 data/datasets/three_gene_train/payload.h5
mkdir -p data/datasets/three_gene_val   && cp /tmp/tg/val.h5   data/datasets/three_gene_val/payload.h5
mkdir -p data/datasets/three_gene_test  && cp /tmp/tg/test.h5  data/datasets/three_gene_test/payload.h5

mkdir -p /tmp/2g && tar xzf ~/Downloads/two_gene_classical.tar.gz -C /tmp/2g
mkdir -p data/datasets/two_gene_classical_val  && cp /tmp/2g/val.h5   data/datasets/two_gene_classical_val/payload.h5
```

The **directory name becomes the dataset id** — `scan-datasets` indexes each directory
under `data/datasets/` by its own name. Configs reference that id, so the directory name
and the `dataset_id` in a config must match exactly. (The Experiment B configs originally
said `two_gene_val` while the directory was `two_gene_classical_val`; the id in the
configs is now the directory name.)

## 3. Index them

```bash
rngrn scan-datasets      # writes manifest.json + index rows; inspects in place, no copying
rngrn list-datasets      # confirm
```

`scan-datasets` is idempotent — re-run it any time you add a dataset. Use `--refresh`
to rebuild existing manifests, `--observable-key` if your observable isn't `final_frame`.

## 4. Use from a config

```yaml
data:
  source: registry
  dataset_id: three_gene_val
  sample_key: sample_0000
model: {N: 3, m: 3, observed_idx: [0, 1, 2]}
```

## Layout compatibility

These datasets are already in the expected shape — per-sample HDF5 groups, each with
`final_frame` (the observable) plus `jacobian` / `x_star` / `D` (+ `interaction_matrix`
on the big 3-gene set) as answer-key quantities the firewall gate quarantines. Verified
working unmodified for both the 3-gene and 2-gene classical sets.

Two exceptions that need conversion first:

- **`three_gene_turing_datasets.tar.gz`** uses a different layout: one directory per
  system, each with a FLAT `data.h5` (top-level `final_frame`, `jacobian`, ... — no
  per-sample groups). Wrap each into a per-sample group before dropping in, or add a
  `from_flat_hdf5` loader to `data/gate.py`.
- **`turing_training_data.npz`** is npz, not HDF5, with `u`/`v` as separate
  `(21, 256, 256)` stacks. Convert to the per-sample HDF5 layout to register it.

## Disk

Unpacked, the big 3-gene set is ~82 MB of HDF5 (train 57 / val 12 / test 13). Nothing
here is large by ML standards; no LFS or release assets needed.
