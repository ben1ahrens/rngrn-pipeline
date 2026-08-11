Place the unpacked 3-gene dataset here: train.h5 / val.h5 / test.h5
(per-sample groups with final_frame (3,H,W) + quarantined jacobian/x_star/interaction_matrix).

**This path is retired.** It served `data.source: hdf5_3gene`, which loads a loose HDF5 file
rather than a registered dataset. The configs that used it (`milestone3_3gene.yaml`,
`nc1_milestone3_3gene.yaml`) were removed on 2026-08-10 — they pointed at a file that no
longer exists, and the registry replaced them.

The `hdf5_3gene` source itself still works (`rngrn.data.gate.from_3gene_hdf5`), so this
directory remains the place to drop a loose dataset if you need that path. For anything else,
use the registry: see `data/datasets/README.md` for which datasets to use and how training
data is generated.
