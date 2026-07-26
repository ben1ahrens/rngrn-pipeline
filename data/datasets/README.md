# Dataset registry

Two ways to get a dataset in here. Both end in the same place; pick whichever suits.

## A. You already downloaded the file — drop it in, then scan

    mkdir -p data/datasets/<dataset_id>
    cp ~/Downloads/val.h5 data/datasets/<dataset_id>/payload.h5   # name it payload.h5
    rngrn scan-datasets

`scan-datasets` walks this directory, writes a `manifest.json` for any dataset that
lacks one (inspecting the file in place — nothing is copied), and adds an index row.
Re-running is a no-op; pass `--refresh` to rebuild manifests. Without a manifest the
loader fails loud (`dataset '<id>' not found`) — the manifest is what tells the
firewall gate which HDF5 key is the observable and which are answer-key quantities.

## B. Let the CLI copy it in for you

    rngrn register-data --dataset-id <id> --payload /path/to/file.h5 --provenance "..."

## Then, from any config

    data:
      source: registry
      dataset_id: <id>
      sample_key: sample_0000

## Note on git

`payload.h5` files are gitignored (they are large binaries). Manifests and the index
ARE tracked, so the repo records *which* datasets an experiment used without carrying
the data. Cloning the repo does not bring the datasets — re-drop them and scan.

Expected per-sample HDF5 layout: a `final_frame` dataset (the observable) plus
answer-key datasets (`jacobian`/`x_star`/`D`/`interaction_matrix`), which the gate
quarantines. Override the observable name with `--observable-key` if yours differs.

---

Legacy note — register a stored/downloaded HDF5 payload by name:

    rngrn register-data --dataset-id <id> --payload /path/to/file.h5 --provenance "..."

Each dataset becomes <id>/{payload.h5, manifest.json}; a `datasets` index (jsonl or
sqlite per tracking.index_backend) lists them. Load from a config with
`data.source: registry` + `data.dataset_id: <id>`. Payloads are git-ignored (large);
manifests + index are kept. Expected per-sample HDF5 layout: a `final_frame` dataset
(the observable) plus answer-key datasets (jacobian/x_star/D/interaction_matrix),
which the firewall gate quarantines.
