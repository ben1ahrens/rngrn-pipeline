Dataset registry. Register a stored/downloaded HDF5 payload here by name:

    rngrn register-data --dataset-id <id> --payload /path/to/file.h5 --provenance "..."

Each dataset becomes <id>/{payload.h5, manifest.json}; a `datasets` index (jsonl or
sqlite per tracking.index_backend) lists them. Load from a config with
`data.source: registry` + `data.dataset_id: <id>`. Payloads are git-ignored (large);
manifests + index are kept. Expected per-sample HDF5 layout: a `final_frame` dataset
(the observable) plus answer-key datasets (jacobian/x_star/D/interaction_matrix),
which the firewall gate quarantines.
