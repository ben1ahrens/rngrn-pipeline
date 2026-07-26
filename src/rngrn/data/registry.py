"""registry.py — the dataset store (ANSWER-KEY side).

A place to KEEP datasets you generated elsewhere, downloaded, and want to load back
into the codebase by a stable name — as opposed to data/cache.py, which is keyed by
the hash of an in-codebase generator spec and is for datasets this package regenerates.

Layout (one directory per dataset, addressed by a human-chosen dataset_id):
    <datasets_root>/<dataset_id>/
        payload.h5        # the field arrays: per-sample final_frame (+ optional
                          #   trajectory) and the QUARANTINED answer-key groups
        manifest.json     # provenance + shape/splits + checksum (METADATA ONLY)

A single index (index.py backend) lists every registered dataset so configs can say
`source: registry, dataset_id: <id>` and load by name. The index holds METADATA ONLY
(id, provenance, n_samples, splits, shapes, checksum) — never field arrays.

Firewall note: this module is answer-key side (it can see truth). The firewall gate
(gate.from_registry) is what splits a registered sample into (RecoveryInput, AnswerKey);
recovery-side code never imports this module.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil

import h5py
import numpy as np

from ..index import open_index

MANIFEST = "manifest.json"
PAYLOAD = "payload.h5"


def _checksum(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def dataset_dir(datasets_root, dataset_id):
    return os.path.join(datasets_root, dataset_id)


def register(datasets_root, dataset_id, payload_path, provenance=None,
             observable_key="final_frame", answer_key_keys=("jacobian", "x_star", "D",
             "interaction_matrix", "kstar"), backend="jsonl", overwrite=False):
    """Copy an existing HDF5 payload into the store under dataset_id and index it.

    payload_path : an HDF5 file whose top-level groups are samples; each sample has
                   an `observable_key` dataset (the frame) plus answer-key datasets.
    provenance   : free-form dict (where it came from, generator, git rev, ...).
    Returns the manifest dict.
    """
    d = dataset_dir(datasets_root, dataset_id)
    if os.path.exists(d) and not overwrite:
        raise FileExistsError(f"dataset '{dataset_id}' already registered at {d} "
                              "(pass overwrite=True to replace)")
    tmp = d + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    dst_payload = os.path.join(tmp, PAYLOAD)
    shutil.copyfile(payload_path, dst_payload)

    # inspect (metadata only) — sample keys, split tags, frame shape
    with h5py.File(dst_payload, "r") as f:
        sample_keys = list(f.keys())
        n_samples = len(sample_keys)
        splits = {}
        frame_shape = None
        for k in sample_keys:
            g = f[k]
            if observable_key in g and frame_shape is None:
                frame_shape = list(g[observable_key].shape)
            sp = g.attrs.get("split") if hasattr(g, "attrs") else None
            if sp is not None:
                splits[str(sp)] = splits.get(str(sp), 0) + 1

    manifest = dict(
        dataset_id=dataset_id, n_samples=n_samples, sample_keys=sample_keys[:1000],
        frame_shape=frame_shape, splits=splits, observable_key=observable_key,
        answer_key_keys=list(answer_key_keys), checksum=_checksum(dst_payload),
        provenance=provenance or {},
    )
    with open(os.path.join(tmp, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    if os.path.exists(d):
        shutil.rmtree(d)
    os.replace(tmp, d)   # atomic promote

    idx = open_index(datasets_root, "datasets", backend)
    # de-dup the index row for this id (append-only ledger; last row wins on read-back)
    idx.append(dict(dataset_id=dataset_id, n_samples=n_samples, frame_shape=str(frame_shape),
                    splits=json.dumps(splits), checksum=manifest["checksum"],
                    source=str((provenance or {}).get("source", "")),
                    path=os.path.relpath(d, datasets_root)))
    return manifest


def build_manifest(payload_file, dataset_id, provenance=None, observable_key="final_frame",
                   answer_key_keys=("jacobian", "x_star", "D", "interaction_matrix", "kstar")):
    """Inspect an HDF5 payload IN PLACE and build its manifest dict (metadata only)."""
    with h5py.File(payload_file, "r") as f:
        sample_keys = list(f.keys())
        splits, frame_shape = {}, None
        for k in sample_keys:
            g = f[k]
            if observable_key in g and frame_shape is None:
                frame_shape = list(g[observable_key].shape)
            sp = g.attrs.get("split") if hasattr(g, "attrs") else None
            if sp is not None:
                splits[str(sp)] = splits.get(str(sp), 0) + 1
    return dict(dataset_id=dataset_id, n_samples=len(sample_keys),
                sample_keys=sample_keys[:1000], frame_shape=frame_shape, splits=splits,
                observable_key=observable_key, answer_key_keys=list(answer_key_keys),
                checksum=_checksum(payload_file), provenance=provenance or {})


def scan(datasets_root, backend="jsonl", observable_key="final_frame", refresh=False):
    """Discover MANUALLY PLACED datasets and index them in place (no copying).

    Workflow this supports: you download a dataset yourself and drop it at
        <datasets_root>/<dataset_id>/payload.h5
    then run `rngrn scan-datasets`. Any such directory missing a manifest.json gets one
    generated from the file itself, and an index row is written so `list-datasets` and
    `source: registry` find it. Existing manifests are left alone unless refresh=True.

    Returns a list of dicts: [{dataset_id, action, n_samples}, ...].
    """
    if not os.path.isdir(datasets_root):
        return []
    out = []
    idx = open_index(datasets_root, "datasets", backend)
    for entry in sorted(os.listdir(datasets_root)):
        d = os.path.join(datasets_root, entry)
        payload = os.path.join(d, PAYLOAD)
        if not os.path.isdir(d) or not os.path.exists(payload):
            continue
        man_path = os.path.join(d, MANIFEST)
        if os.path.exists(man_path) and not refresh:
            action = "already-indexed"
            with open(man_path) as fh:
                man = json.load(fh)
        else:
            man = build_manifest(payload, entry,
                                 provenance={"source": "manually placed; indexed by scan"},
                                 observable_key=observable_key)
            with open(man_path, "w") as fh:
                json.dump(man, fh, indent=2, default=str)
            idx.append(dict(dataset_id=entry, n_samples=man["n_samples"],
                            frame_shape=str(man["frame_shape"]),
                            splits=json.dumps(man["splits"]), checksum=man["checksum"],
                            source="manual", path=entry))
            action = "refreshed" if os.path.exists(man_path) and refresh else "indexed"
        out.append(dict(dataset_id=entry, action=action, n_samples=man.get("n_samples")))
    return out


def load_manifest(datasets_root, dataset_id):
    p = os.path.join(dataset_dir(datasets_root, dataset_id), MANIFEST)
    if not os.path.exists(p):
        raise FileNotFoundError(f"dataset '{dataset_id}' not found in {datasets_root}")
    with open(p) as fh:
        return json.load(fh)


def list_datasets(datasets_root, backend="jsonl"):
    """Return the newest index row per dataset_id (append-only -> last wins)."""
    rows = open_index(datasets_root, "datasets", backend).read()
    by_id = {}
    for r in rows:
        by_id[r.get("dataset_id")] = r     # later rows overwrite earlier
    return list(by_id.values())


def payload_path(datasets_root, dataset_id):
    return os.path.join(dataset_dir(datasets_root, dataset_id), PAYLOAD)
