---
name: harvest-dataset
description: Use when generating a new dataset in rngrn-pipeline, when provisioning payloads into a worktree, or before removing any worktree. Prevents the payload-loss failure that has already destroyed generated data — manifests are tracked in git, payload.h5 files are not.
---

# Datasets and payloads

The asymmetry that causes every problem here: **`manifest.json` and
`data/datasets/datasets.jsonl` are tracked in git; `payload.h5` is gitignored.** So the repo
records which dataset a run used, by id and checksum, without carrying the bytes — and a
checkout looks fully provisioned when it holds no data at all.

A dataset's **directory name under `data/datasets/` is its `dataset_id`**, and a config's
`dataset_id` must match the directory exactly. A mismatch once silently disabled both
Experiment B arms for the life of a branch. A payload with no manifest fails loud — the
loader will not guess.

## The canonical store

`rngrn-pipeline/data/datasets/<id>/payload.h5` — **the main checkout**. Worktrees hold
symlinks into it, never originals.

## If your unit GENERATES a dataset

Learned the hard way on 2026-07-29. A unit generated `three_gene_qvar` and
`three_gene_multiL` inside its own worktree. The manifests were tracked, the payloads were
not, so when the worktree was removed the manifests survived and **the data did not**.

So, as the **last step** of any unit that generates data:

1. **Copy the payload into the canonical store** (`rngrn-pipeline/data/datasets/<id>/`).
   Copy — do not move a file something may still be reading, and do not leave the original as
   the only copy inside a worktree that is about to be deleted.
2. **Verify `registry._checksum` against the tracked `manifest.json`.** A payload that does
   not match its manifest is worse than a missing one.
3. **Record the generating command and seed in a tracked doc.** This is what made recovery
   possible at all: `docs/DATASETS_L.md` carried the exact invocation, and unit 11 had already
   replaced the process-salted `abs(hash(topo))` seed with SHA-256 — so the dataset could be
   regenerated **byte-identically** rather than merely similarly.

## Provisioning a worktree

Use `bash scripts/link_payloads.sh`. It exists because the one-liner that used to be handed
to every agent is actively dangerous:

```bash
# DO NOT USE — this is the bug, kept here so it is recognised
for d in data/datasets/*/; do ln -sf $MAIN/$d/payload.h5 $d/payload.h5; done
```

It assumes every payload already lives in the canonical store. For a freshly generated dataset
it does not, so the loop creates a **dangling symlink** — and re-running it **overwrites a
real generated file with one**. Three of four downstream agents were left unable to load data
at all, and one dataset had to be regenerated from scratch.

The rules any provisioning must obey:

- **Link only when the target is absent or already a symlink.** Never clobber a real file.
- **Never link when the canonical payload does not exist.** No dangling symlinks.
- Cost when done right: ~84 KB of links instead of re-copying ~91 MB.

## Before removing ANY worktree

**Check for real (non-symlink) payloads inside it:**

```bash
find data/datasets -name payload.h5 -type f    # -type f excludes symlinks: these are ORIGINALS
```

Anything that prints is data that exists nowhere else. Harvest it to the canonical store and
verify its checksum before the worktree dies.

## Recovering from a branch checkout that ate the manifests

Checking out a branch that predates the tracked manifests **deletes them from disk**. Payloads
survive. Regenerate with:

```bash
rngrn scan-datasets     # idempotent, inspects in place, copies nothing
```

Per-machine setup is documented in `docs/LOCAL_DATA_SETUP.md`.
