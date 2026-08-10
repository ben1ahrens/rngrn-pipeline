# `data/datasets/` — the registered datasets, and how training data is generated

A dataset's **directory name here is its `dataset_id`**. A config's `dataset_id` must match it
exactly; an id/directory mismatch once silently disabled both arms of an experiment for the
life of a branch.

`payload.h5` is **gitignored**. `manifest.json` and `datasets.jsonl` are **tracked**, so the
repo records which dataset a run used — by id and checksum — without carrying the bytes.

---

## 1. Which datasets to use

| dataset | n | role |
|---|---|---|
| **`turing_spots`** | 5 | **PRIMARY — the training data source.** 512², homogeneous: 5/5 measured `spots`. |
| **`turing_labyrinth`** | 5 | **PRIMARY — the training data source.** 512², **mixed: 3 holes + 2 labyrinths** (see §5). |
| `three_gene_qvar` | 34 | Superseded as primary. The provenance of every canonical system; kept for continuity with runs already made against it. |
| `three_gene_multiL` | 92 | Cross-L transfer only (`PREREGISTRATION.md` §3.5a). 23 systems × 4 domain sizes. |
| `three_gene_train/val/test` | 127 | **Legacy. May not support any k\* claim** — the domain-size leak. Kept as the baseline that shows what the leak looked like. |
| `*_classical_*` | 160 | Dormant. Not 3-gene GRNs, and their generator is not in version control. |

**The legacy exclusion is not stylistic.** Every pre-2026-07-29 generator set
`L = clip(6·2π/k*, 18, 220)`, so `k* = 6·2π/L` exactly and a predictor that never looks at
the image recovers k\* perfectly. On those sets `kstar_rel_err` measures the leak, not the
model.

## 2. First-time setup on a new machine or worktree

A checkout **looks** provisioned and is not: manifests are tracked, payloads are not, so
`rngrn list-datasets` lists everything and the first load fails deep in h5py.

```bash
bash scripts/link_payloads.sh
```

That symlinks every payload from the canonical store — `rngrn-pipeline/data/datasets/`, the
main checkout — into this worktree. It refuses to clobber a real file or create a dangling
link, which is why it exists instead of a hand-rolled `ln -sf` loop.

**The canonical store holds the originals. Worktrees hold symlinks, never originals.** A
worktree that has generated new data holds the only copy until it is harvested (§4).

## 3. Regenerating the canonical training datasets

Everything needed is tracked: the generator, the selection logic, the frozen selection
record, and every seed. Nothing here depends on a file that has been deleted.

```bash
# 1. selection — which systems, and at what periods-per-box.  ~15 min (probe simulations)
.venv/bin/python scripts/canon_select.py
#    -> data/canonical_selection.json          (frozen record, tracked)
#    -> data/canonical_stability_cache.json    (probe verdicts, tracked)

# 2. generation — re-simulate the chosen systems at 512x512.  ~70 min at --procs 10
bash scripts/guarded_run.sh .venv/bin/python -u scripts/canon_generate.py --procs 10

# 3. register — write manifests and index rows
.venv/bin/rngrn scan-datasets --datasets-root data/datasets

# 4. annotate — measured morphology + full-precision k_star_fft
.venv/bin/python scripts/canon_annotate.py
```

Run generation through `scripts/guarded_run.sh`. Host RAM is the binding resource on this
machine (CLAUDE.md §7a) and five sessions have been OOM-killed.

### What each step does, and why it is not optional

- **`canon_select.py`** picks systems from the 57 that can be re-simulated at all — only
  `three_gene_qvar` and `three_gene_multiL` store both `params_json` and `sim_seed`; the 127
  legacy samples store neither and their kinetics are gone. It applies admission gates
  (`peak_bin ≥ 3`, `cv ≥ 0.30`, positive class margin), ranks by distance from the class
  boundary, and requires the morphology label to survive re-simulation at a different box
  size. Periods-per-box are laid out as a geometric ladder and **checked** against a leak
  bar, not assumed adequate.
- **`canon_generate.py`** re-simulates at 512² with `Tmax = 500` non-dimensional units,
  escalating the horizon per sample if the saturation gate is not met, and refuses to ship a
  transient frame.
- **`canon_annotate.py`** writes the measured morphology (§5) and rewrites `k_star_fft` at
  full precision — the generator rounds it to 3 dp, which is harmless at the legacy L ≤ 220
  and breaks the half-integer bin property at the L ≈ 990 this corpus reaches.

### Generating a fresh corpus instead of re-simulating

`scripts/gen_tg3.py` is the tracked generator for the `three_gene_*` families. Modes:
`qvar` (one simulation per screened system), `multiL` (the same system at four domain sizes),
`smoke` (3 samples, pipeline tests only). Commands and seeds for the existing sets are in
`docs/DATASETS_L.md`.

## 4. If you GENERATE data, harvest it before the worktree dies

A generated `payload.h5` is a **real file** in that worktree. Manifests are tracked and
payloads are not, so removing the worktree first leaves the tracked manifest pointing at
nothing. This has destroyed data in this repo before.

```bash
cp data/datasets/<id>/payload.h5      ../../rngrn-pipeline/data/datasets/<id>/payload.h5
cp data/datasets/<id>/manifest.json   ../../rngrn-pipeline/data/datasets/<id>/manifest.json
```

Then verify `registry._checksum` against the tracked manifest, and replace the worktree's
copy with a symlink into the store. `find data/datasets -name payload.h5 -type f` lists any
originals still sitting in a worktree — it should return nothing outside the main checkout.

## 5. Morphology: stored label vs measured

Every sample carries the generator's `morphology` attribute. The canonical `turing_*` sets
additionally carry **`morphology_measured`**, computed from the field by
`scripts/phase_topology.py` — which phase fragments into domains, and whether those domains
are round or worm-like.

They disagree, and the disagreement is real. Across the 57 re-simulatable systems:

| stored label | → spots | → holes | → labyrinth |
|---|---|---|---|
| `spots` (29) | **28** | 0 | 1 |
| `labyrinth` (17) | 3 | **7** | **7** |
| `stripes` (11) | 2 | 1 | 8 |

`spots` is reliable. Only 41 % of `labyrinth` is a labyrinth — most of the rest is a **hole**
pattern, which the generator's rule cannot name because its `phi > 0.66` test can essentially
never fire for a connected bright matrix. **The corpus is not relabelled**: all 413 samples
keep their stored attribute, so no existing number changes meaning.

## 6. Caveats that bite

- **Corpus medians are meaningless at n=5.** The canonical sets have five samples each. In
  particular `kstar_rel_err` medians must not be quoted from them: an oracle blind predictor
  can always sit near the middle of five values, and no period range fixes that. **A
  per-sample result is the unit of evidence.**
- **Every system is pre-screened Turing-unstable, and every sample is conditioned on
  `cv ≥ 0.05`.** You cannot ask this corpus what fraction of random networks pattern, and the
  cv floor is an artefact of the filter.
- **"Patterned" is not "Turing-unstable".** Different claims; see CLAUDE.md §7c.
- **`observables.raps` assumes a periodic tile** — `dk = 2π/L`, no windowing or detrending.
  A cropped real image will bias k\* *silently*. The canonical sets are clean periodic
  masters precisely so degradation can be applied downstream and measured.
- **`payload.h5` may be on the sandbox read-deny list.** A sandboxed `pytest` shows ~15
  spurious `PermissionError` failures that look like code faults. Run with the sandbox off.

## 7. Where the reasoning lives

- `docs/CANONICAL_DATASETS.md` — the canonical sets in detail.
- `docs/HANDOFF_canonical_datasets.md` — zero-context handoff.
- `docs/DATASETS_L.md` — the L-decoupled sets and the leak audit.
- `docs/DECISIONS.md` — `D-CANON-1..5` (canonical sets), `D6` (the L policy), `D15`
  (the 6 px/wavelength floor), `D-TDPLOT-1` (the periodicity gate).
- `docs/LOCAL_DATA_SETUP.md` — per-machine setup.
