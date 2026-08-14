# HANDOFF — the canonical Turing training datasets

> **ARCHIVED 2026-08-14.** This handoff was picked up and completed. §7's two follow-ups are
> struck through and marked DONE, closed by commits `3ecdcca` and `4060461`. Every deliverable
> it promises exists on disk: `data/canonical_selection.json` (tracked),
> `experiments/figures_report/canonical/` (10 per-sample figures plus both galleries), and
> `D-CANON-1..5` in `docs/DECISIONS.md`. It is superseded by
> **`docs/CANONICAL_DATASETS.md`**, which is the current reference for these datasets — read
> that document, not this one, for anything other than history. Two numeric errors present in
> the original text (five-distinct-systems-each in §1; the generator-morphology-class claim in
> §2 read as contradicting §1) were corrected in place before archiving; see
> `docs/CANONICAL_DATASETS.md` for the corrected figures and full defect note.

---

**Branch:** `feature/canonical-datasets`, off `feature/training-data-plots`.
**Date:** 2026-08-10. **Written for a zero-context reader.**

---

## 1. What exists now

Two registered datasets at **512×512** — `turing_labyrinth` promotes 4 distinct 3-gene
systems, `turing_spots` promotes 3 (5 sample-rows each; some systems occupy two rows — see
`docs/CANONICAL_DATASETS.md` §3 defect note, added 2026-08-14) — which the owner has
designated **the training data source** for simulated-data work from 2026-08-10:

| dataset | n | composition | periods | px/wavelength |
|---|---|---|---|---|
| `turing_spots` | 5 | 5/5 measured `spots` — homogeneous | 8, 11, 16, 24, 36 | 14.2–64.0 |
| `turing_labyrinth` | 5 | **3 holes + 2 labyrinths — mixed** | 8, 10, 15, 23, 35 | 14.6–64.0 |

Payloads are harvested into the canonical store (`rngrn-pipeline/data/datasets/`) with
checksums verified against the tracked manifests.

**To use them:** `gate.from_registry("data/datasets", "turing_spots", "sample_0000", N=3,
observed_idx=(0,1,2))`. Verified working. Every sample stores all three species channels, so
`m=1` versus `m=3` is a controlled comparison on identical data.

## 2. What is VERIFIED

- **Every sample is a Turing pattern.** `peak_bin` 7–39 against a floor of 3; `cv` 0.43–1.04
  against a floor of 0.30; all ten kept their generator's own **stored** morphology class at
  512² — true only of the generator's own classifier, which structurally cannot emit `holes`
  (§1: `turing_labyrinth` measures as 3 holes + 2 labyrinths under the separate measured
  classifier, `scripts/phase_topology.measure`; the stored label is unchanged by that
  measurement).
- **The domain-size leak is measured, not assumed.** An oracle blind predictor that reads
  only `L` scores **37.5 %** median error on `turing_spots` and **33.3 %** on
  `turing_labyrinth`, against **0.0 %** on the legacy sets.
- **Resolution beats the legacy corpus on both axes at once** — 14.2 px/wavelength worst case
  (the D15 floor is 6.0) at a k\* half-bin precision of 1.4–6.2 % (legacy: 8.3 %).
- **Every sample reached a steady state.** `Tmax = 500` non-dimensional units, and the
  saturation gate passed for all ten with no escalation.
- **Provenance is complete.** Each sample records the corpus system it was promoted from,
  its `sim_seed`, `params_json`, `tmax`, and its split role.
- Suite: see §6.

## 3. What is NOT verified — read before citing anything

- **No recovery result exists.** Nothing was fitted. These are training *inputs*.
- **"Patterned" is not "Turing-unstable".** σ(k) is evaluated nowhere in this branch.
- **`turing_labyrinth` is not one morphology** (§1). Do not treat it as a labyrinth class.
- **Corpus medians are not meaningful at n=5.** In particular `kstar_rel_err` medians must
  not be quoted: an oracle can always sit near the middle of five values, and no period range
  fixes it (D-CANON-3). A per-sample result is the unit of evidence here.
- **The whole future evidence base is 10 samples, 6 held out** — against 26 held out in the
  `three_gene_qvar` split.
- **`raps` still assumes a periodic tile.** No windowing or detrending, so a cropped real
  image will bias k\* *silently*. These are clean periodic masters so that degradation can be
  applied downstream and measured; that measurement has not been done.
- **The `m<N` objective is an open problem.** `exp06` measured the stationarity residual —
  the only term latent channels enter — as harmful, 9/9 cells collapsing.
- **The circularity cut (0.55) is uncalibrated.** Read off a measured separation, used for
  labelling only, never as a pass condition.

## 4. The three findings that changed the plan

1. **`stripes` is not a stable class in this generator** (D-CANON-2). Every stripes candidate
   flips to labyrinth when only the box size changes; the stripes fraction falls to zero at
   p ≥ 11; `corr(periods, anisotropy) = −0.312`. It is largely a small-box artefact — a small
   box admits few orientations, forcing the pattern onto one axis. The plan began as three
   datasets and ships two.
2. **The generator's `labyrinth` label conflates two morphologies** (D-CANON-5). Only 41 % of
   what it calls labyrinth is one; most of the rest is holes. `spots` is reliable (28/29).
3. **The periods draw nearly reintroduced the leak** (D-CANON-3). Distinct integers are not
   sufficient — the leak error is relative, so spread must be geometric. A first attempt drew
   `{17,22,23,24,28}`, which an oracle fits to 4.5 %.

## 5. Where to start

1. `docs/CANONICAL_DATASETS.md` — what they are, how they were made, how to reproduce.
2. `experiments/figures_report/canonical/` — `g_*` galleries, `d_*` per-sample detail
   (species + spectrum), and `s1_stripes_is_a_small_box_artefact.png` for finding 1.
3. `data/canonical_selection.json` — the frozen record: which systems, why admitted, which
   side of the split.
4. `docs/DECISIONS.md` D-CANON-1..5.

## 6. Reproducing and testing

```bash
bash scripts/link_payloads.sh
.venv/bin/python scripts/canon_select.py       # freeze selection (~15 min of probes)
bash scripts/guarded_run.sh .venv/bin/python -u scripts/canon_generate.py --procs 10
.venv/bin/rngrn scan-datasets --datasets-root data/datasets
.venv/bin/python scripts/canon_annotate.py     # write measured morphology
```

Generation is ~70 min wall-clock at `--procs 10`. **Run the suite with the sandbox
disabled** — `payload.h5` is on the read-deny list, and a sandboxed run shows ~15 spurious
`PermissionError` failures that look like code faults.

## 7. Known follow-ups, none blocking

- ~~**No config points at these datasets.**~~ **DONE — corrected 2026-08-11.** This landed in
  `3ecdcca`, 30 minutes after this handoff was written. `configs/m3_registry.yaml` and
  `configs/nc1_m3_registry.yaml` both name `turing_spots` today (verified). `expA_*.yaml`
  still names `three_gene_val` and `expB_*.yaml` still names `two_gene_classical_val`, and
  that part is **deliberate**: both are pre-registered experiment arms, and expB needs a
  2-species ground truth the 3-gene canonical corpus structurally cannot supply.
- ~~**`docs/PREREGISTRATION.md` §1b and the §1 role amendment want owner sign-off.**~~
  **DONE — corrected 2026-08-11.** The sign-off was already given when this line was written:
  `PREREGISTRATION.md:14-15` records *"Amended 2026-08-10, on owner instruction, before any
  run against the new sets"*, and `DECISIONS.md` D-CANON-4 is marked DECIDED, both landed in
  `4060461` — 75 minutes *before* this handoff was committed. No §3 threshold was touched.
- **A `turing_holes` class is not currently possible**: only 3 gated, stable hole systems
  exist against the 5 required. Screening fresh systems with `gen_tg3` would be the route.
- **Whether genuine box-independent stripes are reachable at all** is open (D-CANON-2). The
  principled lever is the quadratic coefficient of the amplitude equation — Hill kinetics are
  generically asymmetric, which is likely why spots dominate at 54 % of the corpus.
