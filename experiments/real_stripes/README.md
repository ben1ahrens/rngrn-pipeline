# real_stripes — RNGRN on real bacterial-colony stripe images

**Started 2026-08-19 (branch `feature/real-stripes`), owner-requested. Read the caveats before
reading any number out of this directory. Decision record: `docs/DECISIONS.md` D-REAL-1.**

## Purpose

First application of the pipeline to **real data**: 8 stripe-pattern snapshot pairs of one
bacterial colony carrying an engineered 3-node Turing circuit, with only 2 of the 3 nodes
imaged (channels C1, C2) — partial observation, m=2 < N=3. Per frame, recover an RNGRN
(nc1 form, N=3), then run the dynamical-lift simulation on the learned weights and produce
topology / parameter / dispersion comparison graphics against the real images.

There is **no ground truth**. The deliverable is therefore *not* recovery accuracy — it is
(a) cross-frame/cross-run **consistency** of the learned topology, parameters, and dispersion,
and (b) the **lifted-rollout comparison** against the real images. It is a real possibility
that no run comes out Turing-unstable; per §8 that is a reportable outcome, not a failure of
the harness.

**Final scope (2026-08-19): the analysis covers 5 of the 8 frames — 1, 2, 6, 7, 8.** Frames
3, 4, 5 trained successfully but their checkpoints were destroyed by a run-directory collision
(see Incidents below); the owner decided against rerunning them for time. Their rows in
`target_reports.jsonl` remain valid per-frame records; only the on-disk models are gone.

## Data

- Dataset: `data/datasets/stripes_colony_2ch/` — 8 samples, `(2, 96, 96)`, C1 = channel 0.
- Source: `~/projects/personal/rngrn/3N-stripes-images/preview_png` (300×300 grayscale PNGs,
  interior crops of one colony; no colony edge, so no mask).
- Ingestion: `scripts/ingest_stripes.py` — per channel: [0,1] float → least-squares 2nd-order
  polynomial illumination detrend → LANCZOS downsample to 96×96 → rescale to [0,1].
- `L = 1.0` is **arbitrary** (no scale bar); it only sets the units of D and k (CLAUDE.md §7c).
- Measured `k_star_fft` across the 8 samples: **11.44–19.93 rad/L ≈ 1.82–3.17 periods/frame**
  (payload attrs), consistent with the ~100 px dominant stripe wavelength found on the raw
  images during ingestion (D-REAL-1, Evidence).
- **Answer-key disclosure:** the loader hard-requires a per-sample `k_star`
  (`data/gate.py::from_registry`). For this real dataset the FFT peak of channel 0 was written
  into **both** `k_star` and `k_star_fft`. See the caveats.

Regenerate the dataset (deterministic, no seed involved):

```bash
.venv/bin/python scripts/ingest_stripes.py    # writes data/datasets/stripes_colony_2ch/payload.h5
```

## Config

`configs/nc1_stripes_partial.yaml` — nc1 form, `N: 3, m: 2, observed_idx: [0, 1]`, cubic
dispersion backend, `adam_steps: 400, n_restarts: 8, device: cuda`. Weights follow the claim-5
tuned nc1 recipe (`param_prior 1.0, turing 8, kstar 8`) **except** `resid: 0.1`, which the
m<N guard requires to be > 0 and which is **UNCALIBRATED** (see caveats).

## Commands

Main run — 8 frames × 1 seed (seed 0), 4-way parallel serial-CUDA processes inside one
`guarded_run` flock hold (§7a — never launch a trainer without the guard):

```bash
bash scripts/guarded_run.sh bash experiments/real_stripes/run_frames.sh
```

The driver (`run_frames.sh`) pins torch threads to 1 per process (CPU is in use elsewhere;
serial-CUDA needs the CPU only for python-side stepping), applies a 7200 s timeout per frame
(3600 s proved too short under 4-way GPU contention — see Incidents), sets a **per-sample
`tracking.run_name`** (load-bearing — see Incidents), and writes per-frame logs to
`experiments/real_stripes/sample_00NN.log`. Each frame runs:

```bash
.venv/bin/python -m rngrn.cli --runs-root experiments/real_stripes target-report \
  --config configs/nc1_stripes_partial.yaml \
  --dataset-id stripes_colony_2ch --sample-key sample_00NN \
  --form nc1 --seeds 0 --workers 1
```

Analysis + graphics (topologies, parameter families, per-frame and overlaid dispersion
relations, lifted rollout vs observed frame) over every valid run:

```bash
KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python experiments/real_stripes/analyze.py
```

Figures land in `experiments/figures_report/real_stripes/` (tracked); the numbers behind them
in `analysis_summary.json`. The script detects run-dir collisions (checkpoint `kstar_obs` vs
the sample's own anchor) and DROPS clobbered samples loudly rather than analysing frame A's
model under frame B's name.

## Layout

- `run_frames.sh` — the driver above.
- `analyze.py` — checkpoint → lift → graphics, with the collision guard.
- `analysis_summary.json` — per-sample numbers behind the figures.
- `sample_00NN.log` — per-frame stdout/stderr.
- `runs/` — run directories (`stripes_partial_<timestamp>_seed0`), with the usual
  `config/frozen_config.yaml`, `results/`, `checkpoints/`, `figures/`. Read the effective
  config from `frozen_config.yaml`, never from the command line (D-EVID-1).
- `runs.jsonl`, `target_reports.jsonl` — append-only indexes.
- `smoke/` — a 1-restart, 5-step plumbing check. Its numbers are **not findings**; the only
  sayable sentence about it is "the harness runs" (§8).

## Incidents — the 2026-08-19 batch

1. **Timeout kill (launch 1).** Frames 1–4 launched with `timeout 3600`; under 4-way GPU
   contention a frame takes ~57–84 min, so frames 1, 3, 4 were killed at the wire (exit 124,
   19:35). Frame 2 finished. Fix: 7200 s in the driver.
2. **Run-directory collision (launch 2) — the one that cost data.** Run dirs are named
   `<run_name>_<UTC-second>_seedN`. The relaunched frames 1, 3, 4, 5 all started within the
   same wall-clock second, so all four wrote to ONE directory
   (`stripes_partial_20260819_183655_seed0`); the last finisher (frame 1, 21:01) owned the
   surviving checkpoint and frames 3, 4, 5's models were silently destroyed. Detected by
   `analyze.py`'s anchor-vs-checkpoint `kstar_obs` cross-check; confirmed via the
   `run_ids` in `target_reports.jsonl` (four samples → one run_id). Fix, now in the driver
   and mandatory for any rerun: `-o tracking.run_name="stripes_${s}"` makes the run dir
   unique per frame regardless of start second. The owner decided not to rerun 3, 4, 5
   (deadline), so the final analysis covers frames 1, 2, 6, 7, 8.

## Caveats — read before quoting anything

1. **Every k\*-vs-truth score here is CIRCULAR (D-REAL-1).** The sample `k_star` *is* the FFT
   peak the training objective anchors to, because real data has no answer key and the loader
   requires the attribute. `kstar_rel_err` on these runs measures self-consistency with the
   training anchor, never accuracy against truth, and is not comparable to the same metric on
   generated datasets.
2. **Recovery quality is unmeasurable.** No generating model exists. Report cross-run
   consistency and the lifted-simulation comparison; do not report "recovered the circuit".
3. **`resid = 0.1` is UNCALIBRATED.** Required > 0 by the m<N guard in `recover.py`; the only
   related measurement (`experiments/exp06_residual_sweep.json`) found the residual harmful in
   the *fully observed* case. No hidden-channel calibration exists (TUNING.md Milestone 2).
4. **1 seed per frame, 8 restarts (vs the tuned 64) — indicative only, per owner scoping.**
   The batched path refuses m<N, so restarts are serial; CUDA per owner instruction despite
   §7's measurement that the serial path is slower on GPU than CPU.
5. **L is a unit, not a measurement.** All recovered D and k values are in arbitrary
   per-frame units; only dimensionless combinations (e.g. periods per frame, D ratios)
   transfer to the physical colony.
6. **"Turing-unstable" and "patterns" remain different claims** (§7c). The lifted rollout, not
   the dispersion relation, is what closes that gap — and a null result there is reportable.
