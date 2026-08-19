# experiments/claim5_obs_noise/ — paper claim 5: robustness to noisy training data

Four cells, one target (`three_gene_qvar/sample_0001`), 8 seeds each, at four observation-noise
levels. Deliverable is the measured curve, not a pass/fail verdict — see
`docs/DECISIONS.md::D-CLAIM5-1` for the full design rationale, the noise mechanism, and the
disclosed deviation from materialising noisy datasets as their own registered payloads.

## Exact commands

Common flags, run through `scripts/guarded_run.sh` (mandatory — see
`.claude/skills/run-training/SKILL.md`):

```bash
bash scripts/guarded_run.sh timeout 2400 .venv/bin/python -m rngrn.cli \
  --runs-root experiments/claim5_obs_noise/sigma_XXXX target-report \
  --config configs/nc1_m3_registry.yaml --dataset-id three_gene_qvar --sample-key sample_0001 \
  --form nc1 --seeds 0 1 2 3 4 5 6 7 --workers 4 \
  -o train.batched=true -o train.device=cuda -o model.dispersion_backend=cubic \
  -o train.n_restarts=64 -o train.lbfgs_steps=0 -o train.adam_steps=400 \
  -o loss.weights.param_prior=1.0 -o loss.weights.turing=8.0 -o loss.weights.kstar=8.0 \
  [-o data.obs_noise_sigma=SIGMA -o data.obs_noise_seed=SEED]
```

The bracketed noise overrides are omitted entirely for the sigma=0.00 control (exercising the
untouched default code path, not merely a sigma=0 override).

| runs-root | sigma_rel | obs_noise_seed | wall time |
|---|---|---|---|
| `sigma_0p00` | 0.0 (control, no override) | n/a | ~21 min |
| `sigma_0p01` | 0.01 | 4201 | ~26 min |
| `sigma_0p05` | 0.05 | 4205 | ~25 min |
| `sigma_0p20` | 0.20 | 4220 | ~16 min |

The exact sequential driver used: `experiments/claim5_obs_noise/run_cells.sh`. Cells were run
one at a time (never concurrently) because they queue on the repo-global trainer flock and a
second concurrent pool would exceed the host memory ceiling even after the flock clears
(CLAUDE.md §7a). All four ran on the `c2_P_t8k8_consol` winning config (batched CUDA path,
`dispersion_backend=cubic`, m=N=3).

## Seeds

`0 1 2 3 4 5 6 7`, identical across all four levels — the same 8 restart-seed conditions are
compared at each noise level, so any difference between levels is attributable to the injected
noise and not to a different restart-seed draw.

## Caveats

- **One target, 8 seeds per level.** Not a general noise-robustness claim; see D-CLAIM5-1.
- **`sigma_rel` is relative to each observed channel's own clean std**, not an absolute noise
  floor — this dataset's three channels differ in amplitude scale.
- **Noise perturbs the observed frame only**, which also perturbs `kstar_obs` (the FFT-measured
  k-grid anchor). This is deliberate — see D-CLAIM5-1 — not a firewall leak; `kstar_obs` is
  computed from the observed frame at recovery time on every path, noisy or not.
- **Non-monotonic behaviour observed on some columns** (e.g. seed 3's rollout patterns at
  sigma 0.01 and 0.20 but not at 0.00 or 0.05). Read as seed-to-seed variance at these noise
  levels on this one target, not as a systematic reversal — reported plainly rather than
  smoothed over.
- **No calibrated pass/fail threshold exists for this claim.** The probe levels
  `{0.01, 0.05, 0.2}` are UNCALIBRATED by design; see D-CLAIM5-1 for why and what was rejected.
