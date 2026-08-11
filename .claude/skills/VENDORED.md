# Vendored skills — provenance

Skills in this directory are of two kinds. **House skills** (`run-training`, `new-worktree`,
`record-decision`, `harvest-dataset`) are written for this repository and are described in
`CLAUDE.md` §12. **Vendored skills** are copied verbatim from an upstream repository and are
listed below.

Vendored skills are **unmodified**. Do not edit them in place — an edit silently forks the
skill and the next update overwrites it. If one needs a project-specific caveat, record the
caveat here rather than in the vendored file.

| Skill | Upstream | Path | Pinned at | Vendored |
| --- | --- | --- | --- | --- |
| `optimize-for-gpu` | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | `skills/optimize-for-gpu` | `d661d27` | 2026-08-11 |
| `statistical-analysis` | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | `skills/statistical-analysis` | `d661d27` | 2026-08-11 |

Both are MIT-licensed, authored by K-Dense, Inc.

## Why these two

`optimize-for-gpu` is here for the §7 direction of travel — batching restarts × samples so the
GPU stops losing on the serial path. Two things make it usable under §8 rather than a source of
speed claims: it treats the numerical contract as something to preserve and verify (SKILL.md
§1, §5), and it insists GPU timing use CUDA events rather than a CPU timer around an
asynchronous call (SKILL.md §6). The most relevant single line for this repo is its rule *not*
to move code out of a GPU-native framework to use RAPIDS — for PyTorch code it routes you to
removing host round trips and batching, which is exactly what `losses/terms.py::_damped_newton`
needs.

`statistical-analysis` is here to calibrate the thresholds §8 marks as UNCALIBRATED — the ~15 %
k\* tolerance (D-EVID-17) and `coupling_threshold=0.05`. It covers test selection, assumption
checking, effect sizes, power analysis and Bayesian alternatives, and it opens by requiring the
test be committed to *before* looking at results, which is the same discipline as
`docs/PREREGISTRATION.md`.

## Caveats for this repository

- **float64 is safe.** Both were checked for an implicit float32 assumption and neither has one.
  `optimize-for-gpu` gates precision on the contract — *"prefer float32 over float64 when the
  numerical contract allows it"*, with *"validate accuracy"* attached. It will not push you off
  `torch.set_default_dtype(torch.float64)` unprompted.
- **Most of `optimize-for-gpu`'s 15 reference files do not apply here.** They cover cuDF, cuML,
  cuGraph, cuCIM, cuVS, cuSpatial and cuxfilter. This project is PyTorch-native with N ≤ 3
  matrices. The reference files that could matter are `cupy.md`, `numba.md`, `warp.md` and
  `raft.md`. All 15 are vendored anyway, unmodified, because SKILL.md links them by relative
  path and a missing file would fail mid-task. They cost disk, not context — only `SKILL.md`
  loads by default.
- **`statistical-analysis` wants extra dependencies** (`pingouin`, `pymc`, `arviz`) that are not
  in this project's `pyproject.toml`. Installing them is a deliberate act, not something to let
  the skill do inside a worktree venv without thinking about it.
- **`scripts/assumption_checks.py` (652 lines) was reviewed before vendoring.** It imports only
  numpy, pandas, scipy, matplotlib, seaborn and statsmodels — no subprocess, network access, or
  `eval`/`exec`.

## Considered and rejected

Two PyTorch skills from `pytorch/pytorch/.claude/skills/` were fetched, read and **not** kept:

- **`cuda-index-width`** — entirely about 32- vs 64-bit index math in ATen CUDA C++
  (`AT_DISPATCH_INDEX_TYPES`, `canUse32BitIndexMath`, `libtorch_cuda.so` binary size). This
  repository writes no CUDA C++, and its tensors are N ≤ 3; nothing here approaches the 2³¹
  element boundary the skill exists to handle.
- **`pt2-bug-basher`** — written for contributors *to PyTorch itself*: it adds regression tests
  to `test/dynamo/test_repros.py`, validates against PyTorch's `main`, and calls PyTorch's own
  `/pr-review`. Its step 5 also instructs the agent to create a worktree and then
  `ExitWorktree (action: remove)`. Automatic worktree removal is the exact shape of the failure
  in `CLAUDE.md` §6a, where a removed worktree destroyed generated `payload.h5` files. Not worth
  the hazard for the parts that generalise, which are two environment variables:
  `TORCH_LOGS="+dynamo,graph_breaks,recompiles"` and `TORCH_COMPILE_DEBUG=1`.

## Updating

Re-fetch the whole directory from upstream and bump the pinned SHA above; never patch in place.
Diff before accepting — an upstream change to a vendored skill changes agent behaviour in this
repository without touching any of its source.
