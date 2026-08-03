# Future work — deliberately deferred, with the reason and the starting point

Things this project knowingly did not do. Each entry says what the question is, why it was
deferred, what already exists to start from, and what would make it worth taking up. An
entry here is a *scoped decision*, not a backlog item that fell through.

---

## 1. Finite-`mu` robustness — undoing the quasi-steady-state reduction

**Status: parked 2026-08-03 at the owner's direction. Not measured. Nothing in this
repository depends on it.**

### The question

The RNGRN trains on a **quasi-steady-state** reaction (`src/rngrn/model.py::_reaction_raw`):
promoter occupancy is assumed to equilibrate *instantaneously* with transcription-factor
concentration, so the gates never appear as state variables. Real gene regulation does not do
that — TF–promoter binding is fast but finite, and `mu` parameterises the separation between
that fast binding and the slow protein production/degradation the model resolves.

Undo the reduction and the state becomes `(x, G_A, G_R)` of dimension `N + 2N²` (21 at N=3),
in which **only `x` diffuses** — promoter states are bound to DNA, so the diffusion matrix is
`diag(D_x, 0, …, 0)`. The question is whether a circuit that is robustly Turing under the QSS
reduction is still robustly Turing there.

### Why it is not idle curiosity

**Every `turing_volume_*` number in this project is computed on the QSS Jacobian.**
`PREREGISTRATION.md` §3.2 is a QSS statement, and should be worded that way in the paper.

The machinery to lift has existed since the scaffold — `eval/dynamical.py::lift_check` writes
the gate dynamics for both regulation forms — but it is called from exactly one place,
`tests/test_science.py:71`, on a 2-species model at `mu = 1e-4`, and all it checks is that the
*production algebra* converges to the QSS expression as `mu → 0`. **It has never been
simulated in space, never run at finite `mu`, never coupled to diffusion, and never asked
whether a pattern forms or survives perturbation.**

### Why it was deferred

The owner's position, verbatim: *"Separation of time scales is a whole nother problem. My
previous work leads me to believe that training under the QSS is sufficient to discover
circuits that pattern under finite mu."* and *"Just note work on mu for future work if we
have time."*

That is a scoping decision on a defensible prior, and it is recorded as one. It is **not** a
measurement, and this project has not tested it.

### What already exists to start from

| artefact | where | state |
|---|---|---|
| `eval/lifted.py` — lifted state packing, Jacobian, dispersion, fixed-point residual | branch `feature/rngrn-c-mu`, commit `17e9ad2` | **written, NOT validated, NOT tested, no results** |
| `scripts/stage0b_mu.py` — the driver | same branch | partial |
| `mucrit.json` / `mucrit.npz` / `fixedpoint.npz` | `experiments/figures_report/stage0b/arrays/` on that branch | partial, **do not cite** |
| `eval/dynamical.py::lift_check` | `main` line | the algebra, validated in the `mu → 0` limit only |
| `eval/analysis.py::robustness_cloud` | `main` line | the QSS robustness estimator to be reproduced at finite `mu` |
| `PREREGISTRATION.md` §3.6 | withdrawn, retained verbatim | the criterion as it would be written |

### The one structural fact worth keeping

Setting `dG_A/dt = dG_R/dt = 0` recovers the QSS algebraic relations **exactly**, so the
lifted fixed point is `(x*, G_A^qss(x*), G_R^qss(x*))` for **every** `mu`. `mu` changes
stability and dynamics, never the steady state. That is what makes QSS and finite-`mu`
measurements directly comparable at the same `x*`, and it is why §3.6 could inherit §3.2's
calibration rather than needing its own.

### Traps, recorded so they are not rediscovered

- **The integrator can fake the answer.** Gates relax at rate `1/mu`, so a stiff scheme that
  numerically damps the instability manufactures "patterns die at finite `mu`" for free. Any
  finite-`mu` simulation must demonstrate `dt`-convergence **and** reproduce the QSS field at
  `mu ≤ 1e-4` before its results are quoted. The natural scheme is an exact-linear update on
  the gates: `dG/dt = (G_∞(x) − G)/mu` integrates exactly over a step at frozen `x` and is
  stable at any `dt`.
- **Oscillatory ≠ Turing.** Explicit slow gates classically introduce a delay-driven Hopf
  route, so the uniform state can go unstable *before* the `k > 0` band closes. A travelling
  wave is not the pattern this project claims to recover; count stationary leading modes only
  and report the oscillatory fraction separately.
- **`mu` is itself uncertain.** It is a ratio of literature timescales — TF–DNA residence
  (seconds) against protein turnover (tens of minutes to hours) — and will not pin better
  than an order of magnitude. Perturbing `(alpha, delta, D, K)` at a fixed `mu` understates
  the exposure; a cloud that also draws `mu` from its band is the honest version.
- **Use the strict test.** `max Re eig(J) < 0`, never `tr(J) < 0`. Stage 0 measured the trace
  test overcounting by **64×** on 80,000 box-constrained draws, with all 1,196 extra
  acceptances being uniform instabilities rather than Turing patterns.
- **Name the `mu` you mean.** This codebase uses `mu` for two unrelated things, recorded at
  `STATE_OF_THE_SCIENCE.md` §713–714: the lift's fast-variable timescale (this entry) and the
  generator's degradation rate (the RNGRN's `delta`, which `scripts/stage0_bio_viability.py`
  swept). One unit has already confused them; say which you mean.

### What would make it worth taking up

If the paper claims robustness without qualification, or if a reviewer asks whether the
recovered circuits are artefacts of the QSS reduction. The cheapest decisive experiment is
the finite-`mu` robustness cloud on the two circuits that carry the claim — currently
`three_gene_qvar` `sample_0003` and `sample_0004`, prior-ON — read against §3.2's existing
bars. That is hours, not days, because the lifted Jacobian is a 21×21 eigenproblem and the
draw scheme already exists.
