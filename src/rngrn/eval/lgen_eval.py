"""lgen_eval.py — DOES THE RECOVERED GRN PATTERN THE SAME WAY ON A DIFFERENT-SIZED BOX?

THE PREDICTION BEING TESTED (this is why the module exists)
----------------------------------------------------------
The selected wavenumber k* of a reaction-diffusion system is set by the reaction and the
diffusion, not by the container: it is the argmax of sigma(k) = eig(J - k^2 D), and neither
J nor D knows how big the domain is. So for one recovered model simulated on domains of
several sizes L, Turing theory predicts, sharply:

  * the PHYSICAL wavelength — and hence k* in rad/length — is INVARIANT in L;
  * the number of periods across the box scales LINEARLY, q(L) = L*k*/(2*pi);
  * the morphology CLASS (spots / stripes / labyrinth) is preserved.

The failure mode this instrument exists to catch is the opposite one: a pattern that simply
refits whatever periodicity the box wants, so q stays put while k* moves as 1/L. That is a
model that memorised a box, not a network that generalises. The two outcomes are
distinguished by the two headline numbers below, and they are far apart:

    for L/L_train in (0.5, 1, 2, 4) and a system with q(L_train) = 6
                                     kstar_phys_cv     periods_slope_rel_err
      k* invariant (generalises)         0.00                  0.00
      q  pinned to the box (memorised)   0.71                  0.62

(the second row is arithmetic on q == 6 exactly at every L, and it is exactly the negative
control asserted in tests/test_lgen_eval.py.)

NO PASS/FAIL LIVES HERE. `summarise_across_L` returns the raw quantities and a
`verdict_components` dict; thresholds are pre-registered elsewhere (docs/PREREGISTRATION.md)
and applied by the caller. Nothing in this module returns a boolean verdict.

THE GRID RULE — dx IS HELD CONSTANT, n SCALES WITH L
---------------------------------------------------
Changing L at a fixed n changes dx = L/n, so a 4x box at a fixed grid has a quarter of the
pixels per wavelength and its k* estimate degrades — which would show up in
`kstar_phys_cv` as if it were physics. This module therefore scales the grid with the
domain:

    n(L) = nearest EVEN integer to  n_grid * L / L_train        (n_grid = the grid at L_train)

so dx, and hence `pixels_per_wavelength`, is the same at every L to within one pixel.
`pixels_per_wavelength` is recorded on every row regardless, and a row that measures a
pattern below `PPW_FLOOR` RAISES rather than reporting an unresolvable k* as a number.

MEASURED, on the Turing fixture of tests/test_rollout.py (L=60, q~6.2, etdrk4_rfft,
seed 0, collapse-only early stop), varying n at fixed L so that pixels-per-wavelength
sweeps 2.8 -> 30.8, k* referenced to the n=192 run:

    n      16     24     32     48     64     96    128    192
    px/wl  2.76   3.94   5.12   7.64  10.10  14.98  20.32  30.79
    |dk*|  6.9%   2.4%   0.3%   0.8%   1.6%   2.8%   1.0%     --
    class  spots  laby   laby   laby   laby   laby   laby   laby

Two things in that table set PPW_FLOOR. k* is stable to ~3% for every grid down to
px/wl 3.9 and jumps to 6.9% at 2.76; the morphology CLASS survives down to 3.94 and flips
at 2.76. The floor is set at 6.0, ABOVE the measured breakdown with margin, because 6.0 is
also where the dataset generator already sat at its coarsest (`three_gene_qvar`/`multiL`
draw periods-per-box p up to 14 on a 96 grid, i.e. 96/14 = 6.86 px/wl), so a floor of 6
refuses exactly the resolutions the training data never contained. It is a MEASURED
conservative bound, not a tuned one; the +-3% grid-to-grid spread in the table is also the
noise floor of `kstar_phys_cv` and no CV below it should be read as meaningful.

A SECOND, INDEPENDENT RESOLUTION LIMIT, which the grid rule does NOT fix
-----------------------------------------------------------------------
`observables.raps` bins the spectrum on the fundamental dk = 2*pi/L, so the k* estimate is
quantised at a RELATIVE width of dk/k* = 1/q. A small box holds few periods and its k* is
therefore coarsely measured no matter how fine the grid: at q = 3 the bin is 33% wide.
Every row records `kstar_bin_width` and `kstar_bin_width_rel` for this reason. Read
`kstar_phys_cv` next to the largest `kstar_bin_width_rel` in the table; a CV of the same
order is a measurement floor, not a physical spread. Below q = 1 the box cannot hold one
period and k* is meaningless — the recorded diagnostic is what makes that visible, and this
module does NOT invent a q threshold on top of it.

THE MORPHOLOGY CLASS IS THE WEAKEST OF THE THREE PREDICTIONS — MEASURED
----------------------------------------------------------------------
On the same fixture across L/L_train = (0.5, 1, 2, 4), the two INTENSITY features barely
move while the two ANGULAR features drift monotonically with q:

    L      q      skew   kurtosis   ang_conc   ang_entropy   class (centroid bank)
     30   3.23   0.507    -0.806      0.190       0.782      labyrinth  (margin 0.62)
     60   6.34   0.461    -0.739      0.172       0.913      labyrinth  (margin 0.28)
    120  12.25   0.459    -0.776      0.104       0.980      spots      (margin 0.17)
    240  24.64   0.428    -0.751      0.074       0.995      spots      (margin 0.24)

skew and kurtosis — which have no Fourier grid in them — are invariant to <10%, i.e. the
TEXTURE is the same at every L. `ang_conc`/`ang_entropy` are computed on a ring of radius q
in pixel-frequency units, so a bigger box spreads the same power over more ring pixels and
more angles; that is a property of the measurement, not of the pattern. Compounding it, the
default centroid bank was fitted on samples with q ~ 6 on a 96 grid, so a q = 25 field is
out of its distribution, and all four calls above are borderline (margin 0.17-0.62 against
a median 0.49 for correct calls). So `morphology_class_preserved` = 1/3 here does NOT
license "the morphology changed". The raw `morph_vector` is recorded on every row precisely
so a reader can see WHICH features moved, and the class fraction should be read as the
weakest of the three predictions, not the headline. It fails in the other direction too:
the box-pinned NEGATIVE control in tests/test_lgen_eval.py preserves its class perfectly
(1.0) while k* moves as 1/L, so class agreement on its own is not evidence of anything.

THE NONDIM TRAP
---------------
`recover()` reports `params["D"]` as physical on both paths, but the SAVED MODEL keeps
`theta_D` in the units the objective ran in: on the non-dimensional path (`nondim=True`)
`model.D` is D/L_train**2 (recover.py converts for the REPORTED params only). Simulating a
reloaded nondim checkpoint with that D integrates the wrong diffusivity, silently, and the
resulting k* is wrong by a factor of L_train. `physical_model_from_checkpoint` performs the
conversion, verifies it, and RAISES on a checkpoint that does not carry the `nondim`/`L`
extras rather than assuming the dimensional path.

FIREWALL
--------
This module simulates a RECOVERED model and needs no ground truth whatsoever: it imports
`.rollout`, `..observables` and `..scoring.morphology` (itself image-only) and nothing from
the answer-key side. It is NOT on `tests/test_firewall.py`'s RECOVERY_SIDE audit list,
because it imports the scoring package — it runs strictly after recovery, like
`scoring/*`, and must never be called from inside the objective.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ..scoring import morphology as MORPH
from .rollout import simulate

__all__ = [
    "PPW_FLOOR",
    "PPW_FLOOR_PROVENANCE",
    "DEFAULT_N_GRID",
    "grid_for_L",
    "physical_model_from_checkpoint",
    "evaluate_across_L",
    "summarise_across_L",
]

TWO_PI = 2.0 * math.pi

# Minimum pixels per measured wavelength. MEASURED — see the module docstring's table for
# the scan this is set from and why it sits above the observed breakdown.
PPW_FLOOR = 6.0
PPW_FLOOR_PROVENANCE = (
    "measured on the tests/test_rollout.py Turing fixture (L=60, q~6.2): k* stays within "
    "3% of the n=192 reference down to 3.94 px/wavelength and moves 6.9% at 2.76, where "
    "the morphology class also flips labyrinth->spots; 6.0 additionally matches the "
    "coarsest resolution the generators ever produced (p=14 periods on a 96 grid = 6.86 "
    "px/wavelength)"
)

# The grid at L_train when the caller does not say. 96 is the grid every registered
# three_gene payload was generated on (data/datasets/*/manifest.json, `grid`).
DEFAULT_N_GRID = 96


# --------------------------------------------------------------------------------------
# the grid rule
# --------------------------------------------------------------------------------------
def grid_for_L(L: float, L_train: float, n_grid: int, n_max: int = 512,
               n_min: int = 16) -> int:
    """Grid size at domain size `L` that holds dx = L/n fixed at its L_train value.

    Returns the nearest EVEN integer to n_grid * L / L_train (even because the rfft
    integrators and the FFT bin layout are cleanest on an even grid; the rounding moves dx
    by at most one part in n).

    Raises rather than clamping at either end. A clamp would silently change dx — which is
    the one thing this rule exists to hold fixed — and then report a `pixels_per_wavelength`
    the caller never asked for.
    """
    for name, v in (("L", L), ("L_train", L_train)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and > 0; got {v!r}")
    if n_grid < n_min:
        raise ValueError(f"n_grid must be >= n_min={n_min}; got {n_grid}")
    n = int(2 * round(n_grid * (L / L_train) / 2.0))
    if n < n_min:
        raise ValueError(
            f"holding dx fixed at L={L:g} would need a {n}x{n} grid, below n_min={n_min}. "
            f"Raise n_grid (currently {n_grid} at L_train={L_train:g}) or drop this L.")
    if n > n_max:
        raise ValueError(
            f"holding dx fixed at L={L:g} would need a {n}x{n} grid, above n_max={n_max}. "
            f"Lower n_grid (currently {n_grid} at L_train={L_train:g}), shrink the L range, "
            f"or raise n_max deliberately and accept the cost.")
    return n


# --------------------------------------------------------------------------------------
# the nondim trap
# --------------------------------------------------------------------------------------
def physical_model_from_checkpoint(model, payload):
    """Return (model_with_PHYSICAL_D, L_train) for a checkpoint loaded by io.load_checkpoint.

    On the non-dimensional recovery path the checkpointed `theta_D` is log(D/L_train**2), so
    `model.D` is dimensionless. Any rollout of that model at a physical L integrates the
    wrong diffusivity and produces a k* wrong by a factor of L_train, with nothing in the
    output saying so. This applies the inverse map D = D_model * L_train**2 by rewriting
    `theta_D` on a DEEP COPY (the caller's model is left untouched) and then verifies that
    `model.D` really is the physical vector.

    RAISES if the payload lacks `nondim` or `L`. `io.save_checkpoint` has stored both since
    unit 12, but checkpoints written before that carry neither, and there is no way to tell
    a dimensional run from a non-dimensional one after the fact — so an older checkpoint is
    refused rather than assumed dimensional.

    On the dimensional path this is a no-op apart from the copy: theta_D is already
    physical, and it is returned unmodified.
    """
    for key in ("nondim", "L"):
        if key not in payload:
            raise KeyError(
                f"checkpoint payload has no '{key}' extra, so its diffusivities cannot be "
                f"placed in physical units: on the non-dimensional path theta_D is "
                f"log(D/L**2) and on the dimensional path it is log(D), and the two are "
                f"indistinguishable without this flag. Keys present: {sorted(payload)}. "
                f"Re-run recovery (io.save_checkpoint has recorded nondim and L since unit "
                f"12) rather than guessing.")
    nondim = bool(payload["nondim"])
    L_train = float(payload["L"])
    if not np.isfinite(L_train) or L_train <= 0:
        raise ValueError(f"checkpoint L must be finite and > 0; got {payload['L']!r}")

    out = copy.deepcopy(model)
    if not nondim:
        return out, L_train

    D_model = out.D.detach().cpu().numpy().astype(float)
    D_phys = D_model * (L_train ** 2)
    with torch.no_grad():
        out.theta_D.copy_(torch.log(torch.as_tensor(
            D_phys, dtype=out.theta_D.dtype, device=out.theta_D.device)))
    got = out.D.detach().cpu().numpy().astype(float)
    if not np.allclose(got, D_phys, rtol=1e-10, atol=0.0):
        raise RuntimeError(
            f"non-dimensional -> physical D conversion did not take: wanted {D_phys!r}, "
            f"model.D is {got!r}. Refusing to simulate a model whose diffusivity is not "
            f"the one asked for.")
    return out, L_train


# --------------------------------------------------------------------------------------
# per-L evaluation
# --------------------------------------------------------------------------------------
def _dedup_L(L_values, L_train):
    """The L to evaluate at: the caller's, with L_train prepended if it is not among them.

    L_train is always evaluated because it supplies the morphology REFERENCE class, and
    because it is the one size at which the model's behaviour is already known — a reader
    comparing the other rows to it needs it measured the same way, not quoted from a
    different run. Near-duplicates (within 1e-9 relative) are collapsed so the same domain
    is never simulated twice and never double-counted in the summary.
    """
    out: list[float] = []
    for v in [L_train] + [float(x) for x in L_values]:
        v = float(v)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"every L must be finite and > 0; got {v!r}")
        if not any(abs(v - u) <= 1e-9 * max(abs(v), abs(u)) for u in out):
            out.append(v)
    return out


def evaluate_across_L(model, L_train, L_values, *, n_grid=None, seed=0,
                      reference_bank=None, ppw_floor: float = PPW_FLOOR,
                      n_max: int = 512, grid_rule: str = "fixed", keep_fields: bool = False,
                      **sim_kw) -> dict:
    """Simulate ONE recovered model on several domain sizes and measure L-generalisation.

    model     : a recovered RNGRN whose `D` is PHYSICAL. If it came from a checkpoint, pass
                it through `physical_model_from_checkpoint` FIRST — see the module docstring
                on the nondim trap.
    L_train   : the domain size recovery ran on. Supplies the reference morphology class and
                the L/L_train column, and anchors the grid rule.
    L_values  : the domain sizes to evaluate. L_train is added if absent (see `_dedup_L`).
    n_grid    : the grid to simulate on. Defaults to DEFAULT_N_GRID (96, the registered
                payloads' grid).
    grid_rule : 'fixed' (DEFAULT) simulates every L on the SAME n_grid. 'constant_dx' scales
                the grid with L via `grid_for_L` so dx — and hence pixels per wavelength —
                is held at its L_train value.

                'fixed' is the default by owner decision (2026-07-30: "resolution does not
                need to be the same. we can drop resolution to make it faster"), and the cost
                is bounded rather than ignored. Under 'constant_dx' a 3.25x L range needs a
                3.25x grid, i.e. ~10x the cells and ~10x the time at the largest L — the
                dominant cost of the whole evaluation. Under 'fixed', dx grows with L so
                pixels per wavelength FALLS as 1/L, and the protection is `ppw_floor`: a
                patterned row below it raises rather than being reported. Worked example on
                three_gene_multiL system 13 (k*=0.4522, wavelength 13.9, L 55.6->180.7 at
                n=96): 24.0, 13.7, 9.6, 7.4 px/wavelength — the widest real L span in the
                dataset still clears the floor of 6.0 at a fixed 96 grid.

                Use 'constant_dx' when a row would otherwise breach the floor, or when
                comparing k* across L to a precision finer than one FFT bin (the bin width
                itself scales with 1/L at fixed n, and `kstar_bin_width_rel` per row is what
                tells you whether the comparison is bin-limited).
    seed      : the initial-noise seed, the same at every L. The noise FIELD still differs
                between L because the grid differs, so this is reproducibility, not a
                controlled initial condition.
    reference_bank : morphology bank for the class call. Defaults to
                MORPH.default_reference_bank() — the baked-in centroids, measurably weaker
                than a bank of real fields (82.1% vs 89.7% held-out accuracy). Pass
                MORPH.build_reference_bank(...) when the dataset is reachable.
    keep_fields : attach each L's simulated (N, H, W) field to its row under 'field'.
                OFF by default because the fields are ~110 KB each at 96x96/N=3 and most
                callers only want the scalars. Needed to PERSIST the fields — "the pattern at
                each domain size" is itself a figure, so `plotdata.save_lgen_fields` writes
                them to an npz next to the cross-L index rows. The field is the rollout's own
                array, not a copy.
    ppw_floor : minimum pixels per measured wavelength. A patterned row below it RAISES.
    **sim_kw  : forwarded to `rollout.simulate`. `L` and `n` are set by this function and
                are rejected if passed; `record_kstar=False` is rejected because k* is the
                measurement. Defaults applied here: `integrator='etdrk4_rfft'` (identical
                scheme to etdrk4 to 1e-12, ~1.8x cheaper), and `early_stop=True` with
                `saturation_tol=0.0`, which enables ONLY the collapse stop — the rule that
                cannot change a verdict — and disables the uncalibrated saturation rule, so
                no row is truncated by a knob nobody calibrated.

    The steady state x* is computed ONCE and reused at every L, so every row starts from the
    same uniform state perturbed by noise. x* is a property of the reaction alone and does
    not depend on the box; computing it per row would only add solver noise.

    Returns dict(L_train, n_grid, grid_rule, ppw_floor, per_L=[rows], summary={...}).
    Each row carries L, L_over_L_train, n, dx, kstar_phys, periods_per_box,
    pixels_per_wavelength, kstar_bin_width(_rel), patterned, amplitude, morph_vector,
    morph_class, morph_margin, nsteps_run, stopped_reason, seconds, is_L_train.
    """
    for bad in ("L", "n"):
        if bad in sim_kw:
            raise TypeError(
                f"evaluate_across_L sets '{bad}' itself (L from L_values, n from the grid "
                f"rule); passing it would silently break the fixed-dx contract")
    if sim_kw.get("record_kstar", True) is not True:
        raise ValueError("record_kstar=False leaves nothing to measure; k* IS the metric")
    n_grid = DEFAULT_N_GRID if n_grid is None else int(n_grid)
    if not np.isfinite(ppw_floor) or ppw_floor <= 0:
        raise ValueError(f"ppw_floor must be finite and > 0; got {ppw_floor!r}")
    L_train = float(L_train)
    L_eval = _dedup_L(L_values, L_train)
    bank = MORPH.default_reference_bank() if reference_bank is None else reference_bank

    sim_kw.setdefault("integrator", "etdrk4_rfft")
    sim_kw.setdefault("early_stop", True)
    sim_kw.setdefault("check_every", 200)
    sim_kw.setdefault("saturation_tol", 0.0)
    if "xstar" not in sim_kw:
        from ..losses.terms import steady_state
        xs, _ = steady_state(model)
        sim_kw["xstar"] = xs.detach().cpu().numpy()

    if grid_rule not in ("fixed", "constant_dx"):
        raise ValueError(
            f"grid_rule must be 'fixed' or 'constant_dx'; got {grid_rule!r}")

    per_L = []
    for L in L_eval:
        n = n_grid if grid_rule == "fixed" else grid_for_L(L, L_train, n_grid, n_max=n_max)
        res = simulate(model, L=L, n=n, seed=seed, **sim_kw)
        row = _row_for(res, L=L, L_train=L_train, n=n, bank=bank)
        if keep_fields:
            row["field"] = res["fields"]
        if row["patterned"] and row["pixels_per_wavelength"] < ppw_floor:
            # The remedy depends on the grid rule, so compute it per rule rather than
            # quoting one formula for both. Under 'fixed' the grid at THIS L is n_grid
            # itself, so clearing the floor needs n_grid >= ppw_floor * q. Under
            # 'constant_dx' the grid here is n_grid * L/L_train, so the requirement on
            # n_grid carries the L_train/L factor. Getting this backwards sends the reader
            # to a grid size that does not fix their problem.
            need = ppw_floor * row["periods_per_box"]
            if grid_rule == "constant_dx":
                need *= L_train / L
            hint = (f"n_grid >= {int(math.ceil(need))} would clear it here under "
                    f"grid_rule={grid_rule!r}")
            if grid_rule == "fixed":
                hint += (", or pass grid_rule='constant_dx' to scale the grid with L "
                         "(~10x the cost at a 3.25x L span)")
            raise ValueError(
                f"L={L:g} produced a pattern at {row['pixels_per_wavelength']:.2f} pixels "
                f"per wavelength, below the floor of {ppw_floor:g} "
                f"(k*={row['kstar_phys']:.4g}, q={row['periods_per_box']:.3g}, grid {n}). "
                f"Below the floor the k* estimate is not merely noisy, it is wrong "
                f"(see PPW_FLOOR_PROVENANCE). {hint} — rather than reading this row as a "
                f"measurement.")
        per_L.append(row)

    return dict(
        L_train=L_train,
        n_grid=n_grid,
        grid_rule=grid_rule,
        grid_rule_detail=(
            f"n(L) = {n_grid} at every L (grid FIXED; dx grows with L, so "
            f"pixels_per_wavelength falls — read it per row)"
            if grid_rule == "fixed" else
            f"n(L) = nearest even int to {n_grid} * L / {L_train:g} "
            f"(dx held fixed; see grid_for_L)"),
        ppw_floor=float(ppw_floor),
        per_L=per_L,
        summary=summarise_across_L(per_L),
    )


def _row_for(res, L, L_train, n, bank) -> dict:
    """One per-L row from a `rollout.simulate` result. No ground truth is read."""
    amp = float(res["amplitude"]) if np.isfinite(res["amplitude"]) else float("nan")
    patterned = bool(res["patterned"])
    kstar = float(res["kstar"])
    # DELIBERATE NaN, twice over, and both are results rather than errors: `simulate`
    # reports kstar=NaN for a field with no amplitude to measure, and a wavelength that was
    # never formed has no pixels-per-wavelength and no periods-per-box either.
    q = L * kstar / TWO_PI if np.isfinite(kstar) else float("nan")
    ppw = n / q if (np.isfinite(q) and q > 0) else float("nan")

    # An UNPATTERNED field is float noise. train._morphology_rollout withholds such a field
    # from morphology scoring for exactly this reason — classifying noise and calling it a
    # class MATCH is a false positive — so the same guard is applied here, and the class is
    # None rather than a label nobody should trust.
    morph_vec, morph_class, morph_margin = None, None, float("nan")
    if patterned and not res["blew_up"]:
        field0 = np.asarray(res["fields"][0], dtype=float)
        if np.all(np.isfinite(field0)):
            morph_vec = MORPH.morphology_vector(field0)
            call = MORPH.classify_morphology(field0, bank)
            morph_class, morph_margin = call.label, float(call.margin)

    return dict(
        L=float(L),
        L_over_L_train=float(L / L_train),
        is_L_train=bool(abs(L - L_train) <= 1e-9 * max(abs(L), abs(L_train))),
        n=int(n),
        dx=float(L / n),
        kstar_phys=kstar,
        periods_per_box=float(q),
        pixels_per_wavelength=float(ppw),
        # the RAPS bin width: the k* estimate's quantisation, INDEPENDENT of the grid.
        kstar_bin_width=float(TWO_PI / L),
        kstar_bin_width_rel=float((TWO_PI / L) / kstar) if np.isfinite(kstar) and kstar > 0
        else float("nan"),
        patterned=patterned,
        amplitude=amp,
        blew_up=bool(res["blew_up"]),
        morph_vector=(None if morph_vec is None else [float(v) for v in morph_vec]),
        morph_feature_order=list(MORPH.FEATURE_ORDER),
        morph_class=morph_class,
        morph_margin=morph_margin,
        sig_max=float(res["sig_max"]),
        nsteps_run=int(res["nsteps_run"]),
        stopped_reason=str(res["stopped_reason"]),
        seconds=float(res["seconds"]),
    )


# --------------------------------------------------------------------------------------
# the summary — raw quantities only, no verdict
# --------------------------------------------------------------------------------------
def summarise_across_L(per_L) -> dict:
    """Cross-L summary statistics of the rows produced by `evaluate_across_L`.

    Separated out and public because it is the piece the NEGATIVE control exercises: a
    synthetic table whose periodicity is pinned to the box (q constant, k* ~ 1/L) is fed
    through this exact arithmetic to check the metric notices. See tests/test_lgen_eval.py.

    kstar_phys_cv
        Population standard deviation (ddof=0) of kstar_phys over the PATTERNED rows,
        divided by their mean. THE HEADLINE: invariance means ~0. Population rather than
        sample SD so the number does not inflate with 3 or 4 L values, which is the whole
        range this is ever used over. Read it against the largest kstar_bin_width_rel in
        the table and against the ~3% grid-to-grid noise floor in the module docstring.

    periods_slope_rel_err
        Fit periods_per_box against L through the origin, slope = sum(L*q)/sum(L**2), and
        compare it to the theory slope mean(kstar_phys)/(2*pi). Relative error of the fit
        against the prediction. Because q_i == L_i*k_i/(2*pi) identically, the fitted slope
        is the L**2-WEIGHTED mean of k_i/(2*pi) while the prediction is the unweighted mean:
        the two agree exactly when k* is invariant and diverge when k* trends with L, which
        makes this a second, differently-weighted view of the same invariance rather than an
        independent measurement. Report both slopes so that is visible.

    morphology_class_preserved
        Fraction of the patterned NON-reference rows whose class equals the class at L_train.
        `n_class_compared` is the denominator, reported because rows that did not pattern
        have no class and are excluded rather than scored as mismatches — excluding them
        flatters the fraction, so the two numbers must be read together.

    DELIBERATE NaN: every statistic above is NaN when fewer than 2 patterned rows are
    available (a CV or a slope over one point is not a spread), and
    morphology_class_preserved is NaN when the L_train row itself did not pattern (there is
    then no reference class to preserve). NaN here means "not measurable from these rows",
    and `n_patterned` / `n_L` / `n_class_compared` say why.

    No pass/fail boolean is returned, by design: thresholds are pre-registered and applied
    by the caller.
    """
    rows = list(per_L)
    if not rows:
        raise ValueError("summarise_across_L needs at least one per-L row; got none")
    for r in rows:
        missing = {"L", "kstar_phys", "periods_per_box", "patterned",
                   "morph_class", "is_L_train"} - set(r)
        if missing:
            raise KeyError(f"per-L row is missing {sorted(missing)}: {r!r}")

    used = [r for r in rows if r["patterned"] and np.isfinite(r["kstar_phys"])
            and r["kstar_phys"] > 0]
    n_used = len(used)
    ks = np.array([float(r["kstar_phys"]) for r in used])
    Ls = np.array([float(r["L"]) for r in used])
    qs = np.array([float(r["periods_per_box"]) for r in used])

    if n_used >= 2:
        kstar_mean = float(ks.mean())
        kstar_cv = float(ks.std(ddof=0) / kstar_mean)
        slope_fit = float(np.sum(Ls * qs) / np.sum(Ls ** 2))
        slope_pred = kstar_mean / TWO_PI
        slope_rel_err = float(abs(slope_fit - slope_pred) / slope_pred)
    else:
        kstar_mean = float(ks.mean()) if n_used == 1 else float("nan")
        kstar_cv = float("nan")
        slope_fit = float("nan")
        slope_pred = float("nan")
        slope_rel_err = float("nan")

    ref = [r for r in rows if r["is_L_train"]]
    if len(ref) != 1:
        raise ValueError(
            f"expected exactly one row flagged is_L_train, got {len(ref)}. The reference "
            f"class is read off that row, so a missing or duplicated one would silently "
            f"change what morphology_class_preserved compares against.")
    ref_class = ref[0]["morph_class"]
    compared = [r for r in used if not r["is_L_train"] and r["morph_class"] is not None]
    if ref_class is None or not compared:
        class_preserved = float("nan")
    else:
        class_preserved = float(np.mean([r["morph_class"] == ref_class for r in compared]))

    return dict(
        kstar_phys_cv=kstar_cv,
        kstar_phys_mean=kstar_mean,
        periods_slope_fit=slope_fit,
        periods_slope_predicted=slope_pred,
        periods_slope_rel_err=slope_rel_err,
        morphology_class_preserved=class_preserved,
        morphology_class_at_L_train=ref_class,
        n_class_compared=len(compared),
        n_patterned=n_used,
        n_L=len(rows),
        max_kstar_bin_width_rel=float(
            max((r.get("kstar_bin_width_rel", float("nan")) for r in used),
                default=float("nan"))),
        min_pixels_per_wavelength=float(
            min((r.get("pixels_per_wavelength", float("nan")) for r in used),
                default=float("nan"))),
        L_over_L_train_values=[float(r.get("L_over_L_train", float("nan"))) for r in rows],
        # The raw quantities a pre-registered threshold is applied to, gathered in one
        # place. Deliberately contains NO boolean: this module does not decide.
        verdict_components=dict(
            kstar_phys_cv=kstar_cv,
            periods_slope_rel_err=slope_rel_err,
            morphology_class_preserved=class_preserved,
            n_patterned=n_used,
            n_L=len(rows),
            n_class_compared=len(compared),
            max_kstar_bin_width_rel=float(
                max((r.get("kstar_bin_width_rel", float("nan")) for r in used),
                    default=float("nan"))),
        ),
    )
