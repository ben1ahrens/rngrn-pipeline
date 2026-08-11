# Talking points — weight-noise robustness deck

**Open with this, before any figure goes up:** *"These are mock-ups. I have not run the
experiment. I built them to show what a positive result would look like, and to get your view
on whether the controls are the right ones before I spend compute."*

Say it first and the whole meeting is about experimental design, which is what you want. Say
it after the plots and it sounds like a retraction. The figures carry an `ILLUSTRATIVE` tag
in the corner for exactly this reason — point at it once.

One-sentence version of the science: **weight noise makes the optimiser descend a blurred
version of the loss, which favours minima that are wide — and in this problem a wide minimum
*is* a large local Turing volume, which is the quantity Tica report.** That equivalence is
the reason the idea is worth testing rather than just a regulariser someone bolted on.

---

## Figure 1 — the headroom is real, even though the effect is a mock-up

- Panels (a) and (b) are **real measured data**, not mock-ups: the 127 `three_gene` answer
  keys from exp11. Worth flagging, because it means the premise doesn't depend on the
  surrogate.
- **The number to land:** each topology in that set has exactly **one** interaction matrix,
  yet robustness at 10 % noise spans up to **4.3×** within a single topology. So the spread is
  entirely parameter-driven — for the same wiring, some parameterisations are four times more
  robust than others.
- That is the whole argument for the "engine" framing: there is something real for an
  optimiser to find. Without this panel, "make it more robust" has no headroom established.
- Panel (d) is the mechanism in parameter space: from the deterministic solution, ~19 % of
  random directions have left the Turing regime by ‖Δθ‖ = 0.6; from the noise-trained one,
  none have.

## Figure 2 — the headline

- Median local Turing volume at 20 % perturbation: **0.798 → 0.974**, improved on **33 of 33**
  systems, Wilcoxon **p = 2.3 × 10⁻¹⁰**.
- Turing hit rate also rises, **0.772 → 0.850** — more seeds succeed at all, which is a
  separate axis from robustness and matters given the standing 2-of-6 generalisation problem.
- At Tica's measured experimental CV (4.8 %) **everything is already at 1.000**, so that noise
  level cannot discriminate on this data. Useful to say out loud: if we want a headline number
  at the experimentally meaningful noise level, the metric needs to be something finer than
  "fraction still Turing" — this is a real design question, not a detail.

## Figure 3 — the cost, and why I'd lead with it

- **Lead with this panel if he is sceptical.** It is the reason to believe the rest.
- Robustness is bought with wavelength accuracy: on the matched subset, k\* error goes
  **3.17 % → 4.06 %** at 20 % noise, and **6.95 %** at 35 %. There is an interior optimum, not
  a free lunch.
- "Matched subset" means only seeds that reached Turing in **both** arms. That guard matters:
  the noise arms convert previously-failing seeds into Turing ones, so the naive pooled
  comparison mixes a population shift into what looks like an accuracy regression. Mentioning
  this unprompted is the cheapest credibility you will buy all meeting.
- Gain holds in all three morphologies (labyrinth 0.705 → 0.961, spots 0.808 → 0.975, stripes
  0.866 → 0.970) — but stripes is n = 6 systems, so don't lean on it.

## Figure 4 — the controls (the panel he will actually probe)

- Two ways the result could be trivially explained, and both fail:
  **post-hoc jitter** (train deterministically, then perturb once) reaches only 0.825 and
  wrecks k\* to 12.7 %; **wide initialisation, no noise** reaches 0.759 with k\* at 45 %.
- So it is not "any stochasticity helps" and not "more exploration helps". It is specifically
  training *against* the perturbation.
- Panel (c) is the anti-cheat guard: growth rate at k\* and the D-ratio do not collapse. This
  matters because we already know the k\*-anchor has a degenerate minimum where the loss falls
  by *flattening* σ(k) rather than moving its peak. A robustness gain achieved by going limp
  would be a fake result, and this panel is how we would catch it.

## Figure 5 — honest about a regression

- The whole seed distribution shifts, not just the median — worth stating, since house
  convention is to report a seed distribution rather than a best seed.
- **But seed-to-seed k\* spread widens: 3.95 % → 7.81 %.** Noise makes individual runs less
  reproducible even as the population gets more robust. Volunteer this.
- The compensating point: k\* dispersion *inside* the surviving perturbation cloud is
  essentially unchanged (10.5 % at 20 % eval noise, both arms). So perturbation degrades
  *whether* a system patterns much more than *which* wavelength it picks — which is the right
  side of the trade given that the dominant spatial mode is the stated success criterion.

## Figure 6 — the bar that actually has to be cleared

- Fragile systems gain most — but flag that this is **partly structural**, since gain is
  bounded above by (1 − baseline). The ceiling is drawn on the panel. Don't sell ρ = −0.95 as
  a discovery.
- **The reference that matters:** the generator systems' own robustness is 0.755 at 20 %. The
  claim to aim for is not "better than our deterministic arm" but "better than the training
  data" — that is the Tica-equivalent framing.

---

## Questions to expect, and the honest answer

**"Is this just standard weight-noise regularisation?"** Mechanically yes, but the payoff is
unusual: here the flatness that noise selects for is the same object as the robustness metric
the field reports. It is not a proxy for generalisation — it is the target quantity.

**"Why should I believe the surrogate transfers?"** You shouldn't, fully. It optimises
log(J, D) directly, whereas the real model reaches J through gated-Hill kinetics and a Newton
steady-state solve. The surrogate shows the *selection effect* exists when the objective has a
wide-minimum structure; whether the real parameterisation preserves that structure is the
experiment. Also worth conceding: perturbing in log space is *cleaner* than the real thing,
where a single σ on raw θ produces a ~14× spread in effective physical noise — so the
surrogate flatters the mechanism.

**"Is σ_train = 0.20 the recommended setting?"** No — that optimum is a property of this
surrogate's geometry. The dose–response *shape* is the transferable claim, not the location of
the knee.

**"Would this be comparable to Tica's number?"** Not as it stands. They perturb kinetic
parameters and re-derive the steady state and Jacobian; this perturbs the linearisation
directly, with no steady-state solve that can fail. It is a clean upper reference. Making it
comparable is Unit 2 in the pipeline spec.

**"How long to get the real version?"** The measurement code is the main gap — the existing
`robustness_cloud` has four documented defects and its output never reaches the run index.
The arms and pass conditions are specified; what is missing is a correct metric and a noise
injection that acts on physical parameters rather than raw θ.

---

## Two decisions to get out of him, ideally today

1. **Is noise resampled per step, or held per restart?** Per-step gives the smoothed-objective
   behaviour this deck models. Per-restart gives a randomised-prior ensemble — a different
   experiment with a different claim. This should be settled before code, not after.
2. **What counts as "at equal quality"?** The robustness claim is only interesting if pattern
   quality is preserved, so we need a tolerance on k\* degradation and morphology agreement
   fixed *in advance*. Given the house rule that thresholds are pre-registered and never
   revised after seeing results, this is the one number worth arguing about now.

## What not to overclaim

- No RNGRN was trained. Nothing here is evidence about the pipeline.
- The 4.8 % level is saturated on this data, so the headline gain is quoted at 20 % — a noise
  level well above anything experimentally measured. Say so.
- Turing-I vs Turing–Hopf was instrumented but barely varies (3 of 1,684 points). It stays in
  because Tica report it, not because it discriminates here.
- A smoke run at a reduced budget (30 steps, 3 seeds) gives a median gain of only +0.02 at
  p = 0.07. The effect needs an adequate optimisation budget to appear — which is itself worth
  knowing before designing the real sweep.
