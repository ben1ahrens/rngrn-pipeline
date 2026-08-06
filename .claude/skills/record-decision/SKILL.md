---
name: record-decision
description: Use whenever making or changing a science decision in rngrn-pipeline — defining or altering a metric, picking a threshold or pass condition, choosing an estimator, deciding what counts as a control, or marking something UNCALIBRATED. Gives the required DECISIONS.md entry format and the rules for what may be decided locally versus escalated.
---

# Recording a science decision

This project is headed for a paper. The owner has explicitly delegated science decisions:
*"You do not need to ask me for all science. If unsure, do some research to verify."* So
**decide it yourself, and record it.** Run freely on mechanical work *and* on science
decisions — defining or changing a metric, choosing a threshold or pass condition, picking an
estimator, deciding what counts as a control.

What replaces asking is **evidence plus a written record**.

## Every science decision must

1. **Be calibrated against something measured**, not chosen for convenience. A threshold needs
   a baseline or a control behind it. If you cannot calibrate it, mark it **UNCALIBRATED** in
   the code *and* in `docs/DECISIONS.md`. An honest placeholder is fine; a silent guess dressed
   up as a choice is not.
2. **Be researched when you are unsure.** Read the primary source, not a secondary one. This
   has already caught real errors: four claims in `docs/STATE_OF_THE_SCIENCE.md` §2.7 were
   attributed to Matas-Gil & Endres when two came from Kho et al. and two were least-squares
   rather than PINN results — traced by reading the paper *and* the authors' shipped code.
   When docs and source disagree the source wins; when a citation and the paper disagree the
   paper wins.
3. **Land in `docs/DECISIONS.md`** with the date, the evidence, and what was rejected.
4. **Be announced loudly if it changes what an existing number means.** A silently
   non-comparable number is worse than a missing one.

## Two things go to the owner instead

- Anything that would **weaken a pre-registered pass condition** in `docs/PREREGISTRATION.md`.
  That file is binding and is dated ahead of the runs it judges — which is the entire point.
- Anything outside the technical scope: publishing, sharing, spending money, or touching
  systems beyond this repo.

## The entry format

Append to `docs/DECISIONS.md` in the existing house format. Match the surrounding entries:

```markdown
### D<N> — <the decision as a single declarative sentence>

**Date:** YYYY-MM-DD (<what produced it — unit, PR, or session>). **Status:** DECIDED
**Decided by:** <the implementing agent under delegated authority | the owner>

**The decision:** <what is now true in the code, precisely, naming the symbols.>

**Evidence:** <the measurement behind it, with the doc/run it lives in. Numbers, not
adjectives. Cite `docs/<FILE>.md` §N or the run directory under `experiments/`.>

**What was rejected and why:** <the alternative you did not take, and the reason. This
field is not optional — a decision with no rejected alternative was not a decision.>

**Not independently validated:** <optional but expected when the choice is flagged
`[IMPL]` — say exactly what was tested and what was not.>

**Where it lives:** `src/rngrn/<module>.py::<symbol>`; `<config or doc>`.
```

**Status values:**
- `DECIDED` — a choice is made and live in the code/config as described.
- `OPEN` — laid out, not yet chosen; picking it would bias what later numbers mean.
- `SUPERSEDED` — a later decision reversed this one. **Keep the entry.** Numbers measured
  under the old decision are not comparable to numbers measured after it, and that is exactly
  what a reader needs to know.

Evidence-integrity *defects* (as opposed to decisions) go in Part 1 of the file, which is
ordered first deliberately because they change how earlier numbers should be read. A defect
entry adds a **Consequence for earlier results** section saying which prior runs are now
suspect and how to check.

## Marking something UNCALIBRATED

In the code, at the site, naming what would calibrate it:

```python
# UNCALIBRATED: no control arm has been run against this yet. Calibrate from <the control>
# before any reported number depends on it. See docs/DECISIONS.md D<N>.
COUPLING_THRESHOLD = 0.05
```

Then the matching `DECISIONS.md` entry gets `**Status:** OPEN` and states plainly what
measurement would close it. Two known live examples: the `~15%` k\* tolerance in `TUNING.md`
(inherited from the initial template commit, referring to reference frames the current solver
cannot even generate) and `coupling_threshold=0.05` for spare-species inertness.

## Before you write the entry

State what is *not* known as plainly as what is. A caveat that only appears after someone
asks is a caveat that failed.
