# Reporting a number

`CLAUDE.md` §8 is the rule. This is the gate to pass before any number leaves a run directory
and enters a doc, a summary, or a message to the owner.

## Before you write the number down

1. **Where did it come from?** Name the run directory under `experiments/`. Run records,
   frozen configs, results, checkpoints and `arrays/*.npz` are tracked in git precisely so a
   claim can be traced to the run behind it. A number with no run is not reportable.
2. **Was it a tuned run on real data, or a plumbing check?** Short CPU runs verify that a
   config resolves, data loads, and scoring routes. They recover nothing. If it was one of
   those, the only sayable sentence is *"the harness runs"*.
3. **What is its control?** Read an arm against its matched control, never against zero. Every
   experiment here ships one for exactly this reason. A number without its control is
   incomplete, not merely unadorned.
4. **Did the config that ran match the config you think ran?** Check `frozen_config.yaml` and
   `results/train_results.json`, not the command line. Repeated `-o` flags once silently kept
   only the last override (D-EVID-1), so any pre-fix run's effective config must be read from
   the frozen file.
5. **Is the threshold it is judged against calibrated?** If not, say so in the same sentence
   as the number.

## Wording

- Say **"Turing-unstable"** or **"patterns"** — never one as a proxy for the other. They are
  different claims; closing the gap between them is the entire reason `eval/rollout.py` exists
  and why `morphology_match` is scored separately from every dispersion-derived criterion.
- A metric that is NaN **by construction** for an experiment arm is reported as such, with the
  reason. A NaN that arrived any other way is a defect, not a value.
- State what is **not** known as plainly as what is. A caveat that only appears after someone
  asks is a caveat that failed.
- If the number is not comparable to a previously reported one, say so in the same breath.
  A silently non-comparable number is worse than a missing one.

## Verify, don't assert

Turing instability is confirmed numerically. Morphology is classified by measurement. A claim
about the code is checked against the code. If a doc and the source disagree, **the source
wins and the doc gets fixed** — in the same change, not later.
