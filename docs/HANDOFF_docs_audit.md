# HANDOFF — documentation audit and consolidation plan

Branch: `docs/fft-docs-audit`, cut from `feature/fft-training` @ `8b96541`, 2026-08-14.
Audience: a zero-context reader. Nothing in this document has been executed — it is the
audit result plus the proposed plan. **Nothing merges to `feature/fft-training` without the
owner's validation** (CLAUDE.md §2, `.claude/rules/pre-merge-checklist.md`).

Method: six parallel read-only audit units (CLAUDE.md §11 wave floor cleared; no unit ran a
trainer), plus orchestrator verification of every finding reproduced below. Suite measured
unsandboxed on this worktree: **630 passed, 1 skipped, 327 s** (venv verified inside this
worktree). The 627/4 figure some units saw is the same 631 collected with three CUDA-gated
tests in `test_forward_solve.py` skipped on a CPU-only path.

---

## Execution status — updated 2026-08-14

Phases 1 and 2 have been **executed on this branch** (owner-approved scope). Phase 0's
tracking step is done; Phase 0's *commit* step and all of Phase 3 are not.

**Done**
- `.claude/rules/orchestration.md` is now **tracked** — the dangling `@`-import is fixed.
- `three_gene_qvar_smoke` manifest checksum corrected to `b881e667b8b9275b`. **All 14
  payloads now verify clean against their manifests.** Repaired surgically rather than via
  `rngrn scan-datasets --refresh`, which has no per-dataset scoping and would have rewritten
  all 14 manifests and appended 14 index rows — more provenance churn than the defect
  warranted. The one wrong field was the checksum; every other manifest field was verified
  against the payload first.
- §3.1 firewall-guard overclaim corrected in `.claude/rules/pre-merge-checklist.md`, plus a
  new checklist line requiring any new **package-root** module to be added to the test's
  `discovered` set at birth (nothing else catches it). `pytest tests/test_firewall.py` green.
- §3.3–§3.7 corrections applied across `README.md`, `TUNING.md`, `CODE_REALITY.md`,
  `CANONICAL_DATASETS.md`, `DATA_INTO_MODEL.md`, `data/datasets/README.md`,
  `GOAL_tica_equivalent.md`, `TIMESCALE_MU.md`, `FUTURE_WORK.md`,
  `ROBUSTNESS_MEASUREMENT.md`, `LGEN_TRANSFER_FIRST_RESULT.md`,
  `HANDOFF_identifiability.md`, `IDENTIFIABILITY_EXPERIMENTS.md`,
  `.claude/skills/{new-worktree,run-training}/SKILL.md`.
- The 57→41 defect and the split-integrity defect are now **recorded** in
  `CANONICAL_DATASETS.md` and `data/datasets/README.md` as dated defect notes. The rule
  itself was not altered; the remedy remains an owner decision (§1.1).
- §2.1 archives: `HANDOFF_canonical_datasets.md` and `HANDOFF_training_data_plots.md` moved
  to `docs/archive/` with dated supersession headers, bodies intact. Their in-body errors
  were fixed first so the archived copies are not wrong.
- Orchestrator-applied, in the three highest-stakes files: the 12.5 %→**11.96 %** correction
  (SPEC §9.1, PLAN §2), the labyrinth-topology correction (the set spans **three**
  topologies, not `double_inhibitor` alone), the adjoint-residual correction
  (**2.3–6.1e-12**, was "4–7e-13"), and a dated evidence note recording that the
  `/tmp/d1_probe2` 64² artefacts are **gone** (confirmed absent) so every figure sourced to
  them is indicative, not reportable.

**Two briefs were corrected by the units that executed them** — recorded because it matters
for trusting the rest:
- `robustness_cloud` and `robustness_volumes` are **distinct functions**; the audit wording
  conflated them. `robustness_cloud` does run on a recovery but stays in stdout;
  `robustness_volumes` is what reaches the run index as `turing_volume_*`.
- TUNING.md's "the batched path refuses loudly" is **still true** —
  `recover.py:430-435` still raises on `batched=True` with a nonzero spectral weight. Only
  the serial/GPU framing was superseded.
- `TUNING.md:102` had moved to `:125`, not the `:115` the audit predicted.

**Not done, and why**
- `CLAUDE.md`, `.claude/README.md`, the four `.claude/agents/*.md`, and **`docs/DECISIONS.md`**
  were untouched: all carry uncommitted changes in the `feature/fft-training` worktree, and
  editing them here would cause exactly the line-granularity merge damage CLAUDE.md §11
  documents. This defers §3.1's `CLAUDE.md` §5 firewall lists, §3.2's dangerous payload loop,
  and **all of §3.8's ledger repairs** — including opening `D-FFT-15`. Three specific items
  are waiting on that file, verified as the *only* remaining stale references repo-wide:
  `DECISIONS.md:2104` and `:2613` still point at the pre-archive paths
  `docs/HANDOFF_training_data_plots.md` and `docs/HANDOFF_canonical_datasets.md` (now under
  `docs/archive/`), and `:2366` carries the same 57-vs-41 defect corrected elsewhere.
  `DECISIONS.md:2606-2607` also still carries the 12.5 % figure corrected to 11.96 % in
  SPEC §9.1. A repo-wide link check confirms **zero broken markdown links** and no other
  dangling path.
- `docs/STATE_OF_THE_SCIENCE.md` was untouched by design: it is cited **by line number** from
  11+ source files and tests, so editing it breaks source comments. Its corrections need a
  paired source-comment update and are deferred.
- Phase 3 in full — it is blocked on §1.
- `src/rngrn/eval/lifted.py:4` carries the same wrong `STATE_OF_THE_SCIENCE.md §713-714`
  pointer that was fixed in `FUTURE_WORK.md:88` (correct span is §792-794). Source left
  untouched; flagged here.
- `TIMESCALE_MU.md`'s six placeholders: **no rendering script exists**. A repo-wide grep for
  the placeholder strings hits only that document. `stage0b_mu.py` writes arrays and
  `stage0b_figures.py` writes PNGs; neither emits the tables. The renderer has to be written
  before the document can be completed — this is more than a re-run. A dated UNRENDERED
  banner was added in the meantime.

---

## 0. The headline, stated plainly

**The problem is not that there are too many documents. Only two of twenty-four are
genuinely removable.** The rest are either irreplaceable measurement records or actively
load-bearing. What the audit actually found is three different problems wearing the costume
of "too many docs":

1. **Two unreconciled definitions of success** live in the repo simultaneously, and a third
   set of gates in `PREREGISTRATION.md` conflicts with both.
2. **A split-integrity defect in the data itself** — verified against payload attrs, not
   inferred from docs.
3. **Systematic drift** — six different test counts, stale line references en masse, and two
   firewall guards that are documented as stronger than they are.

Deleting documents would not touch any of these. Two of the three are owner decisions.

A prior audit (`4527c6a`, 2026-08-11) reached the verdict *"no doc was stale enough to
remove"*. That pass ran **before** the M1 machinery landed on 08-12, so its conclusions about
code-facing claims were correct then and are not now.

---

## 1. BLOCKING — owner decisions, in priority order

None of these may be settled by an agent. Items 1–3 touch `docs/PREREGISTRATION.md` or the
project goal; item 5 is a publishing decision (CLAUDE.md §10 reserved class).

### 1.1 Split-integrity defect: burned systems occupy held-out slots — VERIFIED

`PREREGISTRATION.md:118-121` binds:

> "A system that appears in `PREVIOUSLY_RUN` may **never** occupy a held-out slot… it is
> declared burned here **so the held-out three stay clean**."

Verified directly from payload attrs (SHA-256 over canonicalised `params_json`):

| kinetics hash | sample A | sample B |
|---|---|---|
| `c6883c26e273` | `turing_labyrinth/sample_0001` — **tuning** (qvar:18, declared burned) | `turing_labyrinth/sample_0003` — **held_out** (multiL:36) |
| `f4bbab1a81cc` | `turing_spots/sample_0000` — **tuning** (qvar:2) | `turing_spots/sample_0002` — **held_out** (multiL:0) |
| `4bcb7502631e` | `turing_spots/sample_0003` — held_out | `turing_spots/sample_0004` — held_out |

**Root cause**, verified: `scripts/canon_select.py:210-211` defines
`row_uid = f"{source_dataset}:{system_id}"` and documents it as "stable identity for a SYSTEM
(kinetics), independent of which replicate we read". That is false across datasets —
`system_id` is numbered per source dataset. `ELIGIBLE_DATASETS` pools `three_gene_qvar` (34
distinct kinetics) and `three_gene_multiL` (23), which **overlap by 16**. The true pool is
**41 distinct systems, not the 57 claimed** in `CANONICAL_DATASETS.md:72`,
`HANDOFF_canonical_datasets.md:10` and `data/datasets/README.md:70`.
`tests/test_canon_select.py:252` asserts dedup on `uid`, so it structurally cannot catch this.

**Mitigation — the FFT track's own evidence pair is CLEAN.** Stage-0 `sample_0000`
(`61355ad8abdf`) and Stage-3 `sample_0004` (`4562199300fa`) are both unique and in no pair.
The Stage-3 labyrinth claim is unaffected. Effective clean held-out: labyrinth 2 of 3;
spots 2 of 3 samples but only **1 distinct system**.

**Needs a ruling on:** whether to record-only, re-derive the selection with a kinetics-keyed
uid, or amend the preregistration. Recording it costs nothing and is reversible; re-deriving
changes the canonical sets.

### 1.2 Two live, conflicting binding gate sets

`PREREGISTRATION.md` §3 binds `kstar_fft_rel_err ≤ 8.3 %` and 8 seeds at
`topology_consistency ≥ 0.75` **plus** a cross-target control gap ≥ 0.25 that the document
itself calls *"the substantive half… the criterion most likely to expose a false success"*.
`SPEC_fourier_training.md` §1/§9.1/§9.3 binds one radial bin (measured **11.96 %** on this
sample) and 5/5 seeds under identity mapping, with **no cross-target control**.
`PREREGISTRATION.md` carries **no amendment** for either (verified by grep).

Adopting the one-bin rule on `sample_0000` *loosens* a pre-registered pass condition —
owner-only. The 5/5-vs-8-seed direction is genuinely ambiguous (stricter agreement fraction,
fewer seeds, no control).

**Recommended, as the only option that changes no threshold:** report **both** bars on every
canonical k\* number and gate on the stricter, until the owner rules.

### 1.3 Two unreconciled statements of the project's objective

`GOAL_tica_equivalent.md` §3, quoting the owner:

> "the only thing I really care about is that the model recovers the same dominant spatial
> mode, pattern morphology." … "This overrides older docs in this repo that emphasise J sign
> structure or parameter recovery as goals."

`SPEC_fourier_training.md` §1 makes R1 **sign-structure reproducibility** the binding gate and
demotes morphology to "reported only, for now". Neither document references the other; grep
for "GOAL_tica" or "dominant spatial mode" in SPEC/PLAN returns nothing.

### 1.4 The forward-map convergence contract (already known, still open)

F-D1-5 options (a)–(d). Two additions the audit contributes:

- **Stage 0 would silently train on nothing.** `forward.py:574` `CONVERGENCE_TOL = 1e-9`; on
  failure `solve()` returns `None` and `losses/total.py:47-52` omits all five spectral terms.
  F-D1-5's measured residuals at the data box are 1.5e-5–4.7e-5 at every grid. So every
  ignited step logs `spec_ignited = 1.0`, burns 623–752 s, discards the result, and trains on
  the baseline objective — while `PLAN:68-71`'s machinery bar ("training ignites and completes
  without SteadyStateError storms") **passes**. No gate on the milestone chain tests
  forward-solve convergence at the training geometry.
- **Option (a) collides with the SPEC's shared-`L` assumption.** A model-adaptive
  `L_solve(θ)` makes `SPEC:186-190`'s "matching `observables.raps` bin-for-bin" undefined for
  the model side, and breaks D-FFT-3's training-side k\* check. Forward-validation F3 is
  unaffected.

### 1.5 Externally published stale claim

`METHODS_fourier_training.md` §9 states *"The pipeline is therefore realised GPU-native at
512²"*, contradicted by F-D1-5 one day later. That file is **published as a claude.ai
artifact**, so the stale claim is externally reachable. Both METHODS drafts are also outside
version control — the only artifacts in this project with zero history, in a repo otherwise
engineered around traceability.

---

## 2. Disposition ledger — stale vs current

Twenty-four `docs/` files plus root and `.claude/`. Verdicts are evidence-backed; see §1 and
§3 for the specifics behind each.

### 2.1 ARCHIVE — completed handoffs, content fully superseded (the only removals)

| Doc | Evidence it is done |
|---|---|
| `HANDOFF_canonical_datasets.md` | Both §7 follow-ups struck through and marked DONE with commits `3ecdcca`, `4060461`; configs verified independently; all §5 artifacts exist; D-CANON-1..5 in the register. Superseded by `CANONICAL_DATASETS.md`. |
| `HANDOFF_training_data_plots.md` | Every deliverable exists — notebook, `scripts/td_figures.py`, tests, 62 figures, `D-TDPLOT-1`, firewall entries. Nothing live remains. |

Archive to `docs/archive/` rather than delete: both are cited from `docs/DECISIONS.md` and
`data/datasets/README.md`.

### 2.2 KEEP AS-IS — verified current, do not touch

`DATASETS_L.md` (cleanest file audited; every claim verified to the digit),
`LOCAL_DATA_SETUP.md` (correctly self-scoping; the dangerous symlink loop is **not** in this
file), `BIO_VIABILITY.md`, `.claude/rules/reporting-numbers.md`, the three
`.claude/skills/{harvest-dataset,record-decision,run-training}/SKILL.md`,
`.claude/hooks/guard_trainer.py`, `.claude/agents/repo-mapper.md`.

### 2.3 KEEP — permanent records, not stale despite predating the pivot

These are the only surviving record of a measurement. Removing any loses evidence.

| Doc | Why it is irreplaceable |
|---|---|
| `STATE_OF_THE_SCIENCE.md` | The evidence ledger; unusually well self-annotated with inline SUPERSEDED/CORRECTED boxes. Its §2.7 misattribution fix is confirmed landed (`c44f7a9`). **Cited by line number from 11+ source files and tests** — reflowing it breaks those citations. |
| `C1_COMPETITIVE_TUNING.md` | Holds the σ(k)-gauge-blindness proof (§8). **SPEC §2 depends on it** — cites its 7/8-seed result as the pivot's own rationale. |
| `C2_NC1_TUNING.md` | Holds the "dead J" upper bound (§1.2), 9/16 seeds admitting no Turing instability for any D. Missed by the 08-11 sweep. |
| `LGEN_TRANSFER_FIRST_RESULT.md` | The only cross-L transfer measurement; system 13 is burned, so it cannot be regenerated. |
| `IDENTIFIABILITY_EXPERIMENTS.md` + `HANDOFF_identifiability.md` | A deliberate science/engineering split, not duplication. Harness real, never run beyond a 6-step dry run. |
| `SPEC` / `PLAN` / `DIAGNOSTICS_fft` / `DECISIONS` / `PREREGISTRATION` | The live FFT track. The four-way role separation is doing real work; no consolidation warranted. Ledger integrity verified clean — D-FFT 1–14 each appear once, all 93 cross-references resolve post-renumber. |

### 2.4 UPDATE — current in role, stale in specifics

`CLAUDE.md`, `README.md`, `TUNING.md`, `CODE_REALITY.md`, `DATA_INTO_MODEL.md`,
`CANONICAL_DATASETS.md`, `data/datasets/README.md`, `.claude/README.md`,
`.claude/rules/pre-merge-checklist.md`, `.claude/agents/{firewall-auditor,numerics-reviewer}.md`,
`.claude/skills/new-worktree/SKILL.md`, `GOAL_tica_equivalent.md`, `TIMESCALE_MU.md`,
`ROBUSTNESS_MEASUREMENT.md`, `FUTURE_WORK.md`, and the five FFT-track docs.

### 2.5 PROMOTE

| Item | Action | Urgency |
|---|---|---|
| `.claude/rules/orchestration.md` | Untracked, yet live `CLAUDE.md:583` `@`-imports it — **dangling in every other worktree** the moment the CLAUDE.md delta commits. Track it. | Highest |
| `CLAUDE.local.md` durable gotchas | The conda interpreter path, in-sandbox commit failure, and WSL2 SIGPIPE push remedy exist in no tracked doc. Promote to `CLAUDE.md` §1 / `new-worktree`. Its ephemeral session log dies with the branch. | High |
| `METHODS_*.md` | Bring under version control as `docs/METHODS_*.md` — owner call (§1.5). | Owner |

---

## 3. Safe corrections — no judgement required

Every item below either tightens a claim, or replaces a number with a measured one, or points
a reference at what it already meant. **None weakens a threshold, tolerance, test or gate.**

### 3.1 Two guards documented as stronger than they are (do these first)

- `CLAUDE.md:191` — *"A new module under `src/rngrn/` is therefore no longer unaudited by
  default"* is **false**. `tests/test_firewall.py:162-166` globs only `losses/` and `eval/`,
  plus three hard-coded root modules; the test's own comment at `:158-160` says so.
  `.claude/rules/pre-merge-checklist.md:40-41` repeats the overclaim. Scope both to what the
  test does, and add a checklist line for new package-root modules.
- `CLAUDE.md` §5 firewall lists are one commit behind: `FORBIDDEN` is **27** names, not 12;
  `RECOVERY_SIDE` is **16** modules, missing `forward.py`, `etdrk4_torch.py`,
  `losses/spectral.py`, `eval/lifted.py`. Replace inlined lists with a pointer to the file.
  Same defect in `DATA_INTO_MODEL.md:341` and `STATE_OF_THE_SCIENCE.md:21-22`.

### 3.2 The dangerous provisioning loop

`CLAUDE.md` §2 still hands out the inline `ln -sfn` loop that `new-worktree/SKILL.md:66` and
`harvest-dataset/SKILL.md` label **"DO NOT USE"**, and that `CLAUDE.md` §6a itself documents
as having caused a real clobber. `scripts/link_payloads.sh` exists precisely to replace it and
is never mentioned. Reduce §2 to a pointer at the script. (Its "~91 MB" is also stale; the
canonical store is 206 MB.)

### 3.3 Test counts

Six distinct numbers in three days: 557 (`CLAUDE.md`, `README.md` ×2, `CODE_REALITY.md`), 618
(`CLAUDE.local.md`), 627/4 and 630/1 (measured today). The live `CLAUDE.md` §3 already forbids
recording a count; that rule has not propagated. **Delete the counts** rather than refresh
them — `README.md:12,81`, `CODE_REALITY.md:182`, `HANDOFF_identifiability.md:15,174`.

### 3.4 Numbers that are simply wrong

| Claim | Actual | Where |
|---|---|---|
| one bin = "12.5 % of k\*" | **11.96 %** (recomputed from `d3/results.json`) | `SPEC:383`, `PLAN:54`, `DECISIONS:2606`, `CLAUDE.local.md:147` — fixing **tightens** |
| `PPW_MIN/MAX = 16/32` | `canon_generate.py:47` → **12.0/64.0** | `CANONICAL_DATASETS.md:156` |
| `Tmax = 260`, capped 8× | `TMAX_BASE = 500.0`, `TMAX_MAX = 2000.0` (4×) | `CANONICAL_DATASETS.md:191-195` |
| "57 distinct systems" | **41** | `CANONICAL_DATASETS.md:72`, `data/datasets/README.md:70`, `HANDOFF_canonical_datasets.md:10` |
| "five distinct systems each" | labyrinth **4**, spots **3** | `CANONICAL_DATASETS.md:51` |
| "the 287 registered samples" | **426 / 1203 channel-instances**; 287 = the 9 pre-08-29 sets | `DATA_INTO_MODEL.md:17,200` — add a dated scope banner |
| adjoint residuals "4–7e-13" | 2.34e-12–6.08e-12 | `DIAGNOSTICS_fft.md:52` |
| "162× CUDA speedup" | a *backend* comparison at fixed device; crossover ~6,400 matrices, CPU wins at n=127 | `run-training/SKILL.md:91`, `PLAN:224-226` |

### 3.5 Claims contradicted by source

- `CODE_REALITY.md:226` *"There are no worktrees"* — six exist. `:223-232` branch counts stale
  (48 branches, 41 merged, 7 unmerged).
- `CODE_REALITY.md:251` and `STATE_OF_THE_SCIENCE.md:9` *"the whole `experiments/` tree is
  gitignored"* — `.gitignore:13-17` says **"RUN RECORDS ARE TRACKED."** This one matters: it is
  why the audit trail is citable at all.
- `ROBUSTNESS_MEASUREMENT.md:125` *"`turing_ok` tests `tr(J) < 0`"* — `eval/analysis.py:72`
  evaluates strictly since D-EVID-11. Same stale claim in `BIO_VIABILITY.md:158,161`,
  `GOAL_tica_equivalent.md:383-384`, `TIMESCALE_MU.md:99-102`.
- `GOAL_tica_equivalent.md:369,396-399` *"the D-ratio prior is unwired / plausibility
  uninstrumented"* — `total.py:134-137` wires it; `validate.py:397` emits
  `plausibility_score`. The same doc contradicts itself at `:184-191`.
- `FUTURE_WORK.md:37-38` *"`eval/lifted.py` NOT tested, no results"* — 704 lines, 16 tests
  green.
- `.claude/README.md` "four agents" vs `CLAUDE.md` §12 "five" — a committed defect predating
  this branch. The uncommitted delta fixes it.
- `numerics-reviewer.md:20-22` *"a finite-difference stencil appearing anywhere is a
  regression"* — `observables.py:131 laplacian_torch` has existed inside the objective
  (`terms.py:337`) since before M1. `CLAUDE.md` §7c's heading is wrong for the same reason,
  independently of its (correct, 2026-08-12) forward-solve correction block.

### 3.6 Stale line references — largest and cheapest class

`losses/terms.py` grew ~240 → 820 lines, `total.py` 27 → 284, `recover.py` → 692,
`model.py` → 558. Roughly 60 `file:line` references are stale across `DATA_INTO_MODEL.md`,
`HANDOFF_identifiability.md`, `TUNING.md`, `BIO_VIABILITY.md`, `CLAUDE.local.md`.
**Caution:** `CODE_REALITY.md`'s three refs (`terms.py:186,:237,:415`) are still exact — do
not "fix" them. And `FUTURE_WORK.md:88-89`'s wrong pointer is duplicated verbatim in
`src/rngrn/eval/lifted.py:4`, so the house "source wins" rule does not resolve it.

### 3.7 Evidence hygiene

- `LGEN_TRANSFER_FIRST_RESULT.md:102-103` cites `runs/…` — the runs exist at
  `experiments/lgen_transfer/runs/…`. Missing path prefix, not a lost run.
- F-D1-2 / F-D1-3's 64² probe numbers — including the projected-GMRES-bias result behind
  D-FFT-10 — trace only to `/tmp/d1_probe2`, which no longer exists. The *decision* stands on
  the committed 96² record, but per `.claude/rules/reporting-numbers.md` those figures are not
  currently reportable. Either mark them "probe, artefacts not retained" or re-run into a
  tracked `experiments/diag_fft/d1_probe64/`.
- `TIMESCALE_MU.md` asserts *"The measurement CONFIRMS it"* while six template placeholders
  (`<!--RESULTS_TABLE-->` …) sit unrendered in the committed file. Backing arrays exist; this
  is a rendering fix, not a re-measurement. `FUTURE_WORK.md` separately marks those arrays
  "partial, do not cite".
- Wall-clock figures in `CLAUDE.md:257-259`, `CODE_REALITY.md:175`,
  `STATE_OF_THE_SCIENCE.md:420` ("CPU 138 ms vs CUDA 418 ms") have **no backing run** anywhere
  under `experiments/`. Not reportable without a re-measure.

### 3.8 Ledger repairs (`docs/DECISIONS.md`)

- Open **D-FFT-15** for the forward-map convergence contract — currently the branch's one
  blocking decision, and it has no entry, violating `PLAN:10-11`. Record as OPEN(owner) with
  options (a)–(d) verbatim; choose nothing.
- D-FFT-11 flag 1 (`:3010-3013`) — annotate SUPERSEDED; the measurement it deferred to
  returned negative.
- D-FFT-13 (`:3103-3121`) — sat512 landed negative; the 512²-relax framing is superseded;
  `warm_max_chunks=40` is unmeasured at the data box.
- D-FFT-12 (`:3029-3031`) — "the three owner flags remain the M2 blockers" is stale; they
  closed 2026-08-13.
- D-FFT-14 (`:3173-3176`) — a `Where it lives` block belonging to D-FFT-11 is misfiled here.
- D-FFT-9 closure 1 (`:2808-2810`) — band-power fractions were measured at the *provisional*
  0.5/1.5 edges, not the adopted 0.60/1.55.
- **Citation-integrity defect:** `CLAUDE.md` §11 and D-PERF-2 both quote `CLAUDE.local.md` for
  "judgement agents keep the default" and "mechanical Explore" — **neither phrase exists** in
  that file. Reword both to state the rule directly rather than restore an untraceable quote.

### 3.9 Data repair

`three_gene_qvar_smoke` checksum mismatch (manifest `693715c7d7a82b19`, actual
`b881e667b8b9275b`) is a **stale manifest, not corruption** — the payload is 3 h 37 m newer
than the manifest, which shares a bulk-scan mtime with all 14 others. The file opens and loads
cleanly and reproduces `DATASETS_L.md:102`'s numbers exactly. Repair is
`rngrn scan-datasets --refresh` **in the canonical store**, not regeneration. All 13 other
payloads verify clean, including both 512² Turing sets.

---

## 4. Two decision points I am holding, not settling

Both earn a `docs/DECISIONS.md` entry. Both are leakage-class, so I am surfacing rather than
ruling.

**D1 — the Stage-0 channel holdout may not be one, and it is LIVE.**
`SPEC/PLAN §2:63-66` asserts `frame.mean()` is channel-0-only "so the anchor cannot see
held-out channels". `losses/total.py:125` computes it over the **full `(m,H,W)` tensor**
(`recover.py:373`). At Stage-0 geometry `observed_idx=(0,1,2)`, the frame-scale anchor
averages the two channels the plan calls held out. The codebase demonstrably can slice —
`recover.py:477` does exactly that for `kstar_obs`. Anchor weight is 2.0, so this is active in
every run today. Channels 1–2 are near-flat (cv≈0.075 vs 0.81), so impact is likely small —
which needs *measuring*, not assuming. Options: amend the doc, amend the code (changes the
objective, breaks bit-identity with prior runs), or redefine the split.

**D2 — `spec_aniso` normalises across held-out bands (inert today).**
`SPEC §4:204-209` forbids normalisers including held-out band content.
`losses/spectral.py:130-137` applies no `band_mask`, normalising by a central 24×24 block that
at Stage-0 geometry spans all of `B_low` and part of `B_harm`. D-FFT-11 decision 2
*deliberately* chose this for parity with the `scoring/morphology` comparator, so code and doc
were each defensible alone. Weight is 0.0, so nothing is contaminated. Recommended: keep
parity, record that `spec_aniso` may not be enabled under the band holdout — must land before
any Stage-0 config with `w_spec_aniso ≠ 0` is frozen.

---

## 5. Proposed execution plan

Phased so that each phase is independently verifiable and independently abandonable. Phases 1
and 2 are safe and mechanical. Phase 3 waits on §1. **No phase merges without owner
validation.**

### Phase 0 — prerequisite (blocks everything)

1. Commit the pending `feature/fft-training` working state (`CLAUDE.md` +97, `DECISIONS.md`
   +45, `.claude/agents/*` model pins) and **track `.claude/rules/orchestration.md`**.
   *Verify:* `git status` clean; `@.claude/rules/orchestration.md` resolves from a second
   worktree. Without this, every claim below is true only in one worktree.

### Phase 1 — safe corrections (§3), no owner input needed

2. Firewall-guard overclaims (§3.1) → *verify:* `pytest tests/test_firewall.py -q` green;
   claims match `RECOVERY_SIDE`/`FORBIDDEN` parsed from source.
3. `CLAUDE.md` §2 → pointer at `link_payloads.sh` (§3.2) → *verify:* a fresh worktree
   provisions from the doc alone.
4. Delete stale test counts (§3.3); fix wrong numbers (§3.4) → *verify:* each new number
   recomputed from its cited artifact.
5. Source-contradicted claims (§3.5) and stale line refs (§3.6) → *verify:* scripted check
   that every `file:line` in `docs/` resolves to the named symbol; keep the three known-good
   `CODE_REALITY.md` refs.
6. Evidence hygiene (§3.7) and ledger repairs (§3.8) → *verify:* every cited run directory
   exists; `D-FFT-N` appears exactly once each.
7. `rngrn scan-datasets --refresh` for `three_gene_qvar_smoke` in the canonical store (§3.9)
   → *verify:* all 14 checksums match manifests.

### Phase 2 — consolidation (§2)

8. Archive the two completed handoffs to `docs/archive/` with a one-line supersession header;
   update the three inbound references → *verify:* no broken links repo-wide.
9. Add supersession/scope headers where a doc reads as current but describes retired work —
   `GOAL_tica_equivalent.md` §3, `DATA_INTO_MODEL.md`, C1/C2 → *verify:* each header names the
   superseding document.
10. Promote `CLAUDE.local.md`'s durable gotchas to tracked docs; **correct its three false
    cheat-sheet bullets first** → *verify:* the promoted text is reproducible on a fresh
    worktree.
11. Render `TIMESCALE_MU.md`'s six placeholders from the committed arrays, **or** mark the doc
    PARKED → *verify:* no `<!--…-->` placeholders remain, or the banner is explicit.

### Phase 3 — blocked on §1

12. Fold F-D1-5 into SPEC §3/§4/§9, PLAN §2 falsifiers and M2, and open D-FFT-15.
13. Apply the owner's rulings on the split defect, the gate conflict, and the objective
    conflict.
14. Update METHODS §9 and decide the artifact's fate.

### What I recommend

Run **Phase 0 immediately** — the dangling `@`-import is a live hazard and the fix is
mechanical. Then Phase 1 in one pass, since it is all evidence-backed and reversible. Hold
Phase 2 step 9 until §1.3 is settled, because the supersession header's wording depends on
which objective wins. Phase 3 needs the owner.

---

## What is NOT established

- No dataset was regenerated, rescanned or re-harvested; no trainer was run; nothing was
  merged.
- Per-sample corpus statistics requiring `payload.h5` reads beyond the split check were not
  recomputed.
- Statistical re-derivations in C1 §3.2/§8, C2 §1/§5/§6, BIO_VIABILITY §1–§3 and
  TIMESCALE_MU §3 were not re-run from their arrays.
- `SPEC §7:325-327`'s "stored `k_star_fft` differs from live `raps()` by 1.6–3.7 %" has **no
  source anywhere** and is load-bearing — SPEC calls it "the only thing keeping the headline
  validation metric non-circular".
