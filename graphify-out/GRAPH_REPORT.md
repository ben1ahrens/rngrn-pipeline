# Graph Report - repo-mapper-agent  (2026-08-11)

## Corpus Check
- 358 files · ~1,704,343 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 5458 nodes · 8908 edges · 309 communities (258 shown, 51 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 855 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8440ff19`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Three Gene MultiL Dataset Samples
- Three Gene Train Dataset Samples
- Overparam Scoring Tests
- Three Gene Classical Train Samples
- Target Report Aggregation Tests
- Repo Conventions & Review Agents
- Two Gene Classical Train Samples
- Figure Generation Scripts
- Cross-L Evaluation Tests
- Batched Recovery Tests
- L-Generalisation Recovery Tests
- Morphology Scoring Tests
- Permutation Scoring Tests
- QSS Reduction & Robustness Analysis
- Three Gene Test Dataset Manifest
- Gate Contract Tests
- Three Gene Qvar Dataset Samples
- Three Gene Val Dataset Manifest
- Dataset Families & Identifiability Docs
- Experiment Arm Scoring Tests
- Turing Hinges & Steady-State Solve
- Three Gene Classical Test Manifest
- Batched RNGRN Model
- C1 Analysis Script
- Bio-Viability Stage0 Script
- Reference RD Solver Models
- Benchmark Grouping Tests
- Reproducibility Scoring Tests
- Cross-Cutting Science Concepts (docs)
- Two Gene Classical Test Manifest
- Three-Gene Dataset Generator
- Rollout Simulation Tests
- Cross-Target Tuning Control Arm
- Claude Settings & Hooks
- Three Gene Classical Val Manifest
- Steady-State Newton Solver
- Dataset Cache & Loading
- Plausibility Scoring Tests
- Firewall Gate & Experiment Configs
- Two Gene Classical Val Manifest
- CLI Commands
- Science Correctness Anchors
- Optim Benchmark Command
- JSONL Run Index
- Config Loading & Override Tests
- Answer Key Gate
- L-Generalisation Eval Module
- Config Identity & Hashing
- Numerics: ETDRK4 Rollout Integrator
- Training History Recorder
- Losses Unit Tests
- RNGRN Model Core (dispersion/jacobian)
- Plot Arrays Artefact Tests
- Experiment Tracking Backend
- Long-Training & Trivial-Baseline Experiments
- L-Generalisation Consistency Scoring
- Three Gene Qvar Smoke Manifest
- Training History & Plausibility Modules
- Loss Weighting Strategies
- Morphology Classification Module
- Cross-L Residual & Frame Tests
- D-Ratio Biological Prior
- Bio-Box Plausibility Config
- C2 Init-Bias Script
- Topology Plotting
- Run Export to CSV
- Composite Loss Module
- RNGRN Model Definition
- Observables & k* Extraction
- Loss Staging & Weighting Tests
- Parameter Vector Utilities
- Data Firewall Enforcement
- Three Gene MultiL Manifest
- Three Gene Train Manifest
- Sweep Runner & No-Op Decisions
- Frame-Scale Anchor & Split Hinges
- Run Metadata & Checkpoint I/O
- Dataset Registry
- Plottable Array Persistence
- Rollout Module (Lift-and-Simulate)
- Recovery Scoring (validate.py)
- Cubic Dispersion Backend Tests
- Robustness Baseline Figure (docs)
- Turing-First Homotopy Experiment
- Ratio-Based Loss Weighting Tests
- Determinism Tests
- Robustness Cloud Tests
- Turing Hinge Split Tests
- Dataset Registry Tests
- Three Gene Classical Train Manifest (partial)
- Three Gene Qvar Manifest (partial)
- Two Gene Classical Train Manifest (partial)
- Robustness Baseline Experiment (exp11)
- Independent Forward RD Solver
- Firewall Import-Direction Tests
- Morphology Separability Tests
- C1 Restarts Readout Script
- C2 Restarts Readout Script
- Cubic Dispersion Benchmark Experiment
- Model-Seed No-Op Regression Tests
- Dispersion Curve Plot Data
- Three Gene Classical Train Answer-Key Fields
- Three Gene Qvar Answer-Key Fields
- Two Gene Classical Train Answer-Key Fields
- C1/C2 Jacobian Probe Script
- C1 Queue Script (worker 1)
- C1 Queue Script (worker 2)
- C1 Queue Script (worker 3)
- C1 Queue Script (worker 4)
- C1 Queue Script (worker 5)
- C1/C2 Repro Script
- C2 Jacobian Probe Script
- C2 Repro Script
- Loss Staging Factor
- Dispersion Argmax-at-k0 Tests
- C2 Cell Launcher Script
- C2 Summary Script
- Spectral Amplitude Experiment
- Stage0 Prior Recovery Launcher
- Comp-Tuning Cell Launcher
- Comp-Tuning Summary Script
- Loss Config & Staging Tests
- Model Device Property
- Cross-L Transfer Decisions (docs)
- No-Op Knob Check Script
- Batched Steady-State Newton
- Dispersion/Turing-OK Consistency Test
- NC1 Immobile-Node Diagnostic
- Guarded Trainer Launcher
- Comp-Tuning Drive Script
- K* Bias Tuning Script
- Morphology Call Result Type
- C2 Queue Script
- Residual Sweep Experiment (exp06)
- Payload Symlink Script
- Scoring Package Init
- Reference Bank Centroids
- Test Fixture Conftest
- Angular Feature Separability Test
- Held-Out Morphology Classification Test
- Train-Split Centroid Test
- Centroid vs Sample Bank Test
- Classification Margin Test
- Within/Between Class Distance Test
- Constant Field NaN-Guard Test
- Train-Scale Provenance Test
- Recovery/Scoring Import Direction Test
- Repo Root
- test_benchmark_grouping.py
- test_plausibility.py
- td_figures.py
- make_figures.py
- terms.py
- test_canon_generate.py
- benchmark.py
- numerics.py
- turing_labyrinth/manifest.json
- turing_spots/manifest.json
- cli.py
- Part 2 — Decisions
- PREREGISTRATION — the pass conditions, fixed before the runs that are judged by them
- lgen.py
- cache.py
- RecoveryInput
- CLAUDE.md — working conventions for this repository
- turing_ok
- plot_topology
- 1. Finite-`mu` robustness — undoing the quasi-steady-state reduction
- test_determinism.py
- open_index
- C2 — tuning the NON-COMPETITIVE (`nc1`) form against the pre-registered criteria
- Measuring robustness — what exists, what is broken, and what the data says
- TrainingHistory
- total.py
- _StubModel
- observables.py
- test_nondim_units.py
- test_plot_arrays.py
- results_of
- Part 1 — Evidence-integrity defects (read this before trusting any pre-2026-07-29 number)
- test_experiment_arms.py
- test_experiment_arms.py
- _Key
- `data/datasets/` — the registered datasets, and how training data is generated
- data/registry.py
- history.py
- index.py
- _softplus_hinge
- canon_generate.py
- plausibility_report
- test_dispersion_cubic.py
- Canonical Turing datasets
- Code reality map — what is wired, what is a stub, what the docs get wrong
- Part 2b — Stage 0: the biological-viability precondition (unit c-bioviab, 2026-08-03)
- Sources consulted
- fit
- stage0_figures.py
- io.py
- test_robustness.py
- stage0_figures.py
- TUNING.md — what Claude Code tunes, implements, and validates on CUDA
- BIO_VIABILITY.md
- Local dataset setup (one-time, per machine)
- L-decoupled and multi-L datasets
- exp02_objective_fix.py
- _runlog.py
- rngrn-pipeline
- _fake_table
- Handoff — identifiability experiments branch
- HANDOFF — visual inspection of the training data
- SqliteIndex
- run_sweep
- test_registry.py
- HANDOFF — the canonical Turing training datasets
- three_gene_classical_train/manifest.json
- three_gene_multiL/manifest.json
- three_gene_train/manifest.json
- two_gene_classical_train/manifest.json
- exp11_robustness_baseline.py
- simulate_to_attractor
- _FakeModel
- _imports_of
- RuntimeError
- _FakeModel
- repo-mapper.md
- Pre-merge checklist
- Running trainers in rngrn-pipeline
- _loo_nearest_member
- `.claude/` — agent configuration for rngrn-pipeline
- Datasets and payloads
- Starting new work in rngrn-pipeline
- Recording a science decision
- dtype
- c1_restarts.py
- c2_restarts.py
- exp10_cubic_dispersion.py
- param_prior_batched
- _arrays
- _arrays
- c1_jacobian_probe.py
- c1_queue2.sh
- c1_queue3.sh
- c1_queue4.sh
- c1_queue5.sh
- c1_queue.sh
- c1_repro.py
- c2_jacobian_probe.py
- c2_repro.py
- tune_comp_noop_check.py
- exp05_pixel_minibatch.py
- Any
- firewall-auditor.md
- Reporting a number
- three_gene_val dataset (registry dataset_id)
- c2_cell.sh
- c2_summary.py
- parseval_channels
- stage0_part4_prior_recovery.sh
- tune_comp_cell.sh
- tune_comp_summarise.py
- _StubModel
- exp02_objective_fix.py
- _StubModel
- evidence-auditor.md
- merge-damage-hunter.md
- numerics-reviewer.md
- c2_noop_check.py
- _turing_JD
- expB_control_matched.yaml (Experiment B control, N=2 matched)
- pre-push
- c2_immobile.py
- exp07_long_training.py
- exp09_anchor_and_ensemble.py
- guarded_run.sh
- run_cell
- tune_comp_kstar_bias.py
- DECISIONS.md — the science decision register
- c2_queue.sh
- exp06_residual_sweep.py
- link_payloads.sh
- scoring/__init__.py
- conftest.py
- test_class_means_separate_the_classes
- test_classify_morphology_beats_baseline_on_held_out
- test_baked_in_centroids_match_the_train_split
- test_centroid_bank_is_weaker_than_the_real_sample_bank
- test_margin_is_smaller_for_incorrect_calls
- test_within_class_distance_is_smaller_than_between_class
- test_constant_field_raises_rather_than_returning_nan
- test_train_scale_matches_the_split_it_claims
- test_scoring_init_imports_no_sibling_modules
- test_overparam_is_not_imported_by_recovery_side
- test_probe_period_stays_inside_the_generator_feasibility_window
- test_scoring_init_imports_no_sibling_modules
- test_overparam_is_not_imported_by_recovery_side
- ndarray
- ndarray
- ndarray
- ndarray
- ndarray
- NamedTuple
- ndarray
- ndarray
- NamedTuple
- ndarray
- ndarray
- ndarray

## God Nodes (most connected - your core abstractions)
1. `sample_keys` - 93 edges
2. `sample_keys` - 93 edges
3. `sample_keys` - 89 edges
4. `sample_keys` - 89 edges
5. `RNGRN` - 89 edges
6. `RNGRN` - 84 edges
7. `sample_keys` - 59 edges
8. `sample_keys` - 59 edges
9. `sample_keys` - 53 edges
10. `sample_keys` - 53 edges

## Surprising Connections (you probably didn't know these)
- `Strict (max Re eig(J)<0) vs loose (tr(J)<0) Turing criterion` --implements--> `turing_ok()`  [INFERRED]
  docs/ROBUSTNESS_MEASUREMENT.md → src/rngrn/eval/analysis.py
- `D-EVID-14: three physics consumers read dimensionless model.D on the non-dimensional path` --rationale_for--> `robustness_cloud()`  [EXTRACTED]
  docs/DECISIONS.md → src/rngrn/eval/analysis.py
- `Split-support Turing hinges (disjoint k-support for stability/instability terms)` --implements--> `turing_hinges_split()`  [INFERRED]
  docs/STATE_OF_THE_SCIENCE.md → src/rngrn/losses/terms.py
- `Frame-scale anchor (firewall-legal x* proxy from image mean)` --implements--> `frame_scale_anchor()`  [INFERRED]
  docs/STATE_OF_THE_SCIENCE.md → src/rngrn/losses/terms.py
- `D-ratio soft prior (biological plausibility, centred 7.5)` --implements--> `param_prior()`  [INFERRED]
  docs/STATE_OF_THE_SCIENCE.md → src/rngrn/losses/terms.py

## Import Cycles
- 2-file cycle: `src/rngrn/losses/__init__.py -> src/rngrn/losses/total.py -> src/rngrn/losses/__init__.py`

## Hyperedges (group relationships)
- **Four read-only reviewer agents guarding rngrn-pipeline invariants** — claude_agents_evidence_auditor_agent, claude_agents_firewall_auditor_agent, claude_agents_merge_damage_hunter_agent, claude_agents_numerics_reviewer_agent, claude_readme_overview [EXTRACTED 1.00]
- **Four procedural skills covering worktree, training, dataset, and decision workflows** — claude_skills_harvest_dataset_skill_guide, claude_skills_new_worktree_skill_guide, claude_skills_record_decision_skill_guide, claude_skills_run_training_skill_guide, claude_readme_overview [EXTRACTED 1.00]
- **Components that jointly form the pre-merge quality gate** — claude_rules_pre_merge_checklist_checklist, claude_agents_firewall_auditor_agent, claude_agents_evidence_auditor_agent, claude_agents_merge_damage_hunter_agent, claude_skills_harvest_dataset_skill_guide [INFERRED 0.85]
- **Milestone 1 fully-observed N=2 baseline recovery family (competitive + nc1 forms)** — configs_milestone1_gm, configs_milestone1_schnak, configs_nc1_milestone1_gm, configs_nc1_milestone1_schnak [INFERRED 0.85]
- **Configs consuming the three_gene_val registry dataset** — configs_expa_control_full, configs_expa_hidden_channel, configs_m3_registry, configs_nc1_m3_registry, dataset_three_gene_val [EXTRACTED 1.00]
- **nc1 (non-competitive) recovery-form config family** — configs_nc1_milestone1_gm, configs_nc1_milestone1_schnak, configs_nc1_milestone2_gm_partial, configs_nc1_milestone3_3gene, configs_nc1_m3_registry, concept_nc1_non_competitive_form [EXTRACTED 1.00]
- **D-EVID-7 k* control correction propagated across tuning and pre-registration docs** — docs_decisions_d_evid_7, docs_bio_viability, docs_c1_competitive_tuning, docs_c2_nc1_tuning, docs_preregistration [INFERRED 0.85]
- **Near-immobile third node mechanism measured independently across four docs** — concept_near_immobile_third_node, docs_bio_viability, docs_c2_nc1_tuning, docs_robustness_measurement, docs_goal_tica_equivalent, docs_decisions_d2 [INFERRED 0.85]
- **Strict-vs-loose Turing criterion defect (D-EVID-11) and its downstream corrections** — docs_decisions_d_evid_11, docs_state_of_the_science, docs_c1_competitive_tuning, docs_robustness_measurement, docs_decisions_d9 [INFERRED 0.85]

## Communities (309 total, 51 thin omitted)

### Community 0 - "Three Gene MultiL Dataset Samples"
Cohesion: 0.04
Nodes (94): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+86 more)

### Community 1 - "Three Gene Train Dataset Samples"
Cohesion: 0.04
Nodes (90): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+82 more)

### Community 2 - "Overparam Scoring Tests"
Cohesion: 0.08
Nodes (44): _as_index_tuple(), _as_square(), observed_block_agreement(), overparam_report(), overparam.py — over-parameterisation metrics for EXPERIMENT B (SCAFFOLD).  STATU, Sign pattern of J with a magnitude-relative dead-zone.      Entries with |J_ij|, Quantify whether the model's SPARE species are inert (Experiment B, half (i))., Compare the OBSERVED sub-block of J_rec against the truth (half (ii)).      This (+36 more)

### Community 3 - "Three Gene Classical Train Samples"
Cohesion: 0.07
Nodes (60): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+52 more)

### Community 4 - "Target Report Aggregation Tests"
Cohesion: 0.08
Nodes (54): aggregate_target_report(), _finite(), _fit_one_seed(), _mean_median(), _pattern_block(), _per_seed_audit(), target_report.py — the target-report driver: the measurement instrument for the, Re-hydrate one run's Jacobian for the reproducibility aggregation, at the REQUES (+46 more)

### Community 5 - "Repo Conventions & Review Agents"
Cohesion: 0.07
Nodes (62): No discretised Laplacian in training — analytic in Fourier space, evidence-auditor agent spec, firewall-auditor agent spec, merge-damage-hunter agent spec, numerics-reviewer agent spec, CLAUDE.md — repository working-conventions contract, .claude/README.md — agent configuration overview, Pre-merge checklist rule (+54 more)

### Community 6 - "Two Gene Classical Train Samples"
Cohesion: 0.07
Nodes (54): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+46 more)

### Community 7 - "Figure Generation Scripts"
Cohesion: 0.15
Nodes (23): _decode(), fig_cross_L(), fig_dispersion(), fig_hyperparams(), fig_parameters(), fig_pattern(), fig_raps(), fig_training() (+15 more)

### Community 8 - "Cross-L Evaluation Tests"
Cohesion: 0.05
Nodes (64): Solve f(x*) = 0 for the model's OWN homogeneous steady state.      Damped Newton, steady_state(), cmd_evaluate(), Re-simulate a saved run's model, at its OWN domain size by default.      THE DEF, _dedup_L(), evaluate_across_L(), grid_for_L(), physical_model_from_checkpoint() (+56 more)

### Community 9 - "Batched Recovery Tests"
Cohesion: 0.06
Nodes (51): TrainConfig, DataFirstStaging, FixedWeighting, WRAPPER that applies `staging_factor` to selected terms of ANY inner strategy., Staging is itself a static schedule, so this is exactly the inner strategy's., _batched_adam_steps(), _frame(), _kgrid() (+43 more)

### Community 10 - "L-Generalisation Recovery Tests"
Cohesion: 0.04
Nodes (50): _frame(), test_lgen.py — L-generalisation: non-dimensional recovery + the cross-L metric (, The residual term is the one place L enters the objective. It must be the SAME, The old floor was an ABSOLUTE 2.0 rad/length, so once kstar_obs < 0.25 the grid, The floor's only legitimate job is to keep the grid non-degenerate when kstar_ob, theta_D = randn(N)*0.5 - 2*log(kstar_obs) => D starts at median 1/kstar_obs**2., low_basal has its own D-ratio scheme (unit 2); kstar_obs must not perturb it., The SAME formula (D_model median = 1/kstar_obs**2) must be right on both paths (+42 more)

### Community 11 - "Morphology Scoring Tests"
Cohesion: 0.06
Nodes (37): _load_split(), test_morphology_scoring.py — tests for rngrn.scoring.morphology.  Morphology (sa, The angular features must carry the stripes signal, and skew the spots signal., A train-only bank must generalise to val+test above the majority baseline., TRAIN_CLASS_MEANS must be the three_gene_train class means it claims to be., The documented ordering between the two banks, pinned on held-out data.      Mea, The margin must carry information about how borderline a call is.      Measured:, The distance must order pairs the way the classes do, on average. (+29 more)

### Community 12 - "Permutation Scoring Tests"
Cohesion: 0.07
Nodes (34): _J3(), _J4(), _pattern(), _permute(), test_permutation_scoring.py — fast CPU tests for rngrn.scoring.permutation.  The, EXPERIMENT A (N=3, m=2): exactly ONE admissible relabelling, so alignment is a, Brute-force the objective independently and confirm the returned score is the ma, EXPERIMENT B: model N=3, truth 2x2. The 2x2 observed block is well-defined and (+26 more)

### Community 13 - "QSS Reduction & Robustness Analysis"
Cohesion: 0.07
Nodes (37): cmd_analyze(), _draw_JD_cloud(), linear_stability(), _model_JD(), _perturb_cloud(), analysis.py — linear stability, Turing conditions, and the robustness cloud.  Re, Draw n_samples independent log-normal multiplicative perturbations of the     PH, Vectorized log-normal cloud around a single PHYSICAL (J, D).      Ported from sc (+29 more)

### Community 14 - "Three Gene Test Dataset Manifest"
Cohesion: 0.06
Nodes (45): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+37 more)

### Community 15 - "Gate Contract Tests"
Cohesion: 0.09
Nodes (33): _bins(), test_gate_contract.py — the gate reads per-sample geometry and BOTH wavenumbers., Firewall spot-check at the new fields: neither k* may ride along on RecoveryInpu, The caller's L must not become a fallback for a file that has none., The FFT number is a diagnostic, not the headline — absent is legal, not fatal., The five file-backed configs must not carry an L that silently loses to the file, The linear k* expressed in FFT bins: k* = 2*pi*n/L  =>  n = k* L / 2pi., MEASURED FACT: every three_gene sample's linear k* is exactly 6.00 FFT bins. (+25 more)

### Community 16 - "Three Gene Qvar Dataset Samples"
Cohesion: 0.05
Nodes (59): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+51 more)

### Community 17 - "Three Gene Val Dataset Manifest"
Cohesion: 0.07
Nodes (44): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+36 more)

### Community 18 - "Dataset Families & Identifiability Docs"
Cohesion: 0.13
Nodes (26): align_permutation(), AlignmentScore, _as_field_stack(), _as_square(), _candidate_perms(), _centred_or_raise(), _cosine_uncentered(), _fro_rel_err() (+18 more)

### Community 19 - "Experiment Arm Scoring Tests"
Cohesion: 0.07
Nodes (27): _leak_instrumentation(), The image-blind control for the k* numbers in this project — ONE PER REFERENCE., Grade a RecoveryResult against an AnswerKey. Returns a flat metric dict.      L, Read a scalar reference wavenumber off the answer key as a float, or NaN if abse, Relative error of the recovered k* against a reference, NaN when no reference., _rel_err(), _rel_ref(), score_recovery() (+19 more)

### Community 20 - "Turing Hinges & Steady-State Solve"
Cohesion: 0.05
Nodes (56): Split-support Turing hinges (disjoint k-support for stability/instability terms), diagnose(), EXP01 — Can optimisation REACH Turing instability from a non-Turing init?  The u, run(), diagnose(), fit(), low_basal_init(), EXP03 - "Turing-first" (homotopy) recovery vs the current random-restart scheme. (+48 more)

### Community 21 - "Three Gene Classical Test Manifest"
Cohesion: 0.07
Nodes (41): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+33 more)

### Community 22 - "Batched RNGRN Model"
Cohesion: 0.06
Nodes (22): Closed-form 2-species sigma(k). SELF-TEST reference; smooth sqrt clamp keeps a, BatchedRNGRN, Gene-regulatory reaction model with learnable, biologically-meaningful params., f(x) for a batch of state vectors. x: (..., N) -> (..., N). Pointwise: no grid., J_ij = d f_i / d x_j at xstar (shape (N,)). Form-agnostic (autodiff of the, sigma(k) = max_i Re eig(J - k^2 diag(D)) over a grid of k. General-N,         di, Closed-form 2-species sigma(k). SELF-TEST reference; smooth sqrt clamp keeps a, max_i Re eig(M) for BATCHED 3x3 M, exactly, via closed-form cubic roots.      Wh (+14 more)

### Community 23 - "C1 Analysis Script"
Cohesion: 0.19
Nodes (16): block_perm(), block_repro(), block_restarts(), block_table(), fmt(), load_cells(), main(), pct() (+8 more)

### Community 24 - "Bio-Viability Stage0 Script"
Cohesion: 0.13
Nodes (28): balanced_picks(), build_batched(), build_model(), draw_batch(), evaluate_batch(), evaluate_draw(), harvest(), _inv_softplus() (+20 more)

### Community 25 - "Reference RD Solver Models"
Cohesion: 0.09
Nodes (16): _dispersion_max(), GiererMeinhardt, _jac_fd(), rd_models.py — reference reaction-diffusion systems (ANSWER-KEY SIDE).  These ar, Base: a reaction f(x), diffusion D, domain L, and an analytic answer key., The quarantined truth: x*, J, k*, sigma_max, D, coefficients. VALIDATION ONLY., Saturated Gierer-Meinhardt (stripes). f = a + u^n/(v(1+K u^n)) - b u ; g = u^n -, Schnakenberg (spots). f = gamma(a - u + u^2 v) ; g = gamma(b - u^2 v). (+8 more)

### Community 26 - "Benchmark Grouping Tests"
Cohesion: 0.10
Nodes (30): reproducibility_markdown(), test_benchmark_grouping.py — cross-seed aggregation in optim.benchmark (D-EVID-1, THE DEFECT: three seeds of one arm must be ONE row with n_seeds == 3., n_seeds alone cannot be audited; the actual seeds must be on the row., Two TARGETS are two rows. Pooling them would be a worse defect than the one fixe, Different configs stay separate — the property config_id was protecting., Rows recorded before `arm_id` existed carry no seed-independent identity.      T, `reproducibility_table` has NO callers — which is exactly why it needs a test. (+22 more)

### Community 27 - "Reproducibility Scoring Tests"
Cohesion: 0.10
Nodes (27): _J3(), test_reproducibility_scoring.py — fast CPU tests for rngrn.scoring.reproducibili, Shrinking the threshold to near-zero recovers the raw (near-machine-precision), `repro_J_vector` must round-trip to the ORIGINAL J, not its sign structure., The prereg §3.1 sweep must give DIFFERENT numbers on a threshold-sensitive     t, A matrix already valued in {-1, 0, 1} passes through unchanged — this is the, This module compares recovered J's to each other, never to a ground truth, so it, The overall scale of J is not pinned by the objective (documented elsewhere in (+19 more)

### Community 28 - "Cross-Cutting Science Concepts (docs)"
Cohesion: 0.11
Nodes (21): Frame-scale anchor (firewall-legal x* proxy from image mean), D-EVID-5: loss.weights.param_prior was a silent no-op in the library path, D10: DEFAULT_SIGN_ZERO_RTOL=0.05 and topology_consistency defined as modal fraction, flagged UNCALIBRATED, D11: gradnorm/ntk weighting strategies now raise rather than silently running fixed weights, D2: d_ratio is largest/second-largest diffusivity, excluding an immobile node, D7: loss.weights.resid defaulted to 0.0 — settled OFF, D8: split hinges + frame-scale anchor promoted into the library (0% to 36.8% Turing), D9: low-basal init implemented but default OFF (+13 more)

### Community 29 - "Two Gene Classical Test Manifest"
Cohesion: 0.07
Nodes (38): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+30 more)

### Community 30 - "Three-Gene Dataset Generator"
Cohesion: 0.09
Nodes (34): anisotropy(), audit_leak(), classify(), expand_multiL(), expand_qvar(), feasible_periods(), jac_fd(), main() (+26 more)

### Community 31 - "Rollout Simulation Tests"
Cohesion: 0.05
Nodes (57): Config, Canonical JSON (sorted keys) of the full config, for hashing., Identity of the EXPERIMENT ARM: this config with the seeds neutralised., _collapsed(), rollout.py — lift-and-simulate: does the recovered circuit actually pattern?  Tr, Has the field decayed so far that 'unpatterned' is already locked in?      THE S, Integrate d x/dt = D lap(x) + f(x) from x* + noise. Returns a result dict., Has the pattern stopped changing? THE STOPPING RULE — read this before trusting (+49 more)

### Community 32 - "Cross-Target Tuning Control Arm"
Cohesion: 0.11
Nodes (25): D-EVID-12: sign-threshold sweep was a no-op that mislabeled its own rtol, agreement(), _cell_key(), load_runs(), tune_comp_crosstarget.py — the CONTROL half of pre-registration 3.1.  3.1 requir, The tuning axes, read off the run's OWN frozen config, never off the command lin, cell/sample/seed -> J, for Turing-reaching runs under one runs-root., report() (+17 more)

### Community 33 - "Claude Settings & Hooks"
Cohesion: 0.06
Nodes (32): allowWrite, hooks, PreToolUse, permissions, allow, ask, deny, sandbox (+24 more)

### Community 34 - "Three Gene Classical Val Manifest"
Cohesion: 0.07
Nodes (36): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+28 more)

### Community 35 - "Steady-State Newton Solver"
Cohesion: 0.07
Nodes (34): build_strategy(), losses/ — the composite objective (RECOVERY SIDE).  terms.py     : differentiabl, compute_terms(), compute_terms_batched(), parts_member(), total.py — assemble the composite objective from named, source-tagged terms.  Ea, Batched twin of `compute_terms`. Returns (term_vals, parts, converged).      `mo, Raised when the model-derived steady state fails to converge (fail-loud). (+26 more)

### Community 36 - "Dataset Cache & Loading"
Cohesion: 0.10
Nodes (28): _checksum(), _dataset_dir(), exists(), generate(), load_frame(), cache.py — the content-addressed dataset cache (Stage 1 core).  data/cache/<data, Generate the dataset for `spec` if absent; return its directory. Idempotent., Load ONLY (frame, L). Answer key is deliberately NOT returned here — see gate.py (+20 more)

### Community 37 - "Plausibility Scoring Tests"
Cohesion: 0.06
Nodes (31): D-ratio soft prior (biological plausibility, centred 7.5), D1: D-ratio prior centred at 7.5 (literature Nodal/Lefty), not ~100 (generator median), param_prior(), Soft biological-plausibility prior on the model's OWN parameters. Recovery-side, _fake_recovery_input(), test_plausibility.py — unit tests for rngrn.scoring.plausibility and rngrn.losse, One runaway alpha[i,j] makes the whole parameter implausible, not just that, A box loaded from an arbitrary, hand-built YAML with numbers that appear nowhere (+23 more)

### Community 38 - "Firewall Gate & Experiment Configs"
Cohesion: 0.24
Nodes (13): model.seed override making train.seed a no-op (D-EVID-4 finding), nc1 (non-competitive) recovery form, model.form override, base.yaml (shared config defaults), milestone1_gm.yaml (Milestone 1, Gierer-Meinhardt stripes), milestone1_schnak.yaml (Milestone 1, Schnakenberg spots), milestone2_gm_partial.yaml (Milestone 2, GM with hidden v channel), milestone3_3gene.yaml (Milestone 3, loose HDF5 3-gene dataset), nc1_milestone1_gm.yaml (Milestone 1 GM, nc1 form) (+5 more)

### Community 39 - "Two Gene Classical Val Manifest"
Cohesion: 0.08
Nodes (35): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+27 more)

### Community 40 - "CLI Commands"
Cohesion: 0.05
Nodes (59): _dedup_L(), evaluate_across_L(), grid_for_L(), physical_model_from_checkpoint(), lgen_eval.py — DOES THE RECOVERED GRN PATTERN THE SAME WAY ON A DIFFERENT-SIZED, Grid size at domain size `L` that holds dx = L/n fixed at its L_train value., Return (model_with_PHYSICAL_D, L_train) for a checkpoint loaded by io.load_check, The L to evaluate at: the caller's, with L_train prepended if it is not among th (+51 more)

### Community 41 - "Science Correctness Anchors"
Cohesion: 0.09
Nodes (26): _b3_fold_trap_model(), _b3_sweep(), test_science.py — cheap correctness anchors (seconds, CPU).  These check the sci, The pitfall-8.1 guard: on stiff diffusion, split-step IMEX collapses to flat whi, The counterexample. tr(J) < 0 but max Re eig(J) > 0 — not Turing, not anything., RE-MEASURED 2026-08-04 after the turing_ok correction (docs/DECISIONS.md D-EVID-, Walk a deterministic ensemble ONCE (the failing solves are the expensive ones) a, THE REGRESSION PIN. terms.steady_state's multistart is a pure rescue path: its f (+18 more)

### Community 42 - "Optim Benchmark Command"
Cohesion: 0.07
Nodes (50): build_parser(), cmd_benchmark(), cmd_export(), cmd_generate_data(), cmd_list_datasets(), cmd_register_data(), cmd_scan_datasets(), cmd_sweep() (+42 more)

### Community 43 - "JSONL Run Index"
Cohesion: 0.14
Nodes (9): _coerce(), JsonlIndex, _now_iso(), index.py — append-only metadata indices with two interchangeable backends.  Both, `where` is a SQL predicate string (no leading WHERE), e.g.         "recovered_tu, Reduce a value to something SQLite can store; dicts/lists -> JSON text., Python-side filter: `where` is a predicate callable row->bool (SQL string, _sql_type() (+1 more)

### Community 44 - "Config Loading & Override Tests"
Cohesion: 0.09
Nodes (36): tune_comp_noop_check.py — prove every swept knob actually CHANGES something.  Fo, The things a live knob must be able to move., run(), _set(), signature(), _load(), apply_overrides(), _build_section() (+28 more)

### Community 45 - "Answer Key Gate"
Cohesion: 0.15
Nodes (23): The recovery/ground-truth data firewall, L-encodes-k*-true domain-size leak, AnswerKey, from_3gene_hdf5(), from_cache(), from_registry(), _observe(), gate.py — THE FIREWALL GATE.  This is the single doorway between the answer-key (+15 more)

### Community 46 - "L-Generalisation Eval Module"
Cohesion: 0.08
Nodes (52): aggregate_target_report(), _finite(), _mean_median(), _pattern_block(), _per_seed_audit(), target_report.py — the target-report driver: the measurement instrument for the, Re-hydrate one run's Jacobian for the reproducibility aggregation, at the REQUES, Combine K per-seed recovery results (on ONE target) into one auditable report. (+44 more)

### Community 47 - "Config Identity & Hashing"
Cohesion: 0.05
Nodes (46): _frame(), test_lgen.py — L-generalisation: non-dimensional recovery + the cross-L metric (, The residual term is the one place L enters the objective. It must be the SAME, theta_D = randn(N)*0.5 - 2*log(kstar_obs) => D starts at median 1/kstar_obs**2., low_basal has its own D-ratio scheme (unit 2); kstar_obs must not perturb it., The SAME formula (D_model median = 1/kstar_obs**2) must be right on both paths, One known frame, three domain sizes: the recovered NETWORK is identical.      Th, L = 1 is the fixed point of the change of variables, so the two paths must coinc (+38 more)

### Community 48 - "Numerics: ETDRK4 Rollout Integrator"
Cohesion: 0.14
Nodes (19): _cached_half_coeffs(), _half_coeffs_cached(), integrate_bdf1_newton_krylov(), integrate_etdrk4(), integrate_etdrk4_rfft(), integrate_imex_split(), _phi_contour(), numerics.py — stiff time integrators for the lift-and-simulate rollout.  Three i (+11 more)

### Community 49 - "Training History Recorder"
Cohesion: 0.13
Nodes (10): Accumulates (step, member) records and renders them as dense numpy arrays., The thinning rule: stride `every`, plus the endpoints unconditionally., Freeze the scalar column order on the first record; fail loud if it changes., A member whose steady state diverged and was abandoned at `step`., Dense arrays for `plotdata.save_plot_arrays`, or {} when nothing was recorded., History provenance for the npz `meta` blob., TrainingHistory, test_empty_history_renders_nothing() (+2 more)

### Community 50 - "Losses Unit Tests"
Cohesion: 0.08
Nodes (37): LossConfig, Weights and the weighting strategy. Weights are STARTING points — Claude Code, _cos(), _flat_grad(), _model_at_steady_state(), test_losses.py — the promoted objective terms (unit 1).  Covers the three terms, Control for the test above: no interior instability -> the instability hinge bit, The whole point of the split: an unstable mode at k < k_min must NOT satisfy the (+29 more)

### Community 51 - "RNGRN Model Core (dispersion/jacobian)"
Cohesion: 0.05
Nodes (24): BatchedRNGRN, build_model(), _low_basal_raw_params(), model.py — the RNGRN model core (VALIDATED science, repackaged as-is).  Gated-pr, Gene-regulatory reaction model with learnable, biologically-meaningful params., f(x) for a batch of state vectors. x: (..., N) -> (..., N). Pointwise: no grid., J_ij = d f_i / d x_j at xstar (shape (N,)). Form-agnostic (autodiff of the, sigma(k) = max_i Re eig(J - k^2 diag(D)) over a grid of k. General-N,         di (+16 more)

### Community 52 - "Plot Arrays Artefact Tests"
Cohesion: 0.17
Nodes (13): _lgen_out(), test_plot_arrays.py — the plottable-array artefact, the trajectory recorder, the, House rule: index rows are FLAT SCALARS. The morph vector becomes one column eac, `plotdata` accepts the ANSWER KEY's (J, D) and `export` reads the scored indexes, The recorder runs INSIDE the training loop, so it must import nothing truth-side, _seed_index(), test_export_all_writes_the_csv_files(), test_history_is_recovery_side_safe() (+5 more)

### Community 53 - "Experiment Tracking Backend"
Cohesion: 0.06
Nodes (11): build_tracker(), NoOpTracker, tracking.py — a Tracker protocol with a no-op default.  Turning on an experiment, Tracker, WandbTracker, Protocol, build_tracker(), NoOpTracker (+3 more)

### Community 54 - "Long-Training & Trivial-Baseline Experiments"
Cohesion: 0.11
Nodes (11): EXP07 - Iteration count as the axis. "Batch size introducing iterations."  THE O, EXP08 - The trivial-baseline control. Is recovered k* actually FROM THE IMAGE?, EXP09 - Two questions at the 4000-step budget found by exp07.  Q1  THE BAD-ANCHO, args_config_id(), _runlog.py — shared auditability helper for the experiment scripts.  The library, sha256 of the canonical (sorted-key) JSON of the script's resolved argument, Mint a run_id + config_id + provenance block for one script invocation.      Cal, Append one flat-scalar row to the SAME run index the library uses     (<runs_roo (+3 more)

### Community 55 - "L-Generalisation Consistency Scoring"
Cohesion: 0.14
Nodes (18): d_ratio(), _group_key(), lgen_consistency(), lgen_table(), _log10_spread(), _mean_of_finite(), modal_sign_agreement(), per_run_lgen_metrics() (+10 more)

### Community 56 - "Three Gene Qvar Smoke Manifest"
Cohesion: 0.08
Nodes (28): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+20 more)

### Community 57 - "Training History & Plausibility Modules"
Cohesion: 0.20
Nodes (12): BoxRow, _in_box(), load_box(), plausibility_report(), plausibility.py — biological viability of a RECOVERED model's OWN parameters.  W, True/False for a cited, bounded row; None ("not applicable") for an UNCITED or, Score a RECOVERED model's own parameters against configs/bio_box.yaml.      alph, One parsed row of bio_box.yaml. `low`/`high` are None for an unbounded or     UN (+4 more)

### Community 58 - "Loss Weighting Strategies"
Cohesion: 0.10
Nodes (22): GradNormWeighting, NTKWeighting, weighting.py — adaptive loss-weighting strategies (Stage 3 outer loop).  A weigh, Base interface. `combine` returns (scalar_total, weights_used)., Linearly ramp selected weights from 0 to base over `warmup` steps. Lets the, GradNorm-style balancing: scale each term so per-term gradient magnitudes match., NTK-PINN weighting: set weights from the diagonal of the neural-tangent-kernel /, Ratio-based loss balancing, as actually shipped by Matas-Gil & Endres     (githu (+14 more)

### Community 59 - "Morphology Classification Module"
Cohesion: 0.08
Nodes (36): _angular(), classify(), kstar_of(), laplacian_5point(), laplacian_spectral(), laplacian_torch(), morphology(), raps() (+28 more)

### Community 60 - "Cross-L Residual & Frame Tests"
Cohesion: 0.06
Nodes (46): LossConfig, Weights and the weighting strategy. Weights are STARTING points — Claude Code, DataFirstStaging, FixedWeighting, WRAPPER that applies `staging_factor` to selected terms of ANY inner strategy., _batched_adam_steps(), _cos(), _flat_grad() (+38 more)

### Community 61 - "D-Ratio Biological Prior"
Cohesion: 0.06
Nodes (48): blob_field(), colorbar_axes(), fake_sample(), flat_field(), test_td_figures.py — the figure contract for the training-data inspection notebo, Fail loud (CLAUDE.md section 4) rather than silently drawing the wrong thing., A field with 5 periods across L must peak at k = 2*pi*5/L., A few isolated Gaussian blobs — HIGH contrast but NOT periodic.      This is the (+40 more)

### Community 62 - "Bio-Box Plausibility Config"
Cohesion: 0.09
Nodes (24): Strict (max Re eig(J)<0) vs loose (tr(J)<0) Turing criterion, bio_box.yaml (biological plausibility box), bio_box.yaml alpha (production-weight scale prior), bio_box.yaml beta (basal production rate, UNCITED), bio_box.yaml d_ratio (diffusivity ratio plausibility box, centre 7.5), bio_box.yaml delta (degradation rate scale prior), D3: bio_box D-ratio upper bound of 60.0 is a reasoned proxy, not a measured bound, D5: k_star_fft (image-derived) is the headline validation target, not analytic k_star (+16 more)

### Community 63 - "C2 Init-Bias Script"
Cohesion: 0.19
Nodes (17): canon_key(), cmd_compare(), cmd_floor(), cmd_moved(), grouped_modal(), init_jacobian(), _load_by_target(), modal_fraction() (+9 more)

### Community 64 - "Topology Plotting"
Cohesion: 0.16
Nodes (17): _edge(), _label(), _layout(), _lighten(), plot_topology(), rngrn_topology.py ================ Flexible "Learned GRN Topology" plot for an R, Draw an auto-regulation self-loop just outside the node, bulging along `outdir`., Render the learned RNGRN topology.      model          : an RNGRN instance (read (+9 more)

### Community 65 - "Run Export to CSV"
Cohesion: 0.21
Nodes (16): export_all(), _melt(), _num(), export.py — the run record as TIDY (long) CSV, ready for pandas / seaborn / ggpl, One row per (run_id, step, member, variable) over every run whose npz has a hist, Write every tidy table under `out_dir`. Returns {filename: n_data_rows}., The numeric reading of a value, or '' when it has none. Bools count as 0/1., One dict per (row, non-identifier column). The whole of tidy, in six lines. (+8 more)

### Community 66 - "Composite Loss Module"
Cohesion: 0.07
Nodes (43): TrainConfig, _frame(), _kgrid(), _pair(), The BATCHED recovery path must be numerically equivalent to the SERIAL reference, The standing model.py contract: J must stay differentiable w.r.t. theta, so the, Every member's x* and its converged flag must match the serial damped Newton., Not merely 'agrees with serial' — f(x*) must be at the Newton tolerance. (+35 more)

### Community 67 - "RNGRN Model Definition"
Cohesion: 0.11
Nodes (9): Registry, build_model(), _low_basal_raw_params(), model.py — the RNGRN model core (VALIDATED science, repackaged as-is).  Gated-pr, Construct a model from a ModelConfig (registry-dispatched by architecture)., Draw raw (unconstrained) theta for the low-basal init (ported from     scripts/e, A tiny name -> implementation registry.  Used for the three varying concerns (mo, Registry (+1 more)

### Community 68 - "Observables & k* Extraction"
Cohesion: 0.08
Nodes (44): _as_index_tuple(), _as_square(), observed_block_agreement(), overparam_report(), overparam.py — over-parameterisation metrics for EXPERIMENT B (SCAFFOLD).  STATU, Sign pattern of J with a magnitude-relative dead-zone.      Entries with |J_ij|, Quantify whether the model's SPARE species are inert (Experiment B, half (i))., Compare the OBSERVED sub-block of J_rec against the truth (half (ii)).      This (+36 more)

### Community 69 - "Loss Staging & Weighting Tests"
Cohesion: 0.06
Nodes (41): RE-MEASURED 2026-08-04 after the turing_ok correction (docs/DECISIONS.md D-EVID-, test_low_basal_init_gains_the_positive_diagonal_but_NOT_turing_reachability(), annotate(), main(), domain_stats(), dominant_wavelength_px(), measure(), measured_morphology() (+33 more)

### Community 70 - "Parameter Vector Utilities"
Cohesion: 0.18
Nodes (9): _d_slice(), _param_block(), _param_vector(), The D entries of a flattened parameter vector (D is last in PARAM_SPEC)., The scalar columns of one member: diagnostics, per-term losses, live weights., One serial restart's state at `step`. Call BEFORE `opt.step()` so the recorded, All B members' state at `step`. `alive` (a (B,) bool tensor/array) skips members, One serial model's physical parameters, flattened in PARAM_SPEC order. (P,) (+1 more)

### Community 71 - "Data Firewall Enforcement"
Cohesion: 0.14
Nodes (16): D-EVID-9: firewall audit was a hand-copied allowlist; new modules unaudited by default, _imports(), test_firewall.py — THE firewall enforcement (not a comment, a test).  No recover, NO MODULE MAY BE UNCLASSIFIED. This is the audit's completeness guarantee., The rule CLAUDE.md §5 states, enforced here rather than only in the scorers' tes, RecoveryInput must expose only frame/L/observed_idx/N — no truth fields., The rule CLAUDE.md §5 states, enforced here rather than only in the scorers' tes, RecoveryInput must expose only frame/L/observed_idx/N — no truth fields. (+8 more)

### Community 72 - "Three Gene MultiL Manifest"
Cohesion: 0.16
Nodes (15): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+7 more)

### Community 73 - "Three Gene Train Manifest"
Cohesion: 0.16
Nodes (15): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+7 more)

### Community 74 - "Sweep Runner & No-Op Decisions"
Cohesion: 0.08
Nodes (41): _collapsed(), rollout.py — lift-and-simulate: does the recovered circuit actually pattern?  Tr, Has the field decayed so far that 'unpatterned' is already locked in?      THE S, Integrate d x/dt = D lap(x) + f(x) from x* + noise. Returns a result dict., Has the pattern stopped changing? THE STOPPING RULE — read this before trusting, _reaction_np_builder(), _saturated(), simulate() (+33 more)

### Community 75 - "Frame-Scale Anchor & Split Hinges"
Cohesion: 0.06
Nodes (40): build_plot_arrays(), lgen_fields_path(), lgen_rows(), load_plot_arrays(), plot_arrays_path(), plot_kgrid(), plotdata.py — the PLOTTABLE arrays of a run, persisted next to its scalars.  WHY, Write `arrays` to a compressed npz at `path`. Returns the file size in bytes. (+32 more)

### Community 76 - "Run Metadata & Checkpoint I/O"
Cohesion: 0.15
Nodes (11): Write the run's identity/provenance block to a sidecar JSON next to the     scri, write_meta(), append_run_index(), atomic_write_json(), load_checkpoint(), io.py — run directories, atomic writes, checkpoints, the run index.  Run layout, Write JSON to a temp file and rename on success (never a half-written file)., Each producer writes its OWN result file (train_results / analysis_results / ... (+3 more)

### Community 77 - "Dataset Registry"
Cohesion: 0.21
Nodes (13): build_manifest(), _checksum(), dataset_dir(), list_datasets(), load_manifest(), payload_path(), registry.py — the dataset store (ANSWER-KEY side).  A place to KEEP datasets you, Inspect an HDF5 payload IN PLACE and build its manifest dict (metadata only). (+5 more)

### Community 78 - "Plottable Array Persistence"
Cohesion: 0.12
Nodes (19): build_plot_arrays(), lgen_fields_path(), lgen_rows(), load_plot_arrays(), plot_arrays_path(), plot_kgrid(), plotdata.py — the PLOTTABLE arrays of a run, persisted next to its scalars.  WHY, Write `arrays` to a compressed npz at `path`. Returns the file size in bytes. (+11 more)

### Community 79 - "Rollout Module (Lift-and-Simulate)"
Cohesion: 0.06
Nodes (33): Selection logic for the canonical Turing datasets.  Synthetic rows only — no pay, Not left to luck: draw_periods checks the bar and rejects a ladder that fails it, The first bar was 0.15 and an i.i.d. draw from 16-32 met it only 5% of the time., px/wavelength = 512/p must stay at least 2x the measured 6.0 floor (D15), and ev, The decoupling strength scales with the SPREAD of p, not its magnitude. The firs, The 127 legacy three_gene samples store no params_json and no sim_seed, so their, Measured: ang_conc drifts 0.190 -> 0.074 across an 8x L span, enough to flip a, Probing at the same p re-runs an identical simulation and proves nothing. (+25 more)

### Community 80 - "Recovery Scoring (validate.py)"
Cohesion: 0.09
Nodes (34): _Key, _load_split(), test_morphology_scoring.py — tests for rngrn.scoring.morphology.  Morphology (sa, A constant intensity rescale must move NEITHER metric.      This is the property, The fixed order is a contract: the scales and banks are indexed by it positional, AnswerKey stand-in. Morphology does not read it, but score_recovery does., No frame at all -> no morphology keys, and an explicit reason. NOT a NaN., target_frame alone is FREE and must be scored; the COMPARISON keys must be absen (+26 more)

### Community 81 - "Cubic Dispersion Backend Tests"
Cohesion: 0.19
Nodes (12): The cubic dispersion backend must be EXACTLY equivalent to the eigvals reference, (J, D) pairs from the registered three_gene answer keys, or skip if absent., Random model states, not just the answer keys — covers wilder theta. Parametrize, The backend is used inside a differentiable loss, so gradients must agree too., Exact for N=3 only — must fail loud rather than silently return nonsense.      R, _real_jacobians(), _sigma_cubic(), _sigma_eig() (+4 more)

### Community 82 - "Robustness Baseline Figure (docs)"
Cohesion: 0.33
Nodes (12): RNGRN robustness baseline figure (3-panel), Node diffusibility ordering (slowest/middle/fastest as-generated) forced non-diffusible, Panel a: fraction still Turing vs multiplicative parameter noise, by topology, Panel b: within-topology robustness spread at 10% noise (0.23-1.00), Panel c: fraction of 127 systems still Turing when a node is forced non-diffusible, "Fraction still strictly Turing" robustness metric, Tica et al. — experimental parameter CV = 4.8% reference line, Cross-repression topology (+4 more)

### Community 83 - "Turing-First Homotopy Experiment"
Cohesion: 0.07
Nodes (34): _J3(), _J4(), _pattern(), _permute(), test_permutation_scoring.py — fast CPU tests for rngrn.scoring.permutation.  The, EXPERIMENT A (N=3, m=2): exactly ONE admissible relabelling, so alignment is a, Brute-force the objective independently and confirm the returned score is the ma, EXPERIMENT B: model N=3, truth 2x2. The 2x2 observed block is well-defined and (+26 more)

### Community 84 - "Ratio-Based Loss Weighting Tests"
Cohesion: 0.05
Nodes (37): 10. Init distribution — solved but deliberately unadopted, 11. The D-ratio prior — implemented, opt-in, with an unresolved tension, 12. Where the recorded knowledge lives, 13.1 Reproduction (real data, `recover()` path, 8 independent seeds), 13.2 Mechanism — the hypothesised veto collapse is REFUTED, 13.3 The fix, and why it is safe on by default, 13.4 Outcome, on the same samples and seeds as 13.1 (adam_steps=2000), 13.5 What this does NOT establish, and one measurement hazard (+29 more)

### Community 85 - "Determinism Tests"
Cohesion: 0.29
Nodes (9): _make_input(), test_determinism.py — recover() is bit-identical across two runs at the same see, Different train.seed -> different recovery. Same train.seed -> bit-identical., Unit B1: recover(seed=s) and recover(seed=s+1) must share NO restart init --, test_dispersion_backend_cubic_rejected_for_wrong_N(), test_model_seed_controls_init_independent_of_train_seed(), test_recover_bit_identical_same_seed(), test_restart_seeds_independent_across_run_seeds() (+1 more)

### Community 86 - "Robustness Cloud Tests"
Cohesion: 0.17
Nodes (11): test_robustness.py — synthetic-Jacobian tests for eval.analysis's robustness clo, robustness_volumes emits exactly the four run-index keys the spec asks for, and, A single sigma_log must produce the SAME relative perturbation regardless of a, Section 3.2: a lognormal factor is always positive, so a perturbed entry can, tr(J) < 0 is necessary but not sufficient for max Re eig(J) < 0. J = diag(5, -3,, The vectorized (n_samples, N, N) linear algebra in _perturb_cloud must give, test_batching_matches_serial(), test_perturbation_preserves_sign_and_structural_zeros() (+3 more)

### Community 87 - "Turing Hinge Split Tests"
Cohesion: 0.09
Nodes (32): block_perm(), block_repro(), block_restarts(), block_table(), fmt(), load_cells(), main(), pct() (+24 more)

### Community 88 - "Dataset Registry Tests"
Cohesion: 0.24
Nodes (9): _make_payload(), test_registry.py — dataset store + pluggable index backend.  Covers: register a, registry.py is ANSWER-KEY side: no recovery-side module may import it., A tiny HDF5 payload: per-sample final_frame + quarantined jacobian/x_star., The manual-drop workflow: put payload.h5 at <root>/<id>/ yourself, then scan., test_index_roundtrip(), test_register_list_load(), test_registry_not_imported_by_recovery_side() (+1 more)

### Community 89 - "Three Gene Classical Train Manifest (partial)"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 90 - "Three Gene Qvar Manifest (partial)"
Cohesion: 0.08
Nodes (25): GradNormWeighting, NTKWeighting, weighting.py — adaptive loss-weighting strategies (Stage 3 outer loop).  A weigh, 0 for the first `off_frac` of the budget, linear 0->1 over the next `ramp_frac`,, Base interface. `combine` returns (scalar_total, weights_used)., Linearly ramp selected weights from 0 to base over `warmup` steps. Lets the, GradNorm-style balancing: scale each term so per-term gradient magnitudes match., NTK-PINN weighting: set weights from the diagonal of the neural-tangent-kernel / (+17 more)

### Community 91 - "Two Gene Classical Train Manifest (partial)"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 92 - "Robustness Baseline Experiment (exp11)"
Cohesion: 0.31
Nodes (8): cloud(), main(), EXP11 — Robustness baseline of the GENERATOR systems, and Tica's immobile node., Deterministic seed from strings/numbers. NOT hash() -- Python salts string     h, (is_turing, kstar, sigma_max) under the STRICT criterion., Fraction of a log-normal cloud around (J, D) that is still Turing.      Perturbs, stable_seed(), turing_strict()

### Community 93 - "Independent Forward RD Solver"
Cohesion: 0.31
Nodes (8): _jac_at(), _laplacian_fft(), solver.py — independent forward RD solver (ANSWER-KEY SIDE).  Integrates a refer, Finite-difference Jacobian of the reaction at a single state x (answer-key side), Integrate the reference system from a perturbed homogeneous state to a     conve, Pointwise reaction over a grid. field: (N,H,W) -> (N,H,W).      The reference re, _reaction_grid(), simulate_to_attractor()

### Community 94 - "Firewall Import-Direction Tests"
Cohesion: 0.22
Nodes (9): _imports_of(), Scoring side. It may reuse the image-only measurement (observables.morphology re, This module scores against a LABEL supplied by the caller, not the answer key., The direction that actually matters: recovery must not reach scoring.      Mirro, scoring/__init__ deliberately imports no submodule, so one failing scorer cannot, test_morphology_scoring_does_not_import_the_recovery_engine(), test_morphology_scoring_reads_no_answer_key(), test_no_recovery_side_module_imports_the_morphology_scorer() (+1 more)

### Community 95 - "Morphology Separability Tests"
Cohesion: 0.25
Nodes (8): _loo_nearest_member(), Leave-one-out 1-NN accuracy in the z-scored space, using the module's own metric, PIN the measured separability so a regression is visible.      Measured 2026-07-, The stated reason for z-scoring: it buys the stripes class.      Unscaled distan, WHY the diagnostic is 2-D: radial averaging destroys the stripes anisotropy., test_2d_spectrum_separates_better_than_radial_average(), test_pinned_separability_of_the_four_stat_space(), test_zscoring_beats_unscaled_on_balanced_accuracy()

### Community 96 - "C1 Restarts Readout Script"
Cohesion: 0.43
Nodes (6): index_sample(), main(), pct(), c2_restarts.py — the SENSITIVE readout for unit C2.  `target_reports.jsonl` scor, sample_key per run_id, from the run index (train_results.json omits it)., run_rows()

### Community 97 - "C2 Restarts Readout Script"
Cohesion: 0.43
Nodes (6): index_sample(), main(), pct(), c2_restarts.py — the SENSITIVE readout for unit C2.  `target_reports.jsonl` scor, sample_key per run_id, from the run index (train_results.json omits it)., run_rows()

### Community 98 - "Cubic Dispersion Benchmark Experiment"
Cohesion: 0.38
Nodes (5): EXP10 - An exact, GPU-friendly dispersion for N=3. Validated, not assumed.  THE, max_i Re eig(J - k^2 diag(D)) for N=3, exactly, by closed-form cubic roots., sigma_cubic(), sigma_eig(), validate()

### Community 99 - "Model-Seed No-Op Regression Tests"
Cohesion: 0.29
Nodes (7): ModelConfig, model.seed pinned in a base config makes train.seed a no-op for everything below, test_base_config_does_not_pin_model_seed(), nondim must default to False so no existing caller silently switches methods., OPT-IN: adopting the fix changes recorded D numbers, so it must be a measured, test_d_init_from_kstar_defaults_to_off(), test_dimensional_path_is_the_default()

### Community 100 - "Dispersion Curve Plot Data"
Cohesion: 0.06
Nodes (35): 0. What the previous attempt at this unit actually left behind, 10.1 CRITERION 3.1's CONTROL — the substantive half, and it is NEGATIVE, 10.2 The other pre-registered criteria on this target, 10.3 What this changes about the unit's conclusion, 10. Cell `baseline` on `sample_0003` — and criterion 3.1's control, at last computable on the PRIMARY dataset, 1. No-op proof — every swept axis, on BOTH execution paths, 2. The criteria table — pre-registered per cell, BEFORE the cells were run, 3.1 `sample_0000` (K=8, `git_sha 570f3c8`, `config_id 30105ae5c671`) (+27 more)

### Community 101 - "Three Gene Classical Train Answer-Key Fields"
Cohesion: 0.48
Nodes (7): answer_key_keys, D, interaction_matrix, jacobian, kstar, x_star, answer_key_keys

### Community 102 - "Three Gene Qvar Answer-Key Fields"
Cohesion: 0.07
Nodes (32): _first_channel(), _leak_instrumentation(), _morphology_metrics(), validate.py — score a recovery against the quarantined answer key.  Scoring prio, Take the single 2-D field a morphology comparison operates on.      Accepts (H,, Morphology block. Returns a flat dict of scalars.      Split by COST, deliberate, The image-blind control for the k* numbers in this project — ONE PER REFERENCE., Grade a RecoveryResult against an AnswerKey. Returns a flat metric dict.      L (+24 more)

### Community 103 - "Two Gene Classical Train Answer-Key Fields"
Cohesion: 0.48
Nodes (7): answer_key_keys, D, interaction_matrix, jacobian, kstar, x_star, answer_key_keys

### Community 104 - "C1/C2 Jacobian Probe Script"
Cohesion: 0.47
Nodes (5): _draw_D(), main(), c2_jacobian_probe.py — is nc1's failure in J, or in D?  Every nc1 baseline run c, Scan D with J fixed. Returns (n_turing, best_sigma, bestD)., turing_capable()

### Community 105 - "C1 Queue Script (worker 1)"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue.sh script

### Community 106 - "C1 Queue Script (worker 2)"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue2.sh script

### Community 107 - "C1 Queue Script (worker 3)"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue3.sh script

### Community 108 - "C1 Queue Script (worker 4)"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue4.sh script

### Community 109 - "C1 Queue Script (worker 5)"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue5.sh script

### Community 110 - "C1/C2 Repro Script"
Cohesion: 0.53
Nodes (5): load(), main(), modal_fraction(), c2_repro.py — criterion 3.1 with its control and its sensitivity, from stored J., sign_key()

### Community 111 - "C2 Jacobian Probe Script"
Cohesion: 0.47
Nodes (5): _draw_D(), main(), c2_jacobian_probe.py — is nc1's failure in J, or in D?  Every nc1 baseline run c, Scan D with J fixed. Returns (n_turing, best_sigma, bestD)., turing_capable()

### Community 112 - "C2 Repro Script"
Cohesion: 0.53
Nodes (5): load(), main(), modal_fraction(), c2_repro.py — criterion 3.1 with its control and its sensitivity, from stored J., sign_key()

### Community 113 - "Loss Staging Factor"
Cohesion: 0.33
Nodes (5): 0 for the first `off_frac` of the budget, linear 0->1 over the next `ramp_frac`,, staging_factor(), exp05 line 87: w = 0 for it < 0.25*steps, then min(1, (it-0.25*steps)/(0.25*step, test_staging_factor_fails_loud_on_a_nonsense_schedule(), test_staging_factor_matches_the_exp05_schedule()

### Community 114 - "Dispersion Argmax-at-k0 Tests"
Cohesion: 0.09
Nodes (33): _bins(), test_gate_contract.py — the gate reads per-sample geometry and BOTH wavenumbers., Firewall spot-check at the new fields: neither k* may ride along on RecoveryInpu, The caller's L must not become a fallback for a file that has none., The FFT number is a diagnostic, not the headline — absent is legal, not fatal., The five file-backed configs must not carry an L that silently loses to the file, The linear k* expressed in FFT bins: k* = 2*pi*n/L  =>  n = k* L / 2pi., MEASURED FACT: every three_gene sample's linear k* is exactly 6.00 FFT bins. (+25 more)

### Community 115 - "C2 Cell Launcher Script"
Cohesion: 0.40
Nodes (4): KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c2_cell.sh script

### Community 116 - "C2 Summary Script"
Cohesion: 0.60
Nodes (4): fmt(), main(), c2_summary.py — read the target_reports table of one or more C2 runs-roots and p, rows()

### Community 117 - "Spectral Amplitude Experiment"
Cohesion: 0.50
Nodes (4): main(), parseval_channels(), EXP12 — Does the power-spectrum AMPLITUDE encode the concentration field?  QUEST, Return (mean_from_DC, var_from_power, mean, var) for one 2-D field.

### Community 118 - "Stage0 Prior Recovery Launcher"
Cohesion: 0.40
Nodes (4): KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, stage0_part4_prior_recovery.sh script

### Community 119 - "Comp-Tuning Cell Launcher"
Cohesion: 0.40
Nodes (4): KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, tune_comp_cell.sh script

### Community 120 - "Comp-Tuning Summary Script"
Cohesion: 0.60
Nodes (4): fmt(), load(), main(), tune_comp_summarise.py — collapse the tune_comp target-report JSONs into one tab

### Community 121 - "Loss Config & Staging Tests"
Cohesion: 0.11
Nodes (30): apply_overrides(), _build_section(), DataConfig, _deep_merge(), from_dict(), load_config(), config.py — typed, hierarchical, hashable config-as-experiment.  Every knob that, Post-training rollout / stiff integration knobs. (+22 more)

### Community 122 - "Model Device Property"
Cohesion: 0.29
Nodes (3): device, Device the parameters live on. Callers that create tensors to feed this, Device the parameters live on. Callers that create tensors to feed this

### Community 123 - "Cross-L Transfer Decisions (docs)"
Cohesion: 0.15
Nodes (13): three_gene_multiL dataset (23 systems x 4 domain sizes), three_gene_qvar_smoke dataset (3 samples, smoke test), D15: cross-L rollouts hold dx fixed, with a measured 6px/wavelength floor, D16: cross-L statistics — population CV of k*, origin-fixed q-vs-L slope, no pass threshold, D6: domain-size leak fixed — vary L with random periods-per-box, D-PLOT-3: cross-L table persisted through run-index machinery; tidy/long exports, First cross-L transfer measurement — 2026-07-30, Persisted artefacts (added 2026-07-30, unit P1) (+5 more)

### Community 124 - "No-Op Knob Check Script"
Cohesion: 0.67
Nodes (3): load(), main(), c2_noop_check.py — prove every swept knob actually changed something.  FOUR sile

### Community 125 - "Batched Steady-State Newton"
Cohesion: 0.08
Nodes (19): build_system(), _dispersion_max(), GiererMeinhardt, _jac_fd(), rd_models.py — reference reaction-diffusion systems (ANSWER-KEY SIDE).  These ar, Base: a reaction f(x), diffusion D, domain L, and an analytic answer key., The quarantined truth: x*, J, k*, sigma_max, D, coefficients. VALIDATION ONLY., Saturated Gierer-Meinhardt (stripes). f = a + u^n/(v(1+K u^n)) - b u ; g = u^n - (+11 more)

### Community 126 - "Dispersion/Turing-OK Consistency Test"
Cohesion: 0.50
Nodes (4): plotdata.sigma_of must be the SAME curve eval.analysis.turing_ok maxes over., A (J, D) pair that is genuinely Turing-unstable, so sigma(k) has a real peak., test_sigma_of_agrees_with_turing_ok(), _turing_JD()

### Community 131 - "Morphology Call Result Type"
Cohesion: 0.06
Nodes (31): 0. Verdict up front, 1.1 `RecoveryInput` — the complete list, 1.2 `L` is read from the sample, and the caller loses, 1.3 Per-loader split: what goes to which side, 1. The recovery input surface, 2.1 Library path — `recover.recover`, 2.2 The four loss terms, 2.3 Dead / inert knobs on the library path (+23 more)

### Community 137 - "Reference Bank Centroids"
Cohesion: 0.12
Nodes (30): balanced_picks(), build_batched(), build_model(), draw_batch(), evaluate_batch(), evaluate_draw(), harvest(), _inv_softplus() (+22 more)

### Community 139 - "Angular Feature Separability Test"
Cohesion: 0.08
Nodes (28): lift_check(), qss_reaction_np(), dynamical.py — the QSS -> dynamical lift and its mu->0 consistency check.  Train, Integrate lifted promoter gates to equilibrium at fixed x, form the production, eval/ — post-training analysis & validation (RECOVERY SIDE, reads only the model, _b3_fold_trap_model(), _b3_sweep(), test_science.py — cheap correctness anchors (seconds, CPU).  These check the sci (+20 more)

### Community 140 - "Held-Out Morphology Classification Test"
Cohesion: 0.10
Nodes (27): _J3(), test_reproducibility_scoring.py — fast CPU tests for rngrn.scoring.reproducibili, Shrinking the threshold to near-zero recovers the raw (near-machine-precision), `repro_J_vector` must round-trip to the ORIGINAL J, not its sign structure., The prereg §3.1 sweep must give DIFFERENT numbers on a threshold-sensitive     t, A matrix already valued in {-1, 0, 1} passes through unchanged — this is the, This module compares recovered J's to each other, never to a ground truth, so it, The overall scale of J is not pinned by the objective (documented elsewhere in (+19 more)

### Community 141 - "Train-Split Centroid Test"
Cohesion: 0.10
Nodes (22): Config, Canonical JSON (sorted keys) of the full config, for hashing., Identity of the EXPERIMENT ARM: this config with the seeds neutralised., _morphology_rollout(), Simulate the recovered model on the TARGET's geometry. Returns (field, row)., Within the pinned arm, train.seed still varies and must still group., It must not become so coarse that different science pools into one row., test_arm_id_is_constant_across_seeds_within_a_FREE_init_arm() (+14 more)

### Community 142 - "Centroid vs Sample Bank Test"
Cohesion: 0.10
Nodes (24): Staging is itself a static schedule, so this is exactly the inner strategy's., _batched_restarts(), _clip_grad_norm_per_member(), FreeScaleLatent, _kgrid_for(), recover.py — per-frame variational recovery (RECOVERY SIDE).  Optimise the RNGRN, Deterministic per-(model_seed, restart) init seed, independent of neighbouring, Clip each BATCH MEMBER's gradient norm independently, in place. Returns the (B,) (+16 more)

### Community 143 - "Classification Margin Test"
Cohesion: 0.13
Nodes (26): align_permutation(), AlignmentScore, _as_field_stack(), _as_square(), _candidate_perms(), _centred_or_raise(), _cosine_uncentered(), _fro_rel_err() (+18 more)

### Community 144 - "Within/Between Class Distance Test"
Cohesion: 0.11
Nodes (24): agreement(), _cell_key(), load_runs(), tune_comp_crosstarget.py — the CONTROL half of pre-registration 3.1.  3.1 requir, The tuning axes, read off the run's OWN frozen config, never off the command lin, cell/sample/seed -> J, for Turing-reaching runs under one runs-root., report(), _as_square() (+16 more)

### Community 145 - "Constant Field NaN-Guard Test"
Cohesion: 0.12
Nodes (25): candidate_table(), class_margin(), compute_stability(), draw_periods(), label_is_stable(), main(), multiL_labels(), native_periods() (+17 more)

### Community 146 - "Train-Scale Provenance Test"
Cohesion: 0.08
Nodes (25): 0. What `mu` is, and one useful fact about it, 1.1 The table, 1.2 Independent replication — and one statistical caveat about it, 1.2b The three things the part-1 table says, 1.3 The trace test overcounts by ~64× here — do not use `turing_ok`'s number, 1.4 The unscored directions, and what the rate is a rate *of*, 1. Is the box Turing-empty? — the acceptance rate, with its denominator, 2.1 The answer: the patterning window lies ENTIRELY INSIDE the box, on 12 of 12 systems (+17 more)

### Community 147 - "Recovery/Scoring Import Direction Test"
Cohesion: 0.11
Nodes (21): RNGRN — inverse-Turing pipeline package.  A per-frame variational pipeline that, _fit_one_seed(), Run recovery K times (one per seed in `seeds`) on ONE target and emit ONE     au, Run fit() for one (cfg, seed, run_id). Never raises: a seed that fails is record, run_target_report(), A tiny name -> implementation registry.  Used for the three varying concerns (mo, _best_restart(), fit() (+13 more)

### Community 149 - "test_benchmark_grouping.py"
Cohesion: 0.14
Nodes (24): reproducibility_markdown(), test_benchmark_grouping.py — cross-seed aggregation in optim.benchmark (D-EVID-1, THE DEFECT: three seeds of one arm must be ONE row with n_seeds == 3., n_seeds alone cannot be audited; the actual seeds must be on the row., Two TARGETS are two rows. Pooling them would be a worse defect than the one fixe, Different configs stay separate — the property config_id was protecting., Rows recorded before `arm_id` existed carry no seed-independent identity.      T, `reproducibility_table` has NO callers — which is exactly why it needs a test. (+16 more)

### Community 150 - "test_plausibility.py"
Cohesion: 0.09
Nodes (14): _fake_recovery_input(), test_plausibility.py — unit tests for rngrn.scoring.plausibility and rngrn.losse, One runaway alpha[i,j] makes the whole parameter implausible, not just that, A box loaded from an arbitrary, hand-built YAML with numbers that appear nowhere, A minimal (model, frame, kgrid, kstar_obs) tuple for exercising losses/total., An adaptive strategy sets a term's weight FROM its value, so the prior would be, An immobile third species (D -> 0) must NOT move the ratio at all, however     s, test_a_single_out_of_box_entry_fails_the_whole_parameter() (+6 more)

### Community 151 - "td_figures.py"
Cohesion: 0.12
Nodes (23): available_datasets(), corpus_summary_figure(), default_datasets_root(), detail_figure(), gallery_figure(), load_samples(), _panel_title(), patterning_verdict() (+15 more)

### Community 152 - "make_figures.py"
Cohesion: 0.15
Nodes (23): _decode(), fig_cross_L(), fig_dispersion(), fig_hyperparams(), fig_parameters(), fig_pattern(), fig_raps(), fig_training() (+15 more)

### Community 153 - "terms.py"
Cohesion: 0.11
Nodes (23): anticollapse(), composite_loss(), _damped_newton(), frame_scale_anchor(), morphology_consistency(), rngrn_objective.py ================== Composite objective for single-frame RNGRN, Re-solve one differentiable Newton polish step from a converged xstar so the ret, Turing hinges on DISJOINT k-support. The promoted default (see total.compute_ter (+15 more)

### Community 154 - "test_canon_generate.py"
Cohesion: 0.12
Nodes (21): Canonical dataset generation: cv tracing, the saturation gate, the payload contr, gate.from_registry RAISES if a sample lacks L or k_star -- it refuses to default, src/rngrn/ never reads `trajectory`; at 512 it would be 6x the storage for nothi, Which corpus sample this was promoted from, and which side of the split it is on, Real images give one observable, but storing all three keeps m=1 vs m=3 a     co, The real contract: gate.from_registry must be able to load what we wrote., A small, fast, known-Turing 3-gene system for integration tests., The trace must be a read-only observation of the same trajectory. (+13 more)

### Community 155 - "benchmark.py"
Cohesion: 0.16
Nodes (21): cmd_benchmark(), build_table(), _col_mean(), _dataset_of(), degradation_markdown(), degradation_table(), _group_key(), _isnum() (+13 more)

### Community 156 - "numerics.py"
Cohesion: 0.13
Nodes (21): _cached_half_coeffs(), _half_coeffs_cached(), integrate_bdf1_newton_krylov(), integrate_etdrk4(), integrate_etdrk4_rfft(), integrate_imex_split(), _phi_contour(), numerics.py — stiff time integrators for the lift-and-simulate rollout.  Three i (+13 more)

### Community 157 - "turing_labyrinth/manifest.json"
Cohesion: 0.10
Nodes (20): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+12 more)

### Community 158 - "turing_spots/manifest.json"
Cohesion: 0.10
Nodes (20): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+12 more)

### Community 159 - "cli.py"
Cohesion: 0.18
Nodes (20): build_parser(), cmd_analyze(), cmd_evaluate(), cmd_export(), cmd_generate_data(), cmd_list_datasets(), cmd_register_data(), cmd_scan_datasets() (+12 more)

### Community 160 - "Part 2 — Decisions"
Cohesion: 0.10
Nodes (20): D10 — `DEFAULT_SIGN_ZERO_RTOL = 0.05` and `topology_consistency := modal_fraction`, both flagged UNCALIBRATED, D11 — `gradnorm`/`ntk` loss-weighting strategies now RAISE rather than silently running fixed weights, D12 — Domain-size design settled by decorrelating L from k\* (see D6), D13 — Both new datasets left unsplit; whether `three_gene_multiL` should split by `system_id` — OPEN, D14 — Generator seeded from `abs(hash(topo))`: fixed going forward, permanent limitation for legacy data, D15 — cross-L rollouts hold **dx** fixed (n scales with L), with a MEASURED floor of 6 pixels per wavelength, D16 — the cross-L statistics: population CV of k\*, an origin-fixed q-vs-L slope, and NO pass threshold, D1 — D-ratio prior centred at 7.5 (literature Nodal/Lefty), not ~100 (generator median) (+12 more)

### Community 161 - "PREREGISTRATION — the pass conditions, fixed before the runs that are judged by them"
Cohesion: 0.10
Nodes (20): 0. What is being claimed, 1. Datasets, and which claims each may support, 1a. The `three_gene_qvar` split — declared 2026-07-29, before any convergence run, 1b. The canonical `turing_*` split — declared 2026-08-10, before the data existed, 2. Seeds, 3.1 Reproducibility — the primary criterion, 3.2 Robustness, 3.3 Same pattern (+12 more)

### Community 162 - "lgen.py"
Cohesion: 0.14
Nodes (18): d_ratio(), _group_key(), lgen_consistency(), lgen_table(), _log10_spread(), _mean_of_finite(), modal_sign_agreement(), per_run_lgen_metrics() (+10 more)

### Community 163 - "cache.py"
Cohesion: 0.17
Nodes (14): _checksum(), _dataset_dir(), exists(), generate(), load_frame(), cache.py — the content-addressed dataset cache (Stage 1 core).  data/cache/<data, Generate the dataset for `spec` if absent; return its directory. Idempotent., Load ONLY (frame, L). Answer key is deliberately NOT returned here — see gate.py (+6 more)

### Community 164 - "RecoveryInput"
Cohesion: 0.21
Nodes (18): AnswerKey, from_3gene_hdf5(), from_cache(), from_registry(), _observe(), gate.py — THE FIREWALL GATE.  This is the single doorway between the answer-key, Read a required scalar sample attribute, or RAISE.      House style is fail-loud, The domain size is a property of the SAMPLE, not of the caller's config.      Co (+10 more)

### Community 165 - "CLAUDE.md — working conventions for this repository"
Cohesion: 0.11
Nodes (17): 10. How far to run before checking in, 11. Subagents and dynamic workflows, 12. `.claude/` — the executable form of this file, 1. Environment, 2. Git, 3. Testing, 4. House style, 5. The data firewall (+9 more)

### Community 166 - "turing_ok"
Cohesion: 0.16
Nodes (17): _draw_JD_cloud(), linear_stability(), _model_JD(), _perturb_cloud(), analysis.py — linear stability, Turing conditions, and the robustness cloud.  Re, Draw n_samples independent log-normal multiplicative perturbations of the     PH, Vectorized log-normal cloud around a single PHYSICAL (J, D).      Ported from sc, (J, D) at the model's own steady state, as numpy. Solves for x* once if xstar is (+9 more)

### Community 167 - "plot_topology"
Cohesion: 0.16
Nodes (17): _edge(), _label(), _layout(), _lighten(), plot_topology(), rngrn_topology.py ================ Flexible "Learned GRN Topology" plot for an R, Draw an auto-regulation self-loop just outside the node, bulging along `outdir`., Render the learned RNGRN topology.      model          : an RNGRN instance (read (+9 more)

### Community 168 - "1. Finite-`mu` robustness — undoing the quasi-steady-state reduction"
Cohesion: 0.13
Nodes (15): Quasi-steady-state (QSS) reaction reduction, 1. Finite-`mu` robustness — undoing the quasi-steady-state reduction, Future work — deliberately deferred, with the reason and the starting point, The one structural fact worth keeping, The question, Traps, recorded so they are not rediscovered, What already exists to start from, What would make it worth taking up (+7 more)

### Community 169 - "test_determinism.py"
Cohesion: 0.15
Nodes (16): ModelConfig, _make_input(), test_determinism.py — recover() is bit-identical across two runs at the same see, Different train.seed -> different recovery. Same train.seed -> bit-identical., Unit B1: recover(seed=s) and recover(seed=s+1) must share NO restart init --, model.seed pinned in a base config makes train.seed a no-op for everything below, test_base_config_does_not_pin_model_seed(), test_dispersion_backend_cubic_rejected_for_wrong_N() (+8 more)

### Community 170 - "open_index"
Cohesion: 0.20
Nodes (16): export_all(), _melt(), _num(), export.py — the run record as TIDY (long) CSV, ready for pandas / seaborn / ggpl, One row per (run_id, step, member, variable) over every run whose npz has a hist, Write every tidy table under `out_dir`. Returns {filename: n_data_rows}., The numeric reading of a value, or '' when it has none. Bools count as 0/1., One dict per (row, non-identifier column). The whole of tidy, in six lines. (+8 more)

### Community 171 - "C2 — tuning the NON-COMPETITIVE (`nc1`) form against the pre-registered criteria"
Cohesion: 0.12
Nodes (16): 0. The cells, 1.1 Criterion 3.1 fails on BOTH halves (`scripts/c2_repro.py`, same runs), 1.2 THE DIAGNOSIS: nc1's failure is TWO failures, and only one of them is D, 1.3 The near-immobile third node is not an artefact — it is where the box D lives, 1.4 The sensitive readout, and why the scored count needs one, 1. The baseline, and what it says the problem is, 2. A CORRECTION TO THE RECORD, made before anything is built on it, 4. No-op checks, per axis (+8 more)

### Community 172 - "Measuring robustness — what exists, what is broken, and what the data says"
Cohesion: 0.12
Nodes (16): 1. The reference quantity, 2. What exists in this repo, 3.1 It perturbs raw θ, so "10 % noise" is not 10 % on anything physical, 3.2 The multiplicative factor is applied to a value that can be negative, 3.3 It silently drops `dispersion_backend`, 3.4 It is serial and slow, 3.5 A fifth issue, in the criterion rather than the cloud, 3. Four defects that make its current output non-comparable (+8 more)

### Community 173 - "TrainingHistory"
Cohesion: 0.13
Nodes (10): Accumulates (step, member) records and renders them as dense numpy arrays., The thinning rule: stride `every`, plus the endpoints unconditionally., Freeze the scalar column order on the first record; fail loud if it changes., A member whose steady state diverged and was abandoned at `step`., Dense arrays for `plotdata.save_plot_arrays`, or {} when nothing was recorded., History provenance for the npz `meta` blob., TrainingHistory, test_empty_history_renders_nothing() (+2 more)

### Community 174 - "total.py"
Cohesion: 0.15
Nodes (14): losses/ — the composite objective (RECOVERY SIDE).  terms.py     : differentiabl, compute_terms(), compute_terms_batched(), parts_member(), total.py — assemble the composite objective from named, source-tagged terms.  Ea, Batched twin of `compute_terms`. Returns (term_vals, parts, converged).      `mo, Raised when the model-derived steady state fails to converge (fail-loud)., Batched twin of `total_loss`. Returns (loss_vec (B,), parts, converged (B,)). (+6 more)

### Community 175 - "_StubModel"
Cohesion: 0.16
Nodes (12): param_prior(), Soft biological-plausibility prior on the model's OWN parameters. Recovery-side, Minimal stand-in exposing exactly the properties param_prior reads (torch     te, Perturbing D away from the centre and taking a small gradient step on theta_D, A third, near-immobile species (very small D) must leave the prior identical to, beta has no bounds in the test box (UNCITED) and param_prior does not even read, _StubModel, test_param_prior_box_hinge_is_zero_inside_and_positive_outside() (+4 more)

### Community 176 - "observables.py"
Cohesion: 0.16
Nodes (15): _angular(), classify(), kstar_of(), laplacian_5point(), laplacian_spectral(), laplacian_torch(), morphology(), raps() (+7 more)

### Community 177 - "test_nondim_units.py"
Cohesion: 0.20
Nodes (13): _fixture(), _Key, _NondimResult, test_nondim_units.py — the physical-D contract on the non-dimensional path (D-EV, THE DEFECT. A nondim result must give the same turing_volume_* as the physical o, `D_observed_max` is a reported diagnostic; on the nondim path it was 1/L**2 off., A RecoveryResult as `recover(nondim=True)` returns one: model.D is D/L**2,     D, The rollout integrates D lap(x) + f(x); D must be the PHYSICAL one at physical L (+5 more)

### Community 178 - "test_plot_arrays.py"
Cohesion: 0.17
Nodes (13): _lgen_out(), test_plot_arrays.py — the plottable-array artefact, the trajectory recorder, the, House rule: index rows are FLAT SCALARS. The morph vector becomes one column eac, `plotdata` accepts the ANSWER KEY's (J, D) and `export` reads the scored indexes, The recorder runs INSIDE the training loop, so it must import nothing truth-side, _seed_index(), test_export_all_writes_the_csv_files(), test_history_is_recovery_side_safe() (+5 more)

### Community 179 - "results_of"
Cohesion: 0.20
Nodes (15): -> [(seed, train_results dict)] for the runs this report names., results_of(), block_invariant(), block_ladder(), block_threshold(), block_verify(), canon_key(), invariants() (+7 more)

### Community 180 - "Part 1 — Evidence-integrity defects (read this before trusting any pre-2026-07-29 number)"
Cohesion: 0.13
Nodes (15): D-EVID-10 — ledger sweep: which recorded rows the 2026-08-04 review affects, D-EVID-11 — `turing_ok` tested `tr(J) < 0`, so uniformly UNSTABLE systems were scored Turing-unstable, D-EVID-12 — the pre-registered sign-threshold sweep was a no-op, and reported the wrong threshold's answer under the requested threshold's label, D-EVID-13 — every cross-seed aggregation was dead: `build_table` grouped on `config_id`, which hashes the seed, D-EVID-14 — three physics consumers read the DIMENSIONLESS `model.D` on the non-dimensional path, D-EVID-15 — what the adversarial re-review of D-EVID-7..14 found, including a regression those fixes introduced, D-EVID-16 — branch consolidation: `model.init` was a silent no-op, and the ledger sweep is redone at 154 rows, D-EVID-1 — `cli.py -o` silently kept only the last override (+7 more)

### Community 181 - "test_experiment_arms.py"
Cohesion: 0.21
Nodes (12): _Key, test_experiment_arms.py — scoring ROUTES correctly per experiment arm.  The unit, Minimal RecoveryResult stand-in: score_recovery needs model, xstar, kstar_model., THE original bug: 3x3 recovered vs 2x2 true used to give a silent NaN., The one cross-arm-comparable column must always be present., fit() derives `arm` from n_true vs N and the hidden set; check the same logic he, _Result, test_arm_classification_matches_shapes() (+4 more)

### Community 182 - "test_experiment_arms.py"
Cohesion: 0.21
Nodes (12): _Key, test_experiment_arms.py — scoring ROUTES correctly per experiment arm.  The unit, Minimal RecoveryResult stand-in: score_recovery needs model, xstar, kstar_model., THE original bug: 3x3 recovered vs 2x2 true used to give a silent NaN., The one cross-arm-comparable column must always be present., fit() derives `arm` from n_true vs N and the hidden set; check the same logic he, _Result, test_arm_classification_matches_shapes() (+4 more)

### Community 183 - "_Key"
Cohesion: 0.17
Nodes (13): _Key, AnswerKey stand-in. Morphology does not read it, but score_recovery does., No frame at all -> no morphology keys, and an explicit reason. NOT a NaN., target_frame alone is FREE and must be scored; the COMPARISON keys must be absen, Morphology depends only on the fields, so the no_true_J arm must still get it., (m, H, W) frames are accepted; channel 0 is scored, matching recovery's k* chann, A grid mismatch between target and model field must raise, not score as 'far'., _Result (+5 more)

### Community 184 - "`data/datasets/` — the registered datasets, and how training data is generated"
Cohesion: 0.14
Nodes (13): Firewall gate (quarantines answer-key HDF5 fields: jacobian/x_star/D/interaction_matrix), registry_example.yaml (example: load a registered dataset), 1. Which datasets to use, 2. First-time setup on a new machine or worktree, 3. Regenerating the canonical training datasets, 4. If you GENERATE data, harvest it before the worktree dies, 5. Morphology: stored label vs measured, 6. Caveats that bite (+5 more)

### Community 185 - "data/registry.py"
Cohesion: 0.21
Nodes (13): build_manifest(), _checksum(), dataset_dir(), list_datasets(), load_manifest(), payload_path(), registry.py — the dataset store (ANSWER-KEY side).  A place to KEEP datasets you, Inspect an HDF5 payload IN PLACE and build its manifest dict (metadata only). (+5 more)

### Community 186 - "history.py"
Cohesion: 0.18
Nodes (10): _d_slice(), _param_block(), _param_vector(), history.py — the TRAINING TRAJECTORY of a recovery, recorded at a thinned cadenc, The D entries of a flattened parameter vector (D is last in PARAM_SPEC)., The scalar columns of one member: diagnostics, per-term losses, live weights., One serial restart's state at `step`. Call BEFORE `opt.step()` so the recorded, All B members' state at `step`. `alive` (a (B,) bool tensor/array) skips members (+2 more)

### Community 187 - "index.py"
Cohesion: 0.18
Nodes (7): _coerce(), JsonlIndex, _now_iso(), index.py — append-only metadata indices with two interchangeable backends.  Both, Reduce a value to something SQLite can store; dicts/lists -> JSON text., Python-side filter: `where` is a predicate callable row->bool (SQL string, _sql_type()

### Community 188 - "_softplus_hinge"
Cohesion: 0.16
Nodes (14): anticollapse_batched(), frame_scale_anchor_batched(), kstar_anchor_batched(), _np(), Zero when the condition holds with margin; smooth positive as it is approached/v, Batched SUPERSEDED shared-support hinges (control arm). Returns ((B,), parts)., Batched disjoint-support Turing hinges (the promoted default). Returns ((B,), pa, Batched log-scale anchor. xstar (B,N), obs_scale a SCALAR frame statistic shared (+6 more)

### Community 189 - "canon_generate.py"
Cohesion: 0.21
Nodes (13): build_dataset(), is_saturated(), main(), params_from_sample(), probe_label(), Write the canonical registry payload: final frame, cv trace, full provenance., Pool worker: re-simulate one selected system at the canonical resolution.      H, Re-simulate every system in one dataset spec at `grid` and write its payload. (+5 more)

### Community 190 - "plausibility_report"
Cohesion: 0.19
Nodes (10): BoxRow, _in_box(), plausibility_report(), plausibility.py — biological viability of a RECOVERED model's OWN parameters.  W, True/False for a cited, bounded row; None ("not applicable") for an UNCITED or, Score a RECOVERED model's own parameters against configs/bio_box.yaml.      alph, One parsed row of bio_box.yaml. `low`/`high` are None for an unbounded or     UN, d_ratio_of() (+2 more)

### Community 191 - "test_dispersion_cubic.py"
Cohesion: 0.19
Nodes (12): The cubic dispersion backend must be EXACTLY equivalent to the eigvals reference, (J, D) pairs from the registered three_gene answer keys, or skip if absent., Random model states, not just the answer keys — covers wilder theta. Parametrize, The backend is used inside a differentiable loss, so gradients must agree too., Exact for N=3 only — must fail loud rather than silently return nonsense.      R, _real_jacobians(), _sigma_cubic(), _sigma_eig() (+4 more)

### Community 192 - "Canonical Turing datasets"
Cohesion: 0.17
Nodes (12): 0. Status: THIS IS THE TRAINING DATA SOURCE, 10. What these datasets do NOT establish, 1. What they are, in one paragraph, 2. Why re-simulate instead of screening fresh systems, 3. How the five per class were chosen, 4. Resolution, and why 512 rather than 1024, 5. The leak, and why periods-per-box is drawn, 6. Why there is no `turing_stripes` (+4 more)

### Community 193 - "Code reality map — what is wired, what is a stub, what the docs get wrong"
Cohesion: 0.17
Nodes (12): 10. The two conventions most likely to be violated by accident, 11. Open decisions currently blocking progress, 1. The single most important structural fact, 2. Stubs and inert code, 3. Where the answer key is and is not allowed, 4. Sample data layout, 5. Run index and scoring plumbing — two traps, 6. Sweep axes: arm vs tuning (+4 more)

### Community 194 - "Part 2b — Stage 0: the biological-viability precondition (unit c-bioviab, 2026-08-03)"
Cohesion: 0.17
Nodes (12): D-BIO-1 — `beta` for the box-viability sampler is the GENERATOR's own basal draw, and is never scored, D-BIO-2 — the D vector is sampled under TWO paired schemes, because the near-immobile node is an INTENDED escape valve, D-BIO-3 — the Turing verdict is the STRICT test, on a LOG k-grid to 10^3.5, D-BIO-4 — the box-constrained sampler is BATCHED, and pinned against the serial reference, D-BIO-5 — the mu sweep varies the GEOMETRIC MEAN of delta, and separately one species at a time, D-BIO-6 — the rollout domain is L = 6 × 2π/k* on a 64×64 grid, D-BIO-7 — the binding parameters (KA/KR) are the FREE directions and are sampled over four decades, D-BIO-8 — part 3 shows the channel with the LARGEST spatial std, and records BOTH pattern verdicts (+4 more)

### Community 195 - "Sources consulted"
Cohesion: 0.17
Nodes (12): D-C1-DIAG — `max diag(J) > 0` adopted as a secondary readout, not as a criterion, D-C1-GAUGE — the J-degeneracy of the objective, and what it does and does not explain, D-C1-TURINGW — `loss.weights.turing` promoted to the lead rate axis, D-CANON-1 — two canonical datasets at 512×512, selected by margin and label stability, D-CANON-2 — `stripes` is not a stable pattern class in this generator; it is largely a small-box artefact, D-CANON-3 — periods-per-box is a geometric ladder over {8..40}, checked against a leak bar, D-CANON-4 — the canonical sets become the training data source, D-CANON-5 — canonical morphology is MEASURED from the field; `turing_labyrinth` is a mixed class (+4 more)

### Community 196 - "fit"
Cohesion: 0.21
Nodes (11): diagnose(), fit(), low_basal_init(), EXP03 - "Turing-first" (homotopy) recovery vs the current random-restart scheme., Disjoint-support hinges (exp02): uniform stability from k=0, instability from, FIREWALL-SAFE prior. Low basal + sub-saturated binding is what admits a positive, turing_hinges_split(), kstar_anchor() (+3 more)

### Community 197 - "stage0_figures.py"
Cohesion: 0.36
Nodes (11): fig_box_acceptance(), fig_mu_boundary(), fig_mu_per_species(), fig_part4(), fig_part4_dispersion(), fig_pattern_gallery(), main(), The recovered model's own D vector, from the tracked results JSON. (+3 more)

### Community 198 - "io.py"
Cohesion: 0.18
Nodes (9): append_run_index(), atomic_write_json(), load_checkpoint(), io.py — run directories, atomic writes, checkpoints, the run index.  Run layout, Write JSON to a temp file and rename on success (never a half-written file)., Each producer writes its OWN result file (train_results / analysis_results / ..., Append one run row to the run index. backend in {'jsonl','sqlite'} (index.py)., read_run_index() (+1 more)

### Community 199 - "test_robustness.py"
Cohesion: 0.17
Nodes (11): test_robustness.py — synthetic-Jacobian tests for eval.analysis's robustness clo, robustness_volumes emits exactly the four run-index keys the spec asks for, and, A single sigma_log must produce the SAME relative perturbation regardless of a, Section 3.2: a lognormal factor is always positive, so a perturbed entry can, tr(J) < 0 is necessary but not sufficient for max Re eig(J) < 0. J = diag(5, -3,, The vectorized (n_samples, N, N) linear algebra in _perturb_cloud must give, test_batching_matches_serial(), test_perturbation_preserves_sign_and_structural_zeros() (+3 more)

### Community 200 - "stage0_figures.py"
Cohesion: 0.36
Nodes (11): fig_box_acceptance(), fig_mu_boundary(), fig_mu_per_species(), fig_part4(), fig_part4_dispersion(), fig_pattern_gallery(), main(), The recovered model's own D vector, from the tracked results JSON. (+3 more)

### Community 201 - "TUNING.md — what Claude Code tunes, implements, and validates on CUDA"
Cohesion: 0.17
Nodes (11): Cross-cutting, L-generalisation (unit 12 — `recover.py::recover(nondim=...)`, `scoring/lgen.py`), Morphology rollout (unit 7), Promoted from the experiments (unit 1), Stage 1 — data generation & storage, Stage 2 — the objective (losses/), Stage 3 — optimisation (inner + outer), Stage 4 — validation / milestones (validate.py, train.fit) (+3 more)

### Community 202 - "BIO_VIABILITY.md"
Cohesion: 0.36
Nodes (8): Jacobian gauge symmetries (transpose, diagonal similarity, permutation) that sigma(k) cannot see, Near-immobile third node mechanism (Tica node C analogue), trivial_kstar_fft_err — the honest image-blind k* control, three_gene_qvar dataset (34 systems, periods-per-box ~ U{3..14}), D-EVID-7: k* leak control measured against the wrong (linear) reference, D-EVID-8: the pre-registered 8.3% k* bar is half an FFT bin, dataset-dependent, model.dispersion — sigma(k) = max Re eig(J - k^2 diag(D)), scoring.plausibility.d_ratio_of — largest/second-largest diffusivity ratio

### Community 203 - "Local dataset setup (one-time, per machine)"
Cohesion: 0.18
Nodes (9): three_gene train/val/test legacy family (127 samples), two_gene_classical dataset family, 1. Download from the Claude Science project, 2. Unpack and drop in, 3. Index them, 4. Use from a config, Disk, Layout compatibility (+1 more)

### Community 204 - "L-decoupled and multi-L datasets"
Cohesion: 0.18
Nodes (11): 1. What was wrong with the existing data, 2. What was generated, 3. Is the leak dead? — the headline number, 4. Storage and auditability, 5. What these sets are FOR, 6. What is NOT done, A reproducibility defect found and fixed, L-decoupled and multi-L datasets (+3 more)

### Community 205 - "exp02_objective_fix.py"
Cohesion: 0.20
Nodes (9): diagnose(), EXP01 — Can optimisation REACH Turing instability from a non-Turing init?  The u, run(), EXP02 - Fix the self-defeating turing_hinges objective.  DIAGNOSIS (measured, ex, Disjoint-support Turing hinges. kgrid[0] must be the ~0 mode., run(), turing_hinges_split(), SUPERSEDED shared-support hinges. Kept because docs and scripts/exp02 reference (+1 more)

### Community 206 - "_runlog.py"
Cohesion: 0.20
Nodes (10): args_config_id(), _runlog.py — shared auditability helper for the experiment scripts.  The library, sha256 of the canonical (sorted-key) JSON of the script's resolved argument, Mint a run_id + config_id + provenance block for one script invocation.      Cal, Write the run's identity/provenance block to a sidecar JSON next to the     scri, Append one flat-scalar row to the SAME run index the library uses     (<runs_roo, record(), start() (+2 more)

### Community 207 - "rngrn-pipeline"
Cohesion: 0.18
Nodes (11): Handoff: the target goal and the state of the science, Identifiability experiments, Install & run, Layout (layered by rate-of-change), rngrn-pipeline, Sandbox note, Tests: run them locally, before you push, The one non-negotiable rule — the firewall (+3 more)

### Community 208 - "_fake_table"
Cohesion: 0.27
Nodes (11): _all_stable(), _fake_table(), 8 synthetic systems: 4 per canonical class, decreasing margin, all gate-clean., Guards the fixture itself: if PREVIOUSLY_RUN changes, these tests must be update, test_each_dataset_gets_distinct_periods(), test_selection_fails_loud_when_a_class_is_short(), test_selection_is_deterministic(), test_selection_returns_the_requested_count_per_class() (+3 more)

### Community 209 - "Handoff — identifiability experiments branch"
Cohesion: 0.20
Nodes (10): Bugs found during integration (already fixed — don't reintroduce), Handoff — identifiability experiments branch, Read these before changing anything, Run it, Scoring API (all scoring-side; never imported by recovery), Status in one line, Update 2026-07-26 — first real-data dry run, What changed on this branch (+2 more)

### Community 210 - "HANDOFF — visual inspection of the training data"
Cohesion: 0.20
Nodes (9): 1. Why this branch exists, 2. What was added, 3. What is VERIFIED, 3a. The claim in §3 that is CIRCULAR, and must not be cited as evidence, 4. The one real finding, 5. What is NOT verified — read this before citing anything here, 6. Firewall position, 7. Where to start if you pick this up (+1 more)

### Community 212 - "run_sweep"
Cohesion: 0.27
Nodes (8): optim/ — the outer loops around fit() (Stages 3 & 6).  sweep     : enumerate con, _as_literal(), _check_axis(), grid_points(), sweep.py — the outer optimisation loop, built OUTWARD from fit().  A sweep enume, axes: {dotted_key: [values]} -> list of override dicts (cartesian product)., Run fit() over the cartesian product of `axes` x `seeds`. Returns metric rows., run_sweep()

### Community 213 - "test_registry.py"
Cohesion: 0.24
Nodes (9): _make_payload(), test_registry.py — dataset store + pluggable index backend.  Covers: register a, registry.py is ANSWER-KEY side: no recovery-side module may import it., A tiny HDF5 payload: per-sample final_frame + quarantined jacobian/x_star., The manual-drop workflow: put payload.h5 at <root>/<id>/ yourself, then scan., test_index_roundtrip(), test_register_list_load(), test_registry_not_imported_by_recovery_side() (+1 more)

### Community 214 - "HANDOFF — the canonical Turing training datasets"
Cohesion: 0.22
Nodes (8): 1. What exists now, 2. What is VERIFIED, 3. What is NOT verified — read before citing anything, 4. The three findings that changed the plan, 5. Where to start, 6. Reproducing and testing, 7. Known follow-ups, none blocking, HANDOFF — the canonical Turing training datasets

### Community 215 - "three_gene_classical_train/manifest.json"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 216 - "three_gene_multiL/manifest.json"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 217 - "three_gene_train/manifest.json"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 218 - "two_gene_classical_train/manifest.json"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 219 - "exp11_robustness_baseline.py"
Cohesion: 0.31
Nodes (8): cloud(), main(), EXP11 — Robustness baseline of the GENERATOR systems, and Tica's immobile node., Deterministic seed from strings/numbers. NOT hash() -- Python salts string     h, (is_turing, kstar, sigma_max) under the STRICT criterion., Fraction of a log-normal cloud around (J, D) that is still Turing.      Perturbs, stable_seed(), turing_strict()

### Community 220 - "simulate_to_attractor"
Cohesion: 0.31
Nodes (8): _jac_at(), _laplacian_fft(), solver.py — independent forward RD solver (ANSWER-KEY SIDE).  Integrates a refer, Finite-difference Jacobian of the reaction at a single state x (answer-key side), Integrate the reference system from a perturbed homogeneous state to a     conve, Pointwise reaction over a grid. field: (N,H,W) -> (N,H,W).      The reference re, _reaction_grid(), simulate_to_attractor()

### Community 221 - "_FakeModel"
Cohesion: 0.25
Nodes (7): param_names(), Column names of one member's parameter trace, e.g. 'KA[0,1]', 'delta[2]'., _FakeModel, _parts(), Duck-typed RNGRN: only the six constrained-parameter properties are read., test_history_column_set_is_frozen_fail_loud(), test_serial_history_renders_dense_arrays_with_nan_for_a_dead_member()

### Community 222 - "_imports_of"
Cohesion: 0.22
Nodes (9): _imports_of(), Scoring side. It may reuse the image-only measurement (observables.morphology re, This module scores against a LABEL supplied by the caller, not the answer key., The direction that actually matters: recovery must not reach scoring.      Mirro, scoring/__init__ deliberately imports no submodule, so one failing scorer cannot, test_morphology_scoring_does_not_import_the_recovery_engine(), test_morphology_scoring_reads_no_answer_key(), test_no_recovery_side_module_imports_the_morphology_scorer() (+1 more)

### Community 223 - "RuntimeError"
Cohesion: 0.28
Nodes (8): RuntimeError, collect(), figure(), main(), is_resimulatable(), Can this sample's simulation be re-run from what the payload stores?      Only t, Re-run the reaction-diffusion simulation behind one training sample.      Return, resimulate()

### Community 224 - "_FakeModel"
Cohesion: 0.25
Nodes (7): param_names(), Column names of one member's parameter trace, e.g. 'KA[0,1]', 'delta[2]'., _FakeModel, _parts(), Duck-typed RNGRN: only the six constrained-parameter properties are read., test_history_column_set_is_frozen_fail_loud(), test_serial_history_renders_dense_arrays_with_nan_for_a_dead_member()

### Community 225 - "repo-mapper.md"
Cohesion: 0.25
Nodes (7): Check staleness every time, and say so, Constraints, Grepping the graph directly is often faster, Landmarks in this repo, Query the graph first, Report, Then verify against the source. Always.

### Community 226 - "Pre-merge checklist"
Cohesion: 0.25
Nodes (7): Data, Evidence, Firewall, Handoff, Merge damage (the class the suite cannot see), Pre-merge checklist, Tests

### Community 227 - "Running trainers in rngrn-pipeline"
Cohesion: 0.25
Nodes (7): 1. Every trainer invocation goes through the guard, 2. `ps`, `pgrep` and `pkill` are BLIND inside the sandbox, 3. Never delete or truncate a file a run might be writing, 4. Where runs go, 5. What a short CPU run is worth, 6. Environment, Running trainers in rngrn-pipeline

### Community 228 - "_loo_nearest_member"
Cohesion: 0.25
Nodes (8): _loo_nearest_member(), Leave-one-out 1-NN accuracy in the z-scored space, using the module's own metric, PIN the measured separability so a regression is visible.      Measured 2026-07-, The stated reason for z-scoring: it buys the stripes class.      Unscaled distan, WHY the diagnostic is 2-D: radial averaging destroys the stripes anisotropy., test_2d_spectrum_separates_better_than_radial_average(), test_pinned_separability_of_the_four_stat_space(), test_zscoring_beats_unscaled_on_balanced_accuracy()

### Community 229 - "`.claude/` — agent configuration for rngrn-pipeline"
Cohesion: 0.29
Nodes (6): Agents, `.claude/` — agent configuration for rngrn-pipeline, Layout, Settings, Skills, The hook

### Community 230 - "Datasets and payloads"
Cohesion: 0.29
Nodes (6): Before removing ANY worktree, Datasets and payloads, If your unit GENERATES a dataset, Provisioning a worktree, Recovering from a branch checkout that ate the manifests, The canonical store

### Community 231 - "Starting new work in rngrn-pipeline"
Cohesion: 0.29
Nodes (6): 1. Create it, 2. Each worktree needs its OWN venv — verify it, 3. A new worktree LOOKS provisioned with datasets but is not, 4. Launching a parallel wave, 5. Finishing, Starting new work in rngrn-pipeline

### Community 232 - "Recording a science decision"
Cohesion: 0.29
Nodes (6): Before you write the entry, Every science decision must, Marking something UNCALIBRATED, Recording a science decision, The entry format, Two things go to the owner instead

### Community 233 - "dtype"
Cohesion: 0.29
Nodes (3): dtype, Parameter dtype (float64 by default). The dispersion eigendecomposition and, Parameter dtype (float64 by default). The dispersion eigendecomposition and

### Community 234 - "c1_restarts.py"
Cohesion: 0.43
Nodes (6): index_sample(), main(), pct(), c2_restarts.py — the SENSITIVE readout for unit C2.  `target_reports.jsonl` scor, sample_key per run_id, from the run index (train_results.json omits it)., run_rows()

### Community 235 - "c2_restarts.py"
Cohesion: 0.43
Nodes (6): index_sample(), main(), pct(), c2_restarts.py — the SENSITIVE readout for unit C2.  `target_reports.jsonl` scor, sample_key per run_id, from the run index (train_results.json omits it)., run_rows()

### Community 236 - "exp10_cubic_dispersion.py"
Cohesion: 0.38
Nodes (5): EXP10 - An exact, GPU-friendly dispersion for N=3. Validated, not assumed.  THE, max_i Re eig(J - k^2 diag(D)) for N=3, exactly, by closed-form cubic roots., sigma_cubic(), sigma_eig(), validate()

### Community 237 - "param_prior_batched"
Cohesion: 0.29
Nodes (7): _load_box_bounds(), param_prior_batched(), Read configs/bio_box.yaml -> {name: (low, high) or None}. None means the row is, Batched twin of `param_prior`. Returns ((B,), parts of (B,) numpy arrays)., losses/terms.py cannot import rngrn.scoring (firewall — see the note atop     pa, test_param_prior_batched_equals_the_serial_prior_member_by_member(), test_terms_and_scoring_box_loaders_agree_on_the_real_box()

### Community 238 - "_arrays"
Cohesion: 0.38
Nodes (7): _arrays(), _field(), The guard that matters: a decayed-noise field must never be saved as a pattern., test_answer_key_curve_absent_without_an_answer_key(), test_model_field_on_a_different_grid_raises(), test_npz_round_trips(), test_unpatterned_rollout_gets_no_model_field()

### Community 239 - "_arrays"
Cohesion: 0.38
Nodes (7): _arrays(), _field(), The guard that matters: a decayed-noise field must never be saved as a pattern., test_answer_key_curve_absent_without_an_answer_key(), test_model_field_on_a_different_grid_raises(), test_npz_round_trips(), test_unpatterned_rollout_gets_no_model_field()

### Community 240 - "c1_jacobian_probe.py"
Cohesion: 0.47
Nodes (5): _draw_D(), main(), c2_jacobian_probe.py — is nc1's failure in J, or in D?  Every nc1 baseline run c, Scan D with J fixed. Returns (n_turing, best_sigma, bestD)., turing_capable()

### Community 241 - "c1_queue2.sh"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue2.sh script

### Community 242 - "c1_queue3.sh"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue3.sh script

### Community 243 - "c1_queue4.sh"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue4.sh script

### Community 244 - "c1_queue5.sh"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue5.sh script

### Community 245 - "c1_queue.sh"
Cohesion: 0.40
Nodes (5): job(), KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c1_queue.sh script

### Community 246 - "c1_repro.py"
Cohesion: 0.53
Nodes (5): load(), main(), modal_fraction(), c2_repro.py — criterion 3.1 with its control and its sensitivity, from stored J., sign_key()

### Community 247 - "c2_jacobian_probe.py"
Cohesion: 0.47
Nodes (5): _draw_D(), main(), c2_jacobian_probe.py — is nc1's failure in J, or in D?  Every nc1 baseline run c, Scan D with J fixed. Returns (n_turing, best_sigma, bestD)., turing_capable()

### Community 248 - "c2_repro.py"
Cohesion: 0.53
Nodes (5): load(), main(), modal_fraction(), c2_repro.py — criterion 3.1 with its control and its sensitivity, from stored J., sign_key()

### Community 249 - "tune_comp_noop_check.py"
Cohesion: 0.40
Nodes (5): tune_comp_noop_check.py — prove every swept knob actually CHANGES something.  Fo, The things a live knob must be able to move., run(), _set(), signature()

### Community 250 - "exp05_pixel_minibatch.py"
Cohesion: 0.47
Nodes (5): fit(), EXP05 - Pixel minibatching: "tons of small batches taken from the same image"., Per-pixel PDE residual on a RANDOM SUBSET of pixels (Endres' L_PDE, batched)., residual_minibatch(), split_hinges()

### Community 251 - "Any"
Cohesion: 0.40
Nodes (5): Any, _latent_diagnostics(), Spatial-contrast diagnostics for the spare species' inferred latent field., _latent_diagnostics(), Spatial-contrast diagnostics for the spare species' inferred latent field.

### Community 252 - "firewall-auditor.md"
Cohesion: 0.40
Nodes (4): Report, Verify, don't assert, What the firewall is, Your job: find what the AST audit structurally cannot

### Community 253 - "Reporting a number"
Cohesion: 0.40
Nodes (4): Before you write the number down, Reporting a number, Verify, don't assert, Wording

### Community 254 - "three_gene_val dataset (registry dataset_id)"
Cohesion: 0.80
Nodes (5): expA_control_full.yaml (Experiment A control, N=3 fully observed), expA_hidden_channel.yaml (Experiment A, hidden third gene), m3_registry.yaml (Milestone 3, registry dataset, N=3 fully observed), nc1_m3_registry.yaml (Milestone 3 registry, nc1 form), three_gene_val dataset (registry dataset_id)

### Community 255 - "c2_cell.sh"
Cohesion: 0.40
Nodes (4): KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, c2_cell.sh script

### Community 256 - "c2_summary.py"
Cohesion: 0.60
Nodes (4): fmt(), main(), c2_summary.py — read the target_reports table of one or more C2 runs-roots and p, rows()

### Community 257 - "parseval_channels"
Cohesion: 0.50
Nodes (4): main(), parseval_channels(), EXP12 — Does the power-spectrum AMPLITUDE encode the concentration field?  QUEST, Return (mean_from_DC, var_from_power, mean, var) for one 2-D field.

### Community 258 - "stage0_part4_prior_recovery.sh"
Cohesion: 0.40
Nodes (4): KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, stage0_part4_prior_recovery.sh script

### Community 259 - "tune_comp_cell.sh"
Cohesion: 0.40
Nodes (4): KMP_AFFINITY, OMP_NUM_THREADS, PYTHONPATH, tune_comp_cell.sh script

### Community 260 - "tune_comp_summarise.py"
Cohesion: 0.60
Nodes (4): fmt(), load(), main(), tune_comp_summarise.py — collapse the tune_comp target-report JSONs into one tab

### Community 262 - "exp02_objective_fix.py"
Cohesion: 0.40
Nodes (4): EXP02 - Fix the self-defeating turing_hinges objective.  DIAGNOSIS (measured, ex, Disjoint-support Turing hinges. kgrid[0] must be the ~0 mode., run(), turing_hinges_split()

### Community 264 - "evidence-auditor.md"
Cohesion: 0.50
Nodes (3): Method, Report, The rules you enforce

### Community 265 - "merge-damage-hunter.md"
Cohesion: 0.50
Nodes (3): Find it mechanically, not by eye, Judging what you find, Report

### Community 266 - "numerics-reviewer.md"
Cohesion: 0.50
Nodes (3): Constraints, The ground truth of this codebase, What to look for

### Community 267 - "c2_noop_check.py"
Cohesion: 0.67
Nodes (3): load(), main(), c2_noop_check.py — prove every swept knob actually changed something.  FOUR sile

### Community 268 - "_turing_JD"
Cohesion: 0.50
Nodes (4): plotdata.sigma_of must be the SAME curve eval.analysis.turing_ok maxes over., A (J, D) pair that is genuinely Turing-unstable, so sigma(k) has a real peak., test_sigma_of_agrees_with_turing_ok(), _turing_JD()

### Community 269 - "expB_control_matched.yaml (Experiment B control, N=2 matched)"
Cohesion: 1.00
Nodes (3): expB_control_matched.yaml (Experiment B control, N=2 matched), expB_overparam.yaml (Experiment B, over-parameterisation N=3 model on N=2 truth), two_gene_classical_val dataset (registry dataset_id)

## Knowledge Gaps
- **676 isolated node(s):** `$schema`, `Bash(git status:*)`, `Bash(git diff:*)`, `Bash(git log:*)`, `Bash(git show:*)` (+671 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **51 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RNGRN` connect `Batched RNGRN Model` to `Repo Conventions & Review Agents`, `exp02_objective_fix.py`, `Cross-L Evaluation Tests`, `Batched Recovery Tests`, `L-Generalisation Recovery Tests`, `Turing Hinges & Steady-State Solve`, `Bio-Viability Stage0 Script`, `Cross-Cutting Science Concepts (docs)`, `Rollout Simulation Tests`, `Steady-State Newton Solver`, `Plausibility Scoring Tests`, `1. Finite-`mu` robustness — undoing the quasi-steady-state reduction`, `Science Correctness Anchors`, `Answer Key Gate`, `Losses Unit Tests`, `test_experiment_arms.py`, `_Key`, `C2 Init-Bias Script`, `RNGRN Model Definition`, `BIO_VIABILITY.md`, `Run Metadata & Checkpoint I/O`, `Cubic Dispersion Backend Tests`, `Determinism Tests`, `dtype`, `exp05_pixel_minibatch.py`, `Model Device Property`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `RNGRN` connect `RNGRN Model Core (dispersion/jacobian)` to `Repo Conventions & Review Agents`, `Reference Bank Centroids`, `Angular Feature Separability Test`, `Train-Split Centroid Test`, `Centroid vs Sample Bank Test`, `Batched RNGRN Model`, `test_plausibility.py`, `Steady-State Newton Solver`, `CLI Commands`, `test_determinism.py`, `Config Identity & Hashing`, `_StubModel`, `test_experiment_arms.py`, `Cross-L Residual & Frame Tests`, `test_dispersion_cubic.py`, `Composite Loss Module`, `RNGRN Model Definition`, `fit`, `io.py`, `Sweep Runner & No-Op Decisions`, `Run Metadata & Checkpoint I/O`, `exp02_objective_fix.py`, `Recovery Scoring (validate.py)`, `dtype`, `param_prior_batched`, `Model Device Property`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `RNGRN — differentiable reaction-diffusion GRN recovery model` connect `Repo Conventions & Review Agents` to `exp02_objective_fix.py`, `Cross-L Evaluation Tests`, `Reference Bank Centroids`, `Batched Recovery Tests`, `Angular Feature Separability Test`, `L-Generalisation Recovery Tests`, `Turing Hinges & Steady-State Solve`, `Bio-Viability Stage0 Script`, `Rollout Simulation Tests`, `CLI Commands`, `Science Correctness Anchors`, `Config Identity & Hashing`, `Losses Unit Tests`, `test_experiment_arms.py`, `test_experiment_arms.py`, `Cross-L Residual & Frame Tests`, `C2 Init-Bias Script`, `test_dispersion_cubic.py`, `Composite Loss Module`, `fit`, `Sweep Runner & No-Op Decisions`, `exp02_objective_fix.py`, `Cubic Dispersion Backend Tests`, `exp05_pixel_minibatch.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `RNGRN` (e.g. with `Quasi-steady-state (QSS) reaction reduction` and `init_jacobian()`) actually correct?**
  _`RNGRN` has 60 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `Bash(git status:*)`, `Bash(git diff:*)` to the rest of the system?**
  _676 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Three Gene MultiL Dataset Samples` be split into smaller, more focused modules?**
  _Cohesion score 0.04209563029055136 - nodes in this community are weakly interconnected._
- **Should `Three Gene Train Dataset Samples` be split into smaller, more focused modules?**
  _Cohesion score 0.043945068664169785 - nodes in this community are weakly interconnected._