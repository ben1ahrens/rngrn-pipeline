# Graph Report - .  (2026-08-06)

## Corpus Check
- Large corpus: 548 files · ~536,360 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2699 nodes · 4309 edges · 149 communities (128 shown, 21 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 426 edges (avg confidence: 0.76)
- Token cost: 0 input · 636,717 output

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

## God Nodes (most connected - your core abstractions)
1. `sample_keys` - 93 edges
2. `sample_keys` - 89 edges
3. `RNGRN` - 89 edges
4. `sample_keys` - 59 edges
5. `sample_keys` - 53 edges
6. `score_recovery()` - 39 edges
7. `sample_keys` - 35 edges
8. `CLAUDE.md — repository working-conventions contract` - 32 edges
9. `docs/DECISIONS.md D-EVID-4 (cited external doc)` - 31 edges
10. `BatchedRNGRN` - 29 edges

## Surprising Connections (you probably didn't know these)
- `L-encodes-k*-true domain-size leak` --conceptually_related_to--> `RecoveryInput`  [INFERRED]
  docs/DATA_INTO_MODEL.md → src/rngrn/data/gate.py
- `Strict (max Re eig(J)<0) vs loose (tr(J)<0) Turing criterion` --implements--> `turing_ok()`  [INFERRED]
  docs/ROBUSTNESS_MEASUREMENT.md → src/rngrn/eval/analysis.py
- `Split-support Turing hinges (disjoint k-support for stability/instability terms)` --implements--> `turing_hinges_split()`  [INFERRED]
  docs/STATE_OF_THE_SCIENCE.md → src/rngrn/losses/terms.py
- `Frame-scale anchor (firewall-legal x* proxy from image mean)` --implements--> `frame_scale_anchor()`  [INFERRED]
  docs/STATE_OF_THE_SCIENCE.md → src/rngrn/losses/terms.py
- `D-ratio soft prior (biological plausibility, centred 7.5)` --implements--> `param_prior()`  [INFERRED]
  docs/STATE_OF_THE_SCIENCE.md → src/rngrn/losses/terms.py

## Import Cycles
- None detected.

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

## Communities (149 total, 21 thin omitted)

### Community 0 - "Three Gene MultiL Dataset Samples"
Cohesion: 0.02
Nodes (93): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+85 more)

### Community 1 - "Three Gene Train Dataset Samples"
Cohesion: 0.02
Nodes (89): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+81 more)

### Community 2 - "Overparam Scoring Tests"
Cohesion: 0.06
Nodes (54): Any, _as_index_tuple(), _as_square(), _latent_diagnostics(), observed_block_agreement(), overparam_report(), ndarray, overparam.py — over-parameterisation metrics for EXPERIMENT B (SCAFFOLD).  STATU (+46 more)

### Community 3 - "Three Gene Classical Train Samples"
Cohesion: 0.03
Nodes (59): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+51 more)

### Community 4 - "Target Report Aggregation Tests"
Cohesion: 0.08
Nodes (53): aggregate_target_report(), _finite(), _mean_median(), _pattern_block(), _per_seed_audit(), ndarray, target_report.py — the target-report driver: the measurement instrument for the, Re-hydrate one run's Jacobian for the reproducibility aggregation, at the REQUES (+45 more)

### Community 5 - "Repo Conventions & Review Agents"
Cohesion: 0.08
Nodes (50): No discretised Laplacian in training — analytic in Fourier space, evidence-auditor agent spec, firewall-auditor agent spec, merge-damage-hunter agent spec, numerics-reviewer agent spec, CLAUDE.md — repository working-conventions contract, .claude/README.md — agent configuration overview, Pre-merge checklist rule (+42 more)

### Community 6 - "Two Gene Classical Train Samples"
Cohesion: 0.04
Nodes (53): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+45 more)

### Community 7 - "Figure Generation Scripts"
Cohesion: 0.06
Nodes (49): _decode(), fig_cross_L(), fig_dispersion(), fig_hyperparams(), fig_parameters(), fig_pattern(), fig_raps(), fig_training() (+41 more)

### Community 8 - "Cross-L Evaluation Tests"
Cohesion: 0.08
Nodes (39): evaluate_across_L(), physical_model_from_checkpoint(), Return (model_with_PHYSICAL_D, L_train) for a checkpoint loaded by io.load_check, Simulate ONE recovered model on several domain sizes and measure L-generalisatio, fixture_across_L(), _nondim_twin(), test_lgen_eval.py — the cross-domain-size instrument, with a POSITIVE and a NEGA, An unknown rule must fail loud, and the rule in force must be in the output. (+31 more)

### Community 9 - "Batched Recovery Tests"
Cohesion: 0.09
Nodes (37): _frame(), _kgrid(), _pair(), The BATCHED recovery path must be numerically equivalent to the SERIAL reference, The standing model.py contract: J must stay differentiable w.r.t. theta, so the, Every member's x* and its converged flag must match the serial damped Newton., Not merely 'agrees with serial' — f(x*) must be at the Newton tolerance., The fail-loud contract becomes a PER-MEMBER FLAG: a member whose steady-state so (+29 more)

### Community 10 - "L-Generalisation Recovery Tests"
Cohesion: 0.06
Nodes (32): test_lgen.py — L-generalisation: non-dimensional recovery + the cross-L metric (, The old floor was an ABSOLUTE 2.0 rad/length, so once kstar_obs < 0.25 the grid, The floor's only legitimate job is to keep the grid non-degenerate when kstar_ob, theta_D = randn(N)*0.5 - 2*log(kstar_obs) => D starts at median 1/kstar_obs**2., low_basal has its own D-ratio scheme (unit 2); kstar_obs must not perturb it., Two seeds at each of two L. Seeds already disagree on one entry at a fixed L, so, A system seen at ONE L cannot demonstrate cross-L consistency and must not be ab, A dimensional run and a non-dimensional run of the same system are different (+24 more)

### Community 11 - "Morphology Scoring Tests"
Cohesion: 0.09
Nodes (34): _Key, _load_split(), test_morphology_scoring.py — tests for rngrn.scoring.morphology.  Morphology (sa, A constant intensity rescale must move NEITHER metric.      This is the property, The fixed order is a contract: the scales and banks are indexed by it positional, AnswerKey stand-in. Morphology does not read it, but score_recovery does., No frame at all -> no morphology keys, and an explicit reason. NOT a NaN., target_frame alone is FREE and must be scored; the COMPARISON keys must be absen (+26 more)

### Community 12 - "Permutation Scoring Tests"
Cohesion: 0.07
Nodes (34): _J3(), _J4(), _pattern(), _permute(), test_permutation_scoring.py — fast CPU tests for rngrn.scoring.permutation.  The, EXPERIMENT A (N=3, m=2): exactly ONE admissible relabelling, so alignment is a, Brute-force the objective independently and confirm the returned score is the ma, EXPERIMENT B: model N=3, truth 2x2. The 2x2 observed block is well-defined and (+26 more)

### Community 13 - "QSS Reduction & Robustness Analysis"
Cohesion: 0.08
Nodes (33): Quasi-steady-state (QSS) reaction reduction, D-EVID-14: three physics consumers read dimensionless model.D on the non-dimensional path, FUTURE_WORK.md — deliberately deferred work, _draw_JD_cloud(), _model_JD(), _perturb_cloud(), analysis.py — linear stability, Turing conditions, and the robustness cloud.  Re, Draw n_samples independent log-normal multiplicative perturbations of the     PH (+25 more)

### Community 14 - "Three Gene Test Dataset Manifest"
Cohesion: 0.06
Nodes (35): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+27 more)

### Community 15 - "Gate Contract Tests"
Cohesion: 0.08
Nodes (35): _bins(), test_gate_contract.py — the gate reads per-sample geometry and BOTH wavenumbers., Firewall spot-check at the new fields: neither k* may ride along on RecoveryInpu, The caller's L must not become a fallback for a file that has none., The FFT number is a diagnostic, not the headline — absent is legal, not fatal., The five file-backed configs must not carry an L that silently loses to the file, The linear k* expressed in FFT bins: k* = 2*pi*n/L  =>  n = k* L / 2pi., MEASURED FACT: every three_gene sample's linear k* is exactly 6.00 FFT bins. (+27 more)

### Community 16 - "Three Gene Qvar Dataset Samples"
Cohesion: 0.06
Nodes (35): sample_0000, sample_0001, sample_0002, sample_0003, sample_0004, sample_0005, sample_0006, sample_0007 (+27 more)

### Community 17 - "Three Gene Val Dataset Manifest"
Cohesion: 0.06
Nodes (34): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+26 more)

### Community 18 - "Dataset Families & Identifiability Docs"
Cohesion: 0.10
Nodes (34): three_gene train/val/test legacy family (127 samples), two_gene_classical dataset family, D-EVID-13: cross-seed aggregation was dead — build_table grouped on config_id (hashes seed), HANDOFF_identifiability.md — engineering state handoff, IDENTIFIABILITY_EXPERIMENTS.md — Experiments A and B, LOCAL_DATA_SETUP.md — one-time per-machine dataset setup, align_permutation(), AlignmentScore (+26 more)

### Community 19 - "Experiment Arm Scoring Tests"
Cohesion: 0.08
Nodes (31): Grade a RecoveryResult against an AnswerKey. Returns a flat metric dict.      L, score_recovery(), _Key, test_experiment_arms.py — scoring ROUTES correctly per experiment arm.  The unit, Minimal RecoveryResult stand-in: score_recovery needs model, xstar, kstar_model., THE original bug: 3x3 recovered vs 2x2 true used to give a silent NaN., The one cross-arm-comparable column must always be present., fit() derives `arm` from n_true vs N and the hidden set; check the same logic he (+23 more)

### Community 20 - "Turing Hinges & Steady-State Solve"
Cohesion: 0.09
Nodes (32): Disjoint-support Turing hinges. kgrid[0] must be the ~0 mode., turing_hinges_split(), anticollapse(), anticollapse_batched(), _damped_newton(), frame_scale_anchor_batched(), kstar_anchor_batched(), _load_box_bounds() (+24 more)

### Community 21 - "Three Gene Classical Test Manifest"
Cohesion: 0.06
Nodes (31): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+23 more)

### Community 22 - "Batched RNGRN Model"
Cohesion: 0.08
Nodes (11): dtype, BatchedRNGRN, Gene-regulatory reaction model with learnable, biologically-meaningful params., Parameter dtype (float64 by default). The dispersion eigendecomposition and, B INDEPENDENT RNGRN members carried as one leading batch dimension.      WHY THI, One member per entry of `seeds`, in order.          Takes the seeds EXPLICITLY r, Member b as a standalone serial RNGRN (parameters DETACHED and copied)., RNGRN (+3 more)

### Community 23 - "C1 Analysis Script"
Cohesion: 0.12
Nodes (30): block_perm(), block_repro(), block_restarts(), block_table(), fmt(), load_cells(), main(), pct() (+22 more)

### Community 24 - "Bio-Viability Stage0 Script"
Cohesion: 0.12
Nodes (30): balanced_picks(), build_batched(), build_model(), draw_batch(), evaluate_batch(), evaluate_draw(), harvest(), _inv_softplus() (+22 more)

### Community 25 - "Reference RD Solver Models"
Cohesion: 0.08
Nodes (18): _dispersion_max(), GiererMeinhardt, _jac_fd(), rd_models.py — reference reaction-diffusion systems (ANSWER-KEY SIDE).  These ar, Base: a reaction f(x), diffusion D, domain L, and an analytic answer key., The quarantined truth: x*, J, k*, sigma_max, D, coefficients. VALIDATION ONLY., Saturated Gierer-Meinhardt (stripes). f = a + u^n/(v(1+K u^n)) - b u ; g = u^n -, Schnakenberg (spots). f = gamma(a - u + u^2 v) ; g = gamma(b - u^2 v). (+10 more)

### Community 26 - "Benchmark Grouping Tests"
Cohesion: 0.10
Nodes (30): reproducibility_markdown(), test_benchmark_grouping.py — cross-seed aggregation in optim.benchmark (D-EVID-1, THE DEFECT: three seeds of one arm must be ONE row with n_seeds == 3., n_seeds alone cannot be audited; the actual seeds must be on the row., Two TARGETS are two rows. Pooling them would be a worse defect than the one fixe, Different configs stay separate — the property config_id was protecting., Rows recorded before `arm_id` existed carry no seed-independent identity.      T, `reproducibility_table` has NO callers — which is exactly why it needs a test. (+22 more)

### Community 27 - "Reproducibility Scoring Tests"
Cohesion: 0.10
Nodes (27): _J3(), test_reproducibility_scoring.py — fast CPU tests for rngrn.scoring.reproducibili, Shrinking the threshold to near-zero recovers the raw (near-machine-precision), `repro_J_vector` must round-trip to the ORIGINAL J, not its sign structure., The prereg §3.1 sweep must give DIFFERENT numbers on a threshold-sensitive     t, A matrix already valued in {-1, 0, 1} passes through unchanged — this is the, This module compares recovered J's to each other, never to a ground truth, so it, The overall scale of J is not pinned by the objective (documented elsewhere in (+19 more)

### Community 28 - "Cross-Cutting Science Concepts (docs)"
Cohesion: 0.12
Nodes (29): Jacobian gauge symmetries (transpose, diagonal similarity, permutation) that sigma(k) cannot see, L-encodes-k*-true domain-size leak, Near-immobile third node mechanism (Tica node C analogue), trivial_kstar_fft_err — the honest image-blind k* control, three_gene_multiL dataset (23 systems x 4 domain sizes), three_gene_qvar dataset (34 systems, periods-per-box ~ U{3..14}), three_gene_qvar_smoke dataset (3 samples, smoke test), BIO_VIABILITY.md — is the biological box Turing-empty? (+21 more)

### Community 29 - "Two Gene Classical Test Manifest"
Cohesion: 0.07
Nodes (28): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+20 more)

### Community 30 - "Three-Gene Dataset Generator"
Cohesion: 0.12
Nodes (28): anisotropy(), audit_leak(), classify(), expand_multiL(), expand_qvar(), feasible_periods(), jac_fd(), main() (+20 more)

### Community 31 - "Rollout Simulation Tests"
Cohesion: 0.10
Nodes (28): Integrate d x/dt = D lap(x) + f(x) from x* + noise. Returns a result dict., simulate(), _fixture_is_turing(), test_rollout.py — the lift-and-simulate rollout: correctness, cost, and invarian, Halving dt must not move the recovered k* or the morphology feature vector., Same physical box, 64x64 vs 96x96: same k* and the same morphology features., A random-init RNGRN is linearly STABLE (sig_max < 0 for all 800 inits measured)., A run cut short by max_steps must say so, so it cannot be read as converged. (+20 more)

### Community 32 - "Cross-Target Tuning Control Arm"
Cohesion: 0.11
Nodes (26): D-EVID-12: sign-threshold sweep was a no-op that mislabeled its own rtol, agreement(), _cell_key(), load_runs(), tune_comp_crosstarget.py — the CONTROL half of pre-registration 3.1.  3.1 requir, The tuning axes, read off the run's OWN frozen config, never off the command lin, cell/sample/seed -> J, for Turing-reaching runs under one runs-root., report() (+18 more)

### Community 33 - "Claude Settings & Hooks"
Cohesion: 0.07
Nodes (26): hooks, PreToolUse, permissions, allow, ask, deny, $schema, Bash(df -h:*) (+18 more)

### Community 34 - "Three Gene Classical Val Manifest"
Cohesion: 0.07
Nodes (26): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+18 more)

### Community 35 - "Steady-State Newton Solver"
Cohesion: 0.11
Nodes (24): RuntimeError, Solve f(x*) = 0 for the model's OWN homogeneous steady state.      Damped Newton, steady_state(), Raised when the model-derived steady state fails to converge (fail-loud)., SteadyStateError, _batched_restarts(), _clip_grad_norm_per_member(), FreeScaleLatent (+16 more)

### Community 36 - "Dataset Cache & Loading"
Cohesion: 0.13
Nodes (22): _checksum(), _dataset_dir(), exists(), generate(), load_frame(), ndarray, cache.py — the content-addressed dataset cache (Stage 1 core).  data/cache/<data, Generate the dataset for `spec` if absent; return its directory. Idempotent. (+14 more)

### Community 37 - "Plausibility Scoring Tests"
Cohesion: 0.08
Nodes (16): _fake_recovery_input(), test_plausibility.py — unit tests for rngrn.scoring.plausibility and rngrn.losse, One runaway alpha[i,j] makes the whole parameter implausible, not just that, A box loaded from an arbitrary, hand-built YAML with numbers that appear nowhere, losses/terms.py cannot import rngrn.scoring (firewall — see the note atop     pa, A minimal (model, frame, kgrid, kstar_obs) tuple for exercising losses/total., An adaptive strategy sets a term's weight FROM its value, so the prior would be, An immobile third species (D -> 0) must NOT move the ratio at all, however     s (+8 more)

### Community 38 - "Firewall Gate & Experiment Configs"
Cohesion: 0.14
Nodes (26): Firewall gate (quarantines answer-key HDF5 fields: jacobian/x_star/D/interaction_matrix), model.seed override making train.seed a no-op (D-EVID-4 finding), nc1 (non-competitive) recovery form, model.form override, base.yaml (shared config defaults), expA_control_full.yaml (Experiment A control, N=3 fully observed), expA_hidden_channel.yaml (Experiment A, hidden third gene), expB_control_matched.yaml (Experiment B control, N=2 matched), expB_overparam.yaml (Experiment B, over-parameterisation N=3 model on N=2 truth) (+18 more)

### Community 39 - "Two Gene Classical Val Manifest"
Cohesion: 0.08
Nodes (25): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+17 more)

### Community 40 - "CLI Commands"
Cohesion: 0.14
Nodes (25): build_parser(), cmd_analyze(), cmd_evaluate(), cmd_export(), cmd_generate_data(), cmd_list_datasets(), cmd_register_data(), cmd_scan_datasets() (+17 more)

### Community 41 - "Science Correctness Anchors"
Cohesion: 0.09
Nodes (24): _b3_fold_trap_model(), _b3_sweep(), test_science.py — cheap correctness anchors (seconds, CPU).  These check the sci, The counterexample. tr(J) < 0 but max Re eig(J) > 0 — not Turing, not anything., RE-MEASURED 2026-08-04 after the turing_ok correction (docs/DECISIONS.md D-EVID-, Walk a deterministic ensemble ONCE (the failing solves are the expensive ones) a, THE REGRESSION PIN. terms.steady_state's multistart is a pure rescue path: its f, Companion to the pin above: on the same ensemble the multistart solve converges (+16 more)

### Community 42 - "Optim Benchmark Command"
Cohesion: 0.15
Nodes (21): cmd_benchmark(), RNGRN — inverse-Turing pipeline package.  A per-frame variational pipeline that, build_table(), _col_mean(), _dataset_of(), degradation_markdown(), degradation_table(), _group_key() (+13 more)

### Community 43 - "JSONL Run Index"
Cohesion: 0.14
Nodes (9): _coerce(), JsonlIndex, _now_iso(), index.py — append-only metadata indices with two interchangeable backends.  Both, `where` is a SQL predicate string (no leading WHERE), e.g.         "recovered_tu, Reduce a value to something SQLite can store; dicts/lists -> JSON text., Python-side filter: `where` is a predicate callable row->bool (SQL string, _sql_type() (+1 more)

### Community 44 - "Config Loading & Override Tests"
Cohesion: 0.16
Nodes (22): apply_overrides(), load_config(), Load a YAML config with optional hierarchical composition via a top-level     `_, Apply dotted-key CLI overrides ('train.adam_lr=0.02'). Returns a new Config., End-to-end on a REAL registered sample: metric -> runs.jsonl -> both tables., End-to-end on a REAL registered sample: the leak control reaches runs.jsonl., test_both_kstar_errors_reach_run_index_and_degradation_table(), test_trivial_kstar_err_reaches_run_index_on_real_dataset() (+14 more)

### Community 45 - "Answer Key Gate"
Cohesion: 0.17
Nodes (21): AnswerKey, from_3gene_hdf5(), from_cache(), from_registry(), _observe(), gate.py — THE FIREWALL GATE.  This is the single doorway between the answer-key, Read a required scalar sample attribute, or RAISE.      House style is fail-loud, The domain size is a property of the SAMPLE, not of the caller's config.      Co (+13 more)

### Community 46 - "L-Generalisation Eval Module"
Cohesion: 0.10
Nodes (22): _dedup_L(), grid_for_L(), lgen_eval.py — DOES THE RECOVERED GRN PATTERN THE SAME WAY ON A DIFFERENT-SIZED, Grid size at domain size `L` that holds dx = L/n fixed at its L_train value., The L to evaluate at: the caller's, with L_train prepended if it is not among th, One per-L row from a `rollout.simulate` result. No ground truth is read., Cross-L summary statistics of the rows produced by `evaluate_across_L`.      Sep, _row_for() (+14 more)

### Community 47 - "Config Identity & Hashing"
Cohesion: 0.14
Nodes (16): Config, Canonical JSON (sorted keys) of the full config, for hashing., Identity of the EXPERIMENT ARM: this config with the seeds neutralised., _morphology_rollout(), Simulate the recovered model on the TARGET's geometry. Returns (field, row)., _AK, _pattern_like(), REGRESSION. A stable model's field decays to float noise; classifying that noise (+8 more)

### Community 48 - "Numerics: ETDRK4 Rollout Integrator"
Cohesion: 0.13
Nodes (21): _cached_half_coeffs(), _half_coeffs_cached(), integrate_bdf1_newton_krylov(), integrate_etdrk4(), integrate_etdrk4_rfft(), integrate_imex_split(), _phi_contour(), numerics.py — stiff time integrators for the lift-and-simulate rollout.  Three i (+13 more)

### Community 49 - "Training History Recorder"
Cohesion: 0.11
Nodes (16): param_names(), Accumulates (step, member) records and renders them as dense numpy arrays., The thinning rule: stride `every`, plus the endpoints unconditionally., A member whose steady state diverged and was abandoned at `step`., Dense arrays for `plotdata.save_plot_arrays`, or {} when nothing was recorded., History provenance for the npz `meta` blob., Column names of one member's parameter trace, e.g. 'KA[0,1]', 'delta[2]'., TrainingHistory (+8 more)

### Community 50 - "Losses Unit Tests"
Cohesion: 0.16
Nodes (19): _flat_grad(), _model_at_steady_state(), test_losses.py — the promoted objective terms (unit 1).  Covers the three terms, Design zero: x*_i == obs_scale for every i., A random-init model plus its own converged steady state. Skips inits whose Newto, The library differentiates through x*; exp05 does not. Whichever is right, the k, A skipped term must be ABSENT, not reported as 0.0 — otherwise a run log says th, The defect that forces recover() to refuse hidden-channel runs at resid weight 0 (+11 more)

### Community 51 - "RNGRN Model Core (dispersion/jacobian)"
Cohesion: 0.13
Nodes (12): f(x) for a batch of state vectors. x: (..., N) -> (..., N). Pointwise: no grid., J_ij = d f_i / d x_j at xstar (shape (N,)). Form-agnostic (autodiff of the, sigma(k) = max_i Re eig(J - k^2 diag(D)) over a grid of k. General-N,         di, Closed-form 2-species sigma(k). SELF-TEST reference; smooth sqrt clamp keeps a, max_i Re eig(M) for BATCHED 3x3 M, exactly, via closed-form cubic roots.      Wh, The pointwise reaction f(x), as a PURE function of the CONSTRAINED parameters., f(x) per member. x: (B, N) -> (B, N). ONE state vector per member.          Deli, J[b]_ij = d f_i / d x_j for member b at xstar[b]. (B,N) -> (B,N,N).          AUT (+4 more)

### Community 52 - "Plot Arrays Artefact Tests"
Cohesion: 0.16
Nodes (18): _arrays(), _field(), _lgen_out(), test_plot_arrays.py — the plottable-array artefact, the trajectory recorder, the, House rule: index rows are FLAT SCALARS. The morph vector becomes one column eac, The recorder runs INSIDE the training loop, so it must import nothing truth-side, The guard that matters: a decayed-noise field must never be saved as a pattern., _seed_index() (+10 more)

### Community 53 - "Experiment Tracking Backend"
Cohesion: 0.12
Nodes (6): Protocol, build_tracker(), NoOpTracker, tracking.py — a Tracker protocol with a no-op default.  Turning on an experiment, Tracker, WandbTracker

### Community 54 - "Long-Training & Trivial-Baseline Experiments"
Cohesion: 0.11
Nodes (12): EXP07 - Iteration count as the axis. "Batch size introducing iterations."  THE O, med(), EXP08 - The trivial-baseline control. Is recovered k* actually FROM THE IMAGE?, EXP09 - Two questions at the 4000-step budget found by exp07.  Q1  THE BAD-ANCHO, args_config_id(), _runlog.py — shared auditability helper for the experiment scripts.  The library, sha256 of the canonical (sorted-key) JSON of the script's resolved argument, Mint a run_id + config_id + provenance block for one script invocation.      Cal (+4 more)

### Community 55 - "L-Generalisation Consistency Scoring"
Cohesion: 0.14
Nodes (18): d_ratio(), _group_key(), lgen_consistency(), lgen_table(), _log10_spread(), _mean_of_finite(), modal_sign_agreement(), per_run_lgen_metrics() (+10 more)

### Community 56 - "Three Gene Qvar Smoke Manifest"
Cohesion: 0.11
Nodes (18): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+10 more)

### Community 57 - "Training History & Plausibility Modules"
Cohesion: 0.15
Nodes (14): history.py — the TRAINING TRAJECTORY of a recovery, recorded at a thinned cadenc, BoxRow, _in_box(), plausibility_report(), plausibility.py — biological viability of a RECOVERED model's OWN parameters.  W, True/False for a cited, bounded row; None ("not applicable") for an UNCITED or, Score a RECOVERED model's own parameters against configs/bio_box.yaml.      alph, One parsed row of bio_box.yaml. `low`/`high` are None for an unbounded or     UN (+6 more)

### Community 58 - "Loss Weighting Strategies"
Cohesion: 0.13
Nodes (12): GradNormWeighting, NTKWeighting, Base interface. `combine` returns (scalar_total, weights_used)., Linearly ramp selected weights from 0 to base over `warmup` steps. Lets the, GradNorm-style balancing: scale each term so per-term gradient magnitudes match., NTK-PINN weighting: set weights from the diagonal of the neural-tangent-kernel /, ScheduledWeighting, WeightingStrategy (+4 more)

### Community 59 - "Morphology Classification Module"
Cohesion: 0.18
Nodes (18): _as_field(), build_reference_bank(), classify_morphology(), morphology_distance(), morphology_vector(), ndarray, morphology.py — MORPHOLOGY as a first-class scored metric (SCORING SIDE).  WHY T, Validate a single 2-D square real field. Raises rather than coercing. (+10 more)

### Community 60 - "Cross-L Residual & Frame Tests"
Cohesion: 0.12
Nodes (18): _frame(), The residual term is the one place L enters the objective. It must be the SAME, The SAME formula (D_model median = 1/kstar_obs**2) must be right on both paths, One known frame, three domain sizes: the recovered NETWORK is identical.      Th, L = 1 is the fixed point of the change of variables, so the two paths must coinc, Fail loud: L <= 0 would divide by zero in the conversion back to physical units., A deterministic patterned 3-channel frame. Content is irrelevant to every assert, A RecoveryInput stand-in (frame, L, observed_idx, N) — the whole firewall surfac (+10 more)

### Community 61 - "D-Ratio Biological Prior"
Cohesion: 0.14
Nodes (14): D-ratio soft prior (biological plausibility, centred 7.5), D1: D-ratio prior centred at 7.5 (literature Nodal/Lefty), not ~100 (generator median), param_prior(), Soft biological-plausibility prior on the model's OWN parameters. Recovery-side, Minimal stand-in exposing exactly the properties param_prior reads (torch     te, Perturbing D away from the centre and taking a small gradient step on theta_D, A third, near-immobile species (very small D) must leave the prior identical to, beta has no bounds in the test box (UNCITED) and param_prior does not even read (+6 more)

### Community 62 - "Bio-Box Plausibility Config"
Cohesion: 0.15
Nodes (18): bio_box.yaml (biological plausibility box), bio_box.yaml alpha (production-weight scale prior), bio_box.yaml beta (basal production rate, UNCITED), bio_box.yaml d_ratio (diffusivity ratio plausibility box, centre 7.5), bio_box.yaml delta (degradation rate scale prior), D3: bio_box D-ratio upper bound of 60.0 is a reasoned proxy, not a measured bound, D5: k_star_fft (image-derived) is the headline validation target, not analytic k_star, D-EVID-11: turing_ok tested tr(J)<0, scoring uniformly-unstable systems as Turing-unstable (+10 more)

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
Cohesion: 0.15
Nodes (14): losses/ — the composite objective (RECOVERY SIDE).  terms.py     : differentiabl, compute_terms(), compute_terms_batched(), parts_member(), total.py — assemble the composite objective from named, source-tagged terms.  Ea, Batched twin of `compute_terms`. Returns (term_vals, parts, converged).      `mo, Batched twin of `total_loss`. Returns (loss_vec (B,), parts, converged (B,))., One member's slice of a batched `parts` dict, in the SHAPE the serial path retur (+6 more)

### Community 67 - "RNGRN Model Definition"
Cohesion: 0.13
Nodes (8): build_model(), _low_basal_raw_params(), model.py — the RNGRN model core (VALIDATED science, repackaged as-is).  Gated-pr, Construct a model from a ModelConfig (registry-dispatched by architecture)., Draw raw (unconstrained) theta for the low-basal init (ported from     scripts/e, A tiny name -> implementation registry.  Used for the three varying concerns (mo, Registry, T

### Community 68 - "Observables & k* Extraction"
Cohesion: 0.19
Nodes (16): _angular(), classify(), kstar_of(), laplacian_5point(), laplacian_spectral(), laplacian_torch(), morphology(), ndarray (+8 more)

### Community 69 - "Loss Staging & Weighting Tests"
Cohesion: 0.15
Nodes (13): TrainConfig, DataFirstStaging, FixedWeighting, WRAPPER that applies `staging_factor` to selected terms of ANY inner strategy., Staging is itself a static schedule, so this is exactly the inner strategy's., _batched_adam_steps(), Minimal stand-in for data.gate.RecoveryInput — (frame, L, observed_idx, N) only., The switch must default to the serial path so no pre-existing number changes met (+5 more)

### Community 70 - "Parameter Vector Utilities"
Cohesion: 0.17
Nodes (11): _d_slice(), _param_block(), _param_vector(), ndarray, The D entries of a flattened parameter vector (D is last in PARAM_SPEC)., Freeze the scalar column order on the first record; fail loud if it changes., The scalar columns of one member: diagnostics, per-term losses, live weights., One serial restart's state at `step`. Call BEFORE `opt.step()` so the recorded (+3 more)

### Community 71 - "Data Firewall Enforcement"
Cohesion: 0.17
Nodes (14): The recovery/ground-truth data firewall, DATA_INTO_MODEL.md — audit of what reaches recovery, D-EVID-9: firewall audit was a hand-copied allowlist; new modules unaudited by default, _imports(), test_firewall.py — THE firewall enforcement (not a comment, a test).  No recover, The rule CLAUDE.md §5 states, enforced here rather than only in the scorers' tes, RecoveryInput must expose only frame/L/observed_idx/N — no truth fields., A module both sides import must be as clean as the recovery side, or it becomes (+6 more)

### Community 72 - "Three Gene MultiL Manifest"
Cohesion: 0.13
Nodes (14): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+6 more)

### Community 73 - "Three Gene Train Manifest"
Cohesion: 0.13
Nodes (14): answer_key_keys, checksum, dataset_id, frame_shape, D, interaction_matrix, jacobian, kstar (+6 more)

### Community 74 - "Sweep Runner & No-Op Decisions"
Cohesion: 0.19
Nodes (14): D-EVID-5: loss.weights.param_prior was a silent no-op in the library path, D-EVID-16: branch consolidation — model.init was a silent no-op; ledger resweep at 154 rows, D-EVID-4: train.seed does not change recovery — every seed replicate is the same draw (model_seed no-op), _as_literal(), _check_axis(), grid_points(), sweep.py — the outer optimisation loop, built OUTWARD from fit().  A sweep enume, axes: {dotted_key: [values]} -> list of override dicts (cartesian product). (+6 more)

### Community 75 - "Frame-Scale Anchor & Split Hinges"
Cohesion: 0.15
Nodes (14): Frame-scale anchor (firewall-legal x* proxy from image mean), Split-support Turing hinges (disjoint k-support for stability/instability terms), D7: loss.weights.resid defaulted to 0.0 — settled OFF, D8: split hinges + frame-scale anchor promoted into the library (0% to 36.8% Turing), composite_loss(), frame_scale_anchor(), Re-solve one differentiable Newton polish step from a converged xstar so the ret, Turing hinges on DISJOINT k-support. The promoted default (see total.compute_ter (+6 more)

### Community 76 - "Run Metadata & Checkpoint I/O"
Cohesion: 0.15
Nodes (11): Write the run's identity/provenance block to a sidecar JSON next to the     scri, write_meta(), append_run_index(), atomic_write_json(), load_checkpoint(), io.py — run directories, atomic writes, checkpoints, the run index.  Run layout, Write JSON to a temp file and rename on success (never a half-written file)., Each producer writes its OWN result file (train_results / analysis_results / ... (+3 more)

### Community 77 - "Dataset Registry"
Cohesion: 0.21
Nodes (13): build_manifest(), _checksum(), dataset_dir(), list_datasets(), load_manifest(), payload_path(), registry.py — the dataset store (ANSWER-KEY side).  A place to KEEP datasets you, Inspect an HDF5 payload IN PLACE and build its manifest dict (metadata only). (+5 more)

### Community 78 - "Plottable Array Persistence"
Cohesion: 0.15
Nodes (13): lgen_fields_path(), lgen_rows(), load_plot_arrays(), plot_arrays_path(), plotdata.py — the PLOTTABLE arrays of a run, persisted next to its scalars.  WHY, Write `arrays` to a compressed npz at `path`. Returns the file size in bytes., Read an npz written by `save_plot_arrays`. `meta` comes back as a dict., Flatten `eval.lgen_eval.evaluate_across_L`'s result into index rows.      Return (+5 more)

### Community 79 - "Rollout Module (Lift-and-Simulate)"
Cohesion: 0.15
Nodes (12): _collapsed(), rollout.py — lift-and-simulate: does the recovered circuit actually pattern?  Tr, Has the field decayed so far that 'unpatterned' is already locked in?      THE S, Has the pattern stopped changing? THE STOPPING RULE — read this before trusting, _reaction_np_builder(), _saturated(), rfft2 stores the non-redundant half of a REAL field's spectrum; irfft2 enforces, Either signal alone is a known false positive; the rule requires both. (+4 more)

### Community 80 - "Recovery Scoring (validate.py)"
Cohesion: 0.17
Nodes (11): _first_channel(), _leak_instrumentation(), _morphology_metrics(), validate.py — score a recovery against the quarantined answer key.  Scoring prio, Take the single 2-D field a morphology comparison operates on.      Accepts (H,, Morphology block. Returns a flat dict of scalars.      Split by COST, deliberate, The image-blind control for the k* numbers in this project — ONE PER REFERENCE., Read a scalar reference wavenumber off the answer key as a float, or NaN if abse (+3 more)

### Community 81 - "Cubic Dispersion Backend Tests"
Cohesion: 0.19
Nodes (12): The cubic dispersion backend must be EXACTLY equivalent to the eigvals reference, (J, D) pairs from the registered three_gene answer keys, or skip if absent., Random model states, not just the answer keys — covers wilder theta. Parametrize, The backend is used inside a differentiable loss, so gradients must agree too., Exact for N=3 only — must fail loud rather than silently return nonsense.      R, _real_jacobians(), _sigma_cubic(), _sigma_eig() (+4 more)

### Community 82 - "Robustness Baseline Figure (docs)"
Cohesion: 0.33
Nodes (12): RNGRN robustness baseline figure (3-panel), Node diffusibility ordering (slowest/middle/fastest as-generated) forced non-diffusible, Panel a: fraction still Turing vs multiplicative parameter noise, by topology, Panel b: within-topology robustness spread at 10% noise (0.23-1.00), Panel c: fraction of 127 systems still Turing when a node is forced non-diffusible, "Fraction still strictly Turing" robustness metric, Tica et al. — experimental parameter CV = 4.8% reference line, Cross-repression topology (+4 more)

### Community 83 - "Turing-First Homotopy Experiment"
Cohesion: 0.21
Nodes (11): diagnose(), fit(), low_basal_init(), EXP03 - "Turing-first" (homotopy) recovery vs the current random-restart scheme., Disjoint-support hinges (exp02): uniform stability from k=0, instability from, FIREWALL-SAFE prior. Low basal + sub-saturated binding is what admits a positive, turing_hinges_split(), kstar_anchor() (+3 more)

### Community 84 - "Ratio-Based Loss Weighting Tests"
Cohesion: 0.27
Nodes (9): Ratio-based loss balancing, as actually shipped by Matas-Gil & Endres     (githu, RatioWeighting, test_weighting.py — unit 13: ratio-based loss balancing + stub strategies.  Cove, _terms(), test_build_strategy_ratio_reads_update_every_from_loss_cfg(), test_fixed_weighting_still_works_as_a_control(), test_ratio_equalises_physics_terms_to_data_loss_after_update(), test_ratio_guards_against_zero_term() (+1 more)

### Community 85 - "Determinism Tests"
Cohesion: 0.23
Nodes (11): Turn on (or off) torch's strict determinism controls.      CUBLAS_WORKSPACE_CONF, set_deterministic(), _make_input(), test_determinism.py — recover() is bit-identical across two runs at the same see, Different train.seed -> different recovery. Same train.seed -> bit-identical., Unit B1: recover(seed=s) and recover(seed=s+1) must share NO restart init --, test_dispersion_backend_cubic_rejected_for_wrong_N(), test_model_seed_controls_init_independent_of_train_seed() (+3 more)

### Community 86 - "Robustness Cloud Tests"
Cohesion: 0.17
Nodes (11): test_robustness.py — synthetic-Jacobian tests for eval.analysis's robustness clo, robustness_volumes emits exactly the four run-index keys the spec asks for, and, A single sigma_log must produce the SAME relative perturbation regardless of a, Section 3.2: a lognormal factor is always positive, so a perturbed entry can, tr(J) < 0 is necessary but not sufficient for max Re eig(J) < 0. J = diag(5, -3,, The vectorized (n_samples, N, N) linear algebra in _perturb_cloud must give, test_batching_matches_serial(), test_perturbation_preserves_sign_and_structural_zeros() (+3 more)

### Community 87 - "Turing Hinge Split Tests"
Cohesion: 0.18
Nodes (9): Control for the test above: no interior instability -> the instability hinge bit, The whole point of the split: an unstable mode at k < k_min must NOT satisfy the, Minimal model surface used by the hinges: jacobian() + dispersion(). Lets a test, Design zero: sigma(0) < 0 (uniform-stable) AND max_{k>=k_min} sigma(k) > 0. The, _StubDispersion, test_split_hinges_are_near_zero_when_both_conditions_hold(), test_split_hinges_ignore_instability_below_k_min(), test_split_hinges_penalise_a_uniformly_stable_dispersion() (+1 more)

### Community 88 - "Dataset Registry Tests"
Cohesion: 0.24
Nodes (9): _make_payload(), test_registry.py — dataset store + pluggable index backend.  Covers: register a, registry.py is ANSWER-KEY side: no recovery-side module may import it., A tiny HDF5 payload: per-sample final_frame + quarantined jacobian/x_star., The manual-drop workflow: put payload.h5 at <root>/<id>/ yourself, then scan., test_index_roundtrip(), test_register_list_load(), test_registry_not_imported_by_recovery_side() (+1 more)

### Community 89 - "Three Gene Classical Train Manifest (partial)"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

### Community 90 - "Three Gene Qvar Manifest (partial)"
Cohesion: 0.22
Nodes (8): checksum, dataset_id, frame_shape, n_samples, observable_key, provenance, source, splits

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
Cohesion: 0.33
Nodes (7): build_plot_arrays(), plot_kgrid(), ndarray, sigma(k) = max_i Re eig(J - k^2 diag(D)) on a numpy (J, D) over a k grid.      T, The k grid both sigma(k) curves are evaluated on: recovery's OWN grid.      Reus, Assemble every array a per-run figure needs. Pure: computes, writes nothing., sigma_of()

### Community 101 - "Three Gene Classical Train Answer-Key Fields"
Cohesion: 0.33
Nodes (6): answer_key_keys, D, interaction_matrix, jacobian, kstar, x_star

### Community 102 - "Three Gene Qvar Answer-Key Fields"
Cohesion: 0.33
Nodes (6): answer_key_keys, D, interaction_matrix, jacobian, kstar, x_star

### Community 103 - "Two Gene Classical Train Answer-Key Fields"
Cohesion: 0.33
Nodes (6): answer_key_keys, D, interaction_matrix, jacobian, kstar, x_star

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
Cohesion: 0.33
Nodes (5): _cos(), THE defect, made visible.      At a random init argmax sigma is the k~0 point (a, The premise the split is built on (docs/STATE_OF_THE_SCIENCE.md 2.2). If this ev, test_argmax_dispersion_is_the_k0_grid_point_at_random_init(), test_old_hinges_collapse_onto_one_direction_split_do_not()

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
Cohesion: 0.40
Nodes (5): LossConfig, Weights and the weighting strategy. Weights are STARTING points — Claude Code, SETTLED OFF (exp06: 9 cells x 8 seeds, all collapsed). The term still exists., test_loss_config_staging_defaults_match_the_promoted_objective(), test_stationarity_residual_weight_defaults_to_zero()

### Community 123 - "Cross-L Transfer Decisions (docs)"
Cohesion: 0.50
Nodes (4): D15: cross-L rollouts hold dx fixed, with a measured 6px/wavelength floor, D16: cross-L statistics — population CV of k*, origin-fixed q-vs-L slope, no pass threshold, D-PLOT-3: cross-L table persisted through run-index machinery; tidy/long exports, First cross-L transfer measurement — 2026-07-30

### Community 124 - "No-Op Knob Check Script"
Cohesion: 0.67
Nodes (3): load(), main(), c2_noop_check.py — prove every swept knob actually changed something.  FOUR sile

### Community 125 - "Batched Steady-State Newton"
Cohesion: 0.50
Nodes (4): Batched damped Newton for f(x*)=0 on B independent members. Returns (x*, converg, Componentwise bracket [lo, hi] containing EVERY positive root of f (both forms)., steady_state_batched(), steady_state_bracket()

### Community 126 - "Dispersion/Turing-OK Consistency Test"
Cohesion: 0.50
Nodes (4): plotdata.sigma_of must be the SAME curve eval.analysis.turing_ok maxes over., A (J, D) pair that is genuinely Turing-unstable, so sigma(k) has a real peak., test_sigma_of_agrees_with_turing_ok(), _turing_JD()

### Community 131 - "Morphology Call Result Type"
Cohesion: 0.67
Nodes (3): MorphologyCall, NamedTuple, Result of a morphology class call.      label   : nearest class in the z-scored

## Knowledge Gaps
- **628 isolated node(s):** `$schema`, `Bash(git status:*)`, `Bash(git diff:*)`, `Bash(git log:*)`, `Bash(git show:*)` (+623 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RNGRN` connect `Batched RNGRN Model` to `Repo Conventions & Review Agents`, `Cross-L Evaluation Tests`, `Batched Recovery Tests`, `L-Generalisation Recovery Tests`, `Morphology Scoring Tests`, `QSS Reduction & Robustness Analysis`, `Experiment Arm Scoring Tests`, `Turing Hinges & Steady-State Solve`, `Bio-Viability Stage0 Script`, `Cross-Cutting Science Concepts (docs)`, `Rollout Simulation Tests`, `Steady-State Newton Solver`, `Plausibility Scoring Tests`, `Science Correctness Anchors`, `Config Identity & Hashing`, `Losses Unit Tests`, `RNGRN Model Core (dispersion/jacobian)`, `Cross-L Residual & Frame Tests`, `D-Ratio Biological Prior`, `C2 Init-Bias Script`, `RNGRN Model Definition`, `Loss Staging & Weighting Tests`, `Data Firewall Enforcement`, `Sweep Runner & No-Op Decisions`, `Run Metadata & Checkpoint I/O`, `Cubic Dispersion Backend Tests`, `Turing-First Homotopy Experiment`, `Determinism Tests`, `Turing Hinge Split Tests`, `Dispersion Argmax-at-k0 Tests`, `Model Device Property`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Why does `score_recovery()` connect `Experiment Arm Scoring Tests` to `Dataset Cache & Loading`, `Repo Conventions & Review Agents`, `Sweep Runner & No-Op Decisions`, `L-Generalisation Recovery Tests`, `Morphology Scoring Tests`, `QSS Reduction & Robustness Analysis`, `Gate Contract Tests`, `Recovery Scoring (validate.py)`, `Config Identity & Hashing`, `Dataset Families & Identifiability Docs`, `Cross-Cutting Science Concepts (docs)`, `Bio-Box Plausibility Config`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `docs/DECISIONS.md D-EVID-4 (cited external doc)` connect `Cross-Cutting Science Concepts (docs)` to `Cross-Target Tuning Control Arm`, `Firewall Gate & Experiment Configs`, `Data Firewall Enforcement`, `Sweep Runner & No-Op Decisions`, `Frame-Scale Anchor & Split Hinges`, `QSS Reduction & Robustness Analysis`, `Dataset Families & Identifiability Docs`, `Cross-L Transfer Decisions (docs)`, `D-Ratio Biological Prior`, `Bio-Box Plausibility Config`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `RNGRN` (e.g. with `Quasi-steady-state (QSS) reaction reduction` and `init_jacobian()`) actually correct?**
  _`RNGRN` has 60 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `Bash(git status:*)`, `Bash(git diff:*)` to the rest of the system?**
  _628 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Three Gene MultiL Dataset Samples` be split into smaller, more focused modules?**
  _Cohesion score 0.021505376344086023 - nodes in this community are weakly interconnected._
- **Should `Three Gene Train Dataset Samples` be split into smaller, more focused modules?**
  _Cohesion score 0.02247191011235955 - nodes in this community are weakly interconnected._