NASA Rover Chemistry Prediction System
=====================================

Overview
- Goal: Generate realistic, validated training data to predict which compounds can form from detected elements under given environmental conditions. The system is planet‑agnostic and focuses on physically plausible chemistry with traceable provenance.
- Outputs: Validated compound suggestions, formation probabilities, limiting factors, and provenance metadata suitable for ML training.

Key Components
- Data models: Core types for environmental conditions, compound suggestions, and validation results are defined in:
  • EnvironmentalConditions: temperature, pressure, pH, humidity, duration, atmosphere, radiation, etc.
  • CompoundSuggestion: formula, name, compound_type, stability, source, confidence.
  • ValidationResult: feasible flag, formation_probability, limiting_factors, validation_details, data_provenance.
  See <mcfile name="data_models.py" path="src/data/data_models.py"></mcfile>

- Validator Orchestrator: EnhancedChemistryValidator coordinates all checks and computes an overall feasibility assessment and probability. See <mcfile name="enhanced_chemistry_validator.py" path="src/core/enhanced_chemistry_validator.py"></mcfile>

Validation Pipeline (What we validate and how)
1) Element availability and stoichiometry
   - Extract elements from the candidate formula and verify availability in the input element set.
   - Check stoichiometric balance with the available molar quantities.
2) Database corroboration
   - Query multiple sources (e.g., NIST, Materials Project, PubChem/RDKit when available). The validator aggregates flags like “nist_found”, “materials_project_found”, etc., which contribute to evidence and confidence.
3) Chemical rules and heuristics
   - Valence, charge balance, common oxidation states, simple incompatibilities, and rule‑based sanity checks for compound_type (ionic, molecular, metallic, network).
4) Environmental suitability
   - Assesses:
     • Temperature windows by compound class/subtype and thermal decomposition risk.
     • Pressure/atmosphere effects (e.g., gas solubility, vacuum stability, high‑pressure effects).
     • Phase‑equilibria‑inspired stability factors using simplified reduced T/P reasoning when applicable.
5) Reaction balancing plausibility
   - Heuristic check to ensure the candidate could be balanced from the available elements.
6) Formation probability and confidence
   - Combines environmental score, chemical rules score, and database corroboration into an overall formation_probability.
   - A compound is marked feasible when formation_probability > 0.30 (tunable threshold in code).
7) Provenance tracking
   - Records which data sources contributed, a computed evidence_score, and stamped validation timestamps.

Dataset Generation (Structured process)
- Inputs per sample
  • Elements: Dict of {element_symbol: moles}.
  • EnvironmentalConditions: Planet‑agnostic snapshot (e.g., moderate lab‑like defaults if none provided).
- Process
  1) Element set sampling or ingestion.
  2) Candidate generation (LLM or fallback heuristics) to produce CompoundSuggestion objects.
  3) Full validation pipeline per candidate via EnhancedChemistryValidator.
  4) Collect ValidationResult objects with feasibility, probabilities, limiting factors, and provenance.
  5) Export results for ML training (e.g., CSV/JSON) including both features and labels.

Reliability and Data Quality Guarantees
- Multi‑source corroboration: Database hits (when libraries and APIs are available) strengthen evidence and raise confidence.
- Rule‑based safeguards: Chemistry rules guard against impossible/implausible structures.
- Environment‑aware scoring: Conditions (T, P, pH, atmosphere) directly influence feasibility and probability.
- Provenance and traceability: Each result includes data_sources, evidence_score, and a validation timestamp, enabling auditability.
- Known limitations
  • Thermodynamic/phase‑equilibria modeling is simplified when advanced libraries are unavailable; treat formation_probability as a ranking signal, not absolute truth.
  • Candidate generation can produce out‑of‑distribution formulas if not constrained by domain priors.
  • Physical realism depends on how input environments are sampled; ensure ranges match your mission scenario.

Planet‑Agnostic Design
- All hard‑coded planetary presets were removed. Use EnvironmentalConditions directly and supply mission‑specific ranges. In the absence of user input, the system defaults to moderate, lab‑like conditions (room temperature, ~1 atm, neutral pH) to remain generic.

How to Use
- Command line
  • From the project root, run: python main.py --help to see available options for sample count, configuration, and output paths. See <mcfile name="main.py" path="main.py"></mcfile>
- Programmatic API (high‑level)
  • Construct EnvironmentalConditions and element dictionaries.
  • Create CompoundSuggestion candidates (or use the generator’s built‑ins).
  • Call EnhancedChemistryValidator.validate_compound_feasibility for each candidate.
  • Persist ValidationResult objects for training.

What the Output Contains (per candidate)
- feasible: bool — True if formation_probability > 0.30.
- formation_probability: float — Composite score from environment, rules, and database corroboration.
- limiting_factors: list[str] — Reasons that penalized feasibility (e.g., missing elements, temperature out of window).
- environmental_validation and chemical_rules: Detailed sub‑scores and rationale.
- data_provenance: sources consulted, evidence_score, method/schema versions, timestamp.
- conditions_used: Echo of the EnvironmentalConditions used for validation.

Best Practices for ML Training
- Constrain candidate generation with chemistry priors to reduce noise.
- Match environment sampling to mission envelopes (e.g., Mars rover local conditions) rather than broad random ranges.
- Keep “hard negatives” (invalid or implausible candidates) and “easy negatives” (missing element cases) for robust decision boundaries.
- Monitor coverage: ensure adequate distribution across compound types and element sets.
- Track seeds and configs to reproduce splits and metrics.

Extensibility
- Add new validators under src/validators and register them in the orchestrator.
- Replace or augment candidate generators to embed domain knowledge.
- Integrate richer thermodynamics/phase diagrams when libraries/data become available to raise fidelity.

File Map (pointers)
- Data models: <mcfile name="data_models.py" path="src/data/data_models.py"></mcfile>
- Validator orchestrator: <mcfile name="enhanced_chemistry_validator.py" path="src/core/enhanced_chemistry_validator.py"></mcfile>
- Environmental validators: <mcfile name="environmental_validators.py" path="src/validators/environmental_validators.py"></mcfile>
- Entry point/CLI: <mcfile name="main.py" path="main.py"></mcfile>

Contact & Support
- Open an issue with a minimal reproducible example, including:
  • Input elements/environment
  • Candidate formula(s)
  • Expected vs. observed behavior
  • Software environment and any optional libraries installed (pymatgen, RDKit, PubChemPy)