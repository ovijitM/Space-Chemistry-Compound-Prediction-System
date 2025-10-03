Space Chemistry: Compound Prediction System

=====================================

A comprehensive, universal chemistry analysis and prediction system for space exploration missions. This advanced system simulates rover drilling operations, analyzes elemental compositions, predicts chemical compound formations, and trains machine learning models for future space chemistry discoveries.

Overview

## 🌟 Overview- Goal: Generate realistic, validated training data to predict which compounds can form from detected elements under given environmental conditions. The system is planet‑agnostic and focuses on physically plausible chemistry with traceable provenance.

- Outputs: Validated compound suggestions, formation probabilities, limiting factors, and provenance metadata suitable for ML training.

The NASA Rover Chemistry Prediction System is a state-of-the-art platform designed to:

Key Components

- **Simulate Universal Drilling Operations**: Works on any unknown environment, not limited to specific planets- Data models: Core types for environmental conditions, compound suggestions, and validation results are defined in:

- **Analyze Elemental Compositions**: Identify and quantify chemical elements found at drilling sites • EnvironmentalConditions: temperature, pressure, pH, humidity, duration, atmosphere, radiation, etc.

- **Predict Compound Formations**: Calculate formation probabilities for potential chemical compounds • CompoundSuggestion: formula, name, compound_type, stability, source, confidence.

- **Train ML Models**: Use collected data to predict future discoveries and chemical behaviors • ValidationResult: feasible flag, formation_probability, limiting_factors, validation_details, data_provenance.

- **Generate Comprehensive Reports**: Provide detailed analysis with beautiful visualizations See <mcfile name="data_models.py" path="src/data/data_models.py"></mcfile>

- **Analyze Known Element Pools**: Identify combinations and formations of known chemical elements under space conditions

- Validator Orchestrator: EnhancedChemistryValidator coordinates all checks and computes an overall feasibility assessment and probability. See <mcfile name="enhanced_chemistry_validator.py" path="src/core/enhanced_chemistry_validator.py"></mcfile>

### 🔬 What It Does

Validation Pipeline (What we validate and how)

1. **Multi-Location Drilling Survey**: Simulates drilling at multiple locations with varying depths and environmental conditions1) Element availability and stoichiometry

2. **Element Detection & Analysis**: Discovers elements, measures abundances, and compares to Earth chemistry - Extract elements from the candidate formula and verify availability in the input element set.

3. **Compound Formation Prediction**: Tests thousands of potential chemical compounds and calculates formation probabilities - Check stoichiometric balance with the available molar quantities.

4. **Machine Learning Training**: Trains multiple ML models to predict:2) Database corroboration

   - Earth similarity percentages - Query multiple sources (e.g., NIST, Materials Project, PubChem/RDKit when available). The validator aggregates flags like “nist_found”, “materials_project_found”, etc., which contribute to evidence and confidence.

   - Discovery potential of new locations3) Chemical rules and heuristics

   - Compound formation success rates - Valence, charge balance, common oxidation states, simple incompatibilities, and rule‑based sanity checks for compound_type (ionic, molecular, metallic, network).

   - Element abundance patterns4) Environmental suitability

5. **Advanced Visualizations**: Creates comprehensive charts, heatmaps, and interactive dashboards - Assesses:

6. **Final Discovery Pool Analysis**: Aggregates all findings into a comprehensive elemental and compound inventory • Temperature windows by compound class/subtype and thermal decomposition risk.

   • Pressure/atmosphere effects (e.g., gas solubility, vacuum stability, high‑pressure effects).

## 🏗️ System Architecture • Phase‑equilibria‑inspired stability factors using simplified reduced T/P reasoning when applicable.

5. Reaction balancing plausibility

### Core Components - Heuristic check to ensure the candidate could be balanced from the available elements.

6. Formation probability and confidence

````- Combines environmental score, chemical rules score, and database corroboration into an overall formation_probability.

NASA-Rover-Chemistry-Prediction-System/   - A compound is marked feasible when formation_probability > 0.30 (tunable threshold in code).

├── enhanced_mission_system.py     # Main orchestrator7) Provenance tracking

├── src/   - Records which data sources contributed, a computed evidence_score, and stamped validation timestamps.

│   ├── core/                      # Core analysis engines

│   │   ├── multi_location_survey.pyDataset Generation (Structured process)

│   │   ├── enhanced_drilling_analyzer.py- Inputs per sample

│   │   ├── ml_trainer.py  • Elements: Dict of {element_symbol: moles}.

│   │   └── simple_visualizer.py  • EnvironmentalConditions: Planet‑agnostic snapshot (e.g., moderate lab‑like defaults if none provided).

│   ├── data/                      # Data models and handlers- Process

│   │   ├── data_models.py  1) Element set sampling or ingestion.

│   │   ├── element_database.py  2) Candidate generation (LLM or fallback heuristics) to produce CompoundSuggestion objects.

│   │   └── thermodynamic_data.py  3) Full validation pipeline per candidate via EnhancedChemistryValidator.

│   ├── validators/                # Chemistry validation systems  4) Collect ValidationResult objects with feasibility, probabilities, limiting factors, and provenance.

│   │   ├── enhanced_chemistry_validator.py  5) Export results for ML training (e.g., CSV/JSON) including both features and labels.

│   │   ├── environmental_validators.py

│   │   └── formation_probability.pyReliability and Data Quality Guarantees

│   └── utils/                     # Utility functions- Multi‑source corroboration: Database hits (when libraries and APIs are available) strengthen evidence and raise confidence.

│       ├── chemistry_utils.py- Rule‑based safeguards: Chemistry rules guard against impossible/implausible structures.

│       └── cache_manager.py- Environment‑aware scoring: Conditions (T, P, pH, atmosphere) directly influence feasibility and probability.

├── cache/                         # Performance cache- Provenance and traceability: Each result includes data_sources, evidence_score, and a validation timestamp, enabling auditability.

├── docs/                          # Documentation- Known limitations

└── tests/                         # Unit tests  • Thermodynamic/phase‑equilibria modeling is simplified when advanced libraries are unavailable; treat formation_probability as a ranking signal, not absolute truth.

```  • Candidate generation can produce out‑of‑distribution formulas if not constrained by domain priors.

  • Physical realism depends on how input environments are sampled; ensure ranges match your mission scenario.

### 🔄 Process Overview

Planet‑Agnostic Design

#### Phase 1: Unified Drilling & Analysis- All hard‑coded planetary presets were removed. Use EnvironmentalConditions directly and supply mission‑specific ranges. In the absence of user input, the system defaults to moderate, lab‑like conditions (room temperature, ~1 atm, neutral pH) to remain generic.

- **Site Selection**: Randomly selects drilling locations across unknown terrain

- **Depth Drilling**: Drills at various depths (0.5m to 8m) with different site typesHow to Use

- **Element Discovery**: Simulates advanced spectrometry to detect chemical elements- Command line

- **Environmental Analysis**: Measures temperature, pH, pressure, and other conditions  • From the project root, run: python main.py --help to see available options for sample count, configuration, and output paths. See <mcfile name="main.py" path="main.py"></mcfile>

- Programmatic API (high‑level)

#### Phase 2: Machine Learning Training  • Construct EnvironmentalConditions and element dictionaries.

- **Data Preparation**: Aggregates drilling data into ML-ready datasets  • Create CompoundSuggestion candidates (or use the generator’s built‑ins).

- **Model Training**: Trains multiple predictive models:  • Call EnhancedChemistryValidator.validate_compound_feasibility for each candidate.

  - RandomForest for Earth similarity prediction  • Persist ValidationResult objects for training.

  - RandomForest for discovery potential assessment

  - Classification models for compound formation successWhat the Output Contains (per candidate)

  - Regression models for element abundance prediction- feasible: bool — True if formation_probability > 0.30.

- formation_probability: float — Composite score from environment, rules, and database corroboration.

#### Phase 3: Prediction Demonstration- limiting_factors: list[str] — Reasons that penalized feasibility (e.g., missing elements, temperature out of window).

- **Future Site Prediction**: Uses trained models to predict outcomes for hypothetical locations- environmental_validation and chemical_rules: Detailed sub‑scores and rationale.

- **Feature Importance Analysis**: Identifies key factors affecting discoveries- data_provenance: sources consulted, evidence_score, method/schema versions, timestamp.

- **Confidence Scoring**: Provides reliability metrics for predictions- conditions_used: Echo of the EnvironmentalConditions used for validation.



#### Phase 4: Advanced VisualizationsBest Practices for ML Training

- **Element Abundance Heatmaps**: Visual representation of element distributions- Constrain candidate generation with chemistry priors to reduce noise.

- **Compound Formation Charts**: Interactive charts showing formation probabilities- Match environment sampling to mission envelopes (e.g., Mars rover local conditions) rather than broad random ranges.

- **Discovery Summary Dashboards**: Comprehensive overview of all findings- Keep “hard negatives” (invalid or implausible candidates) and “easy negatives” (missing element cases) for robust decision boundaries.

- **Correlation Analysis**: Element interaction and relationship mapping- Monitor coverage: ensure adequate distribution across compound types and element sets.

- Track seeds and configs to reproduce splits and metrics.

#### Phase 5: Final Discovery Pool Analysis

- **Comprehensive Inventory**: Complete catalog of discovered elements and compoundsExtensibility

- **Unknown Chemistry Identification**: Highlights space-only discoveries- Add new validators under src/validators and register them in the orchestrator.

- **Statistical Analysis**: Detailed abundance rankings and frequency analysis- Replace or augment candidate generators to embed domain knowledge.

- **Master Visualizations**: Publication-ready charts and dashboards- Integrate richer thermodynamics/phase diagrams when libraries/data become available to raise fidelity.



## 🎯 Expected OutcomesFile Map (pointers)

- Data models: <mcfile name="data_models.py" path="src/data/data_models.py"></mcfile>

### Scientific Discoveries- Validator orchestrator: <mcfile name="enhanced_chemistry_validator.py" path="src/core/enhanced_chemistry_validator.py"></mcfile>

- **Element Pool**: Complete inventory of 40-60+ unique chemical elements- Environmental validators: <mcfile name="environmental_validators.py" path="src/validators/environmental_validators.py"></mcfile>

- **Rare Elements**: 10-15 elements with low Earth abundance found in higher concentrations- Entry point/CLI: <mcfile name="main.py" path="main.py"></mcfile>

- **Compound Library**: 40-50+ potential chemical compounds with formation probabilities

- **Earth Comparison**: Detailed abundance ratios comparing space chemistry to EarthContact & Support

- Open an issue with a minimal reproducible example, including:

### Machine Learning Models  • Input elements/environment

- **Prediction Accuracy**: Trained models for future site assessment  • Candidate formula(s)

- **Discovery Potential**: Algorithms to identify promising drilling locations  • Expected vs. observed behavior

- **Formation Prediction**: Models to predict chemical compound formation success  • Software environment and any optional libraries installed (pymatgen, RDKit, PubChemPy)
- **Pattern Recognition**: AI systems to identify chemical patterns and trends

### Data Products
- **Comprehensive Datasets**: ML-ready data files for further research
- **Interactive Dashboards**: Web-based visualizations for data exploration
- **Scientific Reports**: Detailed analysis reports with publication-quality figures
- **Provenance Tracking**: Complete audit trail of all discoveries and sources

## 🛠️ Setup & Installation

### Prerequisites
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

### 1. Clone Repository
```bash
git clone https://github.com/ovijitM/NASA-Rover-Chemistry-Prediction-System.git
cd NASA-Rover-Chemistry-Prediction-System
````

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually:
pip install numpy pandas scikit-learn matplotlib seaborn
pip install requests beautifulsoup4 lxml
pip install pymatgen pubchempy periodictable
pip install plotly tqdm uncertainties openai
```

### Core Dependencies

- **numpy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib/seaborn**: Statistical visualizations and plotting
- **pymatgen**: Materials science and crystallography
- **pubchempy**: Chemical database access
- **periodictable**: Element properties and data
- **plotly**: Interactive visualizations
- **requests**: Web API interactions
- **tqdm**: Progress bars for long operations
- **uncertainties**: Error propagation and uncertainty calculations
- **openai**: AI-powered compound generation and analysis
- **beautifulsoup4**: Web scraping for chemical databases
- **lxml**: XML parsing for database queries

### Optional Enhancements

```bash
# For enhanced chemical analysis
pip install rdkit-pypi

# For advanced thermodynamics
pip install thermo

# For neural networks
pip install tensorflow torch
```

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# Set Hugging Face token for advanced AI features (optional)
export HUGGINGFACE_TOKEN="your_token_here"

# Set cache directory
export NASA_CACHE_DIR="/path/to/cache"
```

### Setting Hugging Face Token

If you want to use advanced AI features:

1. **Get Token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Create Token**: Generate a new access token
3. **Set in Terminal**:

   ```bash
   # Windows
   set HUGGINGFACE_TOKEN=your_token_here

   # macOS/Linux
   export HUGGINGFACE_TOKEN=your_token_here
   ```

## 🚀 Usage

### Quick Start

```bash
# Run with default settings (6 locations, 8 compounds each)
python enhanced_mission_system.py

# Basic run with custom parameters
python enhanced_mission_system.py --locations 10 --compounds 12
```

### Command Line Arguments

```bash
python enhanced_mission_system.py [OPTIONS]
```

| Parameter      | Type | Default                    | Description                              |
| -------------- | ---- | -------------------------- | ---------------------------------------- |
| `--locations`  | int  | 6                          | Number of drilling locations to survey   |
| `--compounds`  | int  | 8                          | Number of compounds to test per location |
| `--output-dir` | str  | "enhanced_mission_results" | Directory for output files               |
| `--seed`       | int  | 42                         | Random seed for reproducible results     |

### Example Commands

```bash
# Quick survey (3 locations, 4 compounds each)
python enhanced_mission_system.py --locations 3 --compounds 4

# Comprehensive survey (12 locations, 15 compounds each)
python enhanced_mission_system.py --locations 12 --compounds 15 --output-dir "comprehensive_survey"

# Reproducible research run
python enhanced_mission_system.py --seed 12345 --output-dir "research_data_v1"

# Large-scale analysis
python enhanced_mission_system.py --locations 20 --compounds 20 --output-dir "large_scale_analysis"
```

### Parameter Details

#### `--locations` (Drilling Sites)

- **Range**: 1-50 recommended
- **Impact**: More locations = better coverage but longer runtime
- **Typical Values**:
  - Quick test: 3-5 locations
  - Standard survey: 6-10 locations
  - Comprehensive analysis: 12-20 locations

#### `--compounds` (Compounds per Site)

- **Range**: 1-30 recommended
- **Impact**: More compounds = better chemical analysis but slower processing
- **Typical Values**:
  - Basic analysis: 4-6 compounds
  - Standard analysis: 8-12 compounds
  - Deep analysis: 15-25 compounds

#### `--output-dir` (Output Directory)

- **Format**: String path
- **Default**: "enhanced_mission_results"
- **Contains**: All generated data, visualizations, and reports

#### `--seed` (Random Seed)

- **Purpose**: Ensures reproducible results
- **Usage**: Use same seed to replicate exact results
- **Research**: Different seeds explore different scenarios

## 🤖 Machine Learning Models

### Model Architecture

#### 1. Earth Similarity Predictor

- **Algorithm**: RandomForestRegressor
- **Purpose**: Predicts how similar discovered chemistry is to Earth
- **Features**: Element diversity, temperature, location coordinates, abundances
- **Output**: Similarity percentage (0-100%)

#### 2. Discovery Potential Assessor

- **Algorithm**: RandomForestRegressor
- **Purpose**: Evaluates likelihood of significant discoveries
- **Features**: Environmental conditions, element patterns, location characteristics
- **Output**: Discovery potential score (0-100%)

#### 3. Compound Formation Classifier

- **Algorithm**: RandomForestClassifier
- **Purpose**: Predicts whether compounds will successfully form
- **Features**: Temperature, pressure, pH, element availability
- **Output**: Formation success probability

#### 4. Element Abundance Predictor

- **Algorithm**: RandomForestRegressor
- **Purpose**: Predicts elemental abundances at new locations
- **Features**: Environmental factors, location data, geological indicators
- **Output**: Expected abundance percentages

### Feature Engineering

The system automatically generates ML features from:

- **Location Data**: Coordinates, elevation, site type
- **Environmental Data**: Temperature, pH, pressure, depth
- **Chemical Data**: Element counts, abundances, diversity indices
- **Geological Data**: Site characteristics, drilling depth, conditions

### Model Performance

Typical performance metrics:

- **Training Accuracy**: 85-95% depending on data size
- **Cross-Validation**: 5-fold validation for robustness
- **Feature Importance**: Automatically calculated and reported
- **Prediction Confidence**: Uncertainty quantification included

## 📊 Output Files & Results

### Directory Structure

```
enhanced_mission_results/
├── raw_data/
│   ├── drilling_data.json          # Raw drilling results
│   ├── element_analyses.json       # Element discovery data
│   └── compound_analyses.json      # Compound formation data
├── datasets/
│   ├── enhanced_location_dataset.csv    # ML-ready location data
│   ├── enhanced_element_dataset.csv     # ML-ready element data
│   └── enhanced_compound_dataset.csv    # ML-ready compound data
├── trained_models/
│   ├── earth_similarity_model.pkl       # Earth similarity predictor
│   ├── discovery_potential_model.pkl    # Discovery potential assessor
│   ├── compound_formation_model.pkl     # Compound formation classifier
│   └── element_abundance_model.pkl      # Element abundance predictor
├── visualizations/
│   ├── master_discovery_dashboard.png   # Complete overview
│   ├── element_pool_analysis.png        # Element distribution charts
│   ├── compound_pool_analysis.png       # Compound formation charts
│   ├── element_abundance_heatmap.png    # Abundance heatmap
│   ├── discovery_summary.png            # Discovery summary charts
│   ├── element_correlation.png          # Element correlation matrix
│   └── ml_predictions.png               # ML prediction charts
└── reports/
    ├── mission_summary.json             # Complete mission summary
    ├── final_discovery_report.txt       # Human-readable report
    └── data_provenance.json             # Data source tracking
```

### Key Output Files

#### 1. Master Discovery Dashboard

- **File**: `visualizations/master_discovery_dashboard.png`
- **Content**: 4-panel overview with element distributions, discovery categories, formation probabilities, and statistics

#### 2. Element Pool Analysis

- **File**: `visualizations/element_pool_analysis.png`
- **Content**: Detailed element abundance distributions, frequency analysis, Earth ratios, and rarity classifications

#### 3. Compound Pool Analysis

- **File**: `visualizations/compound_pool_analysis.png`
- **Content**: Compound formation probabilities, stability ratings, known vs unknown compounds

#### 4. Enhanced Datasets

- **Location Dataset**: ML-ready data for location-based predictions
- **Element Dataset**: Chemical composition data for element analysis
- **Compound Dataset**: Formation probability data for compound studies

#### 5. Trained Models

- **Format**: Pickled scikit-learn models
- **Usage**: Can be loaded for future predictions
- **Performance**: Includes accuracy metrics and feature importance

## 🧪 Example Output Analysis

### Typical Discovery Results

#### Element Pool (Example)

```
💎 ABUNDANT ELEMENTS (Major Discoveries):
   🌍 Cl: 10.5% avg abundance, 33% frequency (356x Earth)
   🌍 Ca: 5.4% avg abundance, 33% frequency (1.5x Earth)
   ❓ Tc: 4.0% avg abundance, 17% frequency (Unknown on Earth)

💎 RARE ELEMENTS (High Concentration Discoveries):
   � Tc: 4.0% avg abundance - Rare Earth element (356x typical abundance)
   � Po: 2.7% avg abundance - Radioactive element (1250x typical abundance)
   � Hg: 2.7% avg abundance - Heavy metal (890x typical abundance)
```

#### Machine Learning Performance (Example)

```
🧠 ML MODEL PERFORMANCE:
🌍 Earth Similarity Model: R² = 0.85, MSE = 12.4
🔍 Discovery Potential Model: R² = 0.78, MSE = 18.6
⚗️ Compound Formation Model: Accuracy = 92.3%
🧪 Element Abundance Model: R² = 0.71, MSE = 15.2
```

### Statistics Summary

- **Total Elements**: 40-60 unique discoveries
- **Rare Elements**: 10-15 low-abundance Earth elements found in high concentrations
- **Total Compounds**: 40-50 formation candidates
- **Success Rate**: 85-95% valid formations
- **Earth Similarity**: 70-95% depending on location

## 🔬 Advanced Features

### Chemistry Validation System

- **Multi-Database Validation**: Cross-references NIST, PubChem, PyMatGen databases
- **Thermodynamic Analysis**: Advanced stability calculations
- **Environmental Suitability**: Temperature, pressure, pH compatibility assessment
- **Formation Probability**: Sophisticated probability calculations based on multiple factors

### Provenance Tracking

- **Data Sources**: Complete tracking of information sources
- **Validation Methods**: Documentation of validation approaches
- **Timestamp Recording**: Audit trail for all discoveries
- **Evidence Scoring**: Confidence levels for all findings

### Visualization Features

- **Interactive Charts**: Plotly-based interactive visualizations
- **Publication Quality**: High-resolution outputs suitable for research
- **Color-Coded Analysis**: Intuitive color schemes for different discovery types
- **Comprehensive Dashboards**: Multi-panel overviews combining multiple analyses

## 🔧 Troubleshooting

### Common Issues

#### 1. Missing Dependencies

```bash
# Error: Module not found
# Solution: Install missing packages
pip install [missing_package_name]

# Or reinstall all dependencies
pip install -r requirements.txt
```

#### 2. Memory Issues

```bash
# Error: MemoryError during large analysis
# Solution: Reduce analysis size
python enhanced_mission_system.py --locations 3 --compounds 4
```

#### 3. Long Runtime

```bash
# Issue: Analysis takes too long
# Solution: Use smaller parameters for testing
python enhanced_mission_system.py --locations 2 --compounds 3
```

#### 4. Permission Errors

```bash
# Error: Permission denied writing files
# Solution: Run with appropriate permissions or change output directory
python enhanced_mission_system.py --output-dir "./my_results"
```

### Performance Optimization

#### For Faster Results

```bash
# Quick test run
python enhanced_mission_system.py --locations 2 --compounds 3

# Use caching (automatic)
# System automatically caches database queries for faster subsequent runs
```

#### For Maximum Detail

```bash
# Comprehensive analysis
python enhanced_mission_system.py --locations 15 --compounds 20
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/ovijitM/NASA-Rover-Chemistry-Prediction-System.git
cd NASA-Rover-Chemistry-Prediction-System

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # For testing and code formatting
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/
```

### Code Style

```bash
# Format code
black enhanced_mission_system.py src/

# Check style
flake8 enhanced_mission_system.py src/
```

## 📚 Research Applications

### Space Mission Planning

- **Site Selection**: Use discovery potential models to prioritize drilling locations
- **Resource Assessment**: Identify valuable elements for in-situ resource utilization
- **Safety Analysis**: Predict chemical hazards and environmental challenges

### Astrobiology Research

- **Biosignature Detection**: Identify chemical indicators of potential life
- **Prebiotic Chemistry**: Analyze conditions for organic compound formation
- **Extreme Environment Chemistry**: Study chemistry under unusual conditions

### Materials Science

- **Novel Compounds**: Discover new chemical compounds for advanced materials
- **Property Prediction**: Predict material properties from elemental composition
- **Synthesis Planning**: Plan laboratory synthesis of discovered compounds

## 📖 Citation

If you use this system in research, please cite:

```bibtex
@software{nasa_rover_chemistry_2025,
  title={NASA Rover Chemistry Prediction System},
  author={NASA Rover Development Team},
  year={2025},
  version={2.0},
  url={https://github.com/ovijitM/NASA-Rover-Chemistry-Prediction-System}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: Check the `/docs` folder for detailed technical documentation
- **Examples**: See `/examples` for usage examples and tutorials

### Contact Information

- **Project Repository**: https://github.com/ovijitM/NASA-Rover-Chemistry-Prediction-System
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Use GitHub Discussions

## 🎉 Acknowledgments

Special thanks to:

- **NASA**: For inspiring the mission concept
- **PyMatGen**: Materials science computing framework
- **Scikit-learn**: Machine learning tools
- **Pandas/NumPy**: Data analysis foundations
- **Matplotlib/Plotly**: Visualization capabilities
- **Open Source Community**: For providing excellent tools and libraries

---

## 🚀 Ready to Explore?

Start your space chemistry analysis journey:

```bash
# Quick start
python enhanced_mission_system.py

# Or customize your mission
python enhanced_mission_system.py --locations 10 --compounds 15 --output-dir "my_space_mission"
```

**Happy discovering! 🌌🔬⚗️**
