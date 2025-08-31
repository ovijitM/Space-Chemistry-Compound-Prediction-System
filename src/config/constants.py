"""Centralized constants for the chemistry validation system.

This module contains all weights, thresholds, and adjustments used throughout
the validation pipeline to ensure consistency and easy maintenance.
"""

from typing import Dict, Any

# ============================================================================
# FORMATION PROBABILITY WEIGHTS
# ============================================================================

FORMATION_PROBABILITY_WEIGHTS: Dict[str, float] = {
    'thermodynamic': 0.25,
    'kinetic': 0.20,
    'structural': 0.15,
    'evidence': 0.15,
    'environmental': 0.15,
    'chemical_rules': 0.10
}

# Ensure weights sum to 1.0
assert abs(sum(FORMATION_PROBABILITY_WEIGHTS.values()) - 1.0) < 1e-6, "Formation probability weights must sum to 1.0"

# ============================================================================
# COMPOUND TYPE ADJUSTMENTS
# ============================================================================

COMPOUND_TYPE_ADJUSTMENTS: Dict[str, float] = {
    'ionic': 1.1,      # Ionic compounds generally more stable
    'molecular': 1.0,   # Baseline
    'metallic': 0.95,   # Slightly less stable due to complexity
    'network': 0.9      # Network solids can be less stable
}

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

VALIDATION_THRESHOLDS: Dict[str, float] = {
    'formation_probability_threshold': 0.3,
    'confidence_threshold': 0.5,
    'evidence_factor_minimum': 0.2,
    'environmental_score_minimum': 0.3,
    'chemical_rules_minimum': 0.4,
    'thermodynamic_minimum': 0.2,
    'kinetic_minimum': 0.3,
    'structural_minimum': 0.25
}

# ============================================================================
# DATABASE RELIABILITY WEIGHTS
# ============================================================================

DATABASE_WEIGHTS: Dict[str, float] = {
    'materials_project': 0.4,  # High reliability - experimental and computed data
    'nist': 0.3,              # High reliability - experimental data
    'pubchem': 0.2,           # Medium reliability - mixed sources
    'rdkit': 0.1              # Lower reliability - computed properties only
}

# Ensure database weights sum to 1.0
assert abs(sum(DATABASE_WEIGHTS.values()) - 1.0) < 1e-6, "Database weights must sum to 1.0"

# Database reliability and data quality parameters
DATABASE_RELIABILITY: Dict[str, Dict[str, Any]] = {
    'materials_project': {
        'base_reliability': 0.9,
        'data_quality_bonuses': {
            'formation_energy_per_atom': 0.4,
            'band_gap': 0.2,
            'density': 0.1,
            'crystal_structure': 0.2,
            'stability': 0.3
        },
        'uncertainty_factor': 0.05
    },
    'nist': {
        'base_reliability': 0.95,
        'data_quality_bonuses': {
            'formation_enthalpy': 0.5,
            'melting_point': 0.3,
            'boiling_point': 0.3,
            'density': 0.2,
            'heat_capacity': 0.2
        },
        'uncertainty_factor': 0.03
    },
    'pubchem': {
        'base_reliability': 0.7,
        'data_quality_bonuses': {
            'molecular_weight': 0.2,
            'smiles': 0.3,
            'properties': 0.2,
            'synonyms': 0.1,
            'safety_data': 0.1
        },
        'uncertainty_factor': 0.1
    },
    'rdkit': {
        'base_reliability': 0.6,
        'data_quality_bonuses': {
            'valid_molecule': 0.5,
            'descriptors': 0.3,
            'fingerprints': 0.2,
            'conformers': 0.1
        },
        'uncertainty_factor': 0.15
    }
}

# ============================================================================
# HUME-ROTHERY RULES PARAMETERS
# ============================================================================

HUME_ROTHERY_PARAMETERS: Dict[str, Any] = {
    'size_factor_tolerance': 0.15,        # ±15% atomic radius difference
    'electronegativity_tolerance': 0.4,   # ±0.4 electronegativity difference
    'valence_electron_ranges': {          # VEC ranges for different structures
        'fcc': (1.0, 1.4),
        'bcc': (1.4, 1.7),
        'hcp': (1.7, 2.1),
        'complex': (2.1, 3.0)
    },
    'size_mismatch_penalty': 0.1,         # Penalty per 1% size mismatch beyond tolerance
    'electronegativity_penalty': 0.05     # Penalty per 0.1 electronegativity difference beyond tolerance
}

# ============================================================================
# ENVIRONMENTAL FACTOR PARAMETERS
# ============================================================================

ENVIRONMENTAL_PARAMETERS: Dict[str, Any] = {
    'temperature_ranges': {  # Optimal temperature ranges by compound type (K)
        'ionic': (200, 1200),
        'molecular': (150, 800),
        'metallic': (300, 2000),
        'network': (400, 1800)
    },
    'pressure_sensitivity': {  # Pressure sensitivity factors
        'ionic': 0.8,
        'molecular': 1.2,
        'metallic': 0.6,
        'network': 0.7
    },
    'ph_tolerance': {  # pH tolerance ranges
        'ionic': 2.0,
        'molecular': 1.5,
        'metallic': 3.0,
        'network': 2.5
    },
    'aggregation_method': 'geometric_mean'  # 'arithmetic_mean' or 'geometric_mean'
}

# ============================================================================
# UNCERTAINTY PARAMETERS
# ============================================================================

UNCERTAINTY_PARAMETERS: Dict[str, float] = {
    'base_uncertainty': 0.1,              # Base uncertainty for all factors
    'database_uncertainty': 0.05,         # Additional uncertainty per missing database
    'experimental_data_bonus': 0.02,      # Uncertainty reduction for experimental data
    'multiple_source_bonus': 0.01,        # Uncertainty reduction per additional source
    'max_uncertainty': 0.5,               # Maximum allowed uncertainty
    'confidence_scaling': 0.8             # Scaling factor for confidence calculation
}

# ============================================================================
# KINETIC PARAMETERS
# ============================================================================

KINETIC_PARAMETERS: Dict[str, Any] = {
    'element_count_penalty': 0.05,         # Penalty per additional element beyond 2
    'metallic_formula_penalty': 0.02,      # Additional penalty for large metallic formulas
    'activation_energy_estimates': {       # Estimated activation energies (kJ/mol)
        'ionic': 50,
        'molecular': 30,
        'metallic': 80,
        'network': 120
    },
    'temperature_scaling': 0.001           # Temperature scaling factor for kinetic feasibility
}

# ============================================================================
# STRUCTURAL PARAMETERS
# ============================================================================

STRUCTURAL_PARAMETERS: Dict[str, Any] = {
    'simple_ratio_bonus': 0.1,            # Bonus for simple stoichiometric ratios
    'binary_intermetallic_bonus': 0.05,   # Additional bonus for binary intermetallics
    'size_mismatch_threshold': 0.2,       # Threshold for significant size mismatch
    'coordination_preferences': {         # Preferred coordination numbers
        'ionic': [4, 6, 8],
        'molecular': [2, 3, 4],
        'metallic': [8, 12],
        'network': [4, 6]
    }
}

# ============================================================================
# PRECISION AND ROUNDING PARAMETERS
# ============================================================================

PRECISION_PARAMETERS: Dict[str, int] = {
    'calculation_precision': 6,            # Internal calculation precision
    'factor_display_precision': 3,         # Individual factor display precision
    'probability_display_precision': 4,    # Final probability display precision
    'confidence_display_precision': 4      # Confidence display precision
}

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

LOGGING_PARAMETERS: Dict[str, Any] = {
    'default_level': 'INFO',
    'file_level': 'DEBUG',
    'console_level': 'WARNING',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'max_file_size': 10 * 1024 * 1024,    # 10 MB
    'backup_count': 5
}

# ============================================================================
# CACHE PARAMETERS
# ============================================================================

CACHE_PARAMETERS: Dict[str, Any] = {
    'max_cache_size': 10000,
    'cache_ttl': 3600,                     # 1 hour in seconds
    'enable_persistent_cache': True,
    'cache_compression': True
}

# ============================================================================
# PERFORMANCE PARAMETERS
# ============================================================================

PERFORMANCE_PARAMETERS: Dict[str, Any] = {
    'batch_size': 100,
    'max_parallel_workers': 4,
    'timeout_seconds': 30,
    'retry_attempts': 3,
    'rate_limit_delay': 1.0
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_constants():
    """Validate that all constants are within expected ranges."""
    # Check that all weights are positive
    for weight_dict in [FORMATION_PROBABILITY_WEIGHTS, DATABASE_WEIGHTS]:
        for key, value in weight_dict.items():
            assert value > 0, f"Weight {key} must be positive, got {value}"
    
    # Check that all thresholds are between 0 and 1
    for key, value in VALIDATION_THRESHOLDS.items():
        assert 0 <= value <= 1, f"Threshold {key} must be between 0 and 1, got {value}"
    
    # Check that compound type adjustments are reasonable
    for key, value in COMPOUND_TYPE_ADJUSTMENTS.items():
        assert 0.5 <= value <= 1.5, f"Compound type adjustment {key} should be between 0.5 and 1.5, got {value}"
    
    print("All constants validated successfully")

# Validate constants on import
validate_constants()