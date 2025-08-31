"""Enhanced Chemistry Validation System - Backward Compatibility Module.

This module maintains backward compatibility by re-exporting the refactored
EnhancedChemistryValidator class and related components from the new modular structure.

The original monolithic file has been refactored into smaller, focused modules:
- enhanced_chemistry_validator.py: Main validator class
- database_validators.py: Database validation logic
- chemical_rules.py: Chemical rules validation
- environmental_validators.py: Environmental conditions validation
- formation_probability.py: Formation probability calculations
- data_provenance.py: Data provenance tracking
- chemistry_utils.py: Common utility functions
"""

# Import the refactored components
from .enhanced_chemistry_validator import (
    EnhancedChemistryValidator,
    CompoundSuggestion,
    EnvironmentalConditions
)

# Re-export for backward compatibility
__all__ = [
    'EnhancedChemistryValidator',
    'CompoundSuggestion', 
    'EnvironmentalConditions'
]

# Maintain backward compatibility by creating module-level aliases
Validator = EnhancedChemistryValidator  # Common alias