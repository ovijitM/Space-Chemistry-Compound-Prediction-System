"""Enhanced Chemistry Validator - Refactored Main Class.

This module contains the main EnhancedChemistryValidator class that orchestrates
all chemistry validation functionality using the modular components.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Import modular components
from ..validators.database_validators import DatabaseValidators
from ..validators.chemical_rules import ChemicalRulesValidator
from ..validators.environmental_validators import EnvironmentalValidator
from ..validators.formation_probability import FormationProbabilityCalculator
from ..validators.data_provenance import DataProvenanceTracker
from ..utils.chemistry_utils import ChemistryUtils
from ..data.data_models import CompoundSuggestion, EnvironmentalConditions

# Optional chemistry library imports
try:
    from pymatgen.core import Composition, Element
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# CompoundSuggestion and EnvironmentalConditions imported from data_models


class EnhancedChemistryValidator:
    """Enhanced chemistry validator with modular architecture.
    
    This class orchestrates various chemistry validation components to provide
    comprehensive compound feasibility analysis.
    """
    
    def __init__(self, cache_size: int = 1000, enable_thermodynamics: bool = True,
                 enable_phase_equilibria: bool = True, config: Optional[Dict] = None):
        """Initialize the enhanced chemistry validator.
        
        Args:
            cache_size: Size of the validation cache
            enable_thermodynamics: Whether to enable thermodynamic calculations
            enable_phase_equilibria: Whether to enable phase equilibria analysis
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.persistent_cache = None  # Initialize cache if needed
        self.phase_equilibria = None  # Initialize phase equilibria if needed
        
        # Initialize validators
        db_config = getattr(self.config, 'database', {}) if self.config else {}
        self.db_validators = DatabaseValidators(
            config=db_config,
            persistent_cache=self.persistent_cache,
            logger=self.logger
        )
        self.rules_validator = ChemicalRulesValidator()
        self.env_validator = EnvironmentalValidator(
            phase_equilibria=self.phase_equilibria,
            logger=self.logger
        )
        self.probability_calculator = FormationProbabilityCalculator()
        self.provenance_tracker = DataProvenanceTracker()
        self.utils = ChemistryUtils()
        
        # Initialize cache
        self.cache = {}
        self.cache_size = cache_size
        
        # Initialize optional components
        self.thermo_handler = None
        self.phase_equilibria = None
        
        if enable_thermodynamics:
            try:
                # Initialize thermodynamic data handler if available
                # This would be imported from a thermodynamics module
                pass
            except ImportError:
                self.logger.warning("Thermodynamic handler not available")
        
        if enable_phase_equilibria:
            try:
                # Initialize phase equilibria system if available
                # This would be imported from a phase equilibria module
                pass
            except ImportError:
                self.logger.warning("Phase equilibria system not available")
    
    def validate_compound_feasibility(self, compound: CompoundSuggestion,
                                    available_elements: Dict[str, float],
                                    conditions: EnvironmentalConditions) -> Dict[str, Any]:
        """Validate the feasibility of compound formation.
        
        Args:
            compound: Compound suggestion to validate
            available_elements: Available elements and their quantities (moles)
            conditions: Environmental conditions for formation
            
        Returns:
            Comprehensive validation results dictionary
        """
        try:
            # Check cache first
            cache_key = self.utils.get_cache_key(compound, available_elements, conditions)
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for {compound.formula}")
                return self.cache[cache_key]
            
            self.logger.info(f"Validating compound feasibility: {compound.formula}")
            
            # Step 1: Element availability check
            elements_needed = self.utils.extract_elements_from_formula(compound.formula)
            if not all(elem in available_elements for elem in elements_needed):
                return self._cache_and_return(cache_key, {
                    'feasible': False,
                    'reason': 'Missing required elements',
                    'missing_elements': [elem for elem in elements_needed 
                                       if elem not in available_elements],
                    'confidence': 0.0
                })
            
            # Step 2: Stoichiometric analysis
            stoich_result = self.utils.check_stoichiometry(compound.formula, available_elements)
            
            # Step 3: Database validation
            db_validation = self.db_validators.validate_with_databases(compound.formula)
            
            # Step 4: Chemical rules validation
            rules_validation = self.rules_validator.validate_chemical_rules(compound, elements_needed)
            
            # Step 5: Environmental validation
            env_validation = self.env_validator.validate_environmental_conditions(
                compound, conditions
            )
            
            # Step 6: Reaction balancing validation
            reaction_validation = self._validate_reaction_balancing(
                compound, available_elements
            )
            
            # Step 7: Enhanced formation probability calculation
            # Extract scores from validation results
            environmental_score = env_validation.get('overall_score', 0.5)
            chemical_rules_score = rules_validation.get('overall_score', 0.5) if rules_validation else 0.5
            
            formation_result = self.probability_calculator.calculate_enhanced_formation_probability(
                compound, environmental_score, chemical_rules_score, db_validation
            )
            formation_probability = formation_result.get('formation_probability', 0.0)
            
            # Step 8: Thermodynamic data integration
            thermo_data = None
            if self.thermo_handler:
                try:
                    thermo_data = self.thermo_handler.get_thermodynamic_data(compound.formula)
                except Exception as e:
                    self.logger.debug(f"Thermodynamic data retrieval failed: {e}")
            
            # Step 9: Data provenance calculation
            data_sources = []
            if db_validation.get('nist_found'):
                data_sources.append('NIST')
            if db_validation.get('materials_project_found'):
                data_sources.append('Materials Project')
            if db_validation.get('pubchem_found'):
                data_sources.append('PubChem')
            if db_validation.get('rdkit_found'):
                data_sources.append('RDKit')
            
            evidence_score = self.provenance_tracker.calculate_evidence_score(db_validation)
            
            provenance = self.provenance_tracker.create_provenance_record(
                compound.formula, {
                    'database_validation': db_validation,
                    'chemical_rules': rules_validation,
                    'environmental_validation': env_validation,
                    'formation_probability': formation_probability
                }, data_sources, evidence_score
            )
            
            # Step 10: Compile comprehensive results
            result = {
                'feasible': formation_probability > 0.3,
                'formation_probability': formation_probability,
                'confidence': min(0.95, formation_probability + 0.1),
                'stoichiometry': stoich_result,
                'database_validation': db_validation,
                'chemical_rules': rules_validation,
                'environmental_validation': env_validation,
                'reaction_validation': reaction_validation,
                'thermodynamic_data': thermo_data,
                'data_provenance': provenance,
                'validation_timestamp': self.utils.get_timestamp(),
                'conditions_used': conditions.__dict__
            }
            
            # Step 11: Cache and return results
            return self._cache_and_return(cache_key, result)
            
        except Exception as e:
            self.logger.error(f"Validation failed for {compound.formula}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'feasible': False,
                'reason': f'Validation error: {str(e)}',
                'formation_probability': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _cache_and_return(self, cache_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Cache result and return it.
        
        Args:
            cache_key: Key for caching
            result: Result to cache
            
        Returns:
            The cached result
        """
        # Implement LRU cache eviction if needed
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
    
    def _validate_reaction_balancing(self, compound: CompoundSuggestion, 
                                   available_elements: Dict[str, float]) -> Dict[str, Any]:
        """Validate chemical reaction balancing for compound formation.
        
        Args:
            compound: Compound to validate
            available_elements: Available elements and quantities
            
        Returns:
            Reaction validation results
        """
        try:
            # Parse compound formula to get element composition
            element_composition = self._parse_formula_composition(compound.formula)
            
            # Check if we can balance the formation reaction
            balanced_reaction = self._balance_formation_reaction(
                element_composition, available_elements
            )
            
            # Verify stoichiometric requirements
            stoich_valid = self._verify_stoichiometric_requirements(
                element_composition, available_elements, balanced_reaction
            )
            
            # Calculate reaction Gibbs energy if possible
            delta_g_rxn = None
            if balanced_reaction['balanced'] and self.thermo_handler:
                delta_g_rxn = self._calculate_reaction_gibbs_energy(
                    compound.formula, balanced_reaction.get('coefficients', {})
                )
            
            return {
                'valid': balanced_reaction['balanced'] and stoich_valid['valid'],
                'balanced_equation': balanced_reaction.get('equation', ''),
                'stoichiometric_coefficients': balanced_reaction.get('coefficients', {}),
                'limiting_reagent': stoich_valid.get('limiting_reagent', ''),
                'theoretical_yield': stoich_valid.get('theoretical_yield', 0.0),
                'atom_balance': balanced_reaction.get('atom_balance', {}),
                'mass_balance': balanced_reaction.get('mass_balance', True),
                'delta_g_rxn': delta_g_rxn
            }
            
        except Exception as e:
            self.logger.debug(f"Reaction balancing validation failed for {compound.formula}: {e}")
            return {
                'valid': False,
                'error': str(e),
                'balanced_equation': '',
                'stoichiometric_coefficients': {},
                'limiting_reagent': '',
                'theoretical_yield': 0.0,
                'atom_balance': {},
                'mass_balance': False,
                'delta_g_rxn': None
            }
    
    def _parse_formula_composition(self, formula: str) -> Dict[str, int]:
        """Parse chemical formula to extract element composition.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Dictionary mapping elements to their counts
        """
        return self.utils.parse_formula_composition(formula)
    
    def _balance_formation_reaction(self, target_composition: Dict[str, int], 
                                  available_elements: Dict[str, float]) -> Dict[str, Any]:
        """Balance the formation reaction for the target compound.
        
        Args:
            target_composition: Target compound element composition
            available_elements: Available elements and quantities
            
        Returns:
            Balanced reaction information
        """
        return self.utils.balance_formation_reaction(target_composition, available_elements)
    
    def _verify_stoichiometric_requirements(self, target_composition: Dict[str, int],
                                          available_elements: Dict[str, float],
                                          balanced_reaction: Dict[str, Any]) -> Dict[str, Any]:
        """Verify stoichiometric requirements for the balanced reaction.
        
        Args:
            target_composition: Target compound composition
            available_elements: Available elements and quantities
            balanced_reaction: Balanced reaction information
            
        Returns:
            Stoichiometric validation results
        """
        return self.utils.verify_stoichiometric_requirements(
            target_composition, available_elements, balanced_reaction
        )
    
    def _calculate_reaction_gibbs_energy(self, product_formula: str, 
                                       reactant_coefficients: Dict[str, int]) -> Optional[float]:
        """Calculate reaction Gibbs free energy for formation reaction.
        
        Args:
            product_formula: Formula of the product compound
            reactant_coefficients: Dictionary of reactant elements and their coefficients
            
        Returns:
            Reaction Gibbs free energy in kJ/mol, or None if data unavailable
        """
        if not self.thermo_handler:
            return None
            
        try:
            return self.thermo_handler.calculate_reaction_gibbs_energy(
                product_formula, reactant_coefficients
            )
        except Exception as e:
            self.logger.debug(f"Failed to calculate ΔG_rxn for {product_formula}: {e}")
            return None
    
    def get_validation_summary(self, result: Dict[str, Any]) -> str:
        """Generate a human-readable validation summary.
        
        Args:
            result: Validation result dictionary
            
        Returns:
            Human-readable summary string
        """
        if not result.get('feasible', False):
            return f"Compound formation not feasible: {result.get('reason', 'Unknown reason')}"
        
        probability = result.get('formation_probability', 0.0)
        confidence = result.get('confidence', 0.0)
        
        summary = f"Compound formation feasible with {probability:.1%} probability "
        summary += f"(confidence: {confidence:.1%})\n"
        
        # Add key validation points
        if result.get('database_validation'):
            db_valid = sum(1 for v in result['database_validation'].values() 
                          if isinstance(v, dict) and v.get('valid', False))
            summary += f"Database validation: {db_valid} sources confirm feasibility\n"
        
        if result.get('chemical_rules', {}).get('valid'):
            summary += "Chemical rules: All validation checks passed\n"
        
        if result.get('environmental_validation'):
            env_score = result['environmental_validation'].get('overall_score', 0.0)
            summary += f"Environmental compatibility: {env_score:.1%}\n"
        
        return summary
    
    def clear_cache(self):
        """Clear the validation cache."""
        self.cache.clear()
        self.logger.info("Validation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }