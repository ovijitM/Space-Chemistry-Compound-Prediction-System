"""Formation probability calculator module for chemistry compounds.

This module contains calculators for enhanced formation probability,
thermodynamic analysis, and quantitative environmental factors.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from ..utils.chemistry_utils import ChemistryUtils
from ..config.constants import (
    FORMATION_PROBABILITY_WEIGHTS,
    COMPOUND_TYPE_ADJUSTMENTS,
    DATABASE_WEIGHTS,
    DATABASE_RELIABILITY,
    VALIDATION_THRESHOLDS,
    HUME_ROTHERY_PARAMETERS,
    ENVIRONMENTAL_PARAMETERS,
    KINETIC_PARAMETERS,
    STRUCTURAL_PARAMETERS,
    PRECISION_PARAMETERS,
    UNCERTAINTY_PARAMETERS
)


class FormationProbabilityCalculator:
    """Calculator for compound formation probability and thermodynamic analysis."""
    
    def __init__(self, thermo_data_handler=None, logger=None):
        """Initialize formation probability calculator.
        
        Args:
            thermo_data_handler: Thermodynamic data handler instance
            logger: Logger instance
        """
        self.thermo_data_handler = thermo_data_handler
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_enhanced_formation_probability(self, compound, 
                                               environmental_score: float,
                                               chemical_rules_score: float,
                                               db_validation: Dict) -> Dict:
        """Calculate enhanced formation probability with multiple factors.
        
        Args:
            compound: CompoundSuggestion object
            environmental_score: Environmental validation score
            chemical_rules_score: Chemical rules validation score
            db_validation: Database validation results
            
        Returns:
            Dictionary with formation probability analysis
        """
        try:
            # Base thermodynamic probability
            thermo_prob = self._calculate_thermodynamic_probability(compound)
            
            # Kinetic feasibility factor
            kinetic_factor = self._calculate_kinetic_feasibility(compound)
            
            # Structural stability factor
            structural_factor = self._calculate_structural_stability(compound)
            
            # Database evidence factor
            evidence_factor = self._calculate_evidence_factor(db_validation)
            
            # Environmental compatibility factor
            env_factor = min(1.2, environmental_score)
            
            # Chemical rules compliance factor
            rules_factor = min(1.2, chemical_rules_score)
            
            # Calculate weighted probability using centralized weights
            weighted_probability = (
                FORMATION_PROBABILITY_WEIGHTS['thermodynamic'] * thermo_prob +
                FORMATION_PROBABILITY_WEIGHTS['kinetic'] * kinetic_factor +
                FORMATION_PROBABILITY_WEIGHTS['structural'] * structural_factor +
                FORMATION_PROBABILITY_WEIGHTS['evidence'] * evidence_factor +
                FORMATION_PROBABILITY_WEIGHTS['environmental'] * env_factor +
                FORMATION_PROBABILITY_WEIGHTS['chemical_rules'] * rules_factor
            )
            
            # Apply compound type specific adjustments
            type_adjustment = self._get_compound_type_adjustment(compound.compound_type)
            final_probability = min(1.0, weighted_probability * type_adjustment)
            
            # Calculate confidence with uncertainty quantification
            confidence_data = self._calculate_confidence(
                db_validation, thermo_prob, kinetic_factor
            )
            
            return {
                'formation_probability': final_probability,  # Keep full precision
                'confidence': confidence_data['confidence'],  # Keep full precision
                'uncertainty': confidence_data['uncertainty'],
                'confidence_details': confidence_data,  # Full confidence analysis
                'factors': {
                    'thermodynamic': thermo_prob,
                    'kinetic': kinetic_factor,
                    'structural': structural_factor,
                    'evidence': evidence_factor,
                    'environmental': env_factor,
                    'chemical_rules': rules_factor
                },
                'weights': FORMATION_PROBABILITY_WEIGHTS,
                'type_adjustment': type_adjustment
            }
            
        except Exception as e:
            self.logger.error(f"Formation probability calculation failed for {compound.formula}: {e}")
            return {
                'formation_probability': 0.5,
                'confidence': 0.3,
                'factors': {},
                'error': str(e)
            }
    
    def _calculate_thermodynamic_probability(self, compound) -> float:
        """Calculate thermodynamic formation probability.
        
        Args:
            compound: CompoundSuggestion object
            
        Returns:
            Thermodynamic probability (0-1)
        """
        try:
            if self.thermo_data_handler:
                # Get thermodynamic data
                thermo_data = self.thermo_data_handler.get_compound_data(compound.formula)
                
                if thermo_data and 'formation_enthalpy' in thermo_data:
                    delta_h = thermo_data['formation_enthalpy']  # kJ/mol
                    
                    # Convert to probability using Boltzmann-like distribution
                    # More negative ΔH = higher probability
                    if delta_h < -200:  # Very favorable
                        return 0.95
                    elif delta_h < -100:  # Favorable
                        return 0.8 + 0.15 * (1 - abs(delta_h + 100) / 100)
                    elif delta_h < 0:  # Slightly favorable
                        return 0.6 + 0.2 * (1 - abs(delta_h) / 100)
                    elif delta_h < 100:  # Slightly unfavorable
                        return 0.4 * (1 - delta_h / 100)
                    else:  # Very unfavorable
                        return max(0.1, 0.4 * math.exp(-delta_h / 200))
                
                # If no formation enthalpy, try Gibbs free energy
                if thermo_data and 'gibbs_free_energy' in thermo_data:
                    delta_g = thermo_data['gibbs_free_energy']  # kJ/mol
                    
                    # Convert to probability
                    if delta_g < -150:
                        return 0.9
                    elif delta_g < -50:
                        return 0.7 + 0.2 * (1 - abs(delta_g + 50) / 100)
                    elif delta_g < 0:
                        return 0.5 + 0.2 * (1 - abs(delta_g) / 50)
                    elif delta_g < 50:
                        return 0.3 * (1 - delta_g / 50)
                    else:
                        return max(0.05, 0.3 * math.exp(-delta_g / 100))
            
            # Fallback: estimate based on compound properties
            return self._estimate_thermodynamic_probability(compound)
            
        except Exception as e:
            self.logger.debug(f"Thermodynamic probability calculation error: {e}")
            return self._estimate_thermodynamic_probability(compound)
    
    def _estimate_thermodynamic_probability(self, compound) -> float:
        """Estimate thermodynamic probability from compound properties."""
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Base probability by compound type
        base_probs = {
            'ionic': 0.7,      # Generally stable
            'molecular': 0.6,   # Moderate stability
            'metallic': 0.8,    # Usually stable
            'network': 0.9      # Very stable
        }
        
        base_prob = base_probs.get(compound.compound_type, 0.6)
        
        # Adjust based on electronegativity differences
        if len(elements) > 1:
            electronegativity_values = ChemistryUtils.get_electronegativity_values()
            electronegativities = [electronegativity_values.get(elem, 2.0) for elem in elements]
            
            if len(electronegativities) >= 2:
                max_diff = max(electronegativities) - min(electronegativities)
                
                if compound.compound_type == 'ionic':
                    # Large electronegativity difference favors ionic compounds
                    if max_diff > 2.0:
                        base_prob *= 1.2
                    elif max_diff > 1.5:
                        base_prob *= 1.1
                    elif max_diff < 0.5:
                        base_prob *= 0.8
                
                elif compound.compound_type == 'molecular':
                    # Moderate electronegativity difference favors molecular compounds
                    if 0.5 <= max_diff <= 1.5:
                        base_prob *= 1.1
                    elif max_diff > 2.5:
                        base_prob *= 0.7
        
        # Adjust for common vs. rare elements
        common_elements = {'H', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Fe'}
        rare_elements = set(elements.keys()) - common_elements
        
        if rare_elements:
            base_prob *= (0.9 ** len(rare_elements))
        
        # Enhanced metallic intermetallic heuristics with Hume-Rothery rules
        if compound.compound_type == 'metallic' and len(elements) >= 2:
            # Apply Hume-Rothery rules for metallic compound formation
            hume_rothery_results = ChemistryUtils.evaluate_hume_rothery_rules(elements)
            hume_rothery_score = hume_rothery_results.get('overall_score', 0.7)
            
            # Apply Hume-Rothery score with significant weight
            base_prob *= (0.7 + 0.3 * hume_rothery_score)  # Scale between 0.7-1.0
            
            # Traditional ratio and formula size effects (reduced weight due to Hume-Rothery)
            counts = list(elements.values())
            if all(c > 0 for c in counts):
                ratio = max(counts) / min(counts)
                # Common intermetallics often have modest ratios (e.g., AB, AB2)
                if ratio <= 2.0:
                    base_prob *= 1.02  # Reduced from 1.05
                elif ratio >= 8.0:
                    base_prob *= 0.95  # Reduced penalty from 0.9
            
            total_atoms = sum(elements.values())
            # Larger formula units can correlate with lower formation likelihood
            if total_atoms >= 10:
                base_prob *= 0.97  # Reduced penalty from 0.95
            if total_atoms >= 15:
                base_prob *= 0.93  # Reduced penalty from 0.9
        
        return min(1.0, base_prob)
    
    def _calculate_kinetic_feasibility(self, compound) -> float:
        """Calculate kinetic feasibility factor.
        
        Args:
            compound: CompoundSuggestion object
            
        Returns:
            Kinetic feasibility factor (0-1)
        """
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Base kinetic feasibility by compound type
        base_kinetic = {
            'ionic': 0.8,      # Fast ionic reactions
            'molecular': 0.6,   # Variable kinetics
            'metallic': 0.5,    # Slow diffusion processes
            'network': 0.4      # Very slow formation
        }
        
        kinetic_factor = base_kinetic.get(compound.compound_type, 0.6)
        
        # Adjust based on number of elements (complexity)
        num_elements = len(elements)
        if num_elements <= 2:
            kinetic_factor *= 1.2  # Simple compounds form easily
        elif num_elements <= 4:
            kinetic_factor *= 1.0  # Moderate complexity
        else:
            kinetic_factor *= (0.9 ** (num_elements - 4))  # Complex compounds harder to form
        
        # Adjust for specific element combinations
        if compound.compound_type == 'ionic':
            # Metal + nonmetal combinations are kinetically favorable
            metals = {'Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Fe', 'Cu', 'Zn'}
            nonmetals = {'F', 'Cl', 'Br', 'I', 'O', 'S', 'N'}
            
            has_metal = any(elem in metals for elem in elements)
            has_nonmetal = any(elem in nonmetals for elem in elements)
            
            if has_metal and has_nonmetal:
                kinetic_factor *= 1.2
        
        elif compound.compound_type == 'molecular':
            # Organic-like compounds with C-H bonds are kinetically accessible
            if 'C' in elements and 'H' in elements:
                kinetic_factor *= 1.1
            
            # Compounds with multiple bonds may be kinetically hindered
            total_atoms = sum(elements.values())
            if total_atoms > 10:
                kinetic_factor *= 0.9
        
        # Metallic formula size effect to differentiate intermetallics
        if compound.compound_type == 'metallic':
            total_atoms = sum(elements.values())
            if total_atoms > 4:
                # Mild penalty increasing with formula size
                kinetic_factor *= max(0.7, (0.98 ** (total_atoms - 4)))
        
        # Temperature-dependent kinetic effects (assume standard conditions)
        temperature_factor = 1.0  # Assume room temperature
        
        return min(1.0, kinetic_factor * temperature_factor)
    
    def _calculate_structural_stability(self, compound) -> float:
        """Calculate structural stability factor.
        
        Args:
            compound: CompoundSuggestion object
            
        Returns:
            Structural stability factor (0-1)
        """
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Base structural stability by compound type
        base_stability = {
            'ionic': 0.8,      # Crystal lattice stability
            'molecular': 0.6,   # Intermolecular forces
            'metallic': 0.9,    # Metallic bonding
            'network': 0.95     # Covalent network
        }
        
        stability_factor = base_stability.get(compound.compound_type, 0.7)
        
        # Enhanced structural stability for metallic compounds using Hume-Rothery rules
        if compound.compound_type == 'metallic' and len(elements) >= 2:
            # Apply Hume-Rothery rules for structural compatibility
            hume_rothery_results = ChemistryUtils.evaluate_hume_rothery_rules(elements)
            
            # Weight individual factors for structural stability
            size_factor = hume_rothery_results.get('size_factor', 0.7)
            crystal_factor = hume_rothery_results.get('crystal_structure_factor', 0.7)
            
            # Size factor is critical for structural stability
            stability_factor *= (0.8 + 0.2 * size_factor)
            
            # Crystal structure compatibility affects stability
            stability_factor *= (0.9 + 0.1 * crystal_factor)
        
        # Stoichiometry reasonableness (ratio-based, not availability)
        counts = list(elements.values()) if elements else []
        if counts:
            ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            if ratio > 20:
                stability_factor *= 0.85
            elif ratio > 10:
                stability_factor *= 0.9
            elif ratio <= 2.0 and len(elements) == 2 and compound.compound_type == 'metallic':
                # Slight bonus for simple binary intermetallic ratios (reduced due to Hume-Rothery)
                stability_factor *= 1.01  # Reduced from 1.02
        
        # Size mismatch effects for ionic compounds
        if compound.compound_type == 'ionic':
            # Simple size compatibility check
            ionic_radii = {
                'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Mg': 0.72, 'Ca': 1.00, 'Al': 0.54,
                'F': 1.33, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20, 'O': 1.40, 'S': 1.84
            }
            
            radii = [ionic_radii.get(elem, 1.0) for elem in elements]
            if len(radii) >= 2:
                size_ratio = max(radii) / min(radii)
                if size_ratio > 3.0:  # Very large size mismatch
                    stability_factor *= 0.8
                elif size_ratio > 2.0:  # Moderate size mismatch
                    stability_factor *= 0.9
        
        # Molecular geometry considerations
        elif compound.compound_type == 'molecular':
            # Estimate based on number of atoms and likely geometry
            total_atoms = sum(elements.values())
            
            if total_atoms <= 4:  # Small molecules, stable geometries
                stability_factor *= 1.1
            elif total_atoms <= 10:  # Medium molecules
                stability_factor *= 1.0
            else:  # Large molecules, potential strain
                stability_factor *= 0.9
            
            # Symmetry bonus (rough estimate)
            if len(set(elements.values())) == 1:  # All elements have same count
                stability_factor *= 1.05
        
        return min(1.0, stability_factor)
    
    def _calculate_evidence_factor(self, db_validation: Dict) -> float:
        """Calculate enhanced evidence factor using weighted reliability scoring.
        
        Args:
            db_validation: Database validation results
            
        Returns:
            Evidence factor (0-1) based on database reliability and data quality
        """
        if not db_validation:
            return VALIDATION_THRESHOLDS['evidence_factor_minimum']  # No evidence
        
        total_weighted_score = 0.0
        total_possible_score = 0.0
        databases_found = 0
        
        # Calculate weighted evidence score for each database
        for db_name, base_weight in DATABASE_WEIGHTS.items():
            db_key = f"{db_name}_found"
            reliability_config = DATABASE_RELIABILITY.get(db_name, {})
            
            if db_key in db_validation and db_validation[db_key]:
                databases_found += 1
                
                # Start with base reliability score
                base_reliability = reliability_config.get('base_reliability', 0.7)
                db_score = base_weight * base_reliability
                
                # Add data quality bonuses
                if db_name in db_validation and isinstance(db_validation[db_name], dict):
                    db_data = db_validation[db_name]
                    quality_bonuses = reliability_config.get('data_quality_bonuses', {})
                    
                    for data_field, bonus_weight in quality_bonuses.items():
                        if db_data.get(data_field) is not None:
                            # Scale bonus by base weight and data completeness
                            if isinstance(db_data[data_field], (int, float)):
                                # Numerical data gets full bonus
                                db_score += base_weight * bonus_weight * 0.3
                            elif isinstance(db_data[data_field], str) and db_data[data_field].strip():
                                # Non-empty string data gets partial bonus
                                db_score += base_weight * bonus_weight * 0.2
                            elif db_data[data_field] is True:
                                # Boolean true gets moderate bonus
                                db_score += base_weight * bonus_weight * 0.25
                
                total_weighted_score += db_score
            
            # Track maximum possible score for normalization
            max_reliability = reliability_config.get('base_reliability', 0.7)
            max_bonuses = sum(reliability_config.get('data_quality_bonuses', {}).values()) * 0.3
            total_possible_score += base_weight * (max_reliability + max_bonuses)
        
        # Normalize to 0-1 range
        if total_possible_score > 0:
            normalized_score = min(1.0, total_weighted_score / total_possible_score)
        else:
            normalized_score = 0.0
        
        # Apply multi-database confirmation bonus
        if databases_found >= 3:
            normalized_score *= 1.25  # Strong confirmation
        elif databases_found >= 2:
            normalized_score *= 1.15  # Moderate confirmation
        
        # Apply uncertainty reduction for high-quality databases
        high_quality_dbs = sum(1 for db in ['nist', 'materials_project'] 
                              if f"{db}_found" in db_validation and db_validation[f"{db}_found"])
        if high_quality_dbs >= 2:
            normalized_score *= 1.1  # Bonus for multiple high-quality sources
        
        return min(1.0, normalized_score)
    
    def _get_compound_type_adjustment(self, compound_type: str) -> float:
        """Get compound type specific adjustment factor.
        
        Args:
            compound_type: Type of compound
            
        Returns:
            Adjustment factor
        """
        return COMPOUND_TYPE_ADJUSTMENTS.get(compound_type, 1.0)
    
    def _calculate_confidence(self, db_validation: Dict, 
                            thermo_prob: float, kinetic_factor: float) -> Dict[str, float]:
        """Calculate confidence with uncertainty quantification and error propagation.
        
        Args:
            db_validation: Database validation results
            thermo_prob: Thermodynamic probability
            kinetic_factor: Kinetic feasibility factor
            
        Returns:
            Dictionary containing confidence score and uncertainty estimates
        """
        # Start with base uncertainty
        base_uncertainty = UNCERTAINTY_PARAMETERS['base_uncertainty']
        total_uncertainty = base_uncertainty
        
        # Calculate individual factor uncertainties
        uncertainties = {
            'database': self._calculate_database_uncertainty(db_validation),
            'thermodynamic': self._calculate_thermodynamic_uncertainty(thermo_prob),
            'kinetic': self._calculate_kinetic_uncertainty(kinetic_factor),
            'consistency': self._calculate_consistency_uncertainty(thermo_prob, kinetic_factor)
        }
        
        # Propagate uncertainties using root sum of squares for independent factors
        propagated_uncertainty = math.sqrt(
            sum(uncertainty**2 for uncertainty in uncertainties.values())
        )
        
        # Calculate confidence based on evidence and consistency
        evidence_factor = self._calculate_evidence_factor(db_validation)
        base_confidence = 0.5 + 0.3 * evidence_factor
        
        # Adjust confidence based on data availability
        if self.thermo_data_handler:
            base_confidence += 0.1
            
        # Consistency bonus
        consistency_factor = 1.0 - abs(thermo_prob - kinetic_factor)
        base_confidence += 0.1 * consistency_factor
        
        # Extreme value confidence (high certainty at extremes)
        if thermo_prob > 0.8 or thermo_prob < 0.2:
            base_confidence += 0.05
            propagated_uncertainty *= 0.9  # Reduce uncertainty for extreme values
        
        # Apply uncertainty scaling
        confidence_scaling = UNCERTAINTY_PARAMETERS['confidence_scaling']
        final_confidence = base_confidence * confidence_scaling
        
        # Ensure uncertainty doesn't exceed maximum
        max_uncertainty = UNCERTAINTY_PARAMETERS['max_uncertainty']
        final_uncertainty = min(propagated_uncertainty, max_uncertainty)
        
        # Confidence should be inversely related to uncertainty
        uncertainty_penalty = final_uncertainty * 0.5
        adjusted_confidence = max(0.1, final_confidence - uncertainty_penalty)
        
        return {
            'confidence': min(1.0, adjusted_confidence),
            'uncertainty': final_uncertainty,
            'factor_uncertainties': uncertainties,
            'evidence_factor': evidence_factor
        }
    
    def _calculate_database_uncertainty(self, db_validation: Dict) -> float:
        """Calculate uncertainty contribution from database evidence.
        
        Args:
            db_validation: Database validation results
            
        Returns:
            Database uncertainty factor
        """
        if not db_validation:
            return UNCERTAINTY_PARAMETERS['max_uncertainty']
        
        # Count available databases
        available_dbs = sum(1 for db_name in DATABASE_WEIGHTS.keys() 
                           if f"{db_name}_found" in db_validation and db_validation[f"{db_name}_found"])
        
        # Base uncertainty decreases with more databases
        base_db_uncertainty = UNCERTAINTY_PARAMETERS['database_uncertainty']
        if available_dbs == 0:
            return base_db_uncertainty * 4  # High uncertainty with no database evidence
        elif available_dbs == 1:
            return base_db_uncertainty * 2  # Moderate uncertainty with single source
        elif available_dbs >= 3:
            return base_db_uncertainty * 0.5  # Low uncertainty with multiple sources
        else:
            return base_db_uncertainty  # Standard uncertainty with two sources
    
    def _calculate_thermodynamic_uncertainty(self, thermo_prob: float) -> float:
        """Calculate uncertainty contribution from thermodynamic factors.
        
        Args:
            thermo_prob: Thermodynamic probability
            
        Returns:
            Thermodynamic uncertainty factor
        """
        base_uncertainty = UNCERTAINTY_PARAMETERS['base_uncertainty']
        
        # Uncertainty is higher for intermediate values (maximum at 0.5)
        # and lower for extreme values (minimum at 0.0 and 1.0)
        prob_uncertainty = 4 * thermo_prob * (1 - thermo_prob)  # Parabolic: max at 0.5
        
        # Scale by base uncertainty
        return base_uncertainty * (0.5 + prob_uncertainty)
    
    def _calculate_kinetic_uncertainty(self, kinetic_factor: float) -> float:
        """Calculate uncertainty contribution from kinetic factors.
        
        Args:
            kinetic_factor: Kinetic feasibility factor
            
        Returns:
            Kinetic uncertainty factor
        """
        base_uncertainty = UNCERTAINTY_PARAMETERS['base_uncertainty']
        
        # Similar to thermodynamic uncertainty
        kinetic_uncertainty = 4 * kinetic_factor * (1 - kinetic_factor)
        
        # Kinetic factors generally have higher uncertainty than thermodynamic
        return base_uncertainty * (0.7 + kinetic_uncertainty)
    
    def _calculate_consistency_uncertainty(self, thermo_prob: float, kinetic_factor: float) -> float:
        """Calculate uncertainty contribution from factor consistency.
        
        Args:
            thermo_prob: Thermodynamic probability
            kinetic_factor: Kinetic feasibility factor
            
        Returns:
            Consistency uncertainty factor
        """
        base_uncertainty = UNCERTAINTY_PARAMETERS['base_uncertainty']
        
        # Inconsistency between factors increases uncertainty
        inconsistency = abs(thermo_prob - kinetic_factor)
        
        # Scale inconsistency to uncertainty contribution
        consistency_uncertainty = inconsistency * 2  # Linear scaling
        
        return base_uncertainty * consistency_uncertainty
    
    def calculate_quantitative_environmental_factor(self, compound, conditions) -> Dict:
        """Calculate quantitative environmental compatibility factor.
        
        Args:
            compound: CompoundSuggestion object
            conditions: EnvironmentalConditions object
            
        Returns:
            Dictionary with quantitative environmental analysis
        """
        try:
            # Temperature factor with quantitative analysis
            temp_factor = self._calculate_temperature_factor(
                compound, conditions.temperature
            )
            
            # Pressure factor with phase considerations
            pressure_factor = self._calculate_pressure_factor(
                compound, conditions.pressure, conditions.temperature
            )
            
            # Atmospheric compatibility with chemical interactions
            atmosphere_factor = self._calculate_atmosphere_factor(
                compound, conditions.atmosphere, conditions.temperature
            )
            
            # pH effects with speciation analysis
            ph_factor = self._calculate_ph_factor(
                compound, conditions.pH, conditions.temperature
            )
            
            # Combined environmental factor using configured aggregation method
            factors = [temp_factor, pressure_factor, atmosphere_factor, ph_factor]
            
            if ENVIRONMENTAL_PARAMETERS.get('aggregation_method') == 'geometric_mean':
                # Geometric mean: (a * b * c * d)^(1/n)
                combined_factor = math.pow(
                    temp_factor * pressure_factor * atmosphere_factor * ph_factor,
                    1.0 / len(factors)
                )
            else:
                # Fallback to arithmetic multiplication (original behavior)
                combined_factor = (
                    temp_factor * pressure_factor * atmosphere_factor * ph_factor
                )
            
            # Uncertainty analysis
            uncertainty = self._calculate_environmental_uncertainty(
                temp_factor, pressure_factor, atmosphere_factor, ph_factor
            )
            
            return {
                'environmental_factor': combined_factor,
                'uncertainty': uncertainty,
                'components': {
                    'temperature': temp_factor,
                    'pressure': pressure_factor,
                    'atmosphere': atmosphere_factor,
                    'pH': ph_factor
                },
                'conditions': {
                    'temperature': conditions.temperature,
                    'pressure': conditions.pressure,
                    'atmosphere': conditions.atmosphere,
                    'pH': conditions.pH
                }
            }
            
        except Exception as e:
            self.logger.error(f"Environmental factor calculation failed: {e}")
            return {
                'environmental_factor': 0.5,
                'uncertainty': 0.5,
                'error': str(e)
            }
    
    def _calculate_temperature_factor(self, compound, temperature: float) -> float:
        """Calculate quantitative temperature factor."""
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Get thermal stability data
        if self.thermo_data_handler:
            try:
                thermo_data = self.thermo_data_handler.get_compound_data(compound.formula)
                if thermo_data:
                    # Use actual thermal data if available
                    melting_point = thermo_data.get('melting_point')
                    boiling_point = thermo_data.get('boiling_point')
                    
                    if melting_point is not None:
                        if temperature < melting_point - 50:
                            return 1.0  # Solid, stable
                        elif temperature < melting_point:
                            return 0.9  # Near melting
                        elif boiling_point and temperature < boiling_point:
                            return 0.7  # Liquid phase
                        else:
                            return 0.3  # Gas phase or decomposed
            except Exception as e:
                self.logger.debug(f"Thermal data lookup failed: {e}")
        
        # Fallback to estimated thermal stability
        stability_ranges = ChemistryUtils.get_stability_ranges()
        min_temp, max_temp = stability_ranges.get(compound.compound_type, (-100, 500))
        
        if min_temp <= temperature <= max_temp:
            # Within stability range - calculate position factor
            range_center = (min_temp + max_temp) / 2
            range_width = max_temp - min_temp
            deviation = abs(temperature - range_center) / (range_width / 2)
            
            return max(0.7, 1.0 - 0.3 * deviation)
        
        elif temperature < min_temp:
            # Below stability range
            return max(0.2, 1.0 - (min_temp - temperature) / 100)
        
        else:  # temperature > max_temp
            # Above stability range
            return max(0.1, 1.0 - (temperature - max_temp) / 200)
    
    def _calculate_pressure_factor(self, compound, pressure: float, temperature: float) -> float:
        """Calculate quantitative pressure factor with phase considerations."""
        # Phase-dependent pressure effects
        if compound.compound_type == 'molecular':
            # Molecular compounds affected by vapor pressure
            if pressure < 0.001:  # Near vacuum
                return 0.2
            elif pressure < 0.1:  # Low pressure
                return 0.6
            elif 0.1 <= pressure <= 10:  # Normal range
                return 1.0
            elif pressure <= 100:  # High pressure
                return 0.9
            else:  # Very high pressure
                return 0.7
        
        elif compound.compound_type == 'ionic':
            # Ionic compounds generally benefit from pressure
            if pressure < 0.001:
                return 0.3
            elif pressure < 1:
                return 0.8
            else:
                return min(1.1, 1.0 + 0.05 * math.log10(pressure))
        
        elif compound.compound_type in ['metallic', 'network']:
            # Solid compounds stable over wide pressure range
            if pressure < 0.0001:
                return 0.4  # Vacuum effects
            else:
                return min(1.0, 0.9 + 0.1 * math.log10(max(0.001, pressure)))
        
        return 0.8
    
    def _calculate_atmosphere_factor(self, compound, atmosphere: str, temperature: float) -> float:
        """Calculate quantitative atmosphere factor with chemical interactions."""
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Base atmospheric compatibility
        base_factors = {
            'vacuum': 0.5,
            'nitrogen': 0.9,
            'argon': 0.95,
            'air': 0.8,
            'oxygen': 0.6,  # Oxidizing
            'hydrogen': 0.5,  # Reducing
            'CO2': 0.7,
            'water_vapor': 0.6
        }
        
        base_factor = base_factors.get(atmosphere.lower(), 0.7)
        
        # Chemical interaction adjustments
        if atmosphere.lower() == 'oxygen':
            # Oxidation susceptibility
            oxidizable = {'Fe', 'Cu', 'Al', 'Mg', 'Zn', 'Mn'}
            if any(elem in oxidizable for elem in elements):
                # Temperature-dependent oxidation
                if temperature > 200:
                    base_factor *= 0.5
                elif temperature > 100:
                    base_factor *= 0.7
        
        elif atmosphere.lower() == 'hydrogen':
            # Reduction susceptibility
            reducible = {'Fe', 'Cu', 'Ni', 'Co'}
            if any(elem in reducible for elem in elements):
                if temperature > 300:
                    base_factor *= 0.6
        
        elif atmosphere.lower() == 'CO2':
            # Carbonate formation potential
            if compound.compound_type == 'ionic' and any(elem in ['Ca', 'Mg', 'Na', 'K'] for elem in elements):
                base_factor *= 0.8  # May form carbonates
        
        return base_factor
    
    def _calculate_ph_factor(self, compound, ph: float, temperature: float) -> float:
        """Calculate quantitative pH factor with speciation analysis."""
        if ph == 7.0:  # Neutral pH
            return 1.0
        
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # pH-dependent solubility and stability
        if compound.compound_type == 'ionic':
            # Metal hydroxides and oxides
            metals = {'Al', 'Fe', 'Cr', 'Zn', 'Cu', 'Ni', 'Mn'}
            if any(elem in metals for elem in elements):
                if ph < 3:  # Strong acid
                    return 0.2  # Dissolution
                elif ph < 5:  # Weak acid
                    return 0.5
                elif 6 <= ph <= 8:  # Near neutral
                    return 1.0
                elif ph <= 10:  # Weak base
                    return 0.9
                else:  # Strong base
                    return 0.7
            
            # Carbonates
            if 'C' in elements and 'O' in elements:
                if ph < 5:
                    return 0.1  # CO2 evolution
                elif ph < 7:
                    return 0.4
                else:
                    return 1.0
        
        # Temperature effects on pH stability
        temp_factor = 1.0
        if temperature > 100:
            temp_factor = 0.9  # High temperature reduces pH effects
        elif temperature < 0:
            temp_factor = 0.95  # Low temperature reduces reaction rates
        
        # Default pH factor for molecular compounds
        if abs(ph - 7) > 4:  # Extreme pH
            return 0.6 * temp_factor
        elif abs(ph - 7) > 2:  # Moderate pH
            return 0.8 * temp_factor
        else:  # Near neutral
            return 0.95 * temp_factor
    
    def _calculate_environmental_uncertainty(self, temp_factor: float, 
                                           pressure_factor: float,
                                           atmosphere_factor: float, 
                                           ph_factor: float) -> float:
        """Calculate uncertainty in environmental factor prediction."""
        # Base uncertainty
        uncertainty = 0.1
        
        # Higher uncertainty for extreme factors
        factors = [temp_factor, pressure_factor, atmosphere_factor, ph_factor]
        
        for factor in factors:
            if factor < 0.3 or factor > 1.1:
                uncertainty += 0.1  # Extreme conditions increase uncertainty
            elif factor < 0.5 or factor > 0.9:
                uncertainty += 0.05  # Moderate conditions
        
        # Uncertainty from factor consistency
        factor_std = math.sqrt(sum((f - sum(factors)/len(factors))**2 for f in factors) / len(factors))
        uncertainty += factor_std * 0.5
        
        return min(0.8, uncertainty)