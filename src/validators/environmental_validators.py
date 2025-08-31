"""Environmental validation module for chemistry compounds.

This module contains validators for environmental conditions including
temperature, pressure, pH, atmospheric effects, and redox constraints.
"""

import logging
import math
from typing import Dict, Tuple, Optional, Any
from ..utils.chemistry_utils import ChemistryUtils
from ..config.constants import ENVIRONMENTAL_PARAMETERS


class EnvironmentalValidator:
    """Validator for environmental conditions and their effects on compounds."""
    
    def __init__(self, phase_equilibria=None, logger=None):
        """Initialize environmental validator.
        
        Args:
            phase_equilibria: Phase equilibria system instance
            logger: Logger instance
        """
        self.phase_equilibria = phase_equilibria
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_environmental_conditions(self, compound, conditions) -> Dict:
        """Enhanced environmental validation for planetary conditions.
        
        Args:
            compound: CompoundSuggestion object
            conditions: EnvironmentalConditions object
            
        Returns:
            Dictionary with environmental validation results
        """
        factors = {}
        factor_values = []
        
        # Temperature effects
        temp_factor, temp_suitable = self.assess_temperature_effects(
            compound, conditions.temperature
        )
        factor_values.append(temp_factor)
        factors['temperature'] = {
            'suitable': temp_suitable,
            'factor': temp_factor,
            'value': conditions.temperature
        }
        
        # Pressure effects
        pressure_factor, pressure_suitable = self.assess_pressure_effects(
            compound, conditions.pressure
        )
        factor_values.append(pressure_factor)
        factors['pressure'] = {
            'suitable': pressure_suitable,
            'factor': pressure_factor,
            'value': conditions.pressure
        }
        
        # Atmospheric effects
        atmo_factor, atmo_suitable = self.assess_atmospheric_effects(
            compound, conditions.atmosphere
        )
        factor_values.append(atmo_factor)
        factors['atmosphere'] = {
            'suitable': atmo_suitable,
            'factor': atmo_factor,
            'value': conditions.atmosphere
        }
        
        # pH effects (for relevant conditions)
        if conditions.pH != 7.0:
            ph_factor, ph_suitable = self.assess_ph_effects(compound, conditions.pH)
            factor_values.append(ph_factor)
            factors['pH'] = {
                'suitable': ph_suitable,
                'factor': ph_factor,
                'value': conditions.pH
            }
        
        # Redox/Eh constraints
        redox_factor, redox_suitable = self.assess_redox_constraints(
            compound, conditions.atmosphere, conditions.pH
        )
        factor_values.append(redox_factor)
        factors['redox'] = {
            'suitable': redox_suitable,
            'factor': redox_factor,
            'atmosphere': conditions.atmosphere
        }
        
        # Gas solubility and pressure constraints
        solubility_factor, solubility_suitable = self.assess_gas_solubility_constraints(
            compound, conditions.pressure, conditions.temperature, conditions.atmosphere
        )
        factor_values.append(solubility_factor)
        factors['gas_solubility'] = {
            'suitable': solubility_suitable,
            'factor': solubility_factor,
            'pressure': conditions.pressure,
            'temperature': conditions.temperature
        }
        
        # Temperature stability windows
        temp_stability_factor, temp_stability_suitable = self.assess_temperature_stability_windows(
            compound, conditions.temperature
        )
        factor_values.append(temp_stability_factor)
        factors['temperature_stability'] = {
            'suitable': temp_stability_suitable,
            'factor': temp_stability_factor,
            'temperature': conditions.temperature
        }
        
        # Calculate overall score using configured aggregation method
        if ENVIRONMENTAL_PARAMETERS.get('aggregation_method') == 'geometric_mean':
            # Geometric mean: (a * b * c * ...)^(1/n)
            product = 1.0
            for factor in factor_values:
                product *= factor
            score = math.pow(product, 1.0 / len(factor_values))
        else:
            # Fallback to arithmetic multiplication (original behavior)
            score = 1.0
            for factor in factor_values:
                score *= factor
        
        return {
            'overall_score': score,
            'factors': factors,
            'planetary_suitable': score > 0.3
        }
    
    def assess_temperature_effects(self, compound, temperature: float) -> Tuple[float, bool]:
        """Assess temperature effects on compound stability using phase equilibria.
        
        Args:
            compound: CompoundSuggestion object
            temperature: Temperature in Celsius
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        if self.phase_equilibria:
            try:
                return self.calculate_phase_equilibria_effects(compound, temperature, None)
            except Exception as e:
                self.logger.debug(f"Phase equilibria calculation failed for {compound.formula}: {e}")
        
        # Fallback to basic temperature assessment
        stability_ranges = ChemistryUtils.get_stability_ranges()
        
        min_temp, max_temp = stability_ranges.get(compound.compound_type, (-100, 500))
        
        if min_temp <= temperature <= max_temp:
            if -50 <= temperature <= 100:  # Optimal range
                return 1.1, True
            return 1.0, True
        elif temperature < min_temp:
            factor = max(0.2, 1.0 - (min_temp - temperature) / 100)
            return factor, False
        else:  # temperature > max_temp
            factor = max(0.1, 1.0 - (temperature - max_temp) / 200)
            return factor, False
    
    def assess_pressure_effects(self, compound, pressure: float) -> Tuple[float, bool]:
        """Assess pressure effects on compound stability using phase equilibria.
        
        Args:
            compound: CompoundSuggestion object
            pressure: Pressure in atm
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        if self.phase_equilibria:
            try:
                return self.calculate_phase_equilibria_effects(compound, None, pressure)
            except Exception as e:
                self.logger.debug(f"Phase equilibria calculation failed for {compound.formula}: {e}")
        
        # Fallback to basic pressure assessment
        if compound.compound_type == 'molecular' and pressure < 0.001:
            return 0.2, False  # Molecular compounds need some pressure
        elif pressure < 0.0001:  # Near vacuum
            return 0.3, False
        elif 0.1 <= pressure <= 10:  # Reasonable range
            return 1.0, True
        elif pressure > 50:  # Very high pressure
            return 0.7, False
        else:
            return 0.8, True
    
    def assess_atmospheric_effects(self, compound, atmosphere: str) -> Tuple[float, bool]:
        """Assess atmospheric compatibility.
        
        Args:
            compound: CompoundSuggestion object
            atmosphere: Atmospheric composition
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        atmosphere_effects = {
            'vacuum': 0.5,
            'CO2': 0.8,
            'nitrogen': 0.9,
            'argon': 0.9,
            'air': 1.0,
            'oxygen': 0.7,  # Oxidizing
            'hydrogen': 0.6  # Reducing
        }
        
        factor = atmosphere_effects.get(atmosphere.lower(), 0.7)
        
        # Special cases for compound types
        if compound.compound_type == 'molecular' and atmosphere == 'vacuum':
            factor *= 0.5
        elif compound.compound_type == 'ionic' and atmosphere in ['vacuum', 'CO2']:
            factor *= 1.1  # Ionic compounds more stable in these conditions
        
        return factor, factor > 0.6
    
    def assess_ph_effects(self, compound, ph: float) -> Tuple[float, bool]:
        """Enhanced pH effects assessment with compound-specific rules.
        
        Args:
            compound: CompoundSuggestion object
            ph: pH value
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # pH-dependent stability rules based on compound composition
        if compound.compound_type == 'ionic':
            # Metal hydroxides and oxides
            if any(elem in ['Al', 'Fe', 'Cr', 'Zn', 'Cu', 'Ni'] for elem in elements):
                if ph < 4:  # Acidic conditions dissolve many metal compounds
                    return 0.3, False
                elif 6 <= ph <= 9:
                    return 1.0, True
                else:
                    return 0.7, True
            
            # Carbonates and bicarbonates
            if 'C' in elements and 'O' in elements:
                if ph < 6:  # Carbonates dissolve in acidic conditions
                    return 0.2, False
                elif ph > 10:
                    return 1.1, True  # More stable in alkaline conditions
                else:
                    return 0.8, True
            
            # Sulfates and phosphates
            if 'S' in elements or 'P' in elements:
                if 2 <= ph <= 12:  # Generally stable across wide pH range
                    return 1.0, True
                else:
                    return 0.6, False
            
            # General ionic compounds
            if 6 <= ph <= 8:  # Neutral range
                return 1.0, True
            elif ph < 3 or ph > 11:  # Extreme pH
                return 0.4, False
            else:
                return 0.7, True
        
        elif compound.compound_type == 'molecular':
            # Organic acids and bases
            if 'N' in elements:  # Potential amines/amides
                if ph > 9:  # Basic conditions favor nitrogen compounds
                    return 1.1, True
                elif ph < 5:
                    return 0.8, True
                else:
                    return 0.9, True
            
            # Carboxylic acids and esters
            if 'C' in elements and 'O' in elements:
                if ph < 6:  # Acidic conditions
                    return 0.9, True
                elif ph > 10:
                    return 0.7, True
                else:
                    return 1.0, True
            
            # Most other molecular compounds less affected by pH
            return 0.9, True
        
        else:
            # Metallic and network compounds generally pH-independent
            return 0.95, True
    
    def assess_redox_constraints(self, compound, atmosphere: str, ph: float) -> Tuple[float, bool]:
        """Assess redox/Eh constraints on compound stability.
        
        Args:
            compound: CompoundSuggestion object
            atmosphere: Atmospheric composition
            ph: pH value
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Define redox-sensitive elements and their preferred conditions
        oxidizing_atmospheres = ['oxygen', 'air']
        reducing_atmospheres = ['hydrogen', 'CO']
        neutral_atmospheres = ['nitrogen', 'argon', 'CO2']
        
        # Redox-sensitive elements and their oxidation preferences
        redox_sensitive = {
            'Fe': {'oxidizing': [2, 3], 'reducing': [0, 2], 'neutral': [2, 3]},
            'Cu': {'oxidizing': [1, 2], 'reducing': [0, 1], 'neutral': [1, 2]},
            'Mn': {'oxidizing': [2, 3, 4], 'reducing': [0, 2], 'neutral': [2, 3]},
            'Cr': {'oxidizing': [3, 6], 'reducing': [0, 2, 3], 'neutral': [3]},
            'V': {'oxidizing': [4, 5], 'reducing': [2, 3], 'neutral': [3, 4]},
            'Ti': {'oxidizing': [4], 'reducing': [2, 3], 'neutral': [3, 4]},
            'Ni': {'oxidizing': [2, 3], 'reducing': [0, 2], 'neutral': [2]},
            'Co': {'oxidizing': [2, 3], 'reducing': [0, 2], 'neutral': [2, 3]}
        }
        
        # Check for redox-sensitive elements
        redox_elements = [elem for elem in elements if elem in redox_sensitive]
        
        if not redox_elements:
            return 1.0, True  # No redox constraints
        
        # Assess atmosphere compatibility
        atmosphere_type = 'neutral'
        if atmosphere.lower() in oxidizing_atmospheres:
            atmosphere_type = 'oxidizing'
        elif atmosphere.lower() in reducing_atmospheres:
            atmosphere_type = 'reducing'
        
        # Calculate redox compatibility score
        compatibility_score = 1.0
        
        for elem in redox_elements:
            # pH affects redox potential (Nernst equation considerations)
            if atmosphere_type == 'oxidizing':
                if ph < 4:  # Acidic + oxidizing = very oxidizing
                    compatibility_score *= 0.8 if elem in ['Fe', 'Cu', 'Mn'] else 1.0
                elif ph > 10:  # Basic + oxidizing = moderate
                    compatibility_score *= 0.9
            elif atmosphere_type == 'reducing':
                if ph > 10:  # Basic + reducing = very reducing
                    compatibility_score *= 0.7 if elem in ['Cr', 'V', 'Ti'] else 0.9
                elif ph < 4:  # Acidic + reducing = moderate
                    compatibility_score *= 0.85
        
        # Special cases for specific compound types
        if compound.compound_type == 'ionic':
            # Ionic compounds with multiple oxidation states
            if any(elem in ['Fe', 'Cu', 'Mn', 'Cr'] for elem in redox_elements):
                if atmosphere_type == 'oxidizing' and ph < 6:
                    compatibility_score *= 0.7  # May dissolve or change oxidation state
        
        return compatibility_score, compatibility_score > 0.6
    
    def assess_gas_solubility_constraints(self, compound, pressure: float, 
                                        temperature: float, atmosphere: str) -> Tuple[float, bool]:
        """Assess gas solubility and pressure constraints.
        
        Args:
            compound: CompoundSuggestion object
            pressure: Pressure in atm
            temperature: Temperature in Celsius
            atmosphere: Atmospheric composition
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Gas solubility considerations
        if compound.compound_type == 'molecular':
            # Molecular compounds may have gas solubility issues
            if pressure < 0.01:  # Very low pressure
                # Volatile compounds may sublimate/evaporate
                volatile_elements = ['H', 'He', 'N', 'O', 'F', 'Ne', 'Cl', 'Ar']
                if any(elem in volatile_elements for elem in elements):
                    return 0.3, False
            
            # Henry's law considerations for gas dissolution
            if atmosphere in ['CO2', 'oxygen', 'hydrogen']:
                # Gas-liquid equilibrium affects stability
                if 0.1 <= pressure <= 10 and -10 <= temperature <= 100:
                    return 1.0, True
                elif pressure > 50:  # High pressure may force dissolution
                    return 0.8, True
                else:
                    return 0.6, False
        
        # Pressure effects on ionic compounds
        elif compound.compound_type == 'ionic':
            if pressure < 0.001:  # Near vacuum
                # Ionic compounds may decompose in vacuum
                return 0.4, False
            elif pressure > 100:  # Very high pressure
                # May affect crystal structure
                return 0.8, True
            else:
                return 1.0, True
        
        # Gas phase stability
        if atmosphere == 'vacuum':
            # Only very stable compounds survive in vacuum
            if compound.compound_type in ['network', 'metallic']:
                return 0.9, True
            else:
                return 0.4, False
        
        return 0.9, True
    
    def assess_temperature_stability_windows(self, compound, temperature: float) -> Tuple[float, bool]:
        """Assess compound stability within specific temperature windows.
        
        Args:
            compound: CompoundSuggestion object
            temperature: Temperature in Celsius
            
        Returns:
            Tuple of (stability_factor, is_suitable)
        """
        elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
        
        # Define temperature stability windows for different compound types
        stability_windows = {
            'ionic': {
                'oxides': (-200, 1200),
                'halides': (-100, 800),
                'sulfates': (-50, 600),
                'carbonates': (-50, 400),  # Decompose at higher temps
                'nitrates': (-50, 300),   # Thermally unstable
                'hydroxides': (0, 200),   # Dehydrate at higher temps
                'default': (-100, 600)
            },
            'molecular': {
                'hydrocarbons': (-100, 200),
                'alcohols': (-80, 150),
                'organic_acids': (-50, 180),
                'amines': (-60, 120),
                'default': (-80, 150)
            },
            'metallic': {
                'pure_metals': (-200, 1500),
                'alloys': (-200, 1200),
                'default': (-200, 1000)
            },
            'network': {
                'ceramics': (-200, 1800),
                'silicates': (-200, 1400),
                'default': (-200, 1200)
            }
        }
        
        # Determine compound subtype
        compound_subtype = 'default'
        
        if compound.compound_type == 'ionic':
            if 'O' in elements and len(elements) == 2:
                compound_subtype = 'oxides'
            elif any(elem in ['F', 'Cl', 'Br', 'I'] for elem in elements):
                compound_subtype = 'halides'
            elif 'S' in elements and 'O' in elements:
                compound_subtype = 'sulfates'
            elif 'C' in elements and 'O' in elements:
                compound_subtype = 'carbonates'
            elif 'N' in elements and 'O' in elements:
                compound_subtype = 'nitrates'
            elif 'H' in elements and 'O' in elements:
                compound_subtype = 'hydroxides'
        
        elif compound.compound_type == 'molecular':
            if 'C' in elements and 'H' in elements and len(elements) == 2:
                compound_subtype = 'hydrocarbons'
            elif 'C' in elements and 'H' in elements and 'O' in elements:
                if 'N' in elements:
                    compound_subtype = 'amines'
                else:
                    compound_subtype = 'alcohols' if 'COOH' not in compound.formula else 'organic_acids'
        
        elif compound.compound_type == 'network':
            if 'Si' in elements and 'O' in elements:
                compound_subtype = 'silicates'
            elif any(elem in ['Al', 'Mg', 'Ca'] for elem in elements) and 'O' in elements:
                compound_subtype = 'ceramics'
        
        # Get stability window
        min_temp, max_temp = stability_windows[compound.compound_type][compound_subtype]
        
        # Calculate stability factor
        if min_temp <= temperature <= max_temp:
            # Within stability window
            optimal_range = (min_temp + max_temp) / 2
            window_size = max_temp - min_temp
            deviation = abs(temperature - optimal_range)
            
            if deviation <= window_size * 0.2:  # Within 20% of optimal
                return 1.1, True
            elif deviation <= window_size * 0.4:  # Within 40% of optimal
                return 1.0, True
            else:
                return 0.9, True
        
        elif temperature < min_temp:
            # Below stability window
            factor = max(0.2, 1.0 - (min_temp - temperature) / 100)
            return factor, factor > 0.5
        
        else:  # temperature > max_temp
            # Above stability window - thermal decomposition risk
            factor = max(0.1, 1.0 - (temperature - max_temp) / 200)
            return factor, factor > 0.4
    
    def calculate_phase_equilibria_effects(self, compound, 
                                         temperature: Optional[float] = None,
                                         pressure: Optional[float] = None) -> Tuple[float, bool]:
        """Calculate phase equilibria effects on compound stability.
        
        Args:
            compound: CompoundSuggestion object
            temperature: Temperature in Celsius (optional)
            pressure: Pressure in atm (optional)
            
        Returns:
            Tuple of (stability_factor, is_favorable)
        """
        try:
            # Convert temperature to Kelvin if provided
            temp_k = (temperature + 273.15) if temperature is not None else 298.15
            # Convert pressure to Pa if provided
            pressure_pa = (pressure * 101325) if pressure is not None else 101325
            
            # Parse compound formula to get components
            elements = ChemistryUtils.extract_elements_from_formula(compound.formula)
            
            # For single component systems
            if len(elements) == 1:
                element = list(elements.keys())[0]
                stability_factor = self._assess_single_component_stability(
                    element, temp_k, pressure_pa, compound.compound_type
                )
            else:
                # For multi-component systems
                stability_factor = self._assess_multicomponent_stability(
                    elements, temp_k, pressure_pa, compound.compound_type
                )
            
            # Determine if conditions are favorable
            is_favorable = stability_factor > 0.6
            
            return stability_factor, is_favorable
            
        except Exception as e:
            self.logger.debug(f"Phase equilibria calculation error: {e}")
            # Return neutral values on error
            return 0.8, True
    
    def _assess_single_component_stability(self, element: str, temperature: float, 
                                         pressure: float, compound_type: str) -> float:
        """Assess stability for single component using phase equilibria."""
        try:
            # Use phase equilibria system to calculate phase stability
            component_data = {
                'name': element,
                'critical_temperature': self._get_critical_temperature(element),
                'critical_pressure': self._get_critical_pressure(element),
                'acentric_factor': self._get_acentric_factor(element)
            }
            
            # Calculate reduced temperature and pressure
            tr = temperature / component_data['critical_temperature']
            pr = pressure / component_data['critical_pressure']
            
            # Phase stability assessment based on reduced conditions
            if compound_type == 'molecular':
                # Molecular compounds prefer moderate conditions
                if 0.5 <= tr <= 1.2 and 0.1 <= pr <= 5.0:
                    return 1.0
                elif tr > 1.5 or pr > 10.0:
                    return max(0.3, 1.0 - 0.1 * (tr - 1.2) - 0.05 * (pr - 5.0))
                else:
                    return max(0.4, 0.8 + 0.2 * tr)
            
            elif compound_type == 'ionic':
                # Ionic compounds more stable at higher pressures
                if tr <= 2.0 and pr >= 0.5:
                    return min(1.1, 0.8 + 0.1 * pr)
                else:
                    return max(0.2, 1.0 - 0.2 * max(0, tr - 2.0))
            
            elif compound_type == 'metallic':
                # Metallic compounds stable over wide range
                if tr <= 3.0:
                    return min(1.0, 0.9 + 0.05 * pr)
                else:
                    return max(0.3, 1.0 - 0.1 * (tr - 3.0))
            
            else:  # network compounds
                # Network compounds very stable
                if tr <= 4.0:
                    return 1.0
                else:
                    return max(0.5, 1.0 - 0.05 * (tr - 4.0))
                    
        except Exception as e:
            self.logger.debug(f"Single component stability calculation error: {e}")
            return 0.8
    
    def _assess_multicomponent_stability(self, elements: Dict[str, int], 
                                       temperature: float, pressure: float,
                                       compound_type: str) -> float:
        """Assess stability for multi-component systems."""
        try:
            # Calculate average properties weighted by composition
            total_atoms = sum(elements.values())
            avg_stability = 0.0
            
            for element, count in elements.items():
                weight = count / total_atoms
                single_stability = self._assess_single_component_stability(
                    element, temperature, pressure, compound_type
                )
                avg_stability += weight * single_stability
            
            # Apply mixing effects - generally stabilizing for ionic compounds
            if compound_type == 'ionic' and len(elements) > 1:
                avg_stability *= 1.1
            elif compound_type == 'molecular' and len(elements) > 2:
                avg_stability *= 0.95  # Slight destabilization for complex molecules
            
            return min(1.2, avg_stability)
            
        except Exception as e:
            self.logger.debug(f"Multi-component stability calculation error: {e}")
            return 0.8
    
    def _get_critical_temperature(self, element: str) -> float:
        """Get critical temperature for element (K)."""
        # Simplified critical temperatures
        critical_temps = {
            'H': 33.2, 'He': 5.2, 'Li': 3223, 'C': 5100, 'N': 126.2, 'O': 154.6, 'F': 144.3,
            'Na': 2573, 'Mg': 4000, 'Al': 7000, 'Si': 3538, 'P': 994, 'S': 1314, 'Cl': 417,
            'Fe': 9340, 'Cu': 8563, 'Zn': 1180
        }
        return critical_temps.get(element, 1000)  # Default value
    
    def _get_critical_pressure(self, element: str) -> float:
        """Get critical pressure for element (Pa)."""
        # Simplified critical pressures
        critical_pressures = {
            'H': 1.3e6, 'He': 2.3e5, 'Li': 6.7e7, 'C': 1.0e8, 'N': 3.4e6, 'O': 5.0e6, 'F': 5.2e6,
            'Na': 3.5e7, 'Mg': 1.0e8, 'Al': 1.0e8, 'Si': 4.8e7, 'P': 4.3e7, 'S': 2.1e7, 'Cl': 7.7e6,
            'Fe': 1.0e8, 'Cu': 1.0e8, 'Zn': 5.6e7
        }
        return critical_pressures.get(element, 1.0e7)  # Default value
    
    def _get_acentric_factor(self, element: str) -> float:
        """Get acentric factor for element."""
        # Simplified acentric factors
        acentric_factors = {
            'H': -0.22, 'He': -0.39, 'N': 0.04, 'O': 0.02, 'F': 0.05,
            'Cl': 0.07, 'C': 0.1, 'S': 0.25
        }
        return acentric_factors.get(element, 0.1)  # Default value