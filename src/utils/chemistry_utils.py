"""Chemistry utility functions module.

This module contains common utility functions for chemical formula parsing,
element extraction, and other chemistry-related operations.
"""

import re
from typing import Dict, Tuple, Any, List, Optional, Union
from dataclasses import dataclass
import hashlib
import json


class ChemistryUtils:
    """Collection of chemistry utility functions."""
    
    @staticmethod
    def extract_elements_from_formula(formula: str) -> Dict[str, int]:
        """Extract elements and their counts from chemical formula.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Dictionary mapping element symbols to their counts
        """
        # Remove spaces and handle basic cases
        formula = formula.replace(' ', '')
        
        # Pattern to match element symbols and their counts
        pattern = r'([A-Z][a-z]?)([0-9]*)'
        matches = re.findall(pattern, formula)
        
        elements = {}
        for element, count in matches:
            count = int(count) if count else 1
            elements[element] = elements.get(element, 0) + count
        
        return elements
    
    @staticmethod
    def check_stoichiometry(formula: str, element_quantities: Union[Dict[str, Dict], Dict[str, float], None] = None) -> Dict:
        """Check stoichiometric feasibility.
        
        Args:
            formula: Chemical formula
            element_quantities: Available element quantities (either Dict[str, Dict] or Dict[str, float])
                              If None, returns a default insufficient result
            
        Returns:
            Dictionary with stoichiometry analysis results
        """
        # Handle case where element_quantities is missing (defensive programming)
        if element_quantities is None:
            elements = ChemistryUtils.extract_elements_from_formula(formula)
            return {
                'sufficient': False,
                'max_yield_moles': 0,
                'limiting_element': list(elements.keys())[0] if elements else None,
                'element_requirements': elements,
                'error': 'No element quantities provided'
            }
        elements = ChemistryUtils.extract_elements_from_formula(formula)
        
        sufficient = True
        max_yield = float('inf')
        limiting_element = None
        
        for element, required in elements.items():
            if element not in element_quantities:
                sufficient = False
                max_yield = 0
                limiting_element = element
                break
            
            # Handle both old format (Dict[str, Dict]) and new format (Dict[str, float])
            if isinstance(element_quantities[element], dict):
                available = element_quantities[element]['quantity']
            else:
                available = element_quantities[element]
            possible_yield = available / required
            
            if possible_yield < max_yield:
                max_yield = possible_yield
                limiting_element = element
        
        if max_yield == float('inf'):
            max_yield = 0
        
        return {
            'sufficient': sufficient and max_yield > 0,
            'max_yield_moles': max_yield,
            'limiting_element': limiting_element,
            'element_requirements': elements
        }
    
    @staticmethod
    def get_cache_key(compound, element_quantities: Dict, conditions) -> str:
        """Generate cache key for validation result.
        
        Args:
            compound: CompoundSuggestion object
            element_quantities: Element quantities dictionary
            conditions: EnvironmentalConditions object
            
        Returns:
            Cache key string
        """
        elements_str = str(sorted(element_quantities.items()))
        conditions_str = f"{conditions.temperature}_{conditions.pressure}_{conditions.atmosphere}"
        return f"{compound.formula}_{hash(elements_str)}_{hash(conditions_str)}"
    
    @staticmethod
    def get_common_oxidation_states() -> Dict[str, list]:
        """Get common oxidation states for elements.
        
        Returns:
            Dictionary mapping element symbols to lists of common oxidation states
        """
        return {
            'H': [1, -1], 'Li': [1], 'Be': [2], 'B': [3], 'C': [4, -4, 2, -2],
            'N': [5, 3, -3, 4, 2, 1], 'O': [-2, -1], 'F': [-1], 'Na': [1],
            'Mg': [2], 'Al': [3], 'Si': [4, -4], 'P': [5, 3, -3], 'S': [6, 4, 2, -2],
            'Cl': [7, 5, 3, 1, -1], 'K': [1], 'Ca': [2], 'Fe': [3, 2], 'Cu': [2, 1],
            'Zn': [2], 'Ag': [1], 'I': [7, 5, 1, -1]
        }
    
    @staticmethod
    def get_electronegativity_values() -> Dict[str, float]:
        """Get Pauling electronegativity values for elements using PyMatGen with fallback.
        
        Returns:
            Dictionary mapping element symbols to electronegativity values
        """
        # Fallback hardcoded values for cases where PyMatGen is not available
        fallback_values = {
            'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
            'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
            'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
            'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Rb': 0.82, 'Sr': 0.95, 'Ag': 1.93, 'I': 2.66,
            'Pr': 1.13  # Lanthanide for intermetallic heuristics
        }
        
        try:
            # Try to use PyMatGen for more comprehensive and accurate data
            from pymatgen.core import Element
            
            electronegativity_values = {}
            
            # Get all known elements from PyMatGen
            for symbol in fallback_values.keys():
                try:
                    element = Element(symbol)
                    if hasattr(element, 'X') and element.X is not None:
                        electronegativity_values[symbol] = float(element.X)
                    else:
                        electronegativity_values[symbol] = fallback_values[symbol]
                except Exception:
                    electronegativity_values[symbol] = fallback_values[symbol]
            
            # Add additional elements from PyMatGen that might be useful
            additional_elements = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                                 'Cs', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                                 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
            
            for symbol in additional_elements:
                try:
                    element = Element(symbol)
                    if hasattr(element, 'X') and element.X is not None:
                        electronegativity_values[symbol] = float(element.X)
                except Exception:
                    continue  # Skip if element not available
            
            return electronegativity_values
            
        except ImportError:
            # PyMatGen not available, use fallback values
            return fallback_values
        except Exception:
            # Any other error, use fallback values
            return fallback_values
    
    @staticmethod
    def get_stability_ranges() -> Dict[str, Tuple[float, float]]:
        """Get temperature stability ranges for different compound types.
        
        Returns:
            Dictionary mapping compound types to (min_temp, max_temp) tuples
        """
        return {
            'ionic': (-200, 800),
            'molecular': (-100, 300),
            'metallic': (-200, 1000),
            'network': (-200, 1200)
        }
    
    @staticmethod
    def is_noble_gas(element: str) -> bool:
        """Check if element is a noble gas.
        
        Args:
            element: Element symbol
            
        Returns:
            True if element is a noble gas
        """
        noble_gases = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}
        return element in noble_gases
    
    @staticmethod
    def calculate_electronegativity_difference(elem1: str, elem2: str) -> float:
        """Calculate electronegativity difference between two elements.
        
        Args:
            elem1: First element symbol
            elem2: Second element symbol
            
        Returns:
            Absolute electronegativity difference
        """
        electronegativity = ChemistryUtils.get_electronegativity_values()
        
        en1 = electronegativity.get(elem1, 2.0)  # Default to 2.0 if unknown
        en2 = electronegativity.get(elem2, 2.0)
        
        return abs(en1 - en2)
    
    @staticmethod
    def check_charge_balance(elements: Dict[str, int], oxidation_states: Dict[str, int]) -> bool:
        """Check if oxidation states result in charge balance.
        
        Args:
            elements: Dictionary of element symbols to counts
            oxidation_states: Dictionary of element symbols to oxidation states
            
        Returns:
            True if charge balanced
        """
        total_charge = 0
        for element, count in elements.items():
            if element in oxidation_states:
                total_charge += count * oxidation_states[element]
        
        return total_charge == 0
    
    @staticmethod
    def find_possible_oxidation_combinations(elements: Dict[str, int]) -> list:
        """Find possible oxidation state combinations for charge balance.
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            List of possible oxidation state combinations
        """
        common_oxidation_states = ChemistryUtils.get_common_oxidation_states()
        possible_combinations = []
        
        # For binary compounds, check all combinations
        if len(elements) == 2:
            elem1, elem2 = list(elements.keys())
            count1, count2 = elements[elem1], elements[elem2]
            
            ox_states1 = common_oxidation_states.get(elem1, [1, 2, 3, -1, -2, -3])
            ox_states2 = common_oxidation_states.get(elem2, [1, 2, 3, -1, -2, -3])
            
            for ox1 in ox_states1:
                for ox2 in ox_states2:
                    if count1 * ox1 + count2 * ox2 == 0:
                        possible_combinations.append({elem1: ox1, elem2: ox2})
        
        return possible_combinations
    
    @staticmethod
    def parse_formula_composition(formula: str) -> Dict[str, int]:
        """Parse chemical formula to extract element composition.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Dictionary mapping elements to their counts
        """
        # Remove spaces and handle parentheses
        formula = formula.replace(' ', '')
        
        # Simple regex pattern for element counting
        pattern = r'([A-Z][a-z]?)([0-9]*)'
        matches = re.findall(pattern, formula)
        
        composition = {}
        for element, count in matches:
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
            
        return composition
    
    @staticmethod
    def balance_formation_reaction(target_composition: Dict[str, int], 
                                 available_elements: Dict[str, float]) -> Dict[str, Any]:
        """Balance the formation reaction for the target compound.
        
        Args:
            target_composition: Target compound element composition
            available_elements: Available elements and quantities
            
        Returns:
            Balanced reaction information
        """
        try:
            # For simplicity, assume formation from elemental sources
            reactants = []
            coefficients = {}
            
            # Build reactant list from available elements
            for element, count in target_composition.items():
                if element in available_elements and available_elements[element] >= count:
                    reactants.append(element)
                    coefficients[element] = count
                else:
                    return {
                        'balanced': False,
                        'error': f'Insufficient {element} available',
                        'equation': '',
                        'coefficients': {},
                        'atom_balance': {}
                    }
            
            # Create balanced equation string
            reactant_str = ' + '.join([f"{coeff if coeff > 1 else ''}{elem}" 
                                     for elem, coeff in coefficients.items()])
            product_formula = ''.join([f"{elem}{count if count > 1 else ''}" 
                                     for elem, count in target_composition.items()])
            equation = f"{reactant_str} → {product_formula}"
            
            # Verify atom balance
            atom_balance = {}
            for element in target_composition:
                reactant_atoms = coefficients.get(element, 0)
                product_atoms = target_composition[element]
                atom_balance[element] = {
                    'reactants': reactant_atoms,
                    'products': product_atoms,
                    'balanced': reactant_atoms == product_atoms
                }
            
            all_balanced = all(balance['balanced'] for balance in atom_balance.values())
            
            return {
                'balanced': all_balanced,
                'equation': equation,
                'coefficients': coefficients,
                'atom_balance': atom_balance,
                'mass_balance': True  # Simplified assumption
            }
            
        except Exception as e:
            return {
                'balanced': False,
                'error': str(e),
                'equation': '',
                'coefficients': {},
                'atom_balance': {}
            }
    
    @staticmethod
    def verify_stoichiometric_requirements(target_composition: Dict[str, int],
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
        try:
            if not balanced_reaction.get('balanced', False):
                return {'valid': False, 'error': 'Reaction not balanced'}
            
            coefficients = balanced_reaction.get('coefficients', {})
            limiting_reagent = None
            min_yield = float('inf')
            
            # Find limiting reagent
            for element, required_moles in coefficients.items():
                if element in available_elements:
                    available_moles = available_elements[element]
                    possible_yield = available_moles / required_moles
                    
                    if possible_yield < min_yield:
                        min_yield = possible_yield
                        limiting_reagent = element
            
            theoretical_yield = min_yield if min_yield != float('inf') else 0.0
            
            return {
                'valid': theoretical_yield > 0,
                'limiting_reagent': limiting_reagent,
                'theoretical_yield': theoretical_yield,
                'yield_efficiency': min(1.0, theoretical_yield)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'limiting_reagent': '',
                'theoretical_yield': 0.0
            }
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp as ISO format string.
        
        Returns:
            Current timestamp in ISO format
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    @staticmethod
    def evaluate_hume_rothery_rules(elements: Dict[str, int]) -> Dict[str, float]:
        """Evaluate Hume-Rothery rules for metallic compound formation.
        
        The Hume-Rothery rules predict solid solution formation in metallic systems:
        1. Atomic size factor: <15% difference in atomic radii
        2. Electronegativity factor: similar electronegativity values
        3. Valence factor: similar valence electron concentrations
        4. Crystal structure factor: compatible crystal structures
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with rule evaluations and overall compatibility score
        """
        if len(elements) < 2:
            return {'overall_score': 1.0, 'size_factor': 1.0, 'electronegativity_factor': 1.0, 
                   'valence_factor': 1.0, 'crystal_structure_factor': 1.0}
        
        try:
            # Get atomic radii using PyMatGen with fallback
            atomic_radii = {}
            electronegativity_values = ChemistryUtils.get_electronegativity_values()
            
            # Fallback atomic radii (in pm) for common elements
            fallback_radii = {
                'H': 53, 'Li': 167, 'Be': 112, 'B': 87, 'C': 67, 'N': 56, 'O': 48, 'F': 42,
                'Na': 190, 'Mg': 145, 'Al': 118, 'Si': 111, 'P': 98, 'S': 88, 'Cl': 79,
                'K': 243, 'Ca': 194, 'Sc': 184, 'Ti': 176, 'V': 171, 'Cr': 166, 'Mn': 161,
                'Fe': 156, 'Co': 152, 'Ni': 149, 'Cu': 145, 'Zn': 142, 'Ga': 136, 'Ge': 125,
                'As': 114, 'Se': 103, 'Br': 94, 'Rb': 265, 'Sr': 219, 'Y': 212, 'Zr': 206,
                'Nb': 198, 'Mo': 190, 'Tc': 183, 'Ru': 178, 'Rh': 173, 'Pd': 169, 'Ag': 165,
                'Cd': 161, 'In': 156, 'Sn': 145, 'Sb': 133, 'Te': 123, 'I': 115, 'Cs': 298,
                'Ba': 253, 'La': 247, 'Ce': 244, 'Pr': 243, 'Nd': 241, 'Sm': 238, 'Eu': 235,
                'Gd': 233, 'Tb': 230, 'Dy': 228, 'Ho': 226, 'Er': 224, 'Tm': 222, 'Yb': 220,
                'Lu': 217, 'Hf': 208, 'Ta': 200, 'W': 193, 'Re': 188, 'Os': 185, 'Ir': 180,
                'Pt': 177, 'Au': 174, 'Hg': 171, 'Tl': 156, 'Pb': 154, 'Bi': 143
            }
            
            try:
                from pymatgen.core import Element
                for elem in elements.keys():
                    try:
                        element = Element(elem)
                        if hasattr(element, 'atomic_radius') and element.atomic_radius is not None:
                            atomic_radii[elem] = float(element.atomic_radius)
                        else:
                            atomic_radii[elem] = fallback_radii.get(elem, 150.0)
                    except Exception:
                        atomic_radii[elem] = fallback_radii.get(elem, 150.0)
            except ImportError:
                # Use fallback values if PyMatGen not available
                for elem in elements.keys():
                    atomic_radii[elem] = fallback_radii.get(elem, 150.0)
            
            # 1. Atomic size factor (Hume-Rothery Rule 1)
            radii_values = list(atomic_radii.values())
            if len(radii_values) >= 2:
                max_radius = max(radii_values)
                min_radius = min(radii_values)
                size_difference = abs(max_radius - min_radius) / max_radius
                
                if size_difference <= 0.15:  # <15% difference is favorable
                    size_factor = 1.0
                elif size_difference <= 0.25:  # 15-25% is marginal
                    size_factor = 0.7
                else:  # >25% is unfavorable
                    size_factor = 0.3
            else:
                size_factor = 1.0
            
            # 2. Electronegativity factor (Hume-Rothery Rule 2)
            electroneg_values = [electronegativity_values.get(elem, 2.0) for elem in elements.keys()]
            if len(electroneg_values) >= 2:
                max_electroneg = max(electroneg_values)
                min_electroneg = min(electroneg_values)
                electroneg_diff = abs(max_electroneg - min_electroneg)
                
                if electroneg_diff <= 0.4:  # Similar electronegativity
                    electronegativity_factor = 1.0
                elif electroneg_diff <= 0.8:  # Moderate difference
                    electronegativity_factor = 0.8
                else:  # Large difference
                    electronegativity_factor = 0.5
            else:
                electronegativity_factor = 1.0
            
            # 3. Valence factor (simplified - based on common oxidation states)
            # This is a simplified approach; real VEC calculation would need more detailed data
            common_metallic_elements = {
                'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,  # Alkali metals
                'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,  # Alkaline earth
                'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,  # Group 13
                'Sc': 3, 'Y': 3, 'La': 3,  # Group 3
                'Ti': 4, 'Zr': 4, 'Hf': 4,  # Group 4
                'V': 5, 'Nb': 5, 'Ta': 5,  # Group 5
                'Cr': 6, 'Mo': 6, 'W': 6,  # Group 6
                'Mn': 7, 'Tc': 7, 'Re': 7,  # Group 7
                'Fe': 8, 'Ru': 8, 'Os': 8,  # Group 8
                'Co': 9, 'Rh': 9, 'Ir': 9,  # Group 9
                'Ni': 10, 'Pd': 10, 'Pt': 10,  # Group 10
                'Cu': 11, 'Ag': 11, 'Au': 11,  # Group 11
                'Zn': 12, 'Cd': 12, 'Hg': 12,  # Group 12
            }
            
            valence_electrons = [common_metallic_elements.get(elem, 4) for elem in elements.keys()]
            if len(valence_electrons) >= 2:
                valence_range = max(valence_electrons) - min(valence_electrons)
                if valence_range <= 2:  # Similar valence
                    valence_factor = 1.0
                elif valence_range <= 4:  # Moderate difference
                    valence_factor = 0.8
                else:  # Large difference
                    valence_factor = 0.6
            else:
                valence_factor = 1.0
            
            # 4. Crystal structure factor (simplified - based on common structures)
            # This is highly simplified; real analysis would need detailed crystallographic data
            structure_compatible_groups = [
                {'Li', 'Na', 'K', 'Rb', 'Cs'},  # BCC alkali metals
                {'Be', 'Mg', 'Zn', 'Cd'},  # HCP metals
                {'Al', 'Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt'},  # FCC metals
                {'Fe', 'Cr', 'V', 'Nb', 'Ta', 'W', 'Mo'},  # BCC transition metals
                {'Ti', 'Zr', 'Hf', 'Co', 'Ru', 'Os'},  # HCP transition metals
            ]
            
            element_set = set(elements.keys())
            crystal_structure_factor = 0.7  # Default moderate compatibility
            
            for group in structure_compatible_groups:
                if element_set.issubset(group):
                    crystal_structure_factor = 1.0  # High compatibility within group
                    break
                elif len(element_set.intersection(group)) > 0:
                    crystal_structure_factor = 0.8  # Partial compatibility
            
            # Calculate overall score (weighted average)
            overall_score = (
                0.3 * size_factor +
                0.25 * electronegativity_factor +
                0.25 * valence_factor +
                0.2 * crystal_structure_factor
            )
            
            return {
                'overall_score': overall_score,
                'size_factor': size_factor,
                'electronegativity_factor': electronegativity_factor,
                'valence_factor': valence_factor,
                'crystal_structure_factor': crystal_structure_factor,
                'size_difference_percent': size_difference * 100 if 'size_difference' in locals() else 0.0,
                'electronegativity_difference': electroneg_diff if 'electroneg_diff' in locals() else 0.0
            }
            
        except Exception as e:
            # Return neutral scores on error
            return {'overall_score': 0.7, 'size_factor': 0.7, 'electronegativity_factor': 0.7,
                   'valence_factor': 0.7, 'crystal_structure_factor': 0.7, 'error': str(e)}