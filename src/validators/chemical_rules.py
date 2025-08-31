"""Chemical rules validation module.

This module contains validators for chemical rules including oxidation states,
electronegativity compatibility, and basic chemistry constraints.
"""

import logging
from typing import Dict, List, Tuple, Any
from ..utils.chemistry_utils import ChemistryUtils


class ChemicalRulesValidator:
    """Validator for chemical rules and constraints."""
    
    def __init__(self, logger=None):
        """Initialize chemical rules validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_chemical_rules(self, compound, elements: Dict[str, int]) -> Dict:
        """Validate against basic chemical rules.
        
        Args:
            compound: CompoundSuggestion object
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check for impossible combinations
        if self._has_noble_gas_compound(elements):
            issues.append('noble_gas_compound')
        
        # Check oxidation states
        oxidation_validation = self.validate_oxidation_states(compound.formula, elements)
        if not oxidation_validation['valid']:
            issues.extend(oxidation_validation['issues'])
        warnings.extend(oxidation_validation['warnings'])
        
        # Check electronegativity compatibility
        electronegativity_check = self.check_electronegativity_compatibility(elements)
        warnings.extend(electronegativity_check['warnings'])
        
        # Additional checks to derive a quantitative overall score
        valence_check = self.check_valence_constraints(elements)
        warnings.extend(valence_check['warnings'])
        stoich_ratio_check = self.validate_stoichiometric_ratios(elements)
        warnings.extend(stoich_ratio_check['warnings'])
        
        # Derive an overall score in [0,1] to be consumed by probability calculator
        score = 0.7
        
        # Penalize hard issues
        if issues:
            score -= 0.15
        if 'charge_imbalance' in issues:
            score -= 0.1
        if 'noble_gas_compound' in issues:
            score -= 0.2
        
        # Consider electronegativity spread
        max_diff = electronegativity_check.get('max_difference', 0.0)
        if max_diff > 3.0:
            score -= 0.08
        elif max_diff > 2.5:
            score -= 0.05
        
        # Stoichiometric ratio sanity
        if 'large_stoichiometric_ratio' in stoich_ratio_check.get('warnings', []):
            score -= 0.04
        if 'very_large_stoichiometry' in stoich_ratio_check.get('warnings', []):
            score -= 0.03
        
        # Metallic binary heuristics: very skewed ratios are less favorable
        if getattr(compound, 'compound_type', None) == 'metallic' and len(elements) == 2:
            counts = list(elements.values())
            if all(c > 0 for c in counts):
                ratio = max(counts) / min(counts)
                if ratio <= 2.0:
                    score += 0.04
                elif ratio >= 8.0:
                    score -= 0.05
        
        # Light penalty for cumulative warnings
        score -= min(0.1, 0.02 * len(warnings))
        
        # Clamp and round
        score = max(0.2, min(1.0, score))
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'oxidation_details': oxidation_validation,
            'electronegativity_details': electronegativity_check,
            'overall_score': score
        }
    
    def _has_noble_gas_compound(self, elements: Dict[str, int]) -> bool:
        """Check for noble gas compounds (generally impossible).
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            True if contains noble gas in multi-element compound
        """
        noble_gases = {'He', 'Ne', 'Ar'}
        has_noble_gas = any(elem in noble_gases for elem in elements)
        return has_noble_gas and len(elements) > 1
    
    def validate_oxidation_states(self, formula: str, elements: Dict[str, int]) -> Dict:
        """Validate oxidation states for charge balance.
        
        Args:
            formula: Chemical formula
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with oxidation state validation results
        """
        issues = []
        warnings = []
        
        common_oxidation_states = ChemistryUtils.get_common_oxidation_states()
        
        try:
            # For binary compounds, check charge balance
            if len(elements) == 2:
                elem1, elem2 = list(elements.keys())
                count1, count2 = elements[elem1], elements[elem2]
                
                ox_states1 = common_oxidation_states.get(elem1, [1, 2, 3, -1, -2, -3])
                ox_states2 = common_oxidation_states.get(elem2, [1, 2, 3, -1, -2, -3])
                
                # Check if any combination gives charge balance
                balanced = False
                for ox1 in ox_states1:
                    for ox2 in ox_states2:
                        if count1 * ox1 + count2 * ox2 == 0:
                            balanced = True
                            break
                    if balanced:
                        break
                
                if not balanced:
                    issues.append('charge_imbalance')
            
            # Check for impossible oxidation states
            for element in elements:
                if element not in common_oxidation_states:
                    warnings.append(f'unknown_oxidation_states_{element}')
        
        except Exception as e:
            warnings.append(f'oxidation_validation_error: {str(e)}')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def check_electronegativity_compatibility(self, elements: Dict[str, int]) -> Dict:
        """Check electronegativity differences for bond compatibility.
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with electronegativity compatibility results
        """
        electronegativity = ChemistryUtils.get_electronegativity_values()
        
        compatible = True
        max_diff = 0.0
        warnings = []
        
        try:
            element_list = list(elements.keys())
            
            # Check electronegativity differences between all pairs
            for i in range(len(element_list)):
                for j in range(i + 1, len(element_list)):
                    elem1, elem2 = element_list[i], element_list[j]
                    
                    if elem1 in electronegativity and elem2 in electronegativity:
                        diff = abs(electronegativity[elem1] - electronegativity[elem2])
                        max_diff = max(max_diff, diff)
                        
                        # Very large differences might indicate unusual bonding
                        if diff > 3.0:
                            warnings.append(f'large_electronegativity_diff_{elem1}_{elem2}')
                        elif diff > 2.5:
                            warnings.append(f'moderate_electronegativity_diff_{elem1}_{elem2}')
                    else:
                        warnings.append(f'unknown_electronegativity_{elem1 if elem1 not in electronegativity else elem2}')
        
        except Exception as e:
            warnings.append(f'electronegativity_check_error: {str(e)}')
        
        return {
            'compatible': compatible,
            'max_difference': max_diff,
            'warnings': warnings
        }
    
    def check_valence_constraints(self, elements: Dict[str, int]) -> Dict:
        """Check valence electron constraints.
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with valence constraint results
        """
        # Common valence electron counts
        valence_electrons = {
            'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
            'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
            'K': 1, 'Ca': 2, 'Fe': 8, 'Cu': 11, 'Zn': 12, 'Ag': 11, 'I': 7
        }
        
        warnings = []
        total_valence = 0
        
        for element, count in elements.items():
            if element in valence_electrons:
                total_valence += valence_electrons[element] * count
            else:
                warnings.append(f'unknown_valence_{element}')
        
        # Check for reasonable valence electron count
        if total_valence % 2 != 0 and len(elements) > 1:
            warnings.append('odd_valence_electrons')
        
        return {
            'total_valence_electrons': total_valence,
            'warnings': warnings
        }
    
    def check_bond_order_constraints(self, elements: Dict[str, int]) -> Dict:
        """Check bond order and connectivity constraints.
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with bond order constraint results
        """
        # Maximum bonds for common elements
        max_bonds = {
            'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1,
            'Si': 4, 'P': 5, 'S': 6, 'Cl': 7,
            'Fe': 6, 'Cu': 4, 'Zn': 4
        }
        
        warnings = []
        connectivity_issues = []
        
        for element, count in elements.items():
            if element in max_bonds:
                max_bond_count = max_bonds[element]
                
                # For simple heuristic, check if element can form enough bonds
                other_atoms = sum(elements.values()) - count
                if count > 1 and max_bond_count < other_atoms / count:
                    connectivity_issues.append(f'insufficient_bonding_capacity_{element}')
            else:
                warnings.append(f'unknown_bonding_capacity_{element}')
        
        return {
            'connectivity_issues': connectivity_issues,
            'warnings': warnings,
            'valid': len(connectivity_issues) == 0
        }
    
    def validate_stoichiometric_ratios(self, elements: Dict[str, int]) -> Dict:
        """Validate stoichiometric ratios for reasonableness.
        
        Args:
            elements: Dictionary of element symbols to counts
            
        Returns:
            Dictionary with stoichiometric validation results
        """
        warnings = []
        issues = []
        
        # Check for extremely large ratios
        counts = list(elements.values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            
            if max_count / min_count > 20:
                warnings.append('large_stoichiometric_ratio')
            
            # Check for very large absolute numbers
            if max_count > 100:
                warnings.append('very_large_stoichiometry')
        
        # Check for common ratio patterns
        if len(elements) == 2:
            counts = sorted(counts)
            ratio = counts[1] / counts[0] if counts[0] > 0 else float('inf')
            
            # Common ratios: 1:1, 1:2, 1:3, 2:3, etc.
            common_ratios = [1.0, 2.0, 3.0, 1.5, 0.5, 0.33, 0.67]
            if not any(abs(ratio - cr) < 0.1 for cr in common_ratios):
                warnings.append('unusual_stoichiometric_ratio')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }