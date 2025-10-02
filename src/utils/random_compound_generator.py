"""
Random Compound Generator
========================

Generates random chemical compounds from detected elements.
Simulates "alien chemistry" discoveries that need Earth validation.
"""

import random
import itertools
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from src.data.data_models import CompoundSuggestion


@dataclass
class CompoundGenerationRule:
    """Rules for generating chemically plausible compounds"""
    max_elements: int = 4  # Maximum different elements in a compound
    max_total_atoms: int = 20  # Maximum total atoms
    prefer_common_ratios: bool = True  # Favor 1:1, 1:2, 2:1, etc.
    avoid_noble_gases: bool = True  # Skip He, Ne, Ar, etc. in compounds
    metallic_probability: float = 0.15  # Chance of creating metallic compounds
    ionic_probability: float = 0.4  # Chance of ionic vs molecular


class RandomCompoundGenerator:
    """Generates random chemical compounds from available elements"""
    
    def __init__(self, rules: Optional[CompoundGenerationRule] = None):
        self.rules = rules or CompoundGenerationRule()
        self.noble_gases = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
        self.metals = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", 
                      "Al", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", 
                      "Zn", "Ga", "Y", "Zr", "Nb", "Mo", "Ag", "Cd", "In", "Sn",
                      "Sb", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Pb", "Bi"}
        self.nonmetals = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"}
        
    def filter_elements(self, elements: Dict[str, float]) -> Dict[str, float]:
        """Filter out elements that shouldn't form compounds"""
        filtered = {}
        for element, quantity in elements.items():
            if self.rules.avoid_noble_gases and element in self.noble_gases:
                continue
            if quantity > 0.1:  # Only use elements with reasonable quantities
                filtered[element] = quantity
        return filtered
    
    def generate_random_formula(self, elements: List[str]) -> str:
        """Generate a random chemical formula from given elements"""
        
        # Randomly select 1-4 elements
        num_elements = random.randint(1, min(len(elements), self.rules.max_elements))
        selected_elements = random.sample(elements, num_elements)
        
        formula_parts = []
        total_atoms = 0
        
        for element in selected_elements:
            if total_atoms >= self.rules.max_total_atoms:
                break
                
            # Generate random coefficient
            if self.rules.prefer_common_ratios:
                # Favor common ratios like 1, 2, 3, 4
                coefficient = random.choices([1, 2, 3, 4, 5, 6], 
                                           weights=[40, 25, 15, 10, 5, 5])[0]
            else:
                coefficient = random.randint(1, 8)
            
            # Ensure we don't exceed max atoms
            coefficient = min(coefficient, self.rules.max_total_atoms - total_atoms)
            if coefficient <= 0:
                break
                
            total_atoms += coefficient
            
            # Build formula string
            if coefficient == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{coefficient}")
        
        return "".join(formula_parts)
    
    def determine_compound_type(self, formula: str, elements: List[str]) -> str:
        """Determine if compound is ionic, molecular, metallic, or network"""
        
        element_set = set(elements)
        metals_present = element_set.intersection(self.metals)
        nonmetals_present = element_set.intersection(self.nonmetals)
        
        # All metals = metallic compound
        if len(metals_present) > 1 and len(nonmetals_present) == 0:
            return "metallic"
        
        # Metal + nonmetal = usually ionic
        if len(metals_present) > 0 and len(nonmetals_present) > 0:
            if random.random() < self.rules.ionic_probability:
                return "ionic"
            else:
                return "molecular"
        
        # All nonmetals = molecular
        if len(nonmetals_present) > 0 and len(metals_present) == 0:
            # Special cases for network solids
            if element_set.intersection({"Si", "C", "B"}) and len(elements) >= 2:
                if random.random() < 0.2:  # 20% chance for network solid
                    return "network"
            return "molecular"
        
        # Default fallback
        return "molecular"
    
    def estimate_stability(self, formula: str, elements: List[str]) -> str:
        """Estimate compound stability based on heuristics"""
        
        element_set = set(elements)
        
        # High stability indicators
        high_stability_patterns = [
            {"O"},  # Oxides tend to be stable
            {"Cl", "F"},  # Halides
            {"H", "O"},  # Hydroxides/water
            {"Na", "Cl"},  # Simple salts
            {"Ca", "O"},  # Alkaline earth oxides
            {"Fe", "O"},  # Iron oxides
        ]
        
        # Medium stability
        medium_stability_patterns = [
            {"C", "H"},  # Hydrocarbons
            {"N", "H"},  # Amines/ammonia
            {"S", "O"},  # Sulfates/sulfites
        ]
        
        # Check patterns
        for pattern in high_stability_patterns:
            if pattern.issubset(element_set):
                return "high"
        
        for pattern in medium_stability_patterns:
            if pattern.issubset(element_set):
                return "medium"
        
        # Complex compounds with many elements = lower stability
        if len(elements) > 3:
            return "low"
        
        return "medium"
    
    def generate_compound_name(self, formula: str, elements: List[str]) -> str:
        """Generate a plausible compound name"""
        
        # Simple naming rules
        if len(elements) == 1:
            return f"{elements[0]} compound"
        
        if len(elements) == 2:
            element1, element2 = elements[:2]
            
            # Common binary compound patterns
            if element1 in self.metals and element2 in self.nonmetals:
                if element2 == "O":
                    return f"{element1} oxide"
                elif element2 == "Cl":
                    return f"{element1} chloride"
                elif element2 == "S":
                    return f"{element1} sulfide"
                elif element2 == "N":
                    return f"{element1} nitride"
                else:
                    return f"{element1} {element2.lower()}ide"
            
            elif "O" in elements and "H" in elements:
                other_element = [e for e in elements if e not in ["O", "H"]][0] if len(elements) > 2 else elements[0]
                return f"{other_element} hydroxide"
            
            else:
                return f"{element1}-{element2} compound"
        
        # Complex compounds
        if "O" in elements:
            main_element = [e for e in elements if e != "O"][0] if len(elements) > 1 else elements[0]
            return f"{main_element} oxide complex"
        else:
            return f"Multi-element {'-'.join(elements[:2])} compound"
    
    def generate_random_compounds(self, 
                                 detected_elements: Dict[str, float], 
                                 num_compounds: int = 10) -> List[CompoundSuggestion]:
        """
        Generate random compounds from detected elements
        
        Args:
            detected_elements: Dict of element symbols to quantities
            num_compounds: Number of compounds to generate
            
        Returns:
            List of CompoundSuggestion objects
        """
        
        # Filter elements
        usable_elements = self.filter_elements(detected_elements)
        if len(usable_elements) < 2:
            # Add some common elements if we don't have enough
            usable_elements.update({"H": 5.0, "O": 3.0, "C": 2.0})
        
        element_list = list(usable_elements.keys())
        compounds = []
        generated_formulas = set()  # Avoid duplicates
        
        attempts = 0
        while len(compounds) < num_compounds and attempts < num_compounds * 3:
            attempts += 1
            
            try:
                # Generate random formula
                formula = self.generate_random_formula(element_list)
                
                # Skip if we've already generated this formula
                if formula in generated_formulas:
                    continue
                
                generated_formulas.add(formula)
                
                # Extract elements from formula
                formula_elements = self._extract_elements_from_formula(formula)
                
                # Determine properties
                compound_type = self.determine_compound_type(formula, list(formula_elements.keys()))
                stability = self.estimate_stability(formula, list(formula_elements.keys()))
                name = self.generate_compound_name(formula, list(formula_elements.keys()))
                
                # Create compound suggestion
                compound = CompoundSuggestion(
                    formula=formula,
                    name=name,
                    compound_type=compound_type,
                    stability=stability,
                    source="random_generator"
                )
                
                compounds.append(compound)
                
            except Exception as e:
                # Skip problematic compounds
                continue
        
        return compounds
    
    def _extract_elements_from_formula(self, formula: str) -> Dict[str, int]:
        """Extract elements and their counts from a formula string"""
        import re
        
        # Simple regex to extract elements and numbers
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        elements = {}
        for element, count in matches:
            count = int(count) if count else 1
            elements[element] = elements.get(element, 0) + count
        
        return elements


def main():
    """Demo the random compound generator"""
    
    generator = RandomCompoundGenerator()
    
    print("🧪 Random Compound Generator Demo\n")
    
    # Example detected elements (like from Mars)
    detected_elements = {
        "O": 25.0, "Si": 15.0, "Fe": 12.0, "Mg": 8.0,
        "Ca": 5.0, "Al": 4.0, "S": 3.0, "Na": 2.0,
        "H": 1.5, "C": 1.0
    }
    
    print(f"🛸 Detected Elements: {', '.join(detected_elements.keys())}")
    print(f"📊 Total Element Types: {len(detected_elements)}\n")
    
    # Generate random compounds
    compounds = generator.generate_random_compounds(detected_elements, num_compounds=8)
    
    print("🎲 Generated Random Compounds:")
    print("=" * 50)
    
    for i, compound in enumerate(compounds, 1):
        print(f"{i:2d}. {compound.formula:12s} | {compound.name:25s} | {compound.compound_type:10s} | {compound.stability}")
    
    print(f"\n✅ Generated {len(compounds)} random compounds for validation!")
    print("💡 Next step: Check if these compounds exist on Earth...")


if __name__ == "__main__":
    main()