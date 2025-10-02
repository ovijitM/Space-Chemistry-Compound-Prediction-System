"""
Planetary Element Sampling System
=================================

Simulates rover element detection from outer space environments.
Generates realistic element distributions for different planetary scenarios.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PlanetaryEnvironment(Enum):
    """Different planetary environments with element abundance patterns"""
    MARS_SURFACE = "mars_surface"
    MARTIAN_SUBSURFACE = "martian_subsurface"
    LUNAR_SURFACE = "lunar_surface"
    ASTEROID_METALLIC = "asteroid_metallic"
    ASTEROID_CARBONACEOUS = "asteroid_carbonaceous"
    ICY_MOON = "icy_moon"
    VENUS_SURFACE = "venus_surface"
    RANDOM_OUTER_SPACE = "random_outer_space"


@dataclass
class ElementAbundance:
    """Element abundance data for planetary environments"""
    symbol: str
    abundance_ppm: float  # parts per million
    detection_probability: float  # 0.0 to 1.0
    typical_range: Tuple[float, float]  # (min_moles, max_moles)


class PlanetaryElementSampler:
    """Simulates rover element detection from various planetary environments"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.planetary_compositions = self._initialize_planetary_data()
    
    def _initialize_planetary_data(self) -> Dict[PlanetaryEnvironment, List[ElementAbundance]]:
        """Initialize planetary element abundance data based on real space science"""
        
        compositions = {}
        
        # Mars Surface (based on rover data)
        compositions[PlanetaryEnvironment.MARS_SURFACE] = [
            ElementAbundance("O", 45000, 0.95, (20.0, 50.0)),
            ElementAbundance("Si", 21000, 0.90, (10.0, 30.0)),
            ElementAbundance("Fe", 18000, 0.85, (8.0, 25.0)),
            ElementAbundance("Mg", 9000, 0.80, (4.0, 15.0)),
            ElementAbundance("Ca", 4000, 0.75, (2.0, 8.0)),
            ElementAbundance("Al", 3500, 0.70, (1.5, 7.0)),
            ElementAbundance("S", 2000, 0.65, (1.0, 5.0)),
            ElementAbundance("K", 4500, 0.60, (2.0, 8.0)),
            ElementAbundance("Na", 3000, 0.55, (1.0, 6.0)),
            ElementAbundance("Cl", 800, 0.50, (0.5, 3.0)),
            ElementAbundance("Ti", 600, 0.45, (0.3, 2.0)),
            ElementAbundance("Mn", 400, 0.40, (0.2, 1.5)),
            ElementAbundance("Cr", 200, 0.35, (0.1, 1.0)),
            ElementAbundance("P", 1000, 0.30, (0.5, 3.0)),
            ElementAbundance("H", 500, 0.25, (0.2, 2.0)),  # From hydrated minerals
            ElementAbundance("C", 300, 0.20, (0.1, 1.0)),  # Organic compounds
            ElementAbundance("N", 150, 0.15, (0.05, 0.5)),
        ]
        
        # Asteroid Metallic (iron-nickel dominated)
        compositions[PlanetaryEnvironment.ASTEROID_METALLIC] = [
            ElementAbundance("Fe", 35000, 0.98, (50.0, 100.0)),
            ElementAbundance("Ni", 8000, 0.95, (10.0, 30.0)),
            ElementAbundance("Co", 500, 0.80, (0.5, 3.0)),
            ElementAbundance("S", 2000, 0.75, (2.0, 8.0)),
            ElementAbundance("P", 300, 0.70, (0.3, 2.0)),
            ElementAbundance("Cr", 800, 0.65, (0.8, 4.0)),
            ElementAbundance("Mn", 400, 0.60, (0.4, 2.0)),
            ElementAbundance("Cu", 200, 0.55, (0.2, 1.0)),
            ElementAbundance("Zn", 100, 0.50, (0.1, 0.8)),
            ElementAbundance("Pt", 50, 0.30, (0.05, 0.3)),  # Rare platinum group
            ElementAbundance("Au", 20, 0.25, (0.02, 0.1)),
            ElementAbundance("Ir", 10, 0.20, (0.01, 0.05)),
        ]
        
        # Icy Moon (Europa, Enceladus-like)
        compositions[PlanetaryEnvironment.ICY_MOON] = [
            ElementAbundance("O", 50000, 0.99, (30.0, 80.0)),  # Water ice
            ElementAbundance("H", 11000, 0.99, (15.0, 40.0)),  # Water ice
            ElementAbundance("S", 3000, 0.80, (2.0, 10.0)),   # Sulfur compounds
            ElementAbundance("Na", 2000, 0.75, (1.0, 6.0)),   # Salt deposits
            ElementAbundance("Cl", 1500, 0.70, (0.8, 4.0)),   # Salt deposits
            ElementAbundance("Mg", 1000, 0.65, (0.5, 3.0)),
            ElementAbundance("K", 800, 0.60, (0.4, 2.5)),
            ElementAbundance("Ca", 600, 0.55, (0.3, 2.0)),
            ElementAbundance("C", 500, 0.50, (0.2, 2.0)),     # Organic compounds
            ElementAbundance("N", 200, 0.45, (0.1, 1.0)),     # Nitrogen compounds
            ElementAbundance("P", 150, 0.40, (0.08, 0.8)),
            ElementAbundance("Si", 800, 0.35, (0.4, 2.5)),    # Silicate rock
        ]
        
        # Random Outer Space (completely random for discovery)
        compositions[PlanetaryEnvironment.RANDOM_OUTER_SPACE] = [
            ElementAbundance(element, random.uniform(10, 10000), random.uniform(0.1, 0.9), 
                           (random.uniform(0.1, 5.0), random.uniform(5.0, 50.0)))
            for element in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                          "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                          "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                          "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr"]
        ]
        
        return compositions
    
    def simulate_rover_scan(self, 
                           environment: PlanetaryEnvironment = PlanetaryEnvironment.MARS_SURFACE,
                           scan_intensity: float = 1.0,
                           min_elements: int = 3,
                           max_elements: int = 15) -> Dict[str, float]:
        """
        Simulate a rover scanning for elements in a planetary environment
        
        Args:
            environment: Which planetary environment to simulate
            scan_intensity: Detection sensitivity (0.0 to 2.0, higher = more elements detected)
            min_elements: Minimum number of elements to detect
            max_elements: Maximum number of elements to detect
            
        Returns:
            Dict mapping element symbols to quantities (in moles)
        """
        
        composition = self.planetary_compositions.get(environment, 
                                                     self.planetary_compositions[PlanetaryEnvironment.MARS_SURFACE])
        
        detected_elements = {}
        
        for element_data in composition:
            # Adjust detection probability based on scan intensity
            detection_chance = min(1.0, element_data.detection_probability * scan_intensity)
            
            if random.random() < detection_chance:
                # Generate random quantity within typical range
                min_qty, max_qty = element_data.typical_range
                quantity = random.uniform(min_qty, max_qty)
                detected_elements[element_data.symbol] = round(quantity, 2)
        
        # Ensure we have minimum number of elements
        if len(detected_elements) < min_elements:
            # Add some common elements to reach minimum
            common_elements = ["O", "Si", "Fe", "Mg", "Ca", "Al", "H", "C"]
            for element in common_elements:
                if len(detected_elements) >= min_elements:
                    break
                if element not in detected_elements:
                    detected_elements[element] = round(random.uniform(1.0, 10.0), 2)
        
        # Limit to maximum elements
        if len(detected_elements) > max_elements:
            # Keep the elements with highest quantities
            sorted_elements = sorted(detected_elements.items(), key=lambda x: x[1], reverse=True)
            detected_elements = dict(sorted_elements[:max_elements])
        
        return detected_elements
    
    def simulate_universal_scan(self, 
                               scan_intensity: float = 1.0,
                               min_elements: int = 4,
                               max_elements: int = 15) -> Dict[str, float]:
        """
        Universal element scanning for completely unknown environments
        
        This method generates elements without any predefined environmental constraints,
        suitable for truly unknown planetary/space environments.
        """
        
        # All possible elements that could be detected
        all_elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu'
        ]
        
        # Randomly determine how many elements to find
        num_elements = random.randint(min_elements, max_elements)
        
        # Randomly select elements
        selected_elements = random.sample(all_elements, num_elements)
        
        detected_elements = {}
        for element in selected_elements:
            # Generate completely random abundances
            # Some elements might be very rare, others abundant
            base_abundance = random.uniform(0.001, 50.0)  # Very wide range
            
            # Apply scan intensity effect
            detected_quantity = base_abundance * scan_intensity * random.uniform(0.5, 1.5)
            
            # Add some measurement noise
            noise_factor = random.uniform(0.9, 1.1)
            final_quantity = max(0.001, detected_quantity * noise_factor)
            
            detected_elements[element] = final_quantity
        
        return detected_elements
    
    def generate_random_element_combinations(self, num_combinations: int = 10) -> List[Dict[str, float]]:
        """Generate multiple random element combinations for testing"""
        
        combinations = []
        environments = list(PlanetaryEnvironment)
        
        for _ in range(num_combinations):
            # Randomly select environment
            env = random.choice(environments)
            
            # Random scan parameters
            intensity = random.uniform(0.3, 1.8)
            min_elem = random.randint(3, 8)
            max_elem = random.randint(min_elem + 2, 20)
            
            # Generate combination
            combination = self.simulate_rover_scan(env, intensity, min_elem, max_elem)
            combinations.append({
                "elements": combination,
                "environment": env.value,
                "scan_intensity": intensity
            })
        
        return combinations


def main():
    """Demo the planetary element sampling system"""
    
    sampler = PlanetaryElementSampler(random_seed=42)
    
    print("🛸 Planetary Element Sampling System Demo\n")
    
    # Test different environments
    environments = [
        PlanetaryEnvironment.MARS_SURFACE,
        PlanetaryEnvironment.ASTEROID_METALLIC,
        PlanetaryEnvironment.ICY_MOON,
        PlanetaryEnvironment.RANDOM_OUTER_SPACE
    ]
    
    for env in environments:
        print(f"🌍 Environment: {env.value.replace('_', ' ').title()}")
        elements = sampler.simulate_rover_scan(env)
        print(f"   Detected: {len(elements)} elements")
        for element, quantity in sorted(elements.items()):
            print(f"   {element}: {quantity:.2f} moles")
        print()
    
    # Generate random combinations
    print("🎲 Random Element Combinations:")
    combinations = sampler.generate_random_element_combinations(5)
    for i, combo in enumerate(combinations, 1):
        print(f"   Combination {i} ({combo['environment']}):")
        for element, quantity in sorted(combo['elements'].items()):
            print(f"     {element}: {quantity:.2f} moles")
        print()


if __name__ == "__main__":
    main()