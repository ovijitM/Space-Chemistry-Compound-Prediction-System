"""Element database module for managing element properties with planet-independent random relevance."""

import logging
import random
from typing import Dict, Optional

# Optional imports for element data
try:
    from periodictable import elements
    PERIODICTABLE_AVAILABLE = True
except ImportError:
    PERIODICTABLE_AVAILABLE = False
    elements = None
    logging.warning("periodictable not available. Element database functionality will be limited.")


class ElementDatabase:
    """Enhanced element database with caching and planet-independent random relevance scoring."""
    
    def __init__(self):
        self.properties = {}
        self.random_weights = {}
        self._load_element_data()
        self._init_random_weights()
    
    def _load_element_data(self):
        """Load comprehensive element properties."""
        if not PERIODICTABLE_AVAILABLE:
            raise ImportError("periodictable is required for element data")
        
        # Extended list including planetary-relevant elements
        relevant_elements = [
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
        
        for symbol in relevant_elements:
            try:
                element = elements.symbol(symbol)
                self.properties[symbol] = {
                    'atomic_mass': float(element.mass),
                    'atomic_number': int(element.number),
                    'density': getattr(element, 'density', 1.0),
                    'melting_point': getattr(element, 'melting_point', 273.15),
                    'boiling_point': getattr(element, 'boiling_point', 373.15),
                    'electronegativity': getattr(element, 'electronegativity', 1.0),
                    'ionization_energy': getattr(element, 'ionization_energy', 0.0),
                    'atomic_radius': getattr(element, 'atomic_radius', 100.0)
                }
            except Exception as e:
                logging.warning(f"Could not load data for element {symbol}: {e}")
    
    def _init_random_weights(self):
        """Initialize random weights per element (planet-independent)."""
        # Assign a random relevance weight per element; used to scale quantities/availability
        self.random_weights = {el: random.uniform(0.5, 1.5) for el in self.properties.keys()}
    
    def get_planetary_weight(self, element: str, planet: str = 'earth') -> float:
        """Get random weight for element selection (planet-independent; 'planet' ignored)."""
        return self.random_weights.get(element, 1.0)
    
    def get_property(self, element: str, property_name: str, default=None):
        """Get specific property of an element."""
        return self.properties.get(element, {}).get(property_name, default)