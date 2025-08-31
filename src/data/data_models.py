#!/usr/bin/env python3
"""
Data Models Module
==================

Contains all dataclasses, enums, and data structures used throughout
the NASA Rover Chemistry Prediction System.

This module defines:
- Phase equilibria data structures
- Thermodynamic data structures
- Environmental conditions
- Compound suggestions and validation results
- Planetary environment presets
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


# ============================================================================
# PHASE EQUILIBRIA DATA STRUCTURES
# ============================================================================

class PhaseType(Enum):
    """Enumeration of phase types"""
    VAPOR = "vapor"
    LIQUID = "liquid"
    SOLID = "solid"
    SUPERCRITICAL = "supercritical"


@dataclass
class ThermodynamicState:
    """Represents a thermodynamic state point"""
    temperature: float  # K
    pressure: float     # Pa
    composition: Dict[str, float]  # mole fractions
    phase: PhaseType
    
    def __post_init__(self):
        """Validate the thermodynamic state"""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        
        # Normalize composition
        total = sum(self.composition.values())
        if abs(total - 1.0) > 1e-6:
            self.composition = {k: v/total for k, v in self.composition.items()}


class ConvergenceError(Exception):
    """Exception raised when numerical methods fail to converge"""
    pass


# ============================================================================
# THERMODYNAMIC DATA STRUCTURES
# ============================================================================

@dataclass
class NISTThermodynamicData:
    """NIST thermodynamic data structure"""
    enthalpy_formation: Optional[float] = None    # ΔH°f (kJ/mol)
    gibbs_formation: Optional[float] = None       # ΔG°f (kJ/mol)
    entropy_standard: Optional[float] = None      # S° (J/(mol·K))
    heat_capacity: Optional[float] = None         # Cp (J/(mol·K))
    phase: Optional[str] = None                   # solid, liquid, gas
    temperature: float = 298.15                   # Temperature (K)
    source: str = "NIST WebBook"                  # data source
    cas_number: Optional[str] = None              # CAS registry number
    formula: Optional[str] = None                 # Chemical formula


# ============================================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================================

@dataclass
class EnvironmentalConditions:
    """Environmental conditions for chemical reactions"""
    temperature: float  # °C
    pressure: float     # atm
    depth: float        # m
    pH: float          # pH scale
    humidity: float    # %
    reaction_time: float  # hours
    catalyst_present: bool
    atmosphere: str    # 'air', 'CO2', 'vacuum', etc.
    stirring: bool
    light_exposure: str  # 'none', 'ambient', 'UV', 'sunlight'
    radiation_level: float = 0.0  # radiation exposure level
    magnetic_field: float = 0.0   # magnetic field strength


# Planetary environment presets removed to make the system planet-agnostic.
# Use EnvironmentalConditions directly and apply variations programmatically.

# ============================================================================
# COMPOUND AND VALIDATION DATA STRUCTURES
# ============================================================================

@dataclass
class CompoundSuggestion:
    """Represents a suggested chemical compound"""
    formula: str
    name: str
    compound_type: str  # 'molecular', 'ionic', 'metallic', 'network'
    stability: str      # 'high', 'medium', 'low'
    source: str = 'llm'
    confidence: Optional[float] = None
    
    # Data provenance fields
    schema_version: str = '1.0.0'
    method_version: str = '1.0.0'
    evidence_score: float = 0.5  # 0.0-1.0 confidence in data quality


@dataclass
class ValidationResult:
    """Result of compound validation"""
    compound: CompoundSuggestion
    feasible: bool
    formation_probability: float
    limiting_factors: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Data provenance fields
    schema_version: str = '1.0.0'
    method_version: str = '1.0.0'
    evidence_score: float = 0.5  # 0.0-1.0 confidence in validation quality
    data_sources: List[str] = field(default_factory=list)  # Track which data sources were used


# ============================================================================
# THERMODYNAMIC MODELS
# ============================================================================

class ThermodynamicModel(ABC):
    """Abstract base class for thermodynamic models"""
    
    @abstractmethod
    def fugacity_coefficient(self, state: ThermodynamicState, component: str) -> float:
        """Calculate fugacity coefficient for a component"""
        pass
    
    @abstractmethod
    def activity_coefficient(self, state: ThermodynamicState, component: str) -> float:
        """Calculate activity coefficient for a component"""
        pass
    
    @abstractmethod
    def compressibility_factor(self, state: ThermodynamicState) -> float:
        """Calculate compressibility factor"""
        pass


class EquationOfState(ThermodynamicModel):
    """Base class for equations of state"""
    
    def __init__(self, components: List[str]):
        self.components = components
        self.critical_properties = self._load_critical_properties()
        
    def _load_critical_properties(self) -> Dict[str, Dict[str, float]]:
        """Load critical properties for components"""
        # Default critical properties for common components
        properties = {
            'H2O': {'Tc': 647.1, 'Pc': 2.21e7, 'omega': 0.344},
            'CO2': {'Tc': 304.1, 'Pc': 7.38e6, 'omega': 0.225},
            'methane': {'Tc': 190.56, 'Pc': 4599200, 'omega': 0.011},
            'ethane': {'Tc': 305.32, 'Pc': 4872200, 'omega': 0.099},
            'propane': {'Tc': 369.83, 'Pc': 4248000, 'omega': 0.152},
            'n-butane': {'Tc': 425.12, 'Pc': 3796000, 'omega': 0.200},
            'water': {'Tc': 647.1, 'Pc': 22064000, 'omega': 0.345},
            'nitrogen': {'Tc': 126.19, 'Pc': 3395800, 'omega': 0.037},
            'carbon_dioxide': {'Tc': 304.13, 'Pc': 7377300, 'omega': 0.225}
        }
        return {comp: properties.get(comp, {'Tc': 500, 'Pc': 5000000, 'omega': 0.1}) 
                for comp in self.components}