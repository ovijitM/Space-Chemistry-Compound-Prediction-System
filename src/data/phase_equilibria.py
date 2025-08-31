"""Phase Equilibria System Module

This module contains classes for thermodynamic modeling and phase equilibria calculations,
including equations of state, phase stability analysis, and equilibrium calculations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

# Import data models
from .data_models import PhaseType, ThermodynamicState, ConvergenceError, ThermodynamicModel


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


class PengRobinsonEOS(EquationOfState):
    """Peng-Robinson equation of state implementation"""
    
    def __init__(self, components: List[str], binary_interaction_params: Optional[Dict] = None):
        super().__init__(components)
        self.kij = binary_interaction_params or {}
        
    def _alpha_function(self, component: str, temperature: float) -> float:
        """Calculate alpha function for PR EOS"""
        props = self.critical_properties[component]
        Tr = temperature / props['Tc']
        omega = props['omega']
        
        if omega <= 0.49:
            m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        else:
            m = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
            
        return (1 + m * (1 - np.sqrt(Tr)))**2
    
    def _mixing_rules(self, state: ThermodynamicState) -> Tuple[float, float]:
        """Calculate mixture parameters using van der Waals mixing rules"""
        R = 8.314  # J/(mol·K)
        
        # Calculate pure component parameters
        a_pure = {}
        b_pure = {}
        
        for comp in self.components:
            props = self.critical_properties[comp]
            alpha = self._alpha_function(comp, state.temperature)
            a_pure[comp] = 0.45724 * (R * props['Tc'])**2 / props['Pc'] * alpha
            b_pure[comp] = 0.07780 * R * props['Tc'] / props['Pc']
        
        # Mixing rules
        a_mix = 0.0
        b_mix = 0.0
        
        for i, comp_i in enumerate(self.components):
            xi = state.composition.get(comp_i, 0.0)
            b_mix += xi * b_pure[comp_i]
            
            for j, comp_j in enumerate(self.components):
                xj = state.composition.get(comp_j, 0.0)
                kij = self.kij.get((comp_i, comp_j), 0.0)
                aij = np.sqrt(a_pure[comp_i] * a_pure[comp_j]) * (1 - kij)
                a_mix += xi * xj * aij
        
        return a_mix, b_mix
    
    def compressibility_factor(self, state: ThermodynamicState) -> float:
        """Calculate compressibility factor using PR EOS"""
        R = 8.314  # J/(mol·K)
        a_mix, b_mix = self._mixing_rules(state)
        
        A = a_mix * state.pressure / (R * state.temperature)**2
        B = b_mix * state.pressure / (R * state.temperature)
        
        # Solve cubic equation: Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
        coeffs = [1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)]
        roots = np.roots(coeffs)
        
        # Select appropriate root (largest for vapor, smallest for liquid)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        
        if not real_roots:
            raise ConvergenceError("No valid compressibility factor found")
        
        if state.phase == PhaseType.VAPOR:
            return max(real_roots)
        else:
            return min(real_roots)
    
    def fugacity_coefficient(self, state: ThermodynamicState, component: str) -> float:
        """Calculate fugacity coefficient for a component"""
        R = 8.314  # J/(mol·K)
        a_mix, b_mix = self._mixing_rules(state)
        Z = self.compressibility_factor(state)
        
        A = a_mix * state.pressure / (R * state.temperature)**2
        B = b_mix * state.pressure / (R * state.temperature)
        
        # Component-specific parameters
        props = self.critical_properties[component]
        alpha = self._alpha_function(component, state.temperature)
        ai = 0.45724 * (R * props['Tc'])**2 / props['Pc'] * alpha
        bi = 0.07780 * R * props['Tc'] / props['Pc']
        
        # Calculate partial derivatives
        sum_term = 0.0
        for comp_j in self.components:
            xj = state.composition.get(comp_j, 0.0)
            kij = self.kij.get((component, comp_j), 0.0)
            props_j = self.critical_properties[comp_j]
            alpha_j = self._alpha_function(comp_j, state.temperature)
            aj = 0.45724 * (R * props_j['Tc'])**2 / props_j['Pc'] * alpha_j
            aij = np.sqrt(ai * aj) * (1 - kij)
            sum_term += xj * aij
        
        Bi = bi * state.pressure / (R * state.temperature)
        
        ln_phi = (Bi/B) * (Z - 1) - np.log(Z - B) - \
                 (A/(2*np.sqrt(2)*B)) * (2*sum_term/a_mix - Bi/B) * \
                 np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B))
        
        return np.exp(ln_phi)
    
    def activity_coefficient(self, state: ThermodynamicState, component: str) -> float:
        """Activity coefficient (unity for EOS models)"""
        return 1.0


class PhaseStabilityAnalyzer:
    """Performs phase stability analysis and equilibrium calculations"""
    
    def __init__(self, thermo_model: ThermodynamicModel):
        self.thermo_model = thermo_model
        self.tolerance = 1e-8
        self.max_iterations = 100
    
    def tangent_plane_distance(self, state: ThermodynamicState, 
                             trial_composition: Dict[str, float]) -> float:
        """Calculate tangent plane distance for stability analysis"""
        # Create trial state
        trial_state = ThermodynamicState(
            temperature=state.temperature,
            pressure=state.pressure,
            composition=trial_composition,
            phase=state.phase
        )
        
        tpd = 0.0
        for component in state.composition:
            if component in trial_composition and trial_composition[component] > 0:
                # Chemical potential difference
                mu_trial = np.log(trial_composition[component] * 
                                self.thermo_model.fugacity_coefficient(trial_state, component))
                mu_feed = np.log(state.composition[component] * 
                               self.thermo_model.fugacity_coefficient(state, component))
                
                tpd += trial_composition[component] * (mu_trial - mu_feed)
        
        return tpd
    
    def is_stable(self, state: ThermodynamicState) -> bool:
        """Test phase stability using tangent plane analysis"""
        n_components = len(state.composition)
        
        # Test multiple trial compositions
        for _ in range(10):  # Multiple random starting points
            # Generate random trial composition
            trial_fractions = np.random.dirichlet(np.ones(n_components))
            trial_composition = {comp: frac for comp, frac in 
                               zip(state.composition.keys(), trial_fractions)}
            
            try:
                tpd = self.tangent_plane_distance(state, trial_composition)
                if tpd < -self.tolerance:
                    return False  # Unstable
            except Exception as e:
                logging.debug(f"TPD calculation failed: {e}")
                continue
        
        return True  # Stable


class PhaseEquilibriaSystem:
    """Main class for phase equilibria calculations"""
    
    def __init__(self, components: List[str], thermo_model: str = 'peng_robinson'):
        self.components = components
        
        # Initialize thermodynamic model
        if thermo_model == 'peng_robinson':
            self.thermo_model = PengRobinsonEOS(components)
        else:
            self.thermo_model = PengRobinsonEOS(components)  # Default fallback
            
        self.stability_analyzer = PhaseStabilityAnalyzer(self.thermo_model)
    
    def calculate_phase_stability(self, temperature: float, pressure: float, 
                                composition: Dict[str, float]) -> float:
        """Calculate phase stability factor for given conditions"""
        try:
            # Create thermodynamic state
            state = ThermodynamicState(
                temperature=temperature,
                pressure=pressure,
                composition=composition,
                phase=PhaseType.LIQUID  # Default assumption
            )
            
            # Check stability
            is_stable = self.stability_analyzer.is_stable(state)
            
            # Calculate stability factor based on reduced conditions
            stability_factor = 1.0 if is_stable else 0.5
            
            # Apply temperature and pressure corrections
            for comp in composition:
                if comp in self.thermo_model.critical_properties:
                    props = self.thermo_model.critical_properties[comp]
                    Tr = temperature / props['Tc']
                    Pr = pressure / props['Pc']
                    
                    # Adjust stability based on reduced conditions
                    if Tr > 1.5 or Pr > 10.0:
                        stability_factor *= 0.8
                    elif 0.5 <= Tr <= 1.2 and 0.1 <= Pr <= 5.0:
                        stability_factor *= 1.1
            
            return min(1.0, max(0.1, stability_factor))
            
        except Exception as e:
            logging.debug(f"Phase stability calculation error: {e}")
            return 0.8  # Default moderate stability