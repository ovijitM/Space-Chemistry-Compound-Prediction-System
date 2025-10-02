#!/usr/bin/env python3
"""
Enhanced Drilling Analysis System
================================

Provides detailed element and compound analysis for each drilling location
with Earth validation, abundance calculations, and comprehensive reporting.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from src.data.planetary_element_sampler import PlanetaryElementSampler, PlanetaryEnvironment
from src.utils.random_compound_generator import RandomCompoundGenerator
from src.core.enhanced_chemistry_validator import EnhancedChemistryValidator
from src.data.data_models import EnvironmentalConditions


@dataclass
class ElementAnalysis:
    """Detailed analysis of a single element at a location"""
    symbol: str
    quantity: float
    percentage: float
    earth_exists: bool
    earth_abundance_ppm: Optional[float]
    space_vs_earth_ratio: Optional[float]
    rarity_classification: str  # abundant, common, rare, ultra_rare


@dataclass
class CompoundAnalysis:
    """Detailed analysis of a compound at a location"""
    formula: str
    name: str
    formation_probability: float
    earth_exists: bool
    earth_databases_found: List[str]
    validation_confidence: float
    stability_rating: str
    classification: str  # known, unknown, theoretical


@dataclass
class LocationDetailedReport:
    """Comprehensive report for a single drilling location"""
    site_id: str
    coordinates: Tuple[float, float]
    depth: float
    site_type: str
    drilling_timestamp: datetime
    
    # Environmental data
    temperature: float
    pressure: float
    pH: float
    humidity: float
    radiation_level: float
    
    # Element analysis
    total_elements_found: int
    element_analyses: List[ElementAnalysis]
    dominant_elements: List[str]  # Top 5 most abundant
    rare_elements: List[str]      # Elements with <1% abundance
    unknown_elements: List[str]   # Not found on Earth
    
    # Compound analysis
    total_compounds_tested: int
    compound_analyses: List[CompoundAnalysis]
    known_compounds: List[str]    # Found on Earth
    unknown_compounds: List[str]  # Not found on Earth
    theoretical_compounds: List[str]  # Theoretically possible
    
    # Summary statistics
    earth_similarity_score: float  # 0-100% how similar to Earth
    discovery_potential: float     # 0-100% chance of new discoveries
    scientific_value: float        # 0-100% scientific importance


class EnhancedDrillingAnalyzer:
    """Provides enhanced analysis of drilling results with Earth validation"""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.element_sampler = PlanetaryElementSampler(random_seed=random_seed)
        self.compound_generator = RandomCompoundGenerator()  # Remove random_seed parameter
        self.validator = EnhancedChemistryValidator()
        
        # Unknown element naming counter
        self.unknown_element_counter = 0
        self.unknown_element_names = {}
        
        # Earth element abundance data (simplified)
        self.earth_element_abundance = {
            'O': 461000,   # Oxygen - most abundant
            'Si': 282000,  # Silicon  
            'Al': 82300,   # Aluminum
            'Fe': 56300,   # Iron
            'Ca': 41500,   # Calcium
            'Na': 28400,   # Sodium
            'K': 25900,    # Potassium
            'Mg': 23300,   # Magnesium
            'Ti': 5650,    # Titanium
            'H': 1400,     # Hydrogen
            'P': 1050,     # Phosphorus
            'Mn': 950,     # Manganese
            'F': 585,      # Fluorine
            'Ba': 425,     # Barium
            'Sr': 375,     # Strontium
            'S': 350,      # Sulfur
            'C': 200,      # Carbon
            'Zr': 165,     # Zirconium
            'V': 120,      # Vanadium
            'Cl': 145,     # Chlorine
            'Cr': 102,     # Chromium
            'Rb': 90,      # Rubidium
            'Ni': 84,      # Nickel
            'Zn': 70,      # Zinc
            'Ce': 66,      # Cerium
            'Cu': 60,      # Copper
            'Y': 33,       # Yttrium
            'Nd': 28,      # Neodymium
            'Co': 25,      # Cobalt
            'Sc': 22,      # Scandium
            'Li': 20,      # Lithium
            'N': 19,       # Nitrogen
            'Nb': 20,      # Niobium
            'Ga': 19,      # Gallium
            'Pb': 14,      # Lead
            'B': 10,       # Boron
            'Th': 9.6,     # Thorium
            'Pr': 9.2,     # Praseodymium
            'Sm': 7.05,    # Samarium
            'Gd': 6.2,     # Gadolinium
            'Dy': 5.2,     # Dysprosium
            'Er': 3.5,     # Erbium
            'Yb': 3.2,     # Ytterbium
            'Hf': 3.0,     # Hafnium
            'Cs': 3.0,     # Cesium
            'U': 2.7,      # Uranium
            'As': 1.8,     # Arsenic
            'Mo': 1.2,     # Molybdenum
            'Eu': 2.0,     # Europium
            'Tb': 1.2,     # Terbium
            'Ho': 1.3,     # Holmium
            'Tm': 0.52,    # Thulium
            'Lu': 0.8,     # Lutetium
            'Ta': 2.0,     # Tantalum
            'W': 1.25,     # Tungsten
            'Tl': 0.85,    # Thallium
            'Bi': 0.009,   # Bismuth
            'In': 0.25,    # Indium
            'Cd': 0.15,    # Cadmium
            'Ag': 0.075,   # Silver
            'Sb': 0.2,     # Antimony
            'I': 0.45,     # Iodine
            'Te': 0.001,   # Tellurium
            'Sn': 2.3,     # Tin
            'Br': 2.4,     # Bromine
            'Se': 0.05,    # Selenium
            'Ge': 1.5,     # Germanium
            'Kr': 0.001,   # Krypton
            'Xe': 0.00003, # Xenon
            'Ar': 3.5,     # Argon
            'Ne': 0.005,   # Neon
            'He': 0.008,   # Helium
            'Rn': 0.0000004, # Radon
        }
    
    def get_element_display_name(self, element_symbol: str, earth_exists: bool) -> str:
        """Get display name for element - anonymize unknown elements"""
        if earth_exists:
            return element_symbol  # Show real name for known elements
        else:
            # Generate anonymous name for unknown elements
            if element_symbol not in self.unknown_element_names:
                self.unknown_element_counter += 1
                self.unknown_element_names[element_symbol] = f"Unknown-{chr(64 + self.unknown_element_counter)}"  # Unknown-A, Unknown-B, etc.
            return self.unknown_element_names[element_symbol]
    
    def analyze_elements(self, detected_elements: Dict[str, float], 
                        location_id: str) -> List[ElementAnalysis]:
        """Perform detailed analysis of detected elements"""
        
        total_quantity = sum(detected_elements.values())
        element_analyses = []
        
        for element, quantity in detected_elements.items():
            # Calculate percentage
            percentage = (quantity / total_quantity) * 100 if total_quantity > 0 else 0
            
            # Check Earth existence and abundance
            earth_exists = element in self.earth_element_abundance
            earth_abundance = self.earth_element_abundance.get(element, None)
            
            # Calculate space vs Earth ratio
            space_vs_earth_ratio = None
            if earth_exists and earth_abundance:
                space_concentration = quantity * 1000  # Convert to ppm equivalent
                space_vs_earth_ratio = space_concentration / earth_abundance
            
            # Classify rarity
            if percentage >= 10:
                rarity = "abundant"
            elif percentage >= 5:
                rarity = "common"
            elif percentage >= 1:
                rarity = "uncommon"
            elif percentage >= 0.1:
                rarity = "rare"
            else:
                rarity = "ultra_rare"
            
            analysis = ElementAnalysis(
                symbol=element,
                quantity=quantity,
                percentage=percentage,
                earth_exists=earth_exists,
                earth_abundance_ppm=earth_abundance,
                space_vs_earth_ratio=space_vs_earth_ratio,
                rarity_classification=rarity
            )
            element_analyses.append(analysis)
        
        # Sort by abundance
        element_analyses.sort(key=lambda x: x.percentage, reverse=True)
        return element_analyses
    
    def analyze_compounds(self, generated_compounds: List, validated_compounds: List,
                         location_id: str) -> List[CompoundAnalysis]:
        """Perform detailed analysis of generated compounds"""
        
        compound_analyses = []
        
        for i, compound in enumerate(generated_compounds):
            # Get validation data
            validation_data = validated_compounds[i] if i < len(validated_compounds) else {}
            
            # Determine Earth databases that found this compound
            databases_found = []
            database_validation = validation_data.get('database_validation', {})
            
            if database_validation.get('pubchem', {}).get('valid', False):
                databases_found.append('PubChem')
            if database_validation.get('pymatgen', {}).get('valid', False):
                databases_found.append('PyMatGen')
            if database_validation.get('nist', {}).get('valid', False):
                databases_found.append('NIST')
            
            earth_exists = len(databases_found) > 0
            
            # Classify compound
            if earth_exists:
                if len(databases_found) >= 2:
                    classification = "known"
                else:
                    classification = "known"
            else:
                if validation_data.get('feasible', False):
                    classification = "theoretical"
                else:
                    classification = "unknown"
            
            # Determine stability rating
            formation_prob = validation_data.get('formation_probability', 0.0)
            if formation_prob >= 0.8:
                stability = "highly_stable"
            elif formation_prob >= 0.6:
                stability = "stable"
            elif formation_prob >= 0.4:
                stability = "moderately_stable"
            elif formation_prob >= 0.2:
                stability = "unstable"
            else:
                stability = "highly_unstable"
            
            analysis = CompoundAnalysis(
                formula=compound.formula,
                name=compound.name,
                formation_probability=formation_prob,
                earth_exists=earth_exists,
                earth_databases_found=databases_found,
                validation_confidence=validation_data.get('confidence', 0.0),
                stability_rating=stability,
                classification=classification
            )
            compound_analyses.append(analysis)
        
        return compound_analyses
    
    def calculate_earth_similarity_score(self, element_analyses: List[ElementAnalysis],
                                        compound_analyses: List[CompoundAnalysis]) -> float:
        """Calculate how similar this location is to Earth"""
        
        # Element similarity (50% weight)
        earth_element_count = sum(1 for ea in element_analyses if ea.earth_exists)
        total_elements = len(element_analyses)
        element_similarity = (earth_element_count / total_elements) * 50 if total_elements > 0 else 0
        
        # Compound similarity (50% weight)
        earth_compound_count = sum(1 for ca in compound_analyses if ca.earth_exists)
        total_compounds = len(compound_analyses)
        compound_similarity = (earth_compound_count / total_compounds) * 50 if total_compounds > 0 else 0
        
        return element_similarity + compound_similarity
    
    def calculate_discovery_potential(self, element_analyses: List[ElementAnalysis],
                                    compound_analyses: List[CompoundAnalysis]) -> float:
        """Calculate potential for new discoveries based on rare element abundance"""
        
        # Low abundance elements found in high concentrations (30% weight)
        low_abundance_count = sum(1 for ea in element_analyses 
                                 if ea.earth_abundance_ppm and ea.earth_abundance_ppm < 10 and ea.percentage > 1.0)
        total_elements = len(element_analyses)
        low_abundance_score = (low_abundance_count / total_elements) * 30 if total_elements > 0 else 0
        
        # Unknown compounds (40% weight)
        unknown_compound_count = sum(1 for ca in compound_analyses if ca.classification == "unknown")
        total_compounds = len(compound_analyses)
        unknown_compound_score = (unknown_compound_count / total_compounds) * 40 if total_compounds > 0 else 0
        
        # Theoretical compounds (30% weight)
        theoretical_compound_count = sum(1 for ca in compound_analyses if ca.classification == "theoretical")
        theoretical_score = (theoretical_compound_count / total_compounds) * 30 if total_compounds > 0 else 0
        
        return low_abundance_score + unknown_compound_score + theoretical_score
    
    def generate_detailed_location_report(self, location_data, compounds_per_site: int = 8) -> LocationDetailedReport:
        """Generate comprehensive detailed report for a single location"""
        
        site = location_data.drill_site
        env_profile = location_data.environmental_profile
        detected_elements = location_data.detected_elements
        
        print(f"   🔬 Generating detailed analysis for {site.site_id}...")
        
        # Generate compounds for this location
        generated_compounds = self.compound_generator.generate_random_compounds(
            detected_elements, num_compounds=compounds_per_site
        )
        
        # Create environmental conditions
        env_conditions = EnvironmentalConditions(
            temperature=env_profile.temperature,
            pressure=env_profile.pressure,
            depth=env_profile.depth,
            pH=env_profile.pH,
            humidity=env_profile.humidity,
            reaction_time=1.0,
            catalyst_present=False,
            atmosphere=list(env_profile.atmosphere_composition.keys())[0],
            stirring=False,
            light_exposure="solar" if env_profile.solar_irradiance > 500 else "minimal"
        )
        
        # Validate compounds
        validated_compounds = []
        for compound in generated_compounds:
            try:
                validation_result = self.validator.validate_compound_feasibility(
                    compound, detected_elements, env_conditions
                )
                validated_compounds.append(validation_result)
            except Exception as e:
                print(f"         ⚠️ Validation failed for {compound.formula}: {e}")
                validated_compounds.append({})
        
        # Perform detailed analyses
        element_analyses = self.analyze_elements(detected_elements, site.site_id)
        compound_analyses = self.analyze_compounds(generated_compounds, validated_compounds, site.site_id)
        
        # Extract summary lists
        dominant_elements = [ea.symbol for ea in element_analyses[:5]]  # Top 5
        rare_elements = [ea.symbol for ea in element_analyses if ea.percentage < 1.0]
        unknown_elements = [ea.symbol for ea in element_analyses if not ea.earth_exists]
        
        known_compounds = [ca.formula for ca in compound_analyses if ca.earth_exists]
        unknown_compounds = [ca.formula for ca in compound_analyses if ca.classification == "unknown"]
        theoretical_compounds = [ca.formula for ca in compound_analyses if ca.classification == "theoretical"]
        
        # Calculate scores
        earth_similarity = self.calculate_earth_similarity_score(element_analyses, compound_analyses)
        discovery_potential = self.calculate_discovery_potential(element_analyses, compound_analyses)
        scientific_value = (discovery_potential + (100 - earth_similarity)) / 2  # Balance novelty vs familiarity
        
        return LocationDetailedReport(
            site_id=site.site_id,
            coordinates=site.coordinates,
            depth=site.depth,
            site_type=site.site_type,
            drilling_timestamp=site.drilling_time,
            temperature=env_profile.temperature,
            pressure=env_profile.pressure,
            pH=env_profile.pH,
            humidity=env_profile.humidity,
            radiation_level=env_profile.radiation_level,
            total_elements_found=len(detected_elements),
            element_analyses=element_analyses,
            dominant_elements=dominant_elements,
            rare_elements=rare_elements,
            unknown_elements=unknown_elements,
            total_compounds_tested=len(generated_compounds),
            compound_analyses=compound_analyses,
            known_compounds=known_compounds,
            unknown_compounds=unknown_compounds,
            theoretical_compounds=theoretical_compounds,
            earth_similarity_score=earth_similarity,
            discovery_potential=discovery_potential,
            scientific_value=scientific_value
        )
    
    def print_detailed_location_report(self, report: LocationDetailedReport):
        """Print beautiful detailed report for a location"""
        
        print(f"\n" + "🔬" + "=" * 69)
        print(f"📍 DETAILED ANALYSIS: {report.site_id}")
        print(f"🗺️ Location: {report.site_type} at ({report.coordinates[0]:.2f}, {report.coordinates[1]:.2f})")
        print(f"⚡ Depth: {report.depth:.1f}m | 🌡️ Temp: {report.temperature:.1f}°C | pH: {report.pH:.1f}")
        print("=" * 70)
        
        # Element Analysis
        print(f"🧪 ELEMENT DISCOVERY ({report.total_elements_found} elements found):")
        print(f"   Dominant Elements: {', '.join(report.dominant_elements)}")
        
        print(f"\n   📊 Element Breakdown:")
        for ea in report.element_analyses[:8]:  # Show top 8
            earth_status = "🌍" if ea.earth_exists else "❓"
            display_name = self.get_element_display_name(ea.symbol, ea.earth_exists)
            ratio_text = f"({ea.space_vs_earth_ratio:.1f}x Earth)" if ea.space_vs_earth_ratio else ""
            print(f"      {earth_status} {display_name}: {ea.percentage:.1f}% ({ea.rarity_classification}) {ratio_text}")
        
        if report.unknown_elements:
            unknown_display_names = [self.get_element_display_name(elem, False) for elem in report.unknown_elements]
            print(f"   ❓ Unknown Elements (not found on Earth): {', '.join(unknown_display_names)}")
        
        # Compound Analysis
        print(f"\n⚗️ COMPOUND FORMATION ({report.total_compounds_tested} compounds tested):")
        print(f"   🌍 Known on Earth: {len(report.known_compounds)} compounds")
        print(f"   ❓ Unknown: {len(report.unknown_compounds)} compounds") 
        print(f"   🧬 Theoretical: {len(report.theoretical_compounds)} compounds")
        
        print(f"\n   📈 Top Compounds:")
        for ca in sorted(report.compound_analyses, key=lambda x: x.formation_probability, reverse=True)[:5]:
            status_emoji = "🌍" if ca.earth_exists else ("🧬" if ca.classification == "theoretical" else "❓")
            databases = f"[{', '.join(ca.earth_databases_found)}]" if ca.earth_databases_found else ""
            print(f"      {status_emoji} {ca.formula}: {ca.formation_probability:.1f}% probability {databases}")
        
        # Summary Scores
        print(f"\n📊 ANALYSIS SCORES:")
        print(f"   🌍 Earth Similarity: {report.earth_similarity_score:.1f}%")
        print(f"   🔍 Discovery Potential: {report.discovery_potential:.1f}%") 
        print(f"   🏆 Scientific Value: {report.scientific_value:.1f}%")
        print("=" * 70)


def main():
    """Demo the enhanced drilling analyzer"""
    
    from src.core.multi_location_survey import MultiLocationDrillingSurvey
    from src.data.planetary_element_sampler import PlanetaryEnvironment
    
    print("🚀 ENHANCED DRILLING ANALYSIS SYSTEM")
    print("=" * 50)
    
    # Conduct survey
    survey = MultiLocationDrillingSurvey(random_seed=42)
    location_data, cross_analysis = survey.conduct_full_survey(
        num_drilling_sites=3,
        environment=PlanetaryEnvironment.MARS_SURFACE,
        compounds_per_site=6
    )
    
    # Enhanced analysis
    analyzer = EnhancedDrillingAnalyzer(random_seed=42)
    
    print(f"\n🔬 CONDUCTING ENHANCED ANALYSIS...")
    detailed_reports = []
    
    for data in location_data:
        report = analyzer.generate_detailed_location_report(data, compounds_per_site=6)
        detailed_reports.append(report)
        analyzer.print_detailed_location_report(report)
    
    print(f"\n🏁 Enhanced analysis complete for {len(detailed_reports)} locations!")


if __name__ == "__main__":
    main()