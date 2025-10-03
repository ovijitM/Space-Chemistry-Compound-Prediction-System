"""
Multi-Location Drilling Survey System
====================================

Simulates a rover drilling at multiple locations in UNKNOWN ENVIRONMENTS.
This implements the universal vision: 
- Work in ANY unknown environment (no predefined constraints)
- Drill multiple random locations 
- Collect elements + environmental data for each location
- Perform cross-location analysis within that environment
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid

from src.data.planetary_element_sampler import PlanetaryElementSampler, PlanetaryEnvironment
from src.utils.random_compound_generator import RandomCompoundGenerator
from src.core.enhanced_chemistry_validator import EnhancedChemistryValidator
from src.data.data_models import EnvironmentalConditions, CompoundSuggestion
from src.config.configuration import ConfigurationManager


@dataclass
class DrillSite:
    """Represents a single drilling location"""
    site_id: str
    coordinates: Tuple[float, float]  # (latitude, longitude) 
    depth: float  # drilling depth in meters
    site_type: str  # 'surface', 'subsurface', 'crater', 'slope', etc.
    drilling_time: datetime


@dataclass
class EnvironmentalProfile:
    """Complete environmental data for a drilling site"""
    temperature: float  # °C
    pressure: float     # atm
    pH: float
    humidity: float     # %
    radiation_level: float  # background radiation
    magnetic_field: float   # local magnetic field strength
    atmosphere_composition: Dict[str, float]  # gas percentages
    soil_moisture: float    # %
    wind_speed: float      # m/s
    solar_irradiance: float # W/m²
    depth: float           # drilling depth
    geological_context: str # 'sedimentary', 'igneous', 'metamorphic', etc.


@dataclass
class LocationData:
    """Complete data from one drilling location"""
    drill_site: DrillSite
    environmental_profile: EnvironmentalProfile
    detected_elements: Dict[str, float]  # element -> quantity
    generated_compounds: List[CompoundSuggestion]
    validated_compounds: List[Dict]  # validation results
    formation_feasibility: Dict[str, float]  # compound -> probability
    earth_matches: List[str]  # compounds that exist on Earth
    unique_compounds: List[str]  # compounds not found on Earth


@dataclass
class CrossLocationAnalysis:
    """Analysis across all drilling locations"""
    total_locations: int
    common_elements: Dict[str, List[float]]  # element -> [quantities across sites]
    location_specific_elements: Dict[str, List[str]]  # site_id -> unique elements
    common_compounds: Dict[str, int]  # compound -> number of sites where found
    compound_formation_map: Dict[str, Dict[str, float]]  # compound -> {site_id: probability}
    environmental_correlations: Dict[str, float]  # environmental factor -> correlation with compound formation
    element_pool_distribution: Dict[str, Dict[str, float]]  # spatial distribution analysis


class MultiLocationDrillingSurvey:
    """Main system for conducting multi-location drilling surveys"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.element_sampler = PlanetaryElementSampler(random_seed)
        self.compound_generator = RandomCompoundGenerator()
        self.config = ConfigurationManager()
        self.validator = EnhancedChemistryValidator(config=self.config.config)
        
    def generate_drilling_sites(self, 
                               num_sites: int = 8,
                               area_size: float = 10.0) -> List[DrillSite]:
        """Generate realistic drilling sites within a survey area"""
        
        sites = []
        base_time = datetime.now()
        
        # Universal site types for any unknown environment
        site_types = ['surface', 'crater', 'ridge', 'valley', 'plateau', 'depression', 
                     'slope', 'plain', 'anomaly', 'subsurface', 'elevation', 'cavity']
        
        for i in range(num_sites):
            # Generate coordinates within the survey area
            lat = random.uniform(-area_size/2, area_size/2)
            lon = random.uniform(-area_size/2, area_size/2)
            
            # Generate realistic drilling depth (universal approach)
            depth = random.uniform(0.1, 10.0)  # Universal depth range for any material
            
            site = DrillSite(
                site_id=f"DRILL_{i+1:02d}_{uuid.uuid4().hex[:6]}",
                coordinates=(lat, lon),
                depth=depth,
                site_type=random.choice(site_types),
                drilling_time=base_time + timedelta(hours=i*2)  # 2 hours between drills
            )
            sites.append(site)
        
        return sites
    
    def collect_environmental_data(self, 
                                  drill_site: DrillSite) -> EnvironmentalProfile:
        """Collect comprehensive environmental data at a drilling site"""
        
        # Universal environmental parameters (could be any unknown environment)
        # Generate completely random but physically plausible values
        base_temp = random.uniform(-250, 100)  # Extreme cold to moderate hot
        base_pressure = random.uniform(0.0, 3.0)  # Vacuum to high pressure
        
        # Random atmosphere composition
        gas_types = ["CO2", "N2", "Ar", "O2", "CO", "H2O", "H2", "CH4", "NH3", "Unknown_Gas_A", "Unknown_Gas_B"]
        atmosphere = {}
        remaining = 100.0
        for i, gas in enumerate(random.sample(gas_types, random.randint(1, 4))):
            if i == len(gas_types) - 1 or remaining <= 0:
                atmosphere[gas] = max(0, remaining)
                break
            else:
                percentage = random.uniform(0, remaining * 0.8)
                atmosphere[gas] = percentage
                remaining -= percentage
        
        # Normalize atmosphere to 100%
        total = sum(atmosphere.values())
        if total > 0:
            atmosphere = {gas: (pct/total)*100 for gas, pct in atmosphere.items()}
        
        # Add site-specific variations
        temp_variation = random.uniform(-10, 10)
        depth_temp_effect = drill_site.depth * random.uniform(0.5, 2.0)  # deeper = warmer
        
        # Site type effects
        if drill_site.site_type in ['crater', 'crater_rim']:
            temp_variation += random.uniform(-5, 15)  # craters can be warmer
        elif drill_site.site_type in ['slope', 'ridge']:
            temp_variation += random.uniform(-8, 5)   # slopes cooler
        
        profile = EnvironmentalProfile(
            temperature=base_temp + temp_variation + depth_temp_effect,
            pressure=base_pressure * random.uniform(0.8, 1.2),
            pH=random.uniform(6.0, 9.0),
            humidity=random.uniform(0.0, 100.0),  # Universal humidity range
            radiation_level=random.uniform(0.1, 10.0),
            magnetic_field=random.uniform(0.0, 2.0),
            atmosphere_composition=atmosphere,
            soil_moisture=random.uniform(0.0, 15.0),
            wind_speed=random.uniform(0.0, 50.0),
            solar_irradiance=random.uniform(100.0, 1400.0),
            depth=drill_site.depth,
            geological_context=random.choice(['sedimentary', 'igneous', 'metamorphic', 'mixed'])
        )
        
        return profile
    
    def drill_and_analyze_location(self, 
                                  drill_site: DrillSite,
                                  compounds_to_test: int = 8) -> LocationData:
        """Complete drilling and analysis for one location"""
        
        print(f"   🗿 Drilling at {drill_site.site_id} ({drill_site.site_type}, depth: {drill_site.depth:.1f}m)")
        
        # Collect environmental data
        env_profile = self.collect_environmental_data(drill_site)
        
        # Detect elements at this location (universal approach)
        detected_elements = self.element_sampler.simulate_universal_scan(
            scan_intensity=random.uniform(0.7, 1.3),
            min_elements=4,
            max_elements=15
        )
        
        print(f"      📊 Elements: {len(detected_elements)} types, Temp: {env_profile.temperature:.1f}°C, pH: {env_profile.pH:.1f}")
        
        # Generate compounds from detected elements
        generated_compounds = self.compound_generator.generate_random_compounds(
            detected_elements, 
            num_compounds=compounds_to_test
        )
        
        # Create environmental conditions for validation
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
        
        # Validate each compound
        validated_compounds = []
        formation_feasibility = {}
        earth_matches = []
        unique_compounds = []
        
        for compound in generated_compounds:
            try:
                validation_result = self.validator.validate_compound_feasibility(
                    compound, detected_elements, env_conditions
                )
                
                # Check Earth existence
                database_validation = validation_result.get('database_validation', {})
                earth_exists = (database_validation.get('pubchem', {}).get('valid', False) or 
                              database_validation.get('pymatgen', {}).get('valid', False))
                
                validated_compounds.append({
                    'formula': compound.formula,
                    'name': compound.name,
                    'earth_exists': earth_exists,
                    'feasible': validation_result.get('feasible', False),
                    'formation_probability': validation_result.get('formation_probability', 0.0),
                    'confidence': validation_result.get('confidence', 0.0)
                })
                
                formation_feasibility[compound.formula] = validation_result.get('formation_probability', 0.0)
                
                if earth_exists:
                    earth_matches.append(compound.formula)
                else:
                    unique_compounds.append(compound.formula)
                    
            except Exception as e:
                print(f"         ⚠️ Validation failed for {compound.formula}: {e}")
                continue
        
        print(f"      🧪 Tested: {len(validated_compounds)} compounds, Earth matches: {len(earth_matches)}, Unknown: {len(unique_compounds)}")
        
        return LocationData(
            drill_site=drill_site,
            environmental_profile=env_profile,
            detected_elements=detected_elements,
            generated_compounds=generated_compounds,
            validated_compounds=validated_compounds,
            formation_feasibility=formation_feasibility,
            earth_matches=earth_matches,
            unique_compounds=unique_compounds
        )
    
    def conduct_full_survey(self,
                           num_drilling_sites: int = 7,
                           compounds_per_site: int = 8) -> Tuple[List[LocationData], CrossLocationAnalysis]:
        """Conduct complete multi-location drilling survey"""
        
        print(f"🚀 Starting Multi-Location Drilling Survey")
        print(f"🌍 Environment: Unknown Universal Environment")
        print(f"🗿 Drilling Sites: {num_drilling_sites}")
        print(f"⚗️ Compounds per Site: {compounds_per_site}")
        print("=" * 70)
        
        # Generate drilling sites
        drill_sites = self.generate_drilling_sites(num_drilling_sites)
        
        # Drill and analyze each location
        location_data = []
        for i, site in enumerate(drill_sites, 1):
            print(f"\n📍 Location {i}/{num_drilling_sites}")
            data = self.drill_and_analyze_location(site, compounds_per_site)
            location_data.append(data)
        
        # Perform cross-location analysis
        print(f"\n🔍 Performing Cross-Location Analysis...")
        cross_analysis = self.analyze_across_locations(location_data)
        
        return location_data, cross_analysis
    
    def conduct_unified_survey(self,
                              num_drilling_sites: int = 7,
                              compounds_per_site: int = 8,
                              enhanced_analyzer=None) -> Tuple[List[LocationData], List]:
        """Conduct unified drilling survey with immediate detailed analysis"""
        
        print(f"🚀 Starting Unified Drilling & Analysis Survey")
        print(f"🌍 Environment: Unknown Universal Environment")
        print(f"🗿 Drilling Sites: {num_drilling_sites}")
        print(f"⚗️ Compounds per Site: {compounds_per_site}")
        print("=" * 70)
        
        # Generate drilling sites
        drill_sites = self.generate_drilling_sites(num_drilling_sites)
        
        # Drill and analyze each location with immediate detailed analysis
        location_data = []
        detailed_reports = []
        
        for i, site in enumerate(drill_sites, 1):
            print(f"\\n📍 Location {i}/{num_drilling_sites}")
            
            # Basic drilling and analysis
            data = self.drill_and_analyze_location(site, compounds_per_site)
            location_data.append(data)
            
            # Immediately generate and show detailed analysis
            if enhanced_analyzer:
                detailed_report = enhanced_analyzer.generate_detailed_location_report(
                    data, compounds_per_site=compounds_per_site
                )
                detailed_reports.append(detailed_report)
                
                # Print the detailed analysis immediately
                enhanced_analyzer.print_detailed_location_report(detailed_report)
        
        return location_data, detailed_reports
    
    def analyze_across_locations(self, location_data: List[LocationData]) -> CrossLocationAnalysis:
        """Analyze patterns and correlations across all drilling locations"""
        
        # Collect all elements across locations
        all_elements = set()
        for data in location_data:
            all_elements.update(data.detected_elements.keys())
        
        # Find common elements
        common_elements = {}
        for element in all_elements:
            quantities = []
            for data in location_data:
                if element in data.detected_elements:
                    quantities.append(data.detected_elements[element])
                else:
                    quantities.append(0.0)
            common_elements[element] = quantities
        
        # Find location-specific elements
        location_specific_elements = {}
        for data in location_data:
            site_id = data.drill_site.site_id
            unique_to_site = []
            for element in data.detected_elements:
                # Count how many sites have this element
                sites_with_element = sum(1 for d in location_data if element in d.detected_elements)
                if sites_with_element == 1:  # Only found at this site
                    unique_to_site.append(element)
            location_specific_elements[site_id] = unique_to_site
        
        # Find common compounds
        all_compounds = set()
        for data in location_data:
            all_compounds.update([c['formula'] for c in data.validated_compounds])
        
        common_compounds = {}
        compound_formation_map = {}
        
        for compound in all_compounds:
            sites_with_compound = 0
            formation_probs = {}
            
            for data in location_data:
                site_id = data.drill_site.site_id
                for validated_compound in data.validated_compounds:
                    if validated_compound['formula'] == compound:
                        sites_with_compound += 1
                        formation_probs[site_id] = validated_compound['formation_probability']
                        break
                else:
                    formation_probs[site_id] = 0.0
            
            common_compounds[compound] = sites_with_compound
            compound_formation_map[compound] = formation_probs
        
        # Calculate environmental correlations (simplified)
        environmental_correlations = {
            'temperature_compound_diversity': np.corrcoef(
                [len(data.validated_compounds) for data in location_data],
                [data.environmental_profile.temperature for data in location_data]
            )[0, 1],
            'ph_formation_success': np.corrcoef(
                [len(data.earth_matches) for data in location_data],
                [data.environmental_profile.pH for data in location_data]
            )[0, 1],
            'depth_element_diversity': np.corrcoef(
                [len(data.detected_elements) for data in location_data],
                [data.drill_site.depth for data in location_data]
            )[0, 1]
        }
        
        # Element pool distribution analysis
        element_pool_distribution = {}
        for element in all_elements:
            distribution = {}
            for data in location_data:
                site_id = data.drill_site.site_id
                distribution[site_id] = data.detected_elements.get(element, 0.0)
            element_pool_distribution[element] = distribution
        
        return CrossLocationAnalysis(
            total_locations=len(location_data),
            common_elements=common_elements,
            location_specific_elements=location_specific_elements,
            common_compounds=common_compounds,
            compound_formation_map=compound_formation_map,
            environmental_correlations=environmental_correlations,
            element_pool_distribution=element_pool_distribution
        )


def main():
    """Demo the multi-location drilling survey system with user control"""
    
    print("🚀 NASA ROVER MULTI-LOCATION DRILLING SYSTEM")
    print("=" * 50)
    
    print("This system can drill multiple locations on any outer space environment.")
    print("The rover will adapt to whatever planetary surface it encounters.")
    print("\n🌌 Ready to explore any outer space environment!")
    
    # Auto-select a random outer space environment for this mission
    available_envs = [
        PlanetaryEnvironment.MARS_SURFACE,
        PlanetaryEnvironment.ASTEROID_METALLIC, 
        PlanetaryEnvironment.ICY_MOON,
        PlanetaryEnvironment.LUNAR_SURFACE,
        PlanetaryEnvironment.VENUS_SURFACE,
        PlanetaryEnvironment.ASTEROID_CARBONACEOUS
    ]
    
    chosen_env = random.choice(available_envs)
    chosen_name = chosen_env.value.replace('_', ' ').title()
    
    print(f"\n🎯 Mission Target: {chosen_name} (randomly selected)")
    
    # Auto-select drilling parameters for any outer space environment
    num_locations = random.randint(5, 10)  # Random 5-10 locations
    compounds_per_site = random.randint(6, 12)  # Random 6-12 compounds
    
    print(f"\n🗺️ Mission Parameters (auto-selected):")
    print(f"   Surface: {chosen_name}")
    print(f"   Drilling Locations: {num_locations}")
    print(f"   Compounds per Location: {compounds_per_site}")
    print(f"\n🌌 This system works on ANY outer space environment!")
    print()
    
    # Initialize survey system
    survey = MultiLocationDrillingSurvey(random_seed=42)
    
    # Conduct full survey on the randomly selected outer space environment
    location_data, cross_analysis = survey.conduct_full_survey(
        num_drilling_sites=num_locations,
        environment=chosen_env,
        compounds_per_site=compounds_per_site
    )
    
    # Print comprehensive summary
    print(f"\n" + "=" * 70)
    print(f"🏁 OUTER SPACE MISSION COMPLETE - {chosen_name.upper()}")
    print(f"🌌 This system works on ANY outer space environment!")
    print(f"=" * 70)
    print(f"📍 Total Locations Drilled: {cross_analysis.total_locations}")
    print(f"🧪 Unique Elements Found: {len(cross_analysis.common_elements)}")
    print(f"⚗️ Unique Compounds Tested: {len(cross_analysis.common_compounds)}")
    
    # Calculate statistics
    total_earth_matches = sum(len(data.earth_matches) for data in location_data)
    total_unique_compounds = sum(len(data.unique_compounds) for data in location_data)
    avg_earth_matches = np.mean([len(data.earth_matches) for data in location_data])
    avg_unique_compounds = np.mean([len(data.unique_compounds) for data in location_data])
    
    print(f"🌍 Total Earth Matches: {total_earth_matches}")
    print(f"🔬 Total Unknown Compounds: {total_unique_compounds}")
    print(f"📊 Average Earth Matches per Site: {avg_earth_matches:.1f}")
    print(f"🔍 Average Unknown Compounds per Site: {avg_unique_compounds:.1f}")
    
    print(f"\n🔬 Environmental Correlations across {chosen_name}:")
    for factor, correlation in cross_analysis.environmental_correlations.items():
        if not np.isnan(correlation):
            print(f"   {factor.replace('_', ' ').title()}: {correlation:.3f}")
    
    # Show most common elements across all locations
    if cross_analysis.common_elements:
        print(f"\n🧪 Most Abundant Elements across all {num_locations} locations:")
        element_totals = {elem: sum(quantities) for elem, quantities in cross_analysis.common_elements.items()}
        top_elements = sorted(element_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        for element, total_quantity in top_elements:
            avg_quantity = total_quantity / num_locations
            print(f"   {element}: {avg_quantity:.2f}% average abundance")


if __name__ == "__main__":
    main()