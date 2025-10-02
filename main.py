#!/usr/bin/env python3
"""Main entry point for the chemistry validation system."""

import argparse
import json
from pathlib import Path

from src.core.enhanced_chemistry_validator import EnhancedChemistryValidator
from src.data.data_models import CompoundSuggestion, EnvironmentalConditions
from src.config.configuration import ConfigurationManager

def main():
    parser = argparse.ArgumentParser(description='Chemistry Validation System')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, default='output.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = ConfigurationManager(args.config)
    
    # Create validator
    validator = EnhancedChemistryValidator(config=config.config)
    
    # Example usage
    env_conditions = EnvironmentalConditions(
        temperature=25.0, pressure=1.0, depth=0.0, pH=7.0,
        humidity=50.0, reaction_time=1.0, catalyst_present=False,
        atmosphere='air', stirring=False, light_exposure='ambient'
    )
    
    # Define available elements (example quantities in moles)
    available_elements = {
        'H': 10.0,   # Hydrogen
        'O': 5.0,    # Oxygen
        'Na': 2.0,   # Sodium
        'Cl': 2.0,   # Chlorine
        'C': 3.0,    # Carbon
        'N': 1.0,    # Nitrogen
        'Fe': 1.0,   # Iron
        'S': 1.0     # Sulfur
    }
    
    # Generate some example compounds
    compounds = [
        CompoundSuggestion("H2O", "Water", "molecular", "high"),
        CompoundSuggestion("NaCl", "Sodium Chloride", "ionic", "high"),
        CompoundSuggestion("CO2", "Carbon Dioxide", "molecular", "medium"),
    ]
    
    results = []
    for compound in compounds:
        # Fix: Pass all three required arguments
        result = validator.validate_compound_feasibility(compound, available_elements, env_conditions)
        results.append({
            'formula': compound.formula,
            'feasible': result.get('feasible', False),
            'formation_probability': result.get('formation_probability', 0.0),
            'limiting_factors': result.get('limiting_factors', []),
            'confidence': result.get('confidence', 0.0)
        })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nValidation Summary:")
    for result in results:
        print(f"- {result['formula']}: {'✓' if result['feasible'] else '✗'} "
              f"(probability: {result['formation_probability']:.3f})")

if __name__ == "__main__":
    main()