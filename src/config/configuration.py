"""Configuration management module for the chemistry generator."""

import json
import logging
from typing import Dict, Optional
from .constants import VALIDATION_THRESHOLDS, FORMATION_PROBABILITY_WEIGHTS, ENVIRONMENTAL_PARAMETERS


class ConfigurationManager:
    """Manages configuration settings for the chemistry generator."""
    
    DEFAULT_CONFIG = {
        'llm': {
            'models': [
                "deepseek-ai/DeepSeek-V3-0324:fireworks-ai"
            ],
            'timeout': 15,
            'max_retries': 3,
            'temperature': 0.25,
            'max_tokens': 800
        },
        'chemistry': {
            'max_elements': 6,
            'min_elements': 2,
            'reactive_elements_weight': 3.0,
            'formation_probability_threshold': VALIDATION_THRESHOLDS['formation_probability_threshold']
        },
        'validation': {
            'enable_pubchem': True,
            'enable_pymatgen': True,
            'enable_rdkit': True,
            'database_timeout': 5.0,
            'confidence_threshold': VALIDATION_THRESHOLDS['confidence_threshold'],
            'environmental_score_threshold': VALIDATION_THRESHOLDS['environmental_score_minimum'],
            'chemical_rules_threshold': VALIDATION_THRESHOLDS['chemical_rules_minimum'],
            'stoichiometry_threshold': VALIDATION_THRESHOLDS['structural_minimum']
        },
        'performance': {
            'parallel_processing': True,
            'max_workers': 4,
            'rate_limit_delay': 1.5,
            'enable_caching': True
        },
        'materials_project': {
            'enable': True,
            'api_key': None,  # Will read from environment if not set
            'timeout': 10,
            'max_entries': 5
        },
        'environmental': {
            'temperature_ranges': ENVIRONMENTAL_PARAMETERS['temperature_ranges'],
            'pressure_sensitivity': ENVIRONMENTAL_PARAMETERS['pressure_sensitivity'],
            'ph_tolerance': ENVIRONMENTAL_PARAMETERS['ph_tolerance'],
            'aggregation_method': ENVIRONMENTAL_PARAMETERS['aggregation_method']
        },
        'output': {
            'save_format': ['json', 'csv'],
            'timestamp_format': '%Y%m%d_%H%M%S',
            'preview_samples': 2
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            self._merge_config(self.config, user_config)
            logging.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, path: str, default=None):
        """Get configuration value using dot notation (e.g., 'llm.temperature')."""
        keys = path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default