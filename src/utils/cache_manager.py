"""Cache manager for persistent storage of NIST and PubChem lookups.

This module provides a persistent cache system to store thermodynamic and chemical
data lookups to reduce API calls and improve consistency.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path


class PersistentCache:
    """Persistent cache for storing API lookup results."""
    
    def __init__(self, cache_dir: str = "cache", max_age_days: int = 30):
        """Initialize the persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        self.logger = logging.getLogger(__name__)
        
        # Cache files for different data sources
        self.nist_cache_file = self.cache_dir / "nist_cache.json"
        self.pubchem_cache_file = self.cache_dir / "pubchem_cache.json"
        self.mp_cache_file = self.cache_dir / "materials_project_cache.json"
        
        # Load existing caches
        self.nist_cache = self._load_cache(self.nist_cache_file)
        self.pubchem_cache = self._load_cache(self.pubchem_cache_file)
        self.mp_cache = self._load_cache(self.mp_cache_file)
        
        # Warm cache with common species
        self._warm_cache()
        
        # Save the warmed caches
        self._save_cache(self.nist_cache, self.nist_cache_file)
        self._save_cache(self.pubchem_cache, self.pubchem_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load cache from JSON file."""
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Clean expired entries
                    return self._clean_expired_entries(cache_data)
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load cache from {cache_file}: {e}")
            return {}
    
    def _save_cache(self, cache_data: Dict[str, Any], cache_file: Path) -> None:
        """Save cache to JSON file."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save cache to {cache_file}: {e}")
    
    def _clean_expired_entries(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove expired entries from cache."""
        cleaned_cache = {}
        current_time = datetime.now()
        
        for key, entry in cache_data.items():
            if isinstance(entry, dict) and 'timestamp' in entry:
                try:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if current_time - entry_time < self.max_age:
                        cleaned_cache[key] = entry
                except (ValueError, TypeError):
                    # Skip entries with invalid timestamps
                    continue
            else:
                # Keep entries without timestamps (legacy)
                cleaned_cache[key] = entry
        
        return cleaned_cache
    
    def _warm_cache(self) -> None:
        """Pre-populate cache with common chemical species."""
        common_species = {
            'H2O': {
                'nist': {
                    'enthalpy_formation': -285.83,
                    'gibbs_formation': -237.13,
                    'entropy_standard': 69.91,
                    'phase': 'liquid'
                },
                'pubchem': {
                    'cid': 962,
                    'molecular_weight': 18.015,
                    'canonical_smiles': 'O'
                }
            },
            'CO2': {
                'nist': {
                    'enthalpy_formation': -393.51,
                    'gibbs_formation': -394.36,
                    'entropy_standard': 213.74,
                    'phase': 'gas'
                },
                'pubchem': {
                    'cid': 280,
                    'molecular_weight': 44.01,
                    'canonical_smiles': 'C(=O)=O'
                }
            },
            'O2': {
                'nist': {
                    'enthalpy_formation': 0.0,
                    'gibbs_formation': 0.0,
                    'entropy_standard': 205.14,
                    'phase': 'gas'
                },
                'pubchem': {
                    'cid': 977,
                    'molecular_weight': 31.998,
                    'canonical_smiles': 'O=O'
                }
            },
            'N2': {
                'nist': {
                    'enthalpy_formation': 0.0,
                    'gibbs_formation': 0.0,
                    'entropy_standard': 191.61,
                    'phase': 'gas'
                },
                'pubchem': {
                    'cid': 947,
                    'molecular_weight': 28.014,
                    'canonical_smiles': 'N#N'
                }
            },
            'H2': {
                'nist': {
                    'enthalpy_formation': 0.0,
                    'gibbs_formation': 0.0,
                    'entropy_standard': 130.7,
                    'phase': 'gas'
                },
                'pubchem': {
                    'cid': 783,
                    'molecular_weight': 2.016,
                    'canonical_smiles': '[H][H]'
                }
            },
            'NH3': {
                'nist': {
                    'enthalpy_formation': -45.90,
                    'gibbs_formation': -16.45,
                    'entropy_standard': 192.45,
                    'phase': 'gas'
                },
                'pubchem': {
                    'cid': 222,
                    'molecular_weight': 17.031,
                    'canonical_smiles': 'N'
                }
            },
            'CH4': {
                'nist': {
                    'enthalpy_formation': -74.87,
                    'gibbs_formation': -50.72,
                    'entropy_standard': 186.26,
                    'phase': 'gas'
                },
                'pubchem': {
                    'cid': 297,
                    'molecular_weight': 16.043,
                    'canonical_smiles': 'C'
                }
            }
        }
        
        # Add common species to cache if not already present
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Warming cache with {len(common_species)} common chemical species")
        
        for formula, data in common_species.items():
            # Add to NIST cache
            if formula not in self.nist_cache:
                self.nist_cache[formula] = {
                    'data': data['nist'],
                    'timestamp': timestamp,
                    'source': 'warm_cache'
                }
                self.logger.debug(f"Pre-cached NIST data for {formula}")
            
            # Add to PubChem cache
            if formula not in self.pubchem_cache:
                self.pubchem_cache[formula] = {
                    'data': data['pubchem'],
                    'timestamp': timestamp,
                    'source': 'warm_cache'
                }
                self.logger.debug(f"Pre-cached PubChem data for {formula}")
        
        self.logger.info(f"Cache warmed with thermodynamic and chemical data for common species")
    
    def get_nist_data(self, formula: str) -> Optional[Dict[str, Any]]:
        """Get NIST data from cache."""
        entry = self.nist_cache.get(formula)
        if entry and isinstance(entry, dict):
            return entry.get('data')
        return None
    
    def set_nist_data(self, formula: str, data: Dict[str, Any]) -> None:
        """Store NIST data in cache."""
        self.nist_cache[formula] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'source': 'api_lookup'
        }
        self._save_cache(self.nist_cache, self.nist_cache_file)
    
    def get_pubchem_data(self, formula: str) -> Optional[Dict[str, Any]]:
        """Get PubChem data from cache."""
        entry = self.pubchem_cache.get(formula)
        if entry and isinstance(entry, dict):
            return entry.get('data')
        return None
    
    def set_pubchem_data(self, formula: str, data: Dict[str, Any]) -> None:
        """Store PubChem data in cache."""
        self.pubchem_cache[formula] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'source': 'api_lookup'
        }
        self._save_cache(self.pubchem_cache, self.pubchem_cache_file)
    
    def get_mp_data(self, formula: str) -> Optional[Dict[str, Any]]:
        """Get Materials Project data from cache."""
        entry = self.mp_cache.get(formula)
        if entry and isinstance(entry, dict):
            return entry.get('data')
        return None
    
    def set_mp_data(self, formula: str, data: Dict[str, Any]) -> None:
        """Store Materials Project data in cache."""
        self.mp_cache[formula] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'source': 'api_lookup'
        }
        self._save_cache(self.mp_cache, self.mp_cache_file)
    
    def clear_cache(self, cache_type: str = 'all') -> None:
        """Clear cache data.
        
        Args:
            cache_type: Type of cache to clear ('nist', 'pubchem', 'mp', or 'all')
        """
        if cache_type in ['nist', 'all']:
            self.nist_cache.clear()
            self._save_cache(self.nist_cache, self.nist_cache_file)
        
        if cache_type in ['pubchem', 'all']:
            self.pubchem_cache.clear()
            self._save_cache(self.pubchem_cache, self.pubchem_cache_file)
        
        if cache_type in ['mp', 'all']:
            self.mp_cache.clear()
            self._save_cache(self.mp_cache, self.mp_cache_file)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'nist_entries': len(self.nist_cache),
            'pubchem_entries': len(self.pubchem_cache),
            'mp_entries': len(self.mp_cache),
            'cache_dir': str(self.cache_dir),
            'max_age_days': self.max_age.days
        }