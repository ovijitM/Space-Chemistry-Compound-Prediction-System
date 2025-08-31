"""Thermodynamic data handling module for chemistry dataset generation.

This module provides classes for scraping thermodynamic data from NIST WebBook
and handling thermodynamic calculations for compound formation probability.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import requests
from bs4 import BeautifulSoup

from .data_models import NISTThermodynamicData
from ..utils.cache_manager import PersistentCache


class NISTWebBookScraper:
    """Scraper for NIST Chemistry WebBook thermodynamic data."""
    
    def __init__(self, delay: float = 1.0):
        """Initialize the NIST WebBook scraper.
        
        Args:
            delay: Delay between requests in seconds
        """
        self.base_url = "https://webbook.nist.gov/cgi/cbook.cgi"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ChemistryDatasetGenerator/1.0)'
        })
        
    def search_compound(self, formula: str) -> Optional[str]:
        """Search for a compound and return its NIST ID.
        
        Args:
            formula: Chemical formula to search for
            
        Returns:
            NIST compound ID if found, None otherwise
        """
        try:
            params = {
                'Formula': formula,
                'NoIon': 'on',
                'Units': 'SI'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse the response to find compound ID
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for links to compound pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'ID=' in href and 'Mask=' in href:
                    # Extract ID parameter
                    id_match = re.search(r'ID=([^&]+)', href)
                    if id_match:
                        return id_match.group(1)
            
            return None
            
        except Exception as e:
            logging.debug(f"Error searching for compound {formula}: {e}")
            return None
    
    def get_thermodynamic_data(self, compound_id: str) -> Optional[NISTThermodynamicData]:
        """Get thermodynamic data for a compound by its NIST ID.
        
        Args:
            compound_id: NIST compound ID
            
        Returns:
            NISTThermodynamicData object if successful, None otherwise
        """
        try:
            time.sleep(self.delay)  # Rate limiting
            
            params = {
                'ID': compound_id,
                'Mask': '1F',  # Thermochemical data
                'Type': 'JANAFG',
                'Units': 'SI'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse formation data
            formation_data = self._parse_formation_data(soup)
            
            # Parse standard properties
            standard_props = self._parse_standard_properties(soup)
            
            # Combine data
            data = NISTThermodynamicData(
                enthalpy_formation=formation_data.get('enthalpy_formation'),
                gibbs_formation=formation_data.get('gibbs_formation'),
                entropy_standard=standard_props.get('entropy'),
                heat_capacity=standard_props.get('heat_capacity'),
                phase=formation_data.get('phase', 'unknown'),
                cas_number=self._extract_cas_number(soup)
            )
            
            return data
            
        except Exception as e:
            logging.debug(f"Error getting thermodynamic data for ID {compound_id}: {e}")
            return None
    
    def _parse_formation_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse formation enthalpy and Gibbs energy from NIST page.
        
        Args:
            soup: BeautifulSoup object of the NIST page
            
        Returns:
            Dictionary containing formation data
        """
        data = {}
        
        # Look for formation data tables
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    header = cells[0].get_text(strip=True).lower()
                    value_text = cells[1].get_text(strip=True)
                    
                    # Parse enthalpy of formation
                    if 'enthalpy of formation' in header or 'δfh°' in header:
                        value = self._extract_numeric_value(value_text)
                        if value is not None:
                            data['enthalpy_formation'] = value
                    
                    # Parse Gibbs energy of formation
                    elif 'gibbs energy of formation' in header or 'δfg°' in header:
                        value = self._extract_numeric_value(value_text)
                        if value is not None:
                            data['gibbs_formation'] = value
                    
                    # Parse phase information
                    elif 'phase' in header or 'state' in header:
                        phase = value_text.lower()
                        if any(p in phase for p in ['gas', 'vapor']):
                            data['phase'] = 'gas'
                        elif 'liquid' in phase:
                            data['phase'] = 'liquid'
                        elif 'solid' in phase or 'crystal' in phase:
                            data['phase'] = 'solid'
        
        return data
    
    def _parse_standard_properties(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse standard entropy and heat capacity from NIST page.
        
        Args:
            soup: BeautifulSoup object of the NIST page
            
        Returns:
            Dictionary containing standard properties
        """
        data = {}
        
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    header = cells[0].get_text(strip=True).lower()
                    value_text = cells[1].get_text(strip=True)
                    
                    # Parse standard entropy
                    if 'entropy' in header and 'standard' in header:
                        value = self._extract_numeric_value(value_text)
                        if value is not None:
                            data['entropy'] = value
                    
                    # Parse heat capacity
                    elif 'heat capacity' in header or 'cp' in header:
                        value = self._extract_numeric_value(value_text)
                        if value is not None:
                            data['heat_capacity'] = value
        
        return data
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text, handling units and uncertainties.
        
        Args:
            text: Text containing numeric value
            
        Returns:
            Extracted numeric value or None
        """
        try:
            # Remove common units and symbols
            cleaned = re.sub(r'[±∓].*', '', text)  # Remove uncertainty
            cleaned = re.sub(r'[kJ/mol|J/mol|K|°C|°F|atm|bar|Pa]', '', cleaned)
            cleaned = re.sub(r'[()\[\]]', '', cleaned)  # Remove brackets
            cleaned = cleaned.strip()
            
            # Extract first number found
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                return float(match.group())
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    def _extract_cas_number(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract CAS registry number from NIST page.
        
        Args:
            soup: BeautifulSoup object of the NIST page
            
        Returns:
            CAS number if found, None otherwise
        """
        try:
            # Look for CAS number patterns
            text = soup.get_text()
            cas_match = re.search(r'CAS\s*Registry\s*Number[:\s]*(\d+-\d+-\d+)', text, re.IGNORECASE)
            if cas_match:
                return cas_match.group(1)
            
            # Alternative pattern
            cas_match = re.search(r'(\d+-\d+-\d+)', text)
            if cas_match:
                return cas_match.group(1)
            
            return None
            
        except Exception:
            return None
    
    def get_compound_data_by_name(self, formula: str) -> Optional[NISTThermodynamicData]:
        """Get thermodynamic data by compound formula/name.
        
        Args:
            formula: Chemical formula or name
            
        Returns:
            NISTThermodynamicData object if found, None otherwise
        """
        compound_id = self.search_compound(formula)
        if compound_id:
            return self.get_thermodynamic_data(compound_id)
        return None


class ThermodynamicDataHandler:
    """Handler for thermodynamic data retrieval and calculations."""
    
    def __init__(self, cache_size: int = 1000, enable_nist_scraping: bool = True):
        """Initialize the thermodynamic data handler.
        
        Args:
            cache_size: Maximum number of entries to cache
            enable_nist_scraping: Whether to enable NIST WebBook scraping
        """
        self.cache_size = cache_size
        self.cache = {}
        self.enable_nist_scraping = enable_nist_scraping
        
        # Initialize persistent cache
        try:
            self.persistent_cache = PersistentCache()
            logging.info("Persistent cache initialized for thermodynamic data")
        except Exception as e:
            logging.warning(f"Failed to initialize persistent cache: {e}")
            self.persistent_cache = None
        
        # Initialize NIST scraper if enabled
        if enable_nist_scraping:
            try:
                self.nist_scraper = NISTWebBookScraper(delay=1.0)
                logging.info("NIST WebBook scraper initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize NIST scraper: {e}")
                self.nist_scraper = None
        else:
            self.nist_scraper = None
        
        # Load built-in thermodynamic data
        self._load_nist_data()
        self.element_thermodynamics = self._load_element_thermodynamics()
        self._load_element_properties()
        
        logging.info(f"Loaded {len(self.formation_gibbs)} compounds from built-in database")
    
    def _load_nist_data(self):
        """Load internal NIST JANAF thermodynamic data."""
        # Standard formation enthalpies (kJ/mol) at 298.15 K
        self.formation_enthalpies = {
            'H2O': -285.83,
            'CO2': -393.51,
            'CH4': -74.87,
            'NH3': -45.90,
            'H2SO4': -813.99,
            'CaCO3': -1206.92,
            'Fe2O3': -824.2,
            'Al2O3': -1675.7,
            'SiO2': -910.7,
            'NaCl': -411.15,
            'MgO': -601.60,
            'CaO': -635.09,
            'SO2': -296.83,
            'NO2': 33.18,
            'N2O': 82.05,
            'HCl': -92.31,
            'HNO3': -174.10,
            'C2H6': -84.0,
            'C2H4': 52.4,
            'C2H2': 228.2
        }
        
        # Standard Gibbs free energies of formation (kJ/mol) at 298.15 K
        self.formation_gibbs = {
            'H2O': -237.13,
            'CO2': -394.36,
            'CH4': -50.72,
            'NH3': -16.45,
            'H2SO4': -690.00,
            'CaCO3': -1128.79,
            'Fe2O3': -742.2,
            'Al2O3': -1582.3,
            'SiO2': -856.3,
            'NaCl': -384.12,
            'MgO': -569.43,
            'CaO': -604.03,
            'SO2': -300.13,
            'NO2': 51.23,
            'N2O': 103.7,
            'HCl': -95.30,
            'HNO3': -80.71,
            'C2H6': -32.0,
            'C2H4': 68.4,
            'C2H2': 209.9
        }
        
        # Standard entropies (J/(mol·K)) at 298.15 K
        self.standard_entropies = {
            'H2O': 69.91,
            'CO2': 213.74,
            'CH4': 186.26,
            'NH3': 192.45,
            'H2SO4': 156.90,
            'CaCO3': 92.9,
            'Fe2O3': 87.40,
            'Al2O3': 50.92,
            'SiO2': 41.84,
            'NaCl': 72.13,
            'MgO': 26.94,
            'CaO': 39.75,
            'SO2': 248.22,
            'NO2': 240.06,
            'N2O': 219.85,
            'HCl': 186.91,
            'HNO3': 155.60,
            'C2H6': 229.60,
            'C2H4': 219.56,
            'C2H2': 200.94
        }
    
    def _load_element_thermodynamics(self) -> Dict[str, Dict[str, float]]:
        """Load element-specific thermodynamic properties.
        
        Returns:
            Dictionary mapping elements to their thermodynamic properties
        """
        return {
            'H': {'enthalpy_formation': 0.0, 'entropy': 130.7, 'heat_capacity': 28.8},
            'He': {'enthalpy_formation': 0.0, 'entropy': 126.2, 'heat_capacity': 20.8},
            'Li': {'enthalpy_formation': 0.0, 'entropy': 29.1, 'heat_capacity': 24.8},
            'Be': {'enthalpy_formation': 0.0, 'entropy': 9.5, 'heat_capacity': 16.4},
            'B': {'enthalpy_formation': 0.0, 'entropy': 5.9, 'heat_capacity': 11.1},
            'C': {'enthalpy_formation': 0.0, 'entropy': 5.7, 'heat_capacity': 8.5},
            'N': {'enthalpy_formation': 0.0, 'entropy': 191.6, 'heat_capacity': 29.1},
            'O': {'enthalpy_formation': 0.0, 'entropy': 205.2, 'heat_capacity': 29.4},
            'F': {'enthalpy_formation': 0.0, 'entropy': 202.8, 'heat_capacity': 31.3},
            'Ne': {'enthalpy_formation': 0.0, 'entropy': 146.3, 'heat_capacity': 20.8},
            'Na': {'enthalpy_formation': 0.0, 'entropy': 51.3, 'heat_capacity': 28.2},
            'Mg': {'enthalpy_formation': 0.0, 'entropy': 32.7, 'heat_capacity': 24.9},
            'Al': {'enthalpy_formation': 0.0, 'entropy': 28.3, 'heat_capacity': 24.4},
            'Si': {'enthalpy_formation': 0.0, 'entropy': 18.8, 'heat_capacity': 20.0},
            'P': {'enthalpy_formation': 0.0, 'entropy': 41.1, 'heat_capacity': 23.8},
            'S': {'enthalpy_formation': 0.0, 'entropy': 32.1, 'heat_capacity': 22.6},
            'Cl': {'enthalpy_formation': 0.0, 'entropy': 223.1, 'heat_capacity': 33.9},
            'Ar': {'enthalpy_formation': 0.0, 'entropy': 154.8, 'heat_capacity': 20.8},
            'K': {'enthalpy_formation': 0.0, 'entropy': 64.7, 'heat_capacity': 29.6},
            'Ca': {'enthalpy_formation': 0.0, 'entropy': 41.6, 'heat_capacity': 25.9},
            'Fe': {'enthalpy_formation': 0.0, 'entropy': 27.3, 'heat_capacity': 25.1},
            'Ni': {'enthalpy_formation': 0.0, 'entropy': 29.9, 'heat_capacity': 26.1},
            'Cu': {'enthalpy_formation': 0.0, 'entropy': 33.2, 'heat_capacity': 24.4},
            'Zn': {'enthalpy_formation': 0.0, 'entropy': 41.6, 'heat_capacity': 25.4}
        }
    
    def _load_element_properties(self):
        """Load element-specific thermodynamic properties."""
        # Standard formation enthalpies for elements in their standard states are zero
        # These are for common compounds and ions
        self.element_properties = {
            'H': {'ionization_energy': 1312.0, 'electron_affinity': 72.8},
            'C': {'ionization_energy': 1086.5, 'electron_affinity': 121.9},
            'N': {'ionization_energy': 1402.3, 'electron_affinity': -7.0},
            'O': {'ionization_energy': 1313.9, 'electron_affinity': 141.0},
            'F': {'ionization_energy': 1681.0, 'electron_affinity': 328.0},
            'Na': {'ionization_energy': 495.8, 'electron_affinity': 52.8},
            'Mg': {'ionization_energy': 737.7, 'electron_affinity': -40.0},
            'Al': {'ionization_energy': 577.5, 'electron_affinity': 42.5},
            'Si': {'ionization_energy': 786.5, 'electron_affinity': 133.6},
            'P': {'ionization_energy': 1011.8, 'electron_affinity': 72.0},
            'S': {'ionization_energy': 999.6, 'electron_affinity': 200.4},
            'Cl': {'ionization_energy': 1251.2, 'electron_affinity': 349.0},
            'K': {'ionization_energy': 418.8, 'electron_affinity': 48.4},
            'Ca': {'ionization_energy': 589.8, 'electron_affinity': 2.37},
            'Fe': {'ionization_energy': 762.5, 'electron_affinity': 15.7}
        }
    
    def get_thermodynamic_data(self, formula: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive thermodynamic data for a compound.
        
        Args:
            formula: Chemical formula
            
        Returns:
            Dictionary containing thermodynamic data or None
        """
        # Check in-memory cache first
        if formula in self.cache:
            return self.cache[formula]
        
        # Check persistent cache for NIST data
        if self.persistent_cache:
            cached_nist_data = self.persistent_cache.get_nist_data(formula)
            if cached_nist_data:
                logging.debug(f"Using cached NIST data for {formula}")
                # Cache in memory too
                if len(self.cache) >= self.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[formula] = cached_nist_data
                return cached_nist_data
        
        data = {
            'formula': formula,
            'enthalpy_formation': None,
            'gibbs_formation': None,
            'entropy_standard': None,
            'heat_capacity': None,
            'phase': None,
            'source': 'built_in'
        }
        
        # Try internal database first
        if formula in self.formation_enthalpies:
            data['enthalpy_formation'] = self.formation_enthalpies[formula]
            data['gibbs_formation'] = self.formation_gibbs.get(formula)
            data['entropy_standard'] = self.standard_entropies.get(formula)
            data['source'] = 'internal_nist'
        
        # Try NIST WebBook scraping if enabled and no internal data
        elif self.nist_scraper and data['enthalpy_formation'] is None:
            try:
                nist_data = self.nist_scraper.get_compound_data_by_name(formula)
                if nist_data:
                    data['enthalpy_formation'] = nist_data.enthalpy_formation
                    data['gibbs_formation'] = nist_data.gibbs_formation
                    data['entropy_standard'] = nist_data.entropy_standard
                    data['heat_capacity'] = nist_data.heat_capacity
                    data['phase'] = nist_data.phase
                    data['source'] = 'nist_webbook'
                    logging.info(f"Retrieved NIST data for {formula}")
                    
                    # Cache the NIST result persistently
                    if self.persistent_cache:
                        self.persistent_cache.set_nist_data(formula, data)
            except Exception as e:
                logging.debug(f"NIST scraping failed for {formula}: {e}")
        
        # Cache the result in memory
        if self.cache:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[formula] = data
        
        return data if any(v is not None for k, v in data.items() if k not in ['formula', 'source']) else None
    
    def calculate_formation_probability(self, formula: str, temperature: float = 298.15) -> float:
        """Calculate formation probability based on thermodynamic data.
        
        Args:
            formula: Chemical formula
            temperature: Temperature in Kelvin
            
        Returns:
            Formation probability (0-1)
        """
        data = self.get_thermodynamic_data(formula)
        if not data:
            return 0.5  # Default uncertainty
        
        # Use Gibbs free energy if available
        if data['gibbs_formation'] is not None:
            delta_g = data['gibbs_formation']  # kJ/mol
            # Convert to probability using Boltzmann distribution
            # P ∝ exp(-ΔG/RT)
            R = 8.314e-3  # kJ/(mol·K)
            try:
                prob = np.exp(-delta_g / (R * temperature))
                # Normalize to 0-1 range
                return min(1.0, max(0.0, prob / (1 + prob)))
            except (OverflowError, ZeroDivisionError):
                return 0.1 if delta_g > 0 else 0.9
        
        # Fallback to enthalpy-based estimation
        elif data['enthalpy_formation'] is not None:
            delta_h = data['enthalpy_formation']  # kJ/mol
            # Simple heuristic: more negative enthalpy = higher probability
            if delta_h < -200:
                return 0.9
            elif delta_h < -50:
                return 0.7
            elif delta_h < 0:
                return 0.5
            elif delta_h < 100:
                return 0.3
            else:
                return 0.1
        
        return 0.5  # Default uncertainty
    
    def get_stability_assessment(self, formula: str, temperature: float = 298.15) -> Optional[Dict[str, Any]]:
        """Get comprehensive stability assessment for a compound.
        
        Args:
            formula: Chemical formula
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary containing stability metrics
        """
        data = self.get_thermodynamic_data(formula)
        if not data:
            return None
        
        assessment = {
            'formula': formula,
            'temperature': temperature,
            'delta_g_formation': data['gibbs_formation'],
            'delta_h_formation': data['enthalpy_formation'],
            'entropy': data['entropy_standard'],
            'formation_feasible': False,
            'stability_score': 0.0,
            'thermodynamic_favorability': 'unknown'
        }
        
        # Assess formation feasibility
        if data['gibbs_formation'] is not None:
            delta_g = data['gibbs_formation']
            assessment['formation_feasible'] = delta_g < 0
            
            # Calculate stability score (0-1)
            if delta_g < -100:
                assessment['stability_score'] = 0.95
                assessment['thermodynamic_favorability'] = 'highly_favorable'
            elif delta_g < -50:
                assessment['stability_score'] = 0.8
                assessment['thermodynamic_favorability'] = 'favorable'
            elif delta_g < 0:
                assessment['stability_score'] = 0.6
                assessment['thermodynamic_favorability'] = 'moderately_favorable'
            elif delta_g < 50:
                assessment['stability_score'] = 0.3
                assessment['thermodynamic_favorability'] = 'unfavorable'
            else:
                assessment['stability_score'] = 0.1
                assessment['thermodynamic_favorability'] = 'highly_unfavorable'
        
        elif data['enthalpy_formation'] is not None:
            # Fallback to enthalpy-based assessment
            delta_h = data['enthalpy_formation']
            assessment['formation_feasible'] = delta_h < 0
            
            if delta_h < -100:
                assessment['stability_score'] = 0.8
                assessment['thermodynamic_favorability'] = 'enthalpically_favorable'
            elif delta_h < 0:
                assessment['stability_score'] = 0.6
                assessment['thermodynamic_favorability'] = 'moderately_favorable'
            else:
                assessment['stability_score'] = 0.2
                assessment['thermodynamic_favorability'] = 'enthalpically_unfavorable'
        
        return assessment