"""Database validation module for chemistry compounds.

This module contains validators for different chemical databases including
PyMatGen, PubChem, RDKit, and Materials Project.
"""

import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Optional chemistry library imports
try:
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.ext.matproj import MPRester
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import periodictable
    PERIODICTABLE_AVAILABLE = True
except ImportError:
    PERIODICTABLE_AVAILABLE = False

# Check Materials Project availability
MP_AVAILABLE = PYMATGEN_AVAILABLE


class DatabaseValidators:
    """Collection of database validation methods."""
    
    def __init__(self, config: Dict[str, Any], persistent_cache=None, logger=None):
        """Initialize database validators.
        
        Args:
            config: Configuration dictionary
            persistent_cache: Cache instance for storing results
            logger: Logger instance
        """
        self.config = config
        self.persistent_cache = persistent_cache
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_with_databases(self, formula: str) -> Dict:
        """Validate compound against multiple databases."""
        results = {}
        
        # PyMatGen validation
        if PYMATGEN_AVAILABLE:
            results['pymatgen'] = self.validate_with_pymatgen(formula)
        
        # PubChem validation
        if PUBCHEM_AVAILABLE:
            results['pubchem'] = self.validate_with_pubchem(formula)
        
        # RDKit validation
        if RDKIT_AVAILABLE:
            results['rdkit'] = self.validate_with_rdkit(formula)
        
        # Materials Project validation
        if MP_AVAILABLE and self.config.get('materials_project', {}).get('enable', False):
            results['materials_project'] = self.validate_with_materials_project(formula)
        
        # Molecular weight calculation
        if PERIODICTABLE_AVAILABLE:
            results['molecular_weight'] = self.calculate_molecular_weight(formula)
        
        return results
    
    def validate_with_pymatgen(self, formula: str) -> Dict:
        """Validate using PyMatGen materials database."""
        try:
            composition = Composition(formula)
            
            # Estimate formation energy
            formation_energy = self._estimate_formation_energy(composition)
            
            # Check if composition is charge balanced
            is_charge_balanced = composition.charge == 0
            
            return {
                'valid': True,
                'composition': str(composition),
                'formation_energy_estimate': formation_energy,
                'is_charge_balanced': is_charge_balanced,
                'is_stable': formation_energy < 0 if formation_energy is not None else None
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _estimate_formation_energy(self, composition: 'Composition') -> Optional[float]:
        """Estimate formation energy using simple heuristics."""
        try:
            # Simple estimation based on electronegativity differences
            elements = composition.elements
            if len(elements) < 2:
                return 0.0
            
            # Get electronegativity values
            electronegativities = []
            for element in elements:
                # Use simple electronegativity values
                en_values = {
                    'H': 2.2, 'Li': 1.0, 'C': 2.5, 'N': 3.0, 'O': 3.5, 'F': 4.0,
                    'Na': 0.9, 'Mg': 1.2, 'Al': 1.5, 'Si': 1.8, 'P': 2.1, 'S': 2.5, 'Cl': 3.0
                }
                electronegativities.append(en_values.get(str(element), 2.0))
            
            # Estimate based on electronegativity difference
            max_diff = max(electronegativities) - min(electronegativities)
            
            # Simple heuristic: larger differences suggest more stable ionic compounds
            if max_diff > 1.5:
                return -50 * max_diff  # Negative indicates stability
            else:
                return -10 * max_diff  # Less stable
                
        except Exception:
            return None
    
    def validate_with_pubchem(self, formula: str) -> Dict:
        """Validate using PubChem database with SMILES retrieval and caching."""
        # Check cache first
        if self.persistent_cache:
            cached_data = self.persistent_cache.get_pubchem_data(formula)
            if cached_data:
                self.logger.debug(f"Using cached PubChem data for {formula}")
                return cached_data
        
        try:
            compounds = pcp.get_compounds(formula, 'formula')
            
            if compounds:
                compound = compounds[0]
                result = {
                    'valid': True,
                    'cid': compound.cid,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight,
                    'iupac_name': compound.iupac_name
                }
                
                # Fetch canonical SMILES
                try:
                    smiles = compound.canonical_smiles
                    if smiles:
                        result['canonical_smiles'] = smiles
                        
                        # Compute RDKit descriptors from SMILES
                        if RDKIT_AVAILABLE:
                            rdkit_descriptors = self.compute_rdkit_descriptors_from_smiles(smiles)
                            result['rdkit_descriptors'] = rdkit_descriptors
                            
                except Exception as e:
                    self.logger.debug(f"Failed to get SMILES or RDKit descriptors: {e}")
                
                # Cache the result
                if self.persistent_cache:
                    self.persistent_cache.set_pubchem_data(formula, result)
                
                return result
            else:
                error_result = {
                    'valid': False,
                    'error': 'Compound not found in PubChem'
                }
                # Cache negative results too (with shorter TTL)
                if self.persistent_cache:
                    self.persistent_cache.set_pubchem_data(formula, error_result)
                return error_result
        except Exception as e:
            error_result = {
                'valid': False,
                'error': str(e)
            }
            return error_result
    
    def validate_with_rdkit(self, formula: str) -> Dict:
        """Validate using RDKit."""
        try:
            # Try to create molecule from formula (limited capability)
            # RDKit works better with SMILES, but we can do basic validation
            mol = Chem.MolFromFormula(formula)
            
            if mol:
                return {
                    'valid': True,
                    'num_atoms': mol.GetNumAtoms(),
                    'molecular_weight': Descriptors.MolWt(mol)
                }
            else:
                return {
                    'valid': False,
                    'error': 'Invalid molecular formula for RDKit'
                }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def compute_rdkit_descriptors_from_smiles(self, smiles: str) -> Dict:
        """Compute RDKit molecular descriptors from SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'error': 'Invalid SMILES string'}
            
            # Compute key molecular descriptors for ML features
            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),  # Topological Polar Surface Area
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_hbd': Descriptors.NumHDonors(mol),  # Hydrogen bond donors
                'num_hba': Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'valence_electrons': Descriptors.NumValenceElectrons(mol)
            }
            
            # Add basic validity checks
            descriptors['has_valid_valence'] = all(
                atom.GetTotalValence() <= Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())[-1]
                for atom in mol.GetAtoms()
            )
            
            return descriptors
            
        except Exception as e:
             return {'error': str(e)}
    
    def validate_with_materials_project(self, formula: str) -> Dict:
        """Validate using Materials Project database for formation energies and stability."""
        if not MP_AVAILABLE:
            return {'valid': False, 'error': 'Materials Project not available'}
        
        try:
            # Get API key from environment or config
            api_key = self.config.get('materials_project', {}).get('api_key')
            if not api_key:
                api_key = os.environ.get('MP_API_KEY')
            
            if not api_key:
                return {'valid': False, 'error': 'Materials Project API key not found'}
            
            timeout = self.config.get('materials_project', {}).get('timeout', 10)
            max_entries = self.config.get('materials_project', {}).get('max_entries', 5)
            
            with MPRester(api_key) as mpr:
                # Search for entries with this formula
                entries = mpr.get_entries(formula, inc_structure=False)
                
                if not entries:
                    return {'valid': False, 'error': 'No entries found in Materials Project'}
                
                # Limit entries to avoid excessive processing
                entries = entries[:max_entries]
                
                # Find the most stable entry (lowest formation energy per atom)
                most_stable = min(entries, key=lambda e: e.energy_per_atom)
                
                # Calculate energy above hull for stability assessment
                try:
                    # Get all entries for phase diagram construction
                    all_entries = mpr.get_entries_in_chemsys(
                        list(most_stable.composition.element_composition.keys())
                    )
                    
                    # Create phase diagram and calculate energy above hull
                    pd = PhaseDiagram(all_entries)
                    energy_above_hull = pd.get_e_above_hull(most_stable)
                    
                    # Determine stability
                    is_stable = energy_above_hull < 0.025  # eV/atom threshold
                    is_metastable = 0.025 <= energy_above_hull < 0.1
                    
                except Exception as e:
                    self.logger.debug(f"Could not calculate energy above hull: {e}")
                    energy_above_hull = None
                    is_stable = False
                    is_metastable = False
                
                result = {
                    'valid': True,
                    'formation_energy_per_atom': most_stable.energy_per_atom,
                    'total_entries': len(entries),
                    'is_stable': is_stable,
                    'is_metastable': is_metastable,
                    'energy_above_hull': energy_above_hull,
                    'mp_id': getattr(most_stable, 'entry_id', 'unknown')
                }
                
                return result
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def calculate_molecular_weight(self, formula: str) -> Optional[float]:
        """Calculate molecular weight using periodictable."""
        try:
            from chemistry_utils import ChemistryUtils
            elements = ChemistryUtils.extract_elements_from_formula(formula)
            total_weight = 0.0
            
            for element, count in elements.items():
                if hasattr(periodictable, element):
                    atomic_weight = getattr(periodictable, element).mass
                    total_weight += atomic_weight * count
                else:
                    return None
            
            return total_weight
        except Exception:
            return None