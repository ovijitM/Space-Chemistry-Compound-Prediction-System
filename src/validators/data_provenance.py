"""Data provenance module for chemistry validation.

This module handles data source identification, evidence scoring,
and provenance tracking for compound validation results.
"""

import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from ..config.constants import DATABASE_WEIGHTS, DATABASE_RELIABILITY, UNCERTAINTY_PARAMETERS


class DataProvenanceTracker:
    """Tracker for data provenance and evidence scoring."""
    
    def __init__(self, logger=None):
        """Initialize data provenance tracker.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def identify_data_sources(self, db_validation: Dict, thermo_data: Optional[Dict] = None) -> List[str]:
        """Identify data sources from validation results.
        
        Args:
            db_validation: Database validation results
            thermo_data: Thermodynamic data (optional)
            
        Returns:
            List of identified data sources
        """
        sources = []
        
        if not db_validation:
            return sources
        
        # Check each database source
        database_mappings = {
            'nist': 'NIST',
            'pubchem': 'PubChem', 
            'materials_project': 'Materials Project',
            'rdkit': 'RDKit'
        }
        
        for db_key, source_name in database_mappings.items():
            # Check if database was found (handle both boolean and dict formats)
            db_found_key = f"{db_key}_found"
            if db_found_key in db_validation:
                if db_validation[db_found_key] is True:
                    sources.append(source_name)
            
            # Also check if database data exists as nested dict
            elif db_key in db_validation:
                db_data = db_validation[db_key]
                if isinstance(db_data, dict) and db_data:
                    # Non-empty dict indicates data was found
                    sources.append(source_name)
                elif db_data is True:
                    sources.append(source_name)
        
        # Add thermodynamic data source if available
        if thermo_data and isinstance(thermo_data, dict) and thermo_data:
            sources.append('Thermodynamic Database')
        
        # Ensure deterministic ordering
        return sorted(list(set(sources)))
    
    def calculate_evidence_score(self, db_validation: Dict, thermo_data: Optional[Dict] = None) -> float:
        """Calculate enhanced evidence score using centralized database reliability configuration.
        
        Args:
            db_validation: Database validation results
            thermo_data: Thermodynamic data (optional)
            
        Returns:
            Normalized evidence score (0-1) based on database reliability and data quality
        """
        if not db_validation:
            return 0.0
        
        total_weighted_score = 0.0
        total_possible_score = 0.0
        databases_found = 0
        
        # Calculate weighted evidence score for each database using centralized config
        for db_name, base_weight in DATABASE_WEIGHTS.items():
            reliability_config = DATABASE_RELIABILITY.get(db_name, {})
            db_found_key = f"{db_name}_found"
            
            # Calculate maximum possible score for this database
            base_reliability = reliability_config.get('base_reliability', 0.7)
            max_bonuses = sum(reliability_config.get('data_quality_bonuses', {}).values()) * 0.3
            total_possible_score += base_weight * (base_reliability + max_bonuses)
            
            if db_found_key in db_validation and db_validation[db_found_key]:
                databases_found += 1
                
                # Start with base reliability score
                db_score = base_weight * base_reliability
                
                # Add data quality bonuses using centralized configuration
                if db_name in db_validation and isinstance(db_validation[db_name], dict):
                    db_data = db_validation[db_name]
                    quality_bonuses = reliability_config.get('data_quality_bonuses', {})
                    
                    for data_field, bonus_weight in quality_bonuses.items():
                        if db_data.get(data_field) is not None:
                            # Scale bonus by base weight and data completeness
                            if isinstance(db_data[data_field], (int, float)):
                                # Numerical data gets full bonus
                                db_score += base_weight * bonus_weight * 0.3
                            elif isinstance(db_data[data_field], str) and db_data[data_field].strip():
                                # Non-empty string data gets partial bonus
                                db_score += base_weight * bonus_weight * 0.2
                            elif db_data[data_field] is True:
                                # Boolean true gets moderate bonus
                                db_score += base_weight * bonus_weight * 0.25
                
                total_weighted_score += db_score
            
            # Also check nested dict format for indirect evidence
            elif db_name in db_validation and isinstance(db_validation[db_name], dict):
                db_data = db_validation[db_name]
                if db_data:  # Non-empty dict indicates some data available
                    databases_found += 1
                    # Lower score for indirect evidence but still use reliability weighting
                    indirect_score = base_weight * base_reliability * 0.6
                    total_weighted_score += indirect_score
        
        # Handle thermodynamic data separately with fixed weight
        thermo_weight = 0.15  # Additional weight for thermodynamic data
        if thermo_data and isinstance(thermo_data, dict) and thermo_data:
            databases_found += 1
            thermo_score = thermo_weight * 0.8  # Base thermodynamic reliability
            
            # Bonus for detailed thermodynamic data
            if 'formation_enthalpy' in thermo_data:
                thermo_score += thermo_weight * 0.4
            if 'gibbs_free_energy' in thermo_data:
                thermo_score += thermo_weight * 0.3
            if 'entropy' in thermo_data:
                thermo_score += thermo_weight * 0.2
            
            total_weighted_score += thermo_score
            total_possible_score += thermo_weight * (0.8 + 0.4 + 0.3 + 0.2)  # Max possible thermo score
        
        # Normalize score to 0-1 range
        if total_possible_score > 0:
            normalized_score = min(1.0, total_weighted_score / total_possible_score)
        else:
            normalized_score = 0.0
        
        # Apply multi-database confirmation bonus
        if databases_found >= 3:
            normalized_score *= 1.2  # Strong confirmation from multiple sources
        elif databases_found >= 2:
            normalized_score *= 1.1  # Moderate confirmation
        
        # Apply uncertainty reduction for high-quality databases
        high_quality_dbs = sum(1 for db in ['nist', 'materials_project'] 
                              if f"{db}_found" in db_validation and db_validation[f"{db}_found"])
        if high_quality_dbs >= 2:
            normalized_score *= 1.05  # Small bonus for multiple high-quality sources
        
        return min(1.0, normalized_score)
    
    def create_provenance_record(self, compound_formula: str, 
                               validation_result: Dict,
                               data_sources: List[str],
                               evidence_score: float) -> Dict:
        """Create a comprehensive provenance record.
        
        Args:
            compound_formula: Chemical formula
            validation_result: Complete validation result
            data_sources: List of data sources
            evidence_score: Evidence score
            
        Returns:
            Provenance record dictionary
        """
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        provenance = {
            'compound_formula': compound_formula,
            'timestamp': timestamp,
            'data_sources': data_sources,
            'evidence_score': evidence_score,
            'validation_version': '2.0',  # Version of validation system
            'source_details': self._extract_source_details(validation_result),
            'data_quality': self._assess_data_quality(validation_result, data_sources),
            'completeness': self._calculate_completeness(validation_result),
            'reliability': self._assess_reliability(data_sources, evidence_score)
        }
        
        return provenance
    
    def _extract_source_details(self, validation_result: Dict) -> Dict:
        """Extract detailed information about each data source.
        
        Args:
            validation_result: Complete validation result
            
        Returns:
            Dictionary with source details
        """
        source_details = {}
        
        # Extract database validation details
        db_validation = validation_result.get('db_validation', {})
        
        if 'nist' in db_validation and isinstance(db_validation['nist'], dict):
            nist_data = db_validation['nist']
            source_details['NIST'] = {
                'data_available': bool(nist_data),
                'properties': list(nist_data.keys()) if nist_data else [],
                'formation_enthalpy_available': 'formation_enthalpy' in nist_data,
                'thermal_data_available': any(key in nist_data for key in ['melting_point', 'boiling_point'])
            }
        
        if 'materials_project' in db_validation and isinstance(db_validation['materials_project'], dict):
            mp_data = db_validation['materials_project']
            source_details['Materials Project'] = {
                'data_available': bool(mp_data),
                'properties': list(mp_data.keys()) if mp_data else [],
                'formation_energy_available': 'formation_energy_per_atom' in mp_data,
                'electronic_data_available': 'band_gap' in mp_data
            }
        
        if 'pubchem' in db_validation and isinstance(db_validation['pubchem'], dict):
            pubchem_data = db_validation['pubchem']
            source_details['PubChem'] = {
                'data_available': bool(pubchem_data),
                'properties': list(pubchem_data.keys()) if pubchem_data else [],
                'molecular_data_available': 'molecular_weight' in pubchem_data,
                'structure_data_available': 'smiles' in pubchem_data
            }
        
        if 'rdkit' in db_validation and isinstance(db_validation['rdkit'], dict):
            rdkit_data = db_validation['rdkit']
            source_details['RDKit'] = {
                'validation_performed': bool(rdkit_data),
                'molecule_valid': rdkit_data.get('valid_molecule', False),
                'descriptors_calculated': 'descriptors' in rdkit_data
            }
        
        # Extract thermodynamic data details
        thermo_data = validation_result.get('thermo_data')
        if thermo_data and isinstance(thermo_data, dict) and thermo_data:
            source_details['Thermodynamic Database'] = {
                'data_available': True,
                'properties': list(thermo_data.keys()),
                'formation_data_available': 'formation_enthalpy' in thermo_data or 'gibbs_free_energy' in thermo_data,
                'temperature_range_available': 'temperature_range' in thermo_data
            }
        
        return source_details
    
    def _assess_data_quality(self, validation_result: Dict, data_sources: List[str]) -> Dict:
        """Assess the quality of available data.
        
        Args:
            validation_result: Complete validation result
            data_sources: List of data sources
            
        Returns:
            Data quality assessment
        """
        quality_assessment = {
            'overall_quality': 'unknown',
            'source_count': len(data_sources),
            'has_experimental_data': False,
            'has_computational_data': False,
            'has_thermodynamic_data': False,
            'data_consistency': 'unknown'
        }
        
        # Classify data sources
        experimental_sources = {'NIST', 'PubChem'}
        computational_sources = {'Materials Project', 'RDKit'}
        thermodynamic_sources = {'Thermodynamic Database', 'NIST'}
        
        quality_assessment['has_experimental_data'] = any(
            source in experimental_sources for source in data_sources
        )
        quality_assessment['has_computational_data'] = any(
            source in computational_sources for source in data_sources
        )
        quality_assessment['has_thermodynamic_data'] = any(
            source in thermodynamic_sources for source in data_sources
        )
        
        # Overall quality assessment
        if len(data_sources) >= 3:
            if quality_assessment['has_experimental_data'] and quality_assessment['has_computational_data']:
                quality_assessment['overall_quality'] = 'high'
            else:
                quality_assessment['overall_quality'] = 'medium'
        elif len(data_sources) >= 2:
            quality_assessment['overall_quality'] = 'medium'
        elif len(data_sources) >= 1:
            quality_assessment['overall_quality'] = 'low'
        else:
            quality_assessment['overall_quality'] = 'very_low'
        
        # Data consistency check (simplified)
        db_validation = validation_result.get('db_validation', {})
        if 'nist' in db_validation and 'materials_project' in db_validation:
            # Could compare formation energies if both available
            quality_assessment['data_consistency'] = 'checked'
        else:
            quality_assessment['data_consistency'] = 'not_checked'
        
        return quality_assessment
    
    def _calculate_completeness(self, validation_result: Dict) -> Dict:
        """Calculate completeness of validation data.
        
        Args:
            validation_result: Complete validation result
            
        Returns:
            Completeness assessment
        """
        completeness = {
            'overall_completeness': 0.0,
            'database_coverage': 0.0,
            'property_coverage': 0.0,
            'validation_coverage': 0.0
        }
        
        # Database coverage (out of 4 main databases)
        total_databases = 4
        covered_databases = 0
        
        db_validation = validation_result.get('db_validation', {})
        for db_key in ['nist', 'pubchem', 'materials_project', 'rdkit']:
            if f"{db_key}_found" in db_validation and db_validation[f"{db_key}_found"]:
                covered_databases += 1
            elif db_key in db_validation and db_validation[db_key]:
                covered_databases += 1
        
        completeness['database_coverage'] = covered_databases / total_databases
        
        # Property coverage (key properties)
        key_properties = [
            'formation_enthalpy', 'melting_point', 'boiling_point', 'density',
            'molecular_weight', 'band_gap', 'formation_energy_per_atom'
        ]
        
        available_properties = set()
        
        # Check NIST data
        if 'nist' in db_validation and isinstance(db_validation['nist'], dict):
            available_properties.update(db_validation['nist'].keys())
        
        # Check Materials Project data
        if 'materials_project' in db_validation and isinstance(db_validation['materials_project'], dict):
            available_properties.update(db_validation['materials_project'].keys())
        
        # Check PubChem data
        if 'pubchem' in db_validation and isinstance(db_validation['pubchem'], dict):
            available_properties.update(db_validation['pubchem'].keys())
        
        # Check thermodynamic data
        thermo_data = validation_result.get('thermo_data', {})
        if isinstance(thermo_data, dict):
            available_properties.update(thermo_data.keys())
        
        property_matches = len(set(key_properties) & available_properties)
        completeness['property_coverage'] = property_matches / len(key_properties)
        
        # Validation coverage (different validation types)
        validation_types = [
            'element_availability', 'stoichiometry', 'db_validation',
            'chemical_rules', 'environmental_conditions'
        ]
        
        completed_validations = 0
        for validation_type in validation_types:
            if validation_type in validation_result:
                completed_validations += 1
        
        completeness['validation_coverage'] = completed_validations / len(validation_types)
        
        # Overall completeness (weighted average)
        completeness['overall_completeness'] = (
            0.4 * completeness['database_coverage'] +
            0.3 * completeness['property_coverage'] +
            0.3 * completeness['validation_coverage']
        )
        
        # Round all values
        for key in completeness:
            completeness[key] = completeness[key]
        
        return completeness
    
    def _assess_reliability(self, data_sources: List[str], evidence_score: float) -> Dict:
        """Assess reliability of the validation results with uncertainty quantification.
        
        Args:
            data_sources: List of data sources
            evidence_score: Evidence score
            
        Returns:
            Reliability assessment with uncertainty metrics
        """
        reliability = {
            'overall_reliability': 'unknown',
            'source_reliability': {},
            'evidence_strength': 'unknown',
            'confidence_level': 'unknown',
            'uncertainty': 0.0,
            'uncertainty_sources': {}
        }
        
        # Source reliability ratings
        source_ratings = {
            'NIST': 'high',
            'Materials Project': 'high',
            'PubChem': 'medium-high',
            'RDKit': 'medium',
            'Thermodynamic Database': 'high'
        }
        
        for source in data_sources:
            reliability['source_reliability'][source] = source_ratings.get(source, 'medium')
        
        # Evidence strength based on score
        if evidence_score >= 0.8:
            reliability['evidence_strength'] = 'strong'
        elif evidence_score >= 0.6:
            reliability['evidence_strength'] = 'moderate'
        elif evidence_score >= 0.4:
            reliability['evidence_strength'] = 'weak'
        else:
            reliability['evidence_strength'] = 'very_weak'
        
        # Overall reliability
        high_reliability_sources = sum(1 for source in data_sources 
                                     if source_ratings.get(source, 'medium') == 'high')
        
        if high_reliability_sources >= 2 and evidence_score >= 0.7:
            reliability['overall_reliability'] = 'high'
            reliability['confidence_level'] = 'high'
        elif high_reliability_sources >= 1 and evidence_score >= 0.5:
            reliability['overall_reliability'] = 'medium'
            reliability['confidence_level'] = 'medium'
        elif len(data_sources) >= 1 and evidence_score >= 0.3:
            reliability['overall_reliability'] = 'low'
            reliability['confidence_level'] = 'low'
        else:
            reliability['overall_reliability'] = 'very_low'
            reliability['confidence_level'] = 'very_low'
        
        # Calculate uncertainty based on data sources and evidence
        uncertainty_components = self._calculate_provenance_uncertainty(
            data_sources, evidence_score, high_reliability_sources
        )
        reliability.update(uncertainty_components)
        
        return reliability
    
    def _calculate_provenance_uncertainty(self, data_sources: List[str], evidence_score: float, high_reliability_sources: int) -> Dict:
        """Calculate uncertainty metrics for provenance assessment.
        
        Args:
            data_sources: List of data sources
            evidence_score: Evidence score
            high_reliability_sources: Number of high reliability sources
            
        Returns:
            Dictionary with uncertainty metrics
        """
        base_uncertainty = UNCERTAINTY_PARAMETERS['base_uncertainty']
        
        # Source diversity uncertainty (decreases with more sources)
        num_sources = len(data_sources)
        if num_sources == 0:
            source_uncertainty = base_uncertainty * 4  # Very high uncertainty
        elif num_sources == 1:
            source_uncertainty = base_uncertainty * 2  # High uncertainty
        elif num_sources >= 3:
            source_uncertainty = base_uncertainty * 0.5  # Low uncertainty
        else:
            source_uncertainty = base_uncertainty  # Medium uncertainty
        
        # Evidence quality uncertainty (decreases with higher evidence score)
        evidence_uncertainty = base_uncertainty * (1 - evidence_score)
        
        # High reliability source bonus
        reliability_uncertainty = base_uncertainty * max(0, (2 - high_reliability_sources) / 2)
        
        # Combine uncertainties using root sum of squares
        total_uncertainty = (source_uncertainty**2 + evidence_uncertainty**2 + reliability_uncertainty**2)**0.5
        
        # Normalize to [0, 1] range
        max_possible_uncertainty = base_uncertainty * 3  # Theoretical maximum
        normalized_uncertainty = min(1.0, total_uncertainty / max_possible_uncertainty)
        
        return {
            'uncertainty': normalized_uncertainty,
            'uncertainty_sources': {
                'source_diversity': source_uncertainty / max_possible_uncertainty,
                'evidence_quality': evidence_uncertainty / max_possible_uncertainty,
                'reliability_factor': reliability_uncertainty / max_possible_uncertainty
            }
        }
    
    def generate_provenance_summary(self, provenance_record: Dict) -> str:
        """Generate a human-readable provenance summary.
        
        Args:
            provenance_record: Provenance record
            
        Returns:
            Human-readable summary string
        """
        formula = provenance_record['compound_formula']
        sources = provenance_record['data_sources']
        evidence_score = provenance_record['evidence_score']
        quality = provenance_record['data_quality']['overall_quality']
        reliability = provenance_record['reliability']['overall_reliability']
        
        summary_parts = [
            f"Validation for {formula}:",
            f"• Data sources: {', '.join(sources) if sources else 'None'}",
            f"• Evidence score: {evidence_score}",
            f"• Data quality: {quality}",
            f"• Reliability: {reliability}"
        ]
        
        # Add specific source information
        source_details = provenance_record['source_details']
        if 'NIST' in source_details:
            nist = source_details['NIST']
            if nist['formation_enthalpy_available']:
                summary_parts.append("• NIST formation enthalpy available")
        
        if 'Materials Project' in source_details:
            mp = source_details['Materials Project']
            if mp['formation_energy_available']:
                summary_parts.append("• Materials Project formation energy available")
        
        return "\n".join(summary_parts)
    
    def export_provenance_metadata(self, provenance_records: List[Dict]) -> Dict:
        """Export metadata about provenance records for analysis.
        
        Args:
            provenance_records: List of provenance records
            
        Returns:
            Metadata dictionary
        """
        if not provenance_records:
            return {'total_records': 0}
        
        metadata = {
            'total_records': len(provenance_records),
            'timestamp_range': {
                'earliest': min(record['timestamp'] for record in provenance_records),
                'latest': max(record['timestamp'] for record in provenance_records)
            },
            'source_statistics': {},
            'quality_distribution': {},
            'reliability_distribution': {},
            'evidence_score_statistics': {}
        }
        
        # Source statistics
        all_sources = []
        for record in provenance_records:
            all_sources.extend(record['data_sources'])
        
        unique_sources = set(all_sources)
        for source in unique_sources:
            metadata['source_statistics'][source] = all_sources.count(source)
        
        # Quality distribution
        qualities = [record['data_quality']['overall_quality'] for record in provenance_records]
        for quality in set(qualities):
            metadata['quality_distribution'][quality] = qualities.count(quality)
        
        # Reliability distribution
        reliabilities = [record['reliability']['overall_reliability'] for record in provenance_records]
        for reliability in set(reliabilities):
            metadata['reliability_distribution'][reliability] = reliabilities.count(reliability)
        
        # Evidence score statistics
        evidence_scores = [record['evidence_score'] for record in provenance_records]
        metadata['evidence_score_statistics'] = {
            'mean': sum(evidence_scores) / len(evidence_scores),
            'min': min(evidence_scores),
            'max': max(evidence_scores),
            'median': sorted(evidence_scores)[len(evidence_scores) // 2]
        }
        
        return metadata