"""Machine Learning Dataset Creation Pipeline

This module provides comprehensive functionality for creating structured CSV datasets
suitable for machine learning training, with proper feature-label separation,
data normalization, and quality validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import json
import logging

# Optional sklearn imports
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some ML dataset features will be limited.")

from ..data.data_models import ValidationResult, EnvironmentalConditions
from ..config.constants import VALIDATION_THRESHOLDS


class MLDatasetSchema:
    """Defines the schema for ML dataset with clear feature-label separation."""
    
    # Feature columns (input parameters)
    FEATURE_COLUMNS = {
        # Element composition features
        'element_count': 'int',
        'avg_atomic_number': 'float',
        'avg_atomic_mass': 'float',
        'electronegativity_range': 'float',
        'electronegativity_mean': 'float',
        'metallic_ratio': 'float',
        'nonmetal_ratio': 'float',
        'transition_metal_ratio': 'float',
        
        # Environmental condition features
        'temperature_kelvin': 'float',
        'pressure_pa': 'float',
        'ph_value': 'float',
        'radiation_level': 'float',
        'magnetic_field': 'float',
        'atmosphere_encoded': 'int',
        
        # Thermodynamic features
        'enthalpy_estimate': 'float',
        'entropy_estimate': 'float',
        'gibbs_energy_estimate': 'float',
        
        # Chemical rule features
        'oxidation_compatibility': 'float',
        'electronegativity_compatibility': 'float',
        'ionic_radius_compatibility': 'float',
        
        # Database evidence features
        'nist_evidence': 'float',
        'pubchem_evidence': 'float',
        'materials_project_evidence': 'float',
        'rdkit_evidence': 'float',
        'total_evidence_score': 'float',
        
        # Uncertainty quantification features
        'formation_uncertainty': 'float',
        'thermodynamic_uncertainty': 'float',
        'kinetic_uncertainty': 'float',
        'database_uncertainty': 'float'
    }
    
    # Label columns (target variables)
    LABEL_COLUMNS = {
        # Primary targets
        'formation_probability': 'float',  # Main regression target
        'is_feasible': 'bool',             # Main classification target
        
        # Secondary targets
        'feasibility_confidence': 'float',
        'limiting_factor_count': 'int',
        'evidence_quality': 'float'
    }
    
    # Metadata columns (not used for training)
    METADATA_COLUMNS = {
        'sample_id': 'int',
        'timestamp': 'str',
        'compound_formula': 'str',
        'compound_name': 'str',
        'data_sources': 'str',
        'schema_version': 'str',
        'method_version': 'str'
    }
    
    @classmethod
    def get_all_columns(cls) -> Dict[str, str]:
        """Get all columns with their types."""
        return {**cls.FEATURE_COLUMNS, **cls.LABEL_COLUMNS, **cls.METADATA_COLUMNS}
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature column names."""
        return list(cls.FEATURE_COLUMNS.keys())
    
    @classmethod
    def get_label_names(cls) -> List[str]:
        """Get list of label column names."""
        return list(cls.LABEL_COLUMNS.keys())


class MLDatasetCreator:
    """Creates structured ML datasets from chemistry validation results."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ML dataset creator.
        
        Args:
            config: Configuration dictionary for dataset creation
        """
        self.config = config or {}
        self.schema = MLDatasetSchema()
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoders and scalers if sklearn is available
        if SKLEARN_AVAILABLE:
            self.atmosphere_encoder = LabelEncoder()
            self.feature_scaler = StandardScaler()
            self.target_scaler = MinMaxScaler()
        else:
            self.atmosphere_encoder = None
            self.feature_scaler = None
            self.target_scaler = None
        
        # Track data quality metrics
        self.quality_metrics = {
            'total_samples': 0,
            'valid_samples': 0,
            'missing_values': {},
            'outliers': {},
            'data_completeness': 0.0
        }
    
    def create_ml_dataset(self, samples: List[Dict], 
                         target_type: str = 'formation_probability',
                         include_features: List[str] = None,
                         output_path: Optional[str] = None,
                         split_data: bool = True,
                         normalize_features: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create comprehensive ML dataset from chemistry samples.
        
        Args:
            samples: List of sample dictionaries from the main generator
            target_type: Type of target variable ('formation_probability', 'feasible', etc.)
            include_features: Optional list of specific features to include
            output_path: Optional path to save the dataset
            split_data: Whether to create train/test splits
            normalize_features: Whether to normalize feature values
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        self.logger.info(f"Creating ML dataset from {len(samples)} samples")
        
        # Extract features and labels
        dataset_df = self._extract_features_and_labels(samples, target_type, include_features)
        
        # Validate data quality
        self._validate_data_quality(dataset_df)
        
        # Normalize features if requested
        if normalize_features:
            dataset_df = self._normalize_features(dataset_df)
        
        # Generate dataset metadata
        metadata = self._generate_dataset_metadata(dataset_df)
        
        # Create dataset splits if requested
        if split_data:
            splits = self._create_data_splits(dataset_df)
            metadata['splits'] = {name: len(df) for name, df in splits.items()}
            
            # Save dataset if path provided
            if output_path:
                self._save_dataset({'full_dataset': dataset_df, **splits}, output_path)
        else:
            if output_path:
                self._save_dataset({'full_dataset': dataset_df}, output_path)
        
        self.logger.info(f"ML dataset created successfully with {len(dataset_df)} samples")
        return dataset_df, metadata
    
    def _extract_features_and_labels(self, samples: List[Dict], 
                                   target_type: str,
                                   include_features: List[str] = None) -> pd.DataFrame:
        """Extract features and labels from samples."""
        rows = []
        
        for sample in samples:
            # Skip samples without validation results
            if 'validation_results' not in sample or not sample['validation_results']:
                continue
                
            # Get best result for this sample
            best_result = sample.get('best_compound')
            if not best_result:
                # Find best result from validation results
                validation_results = sample['validation_results']
                if validation_results:
                    # Handle both dict and ValidationResult object formats
                    if isinstance(validation_results[0], dict):
                        best_result = max(validation_results, key=lambda r: r.get('formation_probability', 0))
                    else:
                        best_result = max(validation_results, key=lambda r: r.formation_probability)
            
            if not best_result:
                continue
                
            # Extract features
            features = self._extract_sample_features(sample, best_result)
            
            # Extract labels
            labels = self._extract_sample_labels(best_result, target_type)
            
            # Extract metadata
            metadata = self._extract_sample_metadata(sample, best_result)
            
            # Filter features if specified
            if include_features:
                features = {k: v for k, v in features.items() if k in include_features}
            
            # Combine all data
            row = {**features, **labels, **metadata}
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _extract_sample_features(self, sample: Dict, result) -> Dict[str, Any]:
        """Extract feature values from a sample and validation result."""
        features = {}
        
        # Element composition features
        elements = sample['elements']
        features['element_count'] = len(elements)
        
        # Calculate element statistics from element_quantities
        element_quantities = sample.get('element_quantities', {})
        quantities = [data['quantity'] for data in element_quantities.values()]
        
        # Get atomic numbers from ElementDatabase if available
        atomic_numbers = []
        if hasattr(self, 'element_db') and self.element_db:
            for element_symbol in element_quantities.keys():
                if element_symbol in self.element_db.properties:
                    atomic_numbers.append(self.element_db.properties[element_symbol]['atomic_number'])
        
        # Fallback: use a simple mapping if ElementDatabase not available
        if not atomic_numbers:
            element_atomic_numbers = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94
            }
            atomic_numbers = [element_atomic_numbers.get(element_symbol, 0) for element_symbol in element_quantities.keys()]
        
        if atomic_numbers:
            features['avg_atomic_number'] = sum(atomic_numbers) / len(atomic_numbers)
            features['min_atomic_number'] = min(atomic_numbers)
            features['max_atomic_number'] = max(atomic_numbers)
        else:
            features['avg_atomic_number'] = 0.0
            features['min_atomic_number'] = 0.0
            features['max_atomic_number'] = 0.0
            
        if quantities:
            features['total_quantity'] = sum(quantities)
            features['avg_quantity'] = sum(quantities) / len(quantities)
        else:
            features['total_quantity'] = 0.0
            features['avg_quantity'] = 0.0
        
        # Environmental condition features
        conditions = sample['conditions']
        features['temperature_kelvin'] = conditions['temperature']
        features['pressure_pa'] = conditions['pressure']
        features['ph_value'] = conditions['pH']
        features['radiation_level'] = conditions.get('radiation_level', 0.0)
        features['magnetic_field'] = conditions.get('magnetic_field', 0.0)
        features['atmosphere'] = conditions.get('atmosphere', 'air')
        
        # Handle both dict and ValidationResult object formats
        if isinstance(result, dict):
            validation_details = result.get('validation_details', {})
            data_sources = result.get('data_sources', [])
            evidence_score = result.get('evidence_score', 0.0)
        else:
            validation_details = result.validation_details or {}
            data_sources = result.data_sources or []
            evidence_score = result.evidence_score or 0.0
        
        # Thermodynamic features (from validation details)
        features['enthalpy_estimate'] = validation_details.get('enthalpy', 0.0)
        features['entropy_estimate'] = validation_details.get('entropy', 0.0)
        features['gibbs_energy_estimate'] = validation_details.get('gibbs_energy', 0.0)
        
        # Chemical rule features
        chemical_rules = validation_details.get('chemical_rules', {})
        features['oxidation_compatibility'] = chemical_rules.get('oxidation_score', 0.5)
        features['electronegativity_compatibility'] = chemical_rules.get('electronegativity_score', 0.5)
        features['ionic_radius_compatibility'] = chemical_rules.get('ionic_radius_score', 0.5)
        
        # Database evidence features
        features['nist_evidence'] = 1.0 if 'NIST' in data_sources else 0.0
        features['pubchem_evidence'] = 1.0 if 'PubChem' in data_sources else 0.0
        features['materials_project_evidence'] = 1.0 if 'Materials Project' in data_sources else 0.0
        features['rdkit_evidence'] = 1.0 if 'RDKit' in data_sources else 0.0
        features['total_evidence_score'] = evidence_score
        
        # Uncertainty quantification features
        uncertainty_details = validation_details.get('uncertainty', {})
        features['formation_uncertainty'] = uncertainty_details.get('formation', 0.1)
        features['thermodynamic_uncertainty'] = uncertainty_details.get('thermodynamic', 0.1)
        features['kinetic_uncertainty'] = uncertainty_details.get('kinetic', 0.1)
        features['database_uncertainty'] = uncertainty_details.get('database', 0.1)
        
        return features
    
    def _extract_sample_labels(self, result, target_type: str) -> Dict[str, Any]:
        """Extract label values from a validation result."""
        labels = {}
        
        # Handle both dict and ValidationResult object formats
        if isinstance(result, dict):
            labels['formation_probability'] = result.get('formation_probability', 0.0)
            labels['is_feasible'] = result.get('feasible', False)
            confidence = result.get('confidence', 0.5)
            limiting_factors = result.get('limiting_factors', [])
            evidence_score = result.get('evidence_score', 0.0)
        else:
            labels['formation_probability'] = result.formation_probability
            labels['is_feasible'] = result.feasible
            confidence = getattr(result, 'confidence', 0.5)
            limiting_factors = result.limiting_factors or []
            evidence_score = result.evidence_score or 0.0
        
        # Add secondary targets based on target_type
        if target_type in ['all', 'formation_probability', 'feasible']:
            labels['feasibility_confidence'] = confidence
            labels['limiting_factor_count'] = len(limiting_factors)
            labels['evidence_quality'] = evidence_score
        
        return labels
    
    def _extract_sample_metadata(self, sample: Dict, result) -> Dict[str, Any]:
        """Extract metadata from a sample and validation result."""
        # Handle both dict and ValidationResult object formats
        if isinstance(result, dict):
            compound_formula = result.get('formula', '')
            compound_name = result.get('name', '')
            data_sources = result.get('data_sources', [])
            schema_version = result.get('schema_version', '1.0.0')
            method_version = result.get('method_version', '1.0.0')
        else:
            compound_formula = result.compound.formula
            compound_name = result.compound.name or ''
            data_sources = result.data_sources or []
            schema_version = result.schema_version or '1.0.0'
            method_version = result.method_version or '1.0.0'
        
        metadata = {
            'sample_id': sample['sample_id'],
            'timestamp': sample['timestamp'],
            'compound_formula': compound_formula,
            'compound_name': compound_name,
            'data_sources': ', '.join(data_sources),
            'schema_version': schema_version,
            'method_version': method_version
        }
        return metadata
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Validate data quality and update quality metrics."""
        self.quality_metrics['total_samples'] = len(df)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        self.quality_metrics['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Check for outliers (simplified - using IQR method)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        self.quality_metrics['outliers'] = outlier_counts
        
        # Calculate data completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        self.quality_metrics['data_completeness'] = (total_cells - missing_cells) / total_cells
        
        # Count valid samples (no missing target values)
        target_columns = self.schema.get_label_names()
        valid_samples = df[target_columns].dropna().shape[0]
        self.quality_metrics['valid_samples'] = valid_samples
        
        self.logger.info(f"Data quality validation complete: {valid_samples}/{len(df)} valid samples")
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature columns for ML training."""
        df_normalized = df.copy()
        feature_columns = [col for col in self.schema.get_feature_names() if col in df.columns]
        
        # Normalize features if sklearn is available
        if SKLEARN_AVAILABLE and self.feature_scaler and feature_columns:
            df_normalized[feature_columns] = self.feature_scaler.fit_transform(df[feature_columns])
            self.logger.info(f"Normalized {len(feature_columns)} feature columns using StandardScaler")
        elif feature_columns:
            # Simple min-max normalization fallback
            for col in feature_columns:
                if df[col].dtype in ['int64', 'float64']:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if col_max != col_min:
                        df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            self.logger.info(f"Normalized {len(feature_columns)} feature columns using simple min-max scaling")
        
        return df_normalized
    
    def _create_data_splits(self, df: pd.DataFrame, 
                          test_size: float = 0.2, 
                          val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits."""
        if SKLEARN_AVAILABLE and 'is_feasible' in df.columns:
            # Use sklearn for stratified splits
            try:
                # First split: train+val vs test
                train_val, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['is_feasible'])
                
                # Second split: train vs val
                val_size_adjusted = val_size / (1 - test_size)
                train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42, stratify=train_val['is_feasible'])
                
                self.logger.info(f"Created stratified data splits: train={len(train)}, val={len(val)}, test={len(test)}")
            except ValueError:
                # Fallback to random splits if stratification fails
                train_val, test = train_test_split(df, test_size=test_size, random_state=42)
                val_size_adjusted = val_size / (1 - test_size)
                train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42)
                
                self.logger.info(f"Created random data splits: train={len(train)}, val={len(val)}, test={len(test)}")
        else:
            # Simple random splits without sklearn
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            
            test_end = int(len(df) * test_size)
            val_end = test_end + int(len(df) * val_size)
            
            test_indices = indices[:test_end]
            val_indices = indices[test_end:val_end]
            train_indices = indices[val_end:]
            
            test = df.iloc[test_indices].copy()
            val = df.iloc[val_indices].copy()
            train = df.iloc[train_indices].copy()
            
            self.logger.info(f"Created simple random data splits: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return {
            'train': train,
            'validation': val,
            'test': test
        }
    
    def _save_dataset(self, dataset_dict: Dict[str, pd.DataFrame], output_path: str) -> None:
        """Save dataset to CSV files with proper organization."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for split_name, df in dataset_dict.items():
            if isinstance(df, pd.DataFrame):
                filename = f"chemistry_ml_{split_name}_{timestamp}.csv"
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
                self.logger.info(f"Saved {split_name} dataset: {filepath}")
        
        # Save dataset schema and metadata
        schema_info = {
            'feature_columns': self.schema.FEATURE_COLUMNS,
            'label_columns': self.schema.LABEL_COLUMNS,
            'metadata_columns': self.schema.METADATA_COLUMNS,
            'quality_metrics': self.quality_metrics,
            'creation_timestamp': datetime.now().isoformat()
        }
        
        schema_file = output_dir / f"dataset_schema_{timestamp}.json"
        with open(schema_file, 'w') as f:
            json.dump(schema_info, f, indent=2)
        
        self.logger.info(f"Saved dataset schema: {schema_file}")
    
    def _generate_dataset_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset metadata."""
        feature_names = self.schema.get_feature_names()
        label_names = self.schema.get_label_names()
        
        metadata = {
            'dataset_info': {
                'total_samples': len(df),
                'feature_count': len([col for col in feature_names if col in df.columns]),
                'label_count': len([col for col in label_names if col in df.columns]),
                'creation_timestamp': datetime.now().isoformat(),
                'schema_version': '1.0.0'
            },
            'feature_statistics': {},
            'label_statistics': {},
            'quality_metrics': self.quality_metrics
        }
        
        # Calculate feature statistics
        for col in feature_names:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                metadata['feature_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'missing_count': int(df[col].isnull().sum())
                }
        
        # Calculate label statistics
        for col in label_names:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    metadata['label_statistics'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'missing_count': int(df[col].isnull().sum())
                    }
                elif df[col].dtype == 'bool':
                    metadata['label_statistics'][col] = {
                        'true_count': int(df[col].sum()),
                        'false_count': int((~df[col]).sum()),
                        'missing_count': int(df[col].isnull().sum())
                    }
        
        return metadata
    
    def generate_dataset_documentation(self, metadata: Dict[str, Any], output_path: str) -> None:
        """Generate comprehensive dataset documentation."""
        doc_content = self._create_documentation_content(metadata)
        
        doc_file = Path(output_path) / "dataset_documentation.md"
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        
        self.logger.info(f"Generated dataset documentation: {doc_file}")
    
    def _create_documentation_content(self, metadata: Dict[str, Any]) -> str:
        """Create markdown documentation content."""
        content = f"""# Chemistry ML Dataset Documentation

Generated on: {metadata['dataset_info']['creation_timestamp']}
Schema Version: {metadata['dataset_info']['schema_version']}

## Dataset Overview

- **Total Samples**: {metadata['dataset_info']['total_samples']}
- **Feature Count**: {metadata['dataset_info']['feature_count']}
- **Label Count**: {metadata['dataset_info']['label_count']}
- **Data Completeness**: {metadata['quality_metrics']['data_completeness']:.2%}

## Schema Definition

### Feature Columns (Input Parameters)

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
"""
        
        for col, dtype in self.schema.FEATURE_COLUMNS.items():
            content += f"| {col} | {dtype} | Feature description |\n"
        
        content += "\n### Label Columns (Target Variables)\n\n| Column Name | Data Type | Description |\n|-------------|-----------|-------------|\n"
        
        for col, dtype in self.schema.LABEL_COLUMNS.items():
            content += f"| {col} | {dtype} | Target variable description |\n"
        
        content += "\n### Metadata Columns\n\n| Column Name | Data Type | Description |\n|-------------|-----------|-------------|\n"
        
        for col, dtype in self.schema.METADATA_COLUMNS.items():
            content += f"| {col} | {dtype} | Metadata description |\n"
        
        content += f"\n## Data Quality Metrics\n\n- **Valid Samples**: {metadata['quality_metrics']['valid_samples']}/{metadata['quality_metrics']['total_samples']}\n"
        
        if metadata['quality_metrics']['missing_values']:
            content += "\n### Missing Values\n\n"
            for col, count in metadata['quality_metrics']['missing_values'].items():
                content += f"- **{col}**: {count} missing values\n"
        
        if metadata['quality_metrics']['outliers']:
            content += "\n### Outliers Detected\n\n"
            for col, count in metadata['quality_metrics']['outliers'].items():
                content += f"- **{col}**: {count} outliers\n"
        
        content += "\n## Usage Guidelines\n\n1. **Feature Selection**: Use feature columns as input (X) for ML models\n2. **Target Selection**: Use label columns as targets (y) for ML models\n3. **Data Splitting**: Pre-split datasets are provided (train/validation/test)\n4. **Normalization**: Features are pre-normalized using StandardScaler\n5. **Missing Values**: Handle missing values appropriately before training\n\n## Recommended ML Approaches\n\n- **Regression**: Predict `formation_probability` using feature columns\n- **Classification**: Predict `is_feasible` using feature columns\n- **Multi-task Learning**: Predict multiple targets simultaneously\n- **Uncertainty Quantification**: Use uncertainty features for confidence estimation\n"""
        
        return content