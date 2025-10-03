#!/usr/bin/env python3
"""
NASA Rover ML Training System
============================

Trains machine learning models on drilling data to predict element and compound
discoveries for future locations. Includes pattern recognition and prediction capabilities.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib


@dataclass
class MLPrediction:
    """ML prediction result"""
    prediction_type: str  # 'element_abundance', 'compound_formation', 'discovery_potential'
    predicted_value: float
    confidence: float
    factors_influence: Dict[str, float]  # Feature importance


@dataclass 
class MLModelSuite:
    """Complete suite of trained ML models"""
    element_abundance_model: RandomForestRegressor
    compound_formation_model: RandomForestClassifier
    discovery_potential_model: RandomForestRegressor
    earth_similarity_model: RandomForestRegressor
    feature_scaler: StandardScaler
    label_encoders: Dict[str, LabelEncoder]
    feature_names: List[str]
    training_timestamp: datetime
    model_performance: Dict[str, float]


class NASARoverMLTrainer:
    """Advanced ML training system for NASA rover data"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize models
        self.element_abundance_model = RandomForestRegressor(
            n_estimators=100, random_state=random_seed, max_depth=10
        )
        self.compound_formation_model = RandomForestClassifier(
            n_estimators=100, random_state=random_seed, max_depth=10
        )
        self.discovery_potential_model = RandomForestRegressor(
            n_estimators=100, random_state=random_seed, max_depth=10
        )
        self.earth_similarity_model = RandomForestRegressor(
            n_estimators=100, random_state=random_seed, max_depth=10
        )
        
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
    
    def prepare_training_data(self, detailed_reports: List) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Prepare training data from detailed drilling reports"""
        
        print("🔧 Preparing ML training data...")
        
        training_data = []
        
        for report in detailed_reports:
            # Environmental features
            base_features = {
                'temperature': report.temperature,
                'pressure': report.pressure,
                'pH': report.pH,
                'humidity': report.humidity,
                'radiation_level': report.radiation_level,
                'depth': report.depth,
                'site_type_encoded': self._encode_site_type(report.site_type),
                'latitude': report.coordinates[0],
                'longitude': report.coordinates[1],
            }
            
            # Element abundance features (top 10 most common elements)
            common_elements = ['O', 'Si', 'Al', 'Fe', 'Ca', 'Na', 'K', 'Mg', 'Ti', 'H']
            for element in common_elements:
                element_analysis = next((ea for ea in report.element_analyses if ea.symbol == element), None)
                base_features[f'element_{element}_percentage'] = element_analysis.percentage if element_analysis else 0.0
                base_features[f'element_{element}_present'] = 1.0 if element_analysis else 0.0
            
            # Derived features
            base_features['total_elements'] = report.total_elements_found
            base_features['element_diversity'] = len([ea for ea in report.element_analyses if ea.percentage > 1.0])
            base_features['rare_element_count'] = len(report.rare_elements)
            base_features['unknown_element_count'] = len(report.unknown_elements)
            
            # Target variables for different prediction tasks
            targets = {
                'earth_similarity_score': report.earth_similarity_score,
                'discovery_potential': report.discovery_potential,
                'scientific_value': report.scientific_value,
                'compound_success_rate': len(report.known_compounds) / max(1, report.total_compounds_tested) * 100,
                'unknown_compound_rate': len(report.unknown_compounds) / max(1, report.total_compounds_tested) * 100,
            }
            
            # Add compound formation success as classification target
            compound_formation_success = 1 if len(report.known_compounds) >= 3 else 0
            targets['compound_formation_success'] = compound_formation_success
            
            # Combine features and targets
            row_data = {**base_features, **targets}
            training_data.append(row_data)
        
        df = pd.DataFrame(training_data)
        
        # Prepare feature matrix and target arrays
        feature_columns = [col for col in df.columns if col not in [
            'earth_similarity_score', 'discovery_potential', 'scientific_value', 
            'compound_success_rate', 'unknown_compound_rate', 'compound_formation_success'
        ]]
        
        X = df[feature_columns].values
        self.feature_names = feature_columns
        
        targets = {
            'earth_similarity': df['earth_similarity_score'].values,
            'discovery_potential': df['discovery_potential'].values,
            'compound_formation': df['compound_formation_success'].values,
            'element_abundance': df['total_elements'].values,
        }
        
        print(f"   ✅ Prepared {len(df)} training samples with {len(feature_columns)} features")
        return df, targets
    
    def _encode_site_type(self, site_type: str) -> float:
        """Encode site type as numeric value"""
        site_type_map = {
            'surface': 1.0, 'crater': 2.0, 'crater_rim': 3.0, 'slope': 4.0,
            'ridge': 5.0, 'dune': 6.0, 'plateau': 7.0, 'subsurface': 8.0,
            'anomaly': 9.0, 'valley': 10.0
        }
        return site_type_map.get(site_type, 0.0)
    
    def train_models(self, training_df: pd.DataFrame, targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train all ML models and return performance metrics"""
        
        print("🧠 Training ML models...")
        
        # Prepare feature matrix
        feature_columns = [col for col in training_df.columns if col not in [
            'earth_similarity_score', 'discovery_potential', 'scientific_value', 
            'compound_success_rate', 'unknown_compound_rate', 'compound_formation_success'
        ]]
        
        X = training_df[feature_columns].values
        X_scaled = self.feature_scaler.fit_transform(X)
        
        performance = {}
        
        # Train Earth Similarity Model
        print("   🌍 Training Earth similarity prediction model...")
        y_earth = targets['earth_similarity']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_earth, test_size=0.2, random_state=self.random_seed)
        
        self.earth_similarity_model.fit(X_train, y_train)
        y_pred = self.earth_similarity_model.predict(X_test)
        performance['earth_similarity_mse'] = mean_squared_error(y_test, y_pred)
        performance['earth_similarity_r2'] = self.earth_similarity_model.score(X_test, y_test)
        
        # Train Discovery Potential Model
        print("   🔍 Training discovery potential prediction model...")
        y_discovery = targets['discovery_potential']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_discovery, test_size=0.2, random_state=self.random_seed)
        
        self.discovery_potential_model.fit(X_train, y_train)
        y_pred = self.discovery_potential_model.predict(X_test)
        performance['discovery_potential_mse'] = mean_squared_error(y_test, y_pred)
        performance['discovery_potential_r2'] = self.discovery_potential_model.score(X_test, y_test)
        
        # Train Compound Formation Model (Classification)
        print("   ⚗️ Training compound formation success model...")
        y_compound = targets['compound_formation']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_compound, test_size=0.2, random_state=self.random_seed)
        
        self.compound_formation_model.fit(X_train, y_train)
        y_pred = self.compound_formation_model.predict(X_test)
        performance['compound_formation_accuracy'] = accuracy_score(y_test, y_pred)
        
        # Train Element Abundance Model
        print("   🧪 Training element abundance prediction model...")
        y_elements = targets['element_abundance']
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_elements, test_size=0.2, random_state=self.random_seed)
        
        self.element_abundance_model.fit(X_train, y_train)
        y_pred = self.element_abundance_model.predict(X_test)
        performance['element_abundance_mse'] = mean_squared_error(y_test, y_pred)
        performance['element_abundance_r2'] = self.element_abundance_model.score(X_test, y_test)
        
        self.is_trained = True
        print("   ✅ All models trained successfully!")
        
        return performance
    
    def predict_location_potential(self, environmental_features: Dict[str, float]) -> Dict[str, MLPrediction]:
        """Predict potential for a new drilling location"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare feature vector
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in environmental_features:
                feature_vector[i] = environmental_features[feature_name]
            else:
                feature_vector[i] = 0.0  # Default for missing features
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform([feature_vector])
        
        predictions = {}
        
        # Earth Similarity Prediction
        earth_sim_pred = self.earth_similarity_model.predict(feature_vector_scaled)[0]
        earth_sim_confidence = np.mean(self.earth_similarity_model.predict(feature_vector_scaled))
        earth_sim_importance = dict(zip(self.feature_names, self.earth_similarity_model.feature_importances_))
        
        predictions['earth_similarity'] = MLPrediction(
            prediction_type='earth_similarity',
            predicted_value=earth_sim_pred,
            confidence=min(100.0, earth_sim_confidence),
            factors_influence=dict(sorted(earth_sim_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        )
        
        # Discovery Potential Prediction
        discovery_pred = self.discovery_potential_model.predict(feature_vector_scaled)[0]
        discovery_confidence = np.mean(self.discovery_potential_model.predict(feature_vector_scaled))
        discovery_importance = dict(zip(self.feature_names, self.discovery_potential_model.feature_importances_))
        
        predictions['discovery_potential'] = MLPrediction(
            prediction_type='discovery_potential',
            predicted_value=discovery_pred,
            confidence=min(100.0, discovery_confidence),
            factors_influence=dict(sorted(discovery_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        )
        
        # Compound Formation Prediction
        compound_pred_proba = self.compound_formation_model.predict_proba(feature_vector_scaled)[0]
        
        # Handle case where only one class was trained (small dataset)
        if len(compound_pred_proba) == 1:
            # If only one class, assume it's the positive class
            compound_pred = compound_pred_proba[0] * 100
            compound_confidence = compound_pred_proba[0] * 100
        else:
            compound_pred = compound_pred_proba[1] * 100  # Probability of success * 100
            compound_confidence = np.max(compound_pred_proba) * 100
        compound_importance = dict(zip(self.feature_names, self.compound_formation_model.feature_importances_))
        
        predictions['compound_formation'] = MLPrediction(
            prediction_type='compound_formation',
            predicted_value=compound_pred,
            confidence=compound_confidence,
            factors_influence=dict(sorted(compound_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        )
        
        # Element Abundance Prediction
        element_pred = self.element_abundance_model.predict(feature_vector_scaled)[0]
        element_confidence = np.mean(self.element_abundance_model.predict(feature_vector_scaled))
        element_importance = dict(zip(self.feature_names, self.element_abundance_model.feature_importances_))
        
        predictions['element_abundance'] = MLPrediction(
            prediction_type='element_abundance',
            predicted_value=element_pred,
            confidence=min(100.0, abs(element_confidence)),
            factors_influence=dict(sorted(element_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        )
        
        return predictions
    
    def save_model_suite(self, output_path: str) -> str:
        """Save trained models to disk"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save individual models
        model_files = {
            'earth_similarity_model.joblib': self.earth_similarity_model,
            'discovery_potential_model.joblib': self.discovery_potential_model,
            'compound_formation_model.joblib': self.compound_formation_model,
            'element_abundance_model.joblib': self.element_abundance_model,
            'feature_scaler.joblib': self.feature_scaler,
        }
        
        for filename, model in model_files.items():
            joblib.dump(model, output_path / filename)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'model_type': 'NASA_Rover_ML_Suite_v1.0'
        }
        
        with open(output_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model suite saved to: {output_path}")
        return str(output_path)
    
    def load_model_suite(self, model_path: str):
        """Load trained models from disk"""
        
        model_path = Path(model_path)
        
        # Load models
        self.earth_similarity_model = joblib.load(model_path / 'earth_similarity_model.joblib')
        self.discovery_potential_model = joblib.load(model_path / 'discovery_potential_model.joblib')
        self.compound_formation_model = joblib.load(model_path / 'compound_formation_model.joblib')
        self.element_abundance_model = joblib.load(model_path / 'element_abundance_model.joblib')
        self.feature_scaler = joblib.load(model_path / 'feature_scaler.joblib')
        
        # Load metadata
        with open(model_path / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.is_trained = True
        
        print(f"✅ Model suite loaded from: {model_path}")
    
    def print_model_performance(self, performance: Dict[str, float]):
        """Print beautiful model performance report"""
        
        print(f"\n🧠 ML MODEL PERFORMANCE REPORT")
        print("=" * 50)
        
        print(f"🌍 Earth Similarity Model:")
        print(f"   R² Score: {performance.get('earth_similarity_r2', 0):.3f}")
        print(f"   MSE: {performance.get('earth_similarity_mse', 0):.3f}")
        
        print(f"\n🔍 Discovery Potential Model:")
        print(f"   R² Score: {performance.get('discovery_potential_r2', 0):.3f}")
        print(f"   MSE: {performance.get('discovery_potential_mse', 0):.3f}")
        
        print(f"\n⚗️ Compound Formation Model:")
        print(f"   Accuracy: {performance.get('compound_formation_accuracy', 0):.1%}")
        
        print(f"\n🧪 Element Abundance Model:")
        print(f"   R² Score: {performance.get('element_abundance_r2', 0):.3f}")
        print(f"   MSE: {performance.get('element_abundance_mse', 0):.3f}")
        
        print("=" * 50)
    
    def print_prediction_report(self, predictions: Dict[str, MLPrediction], location_name: str = "Unknown"):
        """Print beautiful prediction report"""
        
        print(f"\n🔮 ML PREDICTIONS FOR: {location_name}")
        print("=" * 50)
        
        for pred_type, prediction in predictions.items():
            print(f"\n{self._get_prediction_emoji(pred_type)} {pred_type.replace('_', ' ').title()}:")
            print(f"   Predicted Value: {prediction.predicted_value:.1f}%")
            print(f"   Confidence: {prediction.confidence:.1f}%")
            print(f"   Key Factors:")
            
            for factor, importance in list(prediction.factors_influence.items())[:3]:
                print(f"      • {factor}: {importance:.3f}")
        
        print("=" * 50)
    
    def _get_prediction_emoji(self, pred_type: str) -> str:
        """Get emoji for prediction type"""
        emojis = {
            'earth_similarity': '🌍',
            'discovery_potential': '🔍', 
            'compound_formation': '⚗️',
            'element_abundance': '🧪'
        }
        return emojis.get(pred_type, '📊')


def main():
    """Demo the ML training system"""
    
    print("🧠 NASA ROVER ML TRAINING SYSTEM")
    print("=" * 50)
    
    # This would typically be called after enhanced drilling analysis
    print("Note: This demo requires drilling data from enhanced_drilling_analyzer.py")
    print("Run complete enhanced mission to see ML training in action!")


if __name__ == "__main__":
    main()