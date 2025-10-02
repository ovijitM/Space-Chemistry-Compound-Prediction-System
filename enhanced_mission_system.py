#!/usr/bin/env python3
"""
Complete Enhanced NASA Rover Mission System
==========================================

FULL IMPLEMENTATION with enhanced analysis, ML training, and advanced visualizations.

This is the ultimate NASA rover system that includes:
1.        # Final Mission Report
        print(f"\n🏁 UNIVERSAL MISSION COMPLETE")
        print("=" * 60)
        self._print_final_mission_report("Unknown Environment", mission_summary)Enhanced drilling analysis with detailed Earth validation
2. 🧠 ML training for prediction of future discoveries
3                                                       print(f"   {status} {element}: {data['avg_ab        if hasattr(self, 'final_discovery_analysis'):
            analysis = self.final_discovery_analysis
            categories = analysis['categories']
            
            # Abundant element recommendations.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Promising Compoundsus = "❓ Unknown" if not data['earth_exists'] else "🌍 Known"
                ratio_text = f" ({data['earth_ratio']:.1f}x Earth)" if data['earth_ratio'] > 0 else ""
                print(f"   {status} {element}: {data['avg_abundance']:.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Promising Compounds    print(f"   {status} {element}: {data['avg_ab        if hasattr(self, 'final_discovery_analysis'):
            analysis = self.final_discovery_analysis
            categories = analysis['categories']
            
            # Abundant element recommendations.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Promising Compoundso_text = f" ({data['earth_ratio']:.1f}x         recommendations = []
        
        if hasattr(self, 'final_discovery_analysis'):
            analysis = self.final_discovery_analysis
            categories = analysis['categories']
            
            # Abundant element recommendationsata['earth_ratio'] > 0 else ""
                print(f"   {status} {element}: {data['avg_abundance']:.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Promising Compounds print(f"   {status} {element}: {data['avg_abundance']:.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Promising Compounds     print(f"   {status} {element}: {data['avg_ab            analysis = self.final_discovery_analysis
            categories = analysis['categories']
            
            # Abundant element recommendations:.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Promising Compounds       # Promising Compounds# Promising Compounds# Promising Compoundsnown Elemen            if categories['unknown_elements']:
                recommendations.append(f"🔬 PRIORITY: Further analysis of {len(categories['unknown_elements'])} unknown elements found") (Most Important!)
        if categories['unknown_elements']:
            print(f"\\n⚡ UNKNOWN ELEMENTS (Space-Only Discoveries):")
            for element, data in categories['unknown_elements'].items():
                abundance_ppm = data.get('earth_abundance_ppm', 'Unknown')
                print(f"   ⚡ {element}: {data['avg_abundance']:.1f}% avg abundance, found in {data['locations_found']}/{len(self.detailed_reports)} locations")Unknown Elements (Most Important!)
        if categories['unknown_elements']:
            print(f"\n⚡ UNKNOWN ELEMENTS (Space-Only Discoveries):")
            for element, data in categories['unknown_elements'].items(): Unknown Elements (Most Important!)
        if categorie            if categories['unknown_elements']:
                recommendations.append(f"🔬 PRIORITY: Further analysis of {len(categories['unknown_elements'])} unknown elements found")'unknown_elements']:
            print(f"\n❓ UNKNOWN ELEMENTS (Space-Only Discoveries):")
            for element, data in categories['unknown_elements'].items():
                abundance_ppm = data.get('earth_abundance_ppm', 'Unknown')
                print(f"   ❓ {element}: {data['avg_abundance']:.1f}% avg abundance, found in {data['locations_found']}/{len(self.detailed_reports)} locations")dvanced visualizations and interactive dashboards
4.             if categories['unknown_elements']:
                recommendations.append(f"🚀 PRIORITY: Further analysis of {len(categories['unknown_elements'])} unknown elements found")Comprehensive reporting an        print(f"   🔬 Unknown Elements: {stats['unknown_element_count']}") data analysis
"""

import argparse
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Core systems
from src.core.multi_location_survey import MultiLocationDrillingSurvey
from src.core.enhanced_drilling_analyzer import EnhancedDrillingAnalyzer
from src.core.ml_trainer import NASARoverMLTrainer
from src.core.simple_visualizer import SimpleNASARoverVisualizer
class CompleteEnhancedMissionSystem:
    """Ultimate NASA rover mission system with full capabilities"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize all subsystems
        self.drilling_survey = MultiLocationDrillingSurvey(random_seed=random_seed)
        self.enhanced_analyzer = EnhancedDrillingAnalyzer(random_seed=random_seed)
        self.ml_trainer = NASARoverMLTrainer(random_seed=random_seed)
        self.visualizer = SimpleNASARoverVisualizer()
        
        # Mission data
        self.mission_timestamp = datetime.now()
        self.location_data = []
        self.detailed_reports = []
        self.ml_predictions = {}
        self.visualization_paths = []
    
    def conduct_enhanced_mission(self, 
                                num_drilling_sites: int = 7,
                                compounds_per_site: int = 8,
                                output_dir: str = "enhanced_mission_results") -> Dict:
        """Conduct complete enhanced mission with all features"""
        
        print(f"🚀 UNIVERSAL NASA ROVER MISSION")
        print(f"=" * 60)
        print(f"🌌 Target: Unknown Environment (Universal)")
        print(f"🗿 Drilling Sites: {num_drilling_sites}")
        print(f"⚗️ Compounds per Site: {compounds_per_site}")
        print(f"💾 Output Directory: {output_dir}")
        print(f"🕒 Mission Start: {self.mission_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Phase 1: Unified Drilling and Analysis
        print(f"\\n🔬 PHASE 1: UNIFIED DRILLING & ANALYSIS")
        print("-" * 40)
        
        # Conduct unified drilling with immediate detailed analysis
        location_data, self.detailed_reports = self.drilling_survey.conduct_unified_survey(
            num_drilling_sites=num_drilling_sites,
            compounds_per_site=compounds_per_site,
            enhanced_analyzer=self.enhanced_analyzer
        )
        self.location_data = location_data
        
        # Phase 2: ML Training
        print(f"\\n🧠 PHASE 2: MACHINE LEARNING TRAINING")
        print("-" * 40)
        
        # Prepare training data
        training_df, targets = self.ml_trainer.prepare_training_data(self.detailed_reports)
        
        # Train models
        performance = self.ml_trainer.train_models(training_df, targets)
        self.ml_trainer.print_model_performance(performance)
        
        # Save trained models
        model_path = self.ml_trainer.save_model_suite(str(output_path / "trained_models"))
        
        # Phase 3: Prediction Demonstration
        print(f"\\n🔮 PHASE 3: ML PREDICTION DEMONSTRATION")
        print("-" * 40)
        
        # Create hypothetical new location for prediction
        hypothetical_location = {
            'temperature': random.uniform(-100, 50),
            'pressure': random.uniform(0.1, 2.0),
            'pH': random.uniform(6.0, 9.0),
            'humidity': random.uniform(0.0, 80.0),
            'radiation_level': random.uniform(0.1, 10.0),
            'depth': random.uniform(0.5, 3.0),
            'site_type_encoded': 5.0,  # Ridge
            'latitude': random.uniform(-5, 5),
            'longitude': random.uniform(-5, 5),
        }
        
        # Add element features (simplified)
        for element in ['O', 'Si', 'Al', 'Fe', 'Ca', 'Na', 'K', 'Mg', 'Ti', 'H']:
            hypothetical_location[f'element_{element}_percentage'] = random.uniform(0, 20)
            hypothetical_location[f'element_{element}_present'] = random.choice([0.0, 1.0])
        
        # Additional derived features
        hypothetical_location.update({
            'total_elements': random.randint(5, 15),
            'element_diversity': random.randint(3, 10),
            'rare_element_count': random.randint(0, 3),
            'unknown_element_count': random.randint(0, 2),
        })
        
        # Make predictions
        predictions = self.ml_trainer.predict_location_potential(hypothetical_location)
        self.ml_predictions = predictions
        
        self.ml_trainer.print_prediction_report(predictions, "Hypothetical Future Site")
        
        # Phase 4: Simple Terminal Visualizations
        print(f"\\n🎨 PHASE 4: TERMINAL VISUALIZATIONS")
        print("-" * 40)
        
        viz_output_path = output_path / "visualizations"
        viz_output_path.mkdir(exist_ok=True)
        
        # Generate simple terminal-friendly visualizations
        print("📊 Generating terminal visualizations...")
        
        # Element abundance heatmap
        self.visualizer.create_element_abundance_heatmap(
            self.detailed_reports, 
            str(viz_output_path / "element_abundance_heatmap.png")
        )
        
        # Compound formation charts
        self.visualizer.create_compound_formation_charts(
            self.detailed_reports,
            str(viz_output_path / "compound_formation_charts.png")
        )
        
        # Discovery summary
        self.visualizer.create_discovery_summary_chart(
            self.detailed_reports,
            str(viz_output_path / "discovery_summary.png")
        )
        
        # Element correlation heatmap
        self.visualizer.create_element_correlation_heatmap(
            self.detailed_reports,
            str(viz_output_path / "element_correlation.png")
        )
        
        # ML predictions
        self.visualizer.create_ml_predictions_simple_chart(
            predictions, "Future Site",
            str(viz_output_path / "ml_predictions.png")
        )
        
        # Phase 5: Final Discovery Pool Analysis
        print(f"\\n🏆 PHASE 5: FINAL DISCOVERY POOL ANALYSIS")
        print("-" * 40)
        
        # Generate comprehensive discovery analysis
        final_analysis = self._generate_final_discovery_pool()
        self._print_final_discovery_report(final_analysis)
        
        # Create final discovery visualizations
        self._create_final_discovery_visualizations(final_analysis, viz_output_path)
        
        # Phase 6: Data Export & Reporting
        print(f"\\n💾 PHASE 5: DATA EXPORT & REPORTING")
        print("-" * 40)
        
        # Export detailed reports as JSON
        detailed_reports_data = []
        for report in self.detailed_reports:
            report_dict = {
                'site_id': report.site_id,
                'coordinates': report.coordinates,
                'depth': report.depth,
                'site_type': report.site_type,
                'drilling_timestamp': report.drilling_timestamp.isoformat(),
                'environmental_data': {
                    'temperature': report.temperature,
                    'pressure': report.pressure,
                    'pH': report.pH,
                    'humidity': report.humidity,
                    'radiation_level': report.radiation_level
                },
                'element_analysis': [{
                    'symbol': ea.symbol,
                    'percentage': ea.percentage,
                    'earth_exists': ea.earth_exists,
                    'rarity_classification': ea.rarity_classification
                } for ea in report.element_analyses],
                'compound_analysis': [{
                    'formula': ca.formula,
                    'formation_probability': ca.formation_probability,
                    'earth_exists': ca.earth_exists,
                    'classification': ca.classification,
                    'stability_rating': ca.stability_rating
                } for ca in report.compound_analyses],
                'summary_scores': {
                    'earth_similarity_score': report.earth_similarity_score,
                    'discovery_potential': report.discovery_potential,
                    'scientific_value': report.scientific_value
                }
            }
            detailed_reports_data.append(report_dict)
        
        # Save detailed reports
        with open(output_path / "enhanced_detailed_reports.json", 'w') as f:
            json.dump(detailed_reports_data, f, indent=2)
        
        # Save ML predictions
        predictions_data = {}
        for pred_type, prediction in predictions.items():
            predictions_data[pred_type] = {
                'predicted_value': prediction.predicted_value,
                'confidence': prediction.confidence,
                'top_factors': dict(list(prediction.factors_influence.items())[:3])
            }
        
        with open(output_path / "ml_predictions.json", 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        # Create comprehensive CSV datasets
        self._create_comprehensive_datasets(output_path)
        
        # Mission Summary Report
        mission_summary = self._generate_mission_summary("Unknown Environment", performance)
        
        with open(output_path / "mission_summary.json", 'w') as f:
            json.dump(mission_summary, f, indent=2)
        
        # Final Mission Report
        print(f"\\n🏁 UNIVERSAL MISSION COMPLETE")
        print("=" * 60)
        self._print_final_mission_report("Unknown Environment", mission_summary)
        
        print(f"\\n📁 ALL RESULTS SAVED TO: {output_path}")
        print(f"🌐 VIEW VISUALIZATIONS: {viz_output_path / 'index.html'}")
        print("🎯 Ready for advanced analysis and future missions!")
        
        return mission_summary
    
    def _create_comprehensive_datasets(self, output_path: Path):
        """Create comprehensive CSV datasets for analysis"""
        
        print("📊 Creating comprehensive datasets...")
        
        # Enhanced location dataset
        location_rows = []
        for report in self.detailed_reports:
            row = {
                'site_id': report.site_id,
                'latitude': report.coordinates[0],
                'longitude': report.coordinates[1],
                'depth': report.depth,
                'site_type': report.site_type,
                'temperature': report.temperature,
                'pressure': report.pressure,
                'pH': report.pH,
                'humidity': report.humidity,
                'radiation_level': report.radiation_level,
                'total_elements_found': report.total_elements_found,
                'total_compounds_tested': report.total_compounds_tested,
                'known_compounds_count': len(report.known_compounds),
                'unknown_compounds_count': len(report.unknown_compounds),
                'earth_similarity_score': report.earth_similarity_score,
                'discovery_potential': report.discovery_potential,
                'scientific_value': report.scientific_value,
                'dominant_element_1': report.dominant_elements[0] if report.dominant_elements else '',
                'dominant_element_2': report.dominant_elements[1] if len(report.dominant_elements) > 1 else '',
                'dominant_element_3': report.dominant_elements[2] if len(report.dominant_elements) > 2 else '',
            }
            location_rows.append(row)
        
        location_df = pd.DataFrame(location_rows)
        location_df.to_csv(output_path / "enhanced_location_dataset.csv", index=False)
        
        # Enhanced element dataset
        element_rows = []
        for report in self.detailed_reports:
            for ea in report.element_analyses:
                row = {
                    'site_id': report.site_id,
                    'element_symbol': ea.symbol,
                    'percentage': ea.percentage,
                    'quantity': ea.quantity,
                    'earth_exists': ea.earth_exists,
                    'earth_abundance_ppm': ea.earth_abundance_ppm,
                    'space_vs_earth_ratio': ea.space_vs_earth_ratio,
                    'rarity_classification': ea.rarity_classification,
                    'site_temperature': report.temperature,
                    'site_pH': report.pH,
                    'site_depth': report.depth
                }
                element_rows.append(row)
        
        element_df = pd.DataFrame(element_rows)
        element_df.to_csv(output_path / "enhanced_element_dataset.csv", index=False)
        
        # Enhanced compound dataset
        compound_rows = []
        for report in self.detailed_reports:
            for ca in report.compound_analyses:
                row = {
                    'site_id': report.site_id,
                    'formula': ca.formula,
                    'name': ca.name,
                    'formation_probability': ca.formation_probability,
                    'earth_exists': ca.earth_exists,
                    'databases_found_count': len(ca.earth_databases_found),
                    'databases_found': ','.join(ca.earth_databases_found),
                    'validation_confidence': ca.validation_confidence,
                    'stability_rating': ca.stability_rating,
                    'classification': ca.classification,
                    'site_temperature': report.temperature,
                    'site_pH': report.pH,
                    'site_discovery_potential': report.discovery_potential
                }
                compound_rows.append(row)
        
        compound_df = pd.DataFrame(compound_rows)
        compound_df.to_csv(output_path / "enhanced_compound_dataset.csv", index=False)
        
        print(f"   ✅ Enhanced location dataset: {len(location_df)} rows")
        print(f"   ✅ Enhanced element dataset: {len(element_df)} rows")
        print(f"   ✅ Enhanced compound dataset: {len(compound_df)} rows")
    
    def _generate_mission_summary(self, env_name: str, ml_performance: Dict) -> Dict:
        """Generate comprehensive mission summary"""
        
        # Calculate summary statistics
        total_elements = sum(r.total_elements_found for r in self.detailed_reports)
        total_compounds = sum(r.total_compounds_tested for r in self.detailed_reports)
        total_known_compounds = sum(len(r.known_compounds) for r in self.detailed_reports)
        total_unknown_compounds = sum(len(r.unknown_compounds) for r in self.detailed_reports)
        
        avg_earth_similarity = np.mean([r.earth_similarity_score for r in self.detailed_reports])
        avg_discovery_potential = np.mean([r.discovery_potential for r in self.detailed_reports])
        avg_scientific_value = np.mean([r.scientific_value for r in self.detailed_reports])
        
        # Find best and worst locations
        best_location = max(self.detailed_reports, key=lambda r: r.scientific_value)
        most_earth_like = max(self.detailed_reports, key=lambda r: r.earth_similarity_score)
        highest_discovery = max(self.detailed_reports, key=lambda r: r.discovery_potential)
        
        # Element discoveries
        all_elements = set()
        unknown_elements = set()
        for report in self.detailed_reports:
            for ea in report.element_analyses:
                all_elements.add(ea.symbol)
                if not ea.earth_exists:
                    unknown_elements.add(ea.symbol)
        
        return {
            'mission_metadata': {
                'environment': env_name,
                'total_locations': len(self.detailed_reports),
                'mission_timestamp': self.mission_timestamp.isoformat(),
                'random_seed': self.random_seed
            },
            'discovery_statistics': {
                'total_unique_elements': len(all_elements),
                'unknown_elements': list(unknown_elements),
                'unknown_element_count': len(unknown_elements),
                'total_compounds_tested': total_compounds,
                'known_compounds_found': total_known_compounds,
                'unknown_compounds_found': total_unknown_compounds,
                'compound_success_rate': (total_known_compounds / total_compounds) * 100 if total_compounds > 0 else 0
            },
            'analysis_scores': {
                'average_earth_similarity': avg_earth_similarity,
                'average_discovery_potential': avg_discovery_potential,
                'average_scientific_value': avg_scientific_value
            },
            'notable_locations': {
                'highest_scientific_value': {
                    'site_id': best_location.site_id,
                    'score': best_location.scientific_value,
                    'coordinates': best_location.coordinates
                },
                'most_earth_like': {
                    'site_id': most_earth_like.site_id,
                    'score': most_earth_like.earth_similarity_score,
                    'coordinates': most_earth_like.coordinates
                },
                'highest_discovery_potential': {
                    'site_id': highest_discovery.site_id,
                    'score': highest_discovery.discovery_potential,
                    'coordinates': highest_discovery.coordinates
                }
            },
            'ml_model_performance': ml_performance,
            'top_elements_discovered': list(all_elements)[:10],
            'mission_recommendations': self._generate_mission_recommendations()
        }
    
    def _generate_mission_recommendations(self) -> List[str]:
        """Generate mission recommendations based on findings"""
        
        recommendations = []
        
        # Analyze discovery patterns
        avg_discovery = np.mean([r.discovery_potential for r in self.detailed_reports])
        avg_earth_similarity = np.mean([r.earth_similarity_score for r in self.detailed_reports])
        
        if avg_discovery > 70:
            recommendations.append("High discovery potential detected. Recommend extending mission duration for detailed compound analysis.")
        
        if avg_earth_similarity < 30:
            recommendations.append("Low Earth similarity suggests unique environment. Priority for sample return mission.")
        
        # Element analysis
        all_unknown_elements = set()
        for report in self.detailed_reports:
            for ea in report.element_analyses:
                if not ea.earth_exists:
                    all_unknown_elements.add(ea.symbol)
        
        if len(all_unknown_elements) > 2:
            recommendations.append(f"Multiple unknown elements detected: {', '.join(list(all_unknown_elements)[:3])}. Recommend spectrographic verification.")
        
        # Compound analysis
        total_unknown_compounds = sum(len(r.unknown_compounds) for r in self.detailed_reports)
        if total_unknown_compounds > 5:
            recommendations.append("Significant number of unknown compounds found. Consider in-situ synthesis experiments.")
        
        # Environmental correlations
        temp_range = max(r.temperature for r in self.detailed_reports) - min(r.temperature for r in self.detailed_reports)
        if temp_range > 100:
            recommendations.append("Large temperature variation detected. Temperature may be key factor in compound formation.")
        
        if not recommendations:
            recommendations.append("Standard mission parameters observed. Continue with regular exploration protocol.")
        
        return recommendations
    
    def _generate_final_discovery_pool(self) -> Dict:
        """Generate comprehensive final discovery pool analysis"""
        
        # Collect all elements with their abundances across locations
        element_pool = {}
        compound_pool = {}
        
        for report in self.detailed_reports:
            # Element analysis
            for ea in report.element_analyses:
                if ea.symbol not in element_pool:
                    element_pool[ea.symbol] = {
                        'total_abundance': 0.0,
                        'avg_abundance': 0.0,
                        'max_abundance': 0.0,
                        'locations_found': 0,
                        'earth_exists': ea.earth_exists,
                        'rarity_class': ea.rarity_classification,
                        'earth_ratio': ea.space_vs_earth_ratio if ea.space_vs_earth_ratio else 0.0
                    }
                
                element_pool[ea.symbol]['total_abundance'] += ea.percentage
                element_pool[ea.symbol]['max_abundance'] = max(element_pool[ea.symbol]['max_abundance'], ea.percentage)
                element_pool[ea.symbol]['locations_found'] += 1
            
            # Compound analysis
            for ca in report.compound_analyses:
                if ca.formula not in compound_pool:
                    compound_pool[ca.formula] = {
                        'total_probability': 0.0,
                        'avg_probability': 0.0,
                        'max_probability': 0.0,
                        'locations_found': 0,
                        'earth_exists': ca.earth_exists,
                        'classification': ca.classification,
                        'stability': ca.stability_rating,
                        'databases_found': ca.earth_databases_found
                    }
                
                compound_pool[ca.formula]['total_probability'] += ca.formation_probability
                compound_pool[ca.formula]['max_probability'] = max(compound_pool[ca.formula]['max_probability'], ca.formation_probability)
                compound_pool[ca.formula]['locations_found'] += 1
        
        # Calculate averages
        for element, data in element_pool.items():
            data['avg_abundance'] = data['total_abundance'] / len(self.detailed_reports)
            data['frequency'] = (data['locations_found'] / len(self.detailed_reports)) * 100
        
        for compound, data in compound_pool.items():
            data['avg_probability'] = data['total_probability'] / len(self.detailed_reports)
            data['frequency'] = (data['locations_found'] / len(self.detailed_reports)) * 100
        
        # Sort by abundance/probability
        sorted_elements = dict(sorted(element_pool.items(), key=lambda x: x[1]['avg_abundance'], reverse=True))
        sorted_compounds = dict(sorted(compound_pool.items(), key=lambda x: x[1]['avg_probability'], reverse=True))
        
        # Categorize discoveries
        abundant_elements = {k: v for k, v in sorted_elements.items() if v['avg_abundance'] >= 10.0}
        common_elements = {k: v for k, v in sorted_elements.items() if 1.0 <= v['avg_abundance'] < 10.0}
        rare_elements = {k: v for k, v in sorted_elements.items() if v['avg_abundance'] < 1.0}
        unknown_elements = {k: v for k, v in sorted_elements.items() if not v['earth_exists']}
        
        promising_compounds = {k: v for k, v in sorted_compounds.items() if v['avg_probability'] >= 0.5}
        unknown_compounds = {k: v for k, v in sorted_compounds.items() if not v['earth_exists']}
        
        return {
            'element_pool': sorted_elements,
            'compound_pool': sorted_compounds,
            'categories': {
                'abundant_elements': abundant_elements,
                'common_elements': common_elements,
                'rare_elements': rare_elements,
                'unknown_elements': unknown_elements,
                'promising_compounds': promising_compounds,
                'unknown_compounds': unknown_compounds
            },
            'statistics': {
                'total_unique_elements': len(element_pool),
                'total_unique_compounds': len(compound_pool),
                'unknown_element_count': len(unknown_elements),
                'unknown_compound_count': len(unknown_compounds),
                'abundant_element_count': len(abundant_elements),
                'promising_compound_count': len(promising_compounds)
            }
        }
    
    def _print_final_discovery_report(self, analysis: Dict):
        """Print comprehensive final discovery report"""
        
        print("🏆=" * 35)
        print("🌌 FINAL DISCOVERY POOL ANALYSIS")
        print("🏆=" * 35)
        
        stats = analysis['statistics']
        categories = analysis['categories']
        
        # Overview statistics
        print(f"\\n📊 DISCOVERY OVERVIEW:")
        print(f"   🧪 Total Unique Elements: {stats['total_unique_elements']}")
        print(f"   ⚗️ Total Unique Compounds: {stats['total_unique_compounds']}")
        print(f"   🔬 Unknown Elements: {stats['unknown_element_count']}")
        print(f"   🔬 Unknown Compounds: {stats['unknown_compound_count']}")
        print(f"   💎 Abundant Elements (≥10%): {stats['abundant_element_count']}")
        print(f"   🎯 Promising Compounds (≥0.5%): {stats['promising_compound_count']}")
        
        # Abundant Elements
        if categories['abundant_elements']:
            print(f"\\n💎 ABUNDANT ELEMENTS (Major Discoveries):")
            for element, data in list(categories['abundant_elements'].items())[:5]:
                status = "❓ Unknown" if not data['earth_exists'] else "🌍 Known"
                ratio_text = f" ({data['earth_ratio']:.1f}x Earth)" if data['earth_ratio'] > 0 else ""
                print(f"   {status} {element}: {data['avg_abundance']:.1f}% avg, {data['frequency']:.0f}% frequency{ratio_text}")
        
        # Low Abundance Elements (Most Important!)
        # if categories['low_abundance_elements']:
        #     print(f"\\n🔬 LOW ABUNDANCE ELEMENTS (Rare Earth Discoveries):")
        #     for element, data in categories['low_abundance_elements'].items():
        #         abundance_ppm = data.get('earth_abundance_ppm', 'Unknown')
        #         print(f"   � {element}: {data['avg_abundance']:.1f}% avg abundance, found in {data['locations_found']}/{len(self.detailed_reports)} locations (Earth: {abundance_ppm} ppm)")
        
        # Promising Compounds
        if categories['promising_compounds']:
            print(f"\\n🎯 MOST PROMISING COMPOUNDS:")
            for compound, data in list(categories['promising_compounds'].items())[:5]:
                status = "❓ Unknown" if not data['earth_exists'] else "🌍 Known"
                db_info = f" [{', '.join(data['databases_found'][:2])}]" if data['databases_found'] else ""
                print(f"   {status} {compound}: {data['avg_probability']:.2f}% formation chance{db_info}")
        
        # Unknown Compounds
        if categories['unknown_compounds']:
            print(f"\\n🔬 UNKNOWN COMPOUNDS (Theoretical Formations):")
            for compound, data in categories['unknown_compounds'].items():
                print(f"   🧬 {compound}: {data['avg_probability']:.2f}% formation probability")
        
        # Element Pool Summary
        print(f"\\n🧪 COMPLETE ELEMENT POOL (Top 10 by abundance):")
        for element, data in list(analysis['element_pool'].items())[:10]:
            status = "❓" if not data['earth_exists'] else "🌍"
            rarity = data['rarity_class'][:4].upper()  # ABUN, COMM, RARE
            print(f"   {status} {element}: {data['avg_abundance']:.1f}% ({rarity}) - {data['frequency']:.0f}% locations")
        
        print("🏆=" * 35)
    
    def _create_final_discovery_visualizations(self, analysis: Dict, viz_output_path: Path):
        """Create comprehensive final discovery visualizations"""
        
        print("\\n🎨 Creating Final Discovery Pool Visualizations...")
        
        # Create master discovery dashboard
        self.visualizer.create_master_discovery_dashboard(
            analysis, str(viz_output_path / "master_discovery_dashboard.png")
        )
        
        # Create element pool visualization
        self.visualizer.create_element_pool_analysis(
            analysis['element_pool'], str(viz_output_path / "element_pool_analysis.png")
        )
        
        # Create compound pool visualization  
        self.visualizer.create_compound_pool_analysis(
            analysis['compound_pool'], str(viz_output_path / "compound_pool_analysis.png")
        )
        
        print("   ✅ Final discovery visualizations created!")
    
    def _generate_mission_recommendations(self) -> List[str]:
        """Generate mission recommendations based on findings"""
        recommendations = []
        
        if hasattr(self, 'final_discovery_analysis'):
            analysis = self.final_discovery_analysis
            categories = analysis['categories']
            
            # Unknown element recommendations
            # if categories['low_abundance_elements']:
            #     recommendations.append(f"� PRIORITY: Further analysis of {len(categories['low_abundance_elements'])} low-abundance elements found")
            #     recommendations.append("   Consider extended spectroscopy and isotope analysis")
            
            # Abundant element recommendations  
            if categories['abundant_elements']:
                recommendations.append(f"💎 RESOURCE: {len(categories['abundant_elements'])} abundant elements suitable for resource extraction")
            
            # Unknown compound recommendations
            if categories['unknown_compounds']:
                recommendations.append(f"🧬 RESEARCH: {len(categories['unknown_compounds'])} unknown compounds require laboratory synthesis attempts")
            
            # Promising compound recommendations
            if categories['promising_compounds']:
                recommendations.append(f"🎯 COMPOUNDS: {len(categories['promising_compounds'])} high-probability formations detected")
                
        return recommendations
    
    def _print_final_mission_report(self, env_name: str, mission_summary: Dict):
        """Print beautiful final mission report"""
        
        print(f"🌌 Environment: {env_name}")
        print(f"📍 Locations Surveyed: {mission_summary['mission_metadata']['total_locations']}")
        print(f"⏱️ Mission Duration: {datetime.now() - self.mission_timestamp}")
        
        print(f"\\n🔬 DISCOVERY SUMMARY:")
        ds = mission_summary['discovery_statistics']
        print(f"   🧪 Unique Elements: {ds['total_unique_elements']}")
        print(f"   🔬 Unknown Elements: {ds.get('unknown_element_count', 0)}")
        print(f"   ⚗️ Compounds Tested: {ds['total_compounds_tested']}")
        print(f"   🌍 Known Compounds: {ds['known_compounds_found']}")
        print(f"   🔬 Unknown Compounds: {ds['unknown_compounds_found']}")
        print(f"   📈 Success Rate: {ds['compound_success_rate']:.1f}%")
        
        print(f"\\n📊 ANALYSIS SCORES:")
        scores = mission_summary['analysis_scores']
        print(f"   🌍 Earth Similarity: {scores['average_earth_similarity']:.1f}%")
        print(f"   🔍 Discovery Potential: {scores['average_discovery_potential']:.1f}%")
        print(f"   🏆 Scientific Value: {scores['average_scientific_value']:.1f}%")
        
        print(f"\\n🎯 TOP DISCOVERIES:")
        if ds['unknown_elements']:
            print(f"   ❓ Unknown Elements: {', '.join(ds['unknown_elements'][:5])}")
        
        print(f"\\n💡 MISSION RECOMMENDATIONS:")
        for i, rec in enumerate(mission_summary['mission_recommendations'], 1):
            print(f"   {i}. {rec}")


def main():
    """Main entry point for the complete enhanced mission system"""
    
    parser = argparse.ArgumentParser(description="Complete Enhanced NASA Rover Mission System")
    parser.add_argument("--locations", type=int, default=6, help="Number of drilling locations")

    parser.add_argument("--compounds", type=int, default=8, help="Compounds to test per location")
    parser.add_argument("--output-dir", type=str, default="enhanced_mission_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    

    
    # Initialize and run complete mission
    mission_system = CompleteEnhancedMissionSystem(random_seed=args.seed)
    
    mission_summary = mission_system.conduct_enhanced_mission(
        num_drilling_sites=args.locations,
        compounds_per_site=args.compounds,
        output_dir=args.output_dir
    )
    
    print(f"\\n🎉 MISSION SUCCESS! All systems operational.")


if __name__ == "__main__":
    main()