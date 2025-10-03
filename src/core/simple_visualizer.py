#!/usr/bin/env python3
"""
NASA Rover Simple Terminal Visualizer
====================================

Creates simple, terminal-displayable visualizations using matplotlib
for drilling data, element distributions, and compound formations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend for terminal display
plt.switch_backend('Agg')

class SimpleNASARoverVisualizer:
    """Simple terminal-friendly visualization system"""
    
    def __init__(self):
        # Set up clean styling for terminal output
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Configure matplotlib for better terminal display
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
    def create_element_abundance_heatmap(self, detailed_reports: List, save_path: Optional[str] = None):
        """Create element abundance heatmap"""
        print("\n📊 Creating Element Abundance Heatmap...")
        
        # Collect element data
        element_data = []
        locations = []
        
        for report in detailed_reports:
            locations.append(report.site_id)
            location_elements = {}
            
            for element_analysis in report.element_analyses:
                location_elements[element_analysis.symbol] = element_analysis.percentage
            
            element_data.append(location_elements)
        
        # Create DataFrame
        df = pd.DataFrame(element_data, index=locations)
        df = df.fillna(0)  # Fill missing elements with 0
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='viridis', 
                   cbar_kws={'label': 'Abundance %'})
        plt.title('🧪 Element Abundance Across Drilling Locations', fontsize=16, pad=20)
        plt.xlabel('Chemical Elements', fontsize=12)
        plt.ylabel('Drilling Locations', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
        plt.close()
        
    def create_compound_formation_charts(self, detailed_reports: List, save_path: Optional[str] = None):
        """Create compound formation analysis charts"""
        print("\n⚗️ Creating Compound Formation Charts...")
        
        # Collect compound data
        compound_data = []
        for report in detailed_reports:
            for compound_analysis in report.compound_analyses:
                compound_data.append({
                    'Location': report.site_id,
                    'Compound': compound_analysis.formula,
                    'Formation_Probability': compound_analysis.formation_probability,
                    'Earth_Known': compound_analysis.earth_exists,
                    'Stability': 'High' if compound_analysis.formation_probability > 0.6 else 
                               'Medium' if compound_analysis.formation_probability > 0.3 else 'Low'
                })
        
        df = pd.DataFrame(compound_data)
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('⚗️ Compound Formation Analysis Dashboard', fontsize=18, y=0.98)
        
        # 1. Formation probability by location (box plot)
        sns.boxplot(data=df, x='Location', y='Formation_Probability', ax=ax1)
        ax1.set_title('Formation Probability Distribution by Location')
        ax1.set_ylabel('Formation Probability')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Earth known vs unknown (bar chart)
        earth_counts = df['Earth_Known'].value_counts()
        ax2.bar(['Known on Earth', 'Unknown'], [earth_counts.get(True, 0), earth_counts.get(False, 0)], 
                color=['#1f77b4', '#ff7f0e'])
        ax2.set_title('Earth Database Matches')
        ax2.set_ylabel('Number of Compounds')
        
        # 3. Formation probability histogram
        ax3.hist(df['Formation_Probability'], bins=15, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('Formation Probability Distribution')
        ax3.set_xlabel('Formation Probability')
        ax3.set_ylabel('Frequency')
        
        # 4. Stability classification
        stability_counts = df['Stability'].value_counts()
        colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        bars = ax4.bar(stability_counts.index, stability_counts.values, 
                      color=[colors[x] for x in stability_counts.index])
        ax4.set_title('Compound Stability Classification')
        ax4.set_ylabel('Number of Compounds')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
            
        plt.show()
        plt.close()
        
    def create_discovery_summary_chart(self, detailed_reports: List, save_path: Optional[str] = None):
        """Create discovery summary visualization"""
        print("\n🔍 Creating Discovery Summary Chart...")
        
        # Collect summary data
        locations = []
        total_elements = []
        total_compounds = []
        earth_similarity = []
        discovery_potential = []
        
        for report in detailed_reports:
            locations.append(report.site_id)
            total_elements.append(len(report.element_analyses))
            total_compounds.append(report.total_compounds_tested)
            earth_similarity.append(report.earth_similarity_score)
            discovery_potential.append(report.discovery_potential)
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔍 Discovery Summary Dashboard', fontsize=18, y=0.98)
        
        # 1. Elements discovered per location
        bars1 = ax1.bar(locations, total_elements, color='skyblue', alpha=0.8)
        ax1.set_title('Elements Discovered per Location')
        ax1.set_ylabel('Number of Elements')
        ax1.tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar, value in zip(bars1, total_elements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
        
        # 2. Compounds tested per location
        bars2 = ax2.bar(locations, total_compounds, color='lightgreen', alpha=0.8)
        ax2.set_title('Compounds Tested per Location')
        ax2.set_ylabel('Number of Compounds')
        ax2.tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar, value in zip(bars2, total_compounds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
        
        # 3. Earth similarity scores
        bars3 = ax3.bar(locations, earth_similarity, color='orange', alpha=0.8)
        ax3.set_title('Earth Similarity Scores')
        ax3.set_ylabel('Similarity %')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar, value in zip(bars3, earth_similarity):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 4. Discovery potential scores
        bars4 = ax4.bar(locations, discovery_potential, color='purple', alpha=0.8)
        ax4.set_title('Discovery Potential Scores')
        ax4.set_ylabel('Discovery Potential %')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar, value in zip(bars4, discovery_potential):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
            
        plt.show()
        plt.close()
        
    def create_element_correlation_heatmap(self, detailed_reports: List, save_path: Optional[str] = None):
        """Create element correlation heatmap"""
        print("\n🔗 Creating Element Correlation Heatmap...")
        
        # Collect element abundance data
        element_data = []
        
        for report in detailed_reports:
            location_elements = {}
            for element_analysis in report.element_analyses:
                location_elements[element_analysis.symbol] = element_analysis.percentage
            element_data.append(location_elements)
        
        df = pd.DataFrame(element_data)
        df = df.fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Hide upper triangle
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('🔗 Element Abundance Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
            
        plt.show()
        plt.close()
        
    def create_ml_predictions_simple_chart(self, predictions: Dict, location_name: str, save_path: Optional[str] = None):
        """Create simple ML predictions chart"""
        print(f"\n🔮 Creating ML Predictions Chart for {location_name}...")
        
        # Extract prediction data
        pred_names = []
        pred_values = []
        pred_confidence = []
        
        for name, pred in predictions.items():
            pred_names.append(name.replace('_', ' ').title())
            pred_values.append(pred.predicted_value)
            pred_confidence.append(pred.confidence)
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'🔮 ML Predictions for {location_name}', fontsize=16, y=1.02)
        
        # 1. Prediction values
        bars1 = ax1.bar(pred_names, pred_values, color='steelblue', alpha=0.8)
        ax1.set_title('Predicted Values')
        ax1.set_ylabel('Predicted Value (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 100)
        # Add value labels
        for bar, value in zip(bars1, pred_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. Confidence levels
        bars2 = ax2.bar(pred_names, pred_confidence, color='darkorange', alpha=0.8)
        ax2.set_title('Prediction Confidence')
        ax2.set_ylabel('Confidence (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 100)
        # Add value labels
        for bar, value in zip(bars2, pred_confidence):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
            
        plt.show()
        plt.close()
        
    def create_comprehensive_summary(self, detailed_reports: List, save_path: Optional[str] = None):
        """Create comprehensive mission summary"""
        print("\n📊 Creating Comprehensive Mission Summary...")
        
        # Call all visualization methods
        self.create_element_abundance_heatmap(detailed_reports, 
                                            save_path.replace('.png', '_elements_heatmap.png') if save_path else None)
        
        self.create_compound_formation_charts(detailed_reports,
                                            save_path.replace('.png', '_compounds.png') if save_path else None)
        
        self.create_discovery_summary_chart(detailed_reports,
                                          save_path.replace('.png', '_discovery.png') if save_path else None)
        
        self.create_element_correlation_heatmap(detailed_reports,
                                              save_path.replace('.png', '_correlation.png') if save_path else None)
        
        print("\n✅ All visualizations completed and displayed!")
    
    def create_master_discovery_dashboard(self, analysis: Dict, save_path: str):
        """Create master discovery dashboard showing comprehensive analysis"""
        print("\n🎨 Creating Master Discovery Dashboard...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('🌌 MASTER DISCOVERY POOL DASHBOARD', fontsize=18, fontweight='bold')
        
        # 1. Element abundance pie chart
        element_pool = analysis['element_pool']
        top_elements = list(element_pool.items())[:8]
        
        if top_elements:
            labels = [f"{k} ({v['avg_abundance']:.1f}%)" for k, v in top_elements]
            sizes = [v['avg_abundance'] for k, v in top_elements]
            colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('🧪 Top Elements by Average Abundance', fontsize=14, fontweight='bold')
        
        # 2. Unknown vs Known elements bar chart
        categories = analysis['categories']
        cat_data = {
            'Abundant (≥10%)': len(categories['abundant_elements']),
            'Common (1-10%)': len(categories['common_elements']),
            'Rare (<1%)': len(categories['rare_elements']),
            'Low Abundance': len(categories.get('low_abundance_elements', {}))
        }
        
        bars = ax2.bar(cat_data.keys(), cat_data.values(), 
                      color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax2.set_title('🔍 Element Discovery Categories', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Elements')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Compound formation probability
        compound_pool = analysis['compound_pool']
        top_compounds = list(compound_pool.items())[:10]
        
        if top_compounds:
            comp_names = [k for k, v in top_compounds]
            comp_probs = [v['avg_probability'] for k, v in top_compounds]
            
            bars = ax3.barh(comp_names, comp_probs, color='skyblue')
            ax3.set_title('⚗️ Top Compound Formation Probabilities', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Formation Probability (%)')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{comp_probs[i]:.2f}%', ha='left', va='center')
        
        # 4. Discovery statistics
        stats = analysis['statistics']
        stat_labels = ['Total Elements', 'Total Compounds', 
                      'Low Abundance Elements', 'Unknown Compounds', 'Abundant Elements', 'Promising Compounds']
        stat_values = [stats['total_unique_elements'], stats['total_unique_compounds'],
                      stats.get('low_abundance_element_count', 0), stats['unknown_compound_count'],
                      stats['abundant_element_count'], stats['promising_compound_count']]
        
        bars = ax4.bar(range(len(stat_labels)), stat_values, 
                      color=['#ff7f7f', '#7f7fff', '#ff7fff', '#7fff7f', '#ffff7f', '#7fffff'])
        ax4.set_title('📊 Discovery Statistics Summary', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(stat_labels)))
        ax4.set_xticklabels(stat_labels, rotation=45, ha='right')
        ax4.set_ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Master dashboard saved to: {save_path}")
        plt.show()
        plt.close()
    
    def create_element_pool_analysis(self, element_pool: Dict, save_path: str):
        """Create detailed element pool analysis visualization"""
        print("\n🧪 Creating Element Pool Analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('🧪 COMPLETE ELEMENT POOL ANALYSIS', fontsize=16, fontweight='bold')
        
        # Convert to lists for plotting
        elements = list(element_pool.keys())[:15]  # Top 15
        abundances = [element_pool[e]['avg_abundance'] for e in elements]
        frequencies = [element_pool[e]['frequency'] for e in elements]
        earth_ratios = [element_pool[e]['earth_ratio'] for e in elements]
        
        # 1. Element abundance distribution
        bars = ax1.bar(elements, abundances, color='lightblue', alpha=0.7)
        ax1.set_title('Average Element Abundance Distribution', fontweight='bold')
        ax1.set_ylabel('Average Abundance (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color unknown elements differently
        for i, element in enumerate(elements):
            if not element_pool[element]['earth_exists']:
                bars[i].set_color('red')
                bars[i].set_alpha(0.8)
        
        # 2. Element frequency across locations
        ax2.bar(elements, frequencies, color='lightgreen', alpha=0.7)
        ax2.set_title('Element Frequency Across Locations', fontweight='bold')
        ax2.set_ylabel('Frequency (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Earth ratio comparison (where applicable)
        valid_ratios = [(e, r) for e, r in zip(elements, earth_ratios) if r > 0]
        if valid_ratios:
            ratio_elements, ratio_values = zip(*valid_ratios)
            ax3.bar(ratio_elements, ratio_values, color='orange', alpha=0.7)
            ax3.set_title('Space vs Earth Abundance Ratios', fontweight='bold')
            ax3.set_ylabel('Ratio (Space/Earth)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Earth Level')
            ax3.legend()
        
        # 4. Rarity classification
        rarity_counts = {}
        for element in element_pool.values():
            rarity = element['rarity_class']
            rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1
        
        if rarity_counts:
            ax4.pie(rarity_counts.values(), labels=rarity_counts.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('Element Rarity Classification', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Element pool analysis saved to: {save_path}")
        plt.show()
        plt.close()
    
    def create_compound_pool_analysis(self, compound_pool: Dict, save_path: str):
        """Create detailed compound pool analysis visualization"""
        print("\n⚗️ Creating Compound Pool Analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('⚗️ COMPLETE COMPOUND POOL ANALYSIS', fontsize=16, fontweight='bold')
        
        # Get top compounds for analysis
        compounds = list(compound_pool.keys())[:12]
        probabilities = [compound_pool[c]['avg_probability'] for c in compounds]
        frequencies = [compound_pool[c]['frequency'] for c in compounds]
        
        # 1. Formation probability distribution
        bars = ax1.barh(compounds, probabilities, color='lightcoral', alpha=0.7)
        ax1.set_title('Compound Formation Probabilities', fontweight='bold')
        ax1.set_xlabel('Formation Probability (%)')
        
        # Color unknown compounds differently
        for i, compound in enumerate(compounds):
            if not compound_pool[compound]['earth_exists']:
                bars[i].set_color('purple')
                bars[i].set_alpha(0.8)
        
        # 2. Compound frequency across locations
        ax2.bar(range(len(compounds)), frequencies, color='lightgreen', alpha=0.7)
        ax2.set_title('Compound Frequency Across Locations', fontweight='bold')
        ax2.set_ylabel('Frequency (%)')
        ax2.set_xticks(range(len(compounds)))
        ax2.set_xticklabels(compounds, rotation=45, ha='right')
        
        # 3. Stability ratings
        stability_data = {}
        for compound in compound_pool.values():
            stability = compound.get('stability', 'Unknown')
            stability_data[stability] = stability_data.get(stability, 0) + 1
        
        if stability_data:
            ax3.pie(stability_data.values(), labels=stability_data.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('Compound Stability Distribution', fontweight='bold')
        
        # 4. Known vs Unknown compounds
        known_count = sum(1 for c in compound_pool.values() if c['earth_exists'])
        unknown_count = len(compound_pool) - known_count
        
        ax4.bar(['Known on Earth', 'Unknown/Space-Only'], [known_count, unknown_count],
               color=['blue', 'red'], alpha=0.7)
        ax4.set_title('Known vs Unknown Compounds', fontweight='bold')
        ax4.set_ylabel('Number of Compounds')
        
        # Add value labels
        for i, v in enumerate([known_count, unknown_count]):
            ax4.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Compound pool analysis saved to: {save_path}")
        plt.show()
        plt.close()