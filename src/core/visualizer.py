#!/usr/bin/env python3
"""
NASA Rover Advanced Visualization System
=======================================

Creates stunning visualizations for drilling data, element distributions,
compound formations, and ML predictions with interactive dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime


class NASARoverVisualizer:
    """Advanced visualization system for NASA rover data"""
    
    def __init__(self, style='dark', figsize=(12, 8)):
        # Set up beautiful styling
        plt.style.use('dark_background' if style == 'dark' else 'default')
        sns.set_palette("husl")
        
        self.figsize = figsize
        self.colors = {
            'primary': '#FF6B35',    # Space orange
            'secondary': '#004E89',   # Deep space blue  
            'accent': '#1DB954',      # Discovery green
            'warning': '#FFD23F',     # Caution yellow
            'danger': '#EE4266',      # Alert red
            'earth': '#6B7FD7',       # Earth blue
            'mars': '#CD5C5C',        # Mars red
            'space': '#2E0249'        # Deep space purple
        }
    
    def create_element_abundance_chart(self, detailed_reports: List, save_path: Optional[str] = None) -> str:
        """Create beautiful element abundance visualization"""
        
        print("📊 Creating element abundance visualization...")
        
        # Collect all element data
        all_elements = {}
        location_names = []
        
        for report in detailed_reports:
            location_names.append(report.site_id)
            
        # First, collect all unique elements across all locations
        unique_elements = set()
        for report in detailed_reports:
            for ea in report.element_analyses:
                unique_elements.add(ea.symbol)
        
        # Now create the data structure with all elements for each location
        for element in unique_elements:
            all_elements[element] = []
            
        for report in detailed_reports:
            # Create a dict of elements for this location
            location_elements = {ea.symbol: ea.percentage for ea in report.element_analyses}
            
            # Add data for each element (0 if not found at this location)
            for element in unique_elements:
                all_elements[element].append(location_elements.get(element, 0.0))
        
        # Create DataFrame
        df_elements = pd.DataFrame(all_elements, index=location_names).fillna(0)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Element Abundance Heatmap', 'Top Elements Across Locations',
                'Element Diversity by Location', 'Earth vs Space Elements'
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. Heatmap of element abundance
        heatmap_data = df_elements.T.values
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=location_names,
                y=list(df_elements.columns),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Abundance %")
            ),
            row=1, col=1
        )
        
        # 2. Top elements bar chart
        top_elements = df_elements.mean().nlargest(10)
        fig.add_trace(
            go.Bar(
                x=top_elements.index,
                y=top_elements.values,
                marker_color=self.colors['primary'],
                name="Average Abundance"
            ),
            row=1, col=2
        )
        
        # 3. Element diversity
        diversity_data = [len([ea for ea in report.element_analyses if ea.percentage > 1.0]) for report in detailed_reports]
        fig.add_trace(
            go.Scatter(
                x=location_names,
                y=diversity_data,
                mode='lines+markers',
                marker=dict(size=10, color=self.colors['accent']),
                line=dict(width=3),
                name="Element Diversity"
            ),
            row=2, col=1
        )
        
        # 4. Earth vs Space elements
        earth_elements = []
        space_elements = []
        
        for report in detailed_reports:
            earth_count = sum(1 for ea in report.element_analyses if ea.earth_exists)
            space_count = len(report.element_analyses) - earth_count
            earth_elements.append(earth_count)
            space_elements.append(space_count)
        
        fig.add_trace(
            go.Bar(
                x=location_names,
                y=earth_elements,
                name="Earth Elements",
                marker_color=self.colors['earth']
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=location_names,
                y=space_elements,
                name="Unknown Elements",
                marker_color=self.colors['warning']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🧪 ELEMENT ABUNDANCE ANALYSIS",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            showlegend=True,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(30,30,30,1)'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   ✅ Saved to: {save_path}")
        
        fig.show()
        return save_path or "element_abundance_chart.html"
    
    def create_compound_formation_chart(self, detailed_reports: List, save_path: Optional[str] = None) -> str:
        """Create compound formation success visualization"""
        
        print("⚗️ Creating compound formation visualization...")
        
        # Collect compound data
        compound_data = []
        
        for report in detailed_reports:
            for ca in report.compound_analyses:
                compound_data.append({
                    'location': report.site_id,
                    'formula': ca.formula,
                    'formation_probability': ca.formation_probability,
                    'earth_exists': ca.earth_exists,
                    'classification': ca.classification,
                    'stability': ca.stability_rating,
                    'databases_found': len(ca.earth_databases_found)
                })
        
        df_compounds = pd.DataFrame(compound_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "domain"}],
                   [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                'Compound Formation Success Rate', 'Earth vs Unknown Compounds',
                'Formation Probability Distribution', 'Stability Classification'
            )
        )
        
        # 1. Formation success rate by location
        success_rates = []
        for report in detailed_reports:
            total = report.total_compounds_tested
            successful = len([ca for ca in report.compound_analyses if ca.formation_probability > 0.5])
            success_rates.append((successful / total) * 100 if total > 0 else 0)
        
        fig.add_trace(
            go.Bar(
                x=[r.site_id for r in detailed_reports],
                y=success_rates,
                marker_color=self.colors['accent'],
                name="Success Rate %"
            ),
            row=1, col=1
        )
        
        # 2. Earth vs Unknown compounds pie chart
        earth_count = len(df_compounds[df_compounds['earth_exists'] == True])
        unknown_count = len(df_compounds[df_compounds['earth_exists'] == False])
        
        fig.add_trace(
            go.Pie(
                labels=['Known on Earth', 'Unknown/Space-only'],
                values=[earth_count, unknown_count],
                marker_colors=[self.colors['earth'], self.colors['warning']]
            ),
            row=1, col=2
        )
        
        # 3. Formation probability histogram
        fig.add_trace(
            go.Histogram(
                x=df_compounds['formation_probability'],
                nbinsx=20,
                marker_color=self.colors['primary'],
                opacity=0.7,
                name="Formation Probability"
            ),
            row=2, col=1
        )
        
        # 4. Stability classification
        stability_counts = df_compounds['stability'].value_counts()
        fig.add_trace(
            go.Bar(
                x=stability_counts.index,
                y=stability_counts.values,
                marker_color=self.colors['secondary'],
                name="Stability Count"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="⚗️ COMPOUND FORMATION ANALYSIS",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            showlegend=True,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(30,30,30,1)'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   ✅ Saved to: {save_path}")
        
        fig.show()
        return save_path or "compound_formation_chart.html"
    
    def create_discovery_potential_map(self, detailed_reports: List, save_path: Optional[str] = None) -> str:
        """Create discovery potential 3D visualization"""
        
        print("🗺️ Creating discovery potential map...")
        
        # Extract location data
        lats = [r.coordinates[0] for r in detailed_reports]
        lons = [r.coordinates[1] for r in detailed_reports]
        discovery_potentials = [r.discovery_potential for r in detailed_reports]
        earth_similarities = [r.earth_similarity_score for r in detailed_reports]
        site_ids = [r.site_id for r in detailed_reports]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Main scatter plot
        fig.add_trace(go.Scatter3d(
            x=lats,
            y=lons,
            z=discovery_potentials,
            mode='markers+text',
            marker=dict(
                size=earth_similarities,
                sizeref=2.*max(earth_similarities)/(40.**2),
                sizemode='diameter',
                color=discovery_potentials,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Discovery Potential %")
            ),
            text=site_ids,
            textposition="top center",
            name="Drilling Sites"
        ))
        
        # Add surface mesh for visualization
        lat_range = np.linspace(min(lats), max(lats), 10)
        lon_range = np.linspace(min(lons), max(lons), 10)
        lat_mesh, lon_mesh = np.meshgrid(lat_range, lon_range)
        
        # Interpolate discovery potential for surface
        from scipy.interpolate import griddata
        discovery_mesh = griddata(
            (lats, lons), discovery_potentials,
            (lat_mesh, lon_mesh), method='linear'
        )
        
        fig.add_trace(go.Surface(
            x=lat_mesh,
            y=lon_mesh,
            z=discovery_mesh,
            opacity=0.3,
            showscale=False,
            name="Discovery Surface"
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🗺️ DISCOVERY POTENTIAL MAPPING",
                font=dict(size=20),
                x=0.5
            ),
            scene=dict(
                xaxis_title="Latitude",
                yaxis_title="Longitude", 
                zaxis_title="Discovery Potential %",
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(30,30,30,1)'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   ✅ Saved to: {save_path}")
        
        fig.show()
        return save_path or "discovery_potential_map.html"
    
    def create_ml_prediction_dashboard(self, predictions: Dict, location_name: str, save_path: Optional[str] = None) -> str:
        """Create ML prediction dashboard"""
        
        print("🔮 Creating ML prediction dashboard...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Predictions for {location_name}', 'Confidence Levels',
                'Feature Importance', 'Prediction Comparison'
            ),
            specs=[[{"type": "domain"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "polar"}]]
        )
        
        # 1. Prediction values gauge
        prediction_values = [pred.predicted_value for pred in predictions.values()]
        prediction_names = list(predictions.keys())
        
        for i, (name, pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=pred.predicted_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': name.replace('_', ' ').title()},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': self.colors['primary']},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )
            break  # Show first prediction as gauge
        
        # 2. Confidence levels
        confidence_values = [pred.confidence for pred in predictions.values()]
        fig.add_trace(
            go.Bar(
                x=[name.replace('_', ' ').title() for name in prediction_names],
                y=confidence_values,
                marker_color=self.colors['accent'],
                name="Confidence %"
            ),
            row=1, col=2
        )
        
        # 3. Feature importance (from first prediction)
        first_pred = list(predictions.values())[0]
        factors = list(first_pred.factors_influence.keys())[:5]
        importances = list(first_pred.factors_influence.values())[:5]
        
        fig.add_trace(
            go.Bar(
                x=importances,
                y=factors,
                orientation='h',
                marker_color=self.colors['secondary'],
                name="Feature Importance"
            ),
            row=2, col=1
        )
        
        # 4. Prediction comparison radar
        fig.add_trace(
            go.Scatterpolar(
                r=prediction_values,
                theta=[name.replace('_', ' ').title() for name in prediction_names],
                fill='toself',
                marker_color=self.colors['primary'],
                name="Predictions"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"🔮 ML PREDICTIONS: {location_name}",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            showlegend=True,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(30,30,30,1)'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   ✅ Saved to: {save_path}")
        
        fig.show()
        return save_path or f"ml_predictions_{location_name}.html"
    
    def create_comprehensive_dashboard(self, detailed_reports: List, save_path: Optional[str] = None) -> str:
        """Create comprehensive mission dashboard"""
        
        print("🚀 Creating comprehensive mission dashboard...")
        
        # Create main dashboard with multiple sections
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Mission Overview', 'Element Distribution', 'Compound Success',
                'Earth Similarity', 'Discovery Potential', 'Scientific Value',
                'Environmental Factors', 'Location Types', 'Mission Timeline'
            ),
            specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "domain"}, {"type": "xy"}]]
        )
        
        # Mission overview metrics
        total_locations = len(detailed_reports)
        total_elements = sum(r.total_elements_found for r in detailed_reports)
        total_compounds = sum(r.total_compounds_tested for r in detailed_reports)
        avg_discovery = np.mean([r.discovery_potential for r in detailed_reports])
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_locations,
                title={"text": "Locations Drilled"},
                number={'suffix': " sites"}
            ),
            row=1, col=1
        )
        
        # Element distribution
        all_elements = {}
        for report in detailed_reports:
            for ea in report.element_analyses[:5]:  # Top 5 per location
                if ea.symbol not in all_elements:
                    all_elements[ea.symbol] = 0
                all_elements[ea.symbol] += ea.percentage
        
        top_elements = sorted(all_elements.items(), key=lambda x: x[1], reverse=True)[:8]
        fig.add_trace(
            go.Bar(
                x=[elem[0] for elem in top_elements],
                y=[elem[1] for elem in top_elements],
                marker_color=self.colors['primary'],
                name="Element Abundance"
            ),
            row=1, col=2
        )
        
        # Compound success rates
        success_rates = []
        for report in detailed_reports:
            known = len(report.known_compounds)
            total = report.total_compounds_tested
            success_rates.append((known / total) * 100 if total > 0 else 0)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(success_rates))),
                y=success_rates,
                mode='lines+markers',
                marker=dict(size=8, color=self.colors['accent']),
                name="Success Rate %"
            ),
            row=1, col=3
        )
        
        # Earth similarity scores
        earth_scores = [r.earth_similarity_score for r in detailed_reports]
        fig.add_trace(
            go.Box(
                y=earth_scores,
                marker_color=self.colors['earth'],
                name="Earth Similarity"
            ),
            row=2, col=1
        )
        
        # Discovery potential
        discovery_scores = [r.discovery_potential for r in detailed_reports]
        fig.add_trace(
            go.Box(
                y=discovery_scores,
                marker_color=self.colors['warning'],
                name="Discovery Potential"
            ),
            row=2, col=2
        )
        
        # Scientific value
        scientific_scores = [r.scientific_value for r in detailed_reports]
        fig.add_trace(
            go.Box(
                y=scientific_scores,
                marker_color=self.colors['danger'],
                name="Scientific Value"
            ),
            row=2, col=3
        )
        
        # Environmental factors correlation
        temperatures = [r.temperature for r in detailed_reports]
        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=discovery_scores,
                mode='markers',
                marker=dict(size=10, color=self.colors['secondary']),
                name="Temp vs Discovery"
            ),
            row=3, col=1
        )
        
        # Location types pie chart
        site_types = {}
        for report in detailed_reports:
            site_type = report.site_type
            site_types[site_type] = site_types.get(site_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(site_types.keys()),
                values=list(site_types.values()),
                marker_colors=[self.colors['primary'], self.colors['accent'], self.colors['secondary']]
            ),
            row=3, col=2
        )
        
        # Mission timeline
        depths = [r.depth for r in detailed_reports]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(depths))),
                y=depths,
                mode='lines+markers',
                marker=dict(size=8, color=self.colors['space']),
                name="Drilling Depth"
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🚀 NASA ROVER MISSION COMPREHENSIVE DASHBOARD",
                font=dict(size=24, color='white'),
                x=0.5
            ),
            showlegend=False,
            height=1200,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(30,30,30,1)'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   ✅ Comprehensive dashboard saved to: {save_path}")
        
        fig.show()
        return save_path or "comprehensive_mission_dashboard.html"
    
    def save_all_visualizations(self, detailed_reports: List, output_dir: str = "visualizations"):
        """Generate and save all visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"🎨 Generating all visualizations in: {output_path}")
        
        # Generate all charts
        charts = [
            ("element_abundance.html", self.create_element_abundance_chart),
            ("compound_formation.html", self.create_compound_formation_chart),
            ("discovery_potential_map.html", self.create_discovery_potential_map),
            ("comprehensive_dashboard.html", self.create_comprehensive_dashboard)
        ]
        
        saved_files = []
        for filename, chart_func in charts:
            filepath = output_path / filename
            chart_func(detailed_reports, str(filepath))
            saved_files.append(str(filepath))
        
        # Create index file
        self._create_visualization_index(saved_files, output_path)
        
        print(f"🎨 All visualizations saved to: {output_path}")
        return str(output_path)
    
    def _create_visualization_index(self, chart_files: List[str], output_path: Path):
        """Create HTML index for all visualizations"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NASA Rover Mission Visualizations</title>
            <style>
                body {{ 
                    background: linear-gradient(135deg, #2E0249, #004E89);
                    color: white;
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .card {{ 
                    background: rgba(255,255,255,0.1);
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                    backdrop-filter: blur(10px);
                }}
                .btn {{ 
                    background: #FF6B35;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px;
                }}
                .btn:hover {{ background: #E55A2B; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 NASA Rover Mission Visualizations</h1>
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="card">
                    <h2>📊 Available Visualizations</h2>
                    <a href="element_abundance.html" class="btn">🧪 Element Abundance</a>
                    <a href="compound_formation.html" class="btn">⚗️ Compound Formation</a>
                    <a href="discovery_potential_map.html" class="btn">🗺️ Discovery Map</a>
                    <a href="comprehensive_dashboard.html" class="btn">🚀 Mission Dashboard</a>
                </div>
                
                <div class="card">
                    <h2>🎯 Mission Summary</h2>
                    <p>This visualization suite provides comprehensive analysis of NASA rover drilling missions, 
                    including element discovery, compound formation, and machine learning predictions.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path / "index.html", "w") as f:
            f.write(html_content)


def main():
    """Demo the visualization system"""
    
    print("🎨 NASA ROVER VISUALIZATION SYSTEM")
    print("=" * 50)
    
    # This would typically be called after enhanced drilling analysis
    print("Note: This demo requires drilling data from enhanced_drilling_analyzer.py")
    print("Run complete enhanced mission to see visualizations in action!")


if __name__ == "__main__":
    main()