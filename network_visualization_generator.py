#!/usr/bin/env python3
"""
Network Visualization Generator
Creates updated network diagrams showing current system state and tool chain flow
"""

import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NetworkVisualizationGenerator:
    """Generate network diagrams for the codebase analysis system"""
    
    def __init__(self):
        self.workspace_path = Path(__file__).parent
        self.analysis_data = None
        self.load_analysis_data()
    
    def load_analysis_data(self):
        """Load existing analysis data"""
        try:
            results_path = self.workspace_path / "codebase_analysis_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    self.analysis_data = json.load(f)
                print("✅ Analysis data loaded successfully")
            else:
                print("⚠️ No existing analysis data found")
        except Exception as e:
            print(f"❌ Error loading analysis data: {str(e)}")
    
    def create_tool_chain_flow_diagram(self):
        """Create a diagram showing the tool chain integration flow"""
        print("🎨 Creating Tool Chain Flow Diagram...")
        
        # Create directed graph for tool chain flow
        G = nx.DiGraph()
        
        # Define tool chain components and their relationships
        components = {
            'AE-Lang\nInterpreter': {'pos': (0, 2), 'color': '#FF6B6B', 'size': 3000},
            'Monster\nScanner': {'pos': (2, 2), 'color': '#4ECDC4', 'size': 2500},
            'TheWand\nIntegration': {'pos': (4, 2), 'color': '#45B7D1', 'size': 2500},
            'Auto-Rebuilder': {'pos': (6, 2), 'color': '#96CEB4', 'size': 3000},
            'Pygame\nVisualization': {'pos': (3, 0.5), 'color': '#FFEAA7', 'size': 2000},
            'Analysis\nTools': {'pos': (1, 0.5), 'color': '#DDA0DD', 'size': 2000},
            'Debug\nDashboard': {'pos': (5, 0.5), 'color': '#FFB3BA', 'size': 2000}
        }
        
        # Add nodes
        for component, attrs in components.items():
            G.add_node(component, **attrs)
        
        # Add edges for the main flow
        main_flow = [
            ('AE-Lang\nInterpreter', 'Monster\nScanner'),
            ('Monster\nScanner', 'TheWand\nIntegration'),
            ('TheWand\nIntegration', 'Auto-Rebuilder'),
        ]
        
        # Add supporting connections
        support_flow = [
            ('AE-Lang\nInterpreter', 'Analysis\nTools'),
            ('Monster\nScanner', 'Pygame\nVisualization'),
            ('Auto-Rebuilder', 'Pygame\nVisualization'),
            ('Auto-Rebuilder', 'Debug\nDashboard'),
            ('Analysis\nTools', 'Debug\nDashboard')
        ]
        
        # Add edges to graph
        G.add_edges_from(main_flow, weight=3, color='red', style='solid')
        G.add_edges_from(support_flow, weight=1, color='blue', style='dashed')
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Get positions
        pos = {node: data['pos'] for node, data in components.items()}
        
        # Draw main flow edges (thick red)
        main_edges = [(u, v) for u, v in G.edges() if (u, v) in main_flow]
        nx.draw_networkx_edges(G, pos, edgelist=main_edges, 
                              edge_color='red', width=3, alpha=0.8,
                              arrowsize=20, arrowstyle='->')
        
        # Draw support edges (thin blue, dashed)
        support_edges = [(u, v) for u, v in G.edges() if (u, v) in support_flow]
        nx.draw_networkx_edges(G, pos, edgelist=support_edges,
                              edge_color='blue', width=1, alpha=0.6,
                              arrowsize=15, arrowstyle='->', style='dashed')
        
        # Draw nodes
        node_colors = [components[node]['color'] for node in G.nodes()]
        node_sizes = [components[node]['size'] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, 
                              edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add title and annotations
        plt.title('Tool Chain Integration Flow\nAE-Lang → Monster Scanner → TheWand → Auto-Rebuilder',
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Main Tool Chain Flow'),
            Patch(facecolor='blue', alpha=0.6, label='Supporting Integrations'),
            Patch(facecolor='#FF6B6B', alpha=0.9, label='AE-Lang System'),
            Patch(facecolor='#4ECDC4', alpha=0.9, label='Monitoring/Scanning'),
            Patch(facecolor='#45B7D1', alpha=0.9, label='Integration Bridge'),
            Patch(facecolor='#96CEB4', alpha=0.9, label='Auto-Rebuilder'),
            Patch(facecolor='#FFEAA7', alpha=0.9, label='Visualization'),
            Patch(facecolor='#DDA0DD', alpha=0.9, label='Analysis Tools')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Add status information if available
        if self.analysis_data:
            health = self.analysis_data.get('health_metrics', {})
            overall_health = health.get('overall_health', 0)
            connectivity = health.get('connectivity_score', 0)
            
            status_text = f"System Health: {overall_health:.1%} | Connectivity: {connectivity:.1%}"
            plt.figtext(0.5, 0.02, status_text, ha='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the diagram
        output_path = self.workspace_path / "tool_chain_flow_diagram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Tool Chain Flow Diagram saved: {output_path}")
        
        return output_path
    
    def create_system_health_dashboard(self):
        """Create a comprehensive system health dashboard"""
        print("🎨 Creating System Health Dashboard...")
        
        if not self.analysis_data:
            print("❌ No analysis data available for dashboard")
            return None
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Codebase Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Health Metrics Pie Chart
        health_data = self.analysis_data.get('health_metrics', {})
        overall_health = health_data.get('overall_health', 0)
        
        # Create health score visualization
        health_score = overall_health * 100
        remaining_score = 100 - health_score
        
        ax1.pie([health_score, remaining_score], 
               labels=['Healthy', 'Issues'], 
               colors=['#96CEB4', '#FFB3BA'],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Overall System Health\n{health_score:.1f}%', fontweight='bold')
        
        # 2. Component Statistics Bar Chart
        summary = self.analysis_data.get('summary', {})
        
        components = ['Files', 'Functions', 'Classes', 'Relationships']
        values = [
            summary.get('total_files', 0),
            summary.get('total_functions', 0),
            summary.get('total_classes', 0),
            summary.get('total_relationships', 0) // 100  # Scale down for visibility
        ]
        
        bars = ax2.bar(components, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7'])
        ax2.set_title('System Components', fontweight='bold')
        ax2.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if components[values.index(value)] == 'Relationships':
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value*100:,}', ha='center', va='bottom')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:,}', ha='center', va='bottom')
        
        # 3. Issues Summary
        issues = [
            summary.get('broken_imports', 0),
            summary.get('integration_issues', 0),
            summary.get('isolated_files', 0),
            summary.get('unused_components', 0)
        ]
        issue_labels = ['Broken\nImports', 'Integration\nIssues', 'Isolated\nFiles', 'Unused\nComponents']
        
        bars3 = ax3.bar(issue_labels, issues, color=['#FF6B6B', '#FFB3BA', '#DDA0DD', '#FFEAA7'])
        ax3.set_title('Issues Summary', fontweight='bold')
        ax3.set_ylabel('Count')
        
        # Add value labels
        for bar, value in zip(bars3, issues):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value}', ha='center', va='bottom')
        
        # 4. Connectivity and Quality Metrics
        connectivity = health_data.get('connectivity_score', 0) * 100
        maintainability = health_data.get('maintainability_score', 0) * 100
        
        metrics = ['Connectivity', 'Maintainability', 'Overall Health']
        metric_values = [connectivity, maintainability, health_score]
        
        bars4 = ax4.barh(metrics, metric_values, color=['#4ECDC4', '#96CEB4', '#45B7D1'])
        ax4.set_xlim(0, 100)
        ax4.set_xlabel('Score (%)')
        ax4.set_title('Quality Metrics', fontweight='bold')
        
        # Add percentage labels
        for bar, value in zip(bars4, metric_values):
            width = bar.get_width()
            ax4.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{value:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save the dashboard
        output_path = self.workspace_path / "system_health_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ System Health Dashboard saved: {output_path}")
        
        return output_path
    
    def generate_all_visualizations(self):
        """Generate all available visualizations"""
        print("🎨 Generating All Network Visualizations...")
        
        generated_files = []
        
        # Generate tool chain flow diagram
        try:
            flow_diagram = self.create_tool_chain_flow_diagram()
            if flow_diagram:
                generated_files.append(flow_diagram)
        except Exception as e:
            print(f"❌ Error creating flow diagram: {str(e)}")
        
        # Generate system health dashboard
        try:
            health_dashboard = self.create_system_health_dashboard()
            if health_dashboard:
                generated_files.append(health_dashboard)
        except Exception as e:
            print(f"❌ Error creating health dashboard: {str(e)}")
        
        print(f"\n✅ Generated {len(generated_files)} visualization files:")
        for file_path in generated_files:
            print(f"  📊 {file_path.name}")
        
        return generated_files

def main():
    """Main execution function"""
    try:
        generator = NetworkVisualizationGenerator()
        generated_files = generator.generate_all_visualizations()
        
        if generated_files:
            print(f"\n🎉 Successfully generated {len(generated_files)} visualizations!")
            print("📁 Files are ready for viewing in your workspace.")
            return 0
        else:
            print("⚠️ No visualizations were generated.")
            return 1
    
    except Exception as e:
        print(f"❌ Visualization generation failed: {str(e)}")
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())
