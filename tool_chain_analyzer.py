#!/usr/bin/env python3
"""
Specialized Tool Chain Analyzer
Analyzes the specific integration between AE-Lang, Monster Scanner, TheWand, and Auto-Rebuilder
Identifies critical paths, data flow, and integration points between these core tools
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

@dataclass
class ToolComponent:
    """Represents a tool component in the chain"""
    name: str
    type: str  # 'interpreter', 'scanner', 'generator', 'rebuilder'
    file_path: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    integration_points: List[str]
    data_formats: List[str]
    status: str = "unknown"  # 'working', 'broken', 'isolated', 'partial'

class ToolChainAnalyzer:
    """Analyzes the complete tool chain integration"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.tools: Dict[str, ToolComponent] = {}
        self.data_flow_graph = nx.DiGraph()
        self.integration_issues: List[Dict[str, Any]] = []
        
        # Define expected tool signatures
        self.tool_signatures = {
            'ae_lang': {
                'files': ['AE-Lang_interp.py', 'AE-Lang.yaml'],
                'inputs': ['.ael', '.yaml'],
                'outputs': ['parsed_ast', 'execution_result'],
                'type': 'interpreter'
            },
            'monster_scanner': {
                'files': ['monster_scanner.py', 'scanner.py'],
                'inputs': ['ml_files', 'training_data'],
                'outputs': ['barcode', 'trained_model'],
                'type': 'scanner'
            },
            'the_wand': {
                'files': ['ae_wand_integration_bridge.py', 'wand.py'],
                'inputs': ['project_overview', 'requirements'],
                'outputs': ['generated_code', 'project_structure'],
                'type': 'generator'
            },
            'auto_rebuilder': {
                'files': ['auto_rebuilder.py', 'auto_rebuilder_adapter.py'],
                'inputs': ['broken_code', 'error_reports'],
                'outputs': ['fixed_code', 'integration_report'],
                'type': 'rebuilder'
            }
        }
    
    def analyze_tool_chain(self) -> Dict[str, Any]:
        """Analyze the complete tool chain"""
        print("Analyzing Tool Chain Integration...")
        
        # Step 1: Identify all tools
        self._identify_tools()
        
        # Step 2: Analyze each tool's interface
        for tool_name, tool in self.tools.items():
            self._analyze_tool_interface(tool_name, tool)
        
        # Step 3: Map data flow between tools
        self._map_data_flow()
        
        # Step 4: Identify integration gaps
        self._identify_integration_gaps()
        
        # Step 5: Analyze critical paths
        critical_paths = self._analyze_critical_paths()
        
        # Step 6: Generate tool chain report
        return self._generate_tool_chain_report(critical_paths)
    
    def _identify_tools(self):
        """Identify all tool components in the codebase"""
        for tool_name, signature in self.tool_signatures.items():
            found_files = []
            
            # Look for tool files
            for pattern in signature['files']:
                matches = list(self.root_path.glob(f"**/{pattern}"))
                found_files.extend(matches)
            
            if found_files:
                main_file = found_files[0]  # Use first found file as main
                tool = ToolComponent(
                    name=tool_name,
                    type=signature['type'],
                    file_path=str(main_file.relative_to(self.root_path)),
                    inputs=signature['inputs'].copy(),
                    outputs=signature['outputs'].copy(),
                    dependencies=[],
                    integration_points=[],
                    data_formats=[]
                )
                
                self.tools[tool_name] = tool
                print(f"Found {tool_name}: {tool.file_path}")
            else:
                print(f"Warning: {tool_name} not found (expected files: {signature['files']})")
    
    def _analyze_tool_interface(self, tool_name: str, tool: ToolComponent):
        """Analyze a specific tool's interface and capabilities"""
        file_path = self.root_path / tool.file_path
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze based on tool type
            if tool.type == 'interpreter':
                self._analyze_interpreter(tool, content)
            elif tool.type == 'scanner':
                self._analyze_scanner(tool, content)
            elif tool.type == 'generator':
                self._analyze_generator(tool, content)
            elif tool.type == 'rebuilder':
                self._analyze_rebuilder(tool, content)
            
            tool.status = "working"
            
        except Exception as e:
            print(f"Error analyzing {tool_name}: {e}")
            tool.status = "broken"
    
    def _analyze_interpreter(self, tool: ToolComponent, content: str):
        """Analyze AE-Lang interpreter"""
        # Look for key functions and classes
        patterns = {
            'parse_functions': r'def\s+(parse_\w+|interpret_\w+)',
            'execute_functions': r'def\s+(execute_\w+|run_\w+)',
            'file_handlers': r'def\s+.*\.(ael|yaml)',
            'output_methods': r'def\s+.*output|return|yield'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                tool.integration_points.extend([f"{pattern_type}: {m}" for m in matches])
        
        # Check for specific integrations
        if 'monster_scanner' in content.lower():
            tool.integration_points.append("monster_scanner_integration")
        if 'auto_rebuilder' in content.lower():
            tool.integration_points.append("auto_rebuilder_integration")
    
    def _analyze_scanner(self, tool: ToolComponent, content: str):
        """Analyze Monster Scanner"""
        patterns = {
            'ml_file_processing': r'def\s+.*process.*ml|machine.*learning',
            'barcode_generation': r'def\s+.*barcode|generate.*code',
            'training_functions': r'def\s+.*train|fit|learn',
            'llm_prompting': r'def\s+.*prompt|llm|gpt'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                tool.integration_points.extend([f"{pattern_type}: {m}" for m in matches])
        
        # Check supported file formats
        format_patterns = [r'\.(\w+).*machine.*learning', r'\.(\w+).*ml.*file']
        for pattern in format_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tool.data_formats.extend(matches)
    
    def _analyze_generator(self, tool: ToolComponent, content: str):
        """Analyze TheWand"""
        patterns = {
            'project_analysis': r'def\s+.*analyze.*project|overview',
            'code_generation': r'def\s+.*generate.*code|create.*project',
            'llm_integration': r'def\s+.*llm|gpt|prompt',
            'output_structure': r'def\s+.*structure|organize|build'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                tool.integration_points.extend([f"{pattern_type}: {m}" for m in matches])
        
        # Check for auto-rebuilder handoff
        if 'auto_rebuilder' in content.lower():
            tool.integration_points.append("auto_rebuilder_handoff")
    
    def _analyze_rebuilder(self, tool: ToolComponent, content: str):
        """Analyze Auto-Rebuilder"""
        patterns = {
            'error_detection': r'def\s+.*detect.*error|find.*issue',
            'code_fixing': r'def\s+.*fix|repair|correct',
            'integration_repair': r'def\s+.*integrate|connect|bridge',
            'validation': r'def\s+.*validate|test|verify'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                tool.integration_points.extend([f"{pattern_type}: {m}" for m in matches])
        
        # Check for wand integration
        if 'wand' in content.lower():
            tool.integration_points.append("wand_output_processing")
    
    def _map_data_flow(self):
        """Map data flow between tools"""
        # Define expected flow patterns
        flow_patterns = [
            ('ae_lang', 'monster_scanner', 'ael_scripts'),
            ('monster_scanner', 'the_wand', 'training_data'),
            ('the_wand', 'auto_rebuilder', 'generated_code'),
            ('auto_rebuilder', 'ae_lang', 'fixed_scripts')
        ]
        
        for source, target, data_type in flow_patterns:
            if source in self.tools and target in self.tools:
                self.data_flow_graph.add_edge(source, target, data_type=data_type)
                
                # Check if flow is actually implemented
                flow_implemented = self._check_flow_implementation(source, target, data_type)
                self.data_flow_graph[source][target]['implemented'] = flow_implemented
    
    def _check_flow_implementation(self, source: str, target: str, data_type: str) -> bool:
        """Check if data flow between tools is actually implemented"""
        source_tool = self.tools[source]
        target_tool = self.tools[target]
        
        # Simple heuristic: check if target tool is referenced in source tool
        try:
            source_path = self.root_path / source_tool.file_path
            with open(source_path, 'r', encoding='utf-8') as f:
                source_content = f.read()
            
            # Look for references to target tool
            target_patterns = [
                target_tool.name,
                target_tool.file_path.split('/')[-1].replace('.py', ''),
                data_type
            ]
            
            for pattern in target_patterns:
                if pattern.lower() in source_content.lower():
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _identify_integration_gaps(self):
        """Identify gaps in tool integration"""
        gaps = []
        
        # Check for missing connections
        for source, target, data in self.data_flow_graph.edges(data=True):
            if not data.get('implemented', False):
                gaps.append({
                    'type': 'missing_connection',
                    'source': source,
                    'target': target,
                    'data_type': data.get('data_type', 'unknown'),
                    'severity': 'high'
                })
        
        # Check for isolated tools
        for tool_name, tool in self.tools.items():
            in_degree = self.data_flow_graph.in_degree(tool_name)
            out_degree = self.data_flow_graph.out_degree(tool_name)
            
            if in_degree == 0 and out_degree == 0:
                gaps.append({
                    'type': 'isolated_tool',
                    'tool': tool_name,
                    'severity': 'medium'
                })
        
        # Check for broken tools
        for tool_name, tool in self.tools.items():
            if tool.status == 'broken':
                gaps.append({
                    'type': 'broken_tool',
                    'tool': tool_name,
                    'file_path': tool.file_path,
                    'severity': 'critical'
                })
        
        self.integration_issues = gaps
    
    def _analyze_critical_paths(self) -> List[List[str]]:
        """Identify critical paths through the tool chain"""
        critical_paths = []
        
        # Find all simple paths between tools
        try:
            for source in self.tools:
                for target in self.tools:
                    if source != target:
                        paths = list(nx.all_simple_paths(self.data_flow_graph, source, target))
                        critical_paths.extend(paths)
        except nx.NetworkXNoPath:
            pass
        
        # Sort by path length (longer paths are more critical)
        critical_paths.sort(key=len, reverse=True)
        
        return critical_paths[:5]  # Return top 5 critical paths
    
    def _generate_tool_chain_report(self, critical_paths: List[List[str]]) -> Dict[str, Any]:
        """Generate comprehensive tool chain analysis report"""
        return {
            'tool_chain_summary': {
                'total_tools': len(self.tools),
                'working_tools': len([t for t in self.tools.values() if t.status == 'working']),
                'broken_tools': len([t for t in self.tools.values() if t.status == 'broken']),
                'isolated_tools': len([t for t in self.tools.values() 
                                     if self.data_flow_graph.degree(t.name) == 0]),
                'total_connections': len(self.data_flow_graph.edges()),
                'implemented_connections': len([e for e in self.data_flow_graph.edges(data=True) 
                                              if e[2].get('implemented', False)])
            },
            'tool_details': {name: {
                'type': tool.type,
                'file_path': tool.file_path,
                'status': tool.status,
                'integration_points': tool.integration_points,
                'data_formats': tool.data_formats
            } for name, tool in self.tools.items()},
            'data_flow': {
                'connections': [(source, target, data.get('data_type', 'unknown'), 
                               data.get('implemented', False))
                              for source, target, data in self.data_flow_graph.edges(data=True)],
                'critical_paths': critical_paths
            },
            'integration_issues': self.integration_issues,
            'recommendations': self._generate_tool_specific_recommendations(),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_tool_specific_recommendations(self) -> List[str]:
        """Generate specific recommendations for tool chain improvement"""
        recommendations = []
        
        # Check each expected flow
        expected_flows = [
            ('ae_lang', 'monster_scanner', 'AE-Lang should pass .ael scripts to Monster Scanner for ML training'),
            ('monster_scanner', 'the_wand', 'Monster Scanner should provide training data/barcodes to TheWand'),
            ('the_wand', 'auto_rebuilder', 'TheWand output should automatically feed into Auto-Rebuilder'),
            ('auto_rebuilder', 'ae_lang', 'Auto-Rebuilder should output clean .ael scripts back to AE-Lang')
        ]
        
        for source, target, description in expected_flows:
            if source in self.tools and target in self.tools:
                if not self.data_flow_graph.has_edge(source, target):
                    recommendations.append(f"MISSING: {description}")
                elif not self.data_flow_graph[source][target].get('implemented', False):
                    recommendations.append(f"INCOMPLETE: {description}")
        
        # Tool-specific recommendations
        if 'ae_lang' in self.tools:
            ae_lang = self.tools['ae_lang']
            if 'monster_scanner_integration' not in ae_lang.integration_points:
                recommendations.append("AE-Lang needs integration with Monster Scanner for ML file processing")
        
        if 'auto_rebuilder' in self.tools:
            auto_rebuilder = self.tools['auto_rebuilder']
            if 'wand_output_processing' not in auto_rebuilder.integration_points:
                recommendations.append("Auto-Rebuilder needs specific TheWand output processing capabilities")
        
        # Add consciousness integration recommendations
        consciousness_files = ['ae_theory_production_integration.py', 'ae_consciousness_integration.py']
        found_consciousness = any((self.root_path / f).exists() for f in consciousness_files)
        
        if found_consciousness:
            recommendations.append("Integrate AE Theory consciousness system with tool chain for enhanced performance")
        
        return recommendations

class ToolChainVisualizer:
    """Visualize the tool chain relationships and data flow"""
    
    def __init__(self, analyzer: ToolChainAnalyzer):
        self.analyzer = analyzer
    
    def create_tool_chain_diagram(self, output_file: str = "tool_chain_diagram.png"):
        """Create comprehensive tool chain diagram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left diagram: Tool relationships
        self._draw_tool_relationships(ax1)
        
        # Right diagram: Data flow
        self._draw_data_flow(ax2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Tool chain diagram saved to: {output_file}")
    
    def _draw_tool_relationships(self, ax):
        """Draw tool relationships diagram"""
        ax.set_title("Tool Chain Relationships", fontsize=14, fontweight='bold')
        
        # Position tools in a logical flow
        positions = {
            'ae_lang': (0, 2),
            'monster_scanner': (2, 2),
            'the_wand': (4, 2),
            'auto_rebuilder': (6, 2)
        }
        
        # Draw tools as boxes
        for tool_name, (x, y) in positions.items():
            if tool_name in self.analyzer.tools:
                tool = self.analyzer.tools[tool_name]
                
                # Choose color based on status
                color = {
                    'working': 'lightgreen',
                    'broken': 'lightcoral',
                    'isolated': 'lightyellow',
                    'unknown': 'lightgray'
                }.get(tool.status, 'lightgray')
                
                # Draw tool box
                box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=color, edgecolor='black')
                ax.add_patch(box)
                
                # Add tool name
                ax.text(x, y, tool_name.replace('_', '\n'), 
                       ha='center', va='center', fontweight='bold')
                
                # Add status indicator
                ax.text(x, y-0.45, f"({tool.status})", 
                       ha='center', va='center', fontsize=8)
        
        # Draw connections
        for source, target, data in self.analyzer.data_flow_graph.edges(data=True):
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                
                # Choose arrow color based on implementation
                color = 'green' if data.get('implemented', False) else 'red'
                style = '-' if data.get('implemented', False) else '--'
                
                ax.annotate('', xy=(x2-0.4, y2), xytext=(x1+0.4, y1),
                          arrowprops=dict(arrowstyle='->', color=color, 
                                        linestyle=style, lw=2))
        
        ax.set_xlim(-1, 7)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_data_flow(self, ax):
        """Draw data flow diagram"""
        ax.set_title("Data Flow Analysis", fontsize=14, fontweight='bold')
        
        # Create a more detailed data flow visualization
        flow_data = []
        for source, target, data in self.analyzer.data_flow_graph.edges(data=True):
            flow_data.append({
                'source': source,
                'target': target,
                'data_type': data.get('data_type', 'unknown'),
                'implemented': data.get('implemented', False)
            })
        
        # Draw as a flow chart
        y_pos = len(flow_data)
        for i, flow in enumerate(flow_data):
            y = y_pos - i
            
            # Source box
            color = 'lightgreen' if flow['implemented'] else 'lightcoral'
            ax.barh(y, 1, left=0, color=color, alpha=0.7)
            ax.text(0.5, y, flow['source'], ha='center', va='center', fontsize=10)
            
            # Arrow
            arrow_color = 'green' if flow['implemented'] else 'red'
            ax.annotate('', xy=(2, y), xytext=(1, y),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
            
            # Data type
            ax.text(1.5, y+0.1, flow['data_type'], ha='center', va='bottom', fontsize=8)
            
            # Target box
            ax.barh(y, 1, left=2, color=color, alpha=0.7)
            ax.text(2.5, y, flow['target'], ha='center', va='center', fontsize=10)
        
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0.5, len(flow_data) + 0.5)
        ax.set_xlabel("Data Flow Direction")
        ax.set_yticks([])

# Integration with existing systems
def analyze_current_workspace():
    """Analyze the current workspace with both general and tool-specific analysis"""
    workspace_path = "."
    
    print("Running Comprehensive Codebase Analysis...")
    print("=" * 60)
    
    # Run tool chain analysis
    tool_analyzer = ToolChainAnalyzer(workspace_path)
    tool_results = tool_analyzer.analyze_tool_chain()
    
    # Save tool chain report
    with open("tool_chain_analysis_report.json", 'w') as f:
        json.dump(tool_results, f, indent=2)
    
    # Create visualization
    tool_visualizer = ToolChainVisualizer(tool_analyzer)
    tool_visualizer.create_tool_chain_diagram()
    
    # Print summary
    print("\nTool Chain Analysis Summary:")
    print(f"- Tools Found: {tool_results['tool_chain_summary']['total_tools']}")
    print(f"- Working Tools: {tool_results['tool_chain_summary']['working_tools']}")
    print(f"- Broken Tools: {tool_results['tool_chain_summary']['broken_tools']}")
    print(f"- Connections: {tool_results['tool_chain_summary']['total_connections']}")
    print(f"- Implemented Connections: {tool_results['tool_chain_summary']['implemented_connections']}")
    
    if tool_results['integration_issues']:
        print(f"\nCritical Issues Found: {len(tool_results['integration_issues'])}")
        for issue in tool_results['integration_issues'][:5]:  # Show top 5
            print(f"  - {issue['type']}: {issue.get('tool', issue.get('source', 'unknown'))}")
    
    if tool_results['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(tool_results['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
    
    return tool_results

if __name__ == "__main__":
    analyze_current_workspace()
