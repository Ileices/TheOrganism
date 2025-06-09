#!/usr/bin/env python3
"""
Codebase Relationship Analyzer & Visualizer
Analyzes the entire codebase to create comprehensive relationship diagrams and identify:
- Import dependencies and call chains
- Isolated/disconnected scripts
- Integration gaps and broken connections
- Function/class usage patterns
- Dead code and unused components
- Critical path analysis for debugging
"""

import ast
import os
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import logging

@dataclass
class CodeNode:
    """Represents a code entity (file, function, class, variable)"""
    name: str
    type: str  # 'file', 'function', 'class', 'variable', 'import'
    file_path: str
    line_number: int = 0
    dependencies: List[str] = None
    dependents: List[str] = None
    usage_count: int = 0
    complexity_score: float = 0.0
    is_isolated: bool = False
    integration_health: float = 1.0  # 0.0 = broken, 1.0 = perfect
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.dependents is None:
            self.dependents = []

@dataclass
class CodeRelationship:
    """Represents a relationship between code entities"""
    source: str
    target: str
    relationship_type: str  # 'imports', 'calls', 'inherits', 'uses', 'defines'
    strength: float = 1.0  # Connection strength
    is_broken: bool = False
    error_message: str = ""

class CodebaseAnalyzer:
    """Comprehensive codebase analysis and visualization system"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.nodes: Dict[str, CodeNode] = {}
        self.relationships: List[CodeRelationship] = []
        self.graph = nx.DiGraph()
        
        # Analysis results
        self.isolated_files: List[str] = []
        self.broken_imports: List[Tuple[str, str, str]] = []
        self.unused_functions: List[str] = []
        self.integration_issues: List[Dict[str, Any]] = []
        self.dependency_chains: Dict[str, List[str]] = {}
        
        # File patterns to analyze
        self.code_extensions = {'.py', '.yaml', '.yml', '.json', '.md'}
        self.ignore_patterns = {'__pycache__', '.git', '.vscode', 'node_modules'}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Complete codebase analysis"""
        self.logger.info(f"Starting codebase analysis for: {self.root_path}")
        
        # Step 1: Discover all code files
        files = self._discover_files()
        self.logger.info(f"Found {len(files)} code files")
        
        # Step 2: Parse each file and extract entities
        for file_path in files:
            self._analyze_file(file_path)
        
        # Step 3: Build relationship graph
        self._build_relationship_graph()
        
        # Step 4: Identify issues
        self._identify_isolated_components()
        self._identify_broken_imports()
        self._identify_unused_components()
        self._analyze_integration_health()
        
        # Step 5: Calculate metrics
        analysis_results = self._generate_analysis_report()
        
        self.logger.info("Codebase analysis complete")
        return analysis_results
    
    def _discover_files(self) -> List[Path]:
        """Discover all relevant code files"""
        files = []
        
        for root, dirs, filenames in os.walk(self.root_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            
            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix in self.code_extensions:
                    files.append(file_path)
        
        return files
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single file and extract code entities"""
        try:
            relative_path = str(file_path.relative_to(self.root_path))
            
            # Create file node
            file_node = CodeNode(
                name=file_path.name,
                type='file',
                file_path=relative_path
            )
            self.nodes[relative_path] = file_node
            
            if file_path.suffix == '.py':
                self._analyze_python_file(file_path, relative_path)
            elif file_path.suffix in {'.yaml', '.yml'}:
                self._analyze_yaml_file(file_path, relative_path)
            elif file_path.suffix == '.json':
                self._analyze_json_file(file_path, relative_path)
            elif file_path.suffix == '.md':
                self._analyze_markdown_file(file_path, relative_path)
                
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
    
    def _analyze_python_file(self, file_path: Path, relative_path: str):
        """Analyze Python file using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._extract_import(node, relative_path)
                elif isinstance(node, ast.FunctionDef):
                    self._extract_function(node, relative_path)
                elif isinstance(node, ast.ClassDef):
                    self._extract_class(node, relative_path)
                elif isinstance(node, ast.Call):
                    self._extract_function_call(node, relative_path)
            
            # Calculate complexity
            complexity = self._calculate_complexity(tree)
            self.nodes[relative_path].complexity_score = complexity
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            self.broken_imports.append((relative_path, "SYNTAX_ERROR", str(e)))
        except Exception as e:
            self.logger.error(f"Error parsing Python file {file_path}: {e}")
    
    def _extract_import(self, node: ast.AST, file_path: str):
        """Extract import statements"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_name = alias.name
                self._add_relationship(file_path, import_name, 'imports')
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                import_name = f"{module}.{alias.name}" if module else alias.name
                self._add_relationship(file_path, import_name, 'imports')
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str):
        """Extract function definitions"""
        func_name = f"{file_path}::{node.name}"
        func_node = CodeNode(
            name=node.name,
            type='function',
            file_path=file_path,
            line_number=node.lineno
        )
        self.nodes[func_name] = func_node
        self._add_relationship(file_path, func_name, 'defines')
    
    def _extract_class(self, node: ast.ClassDef, file_path: str):
        """Extract class definitions"""
        class_name = f"{file_path}::{node.name}"
        class_node = CodeNode(
            name=node.name,
            type='class',
            file_path=file_path,
            line_number=node.lineno
        )
        self.nodes[class_name] = class_node
        self._add_relationship(file_path, class_name, 'defines')
        
        # Extract inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                self._add_relationship(class_name, base.id, 'inherits')
    
    def _extract_function_call(self, node: ast.Call, file_path: str):
        """Extract function calls"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self._add_relationship(file_path, func_name, 'calls')
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                self._add_relationship(file_path, f"{obj_name}.{method_name}", 'calls')
    
    def _analyze_yaml_file(self, file_path: Path, relative_path: str):
        """Analyze YAML configuration files"""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Look for script references or imports
            self._extract_yaml_references(data, relative_path)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing YAML file {file_path}: {e}")
    
    def _analyze_json_file(self, file_path: Path, relative_path: str):
        """Analyze JSON configuration files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Look for script references
            self._extract_json_references(data, relative_path)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing JSON file {file_path}: {e}")
    
    def _analyze_markdown_file(self, file_path: Path, relative_path: str):
        """Analyze Markdown files for code references"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for code blocks and file references
            self._extract_markdown_references(content, relative_path)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing Markdown file {file_path}: {e}")
    
    def _extract_yaml_references(self, data: Any, file_path: str, prefix: str = ""):
        """Extract references from YAML data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and (value.endswith('.py') or value.endswith('.yaml')):
                    self._add_relationship(file_path, value, 'references')
                elif isinstance(value, (dict, list)):
                    self._extract_yaml_references(value, file_path, f"{prefix}.{key}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_yaml_references(item, file_path, prefix)
    
    def _extract_json_references(self, data: Any, file_path: str):
        """Extract references from JSON data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and (value.endswith('.py') or 'script' in key.lower()):
                    self._add_relationship(file_path, value, 'references')
                elif isinstance(value, (dict, list)):
                    self._extract_json_references(value, file_path)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_json_references(item, file_path)
    
    def _extract_markdown_references(self, content: str, file_path: str):
        """Extract references from Markdown content"""
        # Look for file references in code blocks and links
        patterns = [
            r'`([^`]+\.py)`',  # `filename.py`
            r'\[([^\]]+\.py)\]',  # [filename.py]
            r'```python[^`]*?([a-zA-Z_][a-zA-Z0-9_]*\.py)',  # Python code blocks
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                self._add_relationship(file_path, match, 'references')
    
    def _add_relationship(self, source: str, target: str, rel_type: str):
        """Add a relationship between code entities"""
        relationship = CodeRelationship(
            source=source,
            target=target,
            relationship_type=rel_type
        )
        self.relationships.append(relationship)
        
        # Update node dependencies
        if source in self.nodes:
            self.nodes[source].dependencies.append(target)
        if target in self.nodes:
            self.nodes[target].dependents.append(source)
            self.nodes[target].usage_count += 1
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate code complexity score"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += len(node.args.args)  # Parameter count
        
        return complexity
    
    def _build_relationship_graph(self):
        """Build NetworkX graph from relationships"""
        for rel in self.relationships:
            self.graph.add_edge(
                rel.source, 
                rel.target, 
                type=rel.relationship_type,
                strength=rel.strength,
                broken=rel.is_broken
            )
    
    def _identify_isolated_components(self):
        """Identify isolated files and components"""
        for node_id, node in self.nodes.items():
            if node.type == 'file':
                # Check if file has any connections
                in_degree = self.graph.in_degree(node_id)
                out_degree = self.graph.out_degree(node_id)
                
                if in_degree == 0 and out_degree == 0:
                    node.is_isolated = True
                    self.isolated_files.append(node_id)
    
    def _identify_broken_imports(self):
        """Identify broken import statements"""
        for rel in self.relationships:
            if rel.relationship_type == 'imports':
                # Check if imported module exists
                imported_module = rel.target
                
                # Try to resolve the import
                if not self._can_resolve_import(imported_module, rel.source):
                    rel.is_broken = True
                    rel.error_message = f"Cannot resolve import: {imported_module}"
                    self.broken_imports.append((rel.source, imported_module, rel.error_message))
    
    def _can_resolve_import(self, module_name: str, source_file: str) -> bool:
        """Check if an import can be resolved"""
        try:
            # Check if it's a local file
            local_path = (self.root_path / f"{module_name}.py")
            if local_path.exists():
                return True
            
            # Check if it's a standard library or installed package
            spec = importlib.util.find_spec(module_name.split('.')[0])
            return spec is not None
            
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
    
    def _identify_unused_components(self):
        """Identify unused functions and classes"""
        for node_id, node in self.nodes.items():
            if node.type in ['function', 'class'] and node.usage_count == 0:
                # Check if it's not a main function or special method
                if not (node.name.startswith('__') or node.name == 'main'):
                    self.unused_functions.append(node_id)
    
    def _analyze_integration_health(self):
        """Analyze integration health across the codebase"""
        for node_id, node in self.nodes.items():
            health_score = 1.0
            issues = []
            
            # Check for broken dependencies
            broken_deps = 0
            for dep in node.dependencies:
                if any(rel.is_broken for rel in self.relationships 
                       if rel.source == node_id and rel.target == dep):
                    broken_deps += 1
            
            if node.dependencies:
                health_score -= (broken_deps / len(node.dependencies)) * 0.5
                if broken_deps > 0:
                    issues.append(f"{broken_deps} broken dependencies")
            
            # Check isolation
            if node.is_isolated and node.type == 'file':
                health_score -= 0.3
                issues.append("Isolated file")
            
            # Check complexity
            if node.complexity_score > 20:  # High complexity threshold
                health_score -= 0.2
                issues.append("High complexity")
            
            node.integration_health = max(0.0, health_score)
            
            if health_score < 0.8 and issues:
                self.integration_issues.append({
                    'node': node_id,
                    'health_score': health_score,
                    'issues': issues
                })
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        return {
            'summary': {
                'total_files': len([n for n in self.nodes.values() if n.type == 'file']),
                'total_functions': len([n for n in self.nodes.values() if n.type == 'function']),
                'total_classes': len([n for n in self.nodes.values() if n.type == 'class']),
                'total_relationships': len(self.relationships),
                'isolated_files': len(self.isolated_files),
                'broken_imports': len(self.broken_imports),
                'unused_components': len(self.unused_functions),
                'integration_issues': len(self.integration_issues)
            },
            'health_metrics': {
                'overall_health': self._calculate_overall_health(),
                'connectivity_score': self._calculate_connectivity_score(),
                'maintainability_score': self._calculate_maintainability_score()
            },
            'detailed_issues': {
                'isolated_files': self.isolated_files,
                'broken_imports': self.broken_imports,
                'unused_components': self.unused_functions,
                'integration_issues': self.integration_issues
            },
            'recommendations': self._generate_recommendations(),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall codebase health score"""
        if not self.nodes:
            return 0.0
        
        total_health = sum(node.integration_health for node in self.nodes.values())
        return total_health / len(self.nodes)
    
    def _calculate_connectivity_score(self) -> float:
        """Calculate connectivity score"""
        total_files = len([n for n in self.nodes.values() if n.type == 'file'])
        if total_files == 0:
            return 0.0
        
        connected_files = total_files - len(self.isolated_files)
        return connected_files / total_files
    
    def _calculate_maintainability_score(self) -> float:
        """Calculate maintainability score"""
        complexity_penalty = sum(min(node.complexity_score / 50, 1.0) 
                                for node in self.nodes.values()) / len(self.nodes)
        broken_penalty = len(self.broken_imports) / max(1, len(self.relationships))
        
        return max(0.0, 1.0 - complexity_penalty - broken_penalty)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if self.isolated_files:
            recommendations.append(f"Connect {len(self.isolated_files)} isolated files to the main codebase")
        
        if self.broken_imports:
            recommendations.append(f"Fix {len(self.broken_imports)} broken import statements")
        
        if self.unused_functions:
            recommendations.append(f"Remove or utilize {len(self.unused_functions)} unused components")
        
        if self.integration_issues:
            critical_issues = [i for i in self.integration_issues if i['health_score'] < 0.5]
            if critical_issues:
                recommendations.append(f"Address {len(critical_issues)} critical integration issues")
        
        # Component-specific recommendations
        auto_rebuilder_files = [f for f in self.nodes.keys() if 'auto_rebuilder' in f.lower()]
        if auto_rebuilder_files:
            recommendations.append("Ensure auto_rebuilder components are properly integrated with TheWand outputs")
        
        monster_scanner_files = [f for f in self.nodes.keys() if 'monster' in f.lower() or 'scanner' in f.lower()]
        if monster_scanner_files:
            recommendations.append("Verify monster scanner ML file processing integration")
        
        return recommendations

class CodebaseVisualizer:
    """Advanced visualization system for codebase relationships"""
    
    def __init__(self, analyzer: CodebaseAnalyzer):
        self.analyzer = analyzer
        self.graph = analyzer.graph
        
    def create_comprehensive_diagram(self, output_file: str = "codebase_diagram.png"):
        """Create comprehensive codebase relationship diagram"""
        plt.figure(figsize=(20, 16))
        
        # Calculate layout
        pos = self._calculate_layout()
        
        # Draw different node types with different colors
        self._draw_nodes(pos)
        
        # Draw relationships with different edge styles
        self._draw_relationships(pos)
        
        # Add legend
        self._add_legend()
        
        # Add title and metadata
        self._add_title_and_metadata()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive codebase diagram saved to: {output_file}")
    
    def _calculate_layout(self) -> Dict[str, Tuple[float, float]]:
        """Calculate optimal layout for visualization"""
        # Use hierarchical layout for better organization
        try:
            # Try to use graphviz layout if available
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        return pos
    
    def _draw_nodes(self, pos: Dict[str, Tuple[float, float]]):
        """Draw nodes with different styles based on type and health"""
        node_colors = {
            'file': '#87CEEB',      # Sky blue
            'function': '#98FB98',   # Pale green
            'class': '#DDA0DD',      # Plum
            'import': '#F0E68C'      # Khaki
        }
        
        for node_type in node_colors:
            nodes_of_type = [n for n, data in self.graph.nodes(data=True) 
                           if self.analyzer.nodes.get(n, CodeNode("", "", "")).type == node_type]
            
            if nodes_of_type:
                # Color based on health
                node_health = [self.analyzer.nodes.get(n, CodeNode("", "", "")).integration_health 
                             for n in nodes_of_type]
                
                nx.draw_networkx_nodes(
                    self.graph, pos, 
                    nodelist=nodes_of_type,
                    node_color=node_health,
                    node_size=[300 + self.analyzer.nodes.get(n, CodeNode("", "", "")).usage_count * 50 
                             for n in nodes_of_type],
                    cmap=plt.cm.RdYlGn,
                    vmin=0, vmax=1,
                    alpha=0.8
                )
        
        # Highlight isolated nodes
        isolated_nodes = [n for n in self.analyzer.isolated_files if n in self.graph.nodes()]
        if isolated_nodes:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=isolated_nodes,
                node_color='red',
                node_size=400,
                alpha=0.6,
                edgecolors='black',
                linewidths=2
            )
    
    def _draw_relationships(self, pos: Dict[str, Tuple[float, float]]):
        """Draw relationships with different styles"""
        edge_styles = {
            'imports': {'color': 'blue', 'style': '-', 'width': 2},
            'calls': {'color': 'green', 'style': '-', 'width': 1},
            'inherits': {'color': 'purple', 'style': '--', 'width': 2},
            'defines': {'color': 'orange', 'style': ':', 'width': 1},
            'references': {'color': 'gray', 'style': '-', 'width': 0.5}
        }
        
        for rel_type, style in edge_styles.items():
            edges_of_type = [(u, v) for u, v, data in self.graph.edges(data=True) 
                           if data.get('type') == rel_type]
            
            if edges_of_type:
                # Identify broken edges
                broken_edges = [(u, v) for u, v in edges_of_type 
                              if self.graph[u][v].get('broken', False)]
                good_edges = [(u, v) for u, v in edges_of_type 
                            if not self.graph[u][v].get('broken', False)]
                
                # Draw good edges
                if good_edges:
                    nx.draw_networkx_edges(
                        self.graph, pos,
                        edgelist=good_edges,
                        edge_color=style['color'],
                        style=style['style'],
                        width=style['width'],
                        alpha=0.7,
                        arrows=True,
                        arrowsize=20
                    )
                
                # Draw broken edges in red
                if broken_edges:
                    nx.draw_networkx_edges(
                        self.graph, pos,
                        edgelist=broken_edges,
                        edge_color='red',
                        style=style['style'],
                        width=style['width'] + 1,
                        alpha=0.8,
                        arrows=True,
                        arrowsize=20
                    )
    
    def _add_legend(self):
        """Add comprehensive legend"""
        legend_elements = [
            mpatches.Circle((0, 0), 0.1, facecolor='#87CEEB', label='Files'),
            mpatches.Circle((0, 0), 0.1, facecolor='#98FB98', label='Functions'),
            mpatches.Circle((0, 0), 0.1, facecolor='#DDA0DD', label='Classes'),
            mpatches.Circle((0, 0), 0.1, facecolor='red', alpha=0.6, label='Isolated'),
            mpatches.Patch(color='blue', label='Imports'),
            mpatches.Patch(color='green', label='Function Calls'),
            mpatches.Patch(color='purple', label='Inheritance'),
            mpatches.Patch(color='orange', label='Definitions'),
            mpatches.Patch(color='red', label='Broken Connections')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    def _add_title_and_metadata(self):
        """Add title and metadata to the diagram"""
        analysis_results = self.analyzer._generate_analysis_report()
        
        title = f"Codebase Relationship Diagram\n"
        title += f"Files: {analysis_results['summary']['total_files']} | "
        title += f"Health: {analysis_results['health_metrics']['overall_health']:.2f} | "
        title += f"Issues: {analysis_results['summary']['integration_issues']}"
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add metadata box
        metadata = (
            f"Connectivity: {analysis_results['health_metrics']['connectivity_score']:.2f}\n"
            f"Maintainability: {analysis_results['health_metrics']['maintainability_score']:.2f}\n"
            f"Broken Imports: {analysis_results['summary']['broken_imports']}\n"
            f"Isolated Files: {analysis_results['summary']['isolated_files']}"
        )
        
        plt.figtext(0.02, 0.02, metadata, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

# Main execution and integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and visualize codebase relationships")
    parser.add_argument("path", help="Path to codebase root directory")
    parser.add_argument("--output", "-o", default="codebase_analysis_report.json", 
                       help="Output file for analysis report")
    parser.add_argument("--diagram", "-d", default="codebase_diagram.png",
                       help="Output file for diagram")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run analysis
    analyzer = CodebaseAnalyzer(args.path)
    results = analyzer.analyze_codebase()
    
    # Save analysis report
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCodebase Analysis Results:")
    print(f"=" * 50)
    print(f"Total Files: {results['summary']['total_files']}")
    print(f"Total Functions: {results['summary']['total_functions']}")
    print(f"Total Classes: {results['summary']['total_classes']}")
    print(f"Overall Health: {results['health_metrics']['overall_health']:.2f}")
    print(f"Connectivity Score: {results['health_metrics']['connectivity_score']:.2f}")
    print(f"Maintainability Score: {results['health_metrics']['maintainability_score']:.2f}")
    print(f"\nIssues Found:")
    print(f"- Isolated Files: {results['summary']['isolated_files']}")
    print(f"- Broken Imports: {results['summary']['broken_imports']}")
    print(f"- Unused Components: {results['summary']['unused_components']}")
    print(f"- Integration Issues: {results['summary']['integration_issues']}")
    
    if results['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Create visualization
    visualizer = CodebaseVisualizer(analyzer)
    visualizer.create_comprehensive_diagram(args.diagram)
    
    print(f"\nAnalysis report saved to: {args.output}")
    print(f"Diagram saved to: {args.diagram}")
