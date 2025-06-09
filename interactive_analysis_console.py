#!/usr/bin/env python3
"""
Interactive Codebase Analysis and Debugging Console
Provides real-time analysis of your comprehensive visualization system
"""

import json
import os
import re
from typing import Dict, List, Any
from datetime import datetime

class InteractiveAnalysisConsole:
    """Interactive console for codebase analysis and debugging"""
    
    def __init__(self):
        self.analysis_results = self.load_analysis_results()
        self.workspace_files = self.scan_workspace()
        
    def load_analysis_results(self) -> Dict[str, Any]:
        """Load existing analysis results"""
        try:
            with open('codebase_analysis_results.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"Error loading analysis: {e}")
            return {}
    
    def scan_workspace(self) -> List[str]:
        """Scan workspace for Python files"""
        python_files = []
        for root, dirs, files in os.walk('.'):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def display_system_overview(self):
        """Display comprehensive system overview"""
        print("🚀 INTERACTIVE CODEBASE ANALYSIS CONSOLE")
        print("=" * 80)
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Workspace: {os.getcwd()}")
        print(f"🔍 Python Files Found: {len(self.workspace_files)}")
        
        if self.analysis_results:
            health = self.analysis_results['health_metrics']
            summary = self.analysis_results['summary']
            
            print(f"\n🎯 SYSTEM HEALTH STATUS")
            print("=" * 30)
            health_score = health['overall_health']
            health_emoji = "🟢" if health_score > 0.9 else "🟡" if health_score > 0.7 else "🔴"
            print(f"Overall Health: {health_score:.1%} {health_emoji}")
            print(f"Connectivity: {health['connectivity_score']:.1%}")
            print(f"Maintainability: {health['maintainability_score']:.1%}")
            
            print(f"\n📊 CODEBASE METRICS")
            print("-" * 20)
            print(f"Files: {summary['total_files']:,}")
            print(f"Functions: {summary['total_functions']:,}")
            print(f"Classes: {summary['total_classes']:,}")
            print(f"Relationships: {summary['total_relationships']:,}")
            
            print(f"\n⚠️  ISSUES DETECTED")
            print("-" * 20)
            print(f"Isolated Files: {summary['isolated_files']} {'✅' if summary['isolated_files'] == 0 else '⚠️'}")
            print(f"Broken Imports: {summary['broken_imports']}")
            print(f"Integration Issues: {summary['integration_issues']}")
        else:
            print("\n⚠️ No analysis results found - system needs initial analysis")
    
    def analyze_tool_chain_integration(self):
        """Analyze tool chain integration status"""
        print(f"\n🔗 TOOL CHAIN INTEGRATION ANALYSIS")
        print("=" * 50)
        
        # Define the tool chain components
        tool_chain = {
            'AE-Lang Interpreter': {
                'files': ['AE-Lang_interp.py', 'enhanced_AE_Lang_interp.py'],
                'description': 'Parses and executes AE-Lang scripts',
                'next_step': 'Monster Scanner'
            },
            'Monster Scanner': {
                'files': ['monster_scanner.py', 'ml_processing.py'],
                'description': 'ML processing and barcode generation',
                'next_step': 'TheWand'
            },
            'TheWand': {
                'files': ['ae_wand_integration_bridge.py'],
                'description': 'Project generation and LLM integration',
                'next_step': 'Auto-Rebuilder'
            },
            'Auto-Rebuilder': {
                'files': ['auto_rebuilder.py', 'auto_rebuilder_adapter.py'],
                'description': 'Error detection and code fixing',
                'next_step': 'Pygame Integration'
            },
            'Pygame Integration': {
                'files': ['AE_equations_sim - pygame.py', 'auto_rebuilder_pygame_adapter.py'],
                'description': 'Real-time visualization and monitoring',
                'next_step': None
            }
        }
        
        print("Integration Flow:")
        print("AE-Lang → Monster Scanner → TheWand → Auto-Rebuilder → Pygame")
        print()
        
        for tool_name, info in tool_chain.items():
            existing_files = [f for f in info['files'] if os.path.exists(f)]
            status = "✅" if existing_files else "❌"
            
            print(f"{status} {tool_name}")
            print(f"   📝 {info['description']}")
            
            if existing_files:
                for file in existing_files:
                    size = os.path.getsize(file)
                    print(f"   📄 {file} ({size:,} bytes)")
            else:
                print(f"   ⚠️  Missing files: {', '.join(info['files'])}")
            
            if info['next_step']:
                print(f"   ➡️  Next: {info['next_step']}")
            print()
    
    def analyze_broken_imports(self):
        """Analyze broken imports in detail"""
        if not self.analysis_results or 'detailed_issues' not in self.analysis_results:
            print("❌ No detailed analysis results available")
            return
            
        broken_imports = self.analysis_results['detailed_issues']['broken_imports']
        
        print(f"\n💔 BROKEN IMPORTS ANALYSIS")
        print("=" * 40)
        print(f"Total Broken Imports: {len(broken_imports)}")
        
        if not broken_imports:
            print("✅ No broken imports detected!")
            return
        
        # Group by import type
        import_categories = {
            'External Libraries': [],
            'Internal Modules': [],
            'System Modules': []
        }
        
        for file, import_name, error in broken_imports[:20]:  # Show first 20
            if any(lib in import_name.lower() for lib in ['cv2', 'librosa', 'pygame', 'websocket', 'pyttsx3']):
                import_categories['External Libraries'].append((file, import_name, error))
            elif '.' in import_name and not import_name.startswith('sys') and not import_name.startswith('os'):
                import_categories['Internal Modules'].append((file, import_name, error))
            else:
                import_categories['System Modules'].append((file, import_name, error))
        
        for category, imports in import_categories.items():
            if imports:
                print(f"\n🔧 {category} ({len(imports)} issues)")
                for file, import_name, error in imports[:5]:  # Show first 5 per category
                    print(f"   📄 {os.path.basename(file)}: {import_name}")
                if len(imports) > 5:
                    print(f"   ... and {len(imports)-5} more")
    
    def analyze_analysis_tools(self):
        """Analyze the analysis tools themselves"""
        print(f"\n🔍 ANALYSIS TOOLS STATUS")
        print("=" * 40)
        
        analysis_tools = {
            'codebase_relationship_analyzer.py': 'Comprehensive relationship mapping and visualization',
            'tool_chain_analyzer.py': 'Specialized tool chain integration analysis',
            'codebase_debug_dashboard.py': 'Interactive debugging interface with GUI',
            'analysis_dashboard.py': 'Summary dashboard for analysis results',
            'live_status_display.py': 'Real-time status monitoring'
        }
        
        for tool, description in analysis_tools.items():
            if os.path.exists(tool):
                size = os.path.getsize(tool)
                lines = self.count_lines(tool)
                print(f"✅ {tool}")
                print(f"   📊 {size:,} bytes, ~{lines:,} lines")
                print(f"   📝 {description}")
            else:
                print(f"❌ {tool} - Missing")
            print()
    
    def count_lines(self, file_path: str) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    def analyze_integration_gaps(self):
        """Analyze potential integration gaps"""
        print(f"\n🔍 INTEGRATION GAP ANALYSIS")
        print("=" * 40)
        
        # Check for critical integration files
        critical_integrations = [
            ('AE-Lang → Auto-Rebuilder', 'AE-Lang_interp.py', 'auto_rebuilder.py'),
            ('Auto-Rebuilder → Pygame', 'auto_rebuilder_pygame_adapter.py', 'AE_equations_sim - pygame.py'),
            ('TheWand → Auto-Rebuilder', 'ae_wand_integration_bridge.py', 'auto_rebuilder.py'),
            ('Consciousness Integration', 'ae_consciousness_integration.py', 'ae_core_consciousness.py')
        ]
        
        gaps_found = []
        
        for integration_name, file1, file2 in critical_integrations:
            file1_exists = os.path.exists(file1)
            file2_exists = os.path.exists(file2)
            
            if file1_exists and file2_exists:
                print(f"✅ {integration_name}: Both components present")
                
                # Check for cross-references
                try:
                    with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
                        content1 = f.read().lower()
                    
                    file2_name = os.path.basename(file2).replace('.py', '').replace(' - ', '_')
                    if file2_name.lower() in content1:
                        print(f"   🔗 Cross-reference detected")
                    else:
                        print(f"   ⚠️  No cross-reference found")
                        gaps_found.append(integration_name)
                        
                except Exception:
                    print(f"   ⚠️  Could not analyze cross-references")
                    
            else:
                status1 = "✅" if file1_exists else "❌"
                status2 = "✅" if file2_exists else "❌"
                print(f"{status1}{status2} {integration_name}: Missing components")
                gaps_found.append(integration_name)
        
        if gaps_found:
            print(f"\n⚠️  Integration gaps detected in: {', '.join(gaps_found)}")
        else:
            print(f"\n✅ No critical integration gaps detected!")
    
    def provide_debugging_recommendations(self):
        """Provide actionable debugging recommendations"""
        print(f"\n💡 DEBUGGING RECOMMENDATIONS")
        print("=" * 40)
        
        recommendations = []
        
        # Based on analysis results
        if self.analysis_results:
            summary = self.analysis_results['summary']
            
            if summary.get('broken_imports', 0) > 0:
                recommendations.append("🔧 Install missing dependencies with pip install commands")
            
            if summary.get('integration_issues', 0) > 0:
                recommendations.append("🔗 Review integration health for critical components")
        
        # Universal recommendations
        recommendations.extend([
            "🎨 Generate network visualization diagrams",
            "🔍 Use interactive debug dashboard for deep analysis",
            "📊 Monitor system health metrics regularly",
            "🔄 Run tool chain verification tests",
            "🚀 Prepare for production deployment"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis suite"""
        self.display_system_overview()
        self.analyze_tool_chain_integration()
        self.analyze_broken_imports()
        self.analyze_analysis_tools()
        self.analyze_integration_gaps()
        self.provide_debugging_recommendations()
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE")
        print("=" * 50)
        
        if self.analysis_results:
            health_score = self.analysis_results['health_metrics']['overall_health']
            if health_score > 0.95:
                print("🟢 SYSTEM STATUS: EXCELLENT - All systems operational")
            elif health_score > 0.8:
                print("🟡 SYSTEM STATUS: GOOD - Minor issues detected")
            else:
                print("🔴 SYSTEM STATUS: NEEDS ATTENTION - Multiple issues found")
        
        print("\n📋 NEXT STEPS:")
        print("1. Address any broken imports by installing dependencies")
        print("2. Run visualization tools to create network diagrams")
        print("3. Use debug dashboard for interactive analysis")
        print("4. Monitor integration health continuously")

def main():
    """Main execution function"""
    console = InteractiveAnalysisConsole()
    console.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
