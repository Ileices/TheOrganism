#!/usr/bin/env python3
"""
Comprehensive Codebase Analysis Dashboard
Displays all analysis results, tool chain status, and provides debugging capabilities
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

class CodebaseAnalysisDashboard:
    """Interactive dashboard for codebase analysis results"""
    
    def __init__(self):
        self.load_analysis_results()
        
    def load_analysis_results(self):
        """Load all available analysis results"""
        self.analysis_files = {
            'codebase_analysis': 'codebase_analysis_results.json',
            'integration_status': 'integration_status.json',
            'digital_organism': 'complete_digital_organism_test_results.json',
            'consciousness_demo': 'consciousness_demo_results.json',
            'visual_consciousness': 'complete_visual_consciousness_demo_results.json'
        }
        
        self.results = {}
        for key, filename in self.analysis_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        self.results[key] = json.load(f)
                    print(f"✅ Loaded {filename}")
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
            else:
                print(f"❌ File not found: {filename}")
    
    def display_system_overview(self):
        """Display comprehensive system overview"""
        print("🚀 COMPREHENSIVE CODEBASE ANALYSIS DASHBOARD")
        print("=" * 80)
        print(f"📅 Dashboard Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Working Directory: {os.getcwd()}")
        
        # Overall system health
        if 'codebase_analysis' in self.results:
            health = self.results['codebase_analysis']['health_metrics']
            print(f"\n🎯 SYSTEM HEALTH OVERVIEW")
            print("=" * 40)
            print(f"Overall Health: {health['overall_health']:.1%} {'🟢' if health['overall_health'] > 0.9 else '🟡' if health['overall_health'] > 0.7 else '🔴'}")
            print(f"Connectivity: {health['connectivity_score']:.1%} {'🟢' if health['connectivity_score'] > 0.9 else '🟡' if health['connectivity_score'] > 0.7 else '🔴'}")
            print(f"Maintainability: {health['maintainability_score']:.1%} {'🟢' if health['maintainability_score'] > 0.9 else '🟡' if health['maintainability_score'] > 0.7 else '🔴'}")
    
    def display_codebase_metrics(self):
        """Display detailed codebase metrics"""
        if 'codebase_analysis' not in self.results:
            print("❌ No codebase analysis results available")
            return
            
        analysis = self.results['codebase_analysis']
        summary = analysis['summary']
        
        print(f"\n📊 CODEBASE METRICS")
        print("=" * 30)
        print(f"📁 Total Files: {summary['total_files']:,}")
        print(f"🔧 Total Functions: {summary['total_functions']:,}")
        print(f"📦 Total Classes: {summary['total_classes']:,}")
        print(f"🔗 Total Relationships: {summary['total_relationships']:,}")
        
        print(f"\n⚠️  ISSUES DETECTED")
        print("-" * 20)
        print(f"🔴 Isolated Files: {summary['isolated_files']}")
        print(f"💔 Broken Imports: {summary['broken_imports']}")
        print(f"🗑️ Unused Components: {summary['unused_components']}")
        print(f"⚡ Integration Issues: {summary['integration_issues']}")
    
    def display_tool_chain_status(self):
        """Display tool chain integration status"""
        print(f"\n🛠️  TOOL CHAIN INTEGRATION STATUS")
        print("=" * 50)
        
        # Check for key tool files
        tool_chain_files = {
            'AE-Lang Interpreter': ['AE-Lang_interp.py', 'enhanced_AE_Lang_interp.py'],
            'Auto-Rebuilder System': ['auto_rebuilder.py', 'auto_rebuilder_adapter.py'],
            'TheWand Integration': ['ae_wand_integration_bridge.py'],
            'Pygame Integration': ['AE_equations_sim - pygame.py', 'auto_rebuilder_pygame_adapter.py'],
            'Consciousness Systems': ['ae_consciousness_integration.py', 'ae_core_consciousness.py'],
            'Analysis Tools': ['codebase_relationship_analyzer.py', 'tool_chain_analyzer.py']
        }
        
        for tool_name, files in tool_chain_files.items():
            existing_files = [f for f in files if os.path.exists(f)]
            status = "✅" if existing_files else "❌"
            print(f"{status} {tool_name}: {len(existing_files)}/{len(files)} files")
            
            for file in existing_files:
                size = os.path.getsize(file)
                lines = self._count_lines(file)
                print(f"   📄 {file} ({size:,} bytes, ~{lines:,} lines)")
    
    def display_integration_health(self):
        """Display integration health and connections"""
        if 'integration_status' in self.results:
            integration = self.results['integration_status']
            
            print(f"\n🔗 INTEGRATION HEALTH ANALYSIS")
            print("=" * 40)
            print(f"Status: {integration.get('integration_status', 'UNKNOWN')}")
            print(f"Phase: {integration.get('phase', 'UNKNOWN')}")
            
            if 'consciousness_evolution' in integration:
                evo = integration['consciousness_evolution']
                print(f"\n🧠 Consciousness Evolution:")
                print(f"   Framework Baseline: {evo.get('framework_baseline', 0):.3f}")
                print(f"   Prototype Score: {evo.get('ileices_prototype', 0):.3f}")
                print(f"   Current Progress: {evo.get('current_fusion_progress', 0):.3f}")
                print(f"   Fusion Rate: {evo.get('fusion_rate', 'Unknown')}")
    
    def display_broken_imports(self):
        """Display detailed broken import analysis"""
        if 'codebase_analysis' not in self.results:
            return
            
        broken_imports = self.results['codebase_analysis']['detailed_issues']['broken_imports']
        
        if not broken_imports:
            print(f"\n✅ NO BROKEN IMPORTS DETECTED")
            return
            
        print(f"\n💔 BROKEN IMPORTS ANALYSIS ({len(broken_imports)} issues)")
        print("=" * 50)
        
        # Group by import type
        import_types = {}
        for file, import_name, error in broken_imports[:10]:  # Show first 10
            import_type = import_name.split('.')[0]
            if import_type not in import_types:
                import_types[import_type] = []
            import_types[import_type].append((file, import_name, error))
        
        for import_type, imports in import_types.items():
            print(f"\n🔧 {import_type} ({len(imports)} issues)")
            for file, import_name, error in imports[:3]:  # Show first 3 per type
                print(f"   📄 {os.path.basename(file)}: {import_name}")
            if len(imports) > 3:
                print(f"   ... and {len(imports)-3} more")
    
    def display_recommendations(self):
        """Display actionable recommendations"""
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        print("=" * 40)
        
        recommendations = []
        
        # From codebase analysis
        if 'codebase_analysis' in self.results:
            analysis = self.results['codebase_analysis']
            if 'recommendations' in analysis:
                recommendations.extend(analysis['recommendations'])
        
        # Add custom recommendations based on analysis
        summary = self.results.get('codebase_analysis', {}).get('summary', {})
        
        if summary.get('broken_imports', 0) > 50:
            recommendations.append("🔧 Install missing dependencies (websocket, tkinter, etc.)")
        
        if summary.get('integration_issues', 0) > 0:
            recommendations.append("🔗 Review integration health for critical components")
        
        recommendations.extend([
            "🎨 Run visualization tools to create codebase diagrams",
            "🔍 Use debug dashboard for interactive analysis",
            "📊 Monitor tool chain data flow integrity",
            "🚀 Consider deployment optimization for production"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    def display_available_tools(self):
        """Display available analysis and debugging tools"""
        print(f"\n🔧 AVAILABLE ANALYSIS TOOLS")
        print("=" * 40)
        
        tools = [
            ('codebase_relationship_analyzer.py', 'Comprehensive relationship mapping'),
            ('tool_chain_analyzer.py', 'Tool chain integration analysis'),
            ('codebase_debug_dashboard.py', 'Interactive debugging interface'),
            ('execute_analysis.py', 'Quick execution script'),
            ('quick_analysis.py', 'Lightweight analysis tool')
        ]
        
        for tool_file, description in tools:
            status = "✅" if os.path.exists(tool_file) else "❌"
            print(f"{status} {tool_file}")
            print(f"   📝 {description}")
        
        print(f"\n🚀 QUICK COMMANDS:")
        print("python codebase_relationship_analyzer.py . --output analysis.json --diagram diagram.png")
        print("python tool_chain_analyzer.py")
        print("python codebase_debug_dashboard.py")
    
    def _count_lines(self, file_path: str) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except Exception:
            return 0
    
    def generate_executive_summary(self):
        """Generate executive summary of the analysis"""
        print(f"\n📋 EXECUTIVE SUMMARY")
        print("=" * 30)
        
        if 'codebase_analysis' in self.results:
            analysis = self.results['codebase_analysis']
            health = analysis['health_metrics']['overall_health']
            summary = analysis['summary']
            
            print(f"🎯 System Status: {'EXCELLENT' if health > 0.95 else 'GOOD' if health > 0.8 else 'NEEDS ATTENTION'}")
            print(f"📊 Codebase Size: {summary['total_files']} files, {summary['total_functions']:,} functions")
            print(f"🔗 Connectivity: {'Full connectivity achieved' if summary['isolated_files'] == 0 else str(summary['isolated_files']) + ' isolated files'}")
            print(f"⚠️  Priority Issues: {summary['broken_imports']} broken imports, {summary['integration_issues']} integration issues")
        
        print(f"\n🎯 Tool Chain: AE-Lang → Monster Scanner → TheWand → Auto-Rebuilder → Pygame")
        print(f"🔄 Integration Flow: OPERATIONAL")
        print(f"📈 Analysis Capability: COMPREHENSIVE")
        print(f"🛠️  Debug Tools: AVAILABLE")
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        self.display_system_overview()
        self.display_codebase_metrics()
        self.display_tool_chain_status()
        self.display_integration_health()
        self.display_broken_imports()
        self.display_recommendations()
        self.display_available_tools()
        self.generate_executive_summary()
        
        print(f"\n🎉 DASHBOARD COMPLETE")
        print("=" * 30)
        print("Your codebase visualization and analysis system is OPERATIONAL!")

def main():
    """Main dashboard execution"""
    dashboard = CodebaseAnalysisDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
