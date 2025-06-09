#!/usr/bin/env python3
"""
Complete Codebase Analysis Runner
Executes all analysis tools and provides comprehensive debugging information
"""

import os
import sys
import json
import traceback
from datetime import datetime

def run_codebase_analysis():
    """Run the comprehensive codebase relationship analyzer"""
    print("🔍 RUNNING CODEBASE RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    try:
        from codebase_relationship_analyzer import CodebaseAnalyzer, CodebaseVisualizer
        
        # Initialize analyzer
        analyzer = CodebaseAnalyzer('.')
        print("✅ Analyzer initialized successfully")
        
        # Run analysis
        print("📊 Analyzing codebase relationships...")
        results = analyzer.analyze_codebase()
        
        # Save results
        with open('codebase_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n📈 ANALYSIS SUMMARY")
        print(f"Total Files: {results['summary']['total_files']}")
        print(f"Total Functions: {results['summary']['total_functions']}")
        print(f"Total Classes: {results['summary']['total_classes']}")
        print(f"Overall Health: {results['health_metrics']['overall_health']:.2f}")
        print(f"Connectivity Score: {results['health_metrics']['connectivity_score']:.2f}")
        print(f"Maintainability Score: {results['health_metrics']['maintainability_score']:.2f}")
        
        print(f"\n⚠️  ISSUES FOUND:")
        print(f"- Isolated Files: {results['summary']['isolated_files']}")
        print(f"- Broken Imports: {results['summary']['broken_imports']}")
        print(f"- Unused Components: {results['summary']['unused_components']}")
        print(f"- Integration Issues: {results['summary']['integration_issues']}")
        
        if results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Create visualization
        print("\n🎨 Creating visualization diagram...")
        visualizer = CodebaseVisualizer(analyzer)
        visualizer.create_comprehensive_diagram('codebase_visualization.png')
        print("✅ Visualization saved to codebase_visualization.png")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in codebase analysis: {str(e)}")
        traceback.print_exc()
        return None

def run_tool_chain_analysis():
    """Run the specialized tool chain analyzer"""
    print("\n🔧 RUNNING TOOL CHAIN ANALYSIS")
    print("=" * 60)
    
    try:
        from tool_chain_analyzer import ToolChainAnalyzer
        
        # Initialize analyzer
        analyzer = ToolChainAnalyzer('.')
        print("✅ Tool chain analyzer initialized")
        
        # Run analysis
        print("🔗 Analyzing tool chain integration...")
        results = analyzer.analyze_tool_chain()
        
        # Save results
        with open('tool_chain_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n🛠️  TOOL CHAIN SUMMARY")
        for tool, info in results['tools'].items():
            print(f"- {tool}: {'✅ Found' if info['found'] else '❌ Missing'}")
            if info['integration_health'] < 0.8:
                print(f"  ⚠️ Integration health: {info['integration_health']:.2f}")
        
        print(f"\n🔄 DATA FLOW ANALYSIS:")
        for flow in results['data_flows']:
            status = "✅" if flow['verified'] else "❌"
            print(f"{status} {flow['from']} → {flow['to']}")
        
        print(f"\n⚡ CRITICAL PATHS:")
        for path in results['critical_paths']:
            print(f"- {' → '.join(path['path'])}")
            if path['bottlenecks']:
                print(f"  ⚠️ Bottlenecks: {', '.join(path['bottlenecks'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in tool chain analysis: {str(e)}")
        traceback.print_exc()
        return None

def analyze_specific_integrations():
    """Analyze specific integration points"""
    print("\n🔍 ANALYZING SPECIFIC INTEGRATIONS")
    print("=" * 60)
    
    # Key files to check
    key_files = [
        'AE-Lang_interp.py',
        'auto_rebuilder.py', 
        'ae_wand_integration_bridge.py',
        'ae_theory_enhanced_auto_rebuilder.py',
        'AE_equations_sim - pygame.py',
        'auto_rebuilder_pygame_adapter.py'
    ]
    
    integration_status = {}
    
    for file in key_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
            integration_status[file] = 'found'
            
            # Quick analysis of imports
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count imports
                import_count = content.count('import ')
                function_count = content.count('def ')
                class_count = content.count('class ')
                
                print(f"   📊 {import_count} imports, {function_count} functions, {class_count} classes")
                
                # Check for key integration patterns
                if 'auto_rebuilder' in content.lower():
                    print(f"   🔗 Contains auto-rebuilder integration")
                if 'ae_lang' in content.lower() or 'aelang' in content.lower():
                    print(f"   🔗 Contains AE-Lang integration")
                if 'pygame' in content.lower():
                    print(f"   🎮 Contains pygame integration")
                if 'wand' in content.lower():
                    print(f"   🪄 Contains TheWand integration")
                    
            except Exception as e:
                print(f"   ⚠️ Error reading file: {str(e)}")
                
        else:
            print(f"❌ {file} - Missing")
            integration_status[file] = 'missing'
    
    return integration_status

def main():
    """Main analysis runner"""
    print("🚀 COMPREHENSIVE CODEBASE ANALYSIS SYSTEM")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    results = {}
    
    # Run all analyses
    results['codebase'] = run_codebase_analysis()
    results['tool_chain'] = run_tool_chain_analysis()
    results['integrations'] = analyze_specific_integrations()
    
    # Save comprehensive results
    with open('comprehensive_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print("=" * 80)
    print("📁 Results saved to:")
    print("- comprehensive_analysis_results.json")
    print("- codebase_analysis_results.json") 
    print("- tool_chain_analysis_results.json")
    print("- codebase_visualization.png")
    
    print(f"\n🔧 Next steps:")
    print("1. Review the analysis results")
    print("2. Address any broken imports or missing integrations")
    print("3. Run the debug dashboard for interactive analysis")
    print("4. Implement fixes for identified issues")

if __name__ == "__main__":
    main()
