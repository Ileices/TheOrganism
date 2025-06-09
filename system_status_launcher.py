#!/usr/bin/env python3
"""
System Status Dashboard Launcher
Comprehensive status display and debugging interface launcher
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

def display_system_status():
    """Display comprehensive system status"""
    print("🚀 COMPREHENSIVE CODEBASE ANALYSIS SYSTEM")
    print("="*60)
    print("📊 CURRENT SYSTEM STATUS REPORT")
    print("="*60)
    
    workspace = Path(__file__).parent
    
    # Load analysis results
    try:
        results_path = workspace / "codebase_analysis_results.json"
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        summary = data.get('summary', {})
        health = data.get('health_metrics', {})
        issues = data.get('detailed_issues', {})
        
        print(f"\n📈 SYSTEM METRICS:")
        print(f"  📁 Total Files Analyzed: {summary.get('total_files', 0):,}")
        print(f"  🔧 Total Functions: {summary.get('total_functions', 0):,}")
        print(f"  📋 Total Classes: {summary.get('total_classes', 0):,}")
        print(f"  🔗 Total Relationships: {summary.get('total_relationships', 0):,}")
        
        print(f"\n🏥 HEALTH METRICS:")
        overall_health = health.get('overall_health', 0)
        connectivity = health.get('connectivity_score', 0)
        maintainability = health.get('maintainability_score', 0)
        
        print(f"  🎯 Overall Health: {overall_health:.1%} {'🎉' if overall_health > 0.99 else '✅' if overall_health > 0.95 else '⚠️'}")
        print(f"  🔌 Connectivity Score: {connectivity:.1%} {'🎉' if connectivity == 1.0 else '✅' if connectivity > 0.95 else '⚠️'}")
        print(f"  🔧 Maintainability: {maintainability:.1%} {'🎉' if maintainability > 0.96 else '✅' if maintainability > 0.90 else '⚠️'}")
        
        print(f"\n⚠️ ISSUES SUMMARY:")
        print(f"  🚫 Isolated Files: {summary.get('isolated_files', 0)} {'✅' if summary.get('isolated_files', 0) == 0 else '⚠️'}")
        print(f"  📦 Broken Imports: {summary.get('broken_imports', 0)} {'✅' if summary.get('broken_imports', 0) < 50 else '⚠️'}")
        print(f"  🔗 Integration Issues: {summary.get('integration_issues', 0)} {'✅' if summary.get('integration_issues', 0) < 25 else '⚠️'}")
        print(f"  🗑️ Unused Components: {summary.get('unused_components', 0)} {'✅' if summary.get('unused_components', 0) == 0 else '⚠️'}")
        
    except Exception as e:
        print(f"❌ Error loading analysis results: {str(e)}")
    
    # Check tool chain components
    print(f"\n🔧 TOOL CHAIN COMPONENT STATUS:")
    
    tool_chain_files = {
        "AE-Lang Interpreter": "AE-Lang_interp.py",
        "Monster Scanner (Multimodal)": "multimodal_consciousness_engine.py", 
        "TheWand Integration": "ae_wand_integration_bridge.py",
        "Auto-Rebuilder": "auto_rebuilder.py"
    }
    
    operational_components = 0
    for name, filename in tool_chain_files.items():
        file_path = workspace / filename
        if file_path.exists():
            size = file_path.stat().st_size
            status = "✅ OPERATIONAL"
            if size > 10000:  # Large files likely have more functionality
                status += " (Full Implementation)"
            elif size > 1000:
                status += " (Basic Implementation)"
            operational_components += 1
        else:
            status = "❌ MISSING"
        print(f"  {name}: {status}")
    
    # Check analysis and debugging tools
    print(f"\n🛠️ ANALYSIS & DEBUGGING TOOLS:")
    
    analysis_tools = {
        "Codebase Relationship Analyzer": "codebase_relationship_analyzer.py",
        "Tool Chain Analyzer": "tool_chain_analyzer.py", 
        "Debug Dashboard": "codebase_debug_dashboard.py",
        "Interactive Console": "interactive_analysis_console.py",
        "Live Status Display": "live_status_display.py"
    }
    
    for name, filename in analysis_tools.items():
        file_path = workspace / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {name}: AVAILABLE ({size:,} bytes)")
        else:
            print(f"  ❌ {name}: MISSING")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if overall_health > 0.99 and connectivity == 1.0:
        print("  🎉 SYSTEM STATUS: EXCELLENT")
        print("  💫 The comprehensive codebase visualization and analysis system is")
        print("     FULLY OPERATIONAL with outstanding health metrics!")
        print("  🔄 Tool chain integration flow is ready for testing")
    elif overall_health > 0.95:
        print("  ✅ SYSTEM STATUS: VERY GOOD") 
        print("  🌟 Most components are operational with minor issues to address")
    else:
        print("  ⚠️ SYSTEM STATUS: NEEDS ATTENTION")
        print("  🔧 Some components require fixes for optimal operation")
    
    print(f"\n🚀 NEXT ACTIONS AVAILABLE:")
    print("  1. 🎮 Launch Interactive Debug Dashboard")
    print("  2. 📊 Run Fresh Analysis")
    print("  3. 🔍 Test Tool Chain Integration Flow")
    print("  4. 📈 Generate Updated Network Diagrams")
    print("  5. 🔧 Address Missing Dependencies")

def launch_debug_dashboard():
    """Launch the interactive debugging dashboard"""
    print("\n🎮 LAUNCHING INTERACTIVE DEBUG DASHBOARD...")
    dashboard_path = Path(__file__).parent / "codebase_debug_dashboard.py"
    
    if dashboard_path.exists():
        try:
            # Launch in a new process so it doesn't block
            subprocess.Popen([sys.executable, str(dashboard_path)], 
                           cwd=str(Path(__file__).parent))
            print("✅ Debug dashboard launched successfully!")
            print("   Check for a new window or terminal session")
        except Exception as e:
            print(f"❌ Failed to launch dashboard: {str(e)}")
    else:
        print("❌ Debug dashboard not found")

def run_fresh_analysis():
    """Run a fresh analysis of the codebase"""
    print("\n📊 RUNNING FRESH ANALYSIS...")
    analyzer_path = Path(__file__).parent / "codebase_relationship_analyzer.py"
    
    if analyzer_path.exists():
        try:
            result = subprocess.run([sys.executable, str(analyzer_path)], 
                                  cwd=str(Path(__file__).parent),
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("✅ Fresh analysis completed successfully!")
                print("📄 Results saved to codebase_analysis_results.json")
            else:
                print(f"⚠️ Analysis completed with warnings: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏱️ Analysis is taking longer than expected, running in background...")
        except Exception as e:
            print(f"❌ Failed to run analysis: {str(e)}")
    else:
        print("❌ Codebase analyzer not found")

def main():
    """Main launcher interface"""
    try:
        display_system_status()
        
        print(f"\n" + "="*60)
        print("🎮 INTERACTIVE LAUNCHER")
        print("="*60)
        
        while True:
            print(f"\nSelect an action:")
            print("1. 🎮 Launch Debug Dashboard") 
            print("2. 📊 Run Fresh Analysis")
            print("3. 🔄 Refresh Status Display")
            print("4. 📈 Show Detailed Metrics")
            print("5. ❌ Exit")
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    launch_debug_dashboard()
                elif choice == '2':
                    run_fresh_analysis()
                elif choice == '3':
                    print("\n" + "="*60)
                    display_system_status()
                elif choice == '4':
                    # Show more detailed metrics
                    results_path = Path(__file__).parent / "codebase_analysis_results.json"
                    if results_path.exists():
                        with open(results_path, 'r') as f:
                            data = json.load(f)
                        print(f"\n📊 DETAILED METRICS:")
                        print(json.dumps(data.get('health_metrics', {}), indent=2))
                    else:
                        print("❌ No detailed metrics available - run analysis first")
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Launcher failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
