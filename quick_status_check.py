#!/usr/bin/env python3
"""
Quick System Status Checker
Provides immediate status of the tool chain components
"""

import os
import sys
from pathlib import Path

def check_system_status():
    """Quick status check of all major components"""
    workspace = Path(__file__).parent
    
    print("🔍 QUICK SYSTEM STATUS CHECK")
    print("="*50)
    
    # Check major files
    major_files = {
        "AE-Lang Interpreter": "AE-Lang_interp.py",
        "Auto-Rebuilder": "auto_rebuilder.py", 
        "Wand Integration Bridge": "ae_wand_integration_bridge.py",
        "Codebase Analyzer": "codebase_relationship_analyzer.py",
        "Tool Chain Analyzer": "tool_chain_analyzer.py",
        "Debug Dashboard": "codebase_debug_dashboard.py",
        "Analysis Results": "codebase_analysis_results.json",
        "Pygame Integration": "AE_equations_sim - pygame.py",
        "Auto-Rebuilder Pygame Adapter": "auto_rebuilder_pygame_adapter.py"
    }
    
    print("\n📁 Core Files Status:")
    operational_count = 0
    for name, filename in major_files.items():
        file_path = workspace / filename
        if file_path.exists():
            size = file_path.stat().st_size
            status = "✅ PRESENT"
            if size > 1000:
                status += f" ({size:,} bytes)"
            operational_count += 1
        else:
            status = "❌ MISSING"
        print(f"  {name}: {status}")
    
    # Check for monster scanner related files
    print(f"\n🔍 Monster Scanner Files:")
    monster_files = list(workspace.glob("*monster*")) + list(workspace.glob("*scanner*"))
    if monster_files:
        for file in monster_files:
            size = file.stat().st_size if file.exists() else 0
            print(f"  ✅ {file.name} ({size:,} bytes)")
    else:
        print("  ⚠️  No explicit monster scanner files found")
    
    # Check analysis results if available
    results_path = workspace / "codebase_analysis_results.json"
    if results_path.exists():
        try:
            import json
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n📊 Previous Analysis Results:")
            summary = data.get('summary', {})
            health = data.get('health_metrics', {})
            
            print(f"  📁 Total Files: {summary.get('total_files', 'Unknown')}")
            print(f"  🔧 Total Functions: {summary.get('total_functions', 'Unknown')}")
            print(f"  📋 Total Classes: {summary.get('total_classes', 'Unknown')}")
            print(f"  🔗 Relationships: {summary.get('total_relationships', 'Unknown')}")
            print(f"  🏥 Overall Health: {health.get('overall_health', 0):.1%}")
            print(f"  🔌 Connectivity: {health.get('connectivity_score', 0):.1%}")
            print(f"  ⚠️  Broken Imports: {summary.get('broken_imports', 'Unknown')}")
            print(f"  🚫 Isolated Files: {summary.get('isolated_files', 'Unknown')}")
            
        except Exception as e:
            print(f"  ❌ Error reading analysis results: {str(e)}")
    
    # Overall status
    total_files = len(major_files)
    health_percentage = (operational_count / total_files) * 100
    
    print(f"\n🎯 OVERALL STATUS:")
    print(f"  Core Files Operational: {operational_count}/{total_files} ({health_percentage:.1f}%)")
    
    if health_percentage >= 90:
        print("  🎉 STATUS: EXCELLENT - System fully operational")
    elif health_percentage >= 75:
        print("  ✅ STATUS: GOOD - Most components operational")
    elif health_percentage >= 50:
        print("  ⚠️  STATUS: FAIR - Some components need attention")
    else:
        print("  ❌ STATUS: POOR - Major components missing")
    
    print("\n" + "="*50)
    
    return health_percentage >= 75

if __name__ == "__main__":
    try:
        status_ok = check_system_status()
        sys.exit(0 if status_ok else 1)
    except Exception as e:
        print(f"❌ Status check failed: {str(e)}")
        sys.exit(2)
