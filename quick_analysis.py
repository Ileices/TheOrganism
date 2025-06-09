#!/usr/bin/env python3
"""
Quick Codebase Analysis Tool
Provides immediate insights into your codebase structure and relationships
"""

import os
import re
import json
from datetime import datetime
from collections import defaultdict, Counter

def analyze_python_files():
    """Analyze all Python files in the current directory"""
    results = {
        'files': {},
        'imports': defaultdict(list),
        'functions': defaultdict(list),
        'classes': defaultdict(list),
        'tool_chain_files': {},
        'integration_patterns': {},
        'summary': {},
        'issues': []
    }
    
    # Key tool chain files to identify
    tool_patterns = {
        'AE-Lang': ['AE-Lang_interp.py', 'enhanced_AE_Lang_interp.py', 'production_ae_lang.py'],
        'Monster_Scanner': ['monster_scanner', 'barcode', 'ml_processing'],
        'TheWand': ['ae_wand_integration_bridge.py', 'The Wand/'],
        'Auto_Rebuilder': ['auto_rebuilder.py', 'auto_rebuilder_adapter.py', 'auto_rebuilder_pygame_adapter.py'],
        'Pygame_Integration': ['AE_equations_sim - pygame.py', 'pygame', 'auto_rebuilder_pygame_adapter.py'],
        'Consciousness': ['consciousness', 'ae_consciousness', 'ae_core_consciousness.py']
    }
    
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files")
    
    # Analyze each file
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic file stats
            lines = content.split('\n')
            file_info = {
                'path': file_path,
                'lines': len(lines),
                'imports': [],
                'functions': [],
                'classes': [],
                'size_bytes': len(content),
                'tool_type': 'unknown'
            }
            
            # Extract imports
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    file_info['imports'].append(line)
                    results['imports'][file_path].append(line)
            
            # Extract functions
            for match in re.finditer(r'^def\s+(\w+)', content, re.MULTILINE):
                func_name = match.group(1)
                file_info['functions'].append(func_name)
                results['functions'][file_path].append(func_name)
            
            # Extract classes
            for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
                class_name = match.group(1)
                file_info['classes'].append(class_name)
                results['classes'][file_path].append(class_name)
            
            # Identify tool type
            file_lower = file_path.lower()
            content_lower = content.lower()
            
            for tool, patterns in tool_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in file_lower or pattern.lower() in content_lower:
                        file_info['tool_type'] = tool
                        results['tool_chain_files'][tool] = results['tool_chain_files'].get(tool, [])
                        results['tool_chain_files'][tool].append(file_path)
                        break
            
            # Check for integration patterns
            integration_keywords = ['integration', 'bridge', 'adapter', 'connector', 'interface']
            for keyword in integration_keywords:
                if keyword in content_lower:
                    results['integration_patterns'][keyword] = results['integration_patterns'].get(keyword, [])
                    results['integration_patterns'][keyword].append(file_path)
            
            results['files'][file_path] = file_info
            
        except Exception as e:
            results['issues'].append(f"Error reading {file_path}: {str(e)}")
    
    return results

def analyze_integrations(results):
    """Analyze integration health and connections"""
    print("\n🔗 ANALYZING INTEGRATIONS")
    print("=" * 50)
    
    # Check for key tool chain connections
    connections = {
        'AE-Lang → Monster Scanner': False,
        'Monster Scanner → TheWand': False,
        'TheWand → Auto-Rebuilder': False,
        'Auto-Rebuilder → Pygame': False,
        'Consciousness → All Tools': False
    }
    
    # Analyze cross-references
    all_content = ""
    for file_path, file_info in results['files'].items():
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_content += f.read().lower() + "\n"
        except:
            continue
    
    # Check connections
    if 'aelang' in all_content and 'monster' in all_content:
        connections['AE-Lang → Monster Scanner'] = True
    
    if 'monster' in all_content and 'wand' in all_content:
        connections['Monster Scanner → TheWand'] = True
    
    if 'wand' in all_content and 'rebuilder' in all_content:
        connections['TheWand → Auto-Rebuilder'] = True
    
    if 'rebuilder' in all_content and 'pygame' in all_content:
        connections['Auto-Rebuilder → Pygame'] = True
    
    if 'consciousness' in all_content:
        connections['Consciousness → All Tools'] = True
    
    return connections

def generate_report(results, connections):
    """Generate comprehensive analysis report"""
    
    # Calculate summary statistics
    total_files = len(results['files'])
    total_lines = sum(f['lines'] for f in results['files'].values())
    total_functions = sum(len(f['functions']) for f in results['files'].values())
    total_classes = sum(len(f['classes']) for f in results['files'].values())
    total_imports = sum(len(f['imports']) for f in results['files'].values())
    
    results['summary'] = {
        'total_files': total_files,
        'total_lines': total_lines,
        'total_functions': total_functions,
        'total_classes': total_classes,
        'total_imports': total_imports,
        'analysis_time': datetime.now().isoformat()
    }
    
    print(f"\n📊 CODEBASE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"📁 Total Files: {total_files}")
    print(f"📄 Total Lines: {total_lines:,}")
    print(f"🔧 Total Functions: {total_functions}")
    print(f"📦 Total Classes: {total_classes}")
    print(f"📥 Total Imports: {total_imports}")
    
    print(f"\n🛠️  TOOL CHAIN STATUS")
    print("-" * 30)
    for tool, files in results['tool_chain_files'].items():
        print(f"✅ {tool}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files
            print(f"   • {os.path.basename(file)}")
        if len(files) > 3:
            print(f"   • ... and {len(files)-3} more")
    
    print(f"\n🔗 INTEGRATION CONNECTIONS")
    print("-" * 40)
    for connection, status in connections.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {connection}")
    
    print(f"\n🔍 INTEGRATION PATTERNS")
    print("-" * 30)
    for pattern, files in results['integration_patterns'].items():
        print(f"🔗 {pattern}: {len(files)} files")
    
    # Identify potential issues
    issues = []
    
    # Check for isolated files
    isolated_files = []
    for file_path, file_info in results['files'].items():
        if len(file_info['imports']) == 0 and len(file_info['functions']) > 0:
            isolated_files.append(file_path)
    
    if isolated_files:
        issues.append(f"Found {len(isolated_files)} potentially isolated files")
    
    # Check for missing tool chain components
    required_tools = ['AE-Lang', 'Auto_Rebuilder', 'Pygame_Integration']
    missing_tools = [tool for tool in required_tools if tool not in results['tool_chain_files']]
    
    if missing_tools:
        issues.append(f"Missing tool chain components: {', '.join(missing_tools)}")
    
    # Check for broken connections
    broken_connections = [conn for conn, status in connections.items() if not status]
    if broken_connections:
        issues.append(f"Potential broken connections: {len(broken_connections)}")
    
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES")
        print("-" * 30)
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print(f"\n✅ NO MAJOR ISSUES DETECTED")
    
    return results, issues

def main():
    """Main analysis function"""
    print("🚀 QUICK CODEBASE ANALYSIS")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    print(f"Analysis time: {datetime.now()}")
    
    # Run analysis
    results = analyze_python_files()
    connections = analyze_integrations(results)
    final_results, issues = generate_report(results, connections)
    
    # Save results
    with open('quick_analysis_results.json', 'w') as f:
        json.dump({
            'results': final_results,
            'connections': connections,
            'issues': issues
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: quick_analysis_results.json")
    
    return final_results, connections, issues

if __name__ == "__main__":
    main()
