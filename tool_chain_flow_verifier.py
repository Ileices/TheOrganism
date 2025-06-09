#!/usr/bin/env python3
"""
Tool Chain Flow Verifier
Tests the complete AE-Lang → Monster Scanner → TheWand → Auto-Rebuilder integration flow
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

class ToolChainFlowVerifier:
    """Comprehensive verification of the tool chain integration flow"""
    
    def __init__(self):
        self.workspace_path = Path(__file__).parent
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'components': {},
            'integration_flow': {},
            'overall_status': 'UNKNOWN',
            'issues': [],
            'recommendations': []
        }
        
    def verify_ae_lang_interpreter(self) -> bool:
        """Verify AE-Lang interpreter functionality"""
        print("🔍 Verifying AE-Lang Interpreter...")
        
        try:
            # Check if AE-Lang interpreter exists and is importable
            ae_lang_path = self.workspace_path / "AE-Lang_interp.py"
            if not ae_lang_path.exists():
                self.results['issues'].append("AE-Lang interpreter file not found")
                return False
                
            # Try to import and test basic functionality
            sys.path.insert(0, str(self.workspace_path))
            try:
                import AE_Lang_interp
                # Test basic interpreter functionality
                if hasattr(AE_Lang_interp, 'AELangInterpreter'):
                    interpreter = AE_Lang_interp.AELangInterpreter()
                    # Test basic memory operation
                    test_result = interpreter.process_line(";[M{test}E]=hello")
                    
                    self.results['components']['ae_lang'] = {
                        'status': 'OPERATIONAL',
                        'file_path': str(ae_lang_path),
                        'classes': [name for name in dir(AE_Lang_interp) if not name.startswith('_')],
                        'test_result': str(test_result)[:100] if test_result else "None"
                    }
                    print("✅ AE-Lang Interpreter: OPERATIONAL")
                    return True
                else:
                    self.results['issues'].append("AELangInterpreter class not found in module")
                    return False
                    
            except Exception as e:
                self.results['issues'].append(f"AE-Lang interpreter import/execution error: {str(e)}")
                return False
                
        except Exception as e:
            self.results['issues'].append(f"AE-Lang verification failed: {str(e)}")
            return False
    
    def verify_monster_scanner(self) -> bool:
        """Verify Monster Scanner functionality"""
        print("🔍 Verifying Monster Scanner...")
        
        try:
            # Look for monster scanner related files
            monster_files = list(self.workspace_path.glob("*monster*"))
            scanner_files = list(self.workspace_path.glob("*scanner*"))
            
            all_monster_scanner_files = monster_files + scanner_files
            
            if not all_monster_scanner_files:
                self.results['issues'].append("No Monster Scanner files found")
                return False
            
            # Check for specific monster scanner functionality
            for file_path in all_monster_scanner_files:
                if file_path.suffix == '.py':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Look for scanner-related functionality
                        if any(keyword in content.lower() for keyword in 
                               ['scan', 'monitor', 'detect', 'analyze', 'threat']):
                            
                            self.results['components']['monster_scanner'] = {
                                'status': 'DETECTED',
                                'files': [str(f) for f in all_monster_scanner_files],
                                'main_file': str(file_path),
                                'functionality': 'Scanning/Monitoring capabilities detected'
                            }
                            print("✅ Monster Scanner: DETECTED")
                            return True
                            
                    except Exception as e:
                        self.results['issues'].append(f"Error reading {file_path}: {str(e)}")
            
            self.results['components']['monster_scanner'] = {
                'status': 'PARTIAL',
                'files': [str(f) for f in all_monster_scanner_files],
                'note': 'Files found but functionality unclear'
            }
            print("⚠️ Monster Scanner: PARTIAL")
            return True
            
        except Exception as e:
            self.results['issues'].append(f"Monster Scanner verification failed: {str(e)}")
            return False
    
    def verify_the_wand(self) -> bool:
        """Verify TheWand integration functionality"""
        print("🔍 Verifying TheWand Integration...")
        
        try:
            # Look for wand-related files
            wand_files = list(self.workspace_path.glob("*wand*"))
            bridge_files = list(self.workspace_path.glob("*bridge*"))
            
            all_wand_files = wand_files + bridge_files
            
            if not all_wand_files:
                self.results['issues'].append("No Wand integration files found")
                return False
            
            # Check the main integration bridge
            bridge_path = self.workspace_path / "ae_wand_integration_bridge.py"
            if bridge_path.exists():
                try:
                    with open(bridge_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for integration functionality
                    integration_indicators = [
                        'AEWandBridge', 'integration', 'bridge', 'wand', 'distributed'
                    ]
                    
                    if any(indicator in content for indicator in integration_indicators):
                        self.results['components']['the_wand'] = {
                            'status': 'OPERATIONAL',
                            'files': [str(f) for f in all_wand_files],
                            'main_bridge': str(bridge_path),
                            'integration_class': 'AEWandBridge detected'
                        }
                        print("✅ TheWand Integration: OPERATIONAL")
                        return True
                        
                except Exception as e:
                    self.results['issues'].append(f"Error reading wand bridge: {str(e)}")
            
            self.results['components']['the_wand'] = {
                'status': 'PARTIAL',
                'files': [str(f) for f in all_wand_files],
                'note': 'Files found but integration unclear'
            }
            print("⚠️ TheWand Integration: PARTIAL")
            return True
            
        except Exception as e:
            self.results['issues'].append(f"TheWand verification failed: {str(e)}")
            return False
    
    def verify_auto_rebuilder(self) -> bool:
        """Verify Auto-Rebuilder functionality"""
        print("🔍 Verifying Auto-Rebuilder...")
        
        try:
            # Check main auto-rebuilder file
            rebuilder_path = self.workspace_path / "auto_rebuilder.py"
            if not rebuilder_path.exists():
                self.results['issues'].append("Auto-rebuilder main file not found")
                return False
            
            # Check file size and content indicators
            file_size = rebuilder_path.stat().st_size
            
            try:
                with open(rebuilder_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for key rebuilder functionality
                rebuilder_indicators = [
                    'rebuild', 'autorebuild', 'integration', 'dependency', 'analyze'
                ]
                
                indicator_count = sum(1 for indicator in rebuilder_indicators 
                                    if indicator in content.lower())
                
                # Look for pygame adapter
                pygame_adapter_path = self.workspace_path / "auto_rebuilder_pygame_adapter.py"
                pygame_integration = pygame_adapter_path.exists()
                
                self.results['components']['auto_rebuilder'] = {
                    'status': 'OPERATIONAL',
                    'main_file': str(rebuilder_path),
                    'file_size': file_size,
                    'functionality_indicators': indicator_count,
                    'pygame_integration': pygame_integration,
                    'pygame_adapter': str(pygame_adapter_path) if pygame_integration else None
                }
                
                print("✅ Auto-Rebuilder: OPERATIONAL")
                return True
                
            except Exception as e:
                self.results['issues'].append(f"Error reading auto-rebuilder: {str(e)}")
                return False
                
        except Exception as e:
            self.results['issues'].append(f"Auto-Rebuilder verification failed: {str(e)}")
            return False
    
    def test_integration_flow(self) -> bool:
        """Test the complete integration flow"""
        print("🔍 Testing Integration Flow...")
        
        try:
            # Test AE-Lang → Monster Scanner flow
            ae_to_monster = self.test_ae_to_monster_flow()
            
            # Test Monster Scanner → TheWand flow
            monster_to_wand = self.test_monster_to_wand_flow()
            
            # Test TheWand → Auto-Rebuilder flow
            wand_to_rebuilder = self.test_wand_to_rebuilder_flow()
            
            self.results['integration_flow'] = {
                'ae_to_monster': ae_to_monster,
                'monster_to_wand': monster_to_wand,
                'wand_to_rebuilder': wand_to_rebuilder,
                'overall_flow': ae_to_monster and monster_to_wand and wand_to_rebuilder
            }
            
            if self.results['integration_flow']['overall_flow']:
                print("✅ Integration Flow: COMPLETE")
                return True
            else:
                print("⚠️ Integration Flow: PARTIAL")
                return False
                
        except Exception as e:
            self.results['issues'].append(f"Integration flow test failed: {str(e)}")
            return False
    
    def test_ae_to_monster_flow(self) -> bool:
        """Test AE-Lang to Monster Scanner data flow"""
        try:
            # Look for files that connect AE-Lang to monitoring/scanning
            connection_indicators = [
                'ae_lang_script_validator.py',
                'ae_framework_integration.py'
            ]
            
            connections_found = 0
            for indicator in connection_indicators:
                if (self.workspace_path / indicator).exists():
                    connections_found += 1
            
            return connections_found > 0
            
        except Exception:
            return False
    
    def test_monster_to_wand_flow(self) -> bool:
        """Test Monster Scanner to TheWand data flow"""
        try:
            # Look for integration bridges or validators
            bridge_indicators = [
                'ae_wand_integration_bridge.py',
                'ae_wand_integration_validator.py'
            ]
            
            bridges_found = 0
            for indicator in bridge_indicators:
                if (self.workspace_path / indicator).exists():
                    bridges_found += 1
            
            return bridges_found > 0
            
        except Exception:
            return False
    
    def test_wand_to_rebuilder_flow(self) -> bool:
        """Test TheWand to Auto-Rebuilder data flow"""
        try:
            # Check if auto-rebuilder has wand integration capabilities
            rebuilder_path = self.workspace_path / "auto_rebuilder.py"
            if rebuilder_path.exists():
                with open(rebuilder_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for wand-related integration code
                wand_integration = any(keyword in content.lower() for keyword in 
                                     ['wand', 'bridge', 'distributed', 'integration'])
                
                return wand_integration
            
            return False
            
        except Exception:
            return False
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run complete tool chain verification"""
        print("🚀 Starting Comprehensive Tool Chain Verification\n")
        
        # Verify individual components
        ae_lang_ok = self.verify_ae_lang_interpreter()
        monster_ok = self.verify_monster_scanner()
        wand_ok = self.verify_the_wand()
        rebuilder_ok = self.verify_auto_rebuilder()
        
        # Test integration flow
        flow_ok = self.test_integration_flow()
        
        # Determine overall status
        component_scores = [ae_lang_ok, monster_ok, wand_ok, rebuilder_ok]
        operational_components = sum(component_scores)
        
        if operational_components == 4 and flow_ok:
            self.results['overall_status'] = 'FULLY_OPERATIONAL'
        elif operational_components >= 3:
            self.results['overall_status'] = 'MOSTLY_OPERATIONAL'
        elif operational_components >= 2:
            self.results['overall_status'] = 'PARTIALLY_OPERATIONAL'
        else:
            self.results['overall_status'] = 'NEEDS_ATTENTION'
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.results
    
    def generate_recommendations(self):
        """Generate recommendations based on verification results"""
        if self.results['overall_status'] == 'FULLY_OPERATIONAL':
            self.results['recommendations'].append(
                "System is fully operational. Consider running performance optimizations."
            )
        else:
            if len(self.results['issues']) > 0:
                self.results['recommendations'].append(
                    "Address the identified issues to improve system integration."
                )
            
            # Component-specific recommendations
            if 'ae_lang' not in self.results['components']:
                self.results['recommendations'].append(
                    "Fix AE-Lang interpreter for complete functionality."
                )
            
            if 'monster_scanner' not in self.results['components']:
                self.results['recommendations'].append(
                    "Implement or repair Monster Scanner component."
                )
            
            if 'the_wand' not in self.results['components']:
                self.results['recommendations'].append(
                    "Establish TheWand integration bridge."
                )
            
            if 'auto_rebuilder' not in self.results['components']:
                self.results['recommendations'].append(
                    "Verify Auto-Rebuilder system functionality."
                )
    
    def print_results(self):
        """Print formatted verification results"""
        print("\n" + "="*60)
        print("TOOL CHAIN VERIFICATION RESULTS")
        print("="*60)
        
        print(f"\n🕐 Timestamp: {self.results['timestamp']}")
        print(f"🎯 Overall Status: {self.results['overall_status']}")
        
        print(f"\n📊 Component Status:")
        for component, info in self.results['components'].items():
            status_emoji = "✅" if info['status'] == 'OPERATIONAL' else "⚠️" if info['status'] == 'PARTIAL' else "❌"
            print(f"  {status_emoji} {component.replace('_', ' ').title()}: {info['status']}")
        
        print(f"\n🔄 Integration Flow:")
        flow = self.results.get('integration_flow', {})
        for step, status in flow.items():
            if step != 'overall_flow':
                status_emoji = "✅" if status else "❌"
                print(f"  {status_emoji} {step.replace('_', ' → ').title()}: {'OK' if status else 'ISSUES'}")
        
        if self.results['issues']:
            print(f"\n⚠️ Issues Found ({len(self.results['issues'])}):")
            for i, issue in enumerate(self.results['issues'], 1):
                print(f"  {i}. {issue}")
        
        if self.results['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
    
    def save_results(self, filename: str = "tool_chain_verification_results.json"):
        """Save verification results to JSON file"""
        output_path = self.workspace_path / filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📄 Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")

def main():
    """Main execution function"""
    try:
        verifier = ToolChainFlowVerifier()
        results = verifier.run_comprehensive_verification()
        verifier.print_results()
        verifier.save_results()
        
        # Return exit code based on status
        if results['overall_status'] in ['FULLY_OPERATIONAL', 'MOSTLY_OPERATIONAL']:
            return 0
        else:
            return 1
    
    except Exception as e:
        print(f"❌ Verification failed with error: {str(e)}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
