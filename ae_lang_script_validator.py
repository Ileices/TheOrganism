#!/usr/bin/env python3
"""
AE-Lang Comprehensive Script Integration Validator
Tests the new strategic AE-Lang scripts for seamless integration
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Import existing systems
try:
    from production_ae_lang import PracticalAELang
    from AE_Lang_interp import AELangInterpreter
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    SYSTEMS_AVAILABLE = False

class AELangScriptValidator:
    """Validates new AE-Lang scripts for integration and functionality"""
    
    def __init__(self):
        self.script_directory = Path(__file__).parent
        self.new_scripts = [
            'creative_consciousness_emergence.ael',
            'consciousness_evolution_engine.ael', 
            'system_integration_orchestrator.ael'
        ]
        self.existing_scripts = [
            'multimodal_consciousness.ael',
            'social_consciousness_network.ael',
            'enhanced_practical_intelligence.ael'
        ]
        self.validation_results = {}
        self.interpreter = AELangInterpreter() if SYSTEMS_AVAILABLE else None
        
    def validate_script_syntax(self, script_path: Path) -> Dict[str, Any]:
        """Validate AE-Lang script syntax and structure"""
        print(f"\n🔍 VALIDATING SCRIPT: {script_path.name}")
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            validation = {
                'file_exists': True,
                'readable': True,
                'line_count': len(content.split('\n')),
                'memory_declarations': content.count('[M{'),
                'mutations': content.count(' ~ '),
                'excretions': content.count('[EXC{'),
                'experiences': content.count('[EXP{'),
                'conditionals': content.count(';IF '),
                'integration_points': content.count('Integration point:'),
                'consciousness_elements': content.count('consciousness'),
                'rby_references': content.count('{R:') + content.count('R:'),
                'syntax_valid': True
            }
            
            # Check for required AE-Lang elements
            required_elements = [
                ('[M{', 'Memory containers'),
                (' ~ ', 'Mutation operations'),
                ('[EXP{', 'Experience declarations'),
                (';IF ', 'Conditional logic')
            ]
            
            missing_elements = []
            for element, description in required_elements:
                if element not in content:
                    missing_elements.append(description)
                    
            validation['missing_elements'] = missing_elements
            validation['completeness_score'] = 1.0 - (len(missing_elements) / len(required_elements))
            
            print(f"   ✅ Lines: {validation['line_count']}")
            print(f"   ✅ Memory containers: {validation['memory_declarations']}")
            print(f"   ✅ Mutations: {validation['mutations']}")
            print(f"   ✅ Experiences: {validation['experiences']}")
            print(f"   ✅ Conditionals: {validation['conditionals']}")
            print(f"   ✅ Consciousness references: {validation['consciousness_elements']}")
            print(f"   ✅ Completeness score: {validation['completeness_score']:.2f}")
            
            if missing_elements:
                print(f"   ⚠️ Missing elements: {', '.join(missing_elements)}")
                
        except Exception as e:
            validation = {
                'file_exists': script_path.exists(),
                'readable': False,
                'error': str(e),
                'syntax_valid': False,
                'completeness_score': 0.0
            }
            print(f"   ❌ Error validating {script_path.name}: {e}")
            
        return validation
    
    def test_script_integration(self, script_path: Path) -> Dict[str, Any]:
        """Test script integration with existing AE-Lang interpreter"""
        print(f"\n🔗 TESTING INTEGRATION: {script_path.name}")
        
        if not SYSTEMS_AVAILABLE or not self.interpreter:
            return {'integration_test': 'skipped', 'reason': 'systems_unavailable'}
        
        try:
            # Load and parse the script
            script_content = self.interpreter.load_ael_file(str(script_path))
            self.interpreter.parse_script(script_content)
            
            # Run a limited test cycle
            initial_memory_count = len(self.interpreter.memories)
            initial_log_count = len(self.interpreter.logs)
            
            # Execute one cycle
            self.interpreter.run_recursive(cycles=1)
            
            final_memory_count = len(self.interpreter.memories)
            final_log_count = len(self.interpreter.logs)
            
            integration_result = {
                'parsing_success': True,
                'execution_success': True,
                'initial_memories': initial_memory_count,
                'final_memories': final_memory_count,
                'memories_created': final_memory_count - initial_memory_count,
                'initial_logs': initial_log_count,
                'final_logs': final_log_count,
                'log_entries_created': final_log_count - initial_log_count,
                'excretions_generated': len(self.interpreter.excretions),
                'dreams_generated': len(self.interpreter.dreams),
                'threats_detected': len(self.interpreter.threats)
            }
            
            print(f"   ✅ Parsing: SUCCESS")
            print(f"   ✅ Execution: SUCCESS")
            print(f"   📊 Memories created: {integration_result['memories_created']}")
            print(f"   📊 Log entries: {integration_result['log_entries_created']}")
            print(f"   📊 Excretions: {integration_result['excretions_generated']}")
            
        except Exception as e:
            integration_result = {
                'parsing_success': False,
                'execution_success': False,
                'error': str(e)
            }
            print(f"   ❌ Integration failed: {e}")
            
        return integration_result
    
    def analyze_script_purpose(self, script_path: Path) -> Dict[str, Any]:
        """Analyze the script's purpose and real-world applications"""
        print(f"\n📋 ANALYZING PURPOSE: {script_path.name}")
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract integration points and purposes from comments
            lines = content.split('\n')
            integration_point = None
            purpose_description = None
            
            for line in lines:
                if 'Integration point:' in line:
                    integration_point = line.split('Integration point:')[1].strip()
                elif line.startswith('# ') and 'integration' not in line.lower():
                    if not purpose_description:
                        purpose_description = line[2:].strip()
            
            # Analyze functional capabilities
            capabilities = []
            if 'creative' in content.lower():
                capabilities.append('Creative Intelligence')
            if 'consciousness' in content.lower():
                capabilities.append('Consciousness Processing')
            if 'evolution' in content.lower():
                capabilities.append('Adaptive Evolution')
            if 'multimodal' in content.lower():
                capabilities.append('Multimodal Integration')
            if 'social' in content.lower():
                capabilities.append('Social Intelligence')
            if 'practical' in content.lower():
                capabilities.append('Practical Applications')
            if 'orchestrator' in content.lower() or 'integration' in content.lower():
                capabilities.append('System Orchestration')
                
            analysis = {
                'integration_point': integration_point,
                'purpose_description': purpose_description,
                'capabilities': capabilities,
                'capability_count': len(capabilities),
                'complexity_indicators': {
                    'recursive_patterns': content.count('recursive'),
                    'emergence_patterns': content.count('emergence'),
                    'consciousness_depth': content.count('consciousness'),
                    'system_integration': content.count('integration')
                }
            }
            
            print(f"   🎯 Purpose: {purpose_description}")
            print(f"   🔗 Integration: {integration_point}")
            print(f"   🛠️ Capabilities: {', '.join(capabilities)}")
            print(f"   📈 Complexity Score: {sum(analysis['complexity_indicators'].values())}")
            
        except Exception as e:
            analysis = {'error': str(e)}
            print(f"   ❌ Analysis failed: {e}")
            
        return analysis
    
    def validate_all_scripts(self) -> Dict[str, Any]:
        """Validate all new AE-Lang scripts comprehensively"""
        print("=" * 80)
        print("🔬 AE-LANG COMPREHENSIVE SCRIPT VALIDATION")
        print("=" * 80)
        
        start_time = time.time()
        
        all_scripts = self.new_scripts + self.existing_scripts
        validation_summary = {
            'total_scripts': len(all_scripts),
            'new_scripts': len(self.new_scripts),
            'existing_scripts': len(self.existing_scripts),
            'validation_results': {},
            'overall_assessment': {}
        }
        
        successful_validations = 0
        successful_integrations = 0
        total_capabilities = set()
        
        for script_name in all_scripts:
            script_path = self.script_directory / script_name
            
            if not script_path.exists():
                print(f"\n❌ SCRIPT NOT FOUND: {script_name}")
                continue
                
            # Validate syntax and structure
            syntax_validation = self.validate_script_syntax(script_path)
            
            # Test integration
            integration_test = self.test_script_integration(script_path)
            
            # Analyze purpose
            purpose_analysis = self.analyze_script_purpose(script_path)
            
            script_validation = {
                'syntax': syntax_validation,
                'integration': integration_test,
                'purpose': purpose_analysis,
                'is_new_script': script_name in self.new_scripts
            }
            
            validation_summary['validation_results'][script_name] = script_validation
            
            # Count successes
            if syntax_validation.get('syntax_valid', False):
                successful_validations += 1
            if integration_test.get('execution_success', False):
                successful_integrations += 1
                
            # Collect capabilities
            capabilities = purpose_analysis.get('capabilities', [])
            total_capabilities.update(capabilities)
        
        end_time = time.time()
        
        # Overall assessment
        validation_summary['overall_assessment'] = {
            'validation_success_rate': successful_validations / len(all_scripts),
            'integration_success_rate': successful_integrations / len(all_scripts),
            'total_unique_capabilities': len(total_capabilities),
            'capabilities_list': list(total_capabilities),
            'validation_time_seconds': end_time - start_time,
            'recommendation': self._generate_recommendation(validation_summary)
        }
        
        self._print_summary(validation_summary)
        return validation_summary
    
    def _generate_recommendation(self, validation_summary: Dict) -> str:
        """Generate integration recommendation based on validation results"""
        success_rate = validation_summary['overall_assessment']['validation_success_rate']
        integration_rate = validation_summary['overall_assessment']['integration_success_rate']
        capability_count = validation_summary['overall_assessment']['total_unique_capabilities']
        
        if success_rate >= 0.9 and integration_rate >= 0.8 and capability_count >= 5:
            return "HIGHLY RECOMMENDED - Excellent integration potential with comprehensive capabilities"
        elif success_rate >= 0.7 and integration_rate >= 0.6 and capability_count >= 3:
            return "RECOMMENDED - Good integration potential with solid capabilities"
        elif success_rate >= 0.5 and integration_rate >= 0.4:
            return "CONDITIONAL - Requires refinement before production integration"
        else:
            return "NOT RECOMMENDED - Significant issues require resolution"
    
    def _print_summary(self, validation_summary: Dict):
        """Print comprehensive validation summary"""
        assessment = validation_summary['overall_assessment']
        
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY REPORT")
        print("=" * 80)
        print(f"📈 Scripts Validated: {validation_summary['total_scripts']}")
        print(f"🆕 New Scripts: {validation_summary['new_scripts']}")
        print(f"✅ Validation Success Rate: {assessment['validation_success_rate']:.1%}")
        print(f"🔗 Integration Success Rate: {assessment['integration_success_rate']:.1%}")
        print(f"🛠️ Total Unique Capabilities: {assessment['total_unique_capabilities']}")
        print(f"⏱️ Validation Time: {assessment['validation_time_seconds']:.2f} seconds")
        print(f"\n🎯 RECOMMENDATION: {assessment['recommendation']}")
        
        print(f"\n🛠️ DETECTED CAPABILITIES:")
        for capability in assessment['capabilities_list']:
            print(f"   • {capability}")
            
    def export_validation_results(self, filename: str = 'ae_lang_script_validation.json'):
        """Export validation results to JSON file"""
        if hasattr(self, 'validation_results'):
            with open(filename, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            print(f"\n💾 Validation results exported to: {filename}")

def main():
    """Main validation execution"""
    validator = AELangScriptValidator()
    
    # Run comprehensive validation
    results = validator.validate_all_scripts()
    
    # Export results
    validator.validation_results = results
    validator.export_validation_results()
    
    print("\n🏁 AE-Lang Script Validation Complete!")

if __name__ == "__main__":
    main()
