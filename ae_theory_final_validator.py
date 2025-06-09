#!/usr/bin/env python3
"""
AE Theory Final Integration Validator
====================================

This script validates that the AE Theory auto-rebuilder integration is ready for production
and performs final compatibility checks with the existing system.
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class AETheoryIntegrationValidator:
    """Validates AE Theory integration readiness"""
    
    def __init__(self):
        self.workspace_path = Path.cwd()
        self.validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "overall_status": "UNKNOWN",
            "deployment_ready": False,
            "detailed_results": {}
        }
    
    async def validate_file_existence(self) -> bool:
        """Validate that all required AE Theory files exist"""
        print("📁 Validating AE Theory file existence...")
        
        required_files = [
            "ae_theory_enhanced_auto_rebuilder.py",
            "ae_theory_advanced_auto_rebuilder.py", 
            "ae_theory_production_integration.py",
            "AE_THEORY_PRODUCTION_DEPLOYMENT_GUIDE.md"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.workspace_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.validation_results["critical_issues"].append(f"Missing files: {missing_files}")
            self.validation_results["detailed_results"]["file_existence"] = "FAILED"
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            self.validation_results["detailed_results"]["file_existence"] = "PASSED"
            print("✅ All required files exist")
            return True
    
    async def validate_import_compatibility(self) -> bool:
        """Validate that AE Theory modules can be imported"""
        print("🔗 Validating import compatibility...")
        
        import_tests = []
        
        # Test enhanced auto-rebuilder import
        try:
            import ae_theory_enhanced_auto_rebuilder
            import_tests.append(("Enhanced AE Auto-Rebuilder", True, None))
            print("✅ Enhanced AE auto-rebuilder imports successfully")
        except Exception as e:
            import_tests.append(("Enhanced AE Auto-Rebuilder", False, str(e)))
            print(f"❌ Enhanced AE auto-rebuilder import failed: {e}")
        
        # Test advanced auto-rebuilder import
        try:
            import ae_theory_advanced_auto_rebuilder
            import_tests.append(("Advanced AE Auto-Rebuilder", True, None))
            print("✅ Advanced AE auto-rebuilder imports successfully")
        except Exception as e:
            import_tests.append(("Advanced AE Auto-Rebuilder", False, str(e)))
            print(f"❌ Advanced AE auto-rebuilder import failed: {e}")
        
        # Test production integration import
        try:
            import ae_theory_production_integration
            import_tests.append(("Production Integration", True, None))
            print("✅ Production integration imports successfully")
        except Exception as e:
            import_tests.append(("Production Integration", False, str(e)))
            print(f"❌ Production integration import failed: {e}")
        
        # Test original system compatibility
        try:
            import auto_rebuilder
            import_tests.append(("Original Auto-Rebuilder", True, None))
            print("✅ Original auto-rebuilder compatibility maintained")
        except Exception as e:
            import_tests.append(("Original Auto-Rebuilder", False, str(e)))
            self.validation_results["warnings"].append(f"Original auto-rebuilder not available: {e}")
        
        try:
            import digital_organism_auto_rebuilder_integration
            import_tests.append(("Digital Organism Integration", True, None))
            print("✅ Digital organism integration compatibility maintained")
        except Exception as e:
            import_tests.append(("Digital Organism Integration", False, str(e)))
            self.validation_results["warnings"].append(f"Digital organism integration not available: {e}")
        
        self.validation_results["detailed_results"]["import_tests"] = import_tests
        
        # Check if critical imports passed
        critical_imports = ["Enhanced AE Auto-Rebuilder", "Advanced AE Auto-Rebuilder", "Production Integration"]
        failed_critical = [test[0] for test in import_tests if test[0] in critical_imports and not test[1]]
        
        if failed_critical:
            self.validation_results["critical_issues"].append(f"Critical import failures: {failed_critical}")
            return False
        else:
            return True
    
    async def validate_ae_theory_functionality(self) -> bool:
        """Validate core AE Theory functionality"""
        print("🧠 Validating AE Theory functionality...")
        
        functionality_tests = []
        
        try:
            # Test RBY Vector functionality
            from ae_theory_advanced_auto_rebuilder import RBYVector
            from decimal import Decimal
            
            rby = RBYVector(Decimal('0.3'), Decimal('0.3'), Decimal('0.4'))
            rby.normalize()
            ae_value = rby.get_ae_constraint()
            
            if abs(float(ae_value) - 1.0) < 0.0001:
                functionality_tests.append(("RBY Vector AE Constraint", True, f"AE = {ae_value}"))
                print(f"✅ RBY Vector AE constraint validated: AE = {ae_value}")
            else:
                functionality_tests.append(("RBY Vector AE Constraint", False, f"AE = {ae_value} (should be 1.0)"))
                print(f"❌ RBY Vector AE constraint failed: AE = {ae_value}")
            
        except Exception as e:
            functionality_tests.append(("RBY Vector AE Constraint", False, str(e)))
            print(f"❌ RBY Vector test failed: {e}")
        
        try:
            # Test Crystallized AE functionality
            from ae_theory_advanced_auto_rebuilder import CrystallizedAE
            
            c_ae = CrystallizedAE()
            c_ae.expand(Decimal('0.1'))
            
            if c_ae.current_size > 0:
                functionality_tests.append(("Crystallized AE Expansion", True, f"Size: {c_ae.current_size}"))
                print(f"✅ Crystallized AE expansion validated: size = {c_ae.current_size}")
            else:
                functionality_tests.append(("Crystallized AE Expansion", False, "No expansion occurred"))
                print("❌ Crystallized AE expansion failed")
            
        except Exception as e:
            functionality_tests.append(("Crystallized AE Expansion", False, str(e)))
            print(f"❌ Crystallized AE test failed: {e}")
        
        try:
            # Test PTAIE Glyph functionality
            from ae_theory_advanced_auto_rebuilder import PTAIEGlyph, RBYVector
            from decimal import Decimal
            
            rby = RBYVector(Decimal('0.33'), Decimal('0.33'), Decimal('0.34'))
            glyph = PTAIEGlyph("test_concept", rby)
            glyph.apply_photonic_compression()
            
            if glyph.photonic_compression_factor > 0:
                functionality_tests.append(("PTAIE Glyph Compression", True, f"Factor: {glyph.photonic_compression_factor}"))
                print(f"✅ PTAIE glyph compression validated: factor = {glyph.photonic_compression_factor}")
            else:
                functionality_tests.append(("PTAIE Glyph Compression", False, "No compression applied"))
                print("❌ PTAIE glyph compression failed")
            
        except Exception as e:
            functionality_tests.append(("PTAIE Glyph Compression", False, str(e)))
            print(f"❌ PTAIE glyph test failed: {e}")
        
        self.validation_results["detailed_results"]["functionality_tests"] = functionality_tests
        
        # Check if critical functionality passed
        failed_tests = [test for test in functionality_tests if not test[1]]
        if failed_tests:
            self.validation_results["warnings"].extend([f"Functionality issue: {test[0]} - {test[2]}" for test in failed_tests])
        
        return len(failed_tests) == 0
    
    async def validate_production_integration(self) -> bool:
        """Validate production integration readiness"""
        print("🏭 Validating production integration...")
        
        try:
            from ae_theory_production_integration import AETheoryProductionConfig, create_ae_theory_production_integration
            
            # Test configuration creation
            config = {
                'workspace_path': str(self.workspace_path),
                'ae_theory_mode': 'advanced',
                'production_mode': True
            }
            
            # Test integration creation (but don't start it)
            integration = await create_ae_theory_production_integration(config)
            
            if integration and integration.config:
                self.validation_results["detailed_results"]["production_integration"] = "PASSED"
                print("✅ Production integration validated successfully")
                return True
            else:
                self.validation_results["detailed_results"]["production_integration"] = "FAILED"
                self.validation_results["critical_issues"].append("Production integration creation failed")
                print("❌ Production integration validation failed")
                return False
            
        except Exception as e:
            self.validation_results["detailed_results"]["production_integration"] = f"FAILED: {e}"
            self.validation_results["critical_issues"].append(f"Production integration error: {e}")
            print(f"❌ Production integration validation failed: {e}")
            return False
    
    async def validate_monitoring_setup(self) -> bool:
        """Validate monitoring and metrics setup"""
        print("📊 Validating monitoring setup...")
        
        try:
            # Check if monitoring directory can be created
            monitoring_dir = self.workspace_path / "ae_theory_monitoring"
            monitoring_dir.mkdir(exist_ok=True)
            
            # Test metrics file creation
            test_metrics = {
                "test_timestamp": datetime.now().isoformat(),
                "test_metric": "validation"
            }
            
            metrics_file = monitoring_dir / "test_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            # Verify file was created
            if metrics_file.exists():
                # Clean up test file
                metrics_file.unlink()
                
                self.validation_results["detailed_results"]["monitoring_setup"] = "PASSED"
                print("✅ Monitoring setup validated successfully")
                return True
            else:
                self.validation_results["detailed_results"]["monitoring_setup"] = "FAILED"
                self.validation_results["warnings"].append("Monitoring directory setup failed")
                print("❌ Monitoring setup validation failed")
                return False
            
        except Exception as e:
            self.validation_results["detailed_results"]["monitoring_setup"] = f"FAILED: {e}"
            self.validation_results["warnings"].append(f"Monitoring setup error: {e}")
            print(f"❌ Monitoring setup validation failed: {e}")
            return False
    
    async def generate_final_report(self):
        """Generate final validation report"""
        print("\n" + "="*60)
        print("📋 AE THEORY INTEGRATION VALIDATION REPORT")
        print("="*60)
        
        # Count results
        passed_tests = sum(1 for result in self.validation_results["detailed_results"].values() 
                          if result == "PASSED")
        total_tests = len(self.validation_results["detailed_results"])
        
        self.validation_results["tests_passed"] = passed_tests
        self.validation_results["tests_failed"] = total_tests - passed_tests
        
        # Determine overall status
        critical_issues = len(self.validation_results["critical_issues"])
        warnings = len(self.validation_results["warnings"])
        
        if critical_issues == 0 and passed_tests == total_tests:
            self.validation_results["overall_status"] = "EXCELLENT"
            self.validation_results["deployment_ready"] = True
        elif critical_issues == 0:
            self.validation_results["overall_status"] = "GOOD"
            self.validation_results["deployment_ready"] = True
        elif critical_issues <= 2:
            self.validation_results["overall_status"] = "ACCEPTABLE"
            self.validation_results["deployment_ready"] = False
        else:
            self.validation_results["overall_status"] = "NEEDS_WORK"
            self.validation_results["deployment_ready"] = False
        
        # Print summary
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Critical Issues: {critical_issues}")
        print(f"Warnings: {warnings}")
        print(f"Overall Status: {self.validation_results['overall_status']}")
        print(f"Deployment Ready: {'✅ YES' if self.validation_results['deployment_ready'] else '❌ NO'}")
        
        # Print detailed results
        print("\n📊 Detailed Test Results:")
        for test_name, result in self.validation_results["detailed_results"].items():
            status_icon = "✅" if result == "PASSED" else "❌"
            print(f"  {status_icon} {test_name}: {result}")
        
        # Print issues and warnings
        if self.validation_results["critical_issues"]:
            print("\n🚨 Critical Issues:")
            for issue in self.validation_results["critical_issues"]:
                print(f"  ❌ {issue}")
        
        if self.validation_results["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in self.validation_results["warnings"]:
                print(f"  ⚠️  {warning}")
        
        # Generate recommendations
        if self.validation_results["deployment_ready"]:
            self.validation_results["recommendations"] = [
                "System is ready for production deployment",
                "Consider running extended stress tests",
                "Monitor initial deployment closely",
                "Keep fallback systems available"
            ]
        else:
            self.validation_results["recommendations"] = [
                "Resolve critical issues before deployment",
                "Review failed tests and fix underlying problems",
                "Run validation again after fixes",
                "Consider gradual rollout approach"
            ]
        
        print("\n💡 Recommendations:")
        for recommendation in self.validation_results["recommendations"]:
            print(f"  💡 {recommendation}")
        
        # Save validation report
        report_file = self.workspace_path / "ae_theory_integration_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\n📄 Validation report saved to: {report_file}")
        print("="*60)
    
    async def run_complete_validation(self):
        """Run complete validation suite"""
        print("🔍 Starting AE Theory Integration Validation")
        print("=" * 50)
        
        validation_steps = [
            ("File Existence", self.validate_file_existence),
            ("Import Compatibility", self.validate_import_compatibility),
            ("AE Theory Functionality", self.validate_ae_theory_functionality),
            ("Production Integration", self.validate_production_integration),
            ("Monitoring Setup", self.validate_monitoring_setup)
        ]
        
        for step_name, step_function in validation_steps:
            print(f"\n🔎 Running: {step_name}")
            try:
                result = await step_function()
                if result:
                    print(f"✅ {step_name}: PASSED")
                else:
                    print(f"❌ {step_name}: FAILED")
            except Exception as e:
                print(f"❌ {step_name}: ERROR - {e}")
                self.validation_results["critical_issues"].append(f"{step_name} validation error: {e}")
        
        # Generate final report
        await self.generate_final_report()
        
        return self.validation_results["deployment_ready"]

async def main():
    """Main validation entry point"""
    validator = AETheoryIntegrationValidator()
    deployment_ready = await validator.run_complete_validation()
    
    if deployment_ready:
        print("\n🎉 AE THEORY INTEGRATION VALIDATION SUCCESSFUL!")
        print("   System is ready for production deployment.")
        sys.exit(0)
    else:
        print("\n⚠️  AE THEORY INTEGRATION VALIDATION INCOMPLETE")
        print("   Please address issues before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
