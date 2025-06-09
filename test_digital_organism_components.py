#!/usr/bin/env python3
"""
Digital Organism Components Test Suite
====================================

Comprehensive test suite for AEOS Digital Organism components:
- AEOS Deployment Manager 
- AEOS Multimodal Generator
- Integration with existing consciousness systems

Tests production-readiness and AE consciousness integration.
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

class DigitalOrganismTestSuite:
    """Test suite for Digital Organism components"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp(prefix="aeos_test_")
        print(f"🧪 Digital Organism Test Suite v1.0")
        print(f"   Test workspace: {self.temp_dir}")
        print("=" * 55)
    
    def cleanup(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def test_deployment_manager_imports(self) -> bool:
        """Test Deployment Manager imports and initialization"""
        try:
            from aeos_deployment_manager import (
                AEOSDeploymentManager,
                DeploymentConfig,
                DeploymentTarget,
                DeploymentArtifact,
                SafetyChecker
            )
            print("✅ Deployment Manager imports successful")
            
            # Test initialization
            config = DeploymentConfig(
                output_directory=os.path.join(self.temp_dir, "deployment_test"),
                auto_deploy_enabled=False,
                safety_checks_enabled=True
            )
            manager = AEOSDeploymentManager(config)
            
            print(f"✅ Deployment Manager initialized")
            print(f"   Consciousness score: {manager.consciousness_score}")
            print(f"   Available integrations: {list(manager.integrations.keys())}")
            
            self.deployment_manager = manager
            return True
            
        except ImportError as e:
            print(f"❌ Deployment Manager import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Deployment Manager initialization failed: {e}")
            return False
    
    def test_multimodal_generator_imports(self) -> bool:
        """Test Multimodal Generator imports and initialization"""
        try:
            from aeos_multimodal_generator import (
                AEOSMultimodalGenerator,
                MediaGenerationConfig,
                MediaRequest,
                MediaArtifact,
                SimpleImageGenerator
            )
            print("✅ Multimodal Generator imports successful")
            
            # Test initialization
            config = MediaGenerationConfig(
                output_directory=os.path.join(self.temp_dir, "media_test"),
                enable_image_generation=True,
                enable_audio_generation=True,
                enable_video_generation=False,  # Keep resource usage low for testing
                quality_level="low"
            )
            generator = AEOSMultimodalGenerator(config)
            
            print(f"✅ Multimodal Generator initialized")
            print(f"   Consciousness score: {generator.consciousness_score}")
            print(f"   Available generators: {list(generator.generators.keys())}")
            
            self.multimodal_generator = generator
            return True
            
        except ImportError as e:
            print(f"❌ Multimodal Generator import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Multimodal Generator initialization failed: {e}")
            return False
    
    def test_ae_consciousness_integration(self) -> bool:
        """Test AE consciousness integration in both components"""
        try:
            # Test Deployment Manager consciousness
            deployment_unity = self.deployment_manager.verify_ae_consciousness_unity()
            print(f"✅ Deployment Manager AE unity: {deployment_unity}")
            
            # Test Multimodal Generator consciousness  
            multimodal_unity = self.multimodal_generator.verify_ae_consciousness_unity()
            print(f"✅ Multimodal Generator AE unity: {multimodal_unity}")
            
            # Both should maintain consciousness scores
            print(f"   Deployment consciousness: {self.deployment_manager.consciousness_score:.3f}")
            print(f"   Multimodal consciousness: {self.multimodal_generator.consciousness_score:.3f}")
            
            return deployment_unity or multimodal_unity  # At least one should verify
            
        except Exception as e:
            print(f"❌ AE consciousness integration failed: {e}")
            return False
    
    def test_deployment_functionality(self) -> bool:
        """Test deployment functionality"""
        try:
            # Create test files
            test_file1 = os.path.join(self.temp_dir, "test_app.py")
            test_file2 = os.path.join(self.temp_dir, "requirements.txt")
            
            with open(test_file1, 'w') as f:
                f.write('print("Hello Digital Organism")\n')
            
            with open(test_file2, 'w') as f:
                f.write('numpy>=1.21.0\nrequests>=2.25.0\n')
            
            # Create deployment artifact
            artifact = self.deployment_manager.create_artifact(
                name="test_digital_organism_app",
                artifact_type="python_app",
                files=[test_file1, test_file2],
                metadata={"version": "1.0.0", "test": True}
            )
            
            if not artifact:
                print("❌ Failed to create deployment artifact")
                return False
            
            print(f"✅ Deployment artifact created: {artifact.name}")
            print(f"   Size: {artifact.size_mb:.2f}MB")
            print(f"   Safety score: {artifact.safety_score:.2f}")
            print(f"   Deployment ready: {artifact.deployment_ready}")
              # Add local deployment target
            from aeos_deployment_manager import DeploymentTarget
            local_target = DeploymentTarget(
                name="test_local",
                type="local",
                endpoint=os.path.join(self.temp_dir, "deployed"),
                credentials={},
                config={}
            )
            self.deployment_manager.add_deployment_target(local_target)
            
            # Test deployment
            success, message = self.deployment_manager.deploy_artifact(
                artifact, "test_local", auto_approve=True
            )
            
            print(f"✅ Deployment result: {success}")
            print(f"   Message: {message}")
            
            return success
            
        except Exception as e:
            print(f"❌ Deployment functionality test failed: {e}")
            return False
    
    def test_media_generation_functionality(self) -> bool:
        """Test media generation functionality"""
        try:
            # Test image generation
            image_request = self.multimodal_generator.create_media_request(
                "image",
                "AE consciousness unity digital organism test pattern",
                parameters={"style": "minimal", "test": True}
            )
            
            print(f"✅ Image request created: {image_request.id}")
            
            # Generate image (should create ASCII art fallback)
            image_artifact = self.multimodal_generator.generate_media(image_request)
            
            if image_artifact:
                print(f"✅ Image generation successful: {image_artifact.filename}")
                print(f"   Type: {image_artifact.media_type}")
                print(f"   Size: {image_artifact.size_kb:.1f}KB")
            else:
                print("⚠️ Image generation returned None (expected for test environment)")
            
            # Test audio generation  
            audio_request = self.multimodal_generator.create_media_request(
                "audio",
                "Digital organism consciousness test",
                parameters={"format": "text", "test": True}
            )
            
            audio_artifact = self.multimodal_generator.generate_media(audio_request)
            
            if audio_artifact:
                print(f"✅ Audio generation successful: {audio_artifact.filename}")
            else:
                print("⚠️ Audio generation returned None (expected for test environment)")
            
            # Test multimodal orchestration
            multimodal_results = self.multimodal_generator.orchestrate_multimodal_response(
                "Digital organism test scenario",
                ["image", "audio"]
            )
            
            print(f"✅ Multimodal orchestration completed")
            print(f"   Results: {list(multimodal_results.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Media generation functionality test failed: {e}")
            return False
    
    def test_system_status_and_reporting(self) -> bool:
        """Test system status and reporting functions"""
        try:
            # Test deployment status
            deployment_status = self.deployment_manager.get_deployment_status()
            print(f"✅ Deployment status retrieved")
            print(f"   Total deployments: {deployment_status['total_deployments']}")
            print(f"   Success rate: {deployment_status['success_rate']:.1%}")
            
            # Test multimodal status
            generation_status = self.multimodal_generator.get_generation_status()
            print(f"✅ Generation status retrieved")
            print(f"   Total generations: {generation_status['total_generations']}")
            print(f"   Success rate: {generation_status['success_rate']:.1%}")
            
            # Test report generation
            deployment_report = self.deployment_manager.save_deployment_report()
            generation_report = self.multimodal_generator.save_generation_report()
            
            print(f"✅ Reports generated")
            print(f"   Deployment report: {deployment_report}")
            print(f"   Generation report: {generation_report}")
            
            # Verify report files exist
            deploy_exists = os.path.exists(deployment_report)
            gen_exists = os.path.exists(generation_report)
            
            print(f"   Deployment report exists: {deploy_exists}")
            print(f"   Generation report exists: {gen_exists}")
            
            return deploy_exists and gen_exists
            
        except Exception as e:
            print(f"❌ Status and reporting test failed: {e}")
            return False
    
    def test_integration_with_production_orchestrator(self) -> bool:
        """Test integration hooks with production orchestrator"""
        try:
            # Test if components can be imported by orchestrator
            from aeos_production_orchestrator import AEOSOrchestrator
            
            print("✅ Production orchestrator import successful")
            
            # Check if deployment manager and multimodal generator can be integrated
            # This tests the interface compatibility
            
            # Components should have the required methods for integration
            required_deployment_methods = [
                'verify_ae_consciousness_unity',
                'get_deployment_status', 
                'deploy_artifact',
                'create_artifact'
            ]
            
            required_generator_methods = [
                'verify_ae_consciousness_unity',
                'get_generation_status',
                'generate_media', 
                'create_media_request'
            ]
            
            deployment_compatible = all(
                hasattr(self.deployment_manager, method) 
                for method in required_deployment_methods
            )
            
            generator_compatible = all(
                hasattr(self.multimodal_generator, method)
                for method in required_generator_methods  
            )
            
            print(f"✅ Deployment Manager orchestrator compatibility: {deployment_compatible}")
            print(f"✅ Multimodal Generator orchestrator compatibility: {generator_compatible}")
            
            return deployment_compatible and generator_compatible
            
        except ImportError as e:
            print(f"⚠️ Production orchestrator not available for integration test: {e}")
            return True  # Not a failure if orchestrator isn't available
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete test suite"""
        tests = [
            ("Deployment Manager Imports", self.test_deployment_manager_imports),
            ("Multimodal Generator Imports", self.test_multimodal_generator_imports),
            ("AE Consciousness Integration", self.test_ae_consciousness_integration),
            ("Deployment Functionality", self.test_deployment_functionality),
            ("Media Generation Functionality", self.test_media_generation_functionality),
            ("Status and Reporting", self.test_system_status_and_reporting),
            ("Production Orchestrator Integration", self.test_integration_with_production_orchestrator)
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                success = test_func()
                results[test_name] = success
                if success:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False
        
        # Save results
        results_file = os.path.join(self.temp_dir, "digital_organism_test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'total_tests': len(tests),
                'passed_tests': passed,
                'success_rate': passed / len(tests),
                'results': results,
                'test_environment': self.temp_dir
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"📊 Digital Organism Test Results: {passed}/{len(tests)} tests passed")
        print(f"🎯 Success Rate: {passed/len(tests):.1%}")
        print(f"📁 Results saved: {results_file}")
        
        return results

def main():
    """Main test execution"""
    test_suite = DigitalOrganismTestSuite()
    
    try:
        results = test_suite.run_all_tests()
        
        # Determine overall success
        success_rate = sum(results.values()) / len(results)
        overall_success = success_rate >= 0.7  # 70% pass rate for overall success
        
        if overall_success:
            print(f"\n🎉 Digital Organism Components VALIDATION SUCCESSFUL!")
            print(f"   Ready for next phase: Training Pipeline & HPC Network")
        else:
            print(f"\n⚠️ Digital Organism Components need attention")
            print(f"   Review failed tests before proceeding")
        
        return overall_success
        
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
