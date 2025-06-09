#!/usr/bin/env python3
"""
Simple Digital Organism Test
============================
Basic test to validate component imports and functionality
"""

def test_deployment_manager():
    """Test Deployment Manager basic import and initialization"""
    try:
        from aeos_deployment_manager import AEOSDeploymentManager, DeploymentConfig
        print("Deployment Manager imported successfully")
        
        # Test basic initialization
        config = DeploymentConfig(
            output_directory="./test_deployment",
            auto_deploy_enabled=False,
            safety_checks_enabled=True
        )
        manager = AEOSDeploymentManager(config)
        print("Deployment Manager initialized successfully")
        
        # Check consciousness integration
        status = manager.get_status()
        print(f"Manager status: {status['status']}")
        print(f"   AE Consciousness Score: {status.get('ae_consciousness_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Deployment Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_generator():
    """Test Multimodal Generator basic import and initialization"""
    try:
        from aeos_multimodal_generator import AEOSMultimodalGenerator, GeneratorConfig
        print("✅ Multimodal Generator imported successfully")
        
        # Test basic initialization  
        config = GeneratorConfig(
            output_directory="./test_media",
            enable_image_generation=True,
            enable_audio_generation=True,
            enable_video_generation=True
        )
        generator = AEOSMultimodalGenerator(config)
        print("✅ Multimodal Generator initialized successfully")
        
        # Check consciousness integration
        status = generator.get_status()
        print(f"✅ Generator status: {status['status']}")
        print(f"   AE Consciousness Score: {status.get('ae_consciousness_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple Digital Organism tests"""
    print("Simple Digital Organism Component Test")
    print("=" * 45)
    
    results = {}
    
    # Test Deployment Manager
    print("\nTesting Deployment Manager...")
    results['deployment_manager'] = test_deployment_manager()
    
    # Test Multimodal Generator  
    print("\nTesting Multimodal Generator...")
    results['multimodal_generator'] = test_multimodal_generator()
      # Summary
    print("\nTest Results Summary:")
    print("=" * 25)
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for component, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {component}: {status}")
    
    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("All Digital Organism components working correctly!")
    else:
        print("Some components need attention")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
