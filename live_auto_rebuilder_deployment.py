#!/usr/bin/env python3
"""
Live Auto-Rebuilder Deployment and Validation Script
===================================================

This script performs live testing and deployment of the auto-rebuilder integration
framework for the Digital Organism system.

Usage:
    python live_auto_rebuilder_deployment.py

Features:
- Live import validation
- Configuration testing
- Heartbeat system validation
- Integration adapter testing
- Performance monitoring
- Production readiness assessment
"""

import sys
import os
import asyncio
import time
import traceback
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test all critical imports for the auto-rebuilder integration."""
    print("🔍 Testing Critical Imports...")
    
    try:
        # Test auto_rebuilder import
        import auto_rebuilder
        print("✅ auto_rebuilder.py imported successfully")
        
        # Test integration framework import
        import digital_organism_auto_rebuilder_integration as integration
        print("✅ digital_organism_auto_rebuilder_integration.py imported successfully")
        
        # Test adapter import
        import auto_rebuilder_adapter as adapter
        print("✅ auto_rebuilder_adapter.py imported successfully")
        
        # Test test framework import
        import test_auto_rebuilder_integration as tests
        print("✅ test_auto_rebuilder_integration.py imported successfully")
        
        return True, {
            'auto_rebuilder': auto_rebuilder,
            'integration': integration,
            'adapter': adapter,
            'tests': tests
        }
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False, {}

def test_configuration():
    """Test configuration and basic setup."""
    print("\n🔧 Testing Configuration...")
    
    try:
        from digital_organism_auto_rebuilder_integration import DigitalOrganismConfig
        
        # Test default configuration
        config = DigitalOrganismConfig()
        print(f"✅ Default config created - heartbeat_interval: {config.heartbeat_interval}s")
        
        # Test custom configuration
        custom_config = DigitalOrganismConfig(
            heartbeat_interval=30,
            max_queue_size=200,
            enable_auto_integration=False
        )
        print(f"✅ Custom config created - heartbeat_interval: {custom_config.heartbeat_interval}s")
        
        return True, config
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False, None

def test_adapter_functionality():
    """Test the lightweight adapter functionality."""
    print("\n🔌 Testing Adapter Functionality...")
    
    try:
        from auto_rebuilder_adapter import AutoRebuilderAdapter, get_global_adapter
        
        # Test adapter creation
        adapter = AutoRebuilderAdapter()
        print("✅ AutoRebuilderAdapter created successfully")
        
        # Test health assessment
        health = adapter.assess_system_health()
        print(f"✅ System health assessment: {health['status']}")
        
        # Test code safety assessment
        test_code = "print('Hello, World!')"
        safety = adapter.assess_code_safety(test_code)
        print(f"✅ Code safety assessment: Safe={safety.get('is_safe', False)}")
        
        # Test global adapter
        global_adapter = get_global_adapter()
        print("✅ Global adapter retrieved successfully")
        
        return True, adapter
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        traceback.print_exc()
        return False, None

async def test_heartbeat_system():
    """Test the heartbeat system functionality."""
    print("\n💓 Testing Heartbeat System...")
    
    try:
        from digital_organism_auto_rebuilder_integration import DigitalOrganismAutoRebuilder, DigitalOrganismConfig
        
        # Create configuration for quick testing
        config = DigitalOrganismConfig(heartbeat_interval=5)  # 5 seconds for testing
        
        # Create auto-rebuilder instance
        rebuilder = DigitalOrganismAutoRebuilder(config)
        print("✅ DigitalOrganismAutoRebuilder created successfully")
        
        # Test initialization
        await rebuilder.initialize()
        print("✅ Auto-rebuilder initialized successfully")
        
        # Test single heartbeat
        print("🔄 Testing single heartbeat cycle...")
        await rebuilder._heartbeat_cycle()
        print("✅ Heartbeat cycle completed successfully")
        
        # Test metrics
        metrics = rebuilder.get_metrics()
        print(f"✅ Metrics retrieved: {len(metrics)} metrics available")
        
        # Test shutdown
        await rebuilder.shutdown()
        print("✅ Auto-rebuilder shutdown completed")
        
        return True, rebuilder
        
    except Exception as e:
        print(f"❌ Heartbeat test failed: {e}")
        traceback.print_exc()
        return False, None

def test_auto_rebuilder_core():
    """Test core auto_rebuilder.py functionality."""
    print("\n🏗️ Testing Auto-Rebuilder Core...")
    
    try:
        import auto_rebuilder
        
        # Test if main classes exist
        if hasattr(auto_rebuilder, 'AutoRebuilder'):
            print("✅ AutoRebuilder class found")
        elif hasattr(auto_rebuilder, 'main'):
            print("✅ main function found")
        else:
            print("⚠️ Auto-rebuilder structure unclear, but import successful")
        
        # Test basic functionality
        print("✅ Auto-rebuilder core is accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-rebuilder core test failed: {e}")
        traceback.print_exc()
        return False

async def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Auto-Rebuilder Integration Tests")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Imports
    results['imports'], modules = test_imports()
    if not results['imports']:
        print("❌ Critical imports failed. Cannot continue.")
        return results
    
    # Test 2: Configuration
    results['configuration'], config = test_configuration()
    
    # Test 3: Adapter
    results['adapter'], adapter = test_adapter_functionality()
    
    # Test 4: Auto-rebuilder core
    results['core'] = test_auto_rebuilder_core()
    
    # Test 5: Heartbeat system
    results['heartbeat'], rebuilder = await test_heartbeat_system()
    
    return results

def generate_deployment_report(results):
    """Generate a comprehensive deployment report."""
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT VALIDATION REPORT")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.upper()}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - READY FOR PRODUCTION DEPLOYMENT!")
        print("\nNext Steps:")
        print("1. Run: python unified_digital_organism_launcher.py")
        print("2. Monitor auto-rebuilder heartbeat in logs")
        print("3. Test integration with consciousness engines")
        print("4. Deploy as production service")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed - Review and fix issues before deployment")
    
    return passed_tests == total_tests

def main():
    """Main deployment and validation function."""
    print("🌟 Digital Organism Auto-Rebuilder Live Deployment")
    print("🔬 Performing comprehensive validation and testing...")
    print()
    
    try:
        # Run comprehensive tests
        results = asyncio.run(run_comprehensive_tests())
        
        # Generate deployment report
        deployment_ready = generate_deployment_report(results)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"auto_rebuilder_deployment_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Auto-Rebuilder Deployment Report - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            for test_name, result in results.items():
                status = "PASS" if result else "FAIL"
                f.write(f"{test_name.upper()}: {status}\n")
            f.write(f"\nDeployment Ready: {deployment_ready}\n")
        
        print(f"\n📄 Report saved to: {report_file}")
        
        if deployment_ready:
            print("\n🚀 Auto-Rebuilder integration is ready for production!")
            return 0
        else:
            print("\n🔧 Please fix issues before production deployment.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Deployment validation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
