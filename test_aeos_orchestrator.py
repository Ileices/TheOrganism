#!/usr/bin/env python3
"""
AEOS Production Orchestrator Test & Validation
============================================

Test script to validate the production orchestrator and advance
to implementing the next phase of Digital Organism components.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import psutil
        print("✅ psutil imported successfully")
    except ImportError as e:
        print(f"❌ psutil import failed: {e}")
        return False
        
    try:
        # Test orchestrator imports
        from aeos_production_orchestrator import AEOSConfiguration, ComponentStatus
        print("✅ AEOS configuration classes imported")
    except ImportError as e:
        print(f"❌ AEOS configuration import failed: {e}")
        return False
        
    try:
        from aeos_production_orchestrator import AEOSOrchestrator
        print("✅ AEOS orchestrator imported")
    except ImportError as e:
        print(f"❌ AEOS orchestrator import failed: {e}")
        return False
        
    return True

def test_configuration():
    """Test configuration system"""
    print("\n🧪 Testing configuration system...")
    
    try:
        from aeos_production_orchestrator import AEOSConfiguration
        config = AEOSConfiguration()
        
        print(f"✅ Configuration created with {len(config.required_components)} components")
        print(f"   Required files: {len(config.required_files)}")
        print(f"   Compression threshold: {config.compression_threshold}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_orchestrator_init():
    """Test orchestrator initialization"""
    print("\n🧪 Testing orchestrator initialization...")
    
    try:
        from aeos_production_orchestrator import AEOSOrchestrator, AEOSConfiguration
        
        config = AEOSConfiguration()
        orchestrator = AEOSOrchestrator(config)
        
        print("✅ Orchestrator initialized successfully")
        
        # Test dependency verification
        missing_deps = orchestrator.verify_dependencies()
        print(f"   Missing dependencies: {len(missing_deps)}")
        
        if missing_deps:
            print("   Missing:")
            for dep in missing_deps[:5]:  # Show first 5
                print(f"     - {dep}")
        
        return True
    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_existing_systems():
    """Test integration with existing consciousness systems"""
    print("\n🧪 Testing existing systems integration...")
    
    # Check for key files
    key_files = [
        "ae_ptaie_consciousness_integration.py",
        "consciousness_emergence_engine.py",
        "multimodal_consciousness_engine.py",
        "ae_universe_launcher.py"
    ]
    
    existing_files = []
    for file in key_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
    
    print(f"   Found {len(existing_files)}/{len(key_files)} key files")
    return len(existing_files) > 0

def main():
    """Main test function"""
    print("🌌 AEOS Production Orchestrator Validation")
    print("   Testing Digital Organism System Components")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Orchestrator Initialization", test_orchestrator_init),
        ("Existing Systems Check", test_existing_systems)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Ready for next phase implementation!")
        return True
    else:
        print("⚠️ Some tests failed - Need to fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
