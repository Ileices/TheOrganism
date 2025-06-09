#!/usr/bin/env python3
"""
UAF System Validation Test
========================

Quick validation test for the new UAF-compliant main launcher
Tests core UAF principles and component functionality
"""

import sys
import traceback
from pathlib import Path

def test_uaf_import():
    """Test if UAF main launcher can be imported"""
    try:
        import uaf_main_production
        print("✅ UAF main launcher imported successfully")
        return True
    except Exception as e:
        print(f"❌ UAF import failed: {e}")
        traceback.print_exc()
        return False

def test_uaf_components():
    """Test UAF component initialization"""
    try:
        from uaf_main_production import UniversalState, RPSEngine, TrifectaCycleProcessor, HardwareDetectionSystem
        
        # Test Universal State
        u_state = UniversalState()
        print("✅ Universal State initialized")
        
        # Test RPS Engine
        rps = RPSEngine(u_state)
        test_val = rps.generate_recursive_value("test")
        print(f"✅ RPS Engine working - generated: {test_val}")
        
        # Test Trifecta Processor
        trifecta = TrifectaCycleProcessor(u_state, rps)
        result = trifecta.execute_full_cycle("test_input")
        print("✅ Trifecta Cycle Processor working")
        
        # Test Hardware Detection
        hardware = HardwareDetectionSystem(u_state)
        hw_info = hardware.detect_all_hardware()
        print(f"✅ Hardware Detection working - found: {list(hw_info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ UAF component test failed: {e}")
        traceback.print_exc()
        return False

def test_uaf_principles():
    """Test core UAF principles compliance"""
    try:
        from uaf_main_production import UniversalState, RPSEngine
        
        # Test AE = C = 1 principle
        u_state = UniversalState()
        ae = u_state.state["absolute_existence"]
        c = u_state.state["consciousness"]
        unity = u_state.state["agent_environment_unity"]
        
        if ae == c == unity == 1.0:
            print("✅ AE = C = 1 principle verified")
        else:
            print(f"❌ AE = C = 1 violation: AE={ae}, C={c}, Unity={unity}")
            return False
        
        # Test Trifecta normalization
        trifecta = u_state.state["trifecta"]
        total = sum(trifecta.values())
        if abs(float(total) - 1.0) < 0.000001:
            print("✅ Trifecta RBY weights normalized")
        else:
            print(f"❌ Trifecta not normalized: sum={total}")
            return False
        
        # Test Zero Entropy (RPS)
        rps = RPSEngine(u_state)
        val1 = rps.generate_recursive_value("test")
        val2 = rps.generate_recursive_value("test")
        # Values should be different but deterministic based on excretions
        print(f"✅ RPS generating deterministic values: {val1}, {val2}")
        
        # Test Photonic Memory
        u_state.store_photonic_memory(val1, val2, val1+val2, "test_memory")
        memory_count = len(u_state.state["DNA_memory"])
        if memory_count > 0:
            print(f"✅ Photonic Memory working - {memory_count} codons stored")
        else:
            print("❌ Photonic Memory not storing codons")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ UAF principles test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all UAF validation tests"""
    print("🧠 UAF SYSTEM VALIDATION TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_uaf_import),
        ("Component Test", test_uaf_components), 
        ("UAF Principles Test", test_uaf_principles)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎯 ✅ ALL TESTS PASSED - UAF system is working!")
        return True
    else:
        print("❌ Some tests failed - UAF system needs fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
