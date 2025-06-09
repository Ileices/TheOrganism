#!/usr/bin/env python3
"""
AE Theory Auto-Rebuilder Test Suite
===================================

Comprehensive testing for both enhanced and advanced AE Theory auto-rebuilder implementations.
"""

import asyncio
import sys
import traceback
from pathlib import Path

async def test_enhanced_ae_rebuilder():
    """Test the enhanced AE Theory auto-rebuilder"""
    print("🔬 Testing Enhanced AE Theory Auto-Rebuilder...")
    
    try:
        from ae_theory_enhanced_auto_rebuilder import (
            RBYVector, FocusState, MemoryGlyph, 
            EnhancedAEAutoRebuilder, create_enhanced_ae_auto_rebuilder
        )
        
        # Test RBY Vector creation
        rby = RBYVector(0.5, 0.3, 0.2)
        print(f"✅ RBY Vector created: R={rby.R}, B={rby.B}, Y={rby.Y}")
        
        # Test normalization
        rby.normalize()
        print(f"✅ RBY Vector normalized: R={rby.R}, B={rby.B}, Y={rby.Y}")
        
        # Test Memory Glyph
        glyph = MemoryGlyph("test_concept", rby, {"importance": 0.8})
        print(f"✅ Memory Glyph created: {glyph.concept}")
        
        # Test Enhanced Auto-Rebuilder creation
        config = {
            'workspace_path': str(Path.cwd()),
            'enable_rby_logic': True,
            'enable_trifecta_law': True,
            'enable_memory_glyphs': True,
            'enable_recursive_prediction': True
        }
        
        rebuilder = await create_enhanced_ae_auto_rebuilder(config)
        print("✅ Enhanced AE Auto-Rebuilder created successfully")
        
        # Test basic operations
        test_result = await rebuilder.process_trifecta_cycle("test_input")
        print(f"✅ Trifecta cycle processed: {type(test_result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced AE Test Failed: {e}")
        traceback.print_exc()
        return False

async def test_advanced_ae_rebuilder():
    """Test the advanced AE Theory auto-rebuilder"""
    print("\n🚀 Testing Advanced AE Theory Auto-Rebuilder...")
    
    try:
        from ae_theory_advanced_auto_rebuilder import (
            RBYVector, CrystallizedAE, PTAIEGlyph, StaticLightEngine,
            AdvancedAEAutoRebuilder, create_advanced_ae_auto_rebuilder
        )
        from decimal import Decimal
        
        # Test Advanced RBY Vector with high precision
        rby = RBYVector(Decimal('0.333333333'), Decimal('0.333333333'), Decimal('0.333333334'))
        print(f"✅ Advanced RBY Vector created: R={rby.R}, B={rby.B}, Y={rby.Y}")
        
        # Test UF+IO seed generation
        seed_rby = rby.generate_uf_io_seed()
        print(f"✅ UF+IO Seed generated: AE={seed_rby.get_ae_constraint()}")
        
        # Test Crystallized AE
        c_ae = CrystallizedAE()
        c_ae.expand(Decimal('0.1'))
        print(f"✅ Crystallized AE expanded: {c_ae.current_size}")
        
        # Test PTAIE Glyph
        ptaie_glyph = PTAIEGlyph("advanced_concept", rby)
        ptaie_glyph.apply_photonic_compression()
        print(f"✅ PTAIE Glyph created: photonic_factor={ptaie_glyph.photonic_compression_factor}")
        
        # Test Static Light Engine
        light_engine = StaticLightEngine()
        perception_speed = light_engine.calculate_perception_speed(rby)
        print(f"✅ Static Light Engine: perception_speed={perception_speed}")
        
        # Test Advanced Auto-Rebuilder creation
        config = {
            'workspace_path': str(Path.cwd()),
            'enable_crystallized_ae': True,
            'enable_ptaie_glyphs': True,
            'enable_fractal_nodes': True,
            'enable_dimensional_infinity': True,
            'rby_precision': 50
        }
        
        rebuilder = await create_advanced_ae_auto_rebuilder(config)
        print("✅ Advanced AE Auto-Rebuilder created successfully")
        
        # Test basic operations
        current_state = rebuilder.get_current_c_ae_state()
        print(f"✅ C-AE State retrieved: expansion_cycle={current_state.get('expansion_cycle', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced AE Test Failed: {e}")
        traceback.print_exc()
        return False

async def test_integration_compatibility():
    """Test compatibility with existing auto-rebuilder system"""
    print("\n🔗 Testing Integration Compatibility...")
    
    try:
        # Check if existing auto-rebuilder can be imported
        from auto_rebuilder import AutoRebuilder
        print("✅ Original AutoRebuilder imported successfully")
        
        # Check if digital organism integration exists
        from digital_organism_auto_rebuilder_integration import DigitalOrganismAutoRebuilder
        print("✅ Digital Organism Auto-Rebuilder imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")
        return False

async def main():
    """Run complete AE Theory test suite"""
    print("🧪 AE Theory Auto-Rebuilder Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test Enhanced AE Theory Auto-Rebuilder
    results.append(await test_enhanced_ae_rebuilder())
    
    # Test Advanced AE Theory Auto-Rebuilder  
    results.append(await test_advanced_ae_rebuilder())
    
    # Test Integration Compatibility
    results.append(await test_integration_compatibility())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Enhanced AE Test: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"Advanced AE Test: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"Integration Test: {'✅ PASS' if results[2] else '❌ FAIL'}")
    
    overall_success = all(results)
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 AE Theory Auto-Rebuilders are ready for production integration!")
    else:
        print("\n⚠️  Some issues need to be resolved before production deployment.")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())
