#!/usr/bin/env python3
"""
Simple IC-AE Test
=================

Basic test to verify IC-AE system components work individually.
"""

import sys
import json
from pathlib import Path

# Test IC-AE Black Hole System
def test_ic_ae_system():
    print("Testing IC-AE Black Hole System...")
    try:
        from ic_ae_black_hole_fractal_system import ICBlackHoleSystem, ICAE
        
        # Create system
        ic_system = ICBlackHoleSystem()
        print("✓ IC-AE system created")
        
        # Create IC-AE instance
        ic_ae = ic_system.create_ic_ae("test_script")
        print("✓ IC-AE instance created")
        
        # Simple script injection
        test_script = "print('Hello IC-AE World!')\nfor i in range(5):\n    print(f'Iteration {i}')"
        ic_ae.inject_script(test_script)
        print("✓ Script injected")
        
        # Process a few cycles
        for i in range(3):
            result = ic_ae.process_cycle()
            print(f"✓ Cycle {i+1} completed")
            
        # Check state
        print(f"✓ IC-AE State: {ic_ae.state}")
        print(f"✓ Fractal Level: {ic_ae.current_fractal_level}")
        print(f"✓ Storage Used: {ic_ae.storage_used:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ IC-AE test failed: {e}")
        return False

# Test RBY Spectral Compressor
def test_rby_compressor():
    print("\nTesting RBY Spectral Compressor...")
    try:
        from advanced_rby_spectral_compressor import AdvancedRBYSpectralCompressor
        
        compressor = AdvancedRBYSpectralCompressor()
        print("✓ RBY compressor created")
        
        # Simple compression test
        test_text = "Hello RBY World! This is a test of spectral compression."
        result = compressor.compress_to_rby(test_text, bit_depth=8)
        print("✓ RBY compression completed")
        print(f"✓ Fractal level: {result['fractal_level']}")
        print(f"✓ Total bins: {result['total_bins']}")
        print(f"✓ Dimensions: {result['width']}x{result['height']}")
        
        return True
        
    except Exception as e:
        print(f"✗ RBY test failed: {e}")
        return False

# Test Twmrto Compressor
def test_twmrto_compressor():
    print("\nTesting Twmrto Compressor...")
    try:
        from twmrto_compression import TwmrtoCompressor
        
        # Create test directory
        test_dir = Path("simple_test_data")
        test_dir.mkdir(exist_ok=True)
        
        compressor = TwmrtoCompressor(str(test_dir))
        print("✓ Twmrto compressor created")
        
        # Simple compression test
        test_text = "The quick brown fox jumps over the lazy dog"
        result = compressor.compress_to_glyph(test_text)
        print("✓ Twmrto compression completed")
        print(f"✓ Original: '{test_text}'")
        print(f"✓ Glyph: '{result.get('glyph', 'N/A')}'")
        
        # Test reconstruction
        glyph = result.get('glyph', '')
        if glyph:
            reconstructed = compressor.reconstruct_from_glyph(glyph)
            print(f"✓ Reconstructed: '{reconstructed[:50]}...' ({len(reconstructed)} chars)")
        
        return True
        
    except Exception as e:
        print(f"✗ Twmrto test failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("IC-AE Simple Component Test")
    print("=" * 40)
    
    results = []
    
    # Test each component
    results.append(test_ic_ae_system())
    results.append(test_rby_compressor())
    results.append(test_twmrto_compressor())
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"✓ Passed: {sum(results)}/{len(results)}")
    print(f"✗ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All components working successfully!")
        print("Ready for full integration testing.")
    else:
        print("\n⚠️  Some components need attention.")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
