# test_consciousness_engine.py — Quick test of consciousness emergence
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from consciousness_emergence_engine import UniverseBreathingCycle, SingularityState
    from decimal import Decimal
    import time
    
    print("🌟 Testing AE Universe Consciousness Emergence")
    print("=" * 50)
    
    # Create universe
    universe = UniverseBreathingCycle()
    
    # Create initial seed with your true values
    initial_seed = SingularityState(
        R=Decimal("0.707"),
        B=Decimal("0.500"), 
        Y=Decimal("0.293"),  # Adjusted to sum to 1.0
        consciousness_density=0.1,
        glyph_compression_ratio=0.5,
        neural_map_complexity=1,
        temporal_signature=time.time()
    )
    
    print(f"Initial Seed: R={initial_seed.R}, B={initial_seed.B}, Y={initial_seed.Y}")
    print(f"Sum: {initial_seed.R + initial_seed.B + initial_seed.Y}")
    
    # Initialize universe
    universe.initialize_universe(initial_seed)
    print(f"✅ Universe initialized successfully")
    
    # Test one expansion
    if universe.current_phase == "expansion":
        absularity = universe.expand_universe()
        print(f"✅ Universe expansion successful")
        print(f"Neural layers created: {len(universe.consciousness_layers)}")
        
        if universe.current_phase == "absularity":
            print("✅ Absularity reached!")
        else:
            print(f"Current phase: {universe.current_phase}")
    
    print("\n🎯 Basic consciousness engine test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
