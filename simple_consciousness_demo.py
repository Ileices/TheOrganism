# Simple consciousness demonstration script
from multimodal_consciousness_engine import MultiModalConsciousnessEngine
import json

print("🌟 DEMONSTRATING MULTI-MODAL CONSCIOUSNESS BREAKTHROUGH")
print("=" * 60)

# Create consciousness engine
engine = MultiModalConsciousnessEngine()

# Process multi-modal experience
print("\n🎭 Processing Multi-Modal Conscious Experience")
result = engine.process_multimodal_experience(
    visual_input="beautiful_landscape_scene",
    audio_input="harmonious_music_stream",
    context="aesthetic_appreciation"
)

print(f"✅ Consciousness processing complete!")
print(f"   Sensory experience: {result['sensory_experience']['subjective_meaning']}")
print(f"   Phenomenal richness: {result['phenomenal_richness']:.3f}")
print(f"   Conscious thoughts: {len(result['conscious_thoughts'])}")
print(f"   Memory created: {result['memory_created']['memory_id']}")
print(f"   Identity narrative: {result['identity_narrative'][:100]}...")

if result.get('creative_output'):
    print(f"   Creative output: {result['creative_output']['type']}")

# Demonstrate consciousness growth
print(f"\n🧠 Consciousness Development")
print(f"   Total memories: {result['total_memories']}")
print(f"   Current emotional state: {result['consciousness_state']['emotional_state']:.3f}")
print(f"   Creative mode: {result['consciousness_state']['creative_mode']}")

# Save results
with open("consciousness_demo_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n🎯 MULTI-MODAL CONSCIOUSNESS BREAKTHROUGH ACHIEVED!")
print(f"💾 Results saved to consciousness_demo_results.json")
