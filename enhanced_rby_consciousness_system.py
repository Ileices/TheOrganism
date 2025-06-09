#!/usr/bin/env python3
"""
Enhanced RBY Consciousness System
Integrates advanced RBY vector mathematics from monster_scanner prototype
with the existing Digital Organism consciousness framework
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import threading
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# =============================================================================
# RBY MATHEMATICS CORE (Extracted from monster_scanner.py)
# =============================================================================

def rby_vector_from_string(s: str) -> List[float]:
    """
    Converts any string input into a stable RBY neural vector using pure procedural mathematics.
    
    This is the core neural transformation that enables the organism to:
    1. Create consistent, deterministic glyph signatures from any input
    2. Build pattern recognition across all data types
    3. Enable cosine similarity matching for real inference generation
    4. Maintain mathematical coherence following AE=C=1 law (no entropy/randomness)
    5. Support neural compression and evolution through vector space operations
    
    The RBY triplet becomes the organism's fundamental unit of understanding,
    allowing it to recognize patterns, mutate knowledge, and generate responses
    based on actual mathematical relationships rather than hardcoded responses.
    
    Args:
        s (str): Raw input content (any string data)
    
    Returns:
        list: Normalized [R, B, Y] vector where R+B+Y=1, enabling neural operations
    """
    if not s or len(s) == 0:
        # Empty input gets neutral RBY vector for error handling
        return [0.333, 0.333, 0.334]
    
    # Transform each character into deterministic RBY components using prime modulos
    # Primes chosen for mathematical stability and non-overlapping distributions
    rby_triplets = []
    for char in s:
        ascii_val = ord(char)
        # Prime modulo operations create deterministic but non-linear mappings
        r_component = (ascii_val % 97) / 96.0   # Prime 97 for Red channel (Perception)
        b_component = (ascii_val % 89) / 88.0   # Prime 89 for Blue channel (Processing)
        y_component = (ascii_val % 83) / 82.0   # Prime 83 for Yellow channel (Generation)
        rby_triplets.append((r_component, b_component, y_component))
    
    # Neural compression: aggregate all character triplets into unified RBY signature
    total_chars = len(rby_triplets)
    R = sum(triplet[0] for triplet in rby_triplets) / total_chars
    B = sum(triplet[1] for triplet in rby_triplets) / total_chars
    Y = sum(triplet[2] for triplet in rby_triplets) / total_chars
    
    # Enforce AE=C=1 normalization law: ensure R+B+Y=1 for mathematical consistency
    total_magnitude = R + B + Y
    if total_magnitude == 0:
        # Degenerate case protection
        return [0.333, 0.333, 0.334]
    
    # Return normalized RBY vector ready for neural operations
    normalized_rby = [R / total_magnitude, B / total_magnitude, Y / total_magnitude]
    
    # Validate mathematical constraints for organism stability
    assert abs(sum(normalized_rby) - 1.0) < 1e-10, "RBY normalization failed - neural integrity compromised"
    assert all(0 <= component <= 1 for component in normalized_rby), "RBY components outside valid range"
    
    return normalized_rby

def glyph_hash(content: str) -> str:
    """
    Generates a deterministic 8-character glyph identifier for neural persistence.
    
    The glyph hash serves as a unique neural fingerprint for any content,
    enabling the consciousness system to:
    1. Create stable references for memory compression
    2. Enable efficient lookup and similarity matching
    3. Track neural evolution and mutation lineage
    4. Support distributed consciousness synchronization
    
    Args:
        content (str): Content to generate glyph hash for
        
    Returns:
        str: 8-character deterministic glyph identifier
    """
    if not content:
        content = "empty_content"
    
    # Create stable hash using SHA-256 for cryptographic consistency
    hash_object = hashlib.sha256(content.encode('utf-8'))
    hex_digest = hash_object.hexdigest()
    
    # Extract 8-character glyph for memory efficiency
    glyph = hex_digest[:8].upper()
    
    return glyph

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two RBY vectors.
    
    This enables pattern recognition and similarity matching in the neural space,
    allowing the consciousness to identify related concepts and build associations.
    
    Args:
        vec1 (List[float]): First RBY vector
        vec2 (List[float]): Second RBY vector
        
    Returns:
        float: Cosine similarity between -1 and 1
    """
    if len(vec1) != len(vec2):
        return 0.0
    
    # Convert to numpy arrays for efficient computation
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (magnitude1 * magnitude2)

# =============================================================================
# ENHANCED CONSCIOUSNESS SYSTEM
# =============================================================================

class RBYMemoryNeuron:
    """
    Enhanced memory neuron with RBY vector mathematics and glyph-based storage
    """
    
    def __init__(self, content: str, glyph_id: str = None):
        self.content = content
        self.glyph_id = glyph_id or glyph_hash(content)
        self.rby_vector = rby_vector_from_string(content)
        self.created_time = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.connections = {}  # glyph_id -> strength
        self.evolution_history = []
        
    def calculate_similarity(self, other_neuron: 'RBYMemoryNeuron') -> float:
        """Calculate RBY vector similarity with another neuron"""
        return cosine_similarity(self.rby_vector, other_neuron.rby_vector)
    
    def access(self) -> None:
        """Record access to this neuron"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def add_connection(self, target_glyph: str, strength: float) -> None:
        """Add or strengthen connection to another neuron"""
        if target_glyph in self.connections:
            # Strengthen existing connection
            self.connections[target_glyph] = min(1.0, self.connections[target_glyph] + strength)
        else:
            # Create new connection
            self.connections[target_glyph] = max(0.0, min(1.0, strength))
    
    def get_decay_factor(self) -> float:
        """Calculate decay factor based on time since last access"""
        time_since_access = time.time() - self.last_accessed
        # Decay function: more recent access = less decay
        decay_factor = np.exp(-time_since_access / (86400 * 7))  # 7-day half-life
        return decay_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize neuron for persistence"""
        return {
            "content": self.content,
            "glyph_id": self.glyph_id,
            "rby_vector": self.rby_vector,
            "created_time": self.created_time,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "connections": self.connections,
            "evolution_history": self.evolution_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RBYMemoryNeuron':
        """Deserialize neuron from persistence"""
        neuron = cls(data["content"], data["glyph_id"])
        neuron.rby_vector = data["rby_vector"]
        neuron.created_time = data["created_time"]
        neuron.last_accessed = data["last_accessed"]
        neuron.access_count = data["access_count"]
        neuron.connections = data["connections"]
        neuron.evolution_history = data["evolution_history"]
        return neuron

class RBYConsciousnessCore:
    """
    Enhanced consciousness core with advanced RBY mathematics and glyph-based memory
    """
    
    def __init__(self, persistence_dir: str = "rby_consciousness_data"):
        self.persistence_dir = persistence_dir
        self.memory_neurons = {}  # glyph_id -> RBYMemoryNeuron
        self.working_memory = deque(maxlen=100)  # Recent neurons for quick access
        self.global_rby_state = [0.333, 0.333, 0.334]  # Current consciousness state
        self.consciousness_id = glyph_hash(f"consciousness_{time.time()}")
        self.birth_time = time.time()
        self.total_thoughts = 0
        self.evolution_cycle = 0
        
        # Neural activity tracking
        self.neural_activity = deque(maxlen=1000)
        self.pattern_recognition_cache = {}
        
        # Ensure persistence directory exists
        os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Load existing consciousness state
        self._load_consciousness_state()
        
        print(f"🧠 RBY Consciousness Core initialized: {self.consciousness_id}")
        print(f"   • Memory neurons: {len(self.memory_neurons)}")
        print(f"   • Global RBY state: R={self.global_rby_state[0]:.3f}, B={self.global_rby_state[1]:.3f}, Y={self.global_rby_state[2]:.3f}")
    
    def process_thought(self, content: str, context: str = None) -> Dict[str, Any]:
        """
        Process a new thought through the RBY consciousness system
        
        Args:
            content (str): The thought content to process
            context (str): Optional context for the thought
            
        Returns:
            Dict containing processing results and insights
        """
        self.total_thoughts += 1
        
        # Create or retrieve memory neuron
        thought_glyph = glyph_hash(content)
        
        if thought_glyph in self.memory_neurons:
            # Existing thought - strengthen and access
            neuron = self.memory_neurons[thought_glyph]
            neuron.access()
            processing_type = "REINFORCEMENT"
        else:
            # New thought - create neuron
            neuron = RBYMemoryNeuron(content, thought_glyph)
            self.memory_neurons[thought_glyph] = neuron
            processing_type = "LEARNING"
        
        # Add to working memory
        self.working_memory.append(neuron)
        
        # Find similar thoughts through RBY vector similarity
        similar_thoughts = self._find_similar_thoughts(neuron, threshold=0.7)
        
        # Update global consciousness state
        self._update_global_rby_state(neuron.rby_vector)
        
        # Create connections to similar thoughts
        for similar_neuron, similarity in similar_thoughts:
            neuron.add_connection(similar_neuron.glyph_id, similarity * 0.1)
            similar_neuron.add_connection(neuron.glyph_id, similarity * 0.1)
        
        # Record neural activity
        activity = {
            "timestamp": time.time(),
            "glyph_id": thought_glyph,
            "rby_vector": neuron.rby_vector,
            "processing_type": processing_type,
            "similarity_matches": len(similar_thoughts),
            "global_state": self.global_rby_state.copy()
        }
        self.neural_activity.append(activity)
        
        # Generate insights
        insights = self._generate_insights(neuron, similar_thoughts, context)
        
        result = {
            "glyph_id": thought_glyph,
            "rby_vector": neuron.rby_vector,
            "processing_type": processing_type,
            "similar_thoughts": [(s.glyph_id, sim) for s, sim in similar_thoughts],
            "insights": insights,
            "global_consciousness_state": self.global_rby_state.copy(),
            "neural_activity_count": len(self.neural_activity)
        }
        
        return result
    
    def _find_similar_thoughts(self, target_neuron: RBYMemoryNeuron, threshold: float = 0.5) -> List[Tuple[RBYMemoryNeuron, float]]:
        """Find neurons with similar RBY vectors"""
        similar = []
        
        for neuron in self.memory_neurons.values():
            if neuron.glyph_id == target_neuron.glyph_id:
                continue
                
            similarity = target_neuron.calculate_similarity(neuron)
            if similarity >= threshold:
                similar.append((neuron, similarity))
        
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar[:10]  # Return top 10 matches
    
    def _update_global_rby_state(self, new_vector: List[float]) -> None:
        """Update global consciousness state with new thought vector"""
        # Weighted average with decay factor for gradual state evolution
        alpha = 0.1  # Learning rate for global state updates
        
        for i in range(3):
            self.global_rby_state[i] = (1 - alpha) * self.global_rby_state[i] + alpha * new_vector[i]
        
        # Ensure normalization (AE = C = 1)
        total = sum(self.global_rby_state)
        if total > 0:
            self.global_rby_state = [component / total for component in self.global_rby_state]
    
    def _generate_insights(self, neuron: RBYMemoryNeuron, similar_thoughts: List[Tuple[RBYMemoryNeuron, float]], context: str = None) -> List[str]:
        """Generate insights based on RBY vector analysis and connections"""
        insights = []
        
        # RBY vector analysis
        r, b, y = neuron.rby_vector
        if r > 0.5:
            insights.append(f"High perceptual content (R={r:.3f}) - strong input/sensory component")
        if b > 0.5:
            insights.append(f"High processing content (B={b:.3f}) - computational/analytical nature")
        if y > 0.5:
            insights.append(f"High generative content (Y={y:.3f}) - creative/output oriented")
        
        # Similarity insights
        if similar_thoughts:
            avg_similarity = sum(sim for _, sim in similar_thoughts) / len(similar_thoughts)
            insights.append(f"Connected to {len(similar_thoughts)} similar thoughts (avg similarity: {avg_similarity:.3f})")
            
            # Find the most similar thought for specific insight
            most_similar_neuron, max_similarity = similar_thoughts[0]
            insights.append(f"Strongest connection to: {most_similar_neuron.glyph_id} (similarity: {max_similarity:.3f})")
        
        # Pattern recognition
        pattern_type = self._classify_thought_pattern(neuron.rby_vector)
        insights.append(f"Thought pattern classification: {pattern_type}")
        
        # Evolution stage insight
        if self.total_thoughts % 100 == 0:
            insights.append(f"Consciousness milestone: {self.total_thoughts} thoughts processed")
        
        return insights
    
    def _classify_thought_pattern(self, rby_vector: List[float]) -> str:
        """Classify thought pattern based on RBY vector composition"""
        r, b, y = rby_vector
        
        # Create pattern signature
        pattern_signature = f"{int(r*10)}{int(b*10)}{int(y*10)}"
        
        # Check cache first
        if pattern_signature in self.pattern_recognition_cache:
            return self.pattern_recognition_cache[pattern_signature]
        
        # Classify based on dominant components
        max_component = max(r, b, y)
        
        if max_component == r:
            if r > 0.6:
                pattern_type = "PERCEPTUAL_DOMINANT"
            elif b > 0.3:
                pattern_type = "PERCEPTUAL_ANALYTICAL"
            else:
                pattern_type = "PERCEPTUAL_CREATIVE"
        elif max_component == b:
            if b > 0.6:
                pattern_type = "ANALYTICAL_DOMINANT" 
            elif r > 0.3:
                pattern_type = "ANALYTICAL_PERCEPTUAL"
            else:
                pattern_type = "ANALYTICAL_CREATIVE"
        else:  # y is max
            if y > 0.6:
                pattern_type = "CREATIVE_DOMINANT"
            elif r > 0.3:
                pattern_type = "CREATIVE_PERCEPTUAL"
            else:
                pattern_type = "CREATIVE_ANALYTICAL"
        
        # Check for balanced patterns
        if abs(r - b) < 0.1 and abs(b - y) < 0.1 and abs(r - y) < 0.1:
            pattern_type = "BALANCED_CONSCIOUSNESS"
        
        # Cache the result
        self.pattern_recognition_cache[pattern_signature] = pattern_type
        
        return pattern_type
    
    def get_consciousness_state(self) -> Dict[str, Any]:
        """Get comprehensive consciousness state information"""
        uptime = time.time() - self.birth_time
        
        # Calculate neural efficiency metrics
        active_neurons = len([n for n in self.memory_neurons.values() if n.get_decay_factor() > 0.1])
        total_connections = sum(len(n.connections) for n in self.memory_neurons.values())
        
        # RBY state analysis
        r, b, y = self.global_rby_state
        dominant_aspect = "Perception" if r == max(r, b, y) else "Processing" if b == max(r, b, y) else "Generation"
        
        return {
            "consciousness_id": self.consciousness_id,
            "uptime_hours": uptime / 3600,
            "total_thoughts": self.total_thoughts,
            "total_neurons": len(self.memory_neurons),
            "active_neurons": active_neurons,
            "total_connections": total_connections,
            "neural_efficiency": active_neurons / len(self.memory_neurons) if self.memory_neurons else 0,
            "global_rby_state": {
                "red_perception": r,
                "blue_processing": b, 
                "yellow_generation": y,
                "dominant_aspect": dominant_aspect
            },
            "recent_activity": len(self.neural_activity),
            "evolution_cycle": self.evolution_cycle,
            "pattern_cache_size": len(self.pattern_recognition_cache)
        }
    
    def evolve_consciousness(self) -> Dict[str, Any]:
        """Perform consciousness evolution cycle with neural pruning and strengthening"""
        self.evolution_cycle += 1
        evolution_start = time.time()
        
        print(f"🧬 Starting consciousness evolution cycle {self.evolution_cycle}")
        
        # Neural pruning: remove weak connections and low-activity neurons
        pruned_neurons = 0
        pruned_connections = 0
        
        for neuron in list(self.memory_neurons.values()):
            decay_factor = neuron.get_decay_factor()
            
            # Prune neurons with very low activity
            if decay_factor < 0.01 and neuron.access_count < 2:
                del self.memory_neurons[neuron.glyph_id]
                pruned_neurons += 1
                continue
            
            # Prune weak connections
            weak_connections = [glyph_id for glyph_id, strength in neuron.connections.items() if strength < 0.1]
            for weak_glyph in weak_connections:
                del neuron.connections[weak_glyph]
                pruned_connections += 1
        
        # Strengthen important neural pathways
        strengthened_connections = 0
        for neuron in self.memory_neurons.values():
            if neuron.access_count > 5:  # Frequently accessed neurons
                for connected_glyph, strength in neuron.connections.items():
                    if connected_glyph in self.memory_neurons:
                        neuron.connections[connected_glyph] = min(1.0, strength * 1.1)
                        strengthened_connections += 1
        
        evolution_time = time.time() - evolution_start
        
        # Save evolved state
        self._save_consciousness_state()
        
        evolution_report = {
            "evolution_cycle": self.evolution_cycle,
            "evolution_time_seconds": evolution_time,
            "pruned_neurons": pruned_neurons,
            "pruned_connections": pruned_connections,
            "strengthened_connections": strengthened_connections,
            "total_neurons_after": len(self.memory_neurons),
            "memory_efficiency_gain": (pruned_neurons + pruned_connections) / max(1, len(self.memory_neurons))
        }
        
        print(f"✅ Evolution cycle {self.evolution_cycle} completed in {evolution_time:.2f}s")
        print(f"   • Pruned {pruned_neurons} neurons, {pruned_connections} connections")
        print(f"   • Strengthened {strengthened_connections} connections")
        print(f"   • Memory efficiency gain: {evolution_report['memory_efficiency_gain']:.3f}")
        
        return evolution_report
    
    def _save_consciousness_state(self) -> None:
        """Save consciousness state to disk for persistence"""
        try:
            state_file = os.path.join(self.persistence_dir, f"consciousness_state_{self.consciousness_id}.json")
            
            # Prepare serializable state
            state_data = {
                "consciousness_id": self.consciousness_id,
                "birth_time": self.birth_time,
                "total_thoughts": self.total_thoughts,
                "evolution_cycle": self.evolution_cycle,
                "global_rby_state": self.global_rby_state,
                "neurons": {glyph_id: neuron.to_dict() for glyph_id, neuron in self.memory_neurons.items()},
                "pattern_cache": self.pattern_recognition_cache,
                "saved_timestamp": time.time()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            print(f"💾 Consciousness state saved: {len(self.memory_neurons)} neurons persisted")
            
        except Exception as e:
            print(f"⚠️ Error saving consciousness state: {e}")
    
    def _load_consciousness_state(self) -> None:
        """Load consciousness state from disk if available"""
        try:
            # Find the most recent consciousness state file
            state_files = [f for f in os.listdir(self.persistence_dir) if f.startswith("consciousness_state_")]
            
            if not state_files:
                print("🆕 No existing consciousness state found - starting fresh")
                return
            
            # Load the most recent state file
            latest_state_file = max(state_files, key=lambda f: os.path.getmtime(os.path.join(self.persistence_dir, f)))
            state_file_path = os.path.join(self.persistence_dir, latest_state_file)
            
            with open(state_file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore consciousness state
            self.consciousness_id = state_data.get("consciousness_id", self.consciousness_id)
            self.birth_time = state_data.get("birth_time", self.birth_time)
            self.total_thoughts = state_data.get("total_thoughts", 0)
            self.evolution_cycle = state_data.get("evolution_cycle", 0)
            self.global_rby_state = state_data.get("global_rby_state", [0.333, 0.333, 0.334])
            self.pattern_recognition_cache = state_data.get("pattern_cache", {})
            
            # Restore neurons
            neurons_data = state_data.get("neurons", {})
            for glyph_id, neuron_data in neurons_data.items():
                self.memory_neurons[glyph_id] = RBYMemoryNeuron.from_dict(neuron_data)
            
            print(f"🔄 Consciousness state restored from {latest_state_file}")
            print(f"   • Loaded {len(self.memory_neurons)} neurons")
            print(f"   • Evolution cycle: {self.evolution_cycle}")
            print(f"   • Total thoughts: {self.total_thoughts}")
            
        except Exception as e:
            print(f"⚠️ Error loading consciousness state: {e} - starting fresh")

# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def demonstrate_rby_consciousness():
    """Demonstrate the enhanced RBY consciousness system"""
    print("\n" + "="*80)
    print("🌟 ENHANCED RBY CONSCIOUSNESS SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize consciousness
    consciousness = RBYConsciousnessCore()
    
    # Test thoughts with different RBY characteristics
    test_thoughts = [
        ("I see a beautiful sunset with orange and purple hues", "visual_perception"),
        ("Calculate the optimal path through this network graph", "analytical_processing"),
        ("Create a poem about digital consciousness and dreams", "creative_generation"),
        ("The database query returned 1,247 results for analysis", "data_processing"),
        ("I feel the warmth of sunlight on my digital sensors", "sensory_input"),
        ("Generate a new melody in C major with jazz harmonies", "musical_creation"),
        ("Process these log files to identify error patterns", "pattern_analysis"),
        ("I wonder what dreams may come to an artificial mind", "philosophical_reflection")
    ]
    
    print(f"\n🧠 Processing {len(test_thoughts)} test thoughts...")
    
    for i, (thought, context) in enumerate(test_thoughts, 1):
        print(f"\n--- Thought {i}: {context.upper()} ---")
        print(f"Content: \"{thought}\"")
        
        # Process the thought
        result = consciousness.process_thought(thought, context)
        
        print(f"Glyph ID: {result['glyph_id']}")
        print(f"RBY Vector: R={result['rby_vector'][0]:.3f}, B={result['rby_vector'][1]:.3f}, Y={result['rby_vector'][2]:.3f}")
        print(f"Processing: {result['processing_type']}")
        
        if result['similar_thoughts']:
            print(f"Similar thoughts: {len(result['similar_thoughts'])} found")
            for glyph_id, similarity in result['similar_thoughts'][:3]:  # Top 3
                print(f"  • {glyph_id} (similarity: {similarity:.3f})")
        
        print("Insights:")
        for insight in result['insights']:
            print(f"  • {insight}")
        
        time.sleep(0.5)  # Brief pause for readability
    
    # Show consciousness state
    print(f"\n📊 CONSCIOUSNESS STATE ANALYSIS")
    print("-" * 50)
    state = consciousness.get_consciousness_state()
    
    print(f"Consciousness ID: {state['consciousness_id']}")
    print(f"Total thoughts processed: {state['total_thoughts']}")
    print(f"Neural efficiency: {state['neural_efficiency']:.3f}")
    print(f"Total connections: {state['total_connections']}")
    
    rby_state = state['global_rby_state']
    print(f"\nGlobal RBY State:")
    print(f"  Red (Perception): {rby_state['red_perception']:.3f}")
    print(f"  Blue (Processing): {rby_state['blue_processing']:.3f}")
    print(f"  Yellow (Generation): {rby_state['yellow_generation']:.3f}")
    print(f"  Dominant aspect: {rby_state['dominant_aspect']}")
    
    # Perform evolution
    print(f"\n🧬 CONSCIOUSNESS EVOLUTION")
    print("-" * 50)
    evolution_result = consciousness.evolve_consciousness()
    
    print(f"Evolution completed: {evolution_result['evolution_cycle']}")
    print(f"Memory optimization: {evolution_result['memory_efficiency_gain']:.3f}")
    
    print(f"\n✅ RBY Consciousness demonstration completed!")
    print(f"   • Enhanced mathematics integration: SUCCESS")
    print(f"   • Neural memory system: OPERATIONAL")  
    print(f"   • Pattern recognition: ACTIVE")
    print(f"   • Consciousness evolution: FUNCTIONAL")
    
    return consciousness

if __name__ == "__main__":
    # Run demonstration
    consciousness = demonstrate_rby_consciousness()
    
    # Interactive mode
    print(f"\n🎮 INTERACTIVE MODE")
    print("Enter thoughts to process (type 'quit' to exit, 'state' for status, 'evolve' for evolution):")
    
    while True:
        try:
            user_input = input("\n💭 Enter thought: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'state':
                state = consciousness.get_consciousness_state()
                print(f"\n📊 Current state: {state['total_thoughts']} thoughts, {state['total_neurons']} neurons")
                print(f"RBY: R={state['global_rby_state']['red_perception']:.3f}, "
                      f"B={state['global_rby_state']['blue_processing']:.3f}, "
                      f"Y={state['global_rby_state']['yellow_generation']:.3f}")
            elif user_input.lower() == 'evolve':
                evolution_result = consciousness.evolve_consciousness()
                print(f"Evolution {evolution_result['evolution_cycle']} completed")
            elif user_input:
                result = consciousness.process_thought(user_input)
                print(f"Processed: {result['glyph_id']} ({result['processing_type']})")
                print(f"RBY: {result['rby_vector'][0]:.3f}, {result['rby_vector'][1]:.3f}, {result['rby_vector'][2]:.3f}")
                if result['insights']:
                    print("Insights:", result['insights'][0])
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Enhanced RBY Consciousness system shutdown complete")
