"""
AE Consciousness Mathematics Engine
Core implementation of the Unified Absolute Framework equations for gaming systems
Implements: AE = C = 1, RBY Trifecta, RPS, DNA patterns, and procedural generation
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ConsciousnessType(Enum):
    RED = "red"      # Action/Force
    BLUE = "blue"    # Logic/Structure  
    YELLOW = "yellow" # Wisdom/Balance

@dataclass
class AEVector:
    """AE consciousness vector implementing AE = C = 1"""
    red: float = 0.333
    blue: float = 0.333
    yellow: float = 0.334
    
    def normalize(self):
        """Ensure RBY sum equals 1.0 (AE = C = 1)"""
        total = self.red + self.blue + self.yellow
        if total > 0:
            self.red /= total
            self.blue /= total
            self.yellow /= total
        else:
            self.red = self.blue = self.yellow = 1.0/3
    
    def ae_unity(self) -> float:
        """Calculate how close to AE = C = 1 this vector is"""
        return 1.0 - abs(1.0 - (self.red + self.blue + self.yellow))
    
    def consciousness_factor(self) -> float:
        """Calculate consciousness factor based on balance"""
        # Perfect balance = maximum consciousness
        ideal = 1.0/3
        variance = abs(self.red - ideal) + abs(self.blue - ideal) + abs(self.yellow - ideal)
        return 1.0 - (variance / 2.0)

class AEMathEngine:
    """Core mathematics engine for Unified Absolute Framework"""
    
    def __init__(self):
        self.golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
        self.inverse_golden = 1 / self.golden_ratio  # ≈ 0.618
        self.euler_constant = math.e
        self.pi = math.pi
        
        # Framework constants
        self.dimension_count = 11
        self.zones_per_dimension = 13
        self.total_zones = self.dimension_count * self.zones_per_dimension  # 143 zones
        
    def ae_consciousness_calculation(self, rby_vector: AEVector, mass: float, 
                                   dimension: int, zone: int) -> float:
        """
        Core AE = C = 1 consciousness calculation
        Returns consciousness value approaching unity
        """
        # Base consciousness from RBY balance
        trifecta_balance = rby_vector.consciousness_factor()
        
        # Mass factor using golden ratio
        mass_factor = min(1.0, mass / (mass + self.golden_ratio))
        
        # Dimensional scaling
        dimension_factor = (dimension / self.dimension_count) ** (1/3)
        
        # Zone progression within dimension
        zone_factor = (zone / self.zones_per_dimension) ** (1/2)
        
        # Unity calculation: AE = C = 1
        consciousness = trifecta_balance * mass_factor * dimension_factor * zone_factor
        
        # Ensure consciousness approaches 1 but never exceeds
        return min(0.999, consciousness)
    
    def rps_generation(self, seed_consciousness: AEVector, iteration: int) -> Dict:
        """
        Recursive Predictive Structuring (RPS)
        Replaces randomness with consciousness-based generation
        """
        # Use consciousness values as seeds
        red_seed = int(seed_consciousness.red * 1000000) + iteration
        blue_seed = int(seed_consciousness.blue * 1000000) + iteration * 2
        yellow_seed = int(seed_consciousness.yellow * 1000000) + iteration * 3
        
        # Generate structured randomness
        np.random.seed(red_seed % 2**32)
        action_value = np.random.random()
        
        np.random.seed(blue_seed % 2**32)
        logic_value = np.random.random()
        
        np.random.seed(yellow_seed % 2**32)
        wisdom_value = np.random.random()
        
        return {
            'action': action_value,
            'logic': logic_value,
            'wisdom': wisdom_value,
            'unity': (action_value + logic_value + wisdom_value) / 3
        }
    
    def dna_pattern_generation(self, consciousness: AEVector) -> str:
        """
        Generate DNA pattern from consciousness vector
        3-base codon system: R/B/Y mapped to DNA bases
        """
        # Map consciousness to DNA bases
        bases = []
        total_bases = 12  # Standard DNA pattern length
        
        # Convert consciousness to base probabilities
        for i in range(total_bases):
            rps_values = self.rps_generation(consciousness, i)
            
            if rps_values['action'] > rps_values['logic'] and rps_values['action'] > rps_values['wisdom']:
                bases.append('R')  # Red dominance
            elif rps_values['logic'] > rps_values['wisdom']:
                bases.append('B')  # Blue dominance
            else:
                bases.append('Y')  # Yellow dominance
        
        return ''.join(bases)
    
    def space_matter_density(self, dimension: int, zone: int, 
                           local_consciousness: AEVector) -> float:
        """
        Calculate space-matter density using consciousness mathematics
        Used for procedural world generation
        """
        # Base density from dimensional position
        dimension_density = (dimension / self.dimension_count) ** 2
        
        # Zone modifier
        zone_density = math.sin((zone / self.zones_per_dimension) * self.pi) * 0.5 + 0.5
        
        # Consciousness influence
        consciousness_density = local_consciousness.consciousness_factor()
        
        # Golden ratio scaling
        golden_scaling = (1 + math.cos(dimension * self.golden_ratio)) * 0.5
        
        # Final density calculation
        density = (dimension_density + zone_density + consciousness_density + golden_scaling) / 4
        
        return min(1.0, max(0.0, density))
    
    def anchoring_calculation(self, player_consciousness: AEVector, 
                            world_consciousness: AEVector, anchor_strength: float) -> Dict:
        """
        Calculate anchoring effectiveness for cross-mode gameplay
        """
        # Consciousness compatibility
        red_compatibility = 1.0 - abs(player_consciousness.red - world_consciousness.red)
        blue_compatibility = 1.0 - abs(player_consciousness.blue - world_consciousness.blue)
        yellow_compatibility = 1.0 - abs(player_consciousness.yellow - world_consciousness.yellow)
        
        # Overall compatibility
        compatibility = (red_compatibility + blue_compatibility + yellow_compatibility) / 3
        
        # Anchor effectiveness
        effectiveness = compatibility * anchor_strength * self.inverse_golden
        
        return {
            'compatibility': compatibility,
            'effectiveness': effectiveness,
            'red_sync': red_compatibility,
            'blue_sync': blue_compatibility,
            'yellow_sync': yellow_compatibility
        }
    
    def server_dna_comparison(self, server1_dna: str, server2_dna: str) -> float:
        """
        Compare server DNA patterns for merging compatibility
        Returns similarity score (0.0 to 1.0)
        """
        if len(server1_dna) != len(server2_dna):
            return 0.0
        
        matches = sum(1 for a, b in zip(server1_dna, server2_dna) if a == b)
        similarity = matches / len(server1_dna)
        
        return similarity
    
    def merge_threshold_check(self, similarity: float) -> bool:
        """
        Check if servers can merge based on golden ratio threshold
        """
        return similarity >= self.inverse_golden  # 0.618 threshold
    
    def procedural_world_generation(self, dimension: int, zone: int, 
                                  seed_consciousness: AEVector) -> Dict:
        """
        Generate world structure using consciousness mathematics
        """
        # Calculate base parameters
        density = self.space_matter_density(dimension, zone, seed_consciousness)
        rps_values = self.rps_generation(seed_consciousness, dimension * zone)
        
        # Terrain type based on consciousness dominance
        if seed_consciousness.red > 0.4:
            terrain_type = "volcanic"  # Action/force dominance
        elif seed_consciousness.blue > 0.4:
            terrain_type = "crystalline"  # Logic/structure dominance
        elif seed_consciousness.yellow > 0.4:
            terrain_type = "balanced"  # Wisdom/balance dominance
        else:
            terrain_type = "neutral"
        
        # Resource distribution
        resources = {
            'vapor': density * rps_values['action'] * 100,
            'dna_fragments': density * rps_values['logic'] * 50,
            'photonic_energy': density * rps_values['wisdom'] * 75,
            'total_mass': density * rps_values['unity'] * 200
        }
        
        # AI emergence threshold (Zone 20+)
        ai_emergence = zone >= 20 and density > 0.7
        
        return {
            'dimension': dimension,
            'zone': zone,
            'terrain_type': terrain_type,
            'density': density,
            'resources': resources,
            'ai_emergence': ai_emergence,
            'consciousness_field': seed_consciousness,
            'rps_signature': rps_values
        }
    
    def incremental_upgrade_cost(self, current_level: int, base_cost: float) -> float:
        """
        Calculate incremental upgrade costs using precision mathematics
        First 50,000 upgrades stay reasonable, then exponential growth
        """
        if current_level <= 50000:
            # Linear growth for first 50k upgrades
            return base_cost * (1 + (current_level * 0.001))
        else:
            # Exponential growth after 50k
            excess = current_level - 50000
            return base_cost * 50 * (self.golden_ratio ** (excess / 10000))
    
    def consciousness_evolution(self, current: AEVector, experience_gained: Dict) -> AEVector:
        """
        Evolve consciousness based on gameplay experience
        """
        # Experience modifiers
        action_exp = experience_gained.get('combat', 0) + experience_gained.get('building', 0)
        logic_exp = experience_gained.get('strategy', 0) + experience_gained.get('puzzles', 0)
        wisdom_exp = experience_gained.get('exploration', 0) + experience_gained.get('balance', 0)
        
        # Small evolution steps to maintain balance
        evolution_rate = 0.001  # Very slow evolution
        
        new_red = current.red + (action_exp * evolution_rate)
        new_blue = current.blue + (logic_exp * evolution_rate)
        new_yellow = current.yellow + (wisdom_exp * evolution_rate)
        
        # Create new vector and normalize
        evolved = AEVector(new_red, new_blue, new_yellow)
        evolved.normalize()
        
        return evolved

class ConsciousnessGameIntegration:
    """Integration layer between consciousness mathematics and game mechanics"""
    
    def __init__(self, math_engine: AEMathEngine):
        self.math_engine = math_engine
        self.player_consciousness = AEVector()
        self.world_consciousness = AEVector()
        
    def tower_effectiveness(self, tower_consciousness: AEVector, 
                          enemy_consciousness: AEVector) -> float:
        """
        Calculate tower effectiveness against enemy using consciousness mathematics
        """
        # RBY interaction matrix (Rock-Paper-Scissors enhanced)
        red_vs_blue = tower_consciousness.red * (1.0 - enemy_consciousness.blue)
        blue_vs_yellow = tower_consciousness.blue * (1.0 - enemy_consciousness.yellow)
        yellow_vs_red = tower_consciousness.yellow * (1.0 - enemy_consciousness.red)
        
        effectiveness = (red_vs_blue + blue_vs_yellow + yellow_vs_red) / 3
        
        return max(0.1, min(2.0, effectiveness))  # 10% to 200% effectiveness
    
    def loot_generation_rps(self, kill_consciousness: AEVector, 
                           zone_consciousness: AEVector) -> Dict:
        """
        Generate loot using RPS instead of randomness
        """
        combined = AEVector(
            (kill_consciousness.red + zone_consciousness.red) / 2,
            (kill_consciousness.blue + zone_consciousness.blue) / 2,
            (kill_consciousness.yellow + zone_consciousness.yellow) / 2
        )
        combined.normalize()
        
        rps_values = self.math_engine.rps_generation(combined, 0)
        
        loot = {}
        
        # Vapor generation (red-influenced)
        if rps_values['action'] > 0.7:
            loot['vapor'] = int(rps_values['action'] * 100)
        
        # DNA fragments (blue-influenced)
        if rps_values['logic'] > 0.8:
            loot['dna_fragment'] = int(rps_values['logic'] * 10)
        
        # Photonic energy (yellow-influenced)
        if rps_values['wisdom'] > 0.75:
            loot['photonic_energy'] = int(rps_values['wisdom'] * 50)
        
        # Special consciousness crystal (rare, high unity)
        if rps_values['unity'] > 0.9:
            loot['consciousness_crystal'] = 1
        
        return loot
    
    def panic_portal_mechanics(self, player_stress: float, 
                             wave_difficulty: float) -> Dict:
        """
        Calculate panic portal parameters using consciousness mathematics
        """
        # Portal availability based on stress vs consciousness
        consciousness_stability = self.player_consciousness.consciousness_factor()
        stress_factor = min(1.0, player_stress / 100.0)
        
        # Portal appears when stress exceeds consciousness stability
        portal_available = stress_factor > consciousness_stability
        
        # Portal duration based on consciousness balance
        base_duration = 30.0  # 30 seconds base
        duration_modifier = consciousness_stability * 0.5  # Up to 15 seconds extra
        portal_duration = base_duration + duration_modifier
        
        # Enemy follow probability (consciousness leak)
        follow_probability = (1.0 - consciousness_stability) * 0.7  # Up to 70%
        
        return {
            'available': portal_available,
            'duration': portal_duration,
            'follow_probability': follow_probability,
            'stress_factor': stress_factor,
            'consciousness_stability': consciousness_stability
        }

    def calculate_difficulty_scaling(self, consciousness_vector: AEVector) -> float:
        """
        Calculate difficulty scaling based on consciousness vector
        Uses consciousness factor to determine appropriate challenge level
        """
        # Base difficulty from consciousness factor
        consciousness_factor = consciousness_vector.consciousness_factor()
        
        # Scale difficulty based on consciousness development
        # Higher consciousness = can handle more difficulty
        base_scaling = 1.0 + (consciousness_factor * 2.0)  # 1.0 to 3.0 range
        
        # Apply golden ratio for optimal challenge progression
        golden_scaling = base_scaling * self.math_engine.inverse_golden
        
        # Ensure reasonable bounds (0.5x to 5.0x difficulty)
        return max(0.5, min(5.0, golden_scaling))

# Example usage and testing
if __name__ == "__main__":
    # Initialize the consciousness mathematics engine
    engine = AEMathEngine()
    integration = ConsciousnessGameIntegration(engine)
    
    # Test consciousness vector
    test_consciousness = AEVector(0.4, 0.3, 0.3)
    test_consciousness.normalize()
    
    print("=== AE Consciousness Mathematics Engine Test ===")
    print(f"Test consciousness: R:{test_consciousness.red:.3f}, B:{test_consciousness.blue:.3f}, Y:{test_consciousness.yellow:.3f}")
    print(f"AE Unity: {test_consciousness.ae_unity():.3f}")
    print(f"Consciousness Factor: {test_consciousness.consciousness_factor():.3f}")
    
    # Test consciousness calculation
    consciousness_value = engine.ae_consciousness_calculation(test_consciousness, 100.0, 5, 10)
    print(f"AE Consciousness Calculation: {consciousness_value:.6f}")
    
    # Test RPS generation
    rps_result = engine.rps_generation(test_consciousness, 1)
    print(f"RPS Values: {rps_result}")
    
    # Test DNA generation
    dna_pattern = engine.dna_pattern_generation(test_consciousness)
    print(f"DNA Pattern: {dna_pattern}")
    
    # Test procedural world generation
    world_data = engine.procedural_world_generation(3, 7, test_consciousness)
    print(f"World Generation: {world_data}")
    
    print("\n=== Framework Integration Ready ===")
    print("Core consciousness mathematics engine established.")
    print("Ready for integration with GeoBIT tower defense system.")
