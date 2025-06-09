"""
Procedural World Generation for Unified Absolute Framework
Generates 11 dimensions × 13 zones using consciousness mathematics
Implements Big Bang transitions, AI emergence, and density-based generation
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from ae_consciousness_mathematics import AEMathEngine, AEVector, ConsciousnessGameIntegration

class ZoneType(Enum):
    VOID = "void"
    LOW_DENSITY = "low_density"
    MEDIUM_DENSITY = "medium_density"
    HIGH_DENSITY = "high_density"
    CONSCIOUSNESS_NEXUS = "consciousness_nexus"
    AI_EMERGENCE = "ai_emergence"
    PORTAL_HUB = "portal_hub"

class TerrainFeature(Enum):
    VOLCANIC = "volcanic"
    CRYSTALLINE = "crystalline"
    BALANCED = "balanced"
    NEUTRAL = "neutral"
    QUANTUM_FOAM = "quantum_foam"
    CONSCIOUSNESS_FIELD = "consciousness_field"

@dataclass
class ZoneData:
    """Complete zone information"""
    dimension: int
    zone: int
    zone_type: ZoneType
    terrain_features: List[TerrainFeature]
    consciousness_field: AEVector
    density: float
    resources: Dict[str, float]
    ai_presence: bool
    ai_advancement_level: int
    portal_connections: List[Tuple[int, int]]  # (dimension, zone) pairs
    environmental_effects: Dict[str, float]
    discovery_difficulty: float
    
@dataclass
class DimensionData:
    """Complete dimension information"""
    dimension_id: int
    zones: Dict[int, ZoneData]
    dimension_consciousness: AEVector
    dimensional_theme: str
    big_bang_center: Tuple[float, float]
    expansion_rate: float
    stability: float

class ProceduralWorldGenerator:
    """Generates procedural worlds using consciousness mathematics"""
    
    def __init__(self, math_engine: AEMathEngine):
        self.math_engine = math_engine
        self.consciousness_integration = ConsciousnessGameIntegration(math_engine)
        
        # World structure
        self.dimensions: Dict[int, DimensionData] = {}
        self.total_dimensions = 11
        self.zones_per_dimension = 13
        
        # Generation parameters
        self.world_seed = 42  # Base seed for consistency
        self.consciousness_variance = 0.3  # How much consciousness can vary
        
        # AI emergence parameters
        self.ai_emergence_threshold = 20  # Zone 20+ in any dimension
        self.ai_advancement_per_zone = 5   # AI advancement per zone beyond threshold
        
        # Portal system
        self.portal_probability = 0.1  # 10% chance per zone
        self.max_portals_per_zone = 3
        
        # Generate all dimensions and zones
        self.generate_universe()
    
    def generate_universe(self):
        """Generate the complete 11×13 universe structure"""
        print("Generating Unified Absolute Framework universe...")
        
        for dimension_id in range(1, self.total_dimensions + 1):
            dimension_data = self.generate_dimension(dimension_id)
            self.dimensions[dimension_id] = dimension_data
            
        # Generate inter-dimensional connections after all zones exist
        self.generate_portal_network()
        
        print(f"Universe generated: {self.total_dimensions} dimensions, "
              f"{self.total_dimensions * self.zones_per_dimension} total zones")
    
    def generate_dimension(self, dimension_id: int) -> DimensionData:
        """Generate a complete dimension with all its zones"""
        # Generate dimensional consciousness
        np.random.seed(self.world_seed + dimension_id)
        
        # Dimensional themes based on consciousness dominance
        dimension_consciousness = self.generate_dimensional_consciousness(dimension_id)
        dimensional_theme = self.determine_dimensional_theme(dimension_consciousness)
        
        # Big Bang center for this dimension
        big_bang_center = (
            np.random.uniform(-1000, 1000),
            np.random.uniform(-1000, 1000)
        )
        
        # Expansion rate based on consciousness
        expansion_rate = dimension_consciousness.consciousness_factor() * 100 + 50
        
        # Dimensional stability
        stability = dimension_consciousness.ae_unity()
        
        # Generate all zones in this dimension
        zones = {}
        for zone_id in range(1, self.zones_per_dimension + 1):
            zone_data = self.generate_zone(dimension_id, zone_id, dimension_consciousness)
            zones[zone_id] = zone_data
        
        return DimensionData(
            dimension_id=dimension_id,
            zones=zones,
            dimension_consciousness=dimension_consciousness,
            dimensional_theme=dimensional_theme,
            big_bang_center=big_bang_center,
            expansion_rate=expansion_rate,
            stability=stability
        )
    
    def generate_dimensional_consciousness(self, dimension_id: int) -> AEVector:
        """Generate consciousness signature for a dimension"""
        # Use dimension ID to create unique but reproducible consciousness
        np.random.seed(self.world_seed + dimension_id * 1000)
        
        # Base consciousness with dimensional variation
        base_red = 0.333 + np.random.uniform(-self.consciousness_variance, self.consciousness_variance)
        base_blue = 0.333 + np.random.uniform(-self.consciousness_variance, self.consciousness_variance)
        base_yellow = 0.334 + np.random.uniform(-self.consciousness_variance, self.consciousness_variance)
        
        consciousness = AEVector(base_red, base_blue, base_yellow)
        consciousness.normalize()
        
        return consciousness
    
    def determine_dimensional_theme(self, consciousness: AEVector) -> str:
        """Determine dimensional theme based on consciousness dominance"""
        if consciousness.red > 0.4:
            return "Action Dimension"  # High energy, combat-focused
        elif consciousness.blue > 0.4:
            return "Logic Dimension"   # Structured, puzzle-based
        elif consciousness.yellow > 0.4:
            return "Wisdom Dimension"  # Balanced, exploration-focused
        else:
            return "Neutral Dimension" # Mixed characteristics
    
    def generate_zone(self, dimension_id: int, zone_id: int, 
                     dimension_consciousness: AEVector) -> ZoneData:
        """Generate a specific zone within a dimension"""
        # Zone-specific seed
        np.random.seed(self.world_seed + dimension_id * 1000 + zone_id * 100)
        
        # Generate zone consciousness (variation of dimensional consciousness)
        zone_consciousness = self.generate_zone_consciousness(dimension_consciousness, zone_id)
        
        # Calculate zone density
        density = self.math_engine.space_matter_density(
            dimension_id, zone_id, zone_consciousness
        )
        
        # Determine zone type based on density and position
        zone_type = self.determine_zone_type(density, zone_id)
        
        # Generate terrain features
        terrain_features = self.generate_terrain_features(zone_consciousness, density)
        
        # Check for AI emergence
        ai_presence = zone_id >= self.ai_emergence_threshold
        ai_advancement_level = max(0, (zone_id - self.ai_emergence_threshold) * 
                                 self.ai_advancement_per_zone) if ai_presence else 0
        
        # Generate resources using RPS
        resources = self.generate_zone_resources(zone_consciousness, density, zone_id)
        
        # Environmental effects
        environmental_effects = self.generate_environmental_effects(
            zone_consciousness, density, ai_presence
        )
        
        # Discovery difficulty
        discovery_difficulty = self.calculate_discovery_difficulty(
            dimension_id, zone_id, density
        )
        
        return ZoneData(
            dimension=dimension_id,
            zone=zone_id,
            zone_type=zone_type,
            terrain_features=terrain_features,
            consciousness_field=zone_consciousness,
            density=density,
            resources=resources,
            ai_presence=ai_presence,
            ai_advancement_level=ai_advancement_level,
            portal_connections=[],  # Filled later by portal network generation
            environmental_effects=environmental_effects,
            discovery_difficulty=discovery_difficulty
        )
    
    def generate_zone_consciousness(self, dimension_consciousness: AEVector, 
                                  zone_id: int) -> AEVector:
        """Generate zone-specific consciousness"""
        # Zone progression affects consciousness
        zone_progression = zone_id / self.zones_per_dimension
        
        # Progressive shift toward balance as zones advance
        balance_factor = zone_progression * 0.2
        
        red = dimension_consciousness.red + np.random.uniform(-0.1, 0.1)
        blue = dimension_consciousness.blue + np.random.uniform(-0.1, 0.1)
        yellow = dimension_consciousness.yellow + np.random.uniform(-0.1, 0.1)
        
        # Apply balance factor for higher zones
        if zone_id > 8:
            target_balance = 1.0 / 3
            red += (target_balance - red) * balance_factor
            blue += (target_balance - blue) * balance_factor
            yellow += (target_balance - yellow) * balance_factor
        
        consciousness = AEVector(red, blue, yellow)
        consciousness.normalize()
        
        return consciousness
    
    def determine_zone_type(self, density: float, zone_id: int) -> ZoneType:
        """Determine zone type based on density and position"""
        if zone_id >= self.ai_emergence_threshold:
            return ZoneType.AI_EMERGENCE
        elif density < 0.2:
            return ZoneType.VOID
        elif density < 0.4:
            return ZoneType.LOW_DENSITY
        elif density < 0.7:
            return ZoneType.MEDIUM_DENSITY
        elif density < 0.9:
            return ZoneType.HIGH_DENSITY
        else:
            return ZoneType.CONSCIOUSNESS_NEXUS
    
    def generate_terrain_features(self, consciousness: AEVector, 
                                density: float) -> List[TerrainFeature]:
        """Generate terrain features for a zone"""
        features = []
        
        # Primary feature based on consciousness dominance
        if consciousness.red > 0.4:
            features.append(TerrainFeature.VOLCANIC)
        elif consciousness.blue > 0.4:
            features.append(TerrainFeature.CRYSTALLINE)
        elif consciousness.yellow > 0.4:
            features.append(TerrainFeature.BALANCED)
        else:
            features.append(TerrainFeature.NEUTRAL)
        
        # Additional features based on density
        if density > 0.8:
            features.append(TerrainFeature.CONSCIOUSNESS_FIELD)
        elif density < 0.3:
            features.append(TerrainFeature.QUANTUM_FOAM)
        
        return features
    
    def generate_zone_resources(self, consciousness: AEVector, 
                              density: float, zone_id: int) -> Dict[str, float]:
        """Generate resources using RPS instead of randomness"""
        rps_values = self.math_engine.rps_generation(consciousness, zone_id)
        
        # Base resource multiplier
        base_multiplier = density * 100
        
        # Zone progression multiplier
        zone_multiplier = 1 + (zone_id * 0.1)
        
        resources = {
            'vapor': rps_values['action'] * base_multiplier * zone_multiplier,
            'dna_fragments': rps_values['logic'] * base_multiplier * 0.5 * zone_multiplier,
            'photonic_energy': rps_values['wisdom'] * base_multiplier * 0.75 * zone_multiplier,
            'total_mass': rps_values['unity'] * base_multiplier * 2 * zone_multiplier,
            'genetic_material': 0  # Special resource for Game of Life mode
        }
        
        # Special resources for high-level zones
        if zone_id >= 10:
            resources['consciousness_crystals'] = rps_values['unity'] * 10
        
        if zone_id >= self.ai_emergence_threshold:
            resources['ai_cores'] = (zone_id - self.ai_emergence_threshold + 1) * 2
            resources['genetic_material'] = rps_values['unity'] * 20
        
        return resources
    
    def generate_environmental_effects(self, consciousness: AEVector, 
                                     density: float, ai_presence: bool) -> Dict[str, float]:
        """Generate environmental effects for the zone"""
        effects = {}
        
        # Consciousness-based effects
        if consciousness.red > 0.5:
            effects['damage_amplification'] = consciousness.red * 0.5
            effects['movement_speed_bonus'] = consciousness.red * 0.3
        
        if consciousness.blue > 0.5:
            effects['logic_enhancement'] = consciousness.blue * 0.4
            effects['strategy_bonus'] = consciousness.blue * 0.2
        
        if consciousness.yellow > 0.5:
            effects['healing_rate_bonus'] = consciousness.yellow * 0.6
            effects['experience_multiplier'] = 1 + consciousness.yellow * 0.5
        
        # Density-based effects
        if density > 0.8:
            effects['resource_generation_bonus'] = density * 0.5
            effects['consciousness_evolution_rate'] = density * 0.3
        elif density < 0.3:
            effects['stealth_bonus'] = (1 - density) * 0.7
            effects['portal_stability'] = (1 - density) * 0.4
        
        # AI presence effects
        if ai_presence:
            effects['ai_adaptation_rate'] = 0.1
            effects['enemy_intelligence_boost'] = 0.5
            effects['technology_discovery_rate'] = 0.3
        
        return effects
    
    def calculate_discovery_difficulty(self, dimension: int, zone: int, 
                                     density: float) -> float:
        """Calculate how difficult it is to discover/access this zone"""
        # Base difficulty increases with dimension and zone
        base_difficulty = (dimension * 0.1) + (zone * 0.05)
        
        # Density affects difficulty
        density_modifier = abs(0.5 - density) * 0.5  # Extreme densities are harder
        
        # Final difficulty (0.0 to 1.0)
        difficulty = min(1.0, base_difficulty + density_modifier)
        
        return difficulty
    
    def generate_portal_network(self):
        """Generate portal connections between zones"""
        for dimension_id, dimension in self.dimensions.items():
            for zone_id, zone in dimension.zones.items():
                # Generate portals based on zone characteristics
                if np.random.random() < self.portal_probability:
                    portal_count = np.random.randint(1, self.max_portals_per_zone + 1)
                    
                    for _ in range(portal_count):
                        # Select random destination
                        dest_dimension = np.random.randint(1, self.total_dimensions + 1)
                        dest_zone = np.random.randint(1, self.zones_per_dimension + 1)
                        
                        # Avoid self-connections
                        if dest_dimension != dimension_id or dest_zone != zone_id:
                            zone.portal_connections.append((dest_dimension, dest_zone))
    
    def get_zone(self, dimension: int, zone: int) -> Optional[ZoneData]:
        """Get zone data for specific dimension and zone"""
        if dimension in self.dimensions and zone in self.dimensions[dimension].zones:
            return self.dimensions[dimension].zones[zone]
        return None
    
    def get_adjacent_zones(self, dimension: int, zone: int) -> List[ZoneData]:
        """Get adjacent zones (dimension±1, zone±1)"""
        adjacent = []
        
        # Adjacent zones in same dimension
        for z in [zone - 1, zone + 1]:
            if 1 <= z <= self.zones_per_dimension:
                zone_data = self.get_zone(dimension, z)
                if zone_data:
                    adjacent.append(zone_data)
        
        # Adjacent zones in neighboring dimensions
        for d in [dimension - 1, dimension + 1]:
            if 1 <= d <= self.total_dimensions:
                zone_data = self.get_zone(d, zone)
                if zone_data:
                    adjacent.append(zone_data)
        
        return adjacent
    
    def calculate_travel_cost(self, from_dim: int, from_zone: int,
                            to_dim: int, to_zone: int) -> float:
        """Calculate cost to travel between zones"""
        # Base cost for dimensional travel
        dimension_cost = abs(to_dim - from_dim) * 50
        
        # Base cost for zone travel
        zone_cost = abs(to_zone - from_zone) * 10
        
        # Difficulty modifiers
        from_zone_data = self.get_zone(from_dim, from_zone)
        to_zone_data = self.get_zone(to_dim, to_zone)
        
        difficulty_cost = 0
        if from_zone_data:
            difficulty_cost += from_zone_data.discovery_difficulty * 25
        if to_zone_data:
            difficulty_cost += to_zone_data.discovery_difficulty * 25
        
        return dimension_cost + zone_cost + difficulty_cost
    
    def get_big_bang_transition_data(self, dimension: int) -> Dict:
        """Get Big Bang transition animation data for dimension"""
        dimension_data = self.dimensions.get(dimension)
        if not dimension_data:
            return {}
        
        return {
            'center': dimension_data.big_bang_center,
            'expansion_rate': dimension_data.expansion_rate,
            'consciousness_signature': dimension_data.dimension_consciousness,
            'stability': dimension_data.stability,
            'theme': dimension_data.dimensional_theme
        }
    
    def get_ai_emergence_zones(self) -> List[ZoneData]:
        """Get all zones with AI emergence"""
        ai_zones = []
        
        for dimension in self.dimensions.values():
            for zone in dimension.zones.values():
                if zone.ai_presence:
                    ai_zones.append(zone)
        
        return ai_zones
    
    def get_zone_summary(self, dimension: int, zone: int) -> Dict:
        """Get summary information for a zone"""
        zone_data = self.get_zone(dimension, zone)
        if not zone_data:
            return {}
        
        return {
            'location': f"D{dimension}:Z{zone}",
            'type': zone_data.zone_type.value,
            'density': zone_data.density,
            'consciousness': {
                'red': zone_data.consciousness_field.red,
                'blue': zone_data.consciousness_field.blue,
                'yellow': zone_data.consciousness_field.yellow
            },
            'ai_presence': zone_data.ai_presence,
            'ai_level': zone_data.ai_advancement_level,
            'discovery_difficulty': zone_data.discovery_difficulty,
            'portal_count': len(zone_data.portal_connections),
            'resources': zone_data.resources
        }

# World generation integration for game systems
class WorldGenerationManager:
    """High-level manager for integrating world generation with game"""
    
    def __init__(self, game_instance):
        self.game = game_instance
        self.math_engine = AEMathEngine()
        self.world_generator = ProceduralWorldGenerator(self.math_engine)
        
        # Current player location
        self.current_dimension = 1
        self.current_zone = 1
        
    def get_current_zone_data(self) -> Optional[ZoneData]:
        """Get data for player's current zone"""
        return self.world_generator.get_zone(self.current_dimension, self.current_zone)
    
    def travel_to_zone(self, dimension: int, zone: int) -> bool:
        """Travel to specified zone if possible"""
        travel_cost = self.world_generator.calculate_travel_cost(
            self.current_dimension, self.current_zone, dimension, zone
        )
        
        # Check if player can afford travel (would check resources in real game)
        # For now, always allow travel
        
        self.current_dimension = dimension
        self.current_zone = zone
        
        print(f"Traveled to Dimension {dimension}, Zone {zone} (Cost: {travel_cost})")
        return True
    
    def get_current_zone_effects(self) -> Dict[str, float]:
        """Get environmental effects for current zone"""
        zone_data = self.get_current_zone_data()
        if zone_data:
            return zone_data.environmental_effects
        return {}

# Example usage
if __name__ == "__main__":
    print("=== Procedural World Generation Test ===")
    
    # Initialize generator
    math_engine = AEMathEngine()
    world_gen = ProceduralWorldGenerator(math_engine)
    
    # Test zone access
    test_zone = world_gen.get_zone(1, 1)
    if test_zone:
        print(f"Zone D1:Z1 - Type: {test_zone.zone_type.value}")
        print(f"Density: {test_zone.density:.3f}")
        print(f"Resources: {test_zone.resources}")
        print(f"AI Presence: {test_zone.ai_presence}")
    
    # Test AI emergence zones
    ai_zones = world_gen.get_ai_emergence_zones()
    print(f"\nAI Emergence Zones: {len(ai_zones)}")
    
    # Test zone summary
    summary = world_gen.get_zone_summary(5, 10)
    print(f"\nZone D5:Z10 Summary: {summary}")
    
    print("\n=== Procedural World Generation Ready ===")
    print("11 dimensions × 13 zones universe generated.")
    print("Ready for integration with multi-mode architecture.")
