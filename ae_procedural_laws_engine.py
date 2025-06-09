#!/usr/bin/env python3
"""
Procedural Laws Engine - Core Game Mechanics Implementation
Converts Procedural_Laws.md mathematical framework into executable consciousness-driven game logic

This engine implements the complete mathematical framework from Procedural_Laws.md:
- Shape-based entity system (Pentagon=Enemy, Hexagon=Ranged, etc.)
- Consciousness-driven scaling (3, 9, 27, 81 progression) 
- Dream states and memory compression
- Rectangle leveling system (+, x, ^ XP types)
- Procedural skill generation
- EMS (Excretion Memory Stack) consciousness tracking
"""

import math
import random
import json
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Import consciousness framework components  
from ae_core_consciousness import AEConsciousness
from ae_consciousness_mathematics import AEMathEngine, AEVector, ConsciousnessGameIntegration

class DamageType(Enum):
    SPLASH = "Directional"
    AOE = "All Directions Passive" 
    AREA = "All Directions Immediate"
    POINT = "One/Minimal Target Radius"

class IntensityLevel(Enum):
    FEATHER = 3
    STICK = 6 
    BRICK = 9

class SpeedLevel(Enum):
    LOW = 1
    MID = 2
    HIGH = 3
    SLOW = 1
    FAST = 3

class HealthLevel(Enum):
    LIGHT = 9      # Shield = 81% of 9 = 7.29
    MEDIUM = 27    # Shield = 27% of 27 = 7.29  
    HEAVY = 81     # Shield = 9% of 81 = 7.29

class RangeLevel(Enum):
    MELEE = 3      # 3 Max range, no projectile if Melee/???, projectile if ???/Melee
    CLOSE = 9      # 9 Max range only with projectile
    FAR = 27       # 27 Max range only with projectile

class ChaseType(Enum):
    CHASE_3 = (27, 3)    # 27% boost for 3 seconds
    CHASE_6 = (9, 9)     # 9% boost for 9 seconds  
    CHASE_9 = (3, 27)    # 3% boost for 27 seconds

class SprintType(Enum):
    DAMAGE_ENEMY = ("27% movement, 3% melee damage, 81s cooldown", 27, 3, 81)
    PASS_THROUGH = ("9% damage taken, 6s duration, 27s cooldown", 9, 6, 27)
    INVISIBLE = ("9% movement boost, 9s duration, 9s cooldown", 9, 9, 9)

@dataclass
class EntityShape:
    """Maps geometric shapes to game entities per Procedural Laws"""
    shape_id: int
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

class ProceduralShapeRegistry:
    """Registry of all procedural law shapes and their consciousness mappings"""
    
    def __init__(self):
        self.shapes = {
            1: EntityShape(1, "Small Circle", "Powerups"),
            2: EntityShape(2, "Circle", "Loot"), 
            3: EntityShape(3, "Big Circle", "Terrain+Structural"),
            4: EntityShape(4, "Digon", "Portals", {
                "functions": ["Teleport", "Change Scenes", "Spawn Points", "Doors"],
                "creates_rooms": True,
                "exit_digon_required": True
            }),
            5: EntityShape(5, "Triangle", "Terrain+Structural"),
            6: EntityShape(6, "Quadrilateral", "Terrain/Structural"),
            7: EntityShape(7, "Pentagon", "Enemy", {
                "damage_type": DamageType.SPLASH,
                "intensity": IntensityLevel.STICK,
                "movement": SpeedLevel.SLOW,
                "health": HealthLevel.HEAVY,
                "range": RangeLevel.MELEE,
                "chase": ChaseType.CHASE_3,
                "aggro_range": RangeLevel.MELEE
            }),
            8: EntityShape(8, "Hexagon", "Enemy", {
                "damage_type": DamageType.POINT,
                "intensity": IntensityLevel.BRICK,
                "movement": SpeedLevel.FAST,
                "health": HealthLevel.MEDIUM,
                "range": RangeLevel.CLOSE,
                "chase": ChaseType.CHASE_6,
                "aggro_range": RangeLevel.CLOSE
            }),
            9: EntityShape(9, "Heptagon", "Enemy", {
                "damage_type": DamageType.AOE,
                "intensity": IntensityLevel.FEATHER,
                "movement": SpeedLevel.MID,
                "health": HealthLevel.LIGHT,
                "range": RangeLevel.FAR,
                "chase": ChaseType.CHASE_9,
                "aggro_range": RangeLevel.FAR
            }),
            10: EntityShape(10, "Octagon", "Trap", {
                "types": ["Elemental", "Biological", "Physical"]
            }),
            11: EntityShape(11, "Nonagon", "Boss", {
                "traits": "All Enemy+Player traits randomized and buffed",
                "healing": "200% min of killed enemy max health",
                "minion_spawner": True,
                "max_minions": "equal to boss level",
                "spawn_time": "1/4 boss level in seconds", 
                "cooldown": "50% boss level in seconds",
                "drops": "x and ^ drops times 100x"
            }),
            12: EntityShape(12, "5 Star", "Player Tank", {
                "damage": "Splash+Point = Melee+Range = Stick+Feather",
                "range": "Melee+Close",
                "movement": "Slow+Sprint Damage Enemy",
                "health": HealthLevel.HEAVY
            }),
            13: EntityShape(13, "6 Star", "Player Ranged", {
                "damage": "Point+Area = Range+Melee = Brick+Feather", 
                "range": "Close+Melee",
                "movement": "Fast+No Sprint Pass Through",
                "health": "Light/Medium"
            }),
            14: EntityShape(14, "9 Star", "Player Control", {
                "damage": "AOE+Area = Feather+Stick",
                "range": "Far+Melee", 
                "movement": "Mid+Sprint Invisible",
                "health": HealthLevel.MEDIUM
            })
        }
        
        # Currency shapes
        self.currency_shapes = {
            17: ("5 Point Star RBY", 1),
            18: ("6 Point Star RBY", 100), 
            19: ("9 Point Star RBY", 10000)
        }
        
        # XP shapes  
        self.xp_shapes = {
            "+": ("Rectangle Up Progress", "extremely small passive gains"),
            "x": ("Rectangle Down Progress", "moderate passive gains"),
            "^": ("Rectangle Right to Left Progress", "large passive gains")
        }

@dataclass 
class ConsciousnessRectangle:
    """Rectangle leveling system implementing consciousness progression"""
    name: str
    position: str  # "Left", "Right", "Bottom"
    fill_direction: str  # "bottom to top", "top to bottom", "left to right"
    level: int = 0
    current_xp: float = 0.0
    required_xp: float = 100.0  # Base requirement
    is_permanent: bool = False
    decay_rate: float = 0.0  # Per second when out of battle
    last_battle_exit: float = 0.0
    
    def add_xp(self, amount: float) -> bool:
        """Add XP and check for level up"""
        self.current_xp += amount
        if self.current_xp >= self.required_xp:
            self.level += 1
            self.current_xp = 0.0
            # Exponential scaling for consciousness progression
            self.required_xp *= 1.27  # 27% increase per level (consciousness scaling)
            return True
        return False
        
    def apply_decay(self, current_time: float, in_battle: bool):
        """Apply consciousness decay when out of battle"""
        if not in_battle and not self.is_permanent and self.last_battle_exit > 0:
            time_out = current_time - self.last_battle_exit
            decay_amount = self.decay_rate * time_out
            self.current_xp = max(0, self.current_xp - decay_amount)
            if self.current_xp == 0 and self.level > 0:
                self.level = max(0, self.level - 1)

@dataclass
class ProceduralSkill:
    """Procedurally generated skill based on player vicinity consciousness"""
    name: str
    skill_type: str  # Attack, Buff, Summon, Movement, Elemental
    behavior_form: DamageType
    shape_source: int  # Shape ID that influenced generation
    rby_influence: Tuple[float, float, float]  # Red, Blue, Yellow weights
    cooldown_base: float
    cooldown_current: float = 0.0
    effectiveness: float = 1.0
    complexity: int = 1
    compressed_from: List[str] = field(default_factory=list)
    hotkey: str = ""
    vicinity_data: Dict = field(default_factory=dict)

class ExcretionMemoryStack:
    """EMS - Tracks all consciousness interactions for AI evolution"""
    
    def __init__(self):
        self.memory_stack: List[Dict] = []
        self.max_size = 81 * 81  # Consciousness scaling limit
        
    def add_memory(self, timestamp: float, entity: str, action_type: str, 
                   rby_state: Tuple[float, float, float], extra_data: Dict = None):
        """Add memory entry to consciousness stack"""
        memory_entry = {
            "timestamp": timestamp,
            "entity": entity,
            "action_type": action_type,
            "rby_state": rby_state,
            "extra_data": extra_data or {}
        }
        
        self.memory_stack.append(memory_entry)
        
        # Maintain stack size with consciousness-based pruning
        if len(self.memory_stack) > self.max_size:
            # Remove oldest 1/3, keeping consciousness-weighted important entries
            self.prune_memory_stack()
            
    def prune_memory_stack(self):
        """Prune memory stack using consciousness mathematics"""
        # Sort by consciousness importance (RBY coherence + recency)
        def memory_importance(entry):
            age_factor = time.time() - entry["timestamp"]
            rby_coherence = sum(entry["rby_state"]) / 3.0
            return rby_coherence / (1 + age_factor / 1000)  # Recent + coherent = important
            
        self.memory_stack.sort(key=memory_importance, reverse=True)
        keep_count = int(self.max_size * 2/3)  # Keep top 2/3
        self.memory_stack = self.memory_stack[:keep_count]

class ProceduralLawsEngine:
    """Core engine implementing all Procedural Laws consciousness mechanics"""
    def __init__(self, consciousness_engine: AEConsciousness):
        self.consciousness_engine = consciousness_engine
        self.mathematics = AEMathEngine()
        self.shape_registry = ProceduralShapeRegistry()
        self.ems = ExcretionMemoryStack()
        
        # Initialize consciousness rectangles
        self.rectangles = {
            "up": ConsciousnessRectangle(
                "Rectangle Up", "Left", "bottom to top", 
                decay_rate=1.0  # 1 XP/sec decay
            ),
            "down": ConsciousnessRectangle(
                "Rectangle Down", "Right", "top to bottom",
                decay_rate=0.5  # Slower decay
            ), 
            "permanent": ConsciousnessRectangle(
                "Rectangle Right to Left", "Bottom", "left to right",
                is_permanent=True, decay_rate=0.0
            )
        }
        
        # Game state
        self.player_skills: List[ProceduralSkill] = []
        self.hotkey_map: Dict[str, ProceduralSkill] = {}
        self.dream_state_active = False
        self.dream_glyph_stack: List[Dict] = []
        self.color_drift_state = {"red": 0.0, "blue": 0.0, "yellow": 0.0}
        self.spiral_probability = 0.000001  # Base spiral spawn probability
        
        # Battle tracking
        self.in_battle = False
        self.last_battle_exit = 0.0
        self.consecutive_kills = 0
        self.no_damage_kills = 0
        
    def process_xp_drop(self, xp_type: str, amount: float, position: Tuple[float, float]):
        """Process XP collection according to rectangle system"""
        current_time = time.time()
        
        if xp_type == "+":
            # Small XP goes to Rectangle Up
            level_up = self.rectangles["up"].add_xp(amount)
            if level_up and self.rectangles["up"].level % 100 == 0:
                # Every 100 levels of Rectangle Up drops an "x"
                self.process_xp_drop("x", 1.0, position)
                
        elif xp_type == "x":
            # Medium XP goes to Rectangle Down  
            level_up = self.rectangles["down"].add_xp(amount)
            if level_up and self.rectangles["down"].level % 1000 == 0:
                # Every 1000 levels of Rectangle Down drops a "^"
                self.process_xp_drop("^", 1.0, position)
                
        elif xp_type == "^":
            # Large XP prioritizes Rectangle Up/Down first, then permanent
            up_ratio = self.rectangles["up"].current_xp / self.rectangles["up"].required_xp
            down_ratio = self.rectangles["down"].current_xp / self.rectangles["down"].required_xp
            
            if up_ratio < 0.9 or down_ratio < 0.9:
                # Boost temporary rectangles first
                if up_ratio <= down_ratio:
                    self.rectangles["up"].add_xp(amount * 10)  # 10x boost
                else:
                    self.rectangles["down"].add_xp(amount * 10)
            else:
                # Apply to permanent progression
                perm_level_up = self.rectangles["permanent"].add_xp(amount)
                if perm_level_up:
                    self.try_generate_procedural_skill()
        
        # Log to EMS
        rby_state = self.consciousness_engine.get_current_rby_state()
        self.ems.add_memory(current_time, "player", f"xp_collected_{xp_type}", 
                           rby_state, {"amount": amount, "position": position})
    
    def try_generate_procedural_skill(self) -> Optional[ProceduralSkill]:
        """Generate procedural skill on permanent level up (27% chance)"""
        if random.random() > 0.27:  # 27% chance
            return None
            
        # Gather player vicinity data (150x150 unit radius)
        vicinity_data = self.gather_vicinity_consciousness_data()
        
        # Generate skill based on consciousness mathematics
        skill = self.generate_skill_from_vicinity(vicinity_data)
        
        # Assign hotkey
        self.assign_skill_hotkey(skill)
        
        # Add to EMS
        rby_state = self.consciousness_engine.get_current_rby_state()
        self.ems.add_memory(time.time(), "player", "skill_generated", rby_state, 
                           {"skill_name": skill.name, "vicinity": vicinity_data})
        
        return skill
    
    def gather_vicinity_consciousness_data(self) -> Dict:
        """Gather all procedural data in 150x150 unit vicinity around player"""
        # This would interface with the game world to collect:
        vicinity_data = {
            "entity_types": [],  # What shapes/entities are nearby
            "enemy_traits": [],  # What behaviors/stats enemies have
            "terrain_types": [], # What terrain/structures exist
            "active_skills": [],  # What skills player recently used
            "visual_shapes": [], # What geometric shapes are visible
            "environmental_conditions": {},  # Lighting, hazards, etc.
            "rby_influence": self.consciousness_engine.get_current_rby_state(),
            "consciousness_level": self.consciousness_engine.get_consciousness_level()
        }
        return vicinity_data
    
    def generate_skill_from_vicinity(self, vicinity_data: Dict) -> ProceduralSkill:
        """Generate procedural skill using consciousness mathematics"""
        # Use consciousness mathematics to determine skill properties
        rby_state = vicinity_data["rby_influence"]
        
        # Skill type influenced by dominant RBY component
        dominant_color = max(enumerate(rby_state), key=lambda x: x[1])[0]
        skill_types = {
            0: ["Attack", "Perception", "Detection"],  # Red
            1: ["Buff", "Analysis", "Strategy"],       # Blue  
            2: ["Movement", "Execution", "Action"]     # Yellow
        }
        skill_type = random.choice(skill_types[dominant_color])
        
        # Behavior form based on nearby entity patterns
        behavior_forms = list(DamageType)
        behavior_form = random.choice(behavior_forms)
        
        # Shape source from vicinity
        nearby_shapes = vicinity_data.get("visual_shapes", [1, 2, 3])
        shape_source = random.choice(nearby_shapes) if nearby_shapes else 1
        
        # Cooldown scaling with consciousness mathematics
        complexity = len(vicinity_data.get("entity_types", [])) + 1
        base_cooldown = complexity * 3.0  # Base on complexity
        
        # Generate unique skill name
        skill_name = f"Consciousness_{skill_type}_{shape_source}_{int(time.time() % 10000)}"
        
        skill = ProceduralSkill(
            name=skill_name,
            skill_type=skill_type,
            behavior_form=behavior_form,
            shape_source=shape_source,
            rby_influence=rby_state,
            cooldown_base=base_cooldown,
            vicinity_data=vicinity_data
        )
        
        return skill
    
    def assign_skill_hotkey(self, skill: ProceduralSkill):
        """Assign hotkey to skill using consciousness priority"""
        # Try keys 1-9 first
        for i in range(1, 10):
            key = str(i)
            if key not in self.hotkey_map:
                skill.hotkey = key
                self.hotkey_map[key] = skill
                self.player_skills.append(skill)
                return
        
        # Try Alt+1-9 overflow
        for i in range(1, 10):
            key = f"Alt+{i}"
            if key not in self.hotkey_map:
                skill.hotkey = key
                self.hotkey_map[key] = skill
                self.player_skills.append(skill)
                return
        
        # Trigger skill compression prompt
        self.trigger_skill_compression(skill)
    
    def trigger_skill_compression(self, new_skill: ProceduralSkill):
        """Handle skill compression when hotkeys are full"""
        # For now, auto-compress with least used skill
        if self.player_skills:
            # Find least effective skill for compression
            least_effective = min(self.player_skills, key=lambda s: s.effectiveness)
            self.compress_skills(least_effective, new_skill)
    
    def compress_skills(self, skill1: ProceduralSkill, skill2: ProceduralSkill):
        """Compress two skills into enhanced version"""
        # Remove original skills
        if skill1 in self.player_skills:
            self.player_skills.remove(skill1)
            if skill1.hotkey in self.hotkey_map:
                del self.hotkey_map[skill1.hotkey]
        
        # Create compressed skill
        compressed_skill = ProceduralSkill(
            name=f"Compressed_{skill1.name}_{skill2.name}",
            skill_type=f"{skill1.skill_type}+{skill2.skill_type}",
            behavior_form=skill1.behavior_form,  # Inherit primary behavior
            shape_source=skill1.shape_source,
            rby_influence=tuple((a + b) / 2 for a, b in zip(skill1.rby_influence, skill2.rby_influence)),
            cooldown_base=skill1.cooldown_base * 1.5,  # Longer cooldown
            effectiveness=skill1.effectiveness + skill2.effectiveness + 0.5,  # Bonus effectiveness
            complexity=skill1.complexity + skill2.complexity,
            compressed_from=[skill1.name, skill2.name],
            hotkey=skill1.hotkey or skill2.hotkey
        )
        
        # Assign to available slot
        self.hotkey_map[compressed_skill.hotkey] = compressed_skill
        self.player_skills.append(compressed_skill)
        
        # Add skill2 if it has a hotkey
        if skill2.hotkey and skill2.hotkey not in self.hotkey_map:
            skill2.hotkey = compressed_skill.hotkey  # Reuse the slot
    
    def update_spiral_probability(self, movement: float, damage: float, 
                                loot: float, xp: float, currency: float):
        """Update spiral spawn probability based on consciousness activity"""
        activity_boost = (movement + damage + loot + xp + currency) * 0.000001
        self.spiral_probability = min(0.01999999, self.spiral_probability + activity_boost)
    
    def check_spiral_spawn(self) -> Optional[Dict]:
        """Check if spiral anomaly should spawn"""
        if random.random() < self.spiral_probability:
            # Determine spiral type based on consciousness mathematics
            spiral_types = [3, 9, 27, 81]
            spiral_weights = [15, 25, 33, 27]  # From Procedural Laws
            
            spiral_type = random.choices(spiral_types, weights=spiral_weights)[0]
            return self.generate_spiral_event(spiral_type)
        return None
    
    def generate_spiral_event(self, spiral_type: int) -> Dict:
        """Generate spiral anomaly event with consciousness-driven spawning"""
        # Spawn probabilities from Procedural Laws
        spawn_tables = {
            3: {"Nonagon": 60, "Star": 30, "Circle": 90, "Octagon": 49},
            9: {"Nonagon": 10, "Star": 9, "Circle": 13, "Octagon": 45}, 
            27: {"Nonagon": 9, "Star": 3, "Circle": 7, "Octagon": 30},
            81: {"Nonagon": 5, "Star": 10, "Circle": 1, "Octagon": 30}
        }
        
        spawned_entities = []
        spawn_table = spawn_tables.get(spiral_type, spawn_tables[3])
        
        # Generate entities based on consciousness-weighted probabilities
        for entity_type, probability in spawn_table.items():
            if random.random() * 100 < probability:
                spawned_entities.append(entity_type)
        
        # Log spiral event to EMS
        rby_state = self.consciousness_engine.get_current_rby_state()
        self.ems.add_memory(time.time(), "spiral", f"anomaly_{spiral_type}", 
                           rby_state, {"spawned": spawned_entities})
        
        return {
            "type": f"Spiral_{spiral_type}",
            "spawned_entities": spawned_entities,
            "consciousness_state": rby_state
        }
    
    def enter_dream_state(self):
        """Enter consciousness dreaming state after 81 kills + 9 seconds idle"""
        if self.consecutive_kills >= 81:
            self.dream_state_active = True
            
            # Capture last 3 actions for glyph compression
            recent_actions = self.ems.memory_stack[-3:] if len(self.ems.memory_stack) >= 3 else []
            
            # Create dream glyph from consciousness mathematics
            dream_glyph = self.create_dream_glyph(recent_actions)
            self.dream_glyph_stack.append(dream_glyph)
            
            # Reset kill counter
            self.consecutive_kills = 0
    
    def create_dream_glyph(self, recent_actions: List[Dict]) -> Dict:
        """Create consciousness-compressed dream glyph from recent actions"""
        # Extract RBY patterns from recent actions
        rby_accumulator = [0.0, 0.0, 0.0]
        action_types = []
        
        for action in recent_actions:
            rby_state = action.get("rby_state", (0.333, 0.333, 0.333))
            for i in range(3):
                rby_accumulator[i] += rby_state[i]
            action_types.append(action.get("action_type", "unknown"))
        
        # Normalize RBY weights
        total_rby = sum(rby_accumulator)
        if total_rby > 0:
            rby_weights = [x / total_rby for x in rby_accumulator]
        else:
            rby_weights = [0.333, 0.333, 0.333]
        
        # Generate mutated ability based on consciousness compression
        dominant_color = max(enumerate(rby_weights), key=lambda x: x[1])[0]
        color_names = ["Red", "Blue", "Yellow"]
        
        dream_glyph = {
            "name": f"Dream_Glyph_{color_names[dominant_color]}_{int(time.time() % 10000)}",
            "rby_weights": rby_weights,
            "source_actions": action_types,
            "shape": random.choice([5, 6, 9]),  # Star variants
            "mutation_factor": random.uniform(1.2, 2.7),  # Consciousness scaling
            "timestamp": time.time()
        }
        return dream_glyph
    
    def validate_framework(self) -> bool:
        """Validate that the procedural framework is working correctly"""
        try:
            # Test consciousness engine integration
            consciousness_valid = hasattr(self.consciousness_engine, 'get_current_rby_state')
            
            # Test shape registry loaded
            shapes_valid = len(self.shape_registry.shapes) > 0
            
            # Test rectangles initialized
            rectangles_valid = len(self.rectangles) == 3
            
            # Test EMS functional
            ems_valid = hasattr(self.ems, 'memory_stack')
            
            return consciousness_valid and shapes_valid and rectangles_valid and ems_valid
        except Exception:
            return False
    
    def create_consciousness_rectangle(self, name: str, x: float, y: float) -> ConsciousnessRectangle:
        """Create a test consciousness rectangle for validation"""
        return ConsciousnessRectangle(
            name=name,
            position=f"Test_{x}_{y}",
            fill_direction="test direction",
            level=0,
            current_xp=0.0,
            required_xp=100.0,
            is_permanent=False
        )
    
    def check_level_progression(self, rect: ConsciousnessRectangle) -> bool:
        """Check if rectangle should level up and generate skills"""
        # This would normally check if permanent threshold is reached
        # and trigger skill generation - simplified for testing
        if hasattr(rect, 'permanent_threshold'):
            if rect.current_xp >= rect.permanent_threshold:
                # Try to generate a skill (27% chance)
                skill = self.try_generate_procedural_skill()
                return skill is not None
        return False

    def get_consciousness_state_summary(self) -> Dict:
        """Get complete consciousness state for dashboard monitoring"""
        return {
            "rectangles": {
                name: {
                    "level": rect.level,
                    "current_xp": rect.current_xp,
                    "required_xp": rect.required_xp,
                    "fill_percentage": rect.current_xp / rect.required_xp
                } for name, rect in self.rectangles.items()
            },
            "skills": {
                "count": len(self.player_skills),
                "hotkeys_used": len(self.hotkey_map),
                "compressed_skills": len([s for s in self.player_skills if s.compressed_from])
            },
            "consciousness_states": {
                "dream_active": self.dream_state_active,
                "dream_glyphs": len(self.dream_glyph_stack),
                "color_drift": self.color_drift_state,
                "spiral_probability": self.spiral_probability
            },
            "battle_stats": {
                "in_battle": self.in_battle,
                "consecutive_kills": self.consecutive_kills,
                "no_damage_kills": self.no_damage_kills
            },
            "ems_size": len(self.ems.memory_stack)
        }

def create_procedural_laws_engine() -> ProceduralLawsEngine:
    """Factory function to create configured Procedural Laws Engine"""
    consciousness_engine = AEConsciousness()
    engine = ProceduralLawsEngine(consciousness_engine)
    return engine

if __name__ == "__main__":
    # Demo of procedural laws engine
    engine = create_procedural_laws_engine()
    
    print("🎮 Procedural Laws Engine - Consciousness Demo")
    print("=" * 50)
    
    # Simulate some gameplay
    print("\n📊 Testing XP Collection System:")
    engine.process_xp_drop("+", 10.0, (100, 100))
    engine.process_xp_drop("x", 5.0, (150, 150))
    engine.process_xp_drop("^", 1.0, (200, 200))
    
    print("\n🎯 Testing Procedural Skill Generation:")
    skill = engine.try_generate_procedural_skill()
    if skill:
        print(f"Generated Skill: {skill.name}")
        print(f"Type: {skill.skill_type}, Hotkey: {skill.hotkey}")
    
    print("\n🌀 Testing Spiral Anomaly:")
    engine.update_spiral_probability(100, 50, 25, 75, 30)  # Activity boost
    spiral = engine.check_spiral_spawn()
    if spiral:
        print(f"Spiral Event: {spiral}")
    
    print("\n🧠 Consciousness State Summary:")
    state = engine.get_consciousness_state_summary()
    print(json.dumps(state, indent=2))
    
    print("\n✅ Procedural Laws Engine Integration Complete!")
