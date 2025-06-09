"""
Multi-Mode Architecture for Unified Absolute Framework
Manages seamless transitions between Tower Defense, MMORPG, and Base Builder modes
Implements consciousness-based mode progression and anchoring systems
"""

import pygame
import math
import random
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from ae_consciousness_mathematics import AEMathEngine, AEVector, ConsciousnessGameIntegration

class GameMode(Enum):
    TOWER_DEFENSE = "tower_defense"
    MMORPG = "mmorpg"
    BASE_BUILDER = "base_builder"
    GAME_OF_LIFE = "game_of_life"
    TRANSITION = "transition"

class TransitionType(Enum):
    PANIC_PORTAL = "panic_portal"
    NORMAL_PORTAL = "normal_portal"
    ANCHORED_RETURN = "anchored_return"
    BIG_BANG_EXIT = "big_bang_exit"

@dataclass
class ModeState:
    """State data for each game mode"""
    mode: GameMode
    player_position: Tuple[float, float] = (0, 0)
    consciousness_state: AEVector = field(default_factory=AEVector)
    resources: Dict[str, float] = field(default_factory=dict)
    unlocked_features: List[str] = field(default_factory=list)
    mode_specific_data: Dict = field(default_factory=dict)

@dataclass
class AnchorPoint:
    """Represents an anchored tower/structure across modes"""
    id: str
    position: Tuple[float, float]
    consciousness_signature: AEVector
    tower_type: str
    upgrade_level: int
    health: float
    max_health: float
    anchored_dimension: int
    anchored_zone: int
    passive_effects: Dict[str, float]
    cross_mode_bonus: float

class MultiModeArchitecture:
    """Core multi-mode system managing all game modes and transitions"""
    
    def __init__(self, math_engine: AEMathEngine):
        self.math_engine = math_engine
        self.consciousness_integration = ConsciousnessGameIntegration(math_engine)
        
        # Current mode state
        self.current_mode = GameMode.TOWER_DEFENSE
        self.previous_mode = None
        self.transition_in_progress = False
        self.transition_timer = 0.0
        
        # Mode states
        self.mode_states = {
            GameMode.TOWER_DEFENSE: ModeState(GameMode.TOWER_DEFENSE),
            GameMode.MMORPG: ModeState(GameMode.MMORPG),
            GameMode.BASE_BUILDER: ModeState(GameMode.BASE_BUILDER),
            GameMode.GAME_OF_LIFE: ModeState(GameMode.GAME_OF_LIFE)
        }
        
        # Anchoring system
        self.anchor_points: Dict[str, AnchorPoint] = {}
        self.max_anchors = 4  # Initially 4 towers, upgradeable to 20
        
        # Cross-mode data
        self.global_consciousness = AEVector()
        self.total_mass = 0.0  # Unified currency
        self.dimension = 1
        self.zone = 1
        
        # Mode unlock requirements
        self.mode_requirements = {
            GameMode.MMORPG: {"towers_placed": 2, "waves_completed": 5},
            GameMode.BASE_BUILDER: {"dimensions_explored": 3, "anchors_placed": 1},
            GameMode.GAME_OF_LIFE: {"consciousness_unity": 0.9, "genetic_material": 100}
        }
        
        # Portal system
        self.panic_portal_cooldown = 300.0  # 5 minutes
        self.panic_portal_timer = 0.0
        self.portal_active = False
        self.portal_duration = 30.0
        self.portal_timer = 0.0
        
        # Following enemies system
        self.following_enemies = []
        self.max_followers = 5
        
    def check_mode_unlocks(self, game_stats: Dict) -> List[GameMode]:
        """Check which modes are unlocked based on progression"""
        unlocked = [GameMode.TOWER_DEFENSE]  # Always available
        
        for mode, requirements in self.mode_requirements.items():
            if self._meets_requirements(requirements, game_stats):
                unlocked.append(mode)
        
        return unlocked
    
    def _meets_requirements(self, requirements: Dict, game_stats: Dict) -> bool:
        """Check if requirements are met for mode unlock"""
        for req_key, req_value in requirements.items():
            if game_stats.get(req_key, 0) < req_value:
                return False
        return True
    
    def initiate_transition(self, target_mode: GameMode, 
                          transition_type: TransitionType = TransitionType.NORMAL_PORTAL,
                          enemies_follow: bool = False) -> bool:
        """Initiate transition between game modes"""
        if self.transition_in_progress:
            return False
        
        # Validate transition
        if not self._validate_transition(target_mode, transition_type):
            return False
        
        print(f"Initiating transition: {self.current_mode.value} -> {target_mode.value}")
        print(f"Transition type: {transition_type.value}")
        
        # Save current mode state
        self._save_current_mode_state()
        
        # Handle enemy following
        if enemies_follow and transition_type == TransitionType.PANIC_PORTAL:
            self._handle_enemy_following()
        
        # Set transition state
        self.previous_mode = self.current_mode
        self.current_mode = GameMode.TRANSITION
        self.transition_in_progress = True
        self.transition_timer = 2.0  # 2-second transition animation
        
        # Store target mode for after transition
        self._target_mode = target_mode
        self._transition_type = transition_type
        
        return True
    
    def _validate_transition(self, target_mode: GameMode, 
                           transition_type: TransitionType) -> bool:
        """Validate if transition is allowed"""
        # Check if mode is unlocked
        # (This would check against actual game stats in real implementation)
        
        # Check panic portal cooldown
        if transition_type == TransitionType.PANIC_PORTAL:
            if self.panic_portal_timer > 0:
                print(f"Panic portal on cooldown: {self.panic_portal_timer:.1f}s remaining")
                return False
        
        # Prevent invalid transitions
        if target_mode == self.current_mode:
            return False
        
        return True
    
    def _save_current_mode_state(self):
        """Save current mode state before transition"""
        current_state = self.mode_states[self.current_mode]
        
        # Save mode-specific data based on current mode
        if self.current_mode == GameMode.TOWER_DEFENSE:
            current_state.mode_specific_data = {
                'current_wave': getattr(self, 'current_wave', 1),
                'towers_placed': len(self.anchor_points),
                'enemies_alive': getattr(self, 'enemies_count', 0)
            }
        elif self.current_mode == GameMode.MMORPG:
            current_state.mode_specific_data = {
                'world_position': getattr(self, 'world_position', (0, 0)),
                'quest_progress': getattr(self, 'quest_progress', {}),
                'discovered_zones': getattr(self, 'discovered_zones', [])
            }
        elif self.current_mode == GameMode.BASE_BUILDER:
            current_state.mode_specific_data = {
                'base_structures': getattr(self, 'base_structures', []),
                'resource_production': getattr(self, 'resource_production', {}),
                'defense_rating': getattr(self, 'defense_rating', 0)
            }
    
    def _handle_enemy_following(self):
        """Handle enemies following through panic portal"""
        # This would interact with the game's enemy system
        self.following_enemies = []
        
        # Select random enemies to follow (up to max_followers)
        # This is a placeholder - actual implementation would get enemies from game state
        enemy_count = random.randint(1, self.max_followers)
        
        for i in range(enemy_count):
            # Generate enemy consciousness for cross-mode behavior
            enemy_consciousness = AEVector(
                random.uniform(0.2, 0.8),
                random.uniform(0.1, 0.6),
                random.uniform(0.1, 0.5)
            )
            enemy_consciousness.normalize()
            
            self.following_enemies.append({
                'id': f"follower_{i}",
                'consciousness': enemy_consciousness,
                'original_mode': self.current_mode,
                'follow_strength': random.uniform(0.5, 1.0)
            })
        
        print(f"{len(self.following_enemies)} enemies will follow through portal")
    
    def complete_transition(self):
        """Complete the mode transition"""
        if not self.transition_in_progress:
            return
        
        # Set new mode
        self.current_mode = self._target_mode
        self.transition_in_progress = False
        
        # Load target mode state
        self._load_mode_state(self.current_mode)
        
        # Handle transition-specific effects
        if self._transition_type == TransitionType.PANIC_PORTAL:
            self.panic_portal_timer = self.panic_portal_cooldown
            # Add stress effect to consciousness
            stress_effect = 0.1
            self.global_consciousness.red += stress_effect
            self.global_consciousness.normalize()
        
        # Spawn following enemies in new mode
        if self.following_enemies:
            self._spawn_following_enemies()
        
        print(f"Transition complete. Now in {self.current_mode.value} mode.")
        
        # Reset transition variables
        self._target_mode = None
        self._transition_type = None
    
    def _load_mode_state(self, mode: GameMode):
        """Load state data for specified mode"""
        state = self.mode_states[mode]
        
        # Restore mode-specific data
        if mode == GameMode.TOWER_DEFENSE:
            # Restore wave state, towers, etc.
            pass
        elif mode == GameMode.MMORPG:
            # Restore world position, quests, etc.
            pass
        elif mode == GameMode.BASE_BUILDER:
            # Restore base structures, production, etc.
            pass
    
    def _spawn_following_enemies(self):
        """Spawn enemies that followed through portal in new mode"""
        for enemy in self.following_enemies:
            # Adapt enemy to new mode
            if self.current_mode == GameMode.MMORPG:
                # Spawn as world mob
                pass
            elif self.current_mode == GameMode.BASE_BUILDER:
                # Spawn as raider
                pass
            elif self.current_mode == GameMode.TOWER_DEFENSE:
                # Spawn as wave enemy with boosted stats
                pass
        
        # Clear following enemies after spawning
        self.following_enemies = []
    
    def place_anchor(self, position: Tuple[float, float], tower_type: str,
                    consciousness: AEVector) -> bool:
        """Place an anchor point for cross-mode gameplay"""
        if len(self.anchor_points) >= self.max_anchors:
            return False
        
        anchor_id = f"anchor_{len(self.anchor_points)}"
        
        # Calculate anchor effectiveness
        anchor_data = self.math_engine.anchoring_calculation(
            consciousness, self.global_consciousness, 1.0
        )
        
        anchor = AnchorPoint(
            id=anchor_id,
            position=position,
            consciousness_signature=consciousness,
            tower_type=tower_type,
            upgrade_level=1,
            health=100.0,
            max_health=100.0,
            anchored_dimension=self.dimension,
            anchored_zone=self.zone,
            passive_effects={
                'healing_rate': anchor_data['effectiveness'] * 0.1,
                'consciousness_boost': anchor_data['compatibility'] * 0.05
            },
            cross_mode_bonus=anchor_data['effectiveness']
        )
        
        self.anchor_points[anchor_id] = anchor
        print(f"Anchor placed: {anchor_id} at {position}")
        return True
    
    def upgrade_anchor_capacity(self, current_upgrades: int) -> int:
        """Calculate anchor capacity based on upgrades"""
        # Linear progression: 1M upgrades = 20 total anchors
        # Base capacity: 4 anchors
        # Formula: 4 + (upgrades / 62500) up to max 20
        additional_capacity = min(16, current_upgrades // 62500)
        return 4 + additional_capacity
    
    def get_cross_mode_benefits(self) -> Dict[str, float]:
        """Calculate benefits from anchored towers across modes"""
        benefits = {
            'passive_healing': 0.0,
            'consciousness_gain': 0.0,
            'resource_bonus': 0.0,
            'experience_multiplier': 1.0
        }
        
        for anchor in self.anchor_points.values():
            benefits['passive_healing'] += anchor.passive_effects['healing_rate']
            benefits['consciousness_gain'] += anchor.passive_effects['consciousness_boost']
            benefits['resource_bonus'] += anchor.cross_mode_bonus * 0.1
            benefits['experience_multiplier'] += anchor.cross_mode_bonus * 0.05
        
        return benefits
    
    def update_panic_portal_system(self, dt: float, player_stress: float,
                                 wave_difficulty: float) -> Dict:
        """Update panic portal availability and mechanics"""
        # Update cooldown
        if self.panic_portal_timer > 0:
            self.panic_portal_timer -= dt
        
        # Update active portal timer
        if self.portal_active:
            self.portal_timer -= dt
            if self.portal_timer <= 0:
                self.portal_active = False
                print("Panic portal has closed!")
        
        # Check portal mechanics
        portal_mechanics = self.consciousness_integration.panic_portal_mechanics(
            player_stress, wave_difficulty
        )
        
        # Portal can spawn if available and conditions met
        can_spawn_portal = (
            self.panic_portal_timer <= 0 and 
            portal_mechanics['available'] and 
            not self.portal_active
        )
        
        return {
            'can_spawn': can_spawn_portal,
            'cooldown_remaining': self.panic_portal_timer,
            'portal_active': self.portal_active,
            'portal_timer': self.portal_timer if self.portal_active else 0,
            'mechanics': portal_mechanics
        }
    
    def spawn_panic_portal(self, world_bounds: Tuple[int, int]) -> Tuple[float, float]:
        """Spawn panic portal at random location"""
        if not self.portal_active and self.panic_portal_timer <= 0:
            # Random position within world bounds
            x = random.uniform(50, world_bounds[0] - 50)
            y = random.uniform(50, world_bounds[1] - 50)
            
            self.portal_active = True
            self.portal_timer = self.portal_duration
            
            print(f"Panic portal spawned at ({x:.1f}, {y:.1f})")
            return (x, y)
        
        return None
    
    def update(self, dt: float):
        """Update multi-mode system"""
        # Update transition
        if self.transition_in_progress:
            self.transition_timer -= dt
            if self.transition_timer <= 0:
                self.complete_transition()
        
        # Update anchor points
        for anchor in self.anchor_points.values():
            # Passive healing
            if anchor.health < anchor.max_health:
                heal_rate = anchor.passive_effects['healing_rate']
                anchor.health += heal_rate * dt
                anchor.health = min(anchor.health, anchor.max_health)
        
        # Update global consciousness based on mode activities
        self._update_global_consciousness(dt)
    
    def _update_global_consciousness(self, dt: float):
        """Update global consciousness based on player activities"""
        # This would be influenced by gameplay actions
        # For now, slow natural drift toward balance
        target_balance = 1.0 / 3
        drift_rate = 0.001 * dt
        
        if self.global_consciousness.red > target_balance:
            self.global_consciousness.red -= drift_rate
        elif self.global_consciousness.red < target_balance:
            self.global_consciousness.red += drift_rate
        
        if self.global_consciousness.blue > target_balance:
            self.global_consciousness.blue -= drift_rate
        elif self.global_consciousness.blue < target_balance:
            self.global_consciousness.blue += drift_rate
        
        if self.global_consciousness.yellow > target_balance:
            self.global_consciousness.yellow -= drift_rate
        elif self.global_consciousness.yellow < target_balance:
            self.global_consciousness.yellow += drift_rate
        
        self.global_consciousness.normalize()
    
    def get_mode_transition_effects(self) -> Dict:
        """Get visual effects for mode transitions"""
        if not self.transition_in_progress:
            return {}
        
        progress = 1.0 - (self.transition_timer / 2.0)
        
        # Big Bang style animation parameters
        return {
            'progress': progress,
            'expansion_radius': progress * 500,
            'particle_count': int(progress * 100),
            'consciousness_swirl': {
                'red_intensity': self.global_consciousness.red * progress,
                'blue_intensity': self.global_consciousness.blue * progress,
                'yellow_intensity': self.global_consciousness.yellow * progress
            },
            'reality_warp': math.sin(progress * math.pi) * 0.1
        }

# Example integration class for connecting to game systems
class GameModeManager:
    """High-level manager for integrating multi-mode system with game"""
    
    def __init__(self, game_instance):
        self.game = game_instance
        self.math_engine = AEMathEngine()
        self.multi_mode = MultiModeArchitecture(self.math_engine)
        
    def handle_panic_button(self) -> bool:
        """Handle panic button press"""
        portal_status = self.multi_mode.update_panic_portal_system(
            0.0, 
            getattr(self.game.player, 'stress_level', 50.0),
            getattr(self.game, 'current_wave', 1) * 10
        )
        
        if portal_status['can_spawn']:
            portal_pos = self.multi_mode.spawn_panic_portal(
                (self.game.screen.get_width(), self.game.screen.get_height())
            )
            
            if portal_pos:
                # Store portal position in game state
                if hasattr(self.game.state_manager.current_state, 'panic_portal_pos'):
                    self.game.state_manager.current_state.panic_portal_pos = portal_pos
                return True
        
        return False
    
    def use_panic_portal(self, enemies_follow: bool = True) -> bool:
        """Use panic portal to transition to MMORPG mode"""
        return self.multi_mode.initiate_transition(
            GameMode.MMORPG,
            TransitionType.PANIC_PORTAL,
            enemies_follow
        )
    
    def get_current_mode_state(self) -> ModeState:
        """Get current mode state"""
        return self.multi_mode.mode_states[self.multi_mode.current_mode]

# Example usage
if __name__ == "__main__":
    print("=== Multi-Mode Architecture Test ===")
    
    # Initialize system
    math_engine = AEMathEngine()
    multi_mode = MultiModeArchitecture(math_engine)
    
    # Test mode transition
    print(f"Current mode: {multi_mode.current_mode.value}")
    
    # Test anchor placement
    test_consciousness = AEVector(0.4, 0.3, 0.3)
    test_consciousness.normalize()
    
    success = multi_mode.place_anchor((100, 100), "helix_tower", test_consciousness)
    print(f"Anchor placed: {success}")
    
    # Test panic portal
    portal_status = multi_mode.update_panic_portal_system(0.0, 75.0, 50.0)
    print(f"Portal status: {portal_status}")
    
    print("\n=== Multi-Mode Architecture Ready ===")
    print("Ready for integration with game systems.")
