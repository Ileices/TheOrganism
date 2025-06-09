"""
Unified Absolute Framework Integration System
Connects consciousness mathematics, multi-mode architecture, and procedural generation
with the existing GeoBIT tower defense game for seamless framework implementation
"""

import pygame
import math
import json
import websocket
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Import framework components
from ae_consciousness_mathematics import AEMathEngine, AEVector, ConsciousnessGameIntegration
from ae_multimode_architecture import MultiModeArchitecture, GameMode, TransitionType, AnchorPoint
from ae_procedural_world_generation import ProceduralWorldGenerator, ZoneData

# Constants for integration
LEADERBOARD_SERVER_URL = "ws://localhost:8765"
CONSCIOUSNESS_UPDATE_INTERVAL = 1.0  # Update every second
FRAMEWORK_VERSION = "1.0.0"

class IntegrationState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCING = "syncing"
    ACTIVE = "active"
    ERROR = "error"

@dataclass
class PlayerFrameworkState:
    """Complete player state in the Unified Absolute Framework"""
    player_id: str
    consciousness: AEVector
    current_dimension: int
    current_zone: int
    total_mass: float
    framework_level: int
    anchors_placed: int
    max_anchors: int
    upgrade_count: int
    mode_progression: Dict[str, int]
    dna_pattern: str
    server_consciousness: AEVector

class UnifiedFrameworkIntegration:
    """Main integration system connecting all framework components"""
    
    def __init__(self, game_instance):
        self.game = game_instance
        self.integration_state = IntegrationState.DISCONNECTED
        
        # Initialize framework components
        self.math_engine = AEMathEngine()
        self.consciousness_integration = ConsciousnessGameIntegration(self.math_engine)
        self.multi_mode = MultiModeArchitecture(self.math_engine)
        self.world_generator = ProceduralWorldGenerator(self.math_engine)
        
        # Player framework state
        self.player_state = PlayerFrameworkState(
            player_id="player_001",
            consciousness=AEVector(),
            current_dimension=1,
            current_zone=1,
            total_mass=0.0,
            framework_level=1,
            anchors_placed=0,
            max_anchors=4,
            upgrade_count=0,
            mode_progression={
                'tower_defense': 1,
                'mmorpg': 0,
                'base_builder': 0,
                'game_of_life': 0
            },
            dna_pattern="",
            server_consciousness=AEVector()
        )
        
        # WebSocket connection for leaderboard server
        self.websocket = None
        self.websocket_thread = None
        self.last_consciousness_update = 0.0
        
        # Framework integration flags
        self.consciousness_enabled = True
        self.multimode_enabled = True
        self.procedural_generation_enabled = True
        self.leaderboard_integration_enabled = True
        
        # Game state synchronization
        self.game_stats_cache = {}
        self.consciousness_evolution_queue = []
        
        # Initialize framework
        self.initialize_framework()
    
    def initialize_framework(self):
        """Initialize the complete Unified Absolute Framework"""
        print("Initializing Unified Absolute Framework...")
        
        # Generate initial consciousness
        self.player_state.consciousness = self._generate_initial_consciousness()
        
        # Generate DNA pattern
        self.player_state.dna_pattern = self.math_engine.dna_pattern_generation(
            self.player_state.consciousness
        )
        
        # Connect to leaderboard server
        if self.leaderboard_integration_enabled:
            self._connect_to_leaderboard_server()
        
        # Initialize current zone
        self._update_current_zone()
        
        print("Framework initialization complete.")
        print(f"Player consciousness: R:{self.player_state.consciousness.red:.3f}, "
              f"B:{self.player_state.consciousness.blue:.3f}, "
              f"Y:{self.player_state.consciousness.yellow:.3f}")
        print(f"DNA pattern: {self.player_state.dna_pattern}")
        
        self.integration_state = IntegrationState.ACTIVE
    
    def _generate_initial_consciousness(self) -> AEVector:
        """Generate initial player consciousness"""
        # Start with slight randomization from perfect balance
        import random
        random.seed(42)  # Reproducible for testing
        
        consciousness = AEVector(
            0.333 + random.uniform(-0.05, 0.05),
            0.333 + random.uniform(-0.05, 0.05),
            0.334 + random.uniform(-0.05, 0.05)
        )
        consciousness.normalize()
        
        return consciousness
    
    def _connect_to_leaderboard_server(self):
        """Connect to the AE leaderboard server"""
        try:
            self.websocket_thread = threading.Thread(
                target=self._websocket_worker,
                daemon=True
            )
            self.websocket_thread.start()
            self.integration_state = IntegrationState.CONNECTING
        except Exception as e:
            print(f"Failed to connect to leaderboard server: {e}")
            self.integration_state = IntegrationState.ERROR
    
    def _websocket_worker(self):
        """WebSocket worker thread for leaderboard integration"""
        try:
            self.websocket = websocket.WebSocketApp(
                LEADERBOARD_SERVER_URL,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )
            self.websocket.run_forever()
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.integration_state = IntegrationState.ERROR
    
    def _on_websocket_open(self, ws):
        """Handle WebSocket connection opened"""
        print("Connected to AE leaderboard server")
        self.integration_state = IntegrationState.CONNECTED
        
        # Send initial player registration
        self._send_player_registration()
    
    def _on_websocket_message(self, ws, message):
        """Handle WebSocket message received"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'consciousness_update':
                self._handle_consciousness_update(data)
            elif message_type == 'server_consciousness':
                self._handle_server_consciousness_update(data)
            elif message_type == 'dna_compatibility':
                self._handle_dna_compatibility(data)
            elif message_type == 'leaderboard_update':
                self._handle_leaderboard_update(data)
                
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket error"""
        print(f"WebSocket error: {error}")
        self.integration_state = IntegrationState.ERROR
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed"""
        print("WebSocket connection closed")
        self.integration_state = IntegrationState.DISCONNECTED
    
    def _send_player_registration(self):
        """Send player registration to leaderboard server"""
        if not self.websocket:
            return
        
        registration_data = {
            'type': 'player_register',
            'player_id': self.player_state.player_id,
            'consciousness': {
                'red': self.player_state.consciousness.red,
                'blue': self.player_state.consciousness.blue,
                'yellow': self.player_state.consciousness.yellow
            },
            'dna_pattern': self.player_state.dna_pattern,
            'dimension': self.player_state.current_dimension,
            'zone': self.player_state.current_zone,
            'total_mass': self.player_state.total_mass,
            'framework_version': FRAMEWORK_VERSION
        }
        
        try:
            self.websocket.send(json.dumps(registration_data))
        except Exception as e:
            print(f"Failed to send registration: {e}")
    
    def update(self, dt: float):
        """Update framework integration systems"""
        if self.integration_state != IntegrationState.ACTIVE:
            return
        
        # Update multi-mode architecture
        if self.multimode_enabled:
            self.multi_mode.update(dt)
        
        # Update consciousness based on gameplay
        self._update_consciousness_from_gameplay(dt)
        
        # Send consciousness updates to server
        self.last_consciousness_update += dt
        if self.last_consciousness_update >= CONSCIOUSNESS_UPDATE_INTERVAL:
            self._send_consciousness_update()
            self.last_consciousness_update = 0.0
        
        # Process consciousness evolution queue
        self._process_consciousness_evolution()
        
        # Update player framework state
        self._synchronize_game_state()
    
    def _update_consciousness_from_gameplay(self, dt: float):
        """Update player consciousness based on gameplay actions"""
        if not self.consciousness_enabled:
            return
        
        # Gather experience from gameplay
        experience = {
            'combat': 0,
            'building': 0,
            'strategy': 0,
            'puzzles': 0,
            'exploration': 0,
            'balance': 0
        }
        
        # Extract experience from game state
        if hasattr(self.game, 'enemies') and self.game.enemies:
            experience['combat'] = len([e for e in self.game.enemies if not e.alive]) * dt
        
        if hasattr(self.game, 'bits') and self.game.bits:
            experience['building'] = len(self.game.bits) * dt * 0.1
        
        # Strategic play (bit placement, upgrades)
        if hasattr(self.game, 'player') and hasattr(self.game.player, 'upgrade_count'):
            experience['strategy'] = self.game.player.upgrade_count * dt * 0.01
        
        # Balanced consciousness maintenance
        consciousness_balance = self.player_state.consciousness.consciousness_factor()
        experience['balance'] = consciousness_balance * dt * 0.5
        
        # Evolve consciousness
        if sum(experience.values()) > 0:
            evolved_consciousness = self.math_engine.consciousness_evolution(
                self.player_state.consciousness, experience
            )
            self.player_state.consciousness = evolved_consciousness
    
    def _send_consciousness_update(self):
        """Send consciousness update to leaderboard server"""
        if not self.websocket:
            return
        
        update_data = {
            'type': 'consciousness_update',
            'player_id': self.player_state.player_id,
            'consciousness': {
                'red': self.player_state.consciousness.red,
                'blue': self.player_state.consciousness.blue,
                'yellow': self.player_state.consciousness.yellow
            },
            'total_mass': self.player_state.total_mass,
            'dimension': self.player_state.current_dimension,
            'zone': self.player_state.current_zone,
            'framework_level': self.player_state.framework_level
        }
        
        try:
            self.websocket.send(json.dumps(update_data))
        except Exception as e:
            print(f"Failed to send consciousness update: {e}")
    
    def _process_consciousness_evolution(self):
        """Process queued consciousness evolution events"""
        for evolution_event in self.consciousness_evolution_queue:
            # Apply evolution to consciousness
            self.player_state.consciousness = self.math_engine.consciousness_evolution(
                self.player_state.consciousness, evolution_event
            )
        
        # Clear processed events
        self.consciousness_evolution_queue.clear()
    
    def _synchronize_game_state(self):
        """Synchronize framework state with game state"""
        # Update total mass from game resources
        if hasattr(self.game, 'player'):
            player = self.game.player
            
            # Calculate total mass from all resources
            vapor = getattr(player, 'vapor', 0)
            dna = getattr(player, 'dna', 0) * 10  # DNA is more valuable
            photon = getattr(player, 'photon', 0) * 5  # Photons are valuable
            
            self.player_state.total_mass = vapor + dna + photon
            
            # Update upgrade count
            self.player_state.upgrade_count = getattr(player, 'upgrade_count', 0)
            
            # Update anchor capacity
            self.player_state.max_anchors = self.multi_mode.upgrade_anchor_capacity(
                self.player_state.upgrade_count
            )
    
    def _update_current_zone(self):
        """Update current zone data and effects"""
        current_zone = self.world_generator.get_zone(
            self.player_state.current_dimension,
            self.player_state.current_zone
        )
        
        if current_zone:
            # Apply zone effects to game
            self._apply_zone_effects(current_zone)
    
    def _apply_zone_effects(self, zone_data: ZoneData):
        """Apply zone environmental effects to game"""
        effects = zone_data.environmental_effects
        
        # Apply effects to player
        if hasattr(self.game, 'player'):
            player = self.game.player
            
            # Healing rate bonus
            if 'healing_rate_bonus' in effects and hasattr(player, 'healing_rate'):
                player.healing_rate *= (1 + effects['healing_rate_bonus'])
            
            # Experience multiplier
            if 'experience_multiplier' in effects and hasattr(player, 'experience_multiplier'):
                player.experience_multiplier = effects['experience_multiplier']
    
    def handle_panic_portal_activation(self) -> bool:
        """Handle panic portal activation"""
        if not self.multimode_enabled:
            return False
        
        # Check with multi-mode system
        portal_status = self.multi_mode.update_panic_portal_system(
            0.0,
            getattr(self.game.player, 'stress_level', 50.0),
            getattr(self.game, 'current_wave', 1) * 10
        )
        
        if portal_status['can_spawn']:
            # Spawn portal in game
            portal_pos = self.multi_mode.spawn_panic_portal(
                (self.game.screen.get_width(), self.game.screen.get_height())
            )
            
            if portal_pos and hasattr(self.game.state_manager.current_state, 'panic_portal_pos'):
                self.game.state_manager.current_state.panic_portal_pos = portal_pos
                
                # Add consciousness evolution for panic usage
                self.consciousness_evolution_queue.append({
                    'combat': 5.0,  # Panic indicates combat stress
                    'strategy': -2.0  # Panic reduces strategic thinking
                })
                
                return True
        
        return False
    
    def handle_portal_usage(self, enemies_follow: bool = True) -> bool:
        """Handle portal usage for mode transition"""
        if not self.multimode_enabled:
            return False
        
        # Initiate transition through multi-mode system
        success = self.multi_mode.initiate_transition(
            GameMode.MMORPG,
            TransitionType.PANIC_PORTAL,
            enemies_follow
        )
        
        if success:
            # Update player progression
            self.player_state.mode_progression['mmorpg'] += 1
            
            # Add consciousness evolution for exploration
            self.consciousness_evolution_queue.append({
                'exploration': 3.0,
                'balance': 1.0
            })
        
        return success
    
    def handle_tower_placement(self, position: Tuple[float, float], 
                             tower_type: str) -> bool:
        """Handle tower placement with consciousness integration"""
        # Calculate tower consciousness based on type and player consciousness
        tower_consciousness = self._calculate_tower_consciousness(tower_type)
        
        # Place anchor through multi-mode system
        success = self.multi_mode.place_anchor(position, tower_type, tower_consciousness)
        
        if success:
            self.player_state.anchors_placed += 1
            
            # Add consciousness evolution for building
            self.consciousness_evolution_queue.append({
                'building': 2.0,
                'strategy': 1.0
            })
        
        return success
    
    def _calculate_tower_consciousness(self, tower_type: str) -> AEVector:
        """Calculate consciousness signature for tower type"""
        # Base consciousness from player
        base = self.player_state.consciousness
        
        # Modify based on tower type
        if tower_type == "helix":  # Red-dominant (action)
            return AEVector(base.red + 0.1, base.blue - 0.05, base.yellow - 0.05)
        elif tower_type == "myco":  # Blue-dominant (logic)
            return AEVector(base.red - 0.05, base.blue + 0.1, base.yellow - 0.05)
        elif tower_type == "photon":  # Yellow-dominant (wisdom)
            return AEVector(base.red - 0.05, base.blue - 0.05, base.yellow + 0.1)
        else:  # Balanced
            return AEVector(base.red, base.blue, base.yellow)
    
    def calculate_tower_effectiveness(self, tower_consciousness: AEVector,
                                   enemy_consciousness: AEVector) -> float:
        """Calculate tower effectiveness using consciousness mathematics"""
        return self.consciousness_integration.tower_effectiveness(
            tower_consciousness, enemy_consciousness
        )
    
    def generate_loot_rps(self, enemy_consciousness: AEVector) -> Dict[str, float]:
        """Generate loot using RPS instead of randomness"""
        current_zone = self.world_generator.get_zone(
            self.player_state.current_dimension,
            self.player_state.current_zone
        )
        
        if current_zone:
            return self.consciousness_integration.loot_generation_rps(
                enemy_consciousness, current_zone.consciousness_field
            )
        
        return {}
    
    def get_current_zone_effects(self) -> Dict[str, float]:
        """Get environmental effects for current zone"""
        current_zone = self.world_generator.get_zone(
            self.player_state.current_dimension,
            self.player_state.current_zone
        )
        
        if current_zone:
            return current_zone.environmental_effects
        
        return {}
    
    def get_framework_ui_data(self) -> Dict:
        """Get UI data for framework display"""
        return {
            'consciousness': {
                'red': self.player_state.consciousness.red,
                'blue': self.player_state.consciousness.blue,
                'yellow': self.player_state.consciousness.yellow,
                'unity': self.player_state.consciousness.ae_unity(),
                'balance': self.player_state.consciousness.consciousness_factor()
            },
            'location': {
                'dimension': self.player_state.current_dimension,
                'zone': self.player_state.current_zone
            },
            'progression': {
                'framework_level': self.player_state.framework_level,
                'total_mass': self.player_state.total_mass,
                'anchors': f"{self.player_state.anchors_placed}/{self.player_state.max_anchors}",
                'upgrades': self.player_state.upgrade_count
            },
            'mode_state': self.multi_mode.current_mode.value,
            'portal_status': self.multi_mode.update_panic_portal_system(0.0, 50.0, 10.0),
            'dna_pattern': self.player_state.dna_pattern,
            'integration_state': self.integration_state.value
        }
    
    def handle_zone_transition(self, target_dimension: int, target_zone: int) -> bool:
        """Handle transition to different zone"""
        if not self.procedural_generation_enabled:
            return False
        
        # Calculate travel cost
        travel_cost = self.world_generator.calculate_travel_cost(
            self.player_state.current_dimension,
            self.player_state.current_zone,
            target_dimension,
            target_zone
        )
        
        # Check if player can afford travel
        if self.player_state.total_mass >= travel_cost:
            self.player_state.total_mass -= travel_cost
            self.player_state.current_dimension = target_dimension
            self.player_state.current_zone = target_zone
            
            # Update zone effects
            self._update_current_zone()
            
            # Add consciousness evolution for exploration
            self.consciousness_evolution_queue.append({
                'exploration': 2.0,
                'wisdom': 1.0
            })
            
            return True
        
        return False
    
    def _handle_consciousness_update(self, data: Dict):
        """Handle consciousness update from server"""
        # Process consciousness synchronization
        pass
    
    def _handle_server_consciousness_update(self, data: Dict):
        """Handle server consciousness update"""
        server_consciousness = data.get('server_consciousness', {})
        self.player_state.server_consciousness = AEVector(
            server_consciousness.get('red', 0.333),
            server_consciousness.get('blue', 0.333),
            server_consciousness.get('yellow', 0.334)
        )
    
    def _handle_dna_compatibility(self, data: Dict):
        """Handle DNA compatibility update"""
        # Process DNA compatibility for server merging
        pass
    
    def _handle_leaderboard_update(self, data: Dict):
        """Handle leaderboard update"""
        # Process leaderboard changes
        pass
    
    def shutdown(self):
        """Shutdown framework integration"""
        if self.websocket:
            self.websocket.close()
        
        print("Unified Absolute Framework integration shutdown complete.")

# Integration helper functions for game system
def integrate_framework_with_game(game_instance) -> UnifiedFrameworkIntegration:
    """Initialize and integrate framework with existing game"""
    integration = UnifiedFrameworkIntegration(game_instance)
    
    # Hook into game events
    if hasattr(game_instance, 'on_tower_placed'):
        game_instance.on_tower_placed = integration.handle_tower_placement
    
    if hasattr(game_instance, 'on_panic_portal'):
        game_instance.on_panic_portal = integration.handle_panic_portal_activation
    
    if hasattr(game_instance, 'on_portal_use'):
        game_instance.on_portal_use = integration.handle_portal_usage
    
    return integration

# Example usage
if __name__ == "__main__":
    print("=== Unified Absolute Framework Integration Test ===")
    
    # Mock game instance for testing
    class MockGame:
        def __init__(self):
            self.screen = pygame.Surface((800, 600))
            self.player = type('Player', (), {
                'vapor': 100.0,
                'dna': 10.0,
                'photon': 5.0,
                'upgrade_count': 50
            })()
            self.bits = []
            self.enemies = []
    
    mock_game = MockGame()
    integration = integrate_framework_with_game(mock_game)
    
    # Test framework data
    ui_data = integration.get_framework_ui_data()
    print(f"Framework UI Data: {ui_data}")
    
    # Test tower placement
    tower_placed = integration.handle_tower_placement((100, 100), "helix")
    print(f"Tower placed: {tower_placed}")
    
    # Test portal activation
    portal_activated = integration.handle_panic_portal_activation()
    print(f"Portal activated: {portal_activated}")
    
    print("\n=== Framework Integration Ready ===")
    print("Complete Unified Absolute Framework integrated with game systems.")
    print("Ready for deployment with existing GeoBIT tower defense game.")
