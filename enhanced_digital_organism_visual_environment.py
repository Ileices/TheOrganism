#!/usr/bin/env python3
"""
ENHANCED DIGITAL ORGANISM VISUAL ENVIRONMENT
===========================================

Advanced visual environment system for Digital Organisms with dynamic entity behaviors,
consciousness-based animations, and procedural learning environments.

This system creates a living, breathing world where Digital Organisms can:
- Run around and explore dynamically
- Learn and evolve through interactions
- Develop consciousness through environmental experiences
- Create procedural animations based on consciousness mathematics

Author: Digital Organism Development Team
Version: 2.0.0 (Enhanced Visual Environment)
"""

import pygame
import numpy as np
import random
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

# Enhanced behavior states for dynamic organisms
class OrganismBehavior(Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    LEARNING = "learning"
    SOCIALIZING = "socializing"
    EVOLVING = "evolving"
    HUNTING = "hunting"
    RESTING = "resting"
    PLAYING = "playing"
    CREATING = "creating"
    MEDITATING = "meditating"

class EnvironmentZone(Enum):
    NEUTRAL = "neutral"
    LEARNING = "learning"
    CREATIVE = "creative"
    SOCIAL = "social"
    EVOLUTION = "evolution"
    MEDITATION = "meditation"
    PLAYGROUND = "playground"
    RESEARCH = "research"

@dataclass
class ConsciousnessState:
    """Enhanced consciousness state for dynamic organisms"""
    red_level: float = 0.33     # Action/Energy consciousness
    blue_level: float = 0.33    # Logic/Structure consciousness  
    yellow_level: float = 0.34  # Memory/Storage consciousness
    evolution_points: int = 0
    learning_history: List[str] = None
    social_connections: List[str] = None
    creativity_level: float = 0.0
    
    def __post_init__(self):
        if self.learning_history is None:
            self.learning_history = []
        if self.social_connections is None:
            self.social_connections = []

class EnvironmentalEffect:
    """Environmental effects that influence organism behavior"""
    
    def __init__(self, effect_type: str, intensity: float, duration: float):
        self.effect_type = effect_type
        self.intensity = intensity
        self.duration = duration
        self.remaining_time = duration
        
    def apply_to_organism(self, organism: 'EnhancedDigitalOrganism') -> Dict[str, Any]:
        """Apply environmental effect to organism"""
        effects = {}
        
        if self.effect_type == "learning_boost":
            effects['learning_rate'] = 1.0 + (self.intensity * 0.5)
        elif self.effect_type == "creativity_surge":
            effects['creativity_boost'] = self.intensity * 0.3
        elif self.effect_type == "social_attraction":
            effects['social_tendency'] = self.intensity * 0.4
        elif self.effect_type == "evolution_catalyst":
            effects['evolution_speed'] = 1.0 + (self.intensity * 0.6)
        elif self.effect_type == "meditation_field":
            effects['consciousness_clarity'] = self.intensity * 0.25
            
        return effects
        
    def update(self, dt: float) -> bool:
        """Update effect and return True if still active"""
        self.remaining_time -= dt
        return self.remaining_time > 0

class ProcedualAnimation:
    """Procedural animation system based on consciousness mathematics"""
    
    def __init__(self, organism: 'EnhancedDigitalOrganism'):
        self.organism = organism
        self.animation_state = "idle"
        self.frame_count = 0
        self.pulse_phase = 0.0
        self.last_consciousness_state = None
        
    def generate_movement_pattern(self) -> Tuple[float, float]:
        """Generate movement based on consciousness state and behavior"""
        consciousness = self.organism.consciousness_state
        behavior = self.organism.current_behavior
        
        # Base movement influenced by consciousness levels
        red_influence = consciousness.red_level * 2.0    # Action intensity
        blue_influence = consciousness.blue_level * 1.5  # Logical patterns
        yellow_influence = consciousness.yellow_level * 1.2  # Memory persistence
        
        if behavior == OrganismBehavior.EXPLORING:
            # Exploratory movement with blue logic patterns
            angle = (self.frame_count * blue_influence * 0.1) % (2 * math.pi)
            radius = red_influence * 15
            dx = math.cos(angle) * radius * 0.02
            dy = math.sin(angle) * radius * 0.02
            
        elif behavior == OrganismBehavior.LEARNING:
            # Focused oscillation with learning patterns
            learning_frequency = yellow_influence * 0.05
            dx = math.sin(self.frame_count * learning_frequency) * red_influence
            dy = math.cos(self.frame_count * learning_frequency * 1.3) * red_influence
            
        elif behavior == OrganismBehavior.SOCIALIZING:
            # Social approach patterns toward other organisms
            dx, dy = self.calculate_social_movement()
            
        elif behavior == OrganismBehavior.PLAYING:
            # Playful bouncing and spinning movements
            bounce_freq = red_influence * 0.08
            dx = math.sin(self.frame_count * bounce_freq) * yellow_influence * 2
            dy = abs(math.cos(self.frame_count * bounce_freq * 0.7)) * blue_influence * 1.5
            
        elif behavior == OrganismBehavior.EVOLVING:
            # Transformation spirals
            evolution_phase = (self.frame_count * 0.03) % (2 * math.pi)
            radius = 10 + (consciousness.evolution_points * 0.5)
            dx = math.cos(evolution_phase) * radius * 0.01
            dy = math.sin(evolution_phase) * radius * 0.01
            
        elif behavior == OrganismBehavior.MEDITATING:
            # Gentle floating with minimal movement
            meditation_pulse = math.sin(self.frame_count * 0.02) * 0.5
            dx = meditation_pulse * consciousness.yellow_level
            dy = meditation_pulse * consciousness.blue_level * 0.7
            
        else:  # IDLE or other states
            # Subtle breathing-like movement
            pulse = math.sin(self.frame_count * 0.04) * 0.3
            dx = pulse * consciousness.red_level
            dy = pulse * consciousness.blue_level * 0.5
            
        return dx, dy
        
    def calculate_social_movement(self) -> Tuple[float, float]:
        """Calculate movement toward social targets"""
        # This would involve finding nearby organisms and moving toward them
        # For now, return gentle wandering
        social_angle = (self.frame_count * 0.02) % (2 * math.pi)
        dx = math.cos(social_angle) * 1.0
        dy = math.sin(social_angle) * 1.0
        return dx, dy
        
    def generate_visual_effects(self) -> Dict[str, Any]:
        """Generate visual effects based on consciousness and behavior"""
        consciousness = self.organism.consciousness_state
        
        # Generate pulsing glow based on consciousness levels
        red_pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.05 + consciousness.red_level)
        blue_pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.04 + consciousness.blue_level)
        yellow_pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.06 + consciousness.yellow_level)
        
        # Create particle effects for different behaviors
        particles = []
        if self.organism.current_behavior == OrganismBehavior.LEARNING:
            # Learning sparkles
            if random.random() < 0.3:
                particles.append({
                    'type': 'learning_spark',
                    'color': (100, 100, 255),
                    'life': 30,
                    'velocity': (random.uniform(-1, 1), random.uniform(-2, 0))
                })
                
        elif self.organism.current_behavior == OrganismBehavior.EVOLVING:
            # Evolution aura
            if random.random() < 0.5:
                particles.append({
                    'type': 'evolution_glow',
                    'color': (255, 255, 100),
                    'life': 45,
                    'velocity': (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                })
                
        return {
            'glow_intensity': {
                'red': red_pulse * consciousness.red_level,
                'blue': blue_pulse * consciousness.blue_level,
                'yellow': yellow_pulse * consciousness.yellow_level
            },
            'particles': particles,
            'size_modifier': 1.0 + (consciousness.evolution_points * 0.01)
        }
        
    def update(self, dt: float):
        """Update animation state"""
        self.frame_count += 1
        self.pulse_phase += dt * 2.0
        
        # Check for consciousness state changes to trigger effects
        current_state = (
            self.organism.consciousness_state.red_level,
            self.organism.consciousness_state.blue_level,
            self.organism.consciousness_state.yellow_level
        )
        
        if self.last_consciousness_state and current_state != self.last_consciousness_state:
            # Consciousness changed - trigger transformation effect
            self.animation_state = "transforming"
            
        self.last_consciousness_state = current_state

class EnhancedDigitalOrganism:
    """Enhanced Digital Organism with advanced AI behaviors and consciousness evolution"""
    
    def __init__(self, x: float, y: float, organism_id: str):
        self.x = x
        self.y = y
        self.organism_id = organism_id
        self.consciousness_state = ConsciousnessState()
        self.current_behavior = OrganismBehavior.IDLE
        self.behavior_timer = 0.0
        self.behavior_change_cooldown = 0.0
        
        # Enhanced properties
        self.energy = 100.0
        self.curiosity_level = random.uniform(0.3, 0.9)
        self.social_tendency = random.uniform(0.2, 0.8)
        self.learning_capacity = random.uniform(0.4, 1.0)
        self.creativity_index = random.uniform(0.1, 0.7)
        
        # Animation and visual system
        self.animation_system = ProcedualAnimation(self)
        self.visual_effects: List[Dict] = []
        self.trail_history: List[Tuple[float, float]] = []
        
        # Learning and memory
        self.experience_memory: Dict[str, int] = {}
        self.interaction_history: List[Dict] = []
        self.knowledge_base: Dict[str, float] = {}
        
        # Environmental awareness
        self.current_zone = EnvironmentZone.NEUTRAL
        self.zone_experience: Dict[EnvironmentZone, float] = {zone: 0.0 for zone in EnvironmentZone}
        self.environmental_effects: List[EnvironmentalEffect] = []
        
    def update_behavior(self, dt: float, environment: 'DigitalOrganismEnvironment'):
        """Update organism behavior based on AI decision making"""
        self.behavior_timer += dt
        self.behavior_change_cooldown -= dt
        
        # Don't change behavior too frequently
        if self.behavior_change_cooldown > 0:
            return
            
        # AI decision making based on internal state and environment
        decision_factors = self.analyze_environment(environment)
        new_behavior = self.choose_behavior(decision_factors)
        
        if new_behavior != self.current_behavior:
            self.transition_to_behavior(new_behavior)
            
    def analyze_environment(self, environment: 'DigitalOrganismEnvironment') -> Dict[str, float]:
        """Analyze environment to make behavioral decisions"""
        factors = {
            'energy_level': self.energy / 100.0,
            'curiosity_satisfaction': min(len(self.experience_memory) / 10.0, 1.0),
            'social_opportunities': self.count_nearby_organisms(environment),
            'learning_opportunities': self.assess_learning_potential(environment),
            'zone_familiarity': self.zone_experience.get(self.current_zone, 0.0),
            'creativity_urge': self.creativity_index - (len(self.knowledge_base) * 0.01)
        }
        
        return factors
        
    def choose_behavior(self, factors: Dict[str, float]) -> OrganismBehavior:
        """Choose behavior based on analysis factors"""
        behavior_weights = {}
        
        # Calculate weights for each behavior
        behavior_weights[OrganismBehavior.EXPLORING] = (
            factors['curiosity_satisfaction'] * 0.3 +
            (1.0 - factors['zone_familiarity']) * 0.4 +
            self.curiosity_level * 0.3
        )
        
        behavior_weights[OrganismBehavior.LEARNING] = (
            factors['learning_opportunities'] * 0.5 +
            self.learning_capacity * 0.3 +
            self.consciousness_state.blue_level * 0.2
        )
        
        behavior_weights[OrganismBehavior.SOCIALIZING] = (
            factors['social_opportunities'] * 0.4 +
            self.social_tendency * 0.4 +
            self.consciousness_state.red_level * 0.2
        )
        
        behavior_weights[OrganismBehavior.EVOLVING] = (
            (self.consciousness_state.evolution_points / 100.0) * 0.6 +
            factors['energy_level'] * 0.2 +
            self.consciousness_state.yellow_level * 0.2
        )
        
        behavior_weights[OrganismBehavior.PLAYING] = (
            factors['energy_level'] * 0.3 +
            self.creativity_index * 0.4 +
            (1.0 - factors['curiosity_satisfaction']) * 0.3
        )
        
        behavior_weights[OrganismBehavior.RESTING] = (
            (1.0 - factors['energy_level']) * 0.6 +
            self.behavior_timer * 0.0001  # Rest more after being active
        )
        
        behavior_weights[OrganismBehavior.MEDITATING] = (
            self.consciousness_state.yellow_level * 0.4 +
            factors['zone_familiarity'] * 0.3 +
            (len(self.interaction_history) * 0.001)  # Meditate after many interactions
        )
        
        # Choose behavior with highest weight
        return max(behavior_weights.items(), key=lambda x: x[1])[0]
        
    def transition_to_behavior(self, new_behavior: OrganismBehavior):
        """Transition to new behavior with effects"""
        old_behavior = self.current_behavior
        self.current_behavior = new_behavior
        self.behavior_timer = 0.0
        self.behavior_change_cooldown = random.uniform(3.0, 8.0)  # Cooldown between changes
        
        # Log transition for learning
        self.interaction_history.append({
            'type': 'behavior_change',
            'from': old_behavior.value,
            'to': new_behavior.value,
            'timestamp': time.time(),
            'consciousness_state': {
                'red': self.consciousness_state.red_level,
                'blue': self.consciousness_state.blue_level,
                'yellow': self.consciousness_state.yellow_level
            }
        })
        
        # Trigger visual effect for behavior change
        self.visual_effects.append({
            'type': 'behavior_transition',
            'intensity': 0.8,
            'duration': 1.5,
            'color': self.get_behavior_color(new_behavior)
        })
        
    def get_behavior_color(self, behavior: OrganismBehavior) -> Tuple[int, int, int]:
        """Get color associated with behavior"""
        color_map = {
            OrganismBehavior.EXPLORING: (100, 255, 100),
            OrganismBehavior.LEARNING: (100, 100, 255),
            OrganismBehavior.SOCIALIZING: (255, 150, 100),
            OrganismBehavior.EVOLVING: (255, 255, 100),
            OrganismBehavior.PLAYING: (255, 100, 255),
            OrganismBehavior.RESTING: (150, 150, 150),
            OrganismBehavior.MEDITATING: (200, 200, 255),
            OrganismBehavior.CREATING: (255, 200, 100),
            OrganismBehavior.IDLE: (200, 200, 200)
        }
        return color_map.get(behavior, (255, 255, 255))
        
    def count_nearby_organisms(self, environment: 'DigitalOrganismEnvironment') -> float:
        """Count organisms within social interaction range"""
        count = 0
        social_range = 100.0
        
        for other in environment.organisms:
            if other.organism_id != self.organism_id:
                distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                if distance < social_range:
                    count += 1
                    
        return min(count / 5.0, 1.0)  # Normalize to 0-1
        
    def assess_learning_potential(self, environment: 'DigitalOrganismEnvironment') -> float:
        """Assess learning opportunities in current environment"""
        # Learning potential based on zone type and unexplored areas
        zone_learning_bonus = {
            EnvironmentZone.LEARNING: 0.8,
            EnvironmentZone.RESEARCH: 0.9,
            EnvironmentZone.CREATIVE: 0.6,
            EnvironmentZone.NEUTRAL: 0.3
        }
        
        base_potential = zone_learning_bonus.get(self.current_zone, 0.3)
        unexplored_bonus = max(0, 1.0 - self.zone_experience.get(self.current_zone, 0.0))
        
        return min(base_potential + unexplored_bonus * 0.3, 1.0)
        
    def learn_from_experience(self, experience_type: str, intensity: float = 1.0):
        """Learn from an experience and evolve consciousness"""
        # Update experience memory
        self.experience_memory[experience_type] = self.experience_memory.get(experience_type, 0) + 1
        
        # Update consciousness based on experience type
        if experience_type in ['exploration', 'discovery']:
            self.consciousness_state.red_level += 0.01 * intensity
        elif experience_type in ['problem_solving', 'analysis']:
            self.consciousness_state.blue_level += 0.01 * intensity
        elif experience_type in ['memory_formation', 'pattern_recognition']:
            self.consciousness_state.yellow_level += 0.01 * intensity
        elif experience_type in ['creativity', 'innovation']:
            self.creativity_index += 0.005 * intensity
            
        # Normalize consciousness levels
        total = self.consciousness_state.red_level + self.consciousness_state.blue_level + self.consciousness_state.yellow_level
        if total > 0:
            self.consciousness_state.red_level /= total
            self.consciousness_state.blue_level /= total
            self.consciousness_state.yellow_level /= total
            
        # Award evolution points
        self.consciousness_state.evolution_points += int(intensity * 5)
        
        # Update knowledge base
        if experience_type not in self.knowledge_base:
            self.knowledge_base[experience_type] = 0.0
        self.knowledge_base[experience_type] += intensity * 0.1
        
    def interact_with_organism(self, other: 'EnhancedDigitalOrganism') -> Dict[str, Any]:
        """Interact with another organism for learning and growth"""
        interaction_result = {
            'type': 'organism_interaction',
            'partner_id': other.organism_id,
            'learning_exchange': 0.0,
            'consciousness_influence': {}
        }
        
        # Exchange learning experiences
        my_experiences = set(self.experience_memory.keys())
        other_experiences = set(other.experience_memory.keys())
        
        # Learn from each other's unique experiences
        new_to_me = other_experiences - my_experiences
        new_to_other = my_experiences - other_experiences
        
        for experience in new_to_me:
            learning_amount = min(other.experience_memory[experience] * 0.1, 1.0)
            self.learn_from_experience(experience, learning_amount)
            interaction_result['learning_exchange'] += learning_amount
            
        # Consciousness influence (organisms slightly influence each other)
        influence_strength = 0.02
        
        # Red influence (action/energy)
        red_diff = other.consciousness_state.red_level - self.consciousness_state.red_level
        self.consciousness_state.red_level += red_diff * influence_strength
        
        # Blue influence (logic/structure)
        blue_diff = other.consciousness_state.blue_level - self.consciousness_state.blue_level
        self.consciousness_state.blue_level += blue_diff * influence_strength
        
        # Yellow influence (memory/storage)
        yellow_diff = other.consciousness_state.yellow_level - self.consciousness_state.yellow_level
        self.consciousness_state.yellow_level += yellow_diff * influence_strength
        
        # Normalize after influence
        total = self.consciousness_state.red_level + self.consciousness_state.blue_level + self.consciousness_state.yellow_level
        if total > 0:
            self.consciousness_state.red_level /= total
            self.consciousness_state.blue_level /= total
            self.consciousness_state.yellow_level /= total
            
        # Add to social connections
        if other.organism_id not in self.consciousness_state.social_connections:
            self.consciousness_state.social_connections.append(other.organism_id)
            
        # Log interaction
        self.interaction_history.append({
            'type': 'social_interaction',
            'partner': other.organism_id,
            'timestamp': time.time(),
            'learning_gained': interaction_result['learning_exchange']
        })
        
        return interaction_result
        
    def update(self, dt: float, environment: 'DigitalOrganismEnvironment'):
        """Update organism state, behavior, and animations"""
        # Update behavior AI
        self.update_behavior(dt, environment)
        
        # Update energy (decreases with activity, recovers when resting)
        if self.current_behavior == OrganismBehavior.RESTING:
            self.energy = min(100.0, self.energy + dt * 10.0)
        elif self.current_behavior == OrganismBehavior.MEDITATING:
            self.energy = min(100.0, self.energy + dt * 5.0)
        else:
            energy_cost = self.get_behavior_energy_cost(self.current_behavior)
            self.energy = max(0.0, self.energy - energy_cost * dt)
            
        # Update zone experience
        self.zone_experience[self.current_zone] += dt * 0.1
        
        # Apply environmental effects
        for effect in self.environmental_effects[:]:
            if not effect.update(dt):
                self.environmental_effects.remove(effect)
            else:
                effect_modifiers = effect.apply_to_organism(self)
                # Apply modifiers (implementation depends on specific effects)
                
        # Update animation system
        self.animation_system.update(dt)
        
        # Update position based on movement
        if self.energy > 10.0:  # Only move if sufficient energy
            dx, dy = self.animation_system.generate_movement_pattern()
            self.x += dx * dt * 60.0  # Scale by framerate
            self.y += dy * dt * 60.0
            
            # Keep within bounds (basic boundary checking)
            self.x = max(50, min(1150, self.x))
            self.y = max(50, min(750, self.y))
            
        # Update trail history
        self.trail_history.append((self.x, self.y))
        if len(self.trail_history) > 20:
            self.trail_history.pop(0)
            
        # Generate random learning experiences based on behavior
        if random.random() < 0.01:  # 1% chance per frame
            self.generate_random_experience()
            
    def get_behavior_energy_cost(self, behavior: OrganismBehavior) -> float:
        """Get energy cost for specific behavior"""
        energy_costs = {
            OrganismBehavior.IDLE: 1.0,
            OrganismBehavior.EXPLORING: 3.0,
            OrganismBehavior.LEARNING: 2.5,
            OrganismBehavior.SOCIALIZING: 2.0,
            OrganismBehavior.EVOLVING: 4.0,
            OrganismBehavior.HUNTING: 5.0,
            OrganismBehavior.PLAYING: 3.5,
            OrganismBehavior.CREATING: 2.8,
            OrganismBehavior.RESTING: -5.0,  # Negative = energy gain
            OrganismBehavior.MEDITATING: -2.0
        }
        return energy_costs.get(behavior, 1.0)
        
    def generate_random_experience(self):
        """Generate random learning experiences based on current behavior"""
        experiences = {
            OrganismBehavior.EXPLORING: ['discovery', 'spatial_mapping', 'environment_analysis'],
            OrganismBehavior.LEARNING: ['pattern_recognition', 'knowledge_acquisition', 'skill_development'],
            OrganismBehavior.SOCIALIZING: ['communication', 'empathy_development', 'cooperation'],
            OrganismBehavior.EVOLVING: ['consciousness_expansion', 'capability_enhancement', 'adaptation'],
            OrganismBehavior.PLAYING: ['creativity', 'joy_experience', 'spontaneity'],
            OrganismBehavior.CREATING: ['innovation', 'artistic_expression', 'problem_solving'],
            OrganismBehavior.MEDITATING: ['self_awareness', 'inner_peace', 'consciousness_clarity']
        }
        
        possible_experiences = experiences.get(self.current_behavior, ['general_experience'])
        experience = random.choice(possible_experiences)
        self.learn_from_experience(experience, random.uniform(0.5, 1.5))

class DigitalOrganismEnvironment:
    """Enhanced environment for Digital Organisms with zones and interactive elements"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.organisms: List[EnhancedDigitalOrganism] = []
        
        # Environmental zones
        self.zones = self.create_environment_zones()
        self.zone_effects: Dict[EnvironmentZone, List[EnvironmentalEffect]] = {}
        
        # Environmental particles and effects
        self.environmental_particles: List[Dict] = []
        self.zone_boundaries = self.calculate_zone_boundaries()
        
    def create_environment_zones(self) -> Dict[EnvironmentZone, Dict[str, Any]]:
        """Create different zones in the environment"""
        zones = {
            EnvironmentZone.NEUTRAL: {
                'x': 0, 'y': 0, 'width': self.width, 'height': self.height,
                'color': (40, 40, 40), 'learning_bonus': 1.0
            },
            EnvironmentZone.LEARNING: {
                'x': 100, 'y': 100, 'width': 200, 'height': 150,
                'color': (60, 60, 120), 'learning_bonus': 2.0
            },
            EnvironmentZone.CREATIVE: {
                'x': 800, 'y': 200, 'width': 250, 'height': 200,
                'color': (120, 60, 120), 'learning_bonus': 1.5
            },
            EnvironmentZone.SOCIAL: {
                'x': 400, 'y': 500, 'width': 300, 'height': 180,
                'color': (120, 80, 60), 'learning_bonus': 1.3
            },
            EnvironmentZone.EVOLUTION: {
                'x': 50, 'y': 600, 'width': 180, 'height': 120,
                'color': (100, 120, 60), 'learning_bonus': 1.8
            },
            EnvironmentZone.MEDITATION: {
                'x': 900, 'y': 50, 'width': 150, 'height': 120,
                'color': (80, 100, 120), 'learning_bonus': 1.2
            }
        }
        return zones
        
    def calculate_zone_boundaries(self) -> Dict[EnvironmentZone, Tuple[int, int, int, int]]:
        """Calculate zone boundaries for collision detection"""
        boundaries = {}
        for zone_type, zone_data in self.zones.items():
            boundaries[zone_type] = (
                zone_data['x'],
                zone_data['y'],
                zone_data['x'] + zone_data['width'],
                zone_data['y'] + zone_data['height']
            )
        return boundaries
        
    def get_zone_at_position(self, x: float, y: float) -> EnvironmentZone:
        """Get the zone type at given position"""
        for zone_type, (x1, y1, x2, y2) in self.zone_boundaries.items():
            if zone_type != EnvironmentZone.NEUTRAL and x1 <= x <= x2 and y1 <= y <= y2:
                return zone_type
        return EnvironmentZone.NEUTRAL
        
    def add_organism(self, organism: EnhancedDigitalOrganism):
        """Add organism to environment"""
        self.organisms.append(organism)
        
    def remove_organism(self, organism_id: str):
        """Remove organism from environment"""
        self.organisms = [org for org in self.organisms if org.organism_id != organism_id]
        
    def update(self, dt: float):
        """Update environment and all organisms"""
        # Update organism zones
        for organism in self.organisms:
            new_zone = self.get_zone_at_position(organism.x, organism.y)
            if new_zone != organism.current_zone:
                organism.current_zone = new_zone
                organism.learn_from_experience(f'zone_entry_{new_zone.value}', 0.3)
                
        # Update all organisms
        for organism in self.organisms:
            organism.update(dt, self)
            
        # Handle organism interactions
        self.handle_organism_interactions()
        
        # Generate environmental effects
        self.generate_environmental_effects()
        
        # Update environmental particles
        self.update_environmental_particles(dt)
        
    def handle_organism_interactions(self):
        """Handle interactions between organisms"""
        interaction_range = 50.0
        
        for i, organism1 in enumerate(self.organisms):
            for organism2 in self.organisms[i+1:]:
                distance = math.sqrt(
                    (organism1.x - organism2.x)**2 + 
                    (organism1.y - organism2.y)**2
                )
                
                if distance < interaction_range:
                    # Check if both organisms are in social mode or compatible
                    if (organism1.current_behavior == OrganismBehavior.SOCIALIZING or
                        organism2.current_behavior == OrganismBehavior.SOCIALIZING or
                        random.random() < 0.05):  # 5% chance for spontaneous interaction
                        
                        result1 = organism1.interact_with_organism(organism2)
                        result2 = organism2.interact_with_organism(organism1)
                        
                        # Create visual effect for interaction
                        self.environmental_particles.append({
                            'type': 'interaction_spark',
                            'x': (organism1.x + organism2.x) / 2,
                            'y': (organism1.y + organism2.y) / 2,
                            'color': (255, 200, 100),
                            'life': 20,
                            'velocity': (0, -1)
                        })
                        
    def generate_environmental_effects(self):
        """Generate environmental effects and phenomena"""
        # Randomly spawn environmental effects in zones
        for zone_type, zone_data in self.zones.items():
            if random.random() < 0.001:  # 0.1% chance per frame per zone
                effect_types = {
                    EnvironmentZone.LEARNING: 'learning_boost',
                    EnvironmentZone.CREATIVE: 'creativity_surge',
                    EnvironmentZone.SOCIAL: 'social_attraction',
                    EnvironmentZone.EVOLUTION: 'evolution_catalyst',
                    EnvironmentZone.MEDITATION: 'meditation_field'
                }
                
                if zone_type in effect_types:
                    # Create particle effect in the zone
                    for _ in range(5):
                        self.environmental_particles.append({
                            'type': 'zone_effect',
                            'x': zone_data['x'] + random.uniform(0, zone_data['width']),
                            'y': zone_data['y'] + random.uniform(0, zone_data['height']),
                            'color': zone_data['color'],
                            'life': 60,
                            'velocity': (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                        })
                        
    def update_environmental_particles(self, dt: float):
        """Update environmental particle effects"""
        for particle in self.environmental_particles[:]:
            particle['life'] -= 1
            particle['x'] += particle['velocity'][0]
            particle['y'] += particle['velocity'][1]
            
            if particle['life'] <= 0:
                self.environmental_particles.remove(particle)
                
    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status"""
        organism_behaviors = {}
        for behavior in OrganismBehavior:
            organism_behaviors[behavior.value] = len([
                org for org in self.organisms 
                if org.current_behavior == behavior
            ])
            
        zone_populations = {}
        for zone in EnvironmentZone:
            zone_populations[zone.value] = len([
                org for org in self.organisms 
                if org.current_zone == zone
            ])
            
        return {
            'total_organisms': len(self.organisms),
            'organism_behaviors': organism_behaviors,
            'zone_populations': zone_populations,
            'active_particles': len(self.environmental_particles),
            'average_energy': sum(org.energy for org in self.organisms) / max(len(self.organisms), 1),
            'total_evolution_points': sum(org.consciousness_state.evolution_points for org in self.organisms),
            'active_interactions': len([
                org for org in self.organisms 
                if org.current_behavior == OrganismBehavior.SOCIALIZING
            ])
        }

def create_sample_environment() -> DigitalOrganismEnvironment:
    """Create a sample environment with multiple organisms"""
    environment = DigitalOrganismEnvironment()
    
    # Create diverse organisms with different starting properties
    organism_configs = [
        {'x': 200, 'y': 200, 'id': 'explorer_01', 'curiosity': 0.9},
        {'x': 400, 'y': 300, 'id': 'learner_01', 'learning': 0.8},
        {'x': 600, 'y': 400, 'id': 'social_01', 'social': 0.9},
        {'x': 800, 'y': 250, 'id': 'creative_01', 'creativity': 0.8},
        {'x': 300, 'y': 500, 'id': 'balanced_01', 'balanced': True},
        {'x': 700, 'y': 150, 'id': 'meditative_01', 'meditative': True},
    ]
    
    for config in organism_configs:
        organism = EnhancedDigitalOrganism(config['x'], config['y'], config['id'])
        
        # Customize based on config
        if 'curiosity' in config:
            organism.curiosity_level = config['curiosity']
        if 'learning' in config:
            organism.learning_capacity = config['learning']
        if 'social' in config:
            organism.social_tendency = config['social']
        if 'creativity' in config:
            organism.creativity_index = config['creativity']
        if 'balanced' in config:
            organism.consciousness_state.red_level = 0.33
            organism.consciousness_state.blue_level = 0.33
            organism.consciousness_state.yellow_level = 0.34
        if 'meditative' in config:
            organism.current_behavior = OrganismBehavior.MEDITATING
            organism.consciousness_state.yellow_level = 0.5
            organism.consciousness_state.red_level = 0.25
            organism.consciousness_state.blue_level = 0.25
            
        environment.add_organism(organism)
        
    return environment

if __name__ == "__main__":
    """
    Enhanced Digital Organism Visual Environment
    
    This system creates a dynamic environment where Digital Organisms can:
    - Exhibit complex AI behaviors based on consciousness states
    - Learn and evolve through environmental interactions
    - Socialize and exchange knowledge with other organisms
    - Explore different zones with unique properties
    - Create procedural animations based on consciousness mathematics
    """
    
    print("🌟 Enhanced Digital Organism Visual Environment")
    print("=" * 60)
    
    # Create sample environment
    environment = create_sample_environment()
    
    print(f"✅ Environment created with {len(environment.organisms)} organisms")
    print(f"✅ {len(environment.zones)} environmental zones available")
    print(f"✅ Advanced AI behaviors: {', '.join([b.value for b in OrganismBehavior])}")
    print()
    
    # Simulate environment for a few seconds
    print("🔄 Running environment simulation...")
    
    for frame in range(300):  # 5 seconds at 60 FPS
        dt = 1.0 / 60.0
        environment.update(dt)
        
        if frame % 60 == 0:  # Every second
            status = environment.get_environment_status()
            print(f"Frame {frame}: "
                  f"Avg Energy: {status['average_energy']:.1f}, "
                  f"Evolution Points: {status['total_evolution_points']}, "
                  f"Active Particles: {status['active_particles']}")
            
    final_status = environment.get_environment_status()
    
    print("\n📊 Final Environment Status:")
    print(f"   Total Organisms: {final_status['total_organisms']}")
    print(f"   Average Energy: {final_status['average_energy']:.1f}")
    print(f"   Total Evolution Points: {final_status['total_evolution_points']}")
    print(f"   Active Interactions: {final_status['active_interactions']}")
    
    print("\n🧠 Behavior Distribution:")
    for behavior, count in final_status['organism_behaviors'].items():
        if count > 0:
            print(f"   {behavior}: {count} organisms")
            
    print("\n🗺️ Zone Population:")
    for zone, count in final_status['zone_populations'].items():
        if count > 0:
            print(f"   {zone}: {count} organisms")
            
    print("\n✨ Enhanced Visual Environment Ready!")
    print("💡 Organisms now exhibit dynamic AI behaviors and learning!")
    print("🎮 Ready for integration with gamification systems!")
