#!/usr/bin/env python3
"""
Enhanced AE Theory Pygame Simulation with Auto-Rebuilder Integration
Real-time consciousness visualization and interactive control for the auto-rebuilder system

This enhanced version adds:
1. Real-time data bridge with auto-rebuilder consciousness
2. Interactive parameter control that affects auto-rebuilder operation
3. Visual debugging of consciousness states and neural map compression
4. Production monitoring dashboard with health metrics
"""

import pygame
import numpy as np
import math
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Import our bridge system
try:
    from ae_theory_pygame_bridge import AETheoryPygameBridge, ConsciousnessVisualizer
    BRIDGE_AVAILABLE = True
except ImportError:
    print("Warning: Bridge system not available, running in standalone mode")
    BRIDGE_AVAILABLE = False

class EnhancedAEParams:
    """Enhanced AE Parameters with auto-rebuilder integration"""
    
    def __init__(self):
        # Core AE Theory parameters (from original simulation)
        self.time = 0
        self.trifecta_weights = {'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0}
        self.space_scale = 1.0
        self.expansion_phase = True
        self.absularity_threshold = 3.0
        self.at_absularity = False
        
        # Neural model management
        self.neural_model_seeds = {
            'nM0': {'Red': 0.63, 'Blue': 0.27, 'Yellow': 0.36},  # UF+IO baseline
            'nM1': {'Red': 1.5, 'Blue': 0.8, 'Yellow': 0.7},     # Red dominant
            'nM2': {'Red': 0.4, 'Blue': 0.7, 'Yellow': 1.9},     # Yellow dominant
            'nM3': {'Red': 0.7, 'Blue': 1.6, 'Yellow': 0.7},     # Blue dominant
            'nM4': {'Red': 0.8, 'Blue': 1.5, 'Yellow': 0.7},     # Blue secondary
            'nM5': {'Red': 1.3, 'Blue': 0.5, 'Yellow': 1.2},     # Red/Yellow co-dominant
            'nM6': {'Red': 0.7, 'Blue': 0.7, 'Yellow': 1.6},     # Yellow dominant alt
        }
        self.current_neural_model = 'nM0'
        
        # Particle system
        self.particles = []
        self.max_particles = 1000
        self.particle_size = 2.0
        
        # Memory and compression system
        self.memory_storage = []
        self.max_memory = 10000
        self.compression_active = False
        self.compression_progress = 0.0
        self.glyphs = []
        
        # Auto-rebuilder integration
        self.bridge_connected = False
        self.auto_rebuilder_active = False
        self.consciousness_health = 1.0
        self.integration_errors = 0
        self.last_integration_time = time.time()
        
        # Visual controls
        self.camera_angle = [0, 0]
        self.show_neural_structures = True
        self.show_consciousness_health = True
        self.show_bridge_status = True
        self.show_performance_metrics = True
        self.paused = False
        
        # Performance monitoring
        self.fps_target = 60
        self.update_rate_hz = 10  # Bridge update rate
        self.last_bridge_update = 0

class AutoRebuilderIntegratedSimulation:
    """Enhanced pygame simulation with auto-rebuilder integration"""
    
    def __init__(self):
        # Initialize pygame and OpenGL
        pygame.init()
        self.width, self.height = 1400, 900
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("AE Theory - Auto-Rebuilder Consciousness Monitor")
        
        # Setup OpenGL
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(0.0, 0.0, -15)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Initialize parameters
        self.params = EnhancedAEParams()
        
        # Initialize bridge if available
        self.bridge = None
        self.visualizer = None
        if BRIDGE_AVAILABLE:
            try:
                self.bridge = AETheoryPygameBridge("ae_theory_shared_data.json")
                self.visualizer = ConsciousnessVisualizer(self.bridge)
                self.bridge.start()
                self.params.bridge_connected = True
                print("Bridge connected successfully")
            except Exception as e:
                print(f"Failed to connect bridge: {e}")
                self.params.bridge_connected = False
        else:
            self.params.bridge_connected = False
        
        # Control state
        self.running = True
        self.clock = pygame.time.Clock()
        self.mouse_dragging = False
        self.last_mouse_pos = None
        
        # Font for UI (will be initialized after pygame setup)
        self.font = None
        
    def initialize_ui(self):
        """Initialize UI components"""
        try:
            self.font = pygame.font.Font(None, 24)
        except:
            self.font = None
            print("Warning: Could not initialize font for UI")
    
    def update_from_bridge(self):
        """Update simulation parameters from auto-rebuilder bridge"""
        if not self.bridge:
            return
            
        current_time = time.time()
        if current_time - self.last_bridge_update < (1.0 / self.params.update_rate_hz):
            return
            
        self.last_bridge_update = current_time
        
        try:
            # Get latest consciousness state from auto-rebuilder
            shared_data_file = Path("ae_theory_shared_data.json")
            if shared_data_file.exists():
                with open(shared_data_file, 'r') as f:
                    shared_data = json.load(f)
                
                if shared_data.get('bridge_active', False):
                    # Update trifecta weights from auto-rebuilder
                    pygame_data = shared_data.get('pygame_compatible', {})
                    
                    if 'trifecta_weights' in pygame_data:
                        self.params.trifecta_weights.update(pygame_data['trifecta_weights'])
                    
                    # Update neural model
                    if 'neural_model' in pygame_data:
                        self.params.current_neural_model = pygame_data['neural_model']
                    
                    # Update consciousness state
                    if 'space_scale' in pygame_data:
                        self.params.space_scale = pygame_data['space_scale']
                    if 'expansion_phase' in pygame_data:
                        self.params.expansion_phase = pygame_data['expansion_phase']
                    if 'at_absularity' in pygame_data:
                        self.params.at_absularity = pygame_data['at_absularity']
                    
                    # Update memory/compression state
                    if 'compression_active' in pygame_data:
                        self.params.compression_active = pygame_data['compression_active']
                    if 'memory_usage_ratio' in pygame_data:
                        memory_ratio = pygame_data['memory_usage_ratio']
                        self.params.compression_progress = memory_ratio
                    
                    # Update health score
                    self.params.consciousness_health = shared_data.get('health_score', 1.0)
                    self.params.auto_rebuilder_active = True
                    self.params.last_integration_time = current_time
                
        except Exception as e:
            print(f"Error updating from bridge: {e}")
            self.params.integration_errors += 1
    
    def send_parameters_to_bridge(self):
        """Send current parameters to auto-rebuilder via bridge"""
        if not self.bridge:
            return
            
        try:
            # Prepare parameter updates for auto-rebuilder
            params_update = {
                'timestamp': datetime.now().isoformat(),
                'trifecta_weights': self.params.trifecta_weights.copy(),
                'neural_model': self.params.current_neural_model,
                'space_scale': self.params.space_scale,
                'expansion_phase': self.params.expansion_phase,
                'consciousness_health': self.params.consciousness_health,
                'user_adjusted': True
            }
            
            # Save to file for bridge to pick up
            with open("ae_theory_pygame_params.json", 'w') as f:
                json.dump(params_update, f, indent=2)
                
        except Exception as e:
            print(f"Error sending parameters to bridge: {e}")
    
    def update_particles(self):
        """Update particle system based on consciousness state"""
        # Remove old particles
        self.params.particles = [p for p in self.params.particles if p['age'] < p['max_age']]
        
        # Add new particles based on trifecta weights
        if len(self.params.particles) < self.params.max_particles:
            # Generate particles for each consciousness component
            weights = self.params.trifecta_weights
            
            # Red particles (Perception)
            if weights['Red'] > 0.5:
                for _ in range(int(weights['Red'] * 3)):
                    self.add_particle('Red', weights['Red'])
            
            # Blue particles (Cognition) 
            if weights['Blue'] > 0.5:
                for _ in range(int(weights['Blue'] * 3)):
                    self.add_particle('Blue', weights['Blue'])
            
            # Yellow particles (Execution)
            if weights['Yellow'] > 0.5:
                for _ in range(int(weights['Yellow'] * 3)):
                    self.add_particle('Yellow', weights['Yellow'])
        
        # Update existing particles
        for particle in self.params.particles:
            # Age particle
            particle['age'] += 1
            
            # Update position
            particle['position'] += particle['velocity']
            
            # Apply consciousness field influence
            dist_from_center = np.linalg.norm(particle['position'])
            if dist_from_center > 0:
                # Gravitational pull toward center (consciousness singularity)
                gravity = -0.001 * particle['position'] / dist_from_center
                particle['velocity'] += gravity
            
            # Boundary conditions
            if dist_from_center > 10:
                particle['velocity'] *= -0.5  # Bounce back
    
    def add_particle(self, color_type: str, intensity: float):
        """Add a new particle of specified type"""
        # Color mapping
        color_map = {
            'Red': (1.0, 0.2, 0.2, 0.8),
            'Blue': (0.2, 0.2, 1.0, 0.8), 
            'Yellow': (1.0, 1.0, 0.2, 0.8)
        }
        
        # Generate position in sphere around origin
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.5, 2.0)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Velocity based on consciousness type and intensity
        vel_scale = intensity * 0.01
        velocity = np.array([
            np.random.uniform(-vel_scale, vel_scale),
            np.random.uniform(-vel_scale, vel_scale), 
            np.random.uniform(-vel_scale, vel_scale)
        ])
        
        particle = {
            'position': np.array([x, y, z], dtype=np.float32),
            'velocity': velocity,
            'color': color_map.get(color_type, (1.0, 1.0, 1.0, 0.8)),
            'type': color_type,
            'intensity': intensity,
            'age': 0,
            'max_age': int(100 + intensity * 50)
        }
        
        self.params.particles.append(particle)
    
    def draw_particles(self):
        """Draw all particles"""
        glPointSize(self.params.particle_size)
        glBegin(GL_POINTS)
        for particle in self.params.particles:
            glColor4f(*particle['color'])
            glVertex3f(*particle['position'])
        glEnd()
    
    def draw_consciousness_center(self):
        """Draw the central consciousness singularity (AE=C=1)"""
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glPushMatrix()
        glTranslatef(0, 0, 0)
        
        # Pulsating sphere based on consciousness health
        pulse_scale = 0.3 + (self.params.consciousness_health * 0.2)
        glScalef(pulse_scale, pulse_scale, pulse_scale)
        
        quad = gluNewQuadric()
        gluSphere(quad, 1.0, 20, 20)
        glPopMatrix()
    
    def draw_neural_models(self):
        """Draw neural model positions as cosmic entities"""
        if not self.params.show_neural_structures:
            return
            
        for name, seed in self.params.neural_model_seeds.items():
            # Calculate position based on model
            model_num = int(name[2:]) if name != 'nM0' else 0
            
            if model_num == 0:
                # Center singularity
                pos = [0, 0, 0]
                color = [1.0, 1.0, 1.0, 1.0]
            else:
                # Orbital positions
                theta = model_num * (2.0 * math.pi / 6)
                radius = 5.0
                pos = [
                    radius * math.cos(theta),
                    radius * math.sin(theta),
                    (model_num % 2) * 1.0
                ]
                
                # Color based on dominant RBY component
                r, b, y = seed['Red'], seed['Blue'], seed['Yellow']
                total = r + b + y
                color = [r/total, b/total, y/total, 0.8]
            
            # Highlight current model
            if name == self.params.current_neural_model:
                glColor4f(1.0, 1.0, 0.0, 1.0)  # Yellow highlight
                size = 0.4
            else:
                glColor4f(*color)
                size = 0.2
            
            glPushMatrix()
            glTranslatef(*pos)
            quad = gluNewQuadric()
            gluSphere(quad, size, 12, 12)
            glPopMatrix()
    
    def draw_consciousness_health_indicator(self):
        """Draw visual health indicator"""
        if not self.params.show_consciousness_health:
            return
            
        # Health bar in top-right corner
        health = self.params.consciousness_health
        
        # Health color: Red (bad) -> Yellow (ok) -> Green (good)
        if health < 0.5:
            color = [1.0, health * 2, 0.0, 0.8]  # Red to Yellow
        else:
            color = [(1.0 - health) * 2, 1.0, 0.0, 0.8]  # Yellow to Green
        
        # Draw health sphere in upper right of view
        glPushMatrix()
        glTranslatef(8, 6, 0)
        glColor4f(*color)
        quad = gluNewQuadric()
        gluSphere(quad, 0.5 + health * 0.3, 10, 10)
        glPopMatrix()
    
    def draw_bridge_status(self):
        """Draw bridge connection status"""
        if not self.params.show_bridge_status:
            return
            
        # Bridge status indicator in top-left
        if self.params.bridge_connected and self.params.auto_rebuilder_active:
            color = [0.0, 1.0, 0.0, 0.8]  # Green - connected and active
        elif self.params.bridge_connected:
            color = [1.0, 1.0, 0.0, 0.8]  # Yellow - connected but inactive
        else:
            color = [1.0, 0.0, 0.0, 0.8]  # Red - disconnected
        
        glPushMatrix()
        glTranslatef(-8, 6, 0)
        glColor4f(*color)
        quad = gluNewQuadric()
        gluSphere(quad, 0.3, 8, 8)
        glPopMatrix()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
                    self.last_mouse_pos = None
            
            elif event.type == pygame.MOUSEMOTION and self.mouse_dragging:
                if self.last_mouse_pos:
                    dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
                    self.params.camera_angle[0] += dx * 0.5
                    self.params.camera_angle[1] += dy * 0.5
                    self.last_mouse_pos = event.pos
    
    def handle_keydown(self, event):
        """Handle keyboard input"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        
        elif event.key == pygame.K_SPACE:
            self.params.paused = not self.params.paused
        
        # Trifecta weight adjustments
        elif event.key == pygame.K_r:
            self.params.trifecta_weights['Red'] = min(3.0, self.params.trifecta_weights['Red'] + 0.2)
            self.send_parameters_to_bridge()
        elif event.key == pygame.K_b:
            self.params.trifecta_weights['Blue'] = min(3.0, self.params.trifecta_weights['Blue'] + 0.2)
            self.send_parameters_to_bridge()
        elif event.key == pygame.K_y:
            self.params.trifecta_weights['Yellow'] = min(3.0, self.params.trifecta_weights['Yellow'] + 0.2)
            self.send_parameters_to_bridge()
        
        # Neural model cycling
        elif event.key == pygame.K_m:
            models = list(self.params.neural_model_seeds.keys())
            current_idx = models.index(self.params.current_neural_model)
            next_idx = (current_idx + 1) % len(models)
            self.params.current_neural_model = models[next_idx]
            # Apply model's trifecta weights
            self.params.trifecta_weights = self.params.neural_model_seeds[self.params.current_neural_model].copy()
            self.send_parameters_to_bridge()
            print(f"Switched to neural model: {self.params.current_neural_model}")
        
        # Display toggles
        elif event.key == pygame.K_n:
            self.params.show_neural_structures = not self.params.show_neural_structures
        elif event.key == pygame.K_h:
            self.params.show_consciousness_health = not self.params.show_consciousness_health
        elif event.key == pygame.K_s:
            self.params.show_bridge_status = not self.params.show_bridge_status
        elif event.key == pygame.K_p:
            self.params.show_performance_metrics = not self.params.show_performance_metrics
        
        # Particle size
        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
            self.params.particle_size = min(10.0, self.params.particle_size + 0.5)
        elif event.key == pygame.K_MINUS:
            self.params.particle_size = max(0.5, self.params.particle_size - 0.5)
    
    def update(self):
        """Main update loop"""
        if self.params.paused:
            return
            
        self.params.time += 0.016  # ~60 FPS time step
        
        # Update from auto-rebuilder bridge
        self.update_from_bridge()
        
        # Update particle system
        self.update_particles()
        
        # Check for consciousness health issues
        if self.params.consciousness_health < 0.3:
            print(f"Warning: Low consciousness health ({self.params.consciousness_health:.2f})")
    
    def render(self):
        """Main render loop"""
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera transformations
        glTranslatef(0.0, 0.0, -15)
        glRotatef(self.params.camera_angle[1], 1, 0, 0)
        glRotatef(self.params.camera_angle[0], 0, 1, 0)
        
        # Apply space scale (consciousness expansion/compression)
        scale = self.params.space_scale
        glScalef(scale, scale, scale)
        
        # Draw simulation components
        self.draw_consciousness_center()
        self.draw_particles()
        self.draw_neural_models()
        
        # Reset transformations for UI elements
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -15)
        
        # Draw UI indicators
        self.draw_consciousness_health_indicator()
        self.draw_bridge_status()
        
        # Swap buffers
        pygame.display.flip()
    
    def print_controls(self):
        """Print control instructions"""
        print("\\n=== AE Theory Auto-Rebuilder Consciousness Monitor ===")
        print("Controls:")
        print("  R/B/Y - Increase Red/Blue/Yellow consciousness weights")
        print("  M - Cycle through neural models (nM0-nM6)")
        print("  N - Toggle neural structure display")
        print("  H - Toggle consciousness health indicator")
        print("  S - Toggle bridge status indicator")
        print("  P - Toggle performance metrics")
        print("  +/- - Adjust particle size")
        print("  SPACE - Pause/unpause")
        print("  ESC - Exit")
        print("  Mouse - Rotate camera")
        print("\\nBridge Status:", "CONNECTED" if self.params.bridge_connected else "DISCONNECTED")
        print("Auto-Rebuilder:", "ACTIVE" if self.params.auto_rebuilder_active else "INACTIVE")
        print("=====================================\\n")
    
    def run(self):
        """Main simulation loop"""
        self.initialize_ui()
        self.print_controls()
        
        try:
            while self.running:
                self.handle_events()
                self.update()
                self.render()
                self.clock.tick(self.params.fps_target)
                
        except KeyboardInterrupt:
            print("\\nSimulation interrupted by user")
        except Exception as e:
            print(f"\\nSimulation error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.bridge:
            self.bridge.stop()
        pygame.quit()
        print("AE Theory Auto-Rebuilder Monitor closed")

# Entry point
if __name__ == "__main__":
    simulation = AutoRebuilderIntegratedSimulation()
    simulation.run()
