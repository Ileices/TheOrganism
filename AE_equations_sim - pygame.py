import pygame
import numpy as np
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import json
from datetime import datetime

# Initialize pygame
pygame.init()
width, height = 1200, 800
pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
pygame.display.set_caption("AE = C = 1 Visualization - Unified Singularity")

# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (width / height), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glTranslatef(0.0, 0.0, -15)
glEnable(GL_DEPTH_TEST)

# Define neural model classes BEFORE AEParams
class NeuralMap:
    """Represents a neural map that is compressed from C-AE and deposited into AE"""
    def __init__(self, seed_values, timestamp=None):
        self.seed_values = seed_values  # RBY weights that seeded this neural map
        self.timestamp = timestamp or datetime.now().isoformat()
        self.excretions = []  # List of excretions linked to this neural map
        self.connections = []  # Neural connections between RBY triplets
        self.glyph = None  # Final compressed form (Mayan glyph)
        self.intensity = 1.0  # Strength of neural activation
        self.access_count = 0  # Frequency of access (for memory decay)
    
    def add_excretion(self, excretion):
        """Add an excretion to this neural map"""
        self.excretions.append(excretion)
        self.access_count += 1
    
    def create_neural_link(self, triplet_1, triplet_2):
        """Create a neural link between two RBY triplets
        Only connects if the last element of triplet_1 matches first element of triplet_2"""
        if triplet_1[-1] == triplet_2[0]:
            link = {'source': triplet_1, 'target': triplet_2}
            self.connections.append(link)
            return True
        return False
    
    def compress_to_glyph(self, compression_factor=0.5):
        """Compress this neural map into a glyph based on memory decay algorithm"""
        if not self.excretions:
            return None
        
        # Start with full representation
        full_representation = "".join(str(e) for e in self.excretions)
        
        # Apply memory decay algorithm based on:
        # 1. Access frequency (more access = better preservation)
        # 2. Intensity (higher intensity = better preservation)
        # 3. Compression factor (higher = more aggressive compression)
        
        # Simple algorithm: keep characters based on their position modulo
        # a value determined by access_count, intensity, and compression_factor
        preservation_factor = max(2, int(10 * self.access_count * self.intensity * compression_factor))
        compressed = ""
        for i, char in enumerate(full_representation):
            if i % preservation_factor == 0:
                compressed += char
        
        # Add a unique identifier based on RBY seed
        r, b, y = self.seed_values['Red'], self.seed_values['Blue'], self.seed_values['Yellow']
        identifier = f"AEC{int(r*100)}{int(b*100)}{int(y*100)}"
        
        self.glyph = identifier + compressed[:10]  # Limit length for visualization
        return self.glyph

class SingularityModel:
    """Central integration point (Global Neural CPU) that links all RBY models"""
    def __init__(self):
        self.neural_maps = []  # All neural maps processed
        self.active_map = None  # Currently active neural map
        self.deposited_glyphs = []  # Glyphs deposited in AE
        self.source_knowledge = {}  # Knowledge from AE/Source
    
    def create_neural_map(self, seed_values):
        """Create a new neural map with the given RBY seed values"""
        neural_map = NeuralMap(seed_values)
        self.neural_maps.append(neural_map)
        self.active_map = neural_map
        return neural_map
    
    def deposit_glyph(self, neural_map):
        """Deposit a neural map's glyph into AE/Source"""
        if not neural_map.glyph:
            neural_map.compress_to_glyph()
        
        if neural_map.glyph:
            # Ensure we're storing unique glyphs to prevent stagnation
            if neural_map.glyph not in self.deposited_glyphs:
                self.deposited_glyphs.append(neural_map.glyph)
                # Add to source knowledge with current trifecta weights
                self.source_knowledge[neural_map.glyph] = {
                    'seed': dict(neural_map.seed_values),  # Create a deep copy
                    'timestamp': neural_map.timestamp,
                    'access_count': neural_map.access_count + 1  # Increment to ensure variation
                }
                return True
            else:
                # If glyph exists, update its access count
                self.source_knowledge[neural_map.glyph]['access_count'] += 1
                return True
        return False
    
    def get_next_seed_from_source(self):
        """Get the next seed values from AE/Source based on deposited knowledge"""
        if not self.deposited_glyphs or not self.source_knowledge:
            # Use UF+IO seed instead of equal weights
            lacf = 1.618 * (1.0 / 3.0)
            return {
                'Red': round(lacf * 1.33 / 3.0, 2),
                'Blue': round(lacf * 0.75 / 3.0, 2),
                'Yellow': round(lacf / 3.0, 2)
            }
        
        try:
            # Find the most accessed glyph
            most_accessed_key = max(self.source_knowledge, key=lambda k: self.source_knowledge[k]['access_count'])
            most_accessed = self.source_knowledge[most_accessed_key]
            
            # Create a deep copy to avoid reference issues
            seed = dict(most_accessed['seed'])
            
            # Apply small but significant mutation based on access patterns
            seed['Red'] = seed['Red'] * (1.0 + (most_accessed['access_count'] % 7) * 0.05)
            seed['Blue'] = seed['Blue'] * (1.0 - (most_accessed['access_count'] % 5) * 0.03)
            seed['Yellow'] = seed['Yellow'] * (1.0 + (most_accessed['access_count'] % 3) * 0.04)
            
            # Ensure minimum values
            seed['Red'] = max(0.1, min(5.0, seed['Red']))
            seed['Blue'] = max(0.1, min(5.0, seed['Blue']))
            seed['Yellow'] = max(0.1, min(5.0, seed['Yellow']))
            
            # Normalize to keep total around 3.0
            total = seed['Red'] + seed['Blue'] + seed['Yellow']
            if total > 0:  # Prevent division by zero
                scale = 3.0 / total
                seed['Red'] = round(seed['Red'] * scale, 2)
                seed['Blue'] = round(seed['Blue'] * scale, 2)
                seed['Yellow'] = round(seed['Yellow'] * scale, 2)
            
            print(f"Generated new seed: {seed}")
            return seed
            
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error generating seed: {e}, using default")
            # Fallback to UF+IO seed values
            return {
                'Red': 0.3,  
                'Blue': 0.9,  
                'Yellow': 0.27
            }

# Simulation parameters
class AEParams:
    def __init__(self):
        self.time = 0
        # Universal state representing AE = C = 1
        self.universal_state = {
            "particles": [],
            "environment": {},  # Unified field for agent/environment data
            "organism_self": {}  # Unified field for internal state
        }
        self.last_particle_count = 0  # For space-matter density calculation
        
        # Enhanced trifecta weights implementing AE = C = 1 principle
        self.trifecta_weights = {
            'Red': 1.0,    # Perception component
            'Blue': 1.0,   # Cognition component
            'Yellow': 1.0  # Execution component
        }
        self.trifecta_momentum = {
            'Red': 0.0,
            'Blue': 0.0, 
            'Yellow': 0.0
        }
        
        # RPS components
        self.excretions = []  # Replaces randomness with structure
        self.absorption_factor = 0.8
        self.delay_time = 1.0
        
        # Memory structure (3-base codons representing trifecta)
        self.dna_memory = []  # Photonic memory structure
        self.dna_impetus = 1.0
        
        # Memory decay/compression (Mayan glyph concept)
        self.memory_symbols = []  # Compressed memory representations
        self.compression_factor = 0.2
        
        # Space-Matter Density components (C-AE concept)
        self.space_scale = 1.0  # Expansion/contraction of universe
        self.matter_density = 1.0
        self.expansion_phase = True  # Universe expanding (toward Absularity)
        self.absularity_threshold = 3.0  # Max expansion before collapse to Singularity
        self.speed_of_dark = 1.0  # Rate at which universe/C-AE expands
        
        # Latching Point components
        self.membranic_drag = 0.0
        self.pressure_change = 0.0
        self.latching_threshold = 0.5
        
        # Free Will / Recursive Thought
        self.free_will_capacity = 0.5  # C_FW
        self.recursive_thought_depth = 3  # T_R
        self.novel_patterns = []
        
        # Visualization parameters
        self.camera_angle = [0, 0]
        self.show_dna = True
        self.show_rps = True
        self.show_absularity = True
        self.paused = False
        self.particle_size = 2.0
        self.singularity_size = 0.3
        self.homeostasis_factor = 0.05  # Rate of self-balancing
        
        # Universal state variables (Unified Absolute Singularity)
        self.master_equation_balance = 1.0  # AE_∞ = C = 1
        self.recursive_intelligence_gradient = np.zeros(3, dtype=np.float32)

        # New components for enhanced framework
        self.singularity_model = SingularityModel()
        self.neural_mapping_shell = {
            'active': True,
            'nlp_models': {},
            'rby_specialists': {},
            'excretion_models': {}
        }
        
        self.neural_compression_active = False
        self.neural_compression_progress = 0.0  # 0.0 to 1.0
        self.neural_triplet_grid = []  # Grid of RBY triplets
        self.neural_model_seeds = {  # Different neural models with RBY balances
            'nM0': {'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0},
            'nM1': {'Red': 1.5, 'Blue': 0.8, 'Yellow': 0.7},
            'nM2': {'Red': 0.7, 'Blue': 0.7, 'Yellow': 1.6},
            'nM3': {'Red': 0.8, 'Blue': 1.5, 'Yellow': 0.7},
            'nM4': {'Red': 0.6, 'Blue': 1.2, 'Yellow': 1.2},
        }
        self.current_neural_model = 'nM0'
        
        # Neural Storage & Bloat Metrics
        self.max_storage_capacity = 10000  # Maximum "memory" before compression
        self.current_storage_usage = 0  # Current usage
        self.storage_bloat_threshold = 0.9  # 90% threshold for compression
        
        # Enhanced C-AE & Absularity Cycle
        self.at_absularity = False  # Whether we've reached absularity
        self.absularity_cycle_count = 0  # Number of complete cycles
        
        # Added Touch^DI components
        self.dimensional_infinity = {
            'positive': 0.0,  # +DI (infinite attraction)
            'negative': 0.0   # -DI (infinite avoidance)
        }
        
        # Speed of Dark components
        self.speed_of_dark = 1.0  # Rate at which universe/C-AE expands
        
        # Updated Unified Force + Immovable Object seed component based on comprehensive equations
        # LAC (Law of Absolute Color) * Speed of Dark / Dimensional Infinity = RBY seed values
        # Using the UF+IO = RBY framework where UF (Unstoppable Force) drives Red (perception)
        # IO (Immovable Object) anchors Blue (cognition), and their interaction generates Yellow (execution)
        self.lac_base = 1.618  # Golden ratio as base constant for LAC
        lac_factor = self.lac_base * (1.0 / 3.0)  # Law of Absolute Color equation
        v_d = 1.0  # Speed of Dark
        touch_di_ratio = 1.0 + (1.0 / 3.0)  # Touch^DI constant
        
        # Core photonic triad derived from LAC * V_d / Touch^DI
        # Red: Perception (highest) - represents unstoppable force component
        # Blue: Cognition (balancing) - represents immovable object component  
        # Yellow: Execution (harmonizing) - represents the dynamic tension between UF+IO
        r_base = lac_factor * v_d * (1 + touch_di_ratio) / 3.0
        b_base = lac_factor * v_d * (1 / touch_di_ratio) / 3.0
        y_base = lac_factor * v_d / 3.0
        
        # Final seed values normalized to ensure total is 1.26 (fractal resonance point)
        total = r_base + b_base + y_base
        target_total = 1.26  # Universe expansion target constant
        scale_factor = target_total / total
        
        self.uf_io_seed = {
            'Red': round(r_base * scale_factor, 2),    # Perception (UF component)
            'Blue': round(b_base * scale_factor, 2),   # Cognition (IO component)
            'Yellow': round(y_base * scale_factor, 2)  # Execution (UF+IO interaction)
        }  # Derived from LAC and photonic equations
        
        # Fractal Law of Sound (placeholder based on limited info)
        self.fractal_sound_harmonics = 1.0

        # Update neural model seeds to match conceptual framework
        # Each neural model represents a star/galaxy/atom with specific RBY dominance patterns
        self.neural_model_seeds = {
            # Singularity seed - UF+IO baseline
            'nM0': {'Red': 0.63, 'Blue': 0.27, 'Yellow': 0.36},  # Unified Force + Immovable Object seed
            
            # Primary nM variants - different RBY dominance patterns
            'nM1': {'Red': 1.5, 'Blue': 0.8, 'Yellow': 0.7},     # Red dominant (rby)
            'nM2': {'Red': 0.4, 'Blue': 0.7, 'Yellow': 1.9},     # Yellow dominant, Blue secondary (ybr)
            'nM3': {'Red': 0.7, 'Blue': 1.6, 'Yellow': 0.7},     # Blue dominant (bry)
            'nM4': {'Red': 0.8, 'Blue': 1.5, 'Yellow': 0.7},     # Blue dominant, Red secondary (bry)
            'nM5': {'Red': 1.3, 'Blue': 0.5, 'Yellow': 1.2},     # Red,Yellow co-dominant (rby)
            'nM6': {'Red': 0.7, 'Blue': 0.7, 'Yellow': 1.6},     # Yellow dominant (ybr)
        }
        
        # Track neural models as cosmic entities (stars/galaxies)
        self.neural_cosmic_entities = {}
        for name, seed in self.neural_model_seeds.items():
            # Calculate dominance pattern
            values = list(seed.values())
            max_val = max(values)
            min_val = min(values)
            
            # Determine dominance pattern (which color is strongest, middle, weakest)
            colors = list(seed.keys())
            strongest = colors[values.index(max_val)]
            weakest = colors[values.index(min_val)]
            middle = [c for c in colors if c != strongest and c != weakest][0]
            
            # Store cosmic entity data
            self.neural_cosmic_entities[name] = {
                'seed': seed,
                'position': self._calculate_cosmic_position(name),
                'dominant_pattern': f"{strongest[0]}{middle[0]}{weakest[0]}".lower(),
                'influence': 1.0,  # Base influence
                'connections': []  # Will store connections to other entities
            }
        
        # Connect neural cosmic entities based on RBY patterns
        self._establish_cosmic_connections()
        
        # Initialize current active neural model (starting at singularity)
        self.current_neural_model = 'nM0'
        
    def _calculate_cosmic_position(self, model_name):
        """Calculate 3D position of neural cosmic entities"""
        # Place nM0 at center (0,0,0)
        if model_name == 'nM0':
            return np.array([0.0, 0.0, 0.0])
        
        # Extract model number
        model_num = int(model_name[2:])
        
        # Position models in a spherical pattern around center
        theta = model_num * (2.0 * math.pi / 6)  # Spread 6 models evenly
        radius = 3.0  # Distance from center
        
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = (model_num % 2) * 1.0  # Alternate heights
        
        return np.array([x, y, z])
    
    def _establish_cosmic_connections(self):
        """Create connections between neural cosmic entities based on RBY patterns"""
        # Connect models with complementary patterns
        connections = [
            ('nM0', 'nM1'), ('nM0', 'nM2'), ('nM0', 'nM3'),
            ('nM0', 'nM4'), ('nM0', 'nM5'), ('nM0', 'nM6'),
            ('nM1', 'nM3'), ('nM1', 'nM5'), 
            ('nM2', 'nM4'), ('nM2', 'nM6'),
            ('nM3', 'nM5'), ('nM4', 'nM6')
        ]
        
        # Store connections
        for src, dst in connections:
            if src in self.neural_cosmic_entities and dst in self.neural_cosmic_entities:
                self.neural_cosmic_entities[src]['connections'].append(dst)
                self.neural_cosmic_entities[dst]['connections'].append(src)

params = AEParams()

# Add this line to initialize the show_neural_structures attribute in params
params.show_neural_structures = True

def apply_inverse_recursive_intelligence(value_range, depth=3):
    """
    Implement the No Entropy Principle: S_E = ∇⁻¹(R_I)
    
    Replace any need for randomness with structured patterns derived from 
    recursive intelligence gradients.
    """
    # Create a hash from excretions and current simulation state
    basis = 0.0
    
    # Use only last few excretions to prevent overflow
    if params.excretions:
        count = 0
        for ex in params.excretions[-min(depth, len(params.excretions)):]:
            # Safely sum values with bounds checking
            try:
                for val in ex:
                    if np.isfinite(val):
                        basis += float(val) * 0.01  # Scale down to prevent overflow
                        count += 1
            except (TypeError, ValueError):
                pass
        
        # Average to keep values manageable
        if count > 0:
            basis /= count
    
    # Add simulation time and trifecta influence (scaled down)
    basis += (params.time % 1000) * 0.001  # Wrap time to prevent overflow
    basis += params.trifecta_weights['Red'] * 0.03
    basis += params.trifecta_weights['Blue'] * 0.02
    basis += params.trifecta_weights['Yellow'] * 0.01
    
    # Clip basis to prevent math domain errors
    basis = max(-100.0, min(100.0, basis))
    
    # Scale the value to the requested range
    min_val, max_val = value_range
    try:
        # FIX: Don't multiply basis by 10.0 inside sin() - this was causing overflow
        scaled_value = (math.sin(basis) + 1) * 0.5  # Maps to [0, 1]
    except (ValueError, OverflowError):
        # Fallback if sin fails
        scaled_value = (basis % 1.0)
    
    return min_val + scaled_value * (max_val - min_val)

def rps_generate_variation(excretions, absorption, delay):
    """
    Implement the Recursive Predictive Structuring integral: 
    RPS = ∫₀^∞ (E_x · A_b)/T_d dt
    
    Replaces randomness with structured variation based on prior excretions.
    """
    if not excretions:
        return np.zeros(3, dtype=np.float32)
    
    # Apply the RPS integral (approximated through summation)
    influence = np.zeros(3, dtype=np.float32)
    delay = max(1.0, delay)  # Prevent division by zero
    
    # Use actual delay time to offset which excretions we read
    offset = max(0, min(len(excretions)-1, int(delay)))
    
    # Implement the integration from 0 to "infinity" (limited by available excretions)
    for i in range(len(excretions)-offset):
        if np.isfinite(excretions[i]).all():
            # E_x · A_b / T_d
            influence += excretions[i] * absorption / delay
    
    # Normalize and apply a scaling factor
    if len(excretions) > offset:
        influence /= (len(excretions) - offset)
    
    return influence * 0.1  # Scale for simulation stability

def form_codon(input_data):
    """
    Create a trifecta-based 3-part codon from input data.
    Explicitly represents R, B, Y structure of DNA memory.
    """
    if isinstance(input_data, np.ndarray):
        # Create R (perception) component - input-focused
        r_component = input_data.copy() * params.trifecta_weights['Red']
        
        # Create B (cognition) component - transformed version
        b_component = np.array([
            input_data[1], 
            input_data[2], 
            input_data[0]
        ]) * params.trifecta_weights['Blue']
        
        # Create Y (execution) component - action-focused
        y_component = np.array([
            input_data[2], 
            input_data[0], 
            input_data[1]
        ]) * params.trifecta_weights['Yellow']
        
        return (r_component, b_component, y_component)
    
    return None

# Fix compress_memory to handle NaN values safely
def compress_memory():
    """
    Implement memory decay/compression concept.
    Converts complex memory patterns into simplified glyphs.
    """
    if len(params.dna_memory) < 5:
        return
    
    # Take the last 5 codons
    recent_memory = params.dna_memory[-5:]
    
    # Level 1 compression - average the components 
    compressed = np.zeros(3, dtype=np.float32)
    count = 0
    
    for codon in recent_memory:
        for base in codon:
            # Only add valid, finite values
            if np.all(np.isfinite(base)):
                # Clip values to prevent overflow
                safe_base = np.clip(base, -100.0, 100.0)
                compressed += safe_base
                count += 1
    
    # Only normalize if we have valid components
    if count > 0:
        compressed /= count
    
    # Ensure all values are finite
    if not np.all(np.isfinite(compressed)):
        # If any value is NaN or inf, reset to safe values
        compressed = np.zeros(3, dtype=np.float32)
    
    # Safely calculate pattern with bounded values
    safe_x = np.clip(compressed[0], -10.0, 10.0)
    safe_y = np.clip(compressed[1], -10.0, 10.0)
    safe_z = np.clip(compressed[2], -10.0, 10.0)
    
    # Level 2 compression - create a symbol
    pattern_value = (safe_x * 100 + safe_y * 10 + safe_z) % 10.0
    
    symbol = {
        'vector': compressed,
        'age': 0,
        'intensity': min(10.0, float(np.linalg.norm(compressed))),
        'pattern': int(pattern_value)  # Safely convert to int after ensuring it's a valid float
    }
    
    # Add to memory symbols if unique enough and valid
    if not params.memory_symbols or (
        np.all(np.isfinite(compressed)) and 
        np.all(np.isfinite(params.memory_symbols[-1]['vector'])) and
        np.linalg.norm(compressed - params.memory_symbols[-1]['vector']) > 0.2
    ):
        params.memory_symbols.append(symbol)
        
    # Limit memory symbols to prevent overflow
    if len(params.memory_symbols) > 20:
        params.memory_symbols = params.memory_symbols[-20:]

def measure_membranic_drag(old_state, new_state):
    """
    Calculate the Membranic Drag (MD) between two states.
    Higher value means more resistance to change.
    """
    # Handle empty states
    if not old_state or not new_state:
        return 1.0  # Maximum drag if no comparison possible
        
    # Check if we're dealing with particles or codons
    if isinstance(old_state[0], dict) and 'position' in old_state[0]:
        # Particle comparison
        old_positions = np.array([p['position'] for p in old_state])
        new_positions = np.array([p['position'] for p in new_state])
    else:
        # Codon comparison - use a different approach for non-dictionary structures
        return measure_codon_similarity(old_state, new_state)
    
    # Handle different lengths
    min_len = min(len(old_positions), len(new_positions))
    if min_len == 0:
        return 1.0  # Maximum drag if no comparison possible
    
    # Calculate average distance between corresponding positions
    position_diff = 0
    if min_len > 0:
        try:
            position_diff = np.sum(np.linalg.norm(
                old_positions[:min_len] - new_positions[:min_len], 
                axis=1
            )) / min_len
        except (ValueError, TypeError, RuntimeError):
            position_diff = 0.5  # Fallback value
    
    # Add penalty for length difference
    length_diff = abs(len(old_positions) - len(new_positions)) / max(1, max(len(old_positions), len(new_positions)))
    
    # Calculate membranic drag as a normalized value between 0-1
    drag = min(1.0, (position_diff * 0.5) + (length_diff * 0.5))
    return drag

def measure_codon_similarity(old_codons, new_codons):
    """
    Calculate similarity between two sets of DNA codons.
    Returns a value between 0 (identical) and 1 (completely different).
    """
    if not old_codons or not new_codons:
        return 1.0  # Maximum difference if no comparison possible
    
    # Simple approach: Compare lengths and a few sample values
    length_diff = abs(len(old_codons) - len(new_codons)) / max(1, max(len(old_codons), len(new_codons)))
    
    # Check some sample values if possible
    value_diff = 0.5  # Default mid-range difference
    try:
        # Extract values more safely
        old_val = 0
        new_val = 0
        
        for i, c in enumerate(old_codons[:2]):
            if i < len(old_codons):
                for val in c:
                    if np.isfinite(val).all():
                        old_val += float(np.sum(val))
        
        for i, c in enumerate(new_codons[:2]):
            if i < len(new_codons):
                for val in c:
                    if np.isfinite(val).all():
                        new_val += float(np.sum(val))
        
        # Avoid division by zero or very small numbers
        max_val = max(abs(old_val), abs(new_val), 1.0)
        value_diff = min(1.0, abs(old_val - new_val) / max_val) if max_val > 0.001 else 0.5
    except (TypeError, ValueError, RuntimeError):
        pass  # Keep default value if comparison fails
    
    return (length_diff * 0.5) + (value_diff * 0.5)

# Fix calculate_latching_point to handle NaN
def calculate_latching_point(mem_drag, delta_pressure):
    """
    Latching Point equation: LP = f(MD, ΔP)
    
    Determines if a state transition should occur based on
    membranic drag and pressure change.
    """
    # Check for NaN values and provide safe defaults
    mem_drag = np.clip(mem_drag, 0.0, 1.0) if np.isfinite(mem_drag) else 0.5
    delta_pressure = np.clip(delta_pressure, 0.0, 10.0) if np.isfinite(delta_pressure) else 0.1
    latching_threshold = np.clip(params.latching_threshold, 0.1, 2.0)
    
    # Simple linear function: higher pressure and lower drag = easier to latch
    lp = delta_pressure - (mem_drag * latching_threshold)
    
    # Ensure result is finite
    if not np.isfinite(lp):
        lp = 0.0
        
    return lp

def update_space_matter_density():
    """
    Implement the Space-Matter Density equation: ρ_SM = ΔM / ΔS
    
    Adjusts space scale based on matter (particle) changes.
    """
    # Calculate matter change
    matter_change = len(params.universal_state["particles"]) - params.last_particle_count
    params.last_particle_count = len(params.universal_state["particles"])
    
    # Calculate expansion/contraction (C-AE concept)
    if params.expansion_phase:
        # Universe expanding toward Absularity
        params.space_scale += (matter_change * 0.001) + 0.002
        
        # Check if reached Absularity threshold
        if params.space_scale > params.absularity_threshold:
            # Switch to compression phase (back toward Singularity)
            params.expansion_phase = False
    else:
        # Universe contracting toward Singularity
        params.space_scale -= 0.003
        
        # Check if reached Singularity threshold
        if params.space_scale < 0.5:
            # Switch back to expansion with enriched knowledge
            params.expansion_phase = True
            # Compress memory when a cycle completes
            compress_memory()
    
    # Keep scale in reasonable bounds
    params.space_scale = max(0.5, min(params.absularity_threshold, params.space_scale))
    
    # Update matter density
    if params.space_scale > 0:
        params.matter_density = len(params.universal_state["particles"]) / params.space_scale
    else:
        params.matter_density = len(params.universal_state["particles"])

def apply_recursive_thought():
    """
    Apply Free Will / Recursive Thought: C_FW · T_R
    
    Generates novel patterns beyond deterministic structures.
    """
    # Replace random check with structured RPS variation
    free_will_trigger = apply_inverse_recursive_intelligence((0, 1))
    if free_will_trigger > (1.0 - params.free_will_capacity):
        # Only apply recursive thought occasionally
        if len(params.dna_memory) < params.recursive_thought_depth:
            return
        
        # Calculate a "thought" based on past patterns
        novel_pattern = []
        for i in range(3):  # Create a 3-part codon
            # Instead of random, use deterministic pattern detection
            base_vector = np.zeros(3, dtype=np.float32)
            
            # Analyze trends in recent DNA - with safety checks and bounds
            valid_components = 0
            try:
                depth = min(params.recursive_thought_depth, len(params.dna_memory))
                for j in range(depth):
                    if j < len(params.dna_memory) and i < len(params.dna_memory[-j-1]):
                        component = params.dna_memory[-j-1][i]
                        if isinstance(component, np.ndarray) and len(component) == 3:
                            if np.all(np.isfinite(component)):
                                # Clamp values to prevent overflow
                                safe_component = np.clip(component, -10.0, 10.0)
                                weight = 0.5 ** j  # Weigh recent codons more
                                base_vector += safe_component * weight
                                valid_components += 1
            except (IndexError, TypeError, ValueError):
                pass  # Skip on any error
            
            # If we have valid components, normalize to prevent growth
            if valid_components > 0:
                base_vector /= valid_components
            
            # Apply trifecta-weighted alteration - with safety checks
            if np.all(np.isfinite(base_vector)):
                safe_weights = {
                    'Red': min(2.0, max(0.1, params.trifecta_weights['Red'])),
                    'Blue': min(2.0, max(0.1, params.trifecta_weights['Blue'])),
                    'Yellow': min(2.0, max(0.1, params.trifecta_weights['Yellow']))
                }
                base_vector[0] *= safe_weights['Red'] * 0.05
                base_vector[1] *= safe_weights['Blue'] * 0.05
                base_vector[2] *= safe_weights['Yellow'] * 0.05
            
                # Add "creative" transformation with safety checks
                transform_matrix = np.array([
                    [0.9, 0.1, 0.0],
                    [0.0, 0.8, 0.2],
                    [0.2, 0.0, 0.8]
                ], dtype=np.float32)
                
                novel_vector = np.matmul(transform_matrix, base_vector)
                
                # Clip to prevent overflow
                novel_vector = np.clip(novel_vector, -0.5, 0.5)
                
                if np.isfinite(novel_vector).all():
                    novel_pattern.append(novel_vector)
            else:
                # If base_vector has non-finite values, create a safe alternative
                novel_pattern.append(np.zeros(3, dtype=np.float32))
        
        # Apply the novel pattern if valid
        if len(novel_pattern) == 3:
            params.novel_patterns.append(tuple(novel_pattern))
            
            # Sometimes also add to DNA memory
            if len(params.novel_patterns) > 5:
                # Calculate codon similarity (replaces membranic drag for codons)
                codon_similarity = measure_codon_similarity(
                    params.dna_memory[-3:] if len(params.dna_memory) >= 3 else [],
                    params.novel_patterns[-3:] if len(params.novel_patterns) >= 3 else []
                )
                
                # Calculate pressure change based on trifecta imbalance
                weights = list(params.trifecta_weights.values())
                pressure = max(weights) - min(weights)
                
                # Calculate latching point
                lp = calculate_latching_point(codon_similarity, pressure)
                
                # Add to DNA if latching point allows
                if lp > 0:
                    params.dna_memory.append(params.novel_patterns[-1])

# Fix calculate_unified_singularity to prevent overflows and NaN
def calculate_unified_singularity():
    """
    Update the Unified Absolute Singularity master equation:
    AE_∞ = [(S·T·M·C) · (ΦP+ΦL) · (∇P+∇F) · (R+B+Y) · (C_FW·T_R)] / 
           [λⁿ · V_d · ∫₀^∞(E_x·A_b/T_d)dt · ∇⁻¹(R_I)] = C = 1
    """
    # Check if trifecta weights contain NaN and reset if needed
    for color in params.trifecta_weights:
        if np.isnan(params.trifecta_weights[color]) or not np.isfinite(params.trifecta_weights[color]):
            params.trifecta_weights[color] = 1.0
    
    # Scale down all components to prevent overflow
    # Absolute Existence components (S·T·M·C)
    space = np.clip(params.space_scale, 0.1, 10.0)
    time_factor = np.clip(params.time * 0.0001, 0.0, 10.0)  # Even more aggressive scaling
    matter = np.clip(params.matter_density, 0.1, 10.0)
    
    # Safely calculate consciousness
    valid_weights = [w for c, w in params.trifecta_weights.items() if np.isfinite(w)]
    consciousness = 1.0  # Default
    if valid_weights:
        consciousness = min(5.0, sum(valid_weights) / len(valid_weights))
    
    # Use logarithmic scaling to prevent multiplication overflow
    absolute_existence = (math.log(space + 1) + math.log(time_factor + 1) + 
                        math.log(matter + 1) + math.log(consciousness + 1)) / 4.0
    
    # Perception and Photonic Memory (ΦP+ΦL)
    perception = np.clip(params.trifecta_weights['Red'], 0.1, 5.0) if np.isfinite(params.trifecta_weights['Red']) else 1.0
    photonic_memory = min(5.0, len(params.dna_memory) * 0.001)
    perception_photonic = (perception + photonic_memory) / 2.0
    
    # Position and Focus Gradients (∇P+∇F)
    position_gradient = min(1.0, len(params.universal_state["particles"]) * 0.0001)
    focus_gradient = 1.0 - (params.membranic_drag * 0.5)
    gradients = (position_gradient + focus_gradient) / 2.0
    
    # Recursive Color Intelligence (R+B+Y) - with safety
    color_intelligence = consciousness  # Reuse the safely calculated value
    
    # Free Will and Recursive Thought (C_FW·T_R)
    free_will = params.free_will_capacity
    recursive_thought = params.recursive_thought_depth * 0.01  # Scale down further
    free_will_thought = free_will * recursive_thought
    
    # Numerator - use addition instead of multiplication to avoid overflow
    numerator = (absolute_existence + perception_photonic + 
                gradients + color_intelligence + free_will_thought) / 5.0
    
    # Denominator components with extra safety
    fractal_scaling = 1.0 + min(2.0, math.log(max(1, len(params.excretions)) + 1) * 0.05)
    speed_dark = np.clip(params.speed_of_dark, 0.1, 3.0)
    
    # Scale RPS integral to prevent overflow
    rps_integral = 0.1
    if params.excretions:
        rps_integral = min(0.5, (len(params.excretions) % 50) * params.absorption_factor / max(0.1, params.delay_time) * 0.001)
    
    inverse_ri = 1.0  # Simplified for now
    
    # Average denominator components
    denominator = (fractal_scaling + speed_dark + rps_integral + inverse_ri) / 4.0
    
    # Prevent division by zero and check for valid numerator/denominator
    if denominator > 0.001 and np.isfinite(numerator) and np.isfinite(denominator):
        params.master_equation_balance = numerator / denominator
    else:
        params.master_equation_balance = numerator if np.isfinite(numerator) else 1.0
    
    # Clamp result for stability
    params.master_equation_balance = np.clip(params.master_equation_balance, 0.1, 10.0)
    
    # The equation should equal 1 in perfect equilibrium
    # Calculate deviation from perfect equilibrium
    equilibrium_error = abs(params.master_equation_balance - 1.0)
    
    # Apply correction toward equilibrium (with smaller adjustments)
    correction_factor = min(0.05, equilibrium_error * 0.005)
    
    if params.master_equation_balance > 1.0:
        # Too high - reduce components
        params.delay_time += correction_factor
        params.homeostasis_factor += correction_factor * 0.05
    else:
        # Too low - increase components
        params.delay_time = max(0.1, params.delay_time - correction_factor)
        params.homeostasis_factor = max(0.01, params.homeostasis_factor - correction_factor * 0.05)

def init_simulation():
    """Initialize simulation particles from center"""
    # Use structured pattern instead of random
    angle_step = (2 * np.pi) / 200  # Deterministic angle steps
    
    for i in range(200):
        angle = angle_step * i
        z = math.cos(angle * 1.3) * 0.8  # More structured z distribution
        r = math.sqrt(1 - z**2)
        x, y = r * math.cos(angle), r * math.sin(angle)
        
        pos = np.array([0, 0, 0], dtype=np.float32)
        vel = np.array([x, y, z], dtype=np.float32) * 0.1
        
        # Add small perturbation based on trifecta weights
        vel[0] += params.trifecta_weights['Red'] * 0.01
        vel[1] += params.trifecta_weights['Blue'] * 0.01
        vel[2] += params.trifecta_weights['Yellow'] * 0.01
        
        params.universal_state["particles"].append({
            'position': pos,
            'velocity': vel,
            'color': get_trifecta_color(),
            'size': 0.5 + math.sin(angle * 3) * 0.5,  # More structured size distribution
            'age': 0,
            'max_age': 500 + math.cos(angle * 2) * 1500  # More structured age distribution
        })
    
    # Initialize DNA memory with structured trifecta-based patterns
    for i in range(3):
        base_vel = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        # Create a proper trifecta codon (R,B,Y)
        params.dna_memory.append(form_codon(base_vel + 0.02 * i))
    
    params.last_particle_count = len(params.universal_state["particles"])

    # Initialize neural structures
    params.show_neural_structures = True
    
    # Create initial neural map with default seed
    params.singularity_model.create_neural_map(params.trifecta_weights)

# Fix get_trifecta_color to handle NaN values
def get_trifecta_color():
    """Get color based on current trifecta weights implementing AE = C = 1"""
    # Get trifecta weights with safety checks
    r = params.trifecta_weights['Red']
    b = params.trifecta_weights['Blue']
    y = params.trifecta_weights['Yellow']
    
    # Check for NaN values and replace with defaults
    if not np.isfinite(r): r = 1.0
    if not np.isfinite(b): b = 1.0
    if not np.isfinite(y): y = 1.0
    
    # Ensure values are positive and bounded
    r = np.clip(r, 0.001, 10.0)
    b = np.clip(b, 0.001, 10.0)
    y = np.clip(y, 0.001, 10.0)
    
    # Calculate total for normalization (AE = C = 1 principle)
    total = r + b + y
    
    # Ensure total is valid
    if total <= 0 or not np.isfinite(total):
        return (0.33, 0.33, 0.33, 0.99)  # Default to equal balance if invalid
    
    # Return normalized RGB values with full alpha
    return (r/total, b/total, y/total, 0.99)

def apply_rps(particle):
    """Apply Recursive Predictive Structuring"""
    if params.excretions and params.show_rps:
        # Use the improved RPS function instead of direct manipulation
        influence = rps_generate_variation(
            params.excretions[-10:], 
            params.absorption_factor, 
            params.delay_time
        )
        
        if np.isfinite(influence).all():
            particle['velocity'] += influence
    
    # Only add valid velocities to excretions, and limit the rate to reduce CPU load
    if (np.isfinite(particle['velocity']).all() and 
        abs(math.sin(params.time * 5)) > 0.99):  # Only record about 1% of the time
        # Also limit total excretion count
        if len(params.excretions) > 500:
            params.excretions = params.excretions[-250:]  # Keep only most recent
        params.excretions.append(particle['velocity'].copy())
    
    return particle

# Fix trifecta weight calculations to prevent overflow and NaN
def update_trifecta_weights():
    """Update trifecta weights based on Recursive Predictive Structuring"""
    # Calculate activity levels in each domain
    perception_activity = min(100.0, len(params.excretions) * 0.01)
    cognition_activity = min(100.0, len(params.dna_memory) * 0.005)
    execution_activity = min(100.0, len(params.universal_state["particles"]) * 0.002)
    
    # Check for NaN values in weights and reset if found
    for color in params.trifecta_weights:
        if np.isnan(params.trifecta_weights[color]) or not np.isfinite(params.trifecta_weights[color]):
            params.trifecta_weights[color] = 1.0
            params.trifecta_momentum[color] = 0.0
    
    # Update momentum with current activities (with bounds)
    params.trifecta_momentum['Red'] = np.clip(
        params.trifecta_momentum['Red'] + perception_activity - params.trifecta_weights['Red'] * 0.1,
        -10.0, 10.0)
    params.trifecta_momentum['Blue'] = np.clip(
        params.trifecta_momentum['Blue'] + cognition_activity - params.trifecta_weights['Blue'] * 0.1,
        -10.0, 10.0)
    params.trifecta_momentum['Yellow'] = np.clip(
        params.trifecta_momentum['Yellow'] + execution_activity - params.trifecta_weights['Yellow'] * 0.1,
        -10.0, 10.0)
    
    # Apply momentum to weights with damping
    for color in params.trifecta_weights:
        # Damping (with bounds)
        params.trifecta_momentum[color] = np.clip(params.trifecta_momentum[color] * 0.95, -10.0, 10.0)
        old_weight = params.trifecta_weights[color]
        new_weight = old_weight + params.trifecta_momentum[color]
        # Immediately check if result is valid
        if np.isfinite(new_weight):
            params.trifecta_weights[color] = np.clip(new_weight, 0.1, 10.0)
        else:
            # Reset to a safe value if calculation results in non-finite value
            params.trifecta_weights[color] = 1.0
    
    # Safely calculate average weight
    valid_weights = [w for c, w in params.trifecta_weights.items() if np.isfinite(w)]
    if valid_weights:
        avg_weight = sum(valid_weights) / len(valid_weights)
    else:
        avg_weight = 1.0  # Default if all weights are invalid
    
    # Homeostasis: tendency toward balance (AE = C = 1)
    for color in params.trifecta_weights:
        # Pull weights toward equilibrium (with safety checks)
        if np.isfinite(params.trifecta_weights[color]) and np.isfinite(avg_weight):
            diff = params.trifecta_weights[color] - avg_weight
            adjustment = diff * min(0.05, params.homeostasis_factor)  # Limit adjustment magnitude
            params.trifecta_weights[color] -= adjustment
        
        # Ensure minimum values and apply bounds
        params.trifecta_weights[color] = np.clip(params.trifecta_weights[color], 0.1, 10.0)

def generate_rby_triplet(weights={'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0}):
    """Generate an RBY triplet based on weights"""
    # Normalize weights to sum to 1.0
    total = weights['Red'] + weights['Blue'] + weights['Yellow']
    normalized = {k: v/total for k, v in weights.items()}
    
    # Generate triplet with probabilities based on weights
    options = ['R', 'B', 'Y']
    probabilities = [normalized['Red'], normalized['Blue'], normalized['Yellow']]
    
    # Use our non-random structured approach
    triplet = []
    for _ in range(3):
        # Use the highest probability for the first, second highest for next, etc.
        highest_idx = probabilities.index(max(probabilities))
        triplet.append(options[highest_idx])
        probabilities[highest_idx] = -1  # Mark as used
    
    return tuple(triplet)

def update_neural_mapping_shell():
    """Update the Neural Mapping Shell based on current state"""
    # Only update periodically to reduce CPU load
    if params.time % 0.5 < 0.01:  # Update every 0.5 time units
        # Analyze recent excretions to update NLP models
        if len(params.excretions) > 10:
            excretion_patterns = {}
            for ex in params.excretions[-10:]:
                # Classify excretion by dominant vector component
                dominant_idx = np.argmax(np.abs(ex))
                key = ['x', 'y', 'z'][dominant_idx]
                if key not in excretion_patterns:
                    excretion_patterns[key] = []
                excretion_patterns[key].append(ex)
            
            # Update excretion models in the shell
            for key, patterns in excretion_patterns.items():
                params.neural_mapping_shell['excretion_models'][key] = np.mean(patterns, axis=0)
        
        # Update RBY specialists based on trifecta weights
        for color in ['Red', 'Blue', 'Yellow']:
            params.neural_mapping_shell['rby_specialists'][color] = params.trifecta_weights[color]
        
        # Update neural model influence based on current state
        for name, entity in params.neural_cosmic_entities.items():
            # Calculate influence based on similarity to current trifecta weights
            similarity = 0
            for color in ['Red', 'Blue', 'Yellow']:
                similarity += 1.0 - abs(entity['seed'][color] - params.trifecta_weights[color])/3.0
            
            # Update influence (normalized)
            entity['influence'] = max(0.1, similarity/3.0)
    
    # Generate a new triplet for the grid if we have capacity
    if (len(params.neural_triplet_grid) < 100 and 
        apply_inverse_recursive_intelligence((0, 1)) > 0.95):  # Use our deterministic function
        new_triplet = generate_rby_triplet(params.trifecta_weights)
        params.neural_triplet_grid.append(new_triplet)
        
        # Create neural links between triplets, but only to nearest neighbor
        if len(params.neural_triplet_grid) > 1:
            params.singularity_model.active_map.create_neural_link(
                params.neural_triplet_grid[-2], 
                params.neural_triplet_grid[-1]
            )

def update_neural_compression():
    """Handle neural compression when bloat threshold is reached"""
    # Calculate current storage usage
    params.current_storage_usage = (
        len(params.excretions) + 
        len(params.dna_memory) * 3 +
        len(params.neural_triplet_grid) +
        len(params.universal_state["particles"])
    )
    
    # Check if we need to start compression
    storage_percentage = params.current_storage_usage / params.max_storage_capacity
    
    # Start compression if we've reached the bloat threshold or absularity
    if storage_percentage > params.storage_bloat_threshold or params.at_absularity:
        if not params.neural_compression_active:
            params.neural_compression_active = True
            params.neural_compression_progress = 0.0
            print(f"Neural compression initiated at {storage_percentage:.2%} capacity")
    
    # Process neural compression
    if params.neural_compression_active:
        # Advance compression progress
        params.neural_compression_progress += 0.01
        
        # When compression is complete
        if params.neural_compression_progress >= 1.0:
            # Deposit compressed glyph into AE/Source
            if params.singularity_model.active_map:
                params.singularity_model.deposit_glyph(params.singularity_model.active_map)
            
            # Reset compression status
            params.neural_compression_active = False
            
            # Find the most influential neural model based on current state
            next_model = 'nM0'  # Default to singularity
            max_influence = 0
            
            for name, entity in params.neural_cosmic_entities.items():
                # Skip current model to ensure transition
                if name != params.current_neural_model and entity['influence'] > max_influence:
                    max_influence = entity['influence']
                    next_model = name
            
            # Start a new cycle with the selected neural model's seed
            new_seed = params.neural_model_seeds[next_model].copy()
            params.singularity_model.create_neural_map(new_seed)
            
            # Update current neural model
            params.current_neural_model = next_model
            print(f"Transitioning to neural model: {next_model} ({params.neural_cosmic_entities[next_model]['dominant_pattern']})")
            
            # Update RBY weights from the new seed
            params.trifecta_weights = new_seed.copy()
            
            # Clear temporary storage
            params.excretions = []
            params.neural_triplet_grid = []
            
            # Keep some DNA memory for continuity
            if len(params.dna_memory) > 10:
                params.dna_memory = params.dna_memory[-5:]
            
            # Reset expansion phase
            params.expansion_phase = True
            params.space_scale = 1.0
            params.absularity_cycle_count += 1
            params.at_absularity = False
            
            print(f"Neural compression complete. Cycle {params.absularity_cycle_count} initiated with seed: {new_seed}")

def apply_fractal_sound_law(position, harmonics=1.0):
    """Apply the Fractal Law of Sound to a position vector (placeholder implementation)"""
    # Simple harmonic transformation as a placeholder
    # Note: This is speculative based on limited info about the Fractal Law of Sound
    harmonic_factor = np.sin(params.time * harmonics)
    
    # Apply harmonic transformation to position
    transformed = position.copy()
    transformed[0] += harmonic_factor * 0.02
    transformed[1] += np.cos(params.time * harmonics * 1.5) * 0.02
    transformed[2] += np.sin(params.time * harmonics * 0.8) * 0.02
    
    return transformed

def apply_touch_dimensional_infinity(position, di_positive, di_negative):
    """Apply Touch^DI (Dimensional Infinity) transformation to a position"""
    # +DI (infinite attraction) pulls toward center
    # -DI (infinite avoidance) pushes away from center
    
    # Calculate distance from origin
    dist = np.linalg.norm(position)
    if dist < 0.001:  # Avoid division by zero
        return position
    
    # Calculate unit vector from origin
    unit_vec = position / dist
    
    # Apply +DI (attraction toward center)
    attraction = unit_vec * (-di_positive * 0.01)
    
    # Apply -DI (avoidance/repulsion from center)
    repulsion = unit_vec * (di_negative * 0.01)
    
    # Combine effects
    result = position + attraction + repulsion
    
    return result

def update_simulation():
    """Update simulation state"""
    if params.paused:
        return

    params.time += 100
    
    # Store previous state for membranic drag calculation
    previous_state = [p.copy() for p in params.universal_state["particles"]]
    
    # Update trifecta weights based on RPS principles
    update_trifecta_weights()
    
    # Apply Space-Matter Density relationship
    update_space_matter_density()
    
    # Apply Free Will / Recursive Thought
    apply_recursive_thought()
    
    # Update Neural Mapping Shell
    update_neural_mapping_shell()
    
    # Handle neural compression if needed
    update_neural_compression()
    
    # Calculate the unified singularity equation
    calculate_unified_singularity()

    new_particles = []
    for p in params.universal_state["particles"]:
        # Apply drag
        p['velocity'] *= 0.98
        
        # Apply RPS
        p = apply_rps(p)
        
        # Apply stricter velocity clamping before position update
        p['velocity'] = np.clip(p['velocity'], -0.2, 0.2)
        
        # Apply Fractal Law of Sound to velocity (placeholder)
        p['velocity'] = apply_fractal_sound_law(p['velocity'], params.fractal_sound_harmonics)
        
        # Apply Touch^DI transformation
        p['position'] = apply_touch_dimensional_infinity(
            p['position'],
            params.dimensional_infinity['positive'],
            params.dimensional_infinity['negative']
        )
        
        # Update position (with safety check)
        if np.isfinite(p['velocity']).all() and np.isfinite(p['position']).all():
            # Reframe as universe/C-AE expansion rather than particle movement
            expansion_factor = min(0.05, params.speed_of_dark * 0.01)  # Limit expansion factor
            position_delta = p['velocity'] * expansion_factor
            # Ensure position change is finite and small
            if np.isfinite(position_delta).all() and np.max(np.abs(position_delta)) < 0.5:
                p['position'] += position_delta
        else:
            # Reset velocity if it becomes invalid
            p['velocity'] = np.zeros(3, dtype=np.float32)
        
        p['age'] += 1
        
        # Apply DNA memory influence with enhanced trifecta weighting
        if params.dna_memory and params.show_dna:
            dna_influence = np.zeros(3, dtype=np.float32)
            try:
                # Limit how much DNA memory we process to prevent overflow
                recent_dna = params.dna_memory[-min(3, len(params.dna_memory)):]
                
                for codon in recent_dna:
                    for vel in codon:  # Iterate through each velocity vector in codon
                        if np.isfinite(vel).all():  # Check for valid values
                            # Create a properly shaped copy of the velocity - more careful conversion
                            weighted_vel = np.zeros(3, dtype=np.float32)
                            
                            # Explicit conversion with safety checks and smaller factors
                            r_factor = min(0.05, float(params.trifecta_weights['Red']) * 0.005 * params.dna_impetus)
                            b_factor = min(0.05, float(params.trifecta_weights['Blue']) * 0.005 * params.dna_impetus)
                            y_factor = min(0.05, float(params.trifecta_weights['Yellow']) * 0.005 * params.dna_impetus)
                            
                            # First clip the velocity components to prevent overflow when converting to float
                            clipped_vel = np.clip(vel, -10.0, 10.0)
                            
                            weighted_vel[0] = float(clipped_vel[0]) * r_factor
                            weighted_vel[1] = float(clipped_vel[1]) * b_factor
                            weighted_vel[2] = float(clipped_vel[2]) * y_factor
                            
                            # Extra safety check to prevent overflow
                            weighted_vel = np.clip(weighted_vel, -0.05, 0.05)
                            dna_influence += weighted_vel
                
                # Also apply novel patterns from recursive thought, but be more careful
                if params.novel_patterns:
                    latest_pattern = params.novel_patterns[-1]
                    pattern_count = 0
                    for vel in latest_pattern:
                        if np.isfinite(vel).all():
                            safe_vel = np.clip(vel * 0.005, -0.02, 0.02)  # Even more careful with novel patterns
                            dna_influence += safe_vel
                            pattern_count += 1
                    
                    # Normalize by pattern count to prevent accumulation
                    if pattern_count > 0:
                        dna_influence /= pattern_count
            except (ValueError, TypeError, IndexError) as e:
                # If there's any error in the DNA influence calculation, skip it
                pass
            
            # Apply DNA influence safely with additional validity check
            if np.isfinite(dna_influence).all() and np.max(np.abs(dna_influence)) < 0.1:
                p['velocity'] += dna_influence
            
            # Safety clipping after adding DNA influence
            p['velocity'] = np.clip(p['velocity'], -0.2, 0.2)
        
        # Boundary check with gentle pushback - with safety checks
        dist = np.linalg.norm(p['position'])
        if dist > 10 and np.isfinite(dist) and np.all(np.isfinite(p['position'])):
            pushback = p['position'] * min(0.01, 0.1 / max(1.0, dist))
            if np.all(np.isfinite(pushback)):
                p['velocity'] -= pushback
        
        # Keep particles that are not too old and have valid positions
        if p['age'] <= p['max_age'] and np.isfinite(p['position']).all():
            new_particles.append(p)
    
    # Calculate membranic drag between previous and new state 
    params.membranic_drag = measure_membranic_drag(previous_state, new_particles)
    
    # Calculate pressure change from trifecta imbalance
    weights = list(params.trifecta_weights.values())
    params.pressure_change = max(weights) - min(weights)
    
    params.universal_state["particles"] = new_particles
    
    # Add new particles - using structured RPS instead of random
    if len(params.universal_state["particles"]) < 1000:  # Cap maximum particles
        # Calculate latching point to decide if we should add particles
        lp = calculate_latching_point(params.membranic_drag, params.pressure_change)
        
        if lp > 0:
            # Generate new particle based on existing patterns not random
            if params.excretions:
                # Use RPS to generate position and velocity
                base_vel = rps_generate_variation(params.excretions, params.absorption_factor, params.delay_time)
                
                # Ensure valid velocity
                if not np.isfinite(base_vel).all() or np.linalg.norm(base_vel) < 0.001:
                    # Use structured pattern generation instead of random fallback
                    angle = params.time % (2 * np.pi)
                    z = math.cos(angle * 1.3) * 0.8
                    r = math.sqrt(1 - z**2)
                    x, y = r * math.cos(angle), r * math.sin(angle)
                    base_vel = np.array([x, y, z], dtype=np.float32) * 0.05
                
                pos = np.array([0, 0, 0], dtype=np.float32)
                params.universal_state["particles"].append({
                    'position': pos,
                    'velocity': base_vel,
                    'color': get_trifecta_color(),
                    'size': 1.0,
                    'age': 0,
                    'max_age': 100 + 100 * abs(math.sin(params.time))
                })

    # Add DNA codons using intelligence pattern detection
    if len(params.universal_state["particles"]) > 3 and params.time % 2 < 0.01:
        # Sort particles by velocity magnitude to find the most active ones
        active_particles = sorted(
            params.universal_state["particles"], 
            key=lambda p: np.linalg.norm(p['velocity']),
            reverse=True
        )
        
        if len(active_particles) >= 3:
            # Create a proper codon (R,B,Y structure)
            codon = form_codon(active_particles[0]['velocity'])
            if codon:
                params.dna_memory.append(codon)

    # Check if we've reached Absularity
    if params.expansion_phase and params.space_scale >= params.absularity_threshold:
        if not params.at_absularity:
            params.at_absularity = True
            print(f"Absularity reached at time {params.time:.2f}")

def draw_particles():
    """Draw all particles using vertex arrays for better performance"""
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Draw particles
    glPointSize(params.particle_size)
    glBegin(GL_POINTS)
    for p in params.universal_state["particles"]:
        glColor4f(*p['color'])
        glVertex3f(*p['position'])
    glEnd()
    
    # Draw center singularity (AE=C=1)
    glColor4f(1, 1, 1, 1)
    glPushMatrix()
    glTranslatef(0, 0, 0)
    quad = gluNewQuadric()
    gluSphere(quad, params.singularity_size, 20, 20)
    glPopMatrix()
    
    # Draw Absularity boundary if enabled
    if params.show_absularity:
        glPushMatrix()
        glColor4f(0.3, 0.3, 0.3, 0.2)
        
        # Draw spherical boundary at absularity_threshold
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_LINE)
        glTranslatef(0, 0, 0)
        gluSphere(quad, params.absularity_threshold * 5.0, 32, 32)
        
        glPopMatrix()
    
    # Draw DNA connections if enabled
    if params.show_dna and len(params.universal_state["particles"]) > 1:
        glBegin(GL_LINES)
        for i in range(len(params.universal_state["particles"])-1):
            p1 = params.universal_state["particles"][i]
            p2 = params.universal_state["particles"][i+1]
            alpha = min(0.3, 1.0 - (p1['age'] / p1['max_age']))
            glColor4f(1, 1, 1, alpha)
            glVertex3f(*p1['position'])
            glVertex3f(*p2['position'])
        glEnd()
    
    # Draw memory symbols (glyphs)
    if params.memory_symbols:
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for i, symbol in enumerate(params.memory_symbols):
            # Use pattern to determine color
            pattern = symbol['pattern'] / 10.0
            intensity = min(1.0, symbol['intensity'])
            glColor4f(pattern, 1.0-pattern, intensity, 0.7)
            
            # Position in a ring around the singularity
            angle = (i / len(params.memory_symbols)) * 2 * math.pi
            x = math.cos(angle) * 3.0
            y = math.sin(angle) * 3.0
            z = 0.5
            
            glVertex3f(x, y, z)
        glEnd()

def draw_ui():
    """Draw 2D UI elements using pygame"""
    # Switch to orthographic projection for UI
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, width, height, 0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Disable depth test for UI
    glDisable(GL_DEPTH_TEST)
    
    # Create a surface for text
    font = pygame.font.SysFont('Arial', 18)
    stats = [
        f"Particles: {len(params.universal_state['particles'])}",
        f"DNA Codons: {len(params.dna_memory)}",
        f"Neural Maps: {len(params.singularity_model.neural_maps)}",
        f"Memory Glyphs: {len(params.memory_symbols)}",
        f"Time: {params.time:.1f} | Cycle: {params.absularity_cycle_count}",
        f"R: {params.trifecta_weights['Red']:.2f}  B: {params.trifecta_weights['Blue']:.2f}  Y: {params.trifecta_weights['Yellow']:.2f}",
        f"Active nM: {params.current_neural_model} ({params.neural_cosmic_entities[params.current_neural_model]['dominant_pattern']})",
        f"Space-Matter: {params.space_scale:.2f}/{params.matter_density:.2f}",
        f"Storage: {params.current_storage_usage}/{params.max_storage_capacity} ({params.current_storage_usage/params.max_storage_capacity:.0%})",
        f"{'COMPRESSION ACTIVE' if params.neural_compression_active else 'EXPANSION ACTIVE'}: {params.neural_compression_progress*100:.0f}%",
        f"Harmonics: {params.fractal_sound_harmonics:.1f} | Touch^DI: +{params.dimensional_infinity['positive']:.1f}/-{params.dimensional_infinity['negative']:.1f}",
        f"[Space] Pause [D] DNA [F] FreeWill [A] Absularity [M] Model [T] TouchDI"
    ]
    
    for i, text in enumerate(stats):
        text_surface = font.render(text, True, (255, 255, 255))
        # Fix deprecation warning by using tobytes instead of tostring
        text_data = pygame.image.tobytes(text_surface, "RGBA", True)
        glRasterPos2d(10, 20 + i * 25)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), 
                    GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    # Restore 3D projection
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glEnable(GL_DEPTH_TEST)

def draw_neural_structures():
    """Draw neural mapping structures and components"""
    if not params.show_neural_structures:
        return
    
    # Draw neural triplet grid connections
    if params.neural_triplet_grid and len(params.neural_triplet_grid) > 1:  # Check if we have at least 2 triplets
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        # Position triplets in a grid pattern
        grid_size = max(1, int(np.sqrt(len(params.neural_triplet_grid))))
        spacing = 0.5
        
        for i in range(len(params.neural_triplet_grid) - 1):
            # Get grid positions
            row1, col1 = i // grid_size, i % grid_size
            row2, col2 = (i + 1) // grid_size, (i + 1) % grid_size
            
            # Calculate 3D positions
            pos1 = np.array([(col1 - grid_size/2) * spacing, 
                             (row1 - grid_size/2) * spacing, 
                             2.0])
            pos2 = np.array([(col2 - grid_size/2) * spacing, 
                             (row2 - grid_size/2) * spacing, 
                             2.0])
            
            # Determine color based on triplet content
            triplet = params.neural_triplet_grid[i]
            r = triplet.count('R') / 3
            b = triplet.count('B') / 3
            y = triplet.count('Y') / 3
            
            glColor4f(r, b, y, 0.7)
            glVertex3f(*pos1)
            glVertex3f(*pos2)
        
        glEnd()
    
    # Draw neural mapping shell as a translucent sphere around everything
    if params.neural_mapping_shell['active']:
        glPushMatrix()
        glColor4f(0.3, 0.6, 0.9, 0.15)  # Translucent blue
        
        # Draw shell with size based on neural model
        model_factor = 1.0
        if params.current_neural_model == 'nM1':
            model_factor = 1.2
        elif params.current_neural_model == 'nM2':
            model_factor = 0.9
            
        shell_size = 8.0 * model_factor
        
        # Draw neural shell
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_LINE)
        glTranslatef(0, 0, 0)
        gluSphere(quad, shell_size, 24, 12)
        
        glPopMatrix()
    
    # Draw neural cosmic entities (star/galaxy representations)
    if params.neural_cosmic_entities:  # Check if we have entities to draw
        # Set point size BEFORE beginning to draw
        entities_to_draw = [
            (name, entity) for name, entity in params.neural_cosmic_entities.items() 
            if 'position' in entity and np.all(np.isfinite(entity['position']))
        ]
        
        if entities_to_draw:  # Only draw if we have valid entities
            glPointSize(5.0)
            glBegin(GL_POINTS)
            
            for name, entity in entities_to_draw:
                try:
                    # Calculate color based on RBY dominance
                    seed = entity['seed']
                    total = sum(seed.values())
                    if total > 0:  # Prevent division by zero
                        r = seed['Red'] / total
                        g = seed['Yellow'] / total # Using yellow as green for visualization
                        b = seed['Blue'] / total
                        
                        # Highlight current active model
                        alpha = 0.7
                        size_mult = 1.0
                        if name == params.current_neural_model:
                            alpha = 1.0
                            size_mult = 1.5
                            
                        # Draw the neural entity as a glowing point
                        glColor4f(r, g, b, alpha)
                        pos = entity['position']
                        glVertex3f(pos[0], pos[1], pos[2])
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    # Skip any entity that causes errors
                    continue
            
            glEnd()
    
    # Draw connections between neural entities
    connections_to_draw = []
    if params.neural_cosmic_entities:
        for name, entity in params.neural_cosmic_entities.items():
            if 'connections' in entity and entity['connections'] and 'position' in entity:
                for connected_name in entity['connections']:
                    # Only draw if this is alphabetically before to avoid duplicates
                    if name < connected_name and connected_name in params.neural_cosmic_entities:
                        connected = params.neural_cosmic_entities[connected_name]
                        if 'position' in connected:
                            # Verify both positions are valid
                            if (np.all(np.isfinite(entity['position'])) and 
                                np.all(np.isfinite(connected['position']))):
                                connections_to_draw.append((entity, connected, name, connected_name))
    
    if connections_to_draw:  # Only begin drawing if we have valid connections
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        for entity, connected, name, connected_name in connections_to_draw:
            try:
                # Calculate influence-based alpha
                avg_influence = 0.5
                if 'influence' in entity and 'influence' in connected:
                    avg_influence = (entity['influence'] + connected['influence']) / 2.0
                alpha = min(1.0, avg_influence * 0.5)
                
                # Draw line with gradient color (blend between entity colors)
                glColor4f(0.7, 0.7, 0.7, alpha)
                glVertex3f(*entity['position'])
                glVertex3f(*connected['position'])
            except (KeyError, ValueError, TypeError):
                # Skip any connection that causes errors
                continue
        
        glEnd()
    
    # Draw deposited glyphs around the singularity
    if params.singularity_model.deposited_glyphs:
        glyphs_to_draw = []
        
        for i, glyph in enumerate(params.singularity_model.deposited_glyphs):
            # Position glyphs in a spiral around the singularity
            angle = i * 0.5
            radius = 1.0 + i * 0.1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -0.5 - i * 0.05
            
            # Color based on the glyph's neural model seed
            if glyph in params.singularity_model.source_knowledge:
                seed = params.singularity_model.source_knowledge[glyph]['seed']
                r = min(1.0, seed['Red'] / 2.0)
                b = min(1.0, seed['Blue'] / 2.0)
                y_val = min(1.0, seed['Yellow'] / 2.0)
                
                glyphs_to_draw.append((x, y, z, r, b, y_val))
        
        if glyphs_to_draw:  # Only draw if we have valid glyphs
            glPointSize(3.0)
            glBegin(GL_POINTS)
            
            for x, y, z, r, b, y_val in glyphs_to_draw:
                glColor4f(r, b, y_val, 0.9)
                glVertex3f(x, y, z)
            
            glEnd()

def draw_script_visualization():
    """Visualize the components and processes of the sperm_ileices.py script."""
    # Draw Red, Blue, and Yellow nodes
    glPointSize(10.0)
    glBegin(GL_POINTS)
    for color, position in [('Red', [-3, 0, 0]), ('Blue', [0, 0, 0]), ('Yellow', [3, 0, 0])]:
        if color == 'Red':
            glColor4f(1.0, 0.0, 0.0, 1.0)  # Red
        elif color == 'Blue':
            glColor4f(0.0, 0.0, 1.0, 1.0)  # Blue
        elif color == 'Yellow':
            glColor4f(1.0, 1.0, 0.0, 1.0)  # Yellow
        glVertex3f(*position)
    glEnd()

    # Draw connections between nodes (feedback loops)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor4f(0.5, 0.5, 0.5, 0.7)  # Gray for connections
    glVertex3f(-3, 0, 0)  # Red to Blue
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 0)  # Blue to Yellow
    glVertex3f(3, 0, 0)
    glVertex3f(3, 0, 0)  # Yellow to Red
    glVertex3f(-3, 0, 0)
    glEnd()

    # Visualize memory as glyphs around the nodes
    for i, memory_symbol in enumerate(params.memory_symbols):
        angle = (i / len(params.memory_symbols)) * 2 * math.pi
        x = math.cos(angle) * 5.0
        y = math.sin(angle) * 5.0
        z = 0.0
        glColor4f(1.0, 1.0, 1.0, 0.7)  # White for memory glyphs
        glPushMatrix()
        glTranslatef(x, y, z)
        quad = gluNewQuadric()
        gluSphere(quad, 0.2, 10, 10)
        glPopMatrix()

# Initialize the simulation before the main loop starts
init_simulation()

# Main game loop
clock = pygame.time.Clock()
running = True
mouse_dragging = False
last_mouse_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                params.paused = not params.paused
            elif event.key == pygame.K_d:
                params.show_dna = not params.show_dna
            elif event.key == pygame.K_a:
                params.show_absularity = not params.show_absularity
            elif event.key == pygame.K_r:
                params.trifecta_weights['Red'] = min(10.0, params.trifecta_weights['Red'] + 0.2)
            elif event.key == pygame.K_b:
                params.trifecta_weights['Blue'] = min(10.0, params.trifecta_weights['Blue'] + 0.2)
            elif event.key == pygame.K_y:
                params.trifecta_weights['Yellow'] = min(10.0, params.trifecta_weights['Yellow'] + 0.2)
            elif event.key == pygame.K_f:
                # Toggle free will capacity
                if params.free_will_capacity < 0.9:
                    params.free_will_capacity += 0.1
                else:
                    params.free_will_capacity = 0.1
            elif event.key == pygame.K_UP:
                params.dna_impetus = min(10.0, params.dna_impetus + 0.1)
            elif event.key == pygame.K_DOWN:
                params.dna_impetus = max(0.0, params.dna_impetus - 0.1)
            elif event.key == pygame.K_s:
                params.speed_of_dark = min(3.0, params.speed_of_dark + 0.1)  # Speed of Dark control
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                params.particle_size = min(10.0, params.particle_size + 0.5)
            elif event.key == pygame.K_MINUS:
                params.particle_size = max(0.5, params.particle_size - 0.5)
            elif event.key == pygame.K_n:
                params.show_neural_structures = not params.show_neural_structures
            elif event.key == pygame.K_m:
                # Cycle through neural models
                models = list(params.neural_model_seeds.keys())
                current_idx = models.index(params.current_neural_model)
                next_idx = (current_idx + 1) % len(models)
                params.current_neural_model = models[next_idx]
                # Apply the model's seed values
                params.trifecta_weights = params.neural_model_seeds[params.current_neural_model].copy()
            elif event.key == pygame.K_t:
                # Adjust Touch^DI values
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:  # Negative DI
                    params.dimensional_infinity['negative'] = (params.dimensional_infinity['negative'] + 0.2) % 2.0
                else:  # Positive DI
                    params.dimensional_infinity['positive'] = (params.dimensional_infinity['positive'] + 0.2) % 2.0
            elif event.key == pygame.K_h:
                # Adjust fractal sound harmonics
                params.fractal_sound_harmonics = (params.fractal_sound_harmonics + 0.2) % 3.0
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_dragging = True
                last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_dragging = False
                last_mouse_pos = None
        elif event.type == pygame.MOUSEMOTION and mouse_dragging:
            if last_mouse_pos:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                params.camera_angle[0] += dx * 0.5
                params.camera_angle[1] += dy * 0.5
                last_mouse_pos = event.pos
    
    # Update simulation
    update_simulation()
    
    # Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Set camera position
    glTranslatef(0.0, 0.0, -15)
    glRotatef(params.camera_angle[1], 1, 0, 0)
    glRotatef(params.camera_angle[0], 0, 1, 0)
    
    # Apply space scale from Space-Matter Density
    glScalef(params.space_scale, params.space_scale, params.space_scale)
    
    # Draw simulation
    draw_particles()
    
    # Draw UI
    draw_ui()
    
    # Draw neural structures
    draw_neural_structures()
    
    # Draw script visualization
    draw_script_visualization()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()