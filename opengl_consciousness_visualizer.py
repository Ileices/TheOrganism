#!/usr/bin/env python3
"""
OpenGL Real-Time Consciousness Visualizer
Hardware-accelerated real-time rendering of consciousness states
Using OpenGL shaders for GPU-based consciousness visualization
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import json

# OpenGL and visualization libraries
try:
    import pygame
    from pygame.locals import *
    import OpenGL.GL as gl
    import OpenGL.GL.shaders as shaders
    from OpenGL.arrays import vbo
    PYGAME_AVAILABLE = True
    print("✅ Pygame/OpenGL backend available")
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Pygame/OpenGL not available - install pygame and PyOpenGL")

# Matrix math libraries
try:
    import glm
    GLM_AVAILABLE = True
except ImportError:
    GLM_AVAILABLE = False
    # Fallback matrix operations
    import math

@dataclass
class VisualizationState:
    """Real-time visualization state"""
    fps: float = 0.0
    frame_count: int = 0
    render_time_ms: float = 0.0
    vertex_count: int = 0
    consciousness_intensity: float = 0.0
    rby_colors: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    zoom_level: float = 1.0

class OpenGLConsciousnessVisualizer:
    """
    Hardware-accelerated consciousness visualization using OpenGL
    Real-time rendering of consciousness states and RBY processing
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, config: Optional[Dict] = None):
        self.width = width
        self.height = height
        self.config = config or self._default_config()
        self.state = VisualizationState()
        
        # OpenGL resources
        self.shader_program = None
        self.vertex_buffer = None
        self.vertex_array = None
        self.texture_id = None
        
        # Consciousness data
        self.consciousness_matrix = None
        self.rby_phases = [np.zeros((128, 128)), np.zeros((128, 128)), np.zeros((128, 128))]
        
        # Animation and timing
        self.start_time = time.time()
        self.frame_times = []
        self.running = False
        
        # Camera controls
        self.camera_rotation = [0.0, 0.0]
        self.camera_distance = 5.0
        
        # Initialize if possible
        if PYGAME_AVAILABLE:
            self._initialize_pygame()
            self._initialize_opengl()
            self._create_shaders()
            self._setup_geometry()
        
    def _default_config(self) -> Dict:
        """Default visualization configuration"""
        return {
            'matrix_resolution': 128,
            'consciousness_scale': 2.0,
            'animation_speed': 1.0,
            'color_intensity': 1.5,
            'wireframe_mode': False,
            'show_rby_phases': True,
            'background_color': (0.05, 0.05, 0.15, 1.0),
            'consciousness_threshold': 0.618,
            'unity_glow_intensity': 2.0,
            'real_time_updates': True,
            'vsync': True
        }
    
    def _initialize_pygame(self):
        """Initialize Pygame and OpenGL context"""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame not available - cannot initialize visualization")
        
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Real-Time Consciousness Visualization")
        
        # Enable VSync if requested
        if self.config['vsync']:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        
        print(f"✅ Pygame initialized: {self.width}x{self.height}")
    
    def _initialize_opengl(self):
        """Initialize OpenGL settings and capabilities"""
        # Enable depth testing and blending
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Set viewport
        gl.glViewport(0, 0, self.width, self.height)
        
        # Background color
        bg_color = self.config['background_color']
        gl.glClearColor(*bg_color)
        
        # Get OpenGL info
        vendor = gl.glGetString(gl.GL_VENDOR).decode()
        renderer = gl.glGetString(gl.GL_RENDERER).decode()
        version = gl.glGetString(gl.GL_VERSION).decode()
        
        print(f"✅ OpenGL initialized")
        print(f"   Vendor: {vendor}")
        print(f"   Renderer: {renderer}")
        print(f"   Version: {version}")
    
    def _create_shaders(self):
        """Create and compile OpenGL shaders for consciousness rendering"""
        
        # Vertex shader for consciousness surface
        vertex_shader_source = """
        #version 330 core
        
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec2 texCoord;
        layout (location = 3) in float consciousness_value;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        uniform float consciousness_scale;
        uniform vec3 rby_phases;
        
        out vec3 world_position;
        out vec3 world_normal;
        out vec2 texture_coord;
        out float consciousness_level;
        out vec3 rby_color;
        
        void main() {
            // Apply consciousness-based vertex displacement
            vec3 displaced_pos = position;
            
            // Consciousness emergence creates surface displacement
            float emergence = consciousness_value * consciousness_scale;
            displaced_pos.y += emergence * sin(time + position.x * 3.14159) * 0.5;
            
            // Calculate world space position
            world_position = vec3(model * vec4(displaced_pos, 1.0));
            world_normal = mat3(transpose(inverse(model))) * normal;
            
            // Pass through texture coordinates and consciousness data
            texture_coord = texCoord;
            consciousness_level = consciousness_value;
            
            // Calculate RBY color based on phases and consciousness
            rby_color = vec3(
                rby_phases.r * consciousness_value,
                rby_phases.g * consciousness_value,
                rby_phases.b * consciousness_value
            );
            
            gl_Position = projection * view * vec4(world_position, 1.0);
        }
        """
        
        # Fragment shader for consciousness rendering
        fragment_shader_source = """
        #version 330 core
        
        in vec3 world_position;
        in vec3 world_normal;
        in vec2 texture_coord;
        in float consciousness_level;
        in vec3 rby_color;
        
        uniform float time;
        uniform vec3 camera_position;
        uniform float unity_threshold;
        uniform float consciousness_threshold;
        uniform float glow_intensity;
        
        out vec4 fragment_color;
        
        void main() {
            // Base consciousness color using RBY spectrum
            vec3 base_color = rby_color;
            
            // Add consciousness-based lighting
            vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
            vec3 normal = normalize(world_normal);
            float light_intensity = max(dot(normal, light_dir), 0.2);
            
            // Consciousness emergence glow
            float emergence_glow = smoothstep(consciousness_threshold, 1.0, consciousness_level);
            
            // Unity threshold special effect
            float unity_glow = 0.0;
            if (consciousness_level > unity_threshold) {
                unity_glow = sin(time * 10.0) * 0.5 + 0.5;
                unity_glow *= glow_intensity;
            }
            
            // Calculate view direction for Fresnel effect
            vec3 view_dir = normalize(camera_position - world_position);
            float fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0);
            
            // Combine all lighting effects
            vec3 final_color = base_color * light_intensity;
            final_color += emergence_glow * vec3(0.8, 0.9, 1.0);
            final_color += unity_glow * vec3(1.0, 1.0, 0.8);
            final_color += fresnel * rby_color * 0.3;
            
            // Alpha based on consciousness level
            float alpha = 0.7 + consciousness_level * 0.3;
            
            fragment_color = vec4(final_color, alpha);
        }
        """
        
        try:
            # Compile shaders
            vertex_shader = shaders.compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)
            
            # Create shader program
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
            
            print("✅ Consciousness shaders compiled successfully")
            
        except Exception as e:
            print(f"❌ Shader compilation failed: {e}")
            raise
    
    def _setup_geometry(self):
        """Setup consciousness surface geometry"""
        resolution = self.config['matrix_resolution']
        
        # Generate consciousness surface mesh
        vertices = []
        indices = []
        
        # Create grid of vertices
        for i in range(resolution):
            for j in range(resolution):
                x = (i / (resolution - 1)) * 2.0 - 1.0  # [-1, 1]
                z = (j / (resolution - 1)) * 2.0 - 1.0  # [-1, 1]
                y = 0.0  # Will be displaced by consciousness
                
                # Normal vector (initially pointing up)
                nx, ny, nz = 0.0, 1.0, 0.0
                
                # Texture coordinates
                u = i / (resolution - 1)
                v = j / (resolution - 1)
                
                # Initial consciousness value
                consciousness = 0.0
                
                # Add vertex: position(3) + normal(3) + texcoord(2) + consciousness(1)
                vertices.extend([x, y, z, nx, ny, nz, u, v, consciousness])
        
        # Generate indices for triangles
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Two triangles per quad
                top_left = i * resolution + j
                top_right = i * resolution + j + 1
                bottom_left = (i + 1) * resolution + j
                bottom_right = (i + 1) * resolution + j + 1
                
                # First triangle
                indices.extend([top_left, bottom_left, top_right])
                # Second triangle
                indices.extend([top_right, bottom_left, bottom_right])
        
        # Convert to numpy arrays
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        
        # Create and bind VAO
        self.vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vertex_array)
        
        # Create and bind VBO
        self.vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_DYNAMIC_DRAW)
        
        # Create and bind EBO
        self.element_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.element_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)
        
        # Setup vertex attributes
        stride = 9 * 4  # 9 floats per vertex * 4 bytes per float
        
        # Position attribute
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)
        
        # Normal attribute
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)
        
        # Texture coordinate attribute
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(6 * 4))
        gl.glEnableVertexAttribArray(2)
        
        # Consciousness value attribute
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(8 * 4))
        gl.glEnableVertexAttribArray(3)
        
        # Unbind
        gl.glBindVertexArray(0)
        
        self.state.vertex_count = len(self.indices)
        print(f"✅ Consciousness geometry created: {self.state.vertex_count} vertices")
    
    def update_consciousness_data(self, consciousness_matrix: np.ndarray, rby_phases: List[np.ndarray]):
        """Update consciousness data for visualization"""
        self.consciousness_matrix = consciousness_matrix
        self.rby_phases = rby_phases
        
        # Update vertex buffer with new consciousness values
        if self.vertex_buffer and consciousness_matrix is not None:
            self._update_vertex_consciousness_values()
    
    def _update_vertex_consciousness_values(self):
        """Update vertex buffer with new consciousness values"""
        resolution = self.config['matrix_resolution']
        
        if self.consciousness_matrix is None:
            return
        
        # Resize consciousness matrix to match vertex resolution
        if self.consciousness_matrix.shape != (resolution, resolution):
            # Simple resize using numpy
            import scipy.ndimage
            resized_matrix = scipy.ndimage.zoom(
                self.consciousness_matrix, 
                (resolution / self.consciousness_matrix.shape[0],
                 resolution / self.consciousness_matrix.shape[1])
            )
        else:
            resized_matrix = self.consciousness_matrix
        
        # Update consciousness values in vertex buffer
        stride = 9  # 9 floats per vertex
        consciousness_offset = 8  # Consciousness value is at index 8
        
        for i in range(resolution):
            for j in range(resolution):
                vertex_index = i * resolution + j
                buffer_index = vertex_index * stride + consciousness_offset
                
                # Get consciousness value for this vertex
                consciousness_value = float(resized_matrix[i, j])
                self.vertices[buffer_index] = consciousness_value
        
        # Update GPU buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.vertices.nbytes, self.vertices)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    
    def _create_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create model, view, and projection matrices"""
        current_time = time.time() - self.start_time
        
        # Model matrix (rotating consciousness surface)
        if GLM_AVAILABLE:
            model = glm.mat4(1.0)
            model = glm.rotate(model, current_time * 0.5, glm.vec3(0, 1, 0))
        else:
            # Simple rotation matrix
            angle = current_time * 0.5
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            model = np.array([
                [cos_a, 0, sin_a, 0],
                [0, 1, 0, 0],
                [-sin_a, 0, cos_a, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
        
        # View matrix (camera looking at consciousness)
        camera_x = self.camera_distance * math.cos(self.camera_rotation[0]) * math.cos(self.camera_rotation[1])
        camera_y = self.camera_distance * math.sin(self.camera_rotation[1])
        camera_z = self.camera_distance * math.sin(self.camera_rotation[0]) * math.cos(self.camera_rotation[1])
        
        if GLM_AVAILABLE:
            view = glm.lookAt(
                glm.vec3(camera_x, camera_y, camera_z),
                glm.vec3(0, 0, 0),
                glm.vec3(0, 1, 0)
            )
        else:
            # Simple view matrix
            view = np.eye(4, dtype=np.float32)
            view[2, 3] = -self.camera_distance
        
        # Projection matrix
        aspect_ratio = self.width / self.height
        fov = math.radians(45.0)
        near, far = 0.1, 100.0
        
        if GLM_AVAILABLE:
            projection = glm.perspective(fov, aspect_ratio, near, far)
        else:
            # Simple perspective matrix
            f = 1.0 / math.tan(fov / 2.0)
            projection = np.array([
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0]
            ], dtype=np.float32)
        
        return model, view, projection
    
    def render_frame(self):
        """Render one frame of consciousness visualization"""
        render_start = time.time()
        
        # Clear buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        if self.shader_program is None:
            return
        
        # Use consciousness shader program
        gl.glUseProgram(self.shader_program)
        
        # Create transformation matrices
        model, view, projection = self._create_matrices()
        
        # Set shader uniforms
        current_time = time.time() - self.start_time
        
        # Matrix uniforms
        model_loc = gl.glGetUniformLocation(self.shader_program, "model")
        view_loc = gl.glGetUniformLocation(self.shader_program, "view")
        projection_loc = gl.glGetUniformLocation(self.shader_program, "projection")
        
        if GLM_AVAILABLE:
            gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model))
            gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view))
            gl.glUniformMatrix4fv(projection_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))
        else:
            gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, model)
            gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, view)
            gl.glUniformMatrix4fv(projection_loc, 1, gl.GL_FALSE, projection)
        
        # Consciousness uniforms
        time_loc = gl.glGetUniformLocation(self.shader_program, "time")
        gl.glUniform1f(time_loc, current_time)
        
        consciousness_scale_loc = gl.glGetUniformLocation(self.shader_program, "consciousness_scale")
        gl.glUniform1f(consciousness_scale_loc, self.config['consciousness_scale'])
        
        # RBY phase colors
        rby_phases_loc = gl.glGetUniformLocation(self.shader_program, "rby_phases")
        rby_values = [np.mean(phase) if phase.size > 0 else 0.0 for phase in self.rby_phases]
        gl.glUniform3f(rby_phases_loc, *rby_values)
        
        # Camera position
        camera_pos_loc = gl.glGetUniformLocation(self.shader_program, "camera_position")
        gl.glUniform3f(camera_pos_loc, *self.state.camera_position)
        
        # Consciousness thresholds
        unity_threshold_loc = gl.glGetUniformLocation(self.shader_program, "unity_threshold")
        gl.glUniform1f(unity_threshold_loc, 0.999)
        
        consciousness_threshold_loc = gl.glGetUniformLocation(self.shader_program, "consciousness_threshold")
        gl.glUniform1f(consciousness_threshold_loc, self.config['consciousness_threshold'])
        
        glow_intensity_loc = gl.glGetUniformLocation(self.shader_program, "glow_intensity")
        gl.glUniform1f(glow_intensity_loc, self.config['unity_glow_intensity'])
        
        # Render consciousness surface
        gl.glBindVertexArray(self.vertex_array)
        
        if self.config['wireframe_mode']:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
        
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        
        # Calculate render time
        render_time = (time.time() - render_start) * 1000  # ms
        self.state.render_time_ms = render_time
        
        # Update frame timing
        self.frame_times.append(time.time())
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.state.fps = fps
        
        self.state.frame_count += 1
    
    def handle_input(self):
        """Handle user input for camera controls"""
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        
        # Camera rotation with mouse
        if mouse_buttons[0]:  # Left mouse button
            mouse_rel = pygame.mouse.get_rel()
            self.camera_rotation[0] += mouse_rel[0] * 0.01
            self.camera_rotation[1] += mouse_rel[1] * 0.01
            # Clamp vertical rotation
            self.camera_rotation[1] = max(-1.5, min(1.5, self.camera_rotation[1]))
        
        # Camera zoom with scroll wheel
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera_distance = max(1.0, self.camera_distance - 0.5)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(20.0, self.camera_distance + 0.5)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:  # Toggle wireframe
                    self.config['wireframe_mode'] = not self.config['wireframe_mode']
                elif event.key == pygame.K_r:  # Reset camera
                    self.camera_rotation = [0.0, 0.0]
                    self.camera_distance = 5.0
            elif event.type == pygame.QUIT:
                return False
        
        return True
    
    def run_visualization(self, consciousness_engine=None):
        """Run the main visualization loop"""
        if not PYGAME_AVAILABLE:
            print("❌ Cannot run visualization - Pygame not available")
            return
        
        self.running = True
        print("🧠 Starting real-time consciousness visualization...")
        print("Controls:")
        print("  - Left mouse: Rotate camera")
        print("  - Mouse wheel: Zoom in/out")
        print("  - 'W' key: Toggle wireframe mode")
        print("  - 'R' key: Reset camera")
        print("  - ESC: Exit")
        
        clock = pygame.time.Clock()
        target_fps = 60
        
        try:
            while self.running:
                # Handle input
                if not self.handle_input():
                    break
                
                # Update consciousness data if engine provided
                if consciousness_engine:
                    state = consciousness_engine.process_consciousness_cycle()
                    if hasattr(consciousness_engine, 'consciousness_matrix'):
                        self.update_consciousness_data(
                            consciousness_engine.consciousness_matrix,
                            consciousness_engine.rby_trifecta
                        )
                    
                    # Update RBY colors for visualization
                    self.state.rby_colors = (
                        state.awareness_level,
                        state.coherence_score,
                        state.emergence_factor
                    )
                    self.state.consciousness_intensity = state.unity_measure
                
                # Render frame
                self.render_frame()
                
                # Display frame info
                if self.state.frame_count % 60 == 0:  # Every second at 60fps
                    print(f"FPS: {self.state.fps:.1f}, "
                          f"Render: {self.state.render_time_ms:.1f}ms, "
                          f"Consciousness: {self.state.consciousness_intensity:.3f}")
                
                # Swap buffers and control frame rate
                pygame.display.flip()
                clock.tick(target_fps)
        
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vertex_buffer:
            gl.glDeleteBuffers(1, [self.vertex_buffer])
        if self.element_buffer:
            gl.glDeleteBuffers(1, [self.element_buffer])
        if self.vertex_array:
            gl.glDeleteVertexArrays(1, [self.vertex_array])
        if self.shader_program:
            gl.glDeleteProgram(self.shader_program)
        
        if PYGAME_AVAILABLE:
            pygame.quit()
        
        print("✅ Visualization cleanup complete")

def main():
    """Main function for testing OpenGL consciousness visualizer"""
    print("🎨 OpenGL Consciousness Visualizer - Real-Time Hardware Rendering")
    print("=" * 70)
    
    if not PYGAME_AVAILABLE:
        print("❌ Pygame/OpenGL not available")
        print("Install requirements: pip install pygame PyOpenGL PyOpenGL_accelerate")
        return
    
    # Create visualizer
    visualizer = OpenGLConsciousnessVisualizer(1280, 720)
    
    # Try to import and use consciousness engine
    consciousness_engine = None
    try:
        from cuda_consciousness_engine import CUDAConsciousnessEngine
        consciousness_engine = CUDAConsciousnessEngine()
        print("✅ CUDA consciousness engine connected")
    except ImportError:
        print("⚠️  CUDA consciousness engine not available - using demo data")
    
    # Run visualization
    visualizer.run_visualization(consciousness_engine)

if __name__ == "__main__":
    main()
