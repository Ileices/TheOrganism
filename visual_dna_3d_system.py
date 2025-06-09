#!/usr/bin/env python3
"""
3D Visual DNA Visualization System
=================================

Advanced 3D visualization system for codebase representation with:
- Spatial encoding in 3D voxel space
- Interactive web-based rendering
- Real-time execution visualization
- RBY color space integration
- WebGL/Three.js export capabilities

Implements the 3D architecture outlined in ADVANCED_VISUALIZATION_ANALYSIS.md
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import base64

# For 3D operations
try:
    import trimesh
    import pygltf2
except ImportError:
    print("WARNING: 3D libraries not available. Install with: pip install trimesh pygltf2")

@dataclass
class Voxel:
    """Individual voxel in 3D space"""
    position: Tuple[int, int, int]
    rby_data: Tuple[float, float, float]  # R, B, Y values
    file_path: str
    complexity_score: float
    connections: List[Tuple[int, int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'position': self.position,
            'rby': self.rby_data,
            'file_path': self.file_path,
            'complexity': self.complexity_score,
            'connections': self.connections,
            'metadata': self.metadata
        }

@dataclass
class VoxelSpace:
    """3D voxel space for codebase representation"""
    dimensions: Tuple[int, int, int]  # X, Y, Z dimensions
    voxels: Dict[Tuple[int, int, int], Voxel] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def place_voxel(self, position: Tuple[int, int, int], voxel: Voxel):
        """Place a voxel at specified position"""
        if self._is_valid_position(position):
            self.voxels[position] = voxel
        else:
            raise ValueError(f"Invalid position {position} for dimensions {self.dimensions}")
    
    def _is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        """Check if position is within bounds"""
        x, y, z = position
        max_x, max_y, max_z = self.dimensions
        return 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z
    
    def get_neighbors(self, position: Tuple[int, int, int]) -> List[Voxel]:
        """Get neighboring voxels"""
        x, y, z = position
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    
                    neighbor_pos = (x + dx, y + dy, z + dz)
                    if neighbor_pos in self.voxels:
                        neighbors.append(self.voxels[neighbor_pos])
        
        return neighbors

class ThreeDimensionalVisualDNA:
    """
    3D Visual DNA system implementing the architecture from analysis
    
    Provides:
    - Spatial encoding of codebase relationships
    - 3D visualization with voxel representation
    - WebGL export for interactive viewing
    - Real-time execution visualization
    """
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.logger = self._setup_logging()
        
        # 3D encoding dimensions
        self.dimensions = {
            'x': 'file_relationships',     # Horizontal connections
            'y': 'complexity_levels',      # Vertical hierarchy
            'z': 'temporal_evolution'      # Time-based changes
        }
        
        # Initialize voxel space
        self.voxel_space = None
        self.resolution = (1000, 1000, 100)  # Default resolution
        
        self.logger.info("3D Visual DNA system initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for 3D system"""
        logger = logging.getLogger("3DVisualDNA")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_3d_visualization(self, codebase_data: Dict) -> Dict[str, Any]:
        """
        Create comprehensive 3D visualization of codebase
        
        Args:
            codebase_data: Collected codebase information
            
        Returns:
            Complete 3D visualization data including voxel space and exports
        """
        self.logger.info("Creating 3D visualization...")
        
        try:
            # Initialize voxel space
            self.voxel_space = VoxelSpace(dimensions=self.resolution)
            
            # Process files into 3D space
            voxel_mapping = self._process_files_to_3d(codebase_data)
            
            # Calculate spatial relationships
            spatial_relationships = self._calculate_spatial_relationships(codebase_data)
            
            # Generate 3D mesh representation
            mesh_data = self._generate_3d_mesh()
            
            # Create WebGL export
            webgl_export = self._create_webgl_export()
            
            # Generate interactive HTML
            interactive_html = self._generate_interactive_html()
            
            # Calculate 3D metrics
            metrics = self._calculate_3d_metrics()
            
            result = {
                'voxel_space': self._serialize_voxel_space(),
                'voxel_mapping': voxel_mapping,
                'spatial_relationships': spatial_relationships,
                'mesh_data': mesh_data,
                'webgl_export': webgl_export,
                'interactive_html': interactive_html,
                'metrics': metrics,
                'rendering_instructions': self._create_rendering_instructions()
            }
            
            self.logger.info("3D visualization created successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating 3D visualization: {e}")
            raise
    
    def _process_files_to_3d(self, codebase_data: Dict) -> Dict[str, Tuple[int, int, int]]:
        """Process files into 3D voxel positions"""
        voxel_mapping = {}
        
        files = codebase_data.get('files', {})
        relationships = codebase_data.get('relationships', {})
        
        # Calculate optimal positions for each file
        for file_path, file_data in files.items():
            try:
                # Calculate position based on multiple factors
                position = self._calculate_optimal_position(
                    file_path, 
                    file_data, 
                    relationships.get(file_path, {})
                )
                
                # Create RBY data for the file
                rby_data = self._calculate_rby_for_file(file_data)
                
                # Create voxel
                voxel = Voxel(
                    position=position,
                    rby_data=rby_data,
                    file_path=file_path,
                    complexity_score=self._calculate_complexity_score(file_data),
                    metadata={
                        'file_size': file_data.get('size', 0),
                        'file_type': file_data.get('type', 'unknown'),
                        'last_modified': file_data.get('last_modified', ''),
                        'line_count': file_data.get('line_count', 0)
                    }
                )
                
                # Place voxel in space
                self.voxel_space.place_voxel(position, voxel)
                voxel_mapping[file_path] = position
                
            except Exception as e:
                self.logger.warning(f"Failed to process file {file_path}: {e}")
                continue
        
        return voxel_mapping
    
    def _calculate_optimal_position(
        self, 
        file_path: str, 
        file_data: Dict, 
        relationships: Dict
    ) -> Tuple[int, int, int]:
        """Calculate optimal 3D position for a file"""
        
        # X-axis: File relationships (connectivity)
        connectivity_score = len(relationships.get('imports', [])) + len(relationships.get('imported_by', []))
        x = min(int(connectivity_score * 50), self.resolution[0] - 1)
        
        # Y-axis: Complexity levels (file complexity)
        complexity = file_data.get('complexity', 0)
        y = min(int(complexity * 10), self.resolution[1] - 1)
        
        # Z-axis: Temporal evolution (based on file modification time)
        # For now, use file type as a proxy for temporal layers
        file_ext = Path(file_path).suffix
        extension_layers = {
            '.py': 10, '.js': 20, '.ts': 25, '.html': 30, '.css': 35,
            '.md': 40, '.json': 45, '.yml': 50, '.yaml': 50, '.txt': 55
        }
        z = extension_layers.get(file_ext, 60)
        
        return (x, y, z)
    
    def _calculate_rby_for_file(self, file_data: Dict) -> Tuple[float, float, float]:
        """Calculate RBY values for file representation"""
        
        # R (Red): Complexity/Error potential
        complexity = file_data.get('complexity', 0)
        r_value = min(complexity / 100.0, 1.0)
        
        # B (Blue): Connectivity/Integration
        imports = len(file_data.get('imports', []))
        b_value = min(imports / 20.0, 1.0)
        
        # Y (Yellow): Activity/Importance
        size = file_data.get('size', 0)
        y_value = min(size / 10000.0, 1.0)
        
        return (r_value, b_value, y_value)
    
    def _calculate_complexity_score(self, file_data: Dict) -> float:
        """Calculate complexity score for a file"""
        
        factors = {
            'size': file_data.get('size', 0) / 1000.0,
            'line_count': file_data.get('line_count', 0) / 100.0,
            'imports': len(file_data.get('imports', [])) * 2.0,
            'functions': len(file_data.get('functions', [])) * 1.5,
            'classes': len(file_data.get('classes', [])) * 3.0
        }
        
        return sum(factors.values())
    
    def _calculate_spatial_relationships(self, codebase_data: Dict) -> Dict[str, Any]:
        """Calculate 3D spatial relationships between files"""
        
        relationships = {}
        files = codebase_data.get('files', {})
        
        for file_path, file_data in files.items():
            file_relationships = []
            
            # Find spatial neighbors
            if file_path in self.voxel_space.voxels:
                position = None
                for pos, voxel in self.voxel_space.voxels.items():
                    if voxel.file_path == file_path:
                        position = pos
                        break
                
                if position:
                    neighbors = self.voxel_space.get_neighbors(position)
                    file_relationships = [n.file_path for n in neighbors]
            
            relationships[file_path] = {
                'spatial_neighbors': file_relationships,
                'import_connections': file_data.get('imports', []),
                'dependency_distance': self._calculate_dependency_distance(file_path, files)
            }
        
        return relationships
    
    def _calculate_dependency_distance(self, file_path: str, files: Dict) -> Dict[str, float]:
        """Calculate 3D distance to dependency files"""
        distances = {}
        
        file_imports = files.get(file_path, {}).get('imports', [])
        
        # Get position of current file
        current_pos = None
        for pos, voxel in self.voxel_space.voxels.items():
            if voxel.file_path == file_path:
                current_pos = pos
                break
        
        if current_pos:
            for import_file in file_imports:
                # Find position of imported file
                import_pos = None
                for pos, voxel in self.voxel_space.voxels.items():
                    if voxel.file_path == import_file:
                        import_pos = pos
                        break
                
                if import_pos:
                    # Calculate 3D Euclidean distance
                    distance = np.sqrt(
                        (current_pos[0] - import_pos[0]) ** 2 +
                        (current_pos[1] - import_pos[1]) ** 2 +
                        (current_pos[2] - import_pos[2]) ** 2
                    )
                    distances[import_file] = float(distance)
        
        return distances
    
    def _generate_3d_mesh(self) -> Dict[str, Any]:
        """Generate 3D mesh representation for rendering"""
        
        vertices = []
        colors = []
        indices = []
        
        vertex_index = 0
        
        for position, voxel in self.voxel_space.voxels.items():
            # Create cube vertices for each voxel
            x, y, z = position
            
            # Cube vertices (8 vertices per cube)
            cube_vertices = [
                [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],        # Bottom face
                [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1] # Top face
            ]
            
            # Add vertices
            vertices.extend(cube_vertices)
            
            # Add colors (RBY data for each vertex)
            r, b, y = voxel.rby_data
            vertex_color = [r, b, y, 1.0]  # RGBA
            colors.extend([vertex_color] * 8)
            
            # Add cube indices (12 triangles = 36 indices per cube)
            cube_indices = [
                # Bottom face
                vertex_index, vertex_index+1, vertex_index+2,
                vertex_index, vertex_index+2, vertex_index+3,
                # Top face
                vertex_index+4, vertex_index+6, vertex_index+5,
                vertex_index+4, vertex_index+7, vertex_index+6,
                # Side faces
                vertex_index, vertex_index+4, vertex_index+5,
                vertex_index, vertex_index+5, vertex_index+1,
                vertex_index+1, vertex_index+5, vertex_index+6,
                vertex_index+1, vertex_index+6, vertex_index+2,
                vertex_index+2, vertex_index+6, vertex_index+7,
                vertex_index+2, vertex_index+7, vertex_index+3,
                vertex_index+3, vertex_index+7, vertex_index+4,
                vertex_index+3, vertex_index+4, vertex_index
            ]
            
            indices.extend(cube_indices)
            vertex_index += 8
        
        return {
            'vertices': vertices,
            'colors': colors,
            'indices': indices,
            'vertex_count': len(vertices),
            'triangle_count': len(indices) // 3,
            'format': 'indexed_triangles'
        }
    
    def _create_webgl_export(self) -> Dict[str, Any]:
        """Create WebGL-compatible export"""
        
        mesh_data = self._generate_3d_mesh()
        
        # Convert to typed arrays for WebGL
        webgl_data = {
            'vertices': {
                'data': mesh_data['vertices'],
                'type': 'Float32Array',
                'stride': 3  # 3 components per vertex (x, y, z)
            },
            'colors': {
                'data': mesh_data['colors'],
                'type': 'Float32Array',
                'stride': 4  # 4 components per color (r, g, b, a)
            },
            'indices': {
                'data': mesh_data['indices'],
                'type': 'Uint16Array'
            },
            'metadata': {
                'vertex_count': mesh_data['vertex_count'],
                'triangle_count': mesh_data['triangle_count'],
                'bounding_box': self._calculate_bounding_box(),
                'camera_position': self._suggest_camera_position()
            }
        }
        
        return webgl_data
    
    def _generate_interactive_html(self) -> str:
        """Generate interactive HTML with Three.js viewer"""
        
        webgl_data = self._create_webgl_export()
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Visual DNA Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; background: #000; }
        canvas { display: block; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>3D Visual DNA Viewer</h3>
        <p>Vertices: {vertex_count}</p>
        <p>Triangles: {triangle_count}</p>
        <p>Files: {file_count}</p>
        <p>Use mouse to orbit, wheel to zoom</p>
    </div>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        
        // Create geometry from Visual DNA data
        const geometry = new THREE.BufferGeometry();
        
        // Vertices
        const vertices = new Float32Array({vertices_json});
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        // Colors
        const colors = new Float32Array({colors_json});
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 4));
        
        // Indices
        const indices = new Uint16Array({indices_json});
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        
        // Material
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
        
        // Mesh
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // Camera position
        camera.position.set({camera_x}, {camera_y}, {camera_z});
        camera.lookAt(0, 0, 0);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
    </script>
</body>
</html>
        """.format(
            vertex_count=webgl_data['metadata']['vertex_count'],
            triangle_count=webgl_data['metadata']['triangle_count'],
            file_count=len(self.voxel_space.voxels),
            vertices_json=json.dumps([coord for vertex in webgl_data['vertices']['data'] for coord in vertex]),
            colors_json=json.dumps([val for color in webgl_data['colors']['data'] for val in color]),
            indices_json=json.dumps(webgl_data['indices']['data']),
            camera_x=webgl_data['metadata']['camera_position'][0],
            camera_y=webgl_data['metadata']['camera_position'][1],
            camera_z=webgl_data['metadata']['camera_position'][2]
        )
        
        return html_template
    
    def _calculate_bounding_box(self) -> Dict[str, Tuple[float, float, float]]:
        """Calculate bounding box of the 3D visualization"""
        
        if not self.voxel_space.voxels:
            return {'min': (0, 0, 0), 'max': (0, 0, 0)}
        
        positions = list(self.voxel_space.voxels.keys())
        
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        min_z = min(pos[2] for pos in positions)
        max_z = max(pos[2] for pos in positions)
        
        return {
            'min': (min_x, min_y, min_z),
            'max': (max_x, max_y, max_z)
        }
    
    def _suggest_camera_position(self) -> Tuple[float, float, float]:
        """Suggest optimal camera position for viewing"""
        
        bbox = self._calculate_bounding_box()
        
        # Center of bounding box
        center_x = (bbox['min'][0] + bbox['max'][0]) / 2
        center_y = (bbox['min'][1] + bbox['max'][1]) / 2
        center_z = (bbox['min'][2] + bbox['max'][2]) / 2
        
        # Distance from center
        distance = max(
            bbox['max'][0] - bbox['min'][0],
            bbox['max'][1] - bbox['min'][1],
            bbox['max'][2] - bbox['min'][2]
        ) * 2
        
        return (center_x + distance, center_y + distance, center_z + distance)
    
    def _calculate_3d_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive 3D visualization metrics"""
        
        metrics = {
            'voxel_count': len(self.voxel_space.voxels),
            'spatial_density': self._calculate_spatial_density(),
            'connectivity_analysis': self._analyze_3d_connectivity(),
            'complexity_distribution': self._analyze_complexity_distribution(),
            'rby_statistics': self._calculate_rby_statistics(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        return metrics
    
    def _calculate_spatial_density(self) -> Dict[str, float]:
        """Calculate spatial density of voxels"""
        
        total_volume = self.resolution[0] * self.resolution[1] * self.resolution[2]
        occupied_volume = len(self.voxel_space.voxels)
        
        return {
            'overall_density': occupied_volume / total_volume,
            'voxel_count': occupied_volume,
            'total_volume': total_volume
        }
    
    def _analyze_3d_connectivity(self) -> Dict[str, Any]:
        """Analyze 3D connectivity patterns"""
        
        connectivity_data = {}
        
        for position, voxel in self.voxel_space.voxels.items():
            neighbors = self.voxel_space.get_neighbors(position)
            connectivity_data[voxel.file_path] = {
                'neighbor_count': len(neighbors),
                'position': position,
                'isolation_score': 1.0 / (len(neighbors) + 1)
            }
        
        return connectivity_data
    
    def _analyze_complexity_distribution(self) -> Dict[str, float]:
        """Analyze complexity distribution across 3D space"""
        
        complexities = [voxel.complexity_score for voxel in self.voxel_space.voxels.values()]
        
        return {
            'mean_complexity': np.mean(complexities),
            'std_complexity': np.std(complexities),
            'min_complexity': np.min(complexities),
            'max_complexity': np.max(complexities)
        }
    
    def _calculate_rby_statistics(self) -> Dict[str, Any]:
        """Calculate RBY color space statistics"""
        
        rby_data = [voxel.rby_data for voxel in self.voxel_space.voxels.values()]
        
        if not rby_data:
            return {}
        
        r_values = [rby[0] for rby in rby_data]
        b_values = [rby[1] for rby in rby_data]
        y_values = [rby[2] for rby in rby_data]
        
        return {
            'r_statistics': {
                'mean': np.mean(r_values),
                'std': np.std(r_values),
                'range': (np.min(r_values), np.max(r_values))
            },
            'b_statistics': {
                'mean': np.mean(b_values),
                'std': np.std(b_values),
                'range': (np.min(b_values), np.max(b_values))
            },
            'y_statistics': {
                'mean': np.mean(y_values),
                'std': np.std(y_values),
                'range': (np.min(y_values), np.max(y_values))
            }
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate rendering performance metrics"""
        
        mesh_data = self._generate_3d_mesh()
        
        return {
            'vertex_count': mesh_data['vertex_count'],
            'triangle_count': mesh_data['triangle_count'],
            'memory_estimate_mb': (
                (mesh_data['vertex_count'] * 3 * 4) +  # Vertices (float32)
                (mesh_data['vertex_count'] * 4 * 4) +  # Colors (float32)
                (len(mesh_data['indices']) * 2)        # Indices (uint16)
            ) / (1024 * 1024),
            'rendering_complexity': 'HIGH' if mesh_data['triangle_count'] > 100000 else 'MEDIUM' if mesh_data['triangle_count'] > 10000 else 'LOW'
        }
    
    def _serialize_voxel_space(self) -> Dict[str, Any]:
        """Serialize voxel space for storage/transmission"""
        
        serialized = {
            'dimensions': self.voxel_space.dimensions,
            'voxel_count': len(self.voxel_space.voxels),
            'voxels': {}
        }
        
        for position, voxel in self.voxel_space.voxels.items():
            pos_key = f"{position[0]},{position[1]},{position[2]}"
            serialized['voxels'][pos_key] = voxel.to_dict()
        
        return serialized
    
    def _create_rendering_instructions(self) -> Dict[str, Any]:
        """Create comprehensive rendering instructions"""
        
        return {
            'webgl_setup': {
                'required_extensions': ['OES_standard_derivatives'],
                'shader_requirements': ['vertex_colors', 'transparency'],
                'buffer_requirements': ['position', 'color', 'index']
            },
            'camera_settings': {
                'fov': 75,
                'near': 0.1,
                'far': 10000,
                'initial_position': self._suggest_camera_position()
            },
            'lighting_setup': {
                'ambient_light': {'color': 0x404040, 'intensity': 0.6},
                'directional_light': {'color': 0xffffff, 'intensity': 0.4, 'position': [1, 1, 1]}
            },
            'interaction_controls': {
                'orbit_controls': True,
                'zoom_enabled': True,
                'pan_enabled': True,
                'damping': True,
                'damping_factor': 0.25
            },
            'performance_optimizations': {
                'frustum_culling': True,
                'level_of_detail': True,
                'instanced_rendering': False  # Set to True for very large datasets
            }
        }
    
    def export_for_web(self, output_dir: str) -> Dict[str, str]:
        """Export complete 3D visualization for web deployment"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all components
        visualization_data = self.create_3d_visualization({})  # Empty for now, will be filled by caller
        
        # Save HTML viewer
        html_path = output_path / "visual_dna_3d_viewer.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(visualization_data['interactive_html'])
        
        # Save JSON data
        json_path = output_path / "visual_dna_3d_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'voxel_space': visualization_data['voxel_space'],
                'webgl_export': visualization_data['webgl_export'],
                'metrics': visualization_data['metrics']
            }, f, indent=2)
        
        # Save rendering instructions
        instructions_path = output_path / "rendering_instructions.json"
        with open(instructions_path, 'w', encoding='utf-8') as f:
            json.dump(visualization_data['rendering_instructions'], f, indent=2)
        
        return {
            'html_viewer': str(html_path),
            'data_file': str(json_path),
            'instructions': str(instructions_path),
            'status': 'success'
        }

if __name__ == "__main__":
    # Example usage
    workspace = r"C:\Users\lokee\Documents\fake_singularity"
    
    # Initialize 3D visualization system
    viz_3d = ThreeDimensionalVisualDNA(workspace)
    
    # Create sample codebase data (in real use, this would come from codebase collector)
    sample_data = {
        'files': {
            'main.py': {
                'size': 1500,
                'complexity': 25,
                'line_count': 75,
                'imports': ['os', 'sys', 'json'],
                'functions': ['main', 'process_data'],
                'classes': []
            },
            'utils.py': {
                'size': 800,
                'complexity': 15,
                'line_count': 40,
                'imports': ['json'],
                'functions': ['helper1', 'helper2'],
                'classes': ['Helper']
            }
        }
    }
    
    # Create 3D visualization
    result = viz_3d.create_3d_visualization(sample_data)
    
    print("3D visualization created!")
    print(f"Voxels: {result['metrics']['voxel_count']}")
    print(f"Triangles: {result['metrics']['performance_metrics']['triangle_count']}")
    
    # Export for web
    web_export = viz_3d.export_for_web("./3d_output")
    print(f"Web export saved to: {web_export['html_viewer']}")
