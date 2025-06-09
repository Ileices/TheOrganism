#!/usr/bin/env python3
"""
ADVANCED RBY SPECTRAL COMPRESSION ENGINE
=======================================

Complete implementation of RBY spectral compression with fractal binning
as specified in the weirdAI.md framework.

Features:
- 3^n fractal level progression (3, 9, 27, 81, 243, 729...)  
- Advanced pixel position determination using space-filling curves
- Bit depth and storage unit optimization
- White/black fill temporal markers for absularity/time tracking
- Complete color spectrum encoding from code/data to RBY triplets
- Universal substrate for memory, code, and intelligence storage

Author: Digital Organism Development Team
Version: 1.0.0 (Revolutionary Spectral Compression)
"""

import numpy as np
import math
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from PIL import Image, ImageDraw
import json
import pickle
from pathlib import Path

# Import PTAIE mapping if available
try:
    from visual_dna_encoder import VisualDNAEncoder
    PTAIE_AVAILABLE = True
except ImportError:
    PTAIE_AVAILABLE = False


class FractalLevel(Enum):
    """Fractal levels for RBY compression (3^n progression)"""
    LEVEL_1 = 3       # 3 bins
    LEVEL_2 = 9       # 9 bins
    LEVEL_3 = 27      # 27 bins
    LEVEL_4 = 81      # 81 bins
    LEVEL_5 = 243     # 243 bins
    LEVEL_6 = 729     # 729 bins
    LEVEL_7 = 2187    # 2187 bins
    LEVEL_8 = 6561    # 6561 bins
    LEVEL_9 = 19683   # 19683 bins
    LEVEL_10 = 59049  # 59049 bins
    LEVEL_11 = 177147 # 177147 bins
    LEVEL_12 = 531441 # 531441 bins


class FillType(Enum):
    """Fill types for temporal/absularity marking"""
    WHITE = "white"     # Early expansion, potential space
    BLACK = "black"     # Late expansion, saturated space
    GRAY = "gray"       # Transitional state
    DATA = "data"       # Occupied by actual data


class BitDepth(Enum):
    """Bit depth options for color precision"""
    BIT_8 = 8     # Standard PNG (24-bit RGB)
    BIT_16 = 16   # High precision (48-bit RGB)
    BIT_32 = 32   # Scientific precision (96-bit RGB)


@dataclass
class RBYPixel:
    """Single RBY pixel with metadata"""
    red: float
    blue: float  
    yellow: float
    position: Tuple[int, int]
    fractal_level: int
    fill_type: FillType = FillType.DATA
    temporal_marker: float = field(default_factory=time.time)
    data_source: Optional[str] = None
    compression_generation: int = 0
    
    def to_rgb(self, bit_depth: BitDepth = BitDepth.BIT_8) -> Tuple[int, int, int]:
        """Convert RBY to RGB values"""
        max_val = (2 ** bit_depth.value) - 1
        
        # Convert RBY to RGB (simplified mapping)
        r = int(self.red * max_val)
        g = int(((self.blue + self.yellow) / 2) * max_val)  # Green from B+Y average
        b = int(self.blue * max_val)
        
        return (r, g, b)
    
    def normalize(self):
        """Normalize RBY values to sum to 1.0"""
        total = self.red + self.blue + self.yellow
        if total > 0:
            self.red /= total
            self.blue /= total
            self.yellow /= total


@dataclass
class SpectralCompressionResult:
    """Result of spectral compression operation"""
    pixels: List[RBYPixel]
    fractal_level: int
    grid_dimensions: Tuple[int, int]
    compression_ratio: float
    fill_statistics: Dict[FillType, int]
    temporal_span: Tuple[float, float]  # (earliest, latest) timestamps
    total_data_units: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class HilbertCurveGenerator:
    """
    Generates Hilbert space-filling curves for optimal pixel positioning
    Preserves locality - similar data stays spatially close
    """
    
    @staticmethod
    def hilbert_curve(n: int) -> List[Tuple[int, int]]:
        """
        Generate Hilbert curve coordinates for n x n grid
        
        Args:
            n: Grid size (should be power of 2)
            
        Returns:
            List of (x, y) coordinates in Hilbert curve order
        """
        if n == 1:
            return [(0, 0)]
        
        # Ensure n is power of 2
        power = int(math.log2(n))
        if 2**power != n:
            power += 1
            n = 2**power
        
        positions = []
        
        def hilbert_recursive(n, x, y, xi, xj, yi, yj):
            if n <= 0:
                return
            if n == 1:
                positions.append((x + (xi + yi) // 2, y + (xj + yj) // 2))
                return
            
            n //= 2
            hilbert_recursive(n, x, y, yi // 2, yj // 2, xi // 2, xj // 2)
            hilbert_recursive(n, x + xi // 2, y + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2)
            hilbert_recursive(n, x + xi // 2 + yi // 2, y + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2, yj // 2)
            hilbert_recursive(n, x + xi // 2 + yi, y + xj // 2 + yj, -yi // 2, -yj // 2, -xi // 2, -xj // 2)
        
        hilbert_recursive(n, 0, 0, n, 0, 0, n)
        return positions[:n*n]


class PTAIEMapping:
    """
    Periodic Table of AI Elements (PTAIE) mapping for RBY conversion
    Enhanced version with comprehensive character support
    """
    
    def __init__(self):
        self.rby_mapping = self._initialize_mapping()
        
    def _initialize_mapping(self) -> Dict[str, Tuple[float, float, float]]:
        """Initialize comprehensive PTAIE RBY mapping"""
        
        if PTAIE_AVAILABLE:
            # Use existing visual DNA encoder mapping
            encoder = VisualDNAEncoder()
            return {k: tuple(v) for k, v in encoder.rby_mapping.items()}
        
        # Fallback mapping based on specification
        mapping = {}
        
        # Letters (A-Z)
        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            # Generate RBY based on character properties
            pos = i / 25.0  # Position in alphabet (0-1)
            
            # Different patterns for different character types
            if char in "AEIOU":  # Vowels
                red = 0.5 + pos * 0.3
                blue = 0.2 + pos * 0.2
                yellow = 0.3 - pos * 0.1
            else:  # Consonants
                red = 0.3 + pos * 0.2
                blue = 0.4 + pos * 0.3
                yellow = 0.3 + pos * 0.2
                
            # Normalize
            total = red + blue + yellow
            mapping[char] = (red/total, blue/total, yellow/total)
            mapping[char.lower()] = mapping[char]  # Add lowercase
        
        # Numbers (0-9)
        for i, char in enumerate("0123456789"):
            pos = i / 9.0
            red = 0.33 + pos * 0.1
            blue = 0.33 + pos * 0.05
            yellow = 0.34 - pos * 0.15
            total = red + blue + yellow
            mapping[char] = (red/total, blue/total, yellow/total)
        
        # Special characters
        special_chars = {
            ' ': (0.2, 0.2, 0.6),    # Space - high yellow
            '\n': (0.1, 0.1, 0.8),   # Newline - very high yellow  
            '\t': (0.15, 0.15, 0.7), # Tab - high yellow
            '.': (0.4, 0.3, 0.3),    # Period
            ',': (0.35, 0.35, 0.3),  # Comma
            ';': (0.4, 0.35, 0.25),  # Semicolon
            ':': (0.45, 0.3, 0.25),  # Colon
            '=': (0.3, 0.5, 0.2),    # Equals - high blue
            '+': (0.2, 0.3, 0.5),    # Plus - high yellow
            '-': (0.4, 0.4, 0.2),    # Minus
            '*': (0.3, 0.2, 0.5),    # Asterisk
            '/': (0.35, 0.25, 0.4),  # Slash
            '\\': (0.4, 0.25, 0.35), # Backslash
            '(': (0.25, 0.45, 0.3),  # Open paren
            ')': (0.3, 0.45, 0.25),  # Close paren
            '[': (0.2, 0.5, 0.3),    # Open bracket
            ']': (0.3, 0.5, 0.2),    # Close bracket
            '{': (0.15, 0.55, 0.3),  # Open brace
            '}': (0.3, 0.55, 0.15),  # Close brace
            '"': (0.4, 0.2, 0.4),    # Quote
            "'": (0.45, 0.15, 0.4),  # Apostrophe
            '`': (0.35, 0.25, 0.4),  # Backtick
            '!': (0.6, 0.2, 0.2),    # Exclamation - high red
            '?': (0.2, 0.6, 0.2),    # Question - high blue
            '@': (0.3, 0.3, 0.4),    # At symbol
            '#': (0.25, 0.35, 0.4),  # Hash
            '$': (0.4, 0.3, 0.3),    # Dollar
            '%': (0.35, 0.35, 0.3),  # Percent
            '^': (0.2, 0.2, 0.6),    # Caret - high yellow
            '&': (0.4, 0.35, 0.25),  # Ampersand
            '|': (0.3, 0.4, 0.3),    # Pipe
            '~': (0.25, 0.25, 0.5),  # Tilde
            '<': (0.2, 0.5, 0.3),    # Less than
            '>': (0.3, 0.5, 0.2),    # Greater than
        }
        
        mapping.update(special_chars)
        
        return mapping
    
    def char_to_rby(self, char: str) -> Tuple[float, float, float]:
        """Convert character to RBY triplet"""
        if char in self.rby_mapping:
            return self.rby_mapping[char]
        
        # Fallback for unknown characters
        char_code = ord(char) % 256
        red = (char_code % 85) / 84.0 * 0.6 + 0.2
        blue = ((char_code // 85) % 85) / 84.0 * 0.6 + 0.2  
        yellow = ((char_code // (85*85)) % 85) / 84.0 * 0.6 + 0.2
        
        total = red + blue + yellow
        return (red/total, blue/total, yellow/total)


class AdvancedRBYSpectralCompressor:
    """
    Advanced RBY Spectral Compression Engine
    
    Implements complete fractal binning with 3^n progression,
    advanced pixel positioning, and temporal absularity marking
    """
    
    def __init__(self, bit_depth: BitDepth = BitDepth.BIT_16):
        self.bit_depth = bit_depth
        self.ptaie_mapping = PTAIEMapping()
        self.hilbert_generator = HilbertCurveGenerator()
        self.compression_history: List[SpectralCompressionResult] = []
        
    def compress_data_to_spectrum(self, 
                                data: Union[str, bytes, List[Any]], 
                                expansion_stage: str = "early",
                                preserve_locality: bool = True) -> SpectralCompressionResult:
        """
        Compress data to RBY spectral representation
        
        Args:
            data: Data to compress (string, bytes, or list)
            expansion_stage: "early", "mid", "late", "absularity"
            preserve_locality: Use Hilbert curve for spatial locality
            
        Returns:
            Complete spectral compression result
        """
        
        print(f"🎨 Compressing data to RBY spectrum...")
        print(f"   Data size: {len(data)} units")
        print(f"   Expansion stage: {expansion_stage}")
        
        # Convert data to RBY pixels
        rby_pixels = self._convert_data_to_rby(data)
        
        # Determine optimal fractal level
        fractal_level = self._get_optimal_fractal_level(len(rby_pixels))
        grid_size = int(math.sqrt(fractal_level))
        
        print(f"   Fractal level: {fractal_level} ({grid_size}x{grid_size})")
        
        # Generate pixel positions
        if preserve_locality and grid_size > 1:
            positions = self._generate_hilbert_positions(grid_size, fractal_level)
        else:
            positions = self._generate_linear_positions(grid_size, fractal_level)
        
        # Assign positions to pixels
        positioned_pixels = []
        for i, pixel in enumerate(rby_pixels):
            if i < len(positions):
                pixel.position = positions[i]
                positioned_pixels.append(pixel)
        
        # Fill remaining positions with temporal markers
        fill_pixels = self._generate_fill_pixels(
            positions[len(rby_pixels):],
            fractal_level,
            expansion_stage
        )
        
        all_pixels = positioned_pixels + fill_pixels
        
        # Calculate statistics
        fill_stats = {}
        for fill_type in FillType:
            fill_stats[fill_type] = sum(1 for p in all_pixels if p.fill_type == fill_type)
        
        temporal_span = (
            min(p.temporal_marker for p in all_pixels),
            max(p.temporal_marker for p in all_pixels)
        )
        
        compression_ratio = len(data) / fractal_level
        
        result = SpectralCompressionResult(
            pixels=all_pixels,
            fractal_level=fractal_level,
            grid_dimensions=(grid_size, grid_size),
            compression_ratio=compression_ratio,
            fill_statistics=fill_stats,
            temporal_span=temporal_span,
            total_data_units=len(data),
            metadata={
                'expansion_stage': expansion_stage,
                'preserve_locality': preserve_locality,
                'bit_depth': self.bit_depth.value
            }
        )
        
        self.compression_history.append(result)
        
        print(f"   ✅ Compression complete")
        print(f"      Pixels: {len(all_pixels)}")
        print(f"      Data pixels: {fill_stats.get(FillType.DATA, 0)}")
        print(f"      White fill: {fill_stats.get(FillType.WHITE, 0)}")
        print(f"      Black fill: {fill_stats.get(FillType.BLACK, 0)}")
        
        return result
    
    def _convert_data_to_rby(self, data: Union[str, bytes, List[Any]]) -> List[RBYPixel]:
        """Convert data units to RBY pixels"""
        
        pixels = []
        
        if isinstance(data, str):
            # Character-by-character conversion
            for i, char in enumerate(data):
                red, blue, yellow = self.ptaie_mapping.char_to_rby(char)
                pixel = RBYPixel(
                    red=red,
                    blue=blue,
                    yellow=yellow,
                    position=(0, 0),  # Will be set later
                    fractal_level=0,   # Will be set later
                    data_source=f"char_{i}:{char}"
                )
                pixels.append(pixel)
                
        elif isinstance(data, bytes):
            # Byte-by-byte conversion
            for i, byte_val in enumerate(data):
                char = chr(byte_val) if byte_val < 128 else f"\\x{byte_val:02x}"
                red, blue, yellow = self.ptaie_mapping.char_to_rby(char)
                pixel = RBYPixel(
                    red=red,
                    blue=blue,
                    yellow=yellow,
                    position=(0, 0),
                    fractal_level=0,
                    data_source=f"byte_{i}:{byte_val}"
                )
                pixels.append(pixel)
                
        elif isinstance(data, list):
            # List item conversion
            for i, item in enumerate(data):
                # Convert item to string representation
                item_str = str(item)
                # Use hash for consistent RBY generation
                item_hash = hashlib.sha256(item_str.encode()).hexdigest()
                
                red = int(item_hash[:8], 16) / (2**32 - 1)
                blue = int(item_hash[8:16], 16) / (2**32 - 1)
                yellow = int(item_hash[16:24], 16) / (2**32 - 1)
                
                # Normalize
                total = red + blue + yellow
                if total > 0:
                    red /= total
                    blue /= total
                    yellow /= total
                
                pixel = RBYPixel(
                    red=red,
                    blue=blue,
                    yellow=yellow,
                    position=(0, 0),
                    fractal_level=0,
                    data_source=f"item_{i}:{type(item).__name__}"
                )
                pixels.append(pixel)
        
        return pixels
    
    def _get_optimal_fractal_level(self, data_size: int) -> int:
        """Get optimal fractal level for data size"""
        
        fractal_levels = [level.value for level in FractalLevel]
        
        for level in fractal_levels:
            if level >= data_size:
                return level
        
        # If data exceeds all predefined levels, calculate next 3^n level
        n = len(fractal_levels) + 1
        while 3**n < data_size:
            n += 1
        
        return 3**n
    
    def _generate_hilbert_positions(self, grid_size: int, total_positions: int) -> List[Tuple[int, int]]:
        """Generate positions using Hilbert curve for locality preservation"""
        
        # Ensure grid_size is power of 2 for Hilbert curve
        hilbert_size = 1
        while hilbert_size < grid_size:
            hilbert_size *= 2
        
        # Generate Hilbert curve
        hilbert_positions = self.hilbert_generator.hilbert_curve(hilbert_size)
        
        # Filter to actual grid size and limit to total_positions
        valid_positions = [
            pos for pos in hilbert_positions 
            if pos[0] < grid_size and pos[1] < grid_size
        ]
        
        # Extend with linear positions if needed
        while len(valid_positions) < total_positions:
            y = len(valid_positions) // grid_size
            x = len(valid_positions) % grid_size
            if y < grid_size:
                valid_positions.append((x, y))
            else:
                break
        
        return valid_positions[:total_positions]
    
    def _generate_linear_positions(self, grid_size: int, total_positions: int) -> List[Tuple[int, int]]:
        """Generate positions in linear order"""
        
        positions = []
        for i in range(total_positions):
            y = i // grid_size
            x = i % grid_size
            if y < grid_size:
                positions.append((x, y))
            else:
                # Extend grid if needed
                extended_size = int(math.sqrt(total_positions)) + 1
                y = i // extended_size
                x = i % extended_size
                positions.append((x, y))
        
        return positions
    
    def _generate_fill_pixels(self, 
                            positions: List[Tuple[int, int]], 
                            fractal_level: int,
                            expansion_stage: str) -> List[RBYPixel]:
        """Generate fill pixels for empty positions"""
        
        fill_pixels = []
        current_time = time.time()
        
        # Determine fill type based on expansion stage
        if expansion_stage in ["early", "mid"]:
            primary_fill = FillType.WHITE
            secondary_fill = FillType.GRAY
        elif expansion_stage == "late":
            primary_fill = FillType.GRAY
            secondary_fill = FillType.BLACK
        else:  # absularity
            primary_fill = FillType.BLACK
            secondary_fill = FillType.GRAY
        
        for i, position in enumerate(positions):
            # Alternate between fill types for visual pattern
            if i % 3 == 0:
                fill_type = primary_fill
            elif i % 3 == 1:
                fill_type = secondary_fill
            else:
                fill_type = FillType.WHITE if primary_fill == FillType.BLACK else FillType.BLACK
            
            # Generate RBY values based on fill type
            if fill_type == FillType.WHITE:
                red, blue, yellow = 0.9, 0.9, 0.9
            elif fill_type == FillType.BLACK:
                red, blue, yellow = 0.1, 0.1, 0.1
            else:  # GRAY
                red, blue, yellow = 0.5, 0.5, 0.5
            
            pixel = RBYPixel(
                red=red,
                blue=blue,
                yellow=yellow,
                position=position,
                fractal_level=fractal_level,
                fill_type=fill_type,
                temporal_marker=current_time + i * 0.001,  # Slight time variation
                data_source=f"fill_{fill_type.value}"
            )
            
            fill_pixels.append(pixel)
        
        return fill_pixels
    
    def render_spectrum_to_image(self, result: SpectralCompressionResult, output_path: str):
        """Render spectral compression result to PNG image"""
        
        grid_width, grid_height = result.grid_dimensions
        
        # Create image
        img = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))
        pixels = img.load()
        
        # Set pixel colors
        for pixel in result.pixels:
            x, y = pixel.position
            if 0 <= x < grid_width and 0 <= y < grid_height:
                rgb = pixel.to_rgb(self.bit_depth)
                pixels[x, y] = rgb
        
        # Save image
        img.save(output_path)
        
        print(f"📸 Spectrum rendered to: {output_path}")
        print(f"   Dimensions: {grid_width}x{grid_height}")
        print(f"   Bit depth: {self.bit_depth.value}-bit per channel")
    
    def save_compression_data(self, result: SpectralCompressionResult, output_path: str):
        """Save complete compression data for reconstruction"""
        
        # Prepare serializable data
        save_data = {
            'metadata': result.metadata,
            'fractal_level': result.fractal_level,
            'grid_dimensions': result.grid_dimensions,
            'compression_ratio': result.compression_ratio,
            'fill_statistics': {k.value: v for k, v in result.fill_statistics.items()},
            'temporal_span': result.temporal_span,
            'total_data_units': result.total_data_units,
            'pixels': []
        }
        
        # Serialize pixels
        for pixel in result.pixels:
            pixel_data = {
                'red': pixel.red,
                'blue': pixel.blue,
                'yellow': pixel.yellow,
                'position': pixel.position,
                'fractal_level': pixel.fractal_level,
                'fill_type': pixel.fill_type.value,
                'temporal_marker': pixel.temporal_marker,
                'data_source': pixel.data_source,
                'compression_generation': pixel.compression_generation
            }
            save_data['pixels'].append(pixel_data)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Compression data saved to: {output_path}")
    
    def load_compression_data(self, input_path: str) -> SpectralCompressionResult:
        """Load compression data from file"""
        
        with open(input_path, 'r') as f:
            save_data = json.load(f)
        
        # Reconstruct pixels
        pixels = []
        for pixel_data in save_data['pixels']:
            pixel = RBYPixel(
                red=pixel_data['red'],
                blue=pixel_data['blue'],
                yellow=pixel_data['yellow'],
                position=tuple(pixel_data['position']),
                fractal_level=pixel_data['fractal_level'],
                fill_type=FillType(pixel_data['fill_type']),
                temporal_marker=pixel_data['temporal_marker'],
                data_source=pixel_data['data_source'],
                compression_generation=pixel_data['compression_generation']
            )
            pixels.append(pixel)
        
        # Reconstruct fill statistics
        fill_stats = {FillType(k): v for k, v in save_data['fill_statistics'].items()}
        
        result = SpectralCompressionResult(
            pixels=pixels,
            fractal_level=save_data['fractal_level'],
            grid_dimensions=tuple(save_data['grid_dimensions']),
            compression_ratio=save_data['compression_ratio'],
            fill_statistics=fill_stats,
            temporal_span=tuple(save_data['temporal_span']),
            total_data_units=save_data['total_data_units'],
            metadata=save_data['metadata']
        )
        
        print(f"📂 Compression data loaded from: {input_path}")
        
        return result


# Utility functions for easy integration

def compress_text_to_rby_spectrum(text: str, output_dir: str = ".", expansion_stage: str = "early") -> str:
    """
    Compress text to RBY spectrum with both image and data output
    
    Args:
        text: Text to compress
        output_dir: Directory for output files
        expansion_stage: Expansion stage for fill logic
        
    Returns:
        Base filename of generated files
    """
    
    compressor = AdvancedRBYSpectralCompressor()
    result = compressor.compress_data_to_spectrum(text, expansion_stage)
    
    # Generate output filenames
    timestamp = int(time.time())
    base_name = f"rby_spectrum_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save image
    img_path = output_dir / f"{base_name}.png"
    compressor.render_spectrum_to_image(result, str(img_path))
    
    # Save data
    data_path = output_dir / f"{base_name}.json" 
    compressor.save_compression_data(result, str(data_path))
    
    return base_name


def compress_file_to_rby_spectrum(file_path: str, output_dir: str = ".", expansion_stage: str = "early") -> str:
    """
    Compress file contents to RBY spectrum
    
    Args:
        file_path: Path to file to compress
        output_dir: Directory for output files  
        expansion_stage: Expansion stage for fill logic
        
    Returns:
        Base filename of generated files
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_name = Path(file_path).stem
    compressor = AdvancedRBYSpectralCompressor()
    result = compressor.compress_data_to_spectrum(content, expansion_stage)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save image
    img_path = output_dir / f"{file_name}_spectrum.png"
    compressor.render_spectrum_to_image(result, str(img_path))
    
    # Save data
    data_path = output_dir / f"{file_name}_spectrum.json"
    compressor.save_compression_data(result, str(data_path))
    
    print(f"📁 File compressed: {file_path} → {file_name}_spectrum.*")
    
    return f"{file_name}_spectrum"


if __name__ == "__main__":
    # Example usage
    print("🎨 Advanced RBY Spectral Compression Engine")
    print("=" * 50)
    
    # Test with sample text
    sample_text = "The cow jumped over the moon"
    print(f"\n🧪 Testing with: '{sample_text}'")
    
    result_name = compress_text_to_rby_spectrum(sample_text, "test_output", "early")
    print(f"✅ Generated: {result_name}.*")
    
    print("\n💡 System ready for advanced RBY spectral compression")
    print("   Use compress_text_to_rby_spectrum() or compress_file_to_rby_spectrum()")
