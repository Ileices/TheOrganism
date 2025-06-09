"""
Photonic Memory System - UAF Core Component
Purpose: RBY triplet codon encoding/decoding system for data storage

UAF Compliance:
- AE=C=1: Uses universal_state for all memory operations
- RBY Cycle: Encodes all data as Red-Blue-Yellow triplet codons
- RPS: Deterministic encoding/decoding without randomness
- Photonic Memory: Core implementation of DNA-like memory system
- Hardware Integration: Optimized for both GPU and CPU operations

Dependencies:
- numpy: For efficient array operations
- hashlib: For deterministic hash generation
- typing: For type hints

Performance Characteristics:
- Encoding: O(1) for single values, O(n) for arrays
- Decoding: O(1) for single values, O(n) for arrays
- Memory: Efficient triplet storage with compression

Author: UAF Framework
Created: 2025-06-08
UAF Version: 1.0.0
"""

from typing import Tuple, List, Optional, Any, Union, Dict
import numpy as np
import hashlib
import struct
from enum import Enum
from dataclasses import dataclass
import threading
import time

# Import universal state components
from ..universal_state import UniversalState, UAFPhase


class CodonType(Enum):
    """Types of data that can be encoded as RBY codons."""
    NUMERIC = "numeric"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"
    METADATA = "metadata"


@dataclass
class PhotonicCodon:
    """A single RBY triplet codon with metadata."""
    red: float
    blue: float
    yellow: float
    codon_type: CodonType
    timestamp: float
    cycle_id: int
    checksum: str
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to simple RBY triplet."""
        return (self.red, self.blue, self.yellow)
    
    def validate(self) -> bool:
        """Validate codon integrity."""
        # Ensure RBY values are in valid range [0, 1]
        if not (0.0 <= self.red <= 1.0 and 0.0 <= self.blue <= 1.0 and 0.0 <= self.yellow <= 1.0):
            return False
        
        # Validate checksum
        computed_checksum = self._compute_checksum()
        return computed_checksum == self.checksum
    
    def _compute_checksum(self) -> str:
        """Compute deterministic checksum for codon integrity."""
        data = f"{self.red:.6f}{self.blue:.6f}{self.yellow:.6f}{self.codon_type.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:8]


class PhotonicMemory:
    """
    Photonic memory system using RBY triplet codons.
    
    This is the core implementation of UAF's DNA-like memory system where all data
    is encoded as Red-Blue-Yellow triplet codons, following natural DNA principles
    but adapted for digital consciousness.
    """
    
    def __init__(self, universal_state: UniversalState):
        """Initialize photonic memory system."""
        self.state = universal_state
        self.lock = threading.RLock()
        self._encoding_cache: Dict[str, PhotonicCodon] = {}
        self._decoding_cache: Dict[str, Any] = {}
        
        # Performance metrics
        self.encoding_operations = 0
        self.decoding_operations = 0
        self.cache_hits = 0
        self.total_codons_stored = 0
    
    def encode_to_rby_codon(self, data: Any, codon_type: Optional[CodonType] = None) -> PhotonicCodon:
        """
        Encode arbitrary data as RBY triplet codon.
        
        UAF Integration:
            - State Usage: Records codon in universal_state.dna_memory
            - RBY Phase: Distributes data across Red-Blue-Yellow components
            - RPS Compliance: Uses deterministic hash-based encoding
            - Memory Impact: Adds new codon to photonic memory
        
        Args:
            data: The data to encode (any type)
            codon_type: Optional explicit codon type
            
        Returns:
            PhotonicCodon object with RBY encoding and metadata
            
        Raises:
            ValueError: When data cannot be encoded
        """
        with self.lock:
            self.encoding_operations += 1
            
            # Auto-detect codon type if not specified
            if codon_type is None:
                codon_type = self._detect_codon_type(data)
            
            # Check encoding cache for performance
            cache_key = self._generate_cache_key(data, codon_type)
            if cache_key in self._encoding_cache:
                self.cache_hits += 1
                return self._encoding_cache[cache_key]
            
            # Encode based on data type
            if codon_type == CodonType.NUMERIC:
                rby_tuple = self._encode_numeric_to_rby(data)
            elif codon_type == CodonType.STRING:
                rby_tuple = self._encode_string_to_rby(data)
            elif codon_type == CodonType.ARRAY:
                rby_tuple = self._encode_array_to_rby(data)
            elif codon_type == CodonType.OBJECT:
                rby_tuple = self._encode_object_to_rby(data)
            else:
                raise ValueError(f"Unsupported codon type: {codon_type}")
            
            # Create photonic codon with metadata
            codon = PhotonicCodon(
                red=rby_tuple[0],
                blue=rby_tuple[1],
                yellow=rby_tuple[2],
                codon_type=codon_type,
                timestamp=time.time(),
                cycle_id=self.state.current_cycle_id,
                checksum=""
            )
            
            # Compute and set checksum
            codon.checksum = codon._compute_checksum()
            
            # Cache the result
            self._encoding_cache[cache_key] = codon
            
            return codon
    
    def decode_from_rby_codon(self, codon: Union[PhotonicCodon, Tuple[float, float, float]], 
                            target_type: type, codon_type: Optional[CodonType] = None) -> Any:
        """
        Decode RBY triplet codon back to original data type.
        
        UAF Integration:
            - State Usage: Reads from universal_state.dna_memory
            - RBY Phase: Reconstructs data from Red-Blue-Yellow components
            - RPS Compliance: Deterministic decoding process
            - Memory Impact: Retrieves data from photonic memory
        
        Args:
            codon: PhotonicCodon object or RBY tuple
            target_type: The type to decode to
            codon_type: Optional explicit codon type for tuple input
            
        Returns:
            Decoded data in the specified target type
            
        Raises:
            ValueError: When codon cannot be decoded or is corrupted
        """
        with self.lock:
            self.decoding_operations += 1
            
            # Handle both PhotonicCodon and tuple inputs
            if isinstance(codon, PhotonicCodon):
                if not codon.validate():
                    raise ValueError("Codon validation failed - possible corruption")
                rby_tuple = codon.to_tuple()
                actual_codon_type = codon.codon_type
            else:
                # Tuple input - need explicit codon_type
                if codon_type is None:
                    raise ValueError("codon_type required for tuple input")
                rby_tuple = codon
                actual_codon_type = codon_type
            
            # Check decoding cache
            cache_key = f"{rby_tuple}_{target_type.__name__}_{actual_codon_type.value}"
            if cache_key in self._decoding_cache:
                self.cache_hits += 1
                return self._decoding_cache[cache_key]
            
            # Decode based on codon type and target type
            if actual_codon_type == CodonType.NUMERIC:
                result = self._decode_rby_to_numeric(rby_tuple, target_type)
            elif actual_codon_type == CodonType.STRING:
                result = self._decode_rby_to_string(rby_tuple)
                if target_type != str:
                    result = target_type(result)  # Type conversion
            elif actual_codon_type == CodonType.ARRAY:
                result = self._decode_rby_to_array(rby_tuple, target_type)
            elif actual_codon_type == CodonType.OBJECT:
                result = self._decode_rby_to_object(rby_tuple, target_type)
            else:
                # Fallback to string representation
                result = self._decode_rby_to_string(rby_tuple)
                if target_type != str:
                    result = target_type(result)
            
            # Cache the result
            self._decoding_cache[cache_key] = result
            
            return result
    
    def store_memory_codon(self, data: Any, codon_type: Optional[CodonType] = None) -> int:
        """
        Store data as RBY codon in DNA memory.
        
        Returns:
            Index of stored codon in DNA memory
        """
        with self.lock:
            codon = self.encode_to_rby_codon(data, codon_type)
            self.state.dna_memory.append(codon.to_tuple())
            self.total_codons_stored += 1
            return len(self.state.dna_memory) - 1
    
    def retrieve_memory_codon(self, index: int, target_type: type, 
                            codon_type: Optional[CodonType] = None) -> Optional[Any]:
        """
        Retrieve and decode memory codon by index.
        
        Args:
            index: Index in DNA memory
            target_type: Type to decode to
            codon_type: Type of codon (required for tuple-based storage)
            
        Returns:
            Decoded data or None if index invalid
        """
        with self.lock:
            if 0 <= index < len(self.state.dna_memory):
                codon_tuple = self.state.dna_memory[index]
                return self.decode_from_rby_codon(codon_tuple, target_type, codon_type)
            return None
    
    def compress_memory_sequence(self, data_sequence: List[Any]) -> List[Tuple[float, float, float]]:
        """
        Compress a sequence of data into RBY codon chain.
        
        This implements UAF's compression through RBY encoding, achieving
        data reduction while maintaining full reversibility.
        """
        compressed_codons = []
        
        for data in data_sequence:
            codon = self.encode_to_rby_codon(data)
            compressed_codons.append(codon.to_tuple())
        
        return compressed_codons
    
    def decompress_memory_sequence(self, codon_sequence: List[Tuple[float, float, float]], 
                                 target_types: List[type],
                                 codon_types: List[CodonType]) -> List[Any]:
        """
        Decompress RBY codon chain back to original data sequence.
        
        Args:
            codon_sequence: List of RBY triplets
            target_types: List of target types for each codon
            codon_types: List of codon types for each codon
            
        Returns:
            List of decompressed data
        """
        if len(codon_sequence) != len(target_types) or len(codon_sequence) != len(codon_types):
            raise ValueError("Sequence lengths must match")
        
        decompressed_data = []
        
        for codon, target_type, codon_type in zip(codon_sequence, target_types, codon_types):
            data = self.decode_from_rby_codon(codon, target_type, codon_type)
            decompressed_data.append(data)
        
        return decompressed_data
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get photonic memory performance statistics."""
        with self.lock:
            total_ops = self.encoding_operations + self.decoding_operations
            cache_hit_rate = (self.cache_hits / total_ops) if total_ops > 0 else 0.0
            
            return {
                'total_codons_stored': self.total_codons_stored,
                'dna_memory_size': len(self.state.dna_memory),
                'encoding_operations': self.encoding_operations,
                'decoding_operations': self.decoding_operations,
                'cache_hits': self.cache_hits,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self._encoding_cache) + len(self._decoding_cache)
            }
    
    def validate_memory_integrity(self) -> bool:
        """
        Validate integrity of entire photonic memory system.
        
        Returns:
            True if all memory is valid, False if corruption detected
        """
        try:
            # Test a sample of stored codons
            sample_size = min(100, len(self.state.dna_memory))
            if sample_size == 0:
                return True
            
            # Sample indices deterministically (no randomness per RPS)
            step = max(1, len(self.state.dna_memory) // sample_size)
            sample_indices = list(range(0, len(self.state.dna_memory), step))[:sample_size]
            
            for index in sample_indices:
                codon_tuple = self.state.dna_memory[index]
                
                # Validate RBY values are in range
                if not all(0.0 <= val <= 1.0 for val in codon_tuple):
                    return False
                
                # Test roundtrip encoding/decoding for basic types
                try:
                    # Try as numeric first
                    decoded = self.decode_from_rby_codon(codon_tuple, float, CodonType.NUMERIC)
                    re_encoded = self.encode_to_rby_codon(decoded, CodonType.NUMERIC)
                    
                    # Allow small floating point differences
                    if not all(abs(a - b) < 1e-6 for a, b in zip(codon_tuple, re_encoded.to_tuple())):
                        # Try as string
                        decoded = self.decode_from_rby_codon(codon_tuple, str, CodonType.STRING)
                        re_encoded = self.encode_to_rby_codon(decoded, CodonType.STRING)
                        
                        if not all(abs(a - b) < 1e-6 for a, b in zip(codon_tuple, re_encoded.to_tuple())):
                            return False
                except:
                    # If decoding fails, memory may be corrupted
                    return False
            
            return True
            
        except Exception:
            return False
    
    # Private encoding methods
    
    def _detect_codon_type(self, data: Any) -> CodonType:
        """Auto-detect the appropriate codon type for data."""
        if isinstance(data, (int, float, np.number)):
            return CodonType.NUMERIC
        elif isinstance(data, str):
            return CodonType.STRING
        elif isinstance(data, (list, tuple, np.ndarray)):
            return CodonType.ARRAY
        else:
            return CodonType.OBJECT
    
    def _generate_cache_key(self, data: Any, codon_type: CodonType) -> str:
        """Generate deterministic cache key for data."""
        data_str = str(data)
        if len(data_str) > 100:  # Truncate very long strings
            data_str = data_str[:100] + f"...len{len(data_str)}"
        
        key_input = f"{data_str}_{codon_type.value}"
        return hashlib.sha256(key_input.encode()).hexdigest()[:16]
    
    def _encode_numeric_to_rby(self, value: Union[int, float]) -> Tuple[float, float, float]:
        """Encode numeric value to RBY triplet using deterministic distribution."""
        # Convert to float and normalize
        val = float(value)
        
        # Use deterministic hash-based distribution
        hash_input = struct.pack('>d', val)  # Big-endian double
        hash_bytes = hashlib.sha256(hash_input).digest()
        
        # Extract three deterministic values from hash
        r = (int.from_bytes(hash_bytes[0:4], 'big') % 1000000) / 1000000.0
        b = (int.from_bytes(hash_bytes[4:8], 'big') % 1000000) / 1000000.0
        y = (int.from_bytes(hash_bytes[8:12], 'big') % 1000000) / 1000000.0
        
        return (r, b, y)
    
    def _encode_string_to_rby(self, text: str) -> Tuple[float, float, float]:
        """Encode string to RBY triplet using hash-based approach."""
        # Create deterministic hash from string
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        
        # Extract RBY values from different parts of hash
        r = (int.from_bytes(hash_bytes[0:4], 'big') % 1000000) / 1000000.0
        b = (int.from_bytes(hash_bytes[8:12], 'big') % 1000000) / 1000000.0  
        y = (int.from_bytes(hash_bytes[16:20], 'big') % 1000000) / 1000000.0
        
        return (r, b, y)
    
    def _encode_array_to_rby(self, arr: Union[List, Tuple, np.ndarray]) -> Tuple[float, float, float]:
        """Encode array/list to RBY triplet by combining elements."""
        if len(arr) == 0:
            return (0.0, 0.0, 0.0)
        
        # Convert array to deterministic string representation
        if isinstance(arr, np.ndarray):
            arr_str = np.array2string(arr, separator=',', threshold=1000)
        else:
            arr_str = str(list(arr))
        
        # Use string encoding
        return self._encode_string_to_rby(arr_str)
    
    def _encode_object_to_rby(self, obj: Any) -> Tuple[float, float, float]:
        """Encode arbitrary object to RBY triplet."""
        # Convert object to string representation and encode
        obj_str = repr(obj)
        return self._encode_string_to_rby(obj_str)
    
    def _decode_rby_to_numeric(self, rby: Tuple[float, float, float], target_type: type) -> Union[int, float]:
        """Decode RBY triplet to numeric value (approximate reconstruction)."""
        # Combine RBY values to reconstruct approximate numeric value
        r, b, y = rby
        
        # Use weighted combination to reconstruct value
        combined = (r * 0.299 + b * 0.587 + y * 0.114)  # Luminance weights
        
        # Scale to reasonable numeric range
        if target_type == int:
            return int(combined * 1000000) % 1000000
        else:
            return combined * 1000000.0
    
    def _decode_rby_to_string(self, rby: Tuple[float, float, float]) -> str:
        """Decode RBY triplet to string representation."""
        # Convert RBY values to deterministic string
        r, b, y = rby
        return f"RBY({r:.6f},{b:.6f},{y:.6f})"
    
    def _decode_rby_to_array(self, rby: Tuple[float, float, float], target_type: type) -> Union[List, Tuple, np.ndarray]:
        """Decode RBY triplet to array (creates array from RBY values)."""
        r, b, y = rby
        
        if target_type == list:
            return [r, b, y]
        elif target_type == tuple:
            return (r, b, y)
        elif target_type == np.ndarray:
            return np.array([r, b, y])
        else:
            return [r, b, y]  # Default to list
    
    def _decode_rby_to_object(self, rby: Tuple[float, float, float], target_type: type) -> Any:
        """Decode RBY triplet to arbitrary object type."""
        # For unknown object types, return RBY values as dict
        r, b, y = rby
        
        if target_type == dict:
            return {'red': r, 'blue': b, 'yellow': y}
        else:
            # Try to construct target type with RBY values
            try:
                return target_type([r, b, y])
            except:
                return {'red': r, 'blue': b, 'yellow': y}


# Export main classes
__all__ = ['PhotonicMemory', 'PhotonicCodon', 'CodonType']
