#!/usr/bin/env python3
"""
Twmrto Compression Interpreter - Memory Decay Compression Method
Implementation of the advanced memory decay compression technique based on AE Theory principles
"""

import re
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import pickle

@dataclass
class MemoryDecayStage:
    """Represents a single stage in the memory decay process"""
    stage_number: int
    content: str
    compression_ratio: float
    decay_method: str
    timestamp: float
    rby_vector: Optional[Tuple[float, float, float]] = None

class TwmrtoCompressor:
    """
    Twmrto Memory Decay Compression Engine
    
    Implements the multi-stage memory decay process:
    1. Character-level compression
    2. Word-level reduction
    3. Concept extraction
    4. Glyph formation
    5. Final symbol creation
    """
    
    def __init__(self, base_decay_rate: float = 0.15):
        self.base_decay_rate = base_decay_rate
        self.compression_history = []
        self.glyph_registry = {}
        self.reconstruction_keys = {}
        
    def compress_full_cycle(self, text: str, preserve_semantics: bool = True) -> Dict[str, Any]:
        """
        Complete Twmrto compression cycle from original text to final glyph
        
        Args:
            text: Original text to compress
            preserve_semantics: Whether to maintain reconstruction keys
            
        Returns:
            Dictionary with all compression stages and reconstruction data
        """
        if not text.strip():
            return {'error': 'Empty input text'}
        
        original_size = len(text)
        stages = []
        current_text = text.strip()
        
        # Stage 1: Character-level decay (remove redundant characters)
        stage1 = self._apply_character_decay(current_text)
        stages.append(MemoryDecayStage(
            stage_number=1,
            content=stage1,
            compression_ratio=len(stage1) / original_size,
            decay_method="character_removal",
            timestamp=time.time()
        ))
        current_text = stage1
        
        # Stage 2: Word-level simplification
        stage2 = self._apply_word_decay(current_text)
        stages.append(MemoryDecayStage(
            stage_number=2,
            content=stage2,
            compression_ratio=len(stage2) / original_size,
            decay_method="word_simplification", 
            timestamp=time.time()
        ))
        current_text = stage2
        
        # Stage 3: Concept extraction
        stage3 = self._apply_concept_decay(current_text)
        stages.append(MemoryDecayStage(
            stage_number=3,
            content=stage3,
            compression_ratio=len(stage3) / original_size,
            decay_method="concept_extraction",
            timestamp=time.time()
        ))
        current_text = stage3
        
        # Stage 4: Pattern abstraction
        stage4 = self._apply_pattern_decay(current_text)
        stages.append(MemoryDecayStage(
            stage_number=4,
            content=stage4,
            compression_ratio=len(stage4) / original_size,
            decay_method="pattern_abstraction",
            timestamp=time.time()
        ))
        current_text = stage4
        
        # Stage 5: Glyph formation
        stage5 = self._apply_glyph_decay(current_text)
        stages.append(MemoryDecayStage(
            stage_number=5,
            content=stage5,
            compression_ratio=len(stage5) / original_size,
            decay_method="glyph_formation",
            timestamp=time.time()
        ))
        current_text = stage5
        
        # Stage 6: Final symbol (Twmrto-like compression)
        final_glyph = self._create_final_glyph(current_text, text)
        stages.append(MemoryDecayStage(
            stage_number=6,
            content=final_glyph,
            compression_ratio=len(final_glyph) / original_size,
            decay_method="final_symbol",
            timestamp=time.time()
        ))
        
        # Generate RBY vectors for each stage
        for stage in stages:
            stage.rby_vector = self._calculate_rby_vector(stage.content, stage.stage_number)
        
        # Create reconstruction keys if preserving semantics
        reconstruction_key = None
        if preserve_semantics:
            reconstruction_key = self._create_reconstruction_key(text, stages)
            self.reconstruction_keys[final_glyph] = reconstruction_key
        
        # Calculate final compression statistics
        final_compression_ratio = len(final_glyph) / original_size
        compression_efficiency = (1 - final_compression_ratio) * 100
        
        result = {
            'original_text': text,
            'original_size': original_size,
            'final_glyph': final_glyph,
            'final_size': len(final_glyph),
            'compression_ratio': final_compression_ratio,
            'compression_efficiency': compression_efficiency,
            'stages': stages,
            'reconstruction_key': reconstruction_key,
            'timestamp': time.time(),
            'method': 'Twmrto_Memory_Decay'
        }
        
        self.compression_history.append(result)
        return result
    
    def _apply_character_decay(self, text: str) -> str:
        """Stage 1: Character-level decay - remove vowels, spaces, redundancy"""
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove vowels from non-essential words
        words = text.split()
        processed_words = []
        
        # Keep first and last words intact, decay middle words
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1:
                processed_words.append(word)
            else:
                # Remove vowels and duplicate consonants
                decayed = re.sub(r'[aeiouAEIOU]', '', word)
                decayed = re.sub(r'(.)\1+', r'\1', decayed)  # Remove duplicate chars
                if decayed:
                    processed_words.append(decayed)
        
        return ' '.join(processed_words)
    
    def _apply_word_decay(self, text: str) -> str:
        """Stage 2: Word-level decay - remove filler words, simplify"""
        # Common filler words to remove
        filler_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = text.split()
        essential_words = [word for word in words if word.lower() not in filler_words]
        
        # Keep only every other word if still too long
        if len(essential_words) > 4:
            essential_words = essential_words[::2]
        
        return ' '.join(essential_words)
    
    def _apply_concept_decay(self, text: str) -> str:
        """Stage 3: Concept extraction - extract key concepts"""
        words = text.split()
        
        # Extract first letter of each significant word (>2 chars)
        concepts = []
        for word in words:
            if len(word) > 2:
                concepts.append(word[:3])  # Take first 3 characters
            else:
                concepts.append(word)
        
        return ' '.join(concepts)
    
    def _apply_pattern_decay(self, text: str) -> str:
        """Stage 4: Pattern abstraction - create pattern-based compression"""
        # Remove spaces and create consonant-vowel patterns
        no_spaces = text.replace(' ', '')
        
        # Extract pattern: consonants and vowels alternating
        pattern = ''
        for i, char in enumerate(no_spaces):
            if i % 3 == 0:  # Keep every third character
                pattern += char.lower()
        
        return pattern
    
    def _apply_glyph_decay(self, text: str) -> str:
        """Stage 5: Glyph formation - create symbolic representation"""
        # Create abbreviation-like glyph
        if len(text) <= 3:
            return text
        
        # Take first, middle, and last characters
        if len(text) >= 3:
            first = text[0]
            middle = text[len(text)//2] if len(text) > 2 else ''
            last = text[-1]
            return f"{first}{middle}{last}".lower()
        
        return text[:3].lower()
    
    def _create_final_glyph(self, processed_text: str, original_text: str) -> str:
        """Stage 6: Create final symbolic glyph (Twmrto-style)"""
        # Create hash-based glyph for uniqueness
        content_hash = hashlib.md5(original_text.encode()).hexdigest()[:4]
        
        # Combine processed text with hash for unique glyph
        if len(processed_text) <= 3:
            glyph_base = processed_text
        else:
            # Create acronym from original text
            words = original_text.split()
            acronym = ''.join([word[0].lower() for word in words if word][:3])
            glyph_base = acronym if acronym else processed_text[:3]
        
        final_glyph = f"{glyph_base}{content_hash}"
        return final_glyph[:8]  # Limit to 8 characters max
    
    def _calculate_rby_vector(self, content: str, stage: int) -> Tuple[float, float, float]:
        """Calculate RBY vector for content based on AE Theory principles"""
        # Base RBY calculations
        content_length = len(content)
        char_diversity = len(set(content.lower()))
        
        # R (Perception) - decreases as content becomes more abstract
        R = max(0.1, 0.8 - (stage * 0.1) + (char_diversity / max(content_length, 1)) * 0.2)
        
        # B (Cognition) - increases with compression (understanding)
        B = min(0.8, 0.2 + (stage * 0.1) + (char_diversity / max(content_length, 1)) * 0.1)
        
        # Y (Execution) - varies based on content action potential
        Y = 1.0 - R - B
        
        # Normalize to ensure R + B + Y = 1.0
        total = R + B + Y
        return (R/total, B/total, Y/total)
    
    def _create_reconstruction_key(self, original: str, stages: List[MemoryDecayStage]) -> Dict[str, Any]:
        """Create reconstruction key for semantic preservation"""
        return {
            'original_words': original.split(),
            'word_count': len(original.split()),
            'char_count': len(original),
            'key_concepts': self._extract_key_concepts(original),
            'stage_transitions': [(s.stage_number, s.content, s.decay_method) for s in stages],
            'rby_progression': [s.rby_vector for s in stages],
            'semantic_hash': hashlib.sha256(original.encode()).hexdigest()
        }
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key semantic concepts from text"""
        # Simple concept extraction - can be enhanced with NLP
        words = text.split()
        
        # Filter for significant words (>3 chars, not common words)
        common_words = {'the', 'and', 'or', 'but', 'with', 'have', 'this', 'that', 'they', 'them'}
        concepts = [word for word in words if len(word) > 3 and word.lower() not in common_words]
        
        return concepts[:5]  # Limit to top 5 concepts
    
    def decompress(self, glyph: str, reconstruction_key: Optional[Dict[str, Any]] = None) -> str:
        """
        Attempt to decompress a Twmrto glyph back to meaningful text
        
        Args:
            glyph: The compressed glyph to decompress
            reconstruction_key: Optional key for better reconstruction
            
        Returns:
            Decompressed text (may be lossy)
        """
        if not reconstruction_key and glyph in self.reconstruction_keys:
            reconstruction_key = self.reconstruction_keys[glyph]
        
        if not reconstruction_key:
            return f"[RECONSTRUCTED: {glyph}]"  # Minimal reconstruction
        
        # Attempt reconstruction using key concepts
        key_concepts = reconstruction_key.get('key_concepts', [])
        original_words = reconstruction_key.get('original_words', [])
        
        if key_concepts:
            reconstructed = ' '.join(key_concepts)
            return f"[RECONSTRUCTED: {reconstructed}]"
        elif len(original_words) <= 3:
            return ' '.join(original_words)
        else:
            return f"[GLYPH: {glyph} | CONCEPTS: {', '.join(key_concepts[:3])}]"
    
    def analyze_compression_efficiency(self) -> Dict[str, Any]:
        """Analyze compression efficiency across all compressions"""
        if not self.compression_history:
            return {'error': 'No compression history available'}
        
        efficiencies = [comp['compression_efficiency'] for comp in self.compression_history]
        ratios = [comp['compression_ratio'] for comp in self.compression_history]
        
        return {
            'total_compressions': len(self.compression_history),
            'average_efficiency': np.mean(efficiencies),
            'best_efficiency': np.max(efficiencies),
            'worst_efficiency': np.min(efficiencies),
            'average_ratio': np.mean(ratios),
            'total_original_size': sum(comp['original_size'] for comp in self.compression_history),
            'total_compressed_size': sum(comp['final_size'] for comp in self.compression_history)
        }
    
    def save_compression_state(self, filepath: str):
        """Save compression history and reconstruction keys"""
        state = {
            'compression_history': self.compression_history,
            'reconstruction_keys': self.reconstruction_keys,
            'glyph_registry': self.glyph_registry,
            'base_decay_rate': self.base_decay_rate,
            'timestamp': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_compression_state(self, filepath: str):
        """Load compression history and reconstruction keys"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.compression_history = state.get('compression_history', [])
        self.reconstruction_keys = state.get('reconstruction_keys', {})
        self.glyph_registry = state.get('glyph_registry', {})
        self.base_decay_rate = state.get('base_decay_rate', 0.15)

class TwmrtoInterpreter:
    """
    High-level interpreter for Twmrto compression/decompression operations
    Provides standardized interface for the compression method
    """
    
    def __init__(self):
        self.compressor = TwmrtoCompressor()
        self.security_mode = False
        
    def compress_text(self, text: str, security_level: int = 1) -> str:
        """
        Compress text using Twmrto method
        
        Args:
            text: Text to compress
            security_level: 1=normal, 2=enhanced, 3=maximum security
            
        Returns:
            Compressed glyph string
        """
        result = self.compressor.compress_full_cycle(text, preserve_semantics=(security_level < 3))
        
        if security_level >= 2:
            # Add security obfuscation
            glyph = result['final_glyph']
            return self._apply_security_obfuscation(glyph, security_level)
        
        return result['final_glyph']
    
    def compress_codebase(self, code_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Compress an entire codebase using Twmrto method
        
        Args:
            code_files: Dictionary of filename -> code content
            
        Returns:
            Compression results for all files
        """
        results = {}
        
        for filename, code_content in code_files.items():
            # Compress each file
            compression_result = self.compressor.compress_full_cycle(code_content)
            
            results[filename] = {
                'original_size': len(code_content),
                'compressed_glyph': compression_result['final_glyph'],
                'compression_ratio': compression_result['compression_ratio'],
                'rby_vectors': [s.rby_vector for s in compression_result['stages']],
                'reconstruction_possible': compression_result['reconstruction_key'] is not None
            }
        
        return results
    
    def _apply_security_obfuscation(self, glyph: str, level: int) -> str:
        """Apply security obfuscation to glyph"""
        if level == 2:
            # ROT13-style character shifting
            obfuscated = ''.join(chr((ord(c) + 13) % 126) if c.isalnum() else c for c in glyph)
            return obfuscated
        elif level == 3:
            # Hash-based obfuscation
            return hashlib.sha256(glyph.encode()).hexdigest()[:8]
        
        return glyph
    
    def benchmark_performance(self, test_texts: List[str]) -> Dict[str, Any]:
        """Benchmark Twmrto compression performance"""
        start_time = time.time()
        
        results = []
        for text in test_texts:
            result = self.compressor.compress_full_cycle(text)
            results.append(result)
        
        end_time = time.time()
        
        # Calculate statistics
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['final_size'] for r in results)
        avg_compression = np.mean([r['compression_efficiency'] for r in results])
        
        return {
            'processing_time': end_time - start_time,
            'texts_processed': len(test_texts),
            'total_original_size': total_original,
            'total_compressed_size': total_compressed,
            'overall_compression_ratio': total_compressed / total_original,
            'average_compression_efficiency': avg_compression,
            'throughput_chars_per_second': total_original / (end_time - start_time)
        }
    
    def reconstruct_from_glyph(self, glyph: str, use_ai_inference: bool = True) -> Optional[str]:
        """
        Reconstruct original text from Twmrto glyph
        
        Args:
            glyph: The compressed glyph to reconstruct from
            use_ai_inference: Whether to use AI inference for reconstruction
            
        Returns:
            Reconstructed text or None if reconstruction fails
        """
        
        if glyph in self.reconstruction_keys:
            # Direct reconstruction using stored keys
            return self._reconstruct_with_keys(glyph)
        elif use_ai_inference:
            # AI-based reconstruction using pattern recognition
            return self._reconstruct_with_ai(glyph)
        else:
            return None
    
    def _reconstruct_with_keys(self, glyph: str) -> str:
        """Reconstruct using stored reconstruction keys"""
        
        reconstruction_key = self.reconstruction_keys[glyph]
        
        # Start with final glyph and work backwards
        current_text = glyph
        
        # Reverse through each stage
        for stage_info in reversed(reconstruction_key['stage_info']):
            current_text = self._reverse_stage(current_text, stage_info)
        
        return current_text
    
    def _reconstruct_with_ai(self, glyph: str) -> str:
        """Reconstruct using AI inference and pattern matching"""
        
        # Analyze glyph patterns
        glyph_patterns = self._analyze_glyph_patterns(glyph)
        
        # Find similar glyphs in registry
        similar_glyphs = self._find_similar_glyphs(glyph, glyph_patterns)
        
        # Generate reconstruction hypotheses
        hypotheses = []
        for similar_glyph, similarity in similar_glyphs:
            if similar_glyph in self.reconstruction_keys:
                hypothesis = self._generate_reconstruction_hypothesis(glyph, similar_glyph)
                hypotheses.append((hypothesis, similarity))
        
        # Return best hypothesis
        if hypotheses:
            hypotheses.sort(key=lambda x: x[1], reverse=True)
            return hypotheses[0][0]
        
        # Fallback: expand glyph based on common patterns
        return self._expand_glyph_fallback(glyph)
    
    def _analyze_glyph_patterns(self, glyph: str) -> Dict[str, Any]:
        """Analyze patterns in glyph for reconstruction"""
        
        patterns = {
            'length': len(glyph),
            'character_distribution': {},
            'vowel_consonant_ratio': 0.0,
            'repeated_chars': [],
            'char_positions': {},
            'phonetic_patterns': []
        }
        
        # Character distribution
        for char in glyph:
            patterns['character_distribution'][char] = patterns['character_distribution'].get(char, 0) + 1
        
        # Vowel/consonant analysis
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for char in glyph if char in vowels)
        consonant_count = sum(1 for char in glyph if char.isalpha() and char not in vowels)
        
        if consonant_count > 0:
            patterns['vowel_consonant_ratio'] = vowel_count / consonant_count
        
        # Repeated characters
        for char in set(glyph):
            if glyph.count(char) > 1:
                patterns['repeated_chars'].append(char)
        
        # Character positions
        for i, char in enumerate(glyph):
            if char not in patterns['char_positions']:
                patterns['char_positions'][char] = []
            patterns['char_positions'][char].append(i)
        
        return patterns
    
    def _find_similar_glyphs(self, target_glyph: str, patterns: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Find similar glyphs in the registry"""
        
        similarities = []
        
        for registered_glyph in self.glyph_registry:
            if registered_glyph == target_glyph:
                continue
                
            similarity = self._calculate_glyph_similarity(target_glyph, registered_glyph, patterns)
            if similarity > 0.3:  # Minimum similarity threshold
                similarities.append((registered_glyph, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]  # Top 5 most similar
    
    def _calculate_glyph_similarity(self, glyph1: str, glyph2: str, patterns1: Dict[str, Any]) -> float:
        """Calculate similarity between two glyphs"""
        
        patterns2 = self._analyze_glyph_patterns(glyph2)
        
        similarity_scores = []
        
        # Length similarity
        length_similarity = 1.0 - abs(patterns1['length'] - patterns2['length']) / max(patterns1['length'], patterns2['length'])
        similarity_scores.append(length_similarity * 0.2)
        
        # Character distribution similarity
        all_chars = set(patterns1['character_distribution'].keys()) | set(patterns2['character_distribution'].keys())
        char_similarity = 0.0
        for char in all_chars:
            count1 = patterns1['character_distribution'].get(char, 0)
            count2 = patterns2['character_distribution'].get(char, 0)
            max_count = max(count1, count2)
            if max_count > 0:
                char_similarity += 1.0 - abs(count1 - count2) / max_count
        
        if all_chars:
            char_similarity /= len(all_chars)
        similarity_scores.append(char_similarity * 0.4)
        
        # Vowel/consonant ratio similarity
        ratio_diff = abs(patterns1['vowel_consonant_ratio'] - patterns2['vowel_consonant_ratio'])
        ratio_similarity = max(0.0, 1.0 - ratio_diff)
        similarity_scores.append(ratio_similarity * 0.2)
        
        # Edit distance similarity
        edit_distance = self._calculate_edit_distance(glyph1, glyph2)
        max_length = max(len(glyph1), len(glyph2))
        edit_similarity = 1.0 - edit_distance / max_length if max_length > 0 else 0.0
        similarity_scores.append(edit_similarity * 0.2)
        
        return sum(similarity_scores)
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _generate_reconstruction_hypothesis(self, target_glyph: str, similar_glyph: str) -> str:
        """Generate reconstruction hypothesis based on similar glyph"""
        
        if similar_glyph not in self.reconstruction_keys:
            return target_glyph
        
        similar_original = self.reconstruction_keys[similar_glyph]['original_text']
        
        # Apply similar transformation pattern to target glyph
        transformation_ratio = len(similar_glyph) / len(similar_original) if similar_original else 1.0
        
        # Estimate original length
        estimated_length = int(len(target_glyph) / transformation_ratio)
        
        # Generate hypothesis by expanding glyph
        hypothesis = self._expand_using_pattern(target_glyph, similar_original, estimated_length)
        
        return hypothesis
    
    def _expand_using_pattern(self, glyph: str, reference_text: str, target_length: int) -> str:
        """Expand glyph using reference text pattern"""
        
        # Simple expansion based on character frequency in reference
        char_freq = {}
        for char in reference_text.lower():
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Build expansion
        expanded = []
        for char in glyph.lower():
            expanded.append(char)
            
            # Add likely following characters based on reference
            if len(expanded) < target_length:
                likely_chars = self._get_likely_following_chars(char, reference_text)
                for likely_char in likely_chars[:2]:  # Add up to 2 characters
                    if len(expanded) < target_length:
                        expanded.append(likely_char)
        
        return ''.join(expanded)
    
    def _get_likely_following_chars(self, char: str, reference_text: str) -> List[str]:
        """Get characters likely to follow given character based on reference"""
        
        following_chars = []
        reference_lower = reference_text.lower()
        
        for i, ref_char in enumerate(reference_lower):
            if ref_char == char and i + 1 < len(reference_lower):
                following_chars.append(reference_lower[i + 1])
        
        # Return most common following characters
        char_counts = {}
        for following_char in following_chars:
            char_counts[following_char] = char_counts.get(following_char, 0) + 1
        
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        return [char for char, count in sorted_chars]
    
    def _expand_glyph_fallback(self, glyph: str) -> str:
        """Fallback expansion when no similar glyphs found"""
        
        # Simple heuristic expansion
        expanded = []
        
        for i, char in enumerate(glyph):
            expanded.append(char)
            
            # Add vowels after consonants
            if char.isalpha() and char.lower() not in 'aeiou':
                if i == len(glyph) - 1 or glyph[i + 1].lower() not in 'aeiou':
                    expanded.append('e')  # Most common vowel
            
            # Add space after certain patterns
            if i < len(glyph) - 1 and char.isalpha() and glyph[i + 1].isupper():
                expanded.append(' ')
        
        return ''.join(expanded)
    
    def _reverse_stage(self, text: str, stage_info: Dict[str, Any]) -> str:
        """Reverse a specific decay stage"""
        
        stage_method = stage_info['method']
        
        if stage_method == "character_removal":
            return self._reverse_character_decay(text, stage_info)
        elif stage_method == "word_simplification":
            return self._reverse_word_decay(text, stage_info)
        elif stage_method == "concept_extraction":
            return self._reverse_concept_decay(text, stage_info)
        elif stage_method == "pattern_abstraction":
            return self._reverse_pattern_decay(text, stage_info)
        elif stage_method == "glyph_formation":
            return self._reverse_glyph_decay(text, stage_info)
        else:
            return text
    
    def _reverse_character_decay(self, text: str, stage_info: Dict[str, Any]) -> str:
        """Reverse character-level decay"""
        
        # Add back removed characters based on stored patterns
        removed_chars = stage_info.get('removed_chars', [])
        positions = stage_info.get('removal_positions', [])
        
        result = list(text)
        for char, pos in zip(removed_chars, positions):
            if pos <= len(result):
                result.insert(pos, char)
        
        return ''.join(result)
    
    def _reverse_word_decay(self, text: str, stage_info: Dict[str, Any]) -> str:
        """Reverse word-level decay"""
        
        # Restore simplified words
        word_mappings = stage_info.get('word_mappings', {})
        
        words = text.split()
        restored_words = []
        
        for word in words:
            if word in word_mappings:
                restored_words.append(word_mappings[word])
            else:
                restored_words.append(word)
        
        return ' '.join(restored_words)
    
    def _reverse_concept_decay(self, text: str, stage_info: Dict[str, Any]) -> str:
        """Reverse concept-level decay"""
        
        # Restore extracted concepts
        concept_mappings = stage_info.get('concept_mappings', {})
        
        for concept, original in concept_mappings.items():
            text = text.replace(concept, original)
        
        return text
    
    def _reverse_pattern_decay(self, text: str, stage_info: Dict[str, Any]) -> str:
        """Reverse pattern-level decay"""
        
        # Restore abstracted patterns
        pattern_mappings = stage_info.get('pattern_mappings', {})
        
        for pattern, original in pattern_mappings.items():
            text = text.replace(pattern, original)
        
        return text
    
    def _reverse_glyph_decay(self, text: str, stage_info: Dict[str, Any]) -> str:
        """Reverse glyph formation"""
        
        # Restore from glyph to pattern
        glyph_mappings = stage_info.get('glyph_mappings', {})
        
        if text in glyph_mappings:
            return glyph_mappings[text]
        
        return text
