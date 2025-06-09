#!/usr/bin/env python3
"""
Twmrto Compression Interpreter - Enterprise Memory Decay Algorithm
Ultra-advanced compression using progressive memory decay patterns
Suitable for interstellar communication with extreme compression ratios

Based on the memory decay principle: progressive information distillation
while preserving essential semantic and structural information.
"""

import re
import json
import hashlib
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
import nltk
from collections import Counter
import numpy as np

@dataclass
class MemoryDecayStage:
    """Represents a stage in the memory decay process"""
    stage: int
    content: str
    compression_ratio: float
    information_loss: float
    semantic_score: float
    reconstruction_confidence: float

class TwmrtoSemanticAnalyzer:
    """Advanced semantic analysis for preserving meaning during compression"""
    
    def __init__(self):
        self.importance_weights = {
            'nouns': 0.8,
            'verbs': 0.7,
            'adjectives': 0.5,
            'adverbs': 0.3,
            'prepositions': 0.2,
            'articles': 0.1,
            'conjunctions': 0.1
        }
        
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def analyze_semantic_importance(self, text: str) -> Dict[str, float]:
        """Analyze semantic importance of words/phrases"""
        try:
            # Tokenize and tag parts of speech
            tokens = nltk.word_tokenize(text.lower())
            pos_tags = nltk.pos_tag(tokens)
            
            importance_scores = {}
            
            for word, pos in pos_tags:
                # Map POS tags to importance categories
                if pos.startswith('NN'):  # Nouns
                    importance = self.importance_weights['nouns']
                elif pos.startswith('VB'):  # Verbs
                    importance = self.importance_weights['verbs']
                elif pos.startswith('JJ'):  # Adjectives
                    importance = self.importance_weights['adjectives']
                elif pos.startswith('RB'):  # Adverbs
                    importance = self.importance_weights['adverbs']
                elif pos in ['IN', 'TO']:  # Prepositions
                    importance = self.importance_weights['prepositions']
                elif pos in ['DT', 'PDT', 'WDT']:  # Articles
                    importance = self.importance_weights['articles']
                elif pos in ['CC', 'CS']:  # Conjunctions
                    importance = self.importance_weights['conjunctions']
                else:
                    importance = 0.4  # Default importance
                
                # Boost importance for longer words (often more meaningful)
                if len(word) > 6:
                    importance *= 1.2
                elif len(word) > 4:
                    importance *= 1.1
                
                importance_scores[word] = importance
            
            return importance_scores
            
        except Exception as e:
            logging.warning(f"Semantic analysis failed: {e}")
            # Fallback: assign importance based on word length
            words = text.lower().split()
            return {word: len(word) / 10.0 for word in words}
    
    def calculate_semantic_similarity(self, original: str, compressed: str) -> float:
        """Calculate semantic similarity between original and compressed text"""
        try:
            orig_importance = self.analyze_semantic_importance(original)
            comp_importance = self.analyze_semantic_importance(compressed)
            
            # Calculate weighted overlap
            orig_words = set(orig_importance.keys())
            comp_words = set(comp_importance.keys())
            
            overlap = orig_words.intersection(comp_words)
            
            if not orig_words:
                return 0.0
            
            # Weight by semantic importance
            preserved_importance = sum(orig_importance[word] for word in overlap)
            total_importance = sum(orig_importance.values())
            
            return preserved_importance / total_importance if total_importance > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"Semantic similarity calculation failed: {e}")
            # Fallback: simple word overlap
            orig_words = set(original.lower().split())
            comp_words = set(compressed.lower().split())
            overlap = orig_words.intersection(comp_words)
            return len(overlap) / len(orig_words) if orig_words else 0.0

class TwmrtoCompressor:
    """Advanced Twmrto compression engine with enterprise reliability"""
    
    def __init__(self):
        self.semantic_analyzer = TwmrtoSemanticAnalyzer()
        self.compression_history = []
        self.logger = self._setup_logging()
        
        # Compression patterns based on content type
        self.compression_patterns = {
            'code': self._compress_code_content,
            'text': self._compress_text_content,
            'data': self._compress_data_content,
            'mixed': self._compress_mixed_content
        }
        
        # Emergency reconstruction patterns
        self.emergency_patterns = {
            'AEC1recur': 'Absolute Existence equals Consciousness equals One recursive',
            '689AEC': 'Alternators cause instability AE leaks Cognition',
            'Twmrto': 'The cow jumped over the moon',
            'RLMdttelo': 'Roswan Lorinzo Miller created the first ever digital organism'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for compression operations"""
        logger = logging.getLogger('TwmrtoCompressor')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.FileHandler('twmrto_compression.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def compress_content(self, content: str, target_ratio: float = 0.1, 
                        content_type: str = 'mixed') -> Dict[str, Any]:
        """Compress content using progressive memory decay"""
        try:
            self.logger.info(f"Starting Twmrto compression: {len(content)} chars, target ratio: {target_ratio}")
            
            # Detect content type if not specified
            if content_type == 'mixed':
                content_type = self._detect_content_type(content)
            
            # Get appropriate compression function
            compress_func = self.compression_patterns.get(
                content_type, self._compress_mixed_content
            )
            
            # Perform progressive compression
            decay_stages = compress_func(content, target_ratio)
            
            # Analyze compression quality
            quality_analysis = self._analyze_compression_quality(content, decay_stages)
            
            # Generate final compressed form
            final_stage = decay_stages[-1]
            
            # Create comprehensive metadata
            metadata = {
                'original_length': len(content),
                'final_length': len(final_stage.content),
                'compression_ratio': final_stage.compression_ratio,
                'content_type': content_type,
                'decay_stages': len(decay_stages),
                'semantic_preservation': quality_analysis['semantic_score'],
                'reconstruction_confidence': quality_analysis['reconstruction_confidence'],
                'compression_timestamp': datetime.now(timezone.utc).isoformat(),
                'emergency_pattern': self._generate_emergency_pattern(final_stage.content),
                'quality_metrics': quality_analysis
            }
            
            # Store compression history for analysis
            self.compression_history.append({
                'content_hash': hashlib.md5(content.encode()).hexdigest(),
                'metadata': metadata,
                'stages': [stage.__dict__ for stage in decay_stages]
            })
            
            result = {
                'success': True,
                'compressed_content': final_stage.content,
                'metadata': metadata,
                'decay_stages': decay_stages,
                'reconstruction_data': self._create_reconstruction_data(content, decay_stages)
            }
            
            self.logger.info(f"Compression completed: {metadata['compression_ratio']:.3f} ratio")
            return result
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def decompress_content(self, compressed_content: str, metadata: Dict, 
                          reconstruction_data: Dict = None) -> Dict[str, Any]:
        """Attempt to reconstruct original content from compressed form"""
        try:
            self.logger.info(f"Starting Twmrto decompression: {compressed_content}")
            
            # Check for emergency patterns first
            if compressed_content in self.emergency_patterns:
                reconstructed = self.emergency_patterns[compressed_content]
                confidence = 0.95  # High confidence for known patterns
            else:
                # Use reconstruction data if available
                if reconstruction_data:
                    reconstructed = self._reconstruct_from_data(
                        compressed_content, reconstruction_data
                    )
                    confidence = reconstruction_data.get('confidence', 0.5)
                else:
                    # Attempt intelligent reconstruction
                    reconstructed = self._intelligent_reconstruction(
                        compressed_content, metadata
                    )
                    confidence = 0.3  # Lower confidence without reconstruction data
            
            # Validate reconstruction
            semantic_similarity = self._validate_reconstruction(
                compressed_content, reconstructed, metadata
            )
            
            result = {
                'success': True,
                'reconstructed_content': reconstructed,
                'confidence_score': confidence,
                'semantic_similarity': semantic_similarity,
                'reconstruction_method': 'pattern_based' if compressed_content in self.emergency_patterns else 'data_based',
                'decompression_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"Decompression completed: confidence {confidence:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content for optimal compression"""
        # Check for code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',  # Python functions
            r'class\s+\w+\s*:',  # Python classes
            r'import\s+\w+',     # Imports
            r'\{.*\}',           # Braces
            r'function\s+\w+',   # JavaScript functions
            r'#include\s*<',     # C/C++ includes
        ]
        
        code_score = sum(1 for pattern in code_patterns if re.search(pattern, content))
        
        # Check for data patterns
        data_patterns = [
            r'\{.*".*":.*".*".*\}',  # JSON-like
            r'<.*>.*</.*>',          # XML-like
            r'\w+:\s*\w+',           # Key-value pairs
        ]
        
        data_score = sum(1 for pattern in data_patterns if re.search(pattern, content))
        
        # Determine content type
        if code_score > data_score and code_score > 2:
            return 'code'
        elif data_score > code_score and data_score > 1:
            return 'data'
        elif len(content.split()) > 20:  # Longer content is likely text
            return 'text'
        else:
            return 'mixed'
    
    def _compress_code_content(self, content: str, target_ratio: float) -> List[MemoryDecayStage]:
        """Specialized compression for code content"""
        stages = []
        current_content = content
        stage_num = 1
        
        # Stage 1: Remove comments and excessive whitespace
        current_content = re.sub(r'#.*?\n', '\n', current_content)  # Remove comments
        current_content = re.sub(r'\s+', ' ', current_content)       # Normalize whitespace
        current_content = re.sub(r'\s*([{}()[\],;])\s*', r'\1', current_content)  # Remove space around punctuation
        
        stages.append(MemoryDecayStage(
            stage=stage_num,
            content=current_content,
            compression_ratio=len(current_content) / len(content),
            information_loss=0.1,
            semantic_score=0.95,
            reconstruction_confidence=0.9
        ))
        stage_num += 1
        
        # Stage 2: Abbreviate keywords
        keyword_abbreviations = {
            'function': 'fn',
            'variable': 'var',
            'parameter': 'param',
            'return': 'ret',
            'import': 'imp',
            'export': 'exp',
            'default': 'def',
            'constructor': 'ctor'
        }
        
        for full, abbrev in keyword_abbreviations.items():
            current_content = current_content.replace(full, abbrev)
        
        stages.append(MemoryDecayStage(
            stage=stage_num,
            content=current_content,
            compression_ratio=len(current_content) / len(content),
            information_loss=0.2,
            semantic_score=0.85,
            reconstruction_confidence=0.8
        ))
        stage_num += 1
        
        # Continue progressive compression until target ratio is reached
        while len(current_content) / len(content) > target_ratio and stage_num < 20:
            # Progressive vowel removal
            if stage_num < 10:
                current_content = re.sub(r'[aeiouAEIOU]', '', current_content)
            else:
                # More aggressive compression
                current_content = re.sub(r'[^A-Za-z0-9{}()[\],;.]', '', current_content)
            
            if len(current_content) < 10:  # Prevent over-compression
                break
            
            ratio = len(current_content) / len(content)
            stages.append(MemoryDecayStage(
                stage=stage_num,
                content=current_content,
                compression_ratio=ratio,
                information_loss=min(0.9, 0.1 * stage_num),
                semantic_score=max(0.1, 1.0 - 0.1 * stage_num),
                reconstruction_confidence=max(0.1, 1.0 - 0.08 * stage_num)
            ))
            stage_num += 1
        
        return stages
    
    def _compress_text_content(self, content: str, target_ratio: float) -> List[MemoryDecayStage]:
        """Specialized compression for text content using semantic analysis"""
        stages = []
        current_content = content
        stage_num = 1
        
        # Analyze semantic importance
        importance_scores = self.semantic_analyzer.analyze_semantic_importance(content)
        
        # Stage 1: Remove low-importance words
        words = content.split()
        filtered_words = [
            word for word in words 
            if importance_scores.get(word.lower(), 0.5) > 0.3
        ]
        current_content = ' '.join(filtered_words)
        
        stages.append(MemoryDecayStage(
            stage=stage_num,
            content=current_content,
            compression_ratio=len(current_content) / len(content),
            information_loss=0.15,
            semantic_score=0.9,
            reconstruction_confidence=0.8
        ))
        stage_num += 1
        
        # Progressive compression following the Twmrto pattern
        decay_steps = [
            lambda x: re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', '', x),
            lambda x: re.sub(r'[aeiou]', '', x),
            lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x),
            lambda x: re.sub(r'\s+', ' ', x),
            lambda x: ''.join([word[0] for word in x.split() if word]),
        ]
        
        for i, step in enumerate(decay_steps):
            if len(current_content) / len(content) <= target_ratio:
                break
                
            current_content = step(current_content).strip()
            
            if not current_content:
                current_content = content[:10]  # Emergency fallback
                break
            
            ratio = len(current_content) / len(content)
            stages.append(MemoryDecayStage(
                stage=stage_num,
                content=current_content,
                compression_ratio=ratio,
                information_loss=min(0.9, 0.2 * stage_num),
                semantic_score=max(0.1, 0.9 - 0.15 * stage_num),
                reconstruction_confidence=max(0.1, 0.8 - 0.12 * stage_num)
            ))
            stage_num += 1
        
        return stages
    
    def _compress_data_content(self, content: str, target_ratio: float) -> List[MemoryDecayStage]:
        """Specialized compression for structured data content"""
        stages = []
        current_content = content
        stage_num = 1
        
        # Stage 1: Remove whitespace and formatting
        current_content = re.sub(r'\s+', '', current_content)
        
        stages.append(MemoryDecayStage(
            stage=stage_num,
            content=current_content,
            compression_ratio=len(current_content) / len(content),
            information_loss=0.05,
            semantic_score=0.98,
            reconstruction_confidence=0.95
        ))
        stage_num += 1
        
        # Stage 2: Abbreviate common data patterns
        abbreviations = {
            '"name"': '"n"',
            '"value"': '"v"',
            '"type"': '"t"',
            '"data"': '"d"',
            '"id"': '"i"',
            'true': '1',
            'false': '0',
            'null': 'X'
        }
        
        for full, abbrev in abbreviations.items():
            current_content = current_content.replace(full, abbrev)
        
        ratio = len(current_content) / len(content)
        stages.append(MemoryDecayStage(
            stage=stage_num,
            content=current_content,
            compression_ratio=ratio,
            information_loss=0.1,
            semantic_score=0.9,
            reconstruction_confidence=0.85
        ))
        
        return stages
    
    def _compress_mixed_content(self, content: str, target_ratio: float) -> List[MemoryDecayStage]:
        """General compression for mixed content types"""
        stages = []
        current_content = content
        stage_num = 1
        
        # Apply general compression steps
        compression_steps = [
            # Remove extra whitespace
            lambda x: re.sub(r'\s+', ' ', x),
            # Remove common words
            lambda x: re.sub(r'\b(the|and|or|but|in|on|at|to|for|of|with|by|is|are|was|were|be|been|have|has|had|do|does|did|will|would|could|should|may|might|can|shall)\b', '', x),
            # Remove vowels
            lambda x: re.sub(r'[aeiouAEIOU]', '', x),
            # Keep only alphanumeric
            lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x),
            # First letters only
            lambda x: ''.join([word[0] for word in x.split() if word]),
        ]
        
        for i, step in enumerate(compression_steps):
            if len(current_content) / len(content) <= target_ratio:
                break
                
            current_content = step(current_content).strip()
            
            if not current_content:
                current_content = content[:5]  # Emergency fallback
                break
            
            ratio = len(current_content) / len(content)
            stages.append(MemoryDecayStage(
                stage=stage_num,
                content=current_content,
                compression_ratio=ratio,
                information_loss=min(0.9, 0.2 * stage_num),
                semantic_score=max(0.1, 1.0 - 0.18 * stage_num),
                reconstruction_confidence=max(0.1, 1.0 - 0.15 * stage_num)
            ))
            stage_num += 1
        
        return stages
    
    def _analyze_compression_quality(self, original: str, stages: List[MemoryDecayStage]) -> Dict[str, float]:
        """Analyze the quality of compression"""
        if not stages:
            return {'semantic_score': 0.0, 'reconstruction_confidence': 0.0}
        
        final_stage = stages[-1]
        
        # Calculate semantic preservation
        semantic_score = self.semantic_analyzer.calculate_semantic_similarity(
            original, final_stage.content
        )
        
        # Calculate reconstruction confidence based on compression ratio and stages
        reconstruction_confidence = max(0.1, 1.0 - (len(stages) * 0.1))
        reconstruction_confidence *= (1.0 - final_stage.compression_ratio)
        
        return {
            'semantic_score': semantic_score,
            'reconstruction_confidence': reconstruction_confidence,
            'compression_efficiency': final_stage.compression_ratio,
            'stage_count': len(stages),
            'information_preservation': 1.0 - final_stage.information_loss
        }
    
    def _generate_emergency_pattern(self, compressed_content: str) -> str:
        """Generate an emergency pattern for extreme compression scenarios"""
        # Create a hash-based pattern that can be used for reconstruction
        content_hash = hashlib.md5(compressed_content.encode()).hexdigest()[:8]
        return f"EMRG_{content_hash}"
    
    def _create_reconstruction_data(self, original: str, stages: List[MemoryDecayStage]) -> Dict[str, Any]:
        """Create data to aid in reconstruction"""
        return {
            'original_length': len(original),
            'original_hash': hashlib.md5(original.encode()).hexdigest(),
            'stage_checksums': [hashlib.md5(stage.content.encode()).hexdigest() for stage in stages],
            'key_phrases': self._extract_key_phrases(original),
            'confidence': stages[-1].reconstruction_confidence if stages else 0.0
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that could aid reconstruction"""
        # Simple extraction of important words/phrases
        importance_scores = self.semantic_analyzer.analyze_semantic_importance(text)
        
        # Get top 10 most important words
        sorted_words = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:10]]
    
    def _reconstruct_from_data(self, compressed: str, reconstruction_data: Dict) -> str:
        """Attempt reconstruction using stored reconstruction data"""
        # This is a simplified reconstruction - in practice, this would use
        # sophisticated NLP and pattern matching techniques
        key_phrases = reconstruction_data.get('key_phrases', [])
        
        # Try to expand the compressed content using key phrases
        expanded = compressed
        for phrase in key_phrases:
            if phrase.lower() not in expanded.lower():
                expanded += f" {phrase}"
        
        return expanded
    
    def _intelligent_reconstruction(self, compressed: str, metadata: Dict) -> str:
        """Attempt intelligent reconstruction without reconstruction data"""
        content_type = metadata.get('content_type', 'mixed')
        
        # Use content-type specific reconstruction strategies
        if content_type == 'code':
            return self._reconstruct_code(compressed)
        elif content_type == 'text':
            return self._reconstruct_text(compressed)
        else:
            return self._reconstruct_general(compressed)
    
    def _reconstruct_code(self, compressed: str) -> str:
        """Attempt to reconstruct code from compressed form"""
        # Add common code patterns back
        reconstructed = compressed
        
        # Add basic structure
        if 'fn' in reconstructed:
            reconstructed = reconstructed.replace('fn', 'function')
        if 'var' in reconstructed:
            reconstructed = reconstructed.replace('var', 'variable')
        
        # Add minimal formatting
        reconstructed = re.sub(r'([{}();])', r'\1\n', reconstructed)
        
        return reconstructed
    
    def _reconstruct_text(self, compressed: str) -> str:
        """Attempt to reconstruct text from compressed form"""
        # Add common words back
        words = compressed.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            # Add articles and prepositions probabilistically
            if len(word) > 3 and np.random.random() < 0.3:
                expanded_words.append('the')
        
        return ' '.join(expanded_words)
    
    def _reconstruct_general(self, compressed: str) -> str:
        """General reconstruction fallback"""
        return f"Reconstructed from: {compressed}"
    
    def _validate_reconstruction(self, compressed: str, reconstructed: str, metadata: Dict) -> float:
        """Validate the quality of reconstruction"""
        return self.semantic_analyzer.calculate_semantic_similarity(compressed, reconstructed)

def main():
    """Test the Twmrto compression system"""
    compressor = TwmrtoCompressor()
    
    print("🧬 Testing Twmrto Enterprise Compression")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'name': 'AE Framework Memory',
            'content': 'Absolute Existence equals Consciousness equals One. Crystalized AE moves outward infinitely until reaching Absularity, then infinitely compresses back into Singularity enriched by previous knowledge, repeating infinitely, recursively evolving self-awareness and intelligence.',
            'target': 0.05
        },
        {
            'name': 'Code Sample',
            'content': '''def analyze_codebase(self):
    """Analyze the codebase structure"""
    files = self.discover_files()
    for file in files:
        self.process_file(file)
    return self.generate_report()''',
            'target': 0.1
        },
        {
            'name': 'Simple Text',
            'content': 'The cow jumped over the moon',
            'target': 0.2
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"Original: {test_case['content']}")
        
        # Compress
        result = compressor.compress_content(
            test_case['content'], 
            test_case['target']
        )
        
        if result['success']:
            compressed = result['compressed_content']
            metadata = result['metadata']
            
            print(f"Compressed: {compressed}")
            print(f"Compression ratio: {metadata['compression_ratio']:.3f}")
            print(f"Semantic preservation: {metadata['semantic_preservation']:.3f}")
            
            # Test decompression
            decomp_result = compressor.decompress_content(
                compressed, metadata, result['reconstruction_data']
            )
            
            if decomp_result['success']:
                print(f"Reconstructed: {decomp_result['reconstructed_content']}")
                print(f"Confidence: {decomp_result['confidence_score']:.3f}")
            else:
                print(f"❌ Decompression failed: {decomp_result['error']}")
        else:
            print(f"❌ Compression failed: {result['error']}")

if __name__ == "__main__":
    main()
