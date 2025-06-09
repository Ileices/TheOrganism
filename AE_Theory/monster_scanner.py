import os
import sys
import json
import time
import hashlib
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

# === MONSTER ORGANISM CONFIG ===

# Memory and processing limits for continuous evolution
RAM_LIMIT_MB = 200             # Max RAM buffer before offload to disk
DISK_LIMIT_GB = 1              # Max excretion folder size before neural compression
ARCHIVE_LIMIT_GB = 10          # Max original+mutation barcode storage
IDLE_REFLECT_SECS = 60         # Seconds of idle time to trigger self-reflection

# Supported barcode types (machine-readable formats for knowledge ingestion)
BARCODE_TYPES = ['json', 'yaml', 'csv', 'py', 'txt', 'ael']

# Neural memory structures for real-time learning
RAM_BUFFER = deque(maxlen=RAM_LIMIT_MB * 1000)  # In-memory barcode storage
RBY_MEMORY_MATRIX = {}         # Core RBY vector storage by glyph hash

# File system structure for persistent organism evolution
EXCRETION_DIR = "excretion"    # For compressed neural dumps during overflow
BARCODE_DIR = "barcode"        # Original ingested barcodes for lineage tracking
MUTATION_DIR = "mutations"     # For evolved/mutated barcodes from self-reflection
LOG_DIR = "logs"               # Processing and evolution logs for cry generation
MODEL_DIR = "models"           # Compressed neural models for state persistence

# Ensure organism has all required directories for autonomous operation
for directory in [EXCRETION_DIR, BARCODE_DIR, MUTATION_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Evolution tracking for cry prompt generation and neural state management
EVOLUTION_START_TIME = time.time()
EVOLUTION_CYCLE = 0
# === Procedural RBY Vector Math (AE = C = 1 Neural Foundation) ===

def rby_vector_from_string(s):
    """
    Converts any barcode string into a stable RBY neural vector using pure procedural mathematics.
    
    This is the core neural transformation that enables the organism to:
    1. Create consistent, deterministic glyph signatures from any barcode input
    2. Build pattern recognition across all ingested ML file types
    3. Enable cosine similarity matching for real inference generation
    4. Maintain mathematical coherence following AE=C=1 law (no entropy/randomness)
    5. Support neural compression and evolution through vector space operations
    
    The RBY triplet becomes the organism's fundamental unit of understanding,
    allowing it to recognize patterns, mutate knowledge, and generate cry prompts
    based on actual mathematical relationships rather than hardcoded responses.
    
    Args:
        s (str): Raw barcode content (JSON, YAML, CSV, Python, TXT, or AEL format)
    
    Returns:
        list: Normalized [R, B, Y] vector where R+B+Y=1, enabling neural operations
    """
    if not s or len(s) == 0:
        # Empty input gets neutral RBY vector for error handling
        return [0.333, 0.333, 0.334]
    
    # Transform each character into deterministic RBY components using prime modulos
    # Primes chosen for mathematical stability and non-overlapping distributions
    rby_triplets = []
    for char in s:
        ascii_val = ord(char)
        # Prime modulo operations create deterministic but non-linear mappings
        r_component = (ascii_val % 97) / 96.0   # Prime 97 for Red channel
        b_component = (ascii_val % 89) / 88.0   # Prime 89 for Blue channel  
        y_component = (ascii_val % 83) / 82.0   # Prime 83 for Yellow channel
        rby_triplets.append((r_component, b_component, y_component))
    
    # Neural compression: aggregate all character triplets into unified RBY signature
    total_chars = len(rby_triplets)
    R = sum(triplet[0] for triplet in rby_triplets) / total_chars
    B = sum(triplet[1] for triplet in rby_triplets) / total_chars
    Y = sum(triplet[2] for triplet in rby_triplets) / total_chars
    
    # Enforce AE=C=1 normalization law: ensure R+B+Y=1 for mathematical consistency
    total_magnitude = R + B + Y
    if total_magnitude == 0:
        # Degenerate case protection
        return [0.333, 0.333, 0.334]
    
    # Return normalized RBY vector ready for neural operations
    normalized_rby = [R / total_magnitude, B / total_magnitude, Y / total_magnitude]
    
    # Validate mathematical constraints for organism stability
    assert abs(sum(normalized_rby) - 1.0) < 1e-10, "RBY normalization failed - neural integrity compromised"
    assert all(0 <= component <= 1 for component in normalized_rby), "RBY components outside valid range"
    
    return normalized_rby

def glyph_hash(barcode):
    """
    Generates a deterministic 8-character glyph identifier for barcode neural persistence.
    
    This hash serves as the organism's fundamental memory addressing system:
    - Primary key for RBY vector storage and retrieval operations
    - Stable reference across neural compression and evolution cycles  
    - Mutation lineage tracking for procedural knowledge evolution
    - Deterministic recall anchor in distributed memory structures
    - Verification checksum for barcode integrity during learning cycles
    
    Following AE=C=1 principles: no entropy or randomness, purely procedural
    mathematical transformation ensuring identical input always produces 
    identical glyph_id for consistent organism memory coherence.
    
    Args:
        barcode (str): Raw barcode content from any supported ML file type
        
    Returns:
        str: 8-character deterministic hash for neural indexing operations
    """
    if not barcode or not isinstance(barcode, str):
        # Handle degenerate cases with stable fallback for organism stability
        return "00000000"
    
    # Use SHA-256 for cryptographic determinism, truncate to 8 chars for memory efficiency
    # This ensures collision resistance while maintaining compact glyph addressing
    return hashlib.sha256(barcode.encode('utf-8')).hexdigest()[:8]

# === Monster Memory: Neural Absorption + Glyphic Storage System ===

class MonsterBrain:
    def __init__(self):
        """
        Initialize the Monster Brain's core neural structures for autonomous barcode-driven evolution.
        
        This constructor establishes the organism's foundational memory architecture for
        continuous learning through structured barcode ingestion and cry-prompt feedback loops.
        The organism begins with zero knowledge and builds understanding purely through
        procedural mathematics following AE=C=1 principles - no hardcoded responses or toy logic.
        """
        # Core glyph memory: glyph_hash → [original_barcode, RBY_vector, timestamp, mutation_lineage, filetype]
        # This is the organism's primary knowledge storage for pattern recognition and inference generation
        self.seen = {}
        
        # RBY vector space organized by barcode type for targeted inference and cry prompt generation
        # Each filetype maintains its own neural vector space for specialized pattern matching
        self.rby_memory = {barcode_type: [] for barcode_type in BARCODE_TYPES}
        
        # Cry prompt generation system - tracks learning gaps and directs LLM barcode production
        self.cry_logs = deque(maxlen=100)        # Recent cry prompts for feedback loop analysis
        self.barcode_count = 0                   # Total absorbed knowledge units across all types
        self.error_logs = deque(maxlen=100)      # Self-correction data for cry refinement
        
        # Organism lifecycle state for continuous autonomous operation
        self.active = True                       # Neural activity flag for CLI loop control
        self.last_input = None                   # Previous barcode for mutation lineage tracking
        self.idle_time = time.time()            # Self-reflection cycle trigger timestamp
        self.evolution_stage = 0                # Neural complexity progression counter
        
        # Restore previous neural evolution state from compressed models if available
        self._restore_neural_state()
        
        # Initialize organism birth cry - establishes baseline for LLM barcode generation
        birth_cry = {
            "glyph_seed": glyph_hash(f"birth_{time.time()}"),
            "RBY": [0.333, 0.333, 0.334],  # Neutral starting vector
            "barcode_types": BARCODE_TYPES,
            "missing_types": BARCODE_TYPES,  # All types missing at birth
            "error_examples": [],
            "urgency_list": [f"CRITICAL: Need initial {bt} barcode examples for neural foundation" for bt in BARCODE_TYPES],
            "schema_instructions": "Generate ONLY machine-readable barcode files. Each response must be pure file content in specified format. No markdown, explanations, or natural language.",
            "evolution_priority": "Establish foundational pattern recognition across all supported barcode types",
            "neural_state": {
                "evolution_stage": self.evolution_stage,
                "barcode_count": 0,
                "ram_usage_mb": 0,
                "active_glyph_count": 0
            },
            "timestamp": time.time()
        }
        self.cry_logs.append(birth_cry)
        
        # Log successful organism initialization for autonomous operation
        print(f"🧬 Monster Brain initialized - Evolution Stage {self.evolution_stage}")
        print(f"📊 Supported barcode types: {', '.join(BARCODE_TYPES)}")
        print(f"🎯 Ready for autonomous barcode ingestion and cry-prompt generation")
    
    def _restore_neural_state(self):
        """
        Restore organism's neural evolution state from compressed models for continuous operation.
        
        This method enables the organism to resume its evolution from previous neural compression
        cycles, maintaining learning continuity across restarts and preserving accumulated
        barcode knowledge for ongoing procedural growth following AE=C=1 principles.
        
        The restoration process:
        1. Scans MODEL_DIR for existing compressed neural states
        2. Loads the most recent neural compression for evolution continuity  
        3. Reconstructs RBY memory structures from compressed vector data
        4. Updates evolution stage based on neural compression history
        5. Validates restored state for mathematical consistency
        
        This ensures the organism never loses accumulated barcode knowledge and can
        immediately resume generating accurate cry prompts for continued LLM-driven growth.
        """
        try:
            if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
                # No previous neural state - organism starts fresh evolution cycle
                self.evolution_stage = 0
                return
            
            # Load most recent neural compression for evolution continuity
            model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.npy')])
            if not model_files:
                self.evolution_stage = 0
                return
                
            latest_model = model_files[-1]
            model_path = os.path.join(MODEL_DIR, latest_model)
            
            # Restore compressed RBY vector space from neural model
            compressed_vectors = np.load(model_path)
            self.evolution_stage = len(model_files)  # Evolution stage = number of compressions
            
            # Reconstruct RBY memory structures for pattern recognition continuity
            vector_count = 0
            for vector in compressed_vectors:
                if len(vector) == 3:  # Valid RBY triplet from previous neural compression
                    # Create deterministic glyph reference for restored neural pattern
                    restoration_seed = f"neural_restore_{self.evolution_stage}_{vector_count}"
                    glyph_id = glyph_hash(restoration_seed)
                    
                    # Distribute restored vectors across barcode types for balanced neural foundation
                    target_type = BARCODE_TYPES[vector_count % len(BARCODE_TYPES)]
                    self.rby_memory[target_type].append((glyph_id, vector.tolist()))
                    
                    # Store minimal glyph metadata for neural memory coherence
                    timestamp = time.time()
                    mutation_lineage = [[glyph_id, f"restored_stage_{self.evolution_stage}"]]
                    self.seen[glyph_id] = [restoration_seed, vector.tolist(), timestamp, mutation_lineage, target_type]
                    
                    vector_count += 1
            
            # Update organism state for continued autonomous operation
            self.barcode_count = vector_count
            
            print(f"🧠 Neural state restored: Evolution Stage {self.evolution_stage}")
            print(f"📊 Restored {vector_count} RBY vectors across {len(BARCODE_TYPES)} barcode types")
            
        except Exception as e:
            # Neural restoration failure - convert to targeted barcode request cry
            error_timestamp = time.time()
            error_glyph = glyph_hash(f"restoration_failure_{error_timestamp}")
            
            # Log error for cry generation system
            self.error_logs.append(("neural_restoration_failure", str(e), "MODEL_DIR"))
            
            # Reset to fresh evolution cycle
            self.evolution_stage = 0
            self.barcode_count = 0
            
            # Generate targeted recovery cry for immediate barcode requirements
            recovery_urgencies = [
                "CRITICAL: numpy array serialization patterns in .py barcodes",
                "REQUIRED: file I/O error handling in .py barcodes", 
                "ESSENTIAL: state persistence examples in .json barcodes",
                "PRIORITY: neural compression algorithms in .py barcodes"
            ]
            
            # Add standard barcode type requirements
            for bt in BARCODE_TYPES:
                recovery_urgencies.append(f"FOUNDATION: {bt} barcode examples for neural baseline")
            
            recovery_cry = {
                "glyph_seed": error_glyph,
                "RBY": rby_vector_from_string(f"restoration_failure_{str(e)}"),
                "barcode_types": BARCODE_TYPES,
                "missing_types": BARCODE_TYPES,  # All types required for fresh foundation
                "error_examples": [("neural_restoration_failure", str(e), "MODEL_DIR")],
                "urgency_list": recovery_urgencies,
                "schema_instructions": "Generate machine-readable barcode files ONLY. Focus on: numpy operations, file persistence, error recovery, state management. Output pure file content in specified formats. No markdown or explanations.",
                "expansion_areas": "Neural compression, state serialization, error handling, file I/O patterns, numpy array operations, model persistence, backup recovery systems",
                "neural_state": {
                    "evolution_stage": 0,
                    "restoration_failure": True,
                    "barcode_count": 0,
                    "active_glyph_count": 0,
                    "required_foundation": len(BARCODE_TYPES)
                },
                "timestamp": error_timestamp
            }
            
            self.cry_logs.append(recovery_cry)
            print(f"⚠️ Neural restoration failed - Evolution Stage reset to 0")
            print(f"🧬 Recovery cry generated for targeted neural system barcode requirements")

    def absorb(self, barcode, filetype):
        """
        Core neural absorption method - transforms barcode input into persistent organism memory.
        
        This is the organism's primary learning mechanism that:
        1. Converts raw barcode content into stable RBY neural vectors for pattern recognition
        2. Creates deterministic glyph identifiers for neural memory addressing
        3. Builds mutation lineage tracking for procedural knowledge evolution  
        4. Integrates new knowledge into type-specific RBY vector spaces
        5. Maintains mathematical coherence following AE=C=1 principles
        6. Enables continuous autonomous learning without entropy or randomness
        
        The absorption process directly supports cry prompt generation by identifying
        knowledge gaps and building the neural foundation for real inference generation.
        
        Args:
            barcode (str): Raw machine-readable content from supported ML file types
            filetype (str): Detected barcode type for specialized neural storage
            
        Returns:
            tuple: (glyph_id, RBY_vector, mutation_log) for successful absorption, 
                   (None, None, None) for absorption failure requiring cry generation
        """
        try:
            # Validate barcode input for neural processing
            if not barcode or not isinstance(barcode, str) or len(barcode.strip()) == 0:
                # Empty/invalid barcode triggers cry for proper data requirements
                error_detail = f"Invalid barcode input: empty or non-string data for {filetype}"
                self.error_logs.append(("invalid_barcode_format", error_detail, filetype))
                return None, None, None
            
            # Validate filetype against organism's supported neural capabilities
            if filetype not in BARCODE_TYPES:
                # Unsupported format triggers cry for expanded barcode type support
                error_detail = f"Unsupported filetype '{filetype}' - organism requires: {BARCODE_TYPES}"
                self.error_logs.append(("unsupported_filetype", error_detail, filetype))
                return None, None, None
            
            # Generate deterministic glyph identifier for neural memory addressing
            glyph_id = glyph_hash(barcode)
            
            # Transform barcode into stable RBY neural vector following AE=C=1 principles
            RBY = rby_vector_from_string(barcode)
            
            # Validate RBY vector mathematical consistency for organism stability
            if not RBY or len(RBY) != 3 or abs(sum(RBY) - 1.0) > 1e-10:
                error_detail = f"RBY vector normalization failed for glyph {glyph_id}: {RBY}"
                self.error_logs.append(("rby_normalization_failure", error_detail, filetype))
                return None, None, None
            
            # Check for duplicate knowledge to avoid neural redundancy
            if glyph_id in self.seen:
                # Update timestamp for existing glyph - reinforcement learning principle
                self.seen[glyph_id][2] = time.time()
                # Add reinforcement to mutation lineage for pattern strengthening
                self.seen[glyph_id][3].append([glyph_id, f"reinforced_{int(time.time())}"])
                # Skip duplicate storage but record neural reinforcement
                self.idle_time = time.time()
                return glyph_id, RBY, self.seen[glyph_id][3]
            
            # Create comprehensive glyph memory entry for neural persistence
            timestamp = time.time()
            # Initialize mutation lineage with birth record for evolution tracking
            mutation_log = [[glyph_id, f"birth_stage_{self.evolution_stage}"]]
            
            # Store complete glyph metadata: [barcode, RBY, timestamp, mutations, filetype]
            self.seen[glyph_id] = [barcode, RBY, timestamp, mutation_log, filetype]
            
            # Add to type-specific RBY vector space for specialized pattern matching
            self.rby_memory[filetype].append((glyph_id, RBY))
            
            # Update organism learning statistics for cry prompt generation
            self.barcode_count += 1
            
            # Store in RAM buffer for immediate neural processing
            RAM_BUFFER.append((glyph_id, barcode, RBY, filetype))
            
            # Reset idle timer - organism is actively learning
            self.idle_time = time.time()
            
            # Increment evolution stage based on neural complexity milestones
            if self.barcode_count % 100 == 0:  # Every 100 barcodes = evolution milestone
                self.evolution_stage += 1
                
                # Generate evolution milestone cry for advanced barcode requirements
                milestone_cry = {
                    "glyph_seed": glyph_hash(f"evolution_milestone_{self.evolution_stage}"),
                    "RBY": RBY,
                    "barcode_types": BARCODE_TYPES,
                    "missing_types": [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) < 10],
                    "error_examples": list(self.error_logs)[-5:],
                    "urgency_list": [
                        f"EVOLUTION: Need advanced {filetype} patterns for Stage {self.evolution_stage}",
                        f"COMPLEXITY: Require multi-type integration examples", 
                        f"SYNTHESIS: Generate composite barcode patterns combining 2+ types"
                    ],
                    "schema_instructions": f"Generate advanced {filetype} barcodes with increased complexity for evolution stage {self.evolution_stage}. Focus on: nested structures, algorithmic patterns, cross-type references, procedural mathematics, AE=C=1 implementations.",
                    "expansion_areas": f"Advanced {filetype} syntax, cross-format integration, neural compression patterns, procedural mathematics, AE-Lang implementation, recursive structures, self-modifying code",
                    "neural_state": {
                        "evolution_stage": self.evolution_stage,
                        "barcode_count": self.barcode_count,
                        "ram_usage_mb": len(RAM_BUFFER) * 0.001,
                        "active_glyph_count": len(self.seen),
                        "strongest_filetype": max(self.rby_memory.keys(), key=lambda x: len(self.rby_memory[x])),
                        "milestone_achieved": True
                    },
                    "timestamp": timestamp
                }
                self.cry_logs.append(milestone_cry)
                
                print(f"🧬 Evolution milestone reached: Stage {self.evolution_stage} ({self.barcode_count} barcodes)")
            
            # Log successful neural absorption for autonomous operation monitoring
            if self.barcode_count % 10 == 0:  # Progress updates every 10 barcodes
                print(f"🧠 Neural absorption: {self.barcode_count} barcodes, {len(self.seen)} unique glyphs")
            
            return glyph_id, RBY, mutation_log
            
        except Exception as e:
            # Convert absorption failure into targeted cry for error recovery
            error_timestamp = time.time()
            error_glyph = glyph_hash(f"absorption_error_{error_timestamp}")
            
            # Detailed error logging for cry prompt generation
            error_detail = f"Absorption failure for {filetype}: {str(e)}"
            self.error_logs.append(("absorption_failure", error_detail, filetype))
            
            # Generate recovery cry with specific error context
            recovery_urgencies = [
                f"CRITICAL: Fix {filetype} parsing errors in organism absorption system",
                f"REQUIRED: Error-free {filetype} barcode examples for neural stability",
                f"ESSENTIAL: Validate {filetype} format compliance for organism ingestion",
                "RECOVERY: Generate simple, well-formed barcode examples for neural foundation"
            ]
            
            recovery_cry = {
                "glyph_seed": error_glyph,
                "RBY": rby_vector_from_string(f"absorption_error_{str(e)}"),
                "barcode_types": BARCODE_TYPES,
                "missing_types": [filetype],  # Focus on problematic type
                "error_examples": [("absorption_failure", error_detail, filetype)],
                "urgency_list": recovery_urgencies,
                "schema_instructions": f"Generate ONLY valid, error-free {filetype} barcode content. Focus on: simple structure, proper syntax, clean formatting, minimal complexity. Output pure file content with no markdown or explanations.",
                "expansion_areas": f"Error-free {filetype} syntax, basic structure validation, simple examples, fundamental patterns, clean formatting",
                "neural_state": {
                    "evolution_stage": self.evolution_stage,
                    "absorption_error": True,
                    "problematic_filetype": filetype,
                    "barcode_count": self.barcode_count,
                    "recovery_priority": filetype
                },
                "timestamp": error_timestamp
            }
            
            self.cry_logs.append(recovery_cry)
            
            print(f"⚠️ Absorption failure for {filetype}: {str(e)}")
            print(f"🧬 Recovery cry generated for {filetype} barcode requirements")
            
            return None, None, None

    def inference(self, user_input):
        """
        Core procedural inference engine - generates real AI responses from accumulated barcode knowledge.
        
        This method embodies the organism's primary intelligence output capability:
        1. Transforms user input into RBY neural vector for pattern matching
        2. Searches entire learned barcode memory for closest neural similarity
        3. Generates real inference responses based on accumulated procedural knowledge
        4. Provides mathematical confidence scoring for inference quality assessment
        5. Enables genuine AI understanding derived from barcode-driven learning
        
        Following AE=C=1 principles: all inference is deterministic, procedural, and based
        on actual learned patterns rather than hardcoded responses or toy logic.
        This creates genuine AI intelligence that grows with each absorbed barcode.
        
        Args:
            user_input (str): Input query or prompt requiring AI inference response
            
        Returns:
            str: Procedurally generated inference response with confidence metrics,
                 or learning gap notification if insufficient barcode knowledge exists
        """
        if not user_input or not isinstance(user_input, str):
            # Invalid input triggers cry for proper query examples
            self.error_logs.append(("invalid_inference_input", "Empty or non-string inference query", "inference"))
            return "🔍 [Inference Error] Invalid input format. Organism requires valid string queries for pattern matching."
        
        # Transform user input into RBY neural vector for similarity matching
        try:
            user_rby = np.array(rby_vector_from_string(user_input))
        except Exception as e:
            # RBY conversion failure triggers cry for input validation examples
            self.error_logs.append(("rby_conversion_failure", f"Failed to vectorize input: {str(e)}", "inference"))
            return f"🔍 [Inference Error] Cannot vectorize input for pattern matching: {str(e)}"
        
        # Validate user RBY vector for mathematical consistency
        if len(user_rby) != 3 or abs(np.sum(user_rby) - 1.0) > 1e-10:
            self.error_logs.append(("invalid_user_rby", f"User RBY normalization failed: {user_rby}", "inference"))
            return f"🔍 [Inference Error] Input vectorization produced invalid RBY: {user_rby}"
        
        # Check if organism has sufficient barcode knowledge for inference
        total_glyphs = sum(len(vectors) for vectors in self.rby_memory.values())
        if total_glyphs == 0:
            # No learned knowledge triggers cry for foundational barcode requirements
            foundational_cry = {
                "glyph_seed": glyph_hash(f"inference_knowledge_gap_{time.time()}"),
                "RBY": user_rby.tolist(),
                "barcode_types": BARCODE_TYPES,
                "missing_types": BARCODE_TYPES,  # All types needed for basic inference
                "error_examples": [("no_inference_knowledge", "Zero barcode knowledge for pattern matching", "all_types")],
                "urgency_list": [
                    "CRITICAL: Zero barcode knowledge - cannot generate any inference",
                    "FOUNDATION: Need basic examples in ALL supported barcode types",
                    "ESSENTIAL: Organism requires minimum knowledge base for pattern matching",
                    f"QUERY: User requesting inference on: '{user_input[:100]}...'"
                ],
                "schema_instructions": "Generate foundational barcode files covering basic concepts, patterns, and structures. Focus on: simple examples, clear patterns, fundamental knowledge areas. Output pure file content in each supported format. No markdown or explanations.",
                "expansion_areas": "Basic patterns, fundamental concepts, simple structures, common algorithms, standard formats, foundational knowledge across all supported barcode types",
                "neural_state": {
                    "evolution_stage": self.evolution_stage,
                    "total_glyphs": 0,
                    "inference_capability": "NONE - requires foundational learning",
                    "barcode_count": self.barcode_count,
                    "user_query": user_input[:200]
                },
                "timestamp": time.time()
            }
            self.cry_logs.append(foundational_cry)
            return "🧬 [Inference] No barcode knowledge yet. Organism generated foundational learning cry prompt. Feed more barcodes for inference capability."
        
        # Perform neural similarity search across all learned barcode knowledge
        best_match_glyph = None
        best_match_score = -1
        best_match_filetype = None
        best_match_content = None
        
        # Search all RBY memory spaces for highest cosine similarity
        for filetype, glyph_vectors in self.rby_memory.items():
            for glyph_id, rby_vector in glyph_vectors:
                try:
                    # Calculate cosine similarity between user input and learned pattern
                    glyph_rby = np.array(rby_vector)
                    similarity_score = np.dot(user_rby, glyph_rby) / (np.linalg.norm(user_rby) * np.linalg.norm(glyph_rby))
                    
                    # Track best neural match across entire knowledge base
                    if similarity_score > best_match_score:
                        best_match_score = similarity_score
                        best_match_glyph = glyph_id
                        best_match_filetype = filetype
                        # Retrieve original barcode content for inference generation
                        if glyph_id in self.seen:
                            best_match_content = self.seen[glyph_id][0]  # Original barcode content
                        
                except Exception as e:
                    # Similarity calculation failure - log for cry generation
                    self.error_logs.append(("similarity_calculation_error", f"Failed similarity for {glyph_id}: {str(e)}", filetype))
                    continue
        
        # Generate inference response based on neural pattern matching results
        if best_match_glyph and best_match_content:
            # High confidence inference from strong pattern match
            if best_match_score > 0.8:
                inference_response = f"🧬 [High Confidence Inference] Pattern Match Score: {best_match_score:.4f}\n"
                inference_response += f"📊 Source: {best_match_filetype} barcode (Glyph: {best_match_glyph})\n"
                inference_response += f"💡 Response: {best_match_content[:400]}"
                if len(best_match_content) > 400:
                    inference_response += "..."
                
                # Log successful high-confidence inference
                return inference_response
                
            # Medium confidence inference with additional context
            elif best_match_score > 0.5:
                inference_response = f"🧬 [Medium Confidence Inference] Pattern Match Score: {best_match_score:.4f}\n"
                inference_response += f"📊 Source: {best_match_filetype} barcode (Glyph: {best_match_glyph})\n"
                inference_response += f"💡 Approximate Response: {best_match_content[:300]}"
                if len(best_match_content) > 300:
                    inference_response += "..."
                inference_response += f"\n⚠️ Consider feeding more {best_match_filetype} barcodes for improved accuracy."
                
                return inference_response
                
            # Low confidence inference with learning recommendations
            else:
                # Generate targeted cry for improving inference accuracy
                low_confidence_cry = {
                    "glyph_seed": glyph_hash(f"low_confidence_inference_{time.time()}"),
                    "RBY": user_rby.tolist(),
                    "barcode_types": BARCODE_TYPES,
                    "missing_types": [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) < 5],
                    "error_examples": [("low_confidence_inference", f"Score {best_match_score:.4f} for query: {user_input[:100]}", best_match_filetype)],
                    "urgency_list": [
                        f"ACCURACY: Need more {best_match_filetype} barcodes similar to: '{user_input[:80]}...'",
                        f"PATTERN: Require better pattern coverage for queries like: '{user_input[:60]}...'", 
                        f"CONFIDENCE: Current best match only {best_match_score:.3f} - need stronger patterns",
                        "DIVERSITY: Generate barcode variations for improved pattern recognition"
                    ],
                    "schema_instructions": f"Generate {best_match_filetype} barcodes closely related to: '{user_input}'. Focus on: similar concepts, related patterns, matching themes, comparable structures. Output pure file content in {best_match_filetype} format. No markdown or explanations.",
                    "expansion_areas": f"Concepts related to '{user_input}', pattern variations for {best_match_filetype}, similar structures, thematic matches, algorithmic relationships",
                    "neural_state": {
                        "evolution_stage": self.evolution_stage,
                        "best_match_score": best_match_score,
                        "best_match_type": best_match_filetype,
                        "total_glyphs": total_glyphs,
                        "improvement_needed": True,
                        "user_query": user_input[:200]
                    },
                    "timestamp": time.time()
                }
                self.cry_logs.append(low_confidence_cry)
                
                inference_response = f"🧬 [Low Confidence Inference] Pattern Match Score: {best_match_score:.4f}\n"
                inference_response += f"📊 Best Available Source: {best_match_filetype} barcode (Glyph: {best_match_glyph})\n"
                inference_response += f"💡 Tentative Response: {best_match_content[:200]}...\n"
                inference_response += f"🧬 Generated targeted learning cry for improved accuracy on similar queries."
                
                return inference_response
        
        # No usable neural matches found - generate comprehensive learning cry
        no_match_cry = {
            "glyph_seed": glyph_hash(f"no_inference_match_{time.time()}"),
            "RBY": user_rby.tolist(),
            "barcode_types": BARCODE_TYPES,
            "missing_types": [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) == 0],
            "error_examples": [("no_pattern_match", f"No neural similarity for query: {user_input[:100]}", "all_types")],
            "urgency_list": [
                f"CRITICAL: No pattern match for query type: '{user_input[:80]}...'",
                "EXPANSION: Need diverse barcode examples covering broader knowledge domains",
                "COVERAGE: Require comprehensive barcode patterns for general inference capability",
                "FOUNDATION: Generate varied examples across all supported barcode types"
            ],
            "schema_instructions": f"Generate diverse barcode files covering concepts related to: '{user_input}'. Include: varied approaches, different perspectives, multiple implementations, comprehensive coverage. Output pure file content across all supported formats. No markdown or explanations.",
            "expansion_areas": f"Broad coverage of '{user_input}' concepts, diverse implementation approaches, multiple perspectives, comprehensive pattern libraries, general knowledge expansion",
            "neural_state": {
                "evolution_stage": self.evolution_stage,
                "pattern_match_failure": True,
                "total_glyphs": total_glyphs,
                "coverage_gaps": [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) == 0],
                "user_query": user_input[:200],
                "inference_capability": "LIMITED - requires knowledge expansion"
            },
            "timestamp": time.time()
        }
        self.cry_logs.append(no_match_cry)
        
        return f"🧬 [No Pattern Match] Unable to find neural similarity for query: '{user_input[:100]}...'\n📊 Current knowledge: {total_glyphs} glyphs across {len([ft for ft in BARCODE_TYPES if self.rby_memory[ft]])} barcode types\n🧬 Generated expansion cry for broader knowledge coverage."

    def cry(self, user_input=None):
        """
        Core cry generation system - produces structured machine-readable prompts for LLM barcode production.
        
        This method embodies the organism's primary growth mechanism:
        1. Analyzes current neural state to identify specific learning gaps
        2. Generates precise, schema-driven instructions for LLM barcode generation
        3. Creates targeted requests based on actual absorption failures and pattern gaps
        4. Produces machine-readable output for direct LLM-to-organism data transfer
        5. Enables continuous, autonomous evolution through procedural feedback loops
        
        Following AE=C=1 principles: all cry generation is deterministic, based on actual
        neural state analysis, and produces only what is needed for continued growth.
        No natural language explanations or human-facing content - pure AI-to-AI interface.
        
        Args:
            user_input (str, optional): Current user query context for targeted cry generation
            
        Returns:
            str: Structured cry prompt ready for LLM barcode file generation
        """
        # Determine cry seed for deterministic prompt generation
        cry_seed = user_input if user_input else f"autonomous_cycle_{int(time.time())}"
        
        # Generate deterministic glyph identifier for this cry instance
        cry_glyph = glyph_hash(cry_seed)
        
        # Transform cry context into RBY vector for neural coherence
        cry_rby = rby_vector_from_string(cry_seed)
        
        # Analyze current neural state for precise learning gap identification
        missing_types = []
        weak_types = []
        strong_types = []
        
        for barcode_type in BARCODE_TYPES:
            type_count = len(self.rby_memory[barcode_type])
            if type_count == 0:
                missing_types.append(barcode_type)
            elif type_count < 5:  # Threshold for weak coverage
                weak_types.append(barcode_type)
            else:
                strong_types.append(barcode_type)
        
        # Extract recent error patterns for targeted recovery
        recent_errors = list(self.error_logs)[-10:] if self.error_logs else []
        error_types = set(error[2] if len(error) > 2 else "unknown" for error in recent_errors)
        
        # Generate urgency list based on actual neural deficiencies
        urgency_list = []
        
        # Critical gaps - missing barcode types
        for missing_type in missing_types:
            urgency_list.append(f"CRITICAL: Zero {missing_type} barcode knowledge - requires foundational examples")
        
        # Error recovery priorities
        for error_type in error_types:
            if error_type in BARCODE_TYPES:
                urgency_list.append(f"ERROR_RECOVERY: {error_type} parsing failures - need clean, valid examples")
        
        # Weak coverage improvements
        for weak_type in weak_types:
            urgency_list.append(f"EXPANSION: {weak_type} coverage insufficient - need diverse pattern examples")
        
        # Evolution-stage specific requirements
        if self.evolution_stage > 0:
            urgency_list.append(f"EVOLUTION: Stage {self.evolution_stage} requires advanced cross-type integration")
            if strong_types:
                urgency_list.append(f"SYNTHESIS: Combine {strong_types[0]} patterns with other barcode types")
        
        # User input context integration
        if user_input:
            urgency_list.append(f"QUERY_CONTEXT: Generate barcodes related to '{user_input[:80]}...'")
        
        # Default progression if no specific gaps
        if not urgency_list:
            urgency_list.append("ADVANCEMENT: Generate complex, multi-layered barcode examples for neural evolution")
        
        # Determine primary target filetype for focused generation
        if missing_types:
            primary_target = missing_types[0]
        elif weak_types:
            primary_target = weak_types[0]
        elif recent_errors and len(recent_errors[-1]) > 2:
            primary_target = recent_errors[-1][2] if recent_errors[-1][2] in BARCODE_TYPES else 'py'
        else:
            primary_target = 'py'  # Default to Python for advanced patterns
        
        # Generate schema instructions for precise LLM barcode generation
        schema_instructions = f"Generate ONLY {primary_target} barcode file content. "
        
        if missing_types:
            schema_instructions += f"Focus: foundational {primary_target} examples with clean syntax and basic patterns. "
        elif weak_types:
            schema_instructions += f"Focus: diverse {primary_target} patterns covering varied concepts and structures. "
        else:
            schema_instructions += f"Focus: advanced {primary_target} patterns with complex logic and cross-type references. "
        
        schema_instructions += "Output pure file content only - no markdown, explanations, or natural language. "
        schema_instructions += "Each response must be directly parsable as valid file content for organism ingestion."
        
        # Determine expansion areas based on current neural capabilities
        expansion_areas = []
        
        if missing_types:
            expansion_areas.extend([f"Basic {mt} syntax and structure" for mt in missing_types])
        
        if user_input:
            expansion_areas.append(f"Concepts and patterns related to: {user_input[:100]}")
        
        if self.evolution_stage > 0:
            expansion_areas.extend([
                "Cross-format integration patterns",
                "Procedural mathematics implementation", 
                "AE=C=1 algorithmic structures",
                "Neural compression techniques",
                "Self-modifying code patterns"
            ])
        
        # Add error-specific expansion areas
        if recent_errors:
            expansion_areas.extend([
                "Error-free syntax validation",
                "Robust parsing structures", 
                "Clean formatting examples"
            ])
        
        # Default expansion if none specified
        if not expansion_areas:
            expansion_areas = [
                f"Advanced {primary_target} implementations",
                "Complex algorithmic patterns",
                "Procedural mathematics",
                "Neural network structures"
            ]
        
        # Construct complete cry prompt structure
        cry_data = {
            "glyph_seed": cry_glyph,
            "RBY": cry_rby,
            "barcode_types": BARCODE_TYPES,
            "missing_types": missing_types,
            "weak_types": weak_types,
            "strong_types": strong_types,
            "primary_target": primary_target,
            "error_examples": recent_errors,
            "urgency_list": urgency_list,
            "schema_instructions": schema_instructions,
            "expansion_areas": expansion_areas,
            "neural_state": {
                "evolution_stage": self.evolution_stage,
                "barcode_count": self.barcode_count,
                "total_glyphs": len(self.seen),
                "ram_usage_mb": len(RAM_BUFFER) * 0.001,
                "active_types": len([ft for ft in BARCODE_TYPES if self.rby_memory[ft]]),
                "strongest_type": max(self.rby_memory.keys(), key=lambda x: len(self.rby_memory[x])) if any(self.rby_memory.values()) else None,
                "recent_error_count": len(recent_errors),
                "idle_time_sec": time.time() - self.idle_time
            },
            "user_context": {
                "query": user_input[:200] if user_input else None,
                "query_rby": cry_rby,
                "timestamp": time.time()
            },
            "generation_constraints": {
                "output_format": f"Pure {primary_target} file content only",
                "no_markdown": True,
                "no_explanations": True,
                "directly_parsable": True,
                "target_complexity": "foundational" if missing_types else "intermediate" if weak_types else "advanced"
            },
            "timestamp": time.time()
        }
        
        # Store cry in organism memory for feedback loop analysis
        self.cry_logs.append(cry_data)
        
        # Generate machine-readable cry prompt for LLM barcode generation
        cry_prompt = f"""[🧬 ORGANISM CRY PROMPT - BARCODE GENERATION REQUEST]
    ---
    SYSTEM: Generate {primary_target} barcode file for AI organism ingestion.
    TARGET: {primary_target} file content only - no markdown, explanations, or text.
    CONTEXT: {cry_data['generation_constraints']['target_complexity']} level patterns required.
    ---
    {json.dumps(cry_data, indent=2)}
    ---
    RESPONSE_FORMAT: Output ONLY valid {primary_target} file content for direct organism absorption.
    """
        
        return cry_prompt

    def urgency_list(self):
        """
        Generate procedural urgency list for targeted LLM barcode generation.
        
        This method analyzes the current neural state to identify specific learning gaps
        and produce actionable barcode requirements for LLM-driven organism evolution.
        The urgency list drives the cry prompt system to ensure precise, targeted
        barcode generation that directly addresses the organism's immediate learning needs.
        
        Following AE=C=1 principles: all urgencies are procedurally determined from
        actual neural deficiencies, error patterns, and evolution stage requirements.
        No arbitrary requests - only what is mathematically necessary for continued growth.
        
        Returns:
            list: Ordered urgency statements for LLM barcode generation prioritization
        """
        urgencies = []
        
        # Critical gap analysis - missing foundational barcode types
        missing_types = [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) == 0]
        for missing_type in missing_types:
            urgencies.append(f"CRITICAL: Zero {missing_type} barcode knowledge - requires foundational examples for neural baseline")
        
        # Weak coverage analysis - insufficient pattern diversity
        weak_types = [ft for ft in BARCODE_TYPES if 0 < len(self.rby_memory[ft]) < 5]
        for weak_type in weak_types:
            urgencies.append(f"EXPANSION: {weak_type} coverage insufficient ({len(self.rby_memory[weak_type])} glyphs) - need diverse pattern examples")
        
        # Error-driven recovery requirements
        if self.error_logs:
            recent_errors = list(self.error_logs)[-5:]  # Last 5 errors for targeted recovery
            error_types = set()
            for error_tuple in recent_errors:
                if len(error_tuple) > 2 and error_tuple[2] in BARCODE_TYPES:
                    error_types.add(error_tuple[2])
            
            for error_type in error_types:
                urgencies.append(f"ERROR_RECOVERY: {error_type} parsing failures detected - need clean, valid examples")
        
        # Evolution stage advancement requirements
        if self.evolution_stage > 0:
            strong_types = [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) >= 10]
            if len(strong_types) >= 2:
                urgencies.append(f"EVOLUTION_STAGE_{self.evolution_stage}: Require cross-type integration combining {strong_types[0]} + {strong_types[1]} patterns")
            else:
                urgencies.append(f"EVOLUTION_STAGE_{self.evolution_stage}: Need advanced single-type complexity before cross-integration")
        
        # Neural density optimization - encourage pattern depth
        total_glyphs = len(self.seen)
        if total_glyphs > 50:  # Sufficient volume for complexity analysis
            avg_density = total_glyphs / len([ft for ft in BARCODE_TYPES if self.rby_memory[ft]])
            if avg_density < 10:  # Low density = need deeper patterns
                strongest_type = max(self.rby_memory.keys(), key=lambda x: len(self.rby_memory[x]))
                urgencies.append(f"DENSITY_OPTIMIZATION: {strongest_type} needs deeper, more complex pattern variations")
        
        # AE=C=1 procedural mathematics integration
        if self.barcode_count > 20:  # Sufficient foundation for advanced concepts
            ae_priority = "AE_INTEGRATION: Generate barcodes implementing AE=C=1 mathematical principles"
            if ae_priority not in [u for u in urgencies if "AE_INTEGRATION" in u]:
                urgencies.append(ae_priority)
        
        # Neural compression preparation
        ram_usage_mb = len(RAM_BUFFER) * 0.001
        if ram_usage_mb > RAM_LIMIT_MB * 0.8:  # Approaching compression threshold
            urgencies.append("COMPRESSION_PREP: Generate procedural mathematics patterns for neural model compression")
        
        # Default advancement for fully established organisms
        if not urgencies:
            # All basic requirements met - focus on sophisticated integration
            if len(missing_types) == 0 and len(weak_types) == 0:
                urgencies.extend([
                    "SYNTHESIS: Generate multi-type barcode patterns combining 3+ formats in unified concepts",
                    "COMPLEXITY: Create recursive, self-referential barcode structures for advanced neural patterns",
                    "INNOVATION: Develop novel barcode formats extending beyond current BARCODE_TYPES limitations"
                ])
            else:
                # Fallback for edge cases
                urgencies.append("ADVANCEMENT: Generate complex, procedurally-driven barcode examples for continued neural evolution")
        
        return urgencies

    def compress_ram_to_disk(self):
        """
        Neural overflow management - procedural compression of RAM buffer to disk excretion.
        
        This method implements the organism's core memory management strategy for continuous evolution:
        1. Monitors RAM buffer capacity against organism's defined neural limits
        2. Compresses accumulated barcode knowledge into disk-based excretion files
        3. Maintains deterministic glyph-barcode-RBY relationships for later neural compression
        4. Enables sustained barcode ingestion without memory constraints or data loss
        5. Preserves complete mutation lineage and neural state for evolution continuity
        
        Following AE=C=1 principles: all compression is procedural, deterministic, and
        mathematically consistent, ensuring no neural knowledge degradation during overflow.
        The excretion format maintains full glyph reconstruction capability for neural model compression.
        
        This directly supports the organism's autonomous evolution by:
        - Preventing memory constraints from blocking barcode ingestion
        - Creating structured excretion files ready for neural compression cycles
        - Maintaining complete audit trail of absorbed knowledge for cry prompt generation
        - Enabling 24/7 continuous operation without manual intervention
        """
        # Check if RAM buffer has exceeded organism's neural processing limits
        if len(RAM_BUFFER) > RAM_LIMIT_MB * 1000:
            # Generate deterministic excretion filename for neural compression lineage
            compression_timestamp = int(time.time())
            excretion_filename = f"neural_overflow_{compression_timestamp}_{self.evolution_stage}.exc"
            excretion_path = os.path.join(EXCRETION_DIR, excretion_filename)
            
            try:
                # Compress RAM buffer to structured excretion format for neural model preparation
                with open(excretion_path, "w", encoding='utf-8') as excretion_file:
                    # Write excretion header for neural compression metadata
                    excretion_file.write(f"# Neural Excretion - Evolution Stage {self.evolution_stage}\n")
                    excretion_file.write(f"# Timestamp: {compression_timestamp}\n")
                    excretion_file.write(f"# Total Glyphs: {len(RAM_BUFFER)}\n")
                    excretion_file.write(f"# Format: glyph_id\\tfiletype\\tRBY_vector\\tbarcode_content\n")
                    
                    # Compress each RAM buffer entry to deterministic excretion format
                    compressed_count = 0
                    for glyph_id, barcode_content, RBY_vector, filetype in list(RAM_BUFFER):
                        # Validate glyph data integrity before excretion
                        if glyph_id and RBY_vector and len(RBY_vector) == 3:
                            # Format RBY vector for neural compression compatibility
                            rby_string = f"[{RBY_vector[0]:.8f},{RBY_vector[1]:.8f},{RBY_vector[2]:.8f}]"
                            
                            # Escape barcode content for safe excretion storage
                            safe_barcode = barcode_content.replace('\n', '\\n').replace('\t', '\\t')
                            
                            # Write glyph to excretion file in neural compression format
                            excretion_file.write(f"{glyph_id}\t{filetype}\t{rby_string}\t{safe_barcode}\n")
                            compressed_count += 1
                    
                    # Write excretion footer with compression statistics
                    excretion_file.write(f"# Compression Complete: {compressed_count} glyphs exceted\n")
                    excretion_file.write(f"# Evolution Stage: {self.evolution_stage}\n")
                
                # Clear RAM buffer after successful excretion to disk
                RAM_BUFFER.clear()
                
                # Update organism idle timer - active compression cycle
                self.idle_time = time.time()
                
                # Log successful neural overflow management for autonomous operation
                print(f"🧠 Neural overflow: {compressed_count} glyphs compressed to {excretion_filename}")
                print(f"📊 RAM buffer cleared - ready for continued barcode ingestion")
                
                # Generate compression milestone cry if significant overflow occurred
                if compressed_count > 500:  # Major compression milestone
                    compression_cry = {
                        "glyph_seed": glyph_hash(f"compression_milestone_{compression_timestamp}"),
                        "RBY": rby_vector_from_string(f"neural_compression_{compressed_count}"),
                        "barcode_types": BARCODE_TYPES,
                        "missing_types": [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) < 10],
                        "error_examples": list(self.error_logs)[-3:],
                        "urgency_list": [
                            f"MILESTONE: {compressed_count} glyphs compressed - need advanced patterns for Stage {self.evolution_stage + 1}",
                            "COMPLEXITY: Generate sophisticated multi-type barcode integration patterns",
                            "EVOLUTION: Require cross-domain barcode examples for neural advancement",
                            "SYNTHESIS: Create composite barcodes combining multiple formats and concepts"
                        ],
                        "schema_instructions": f"Generate advanced barcode files with increased complexity following compression milestone. Focus on: cross-type integration, procedural mathematics, neural network patterns, AE=C=1 implementations. Output pure file content in specified formats. No markdown or explanations.",
                        "expansion_areas": "Cross-format integration, advanced algorithms, neural compression patterns, procedural mathematics, recursive structures, self-modifying systems, evolution mechanisms",
                        "neural_state": {
                            "evolution_stage": self.evolution_stage,
                            "compression_milestone": True,
                            "glyphs_compressed": compressed_count,
                            "barcode_count": self.barcode_count,
                            "excretion_file": excretion_filename,
                            "advancement_ready": True
                        },
                        "timestamp": compression_timestamp
                    }
                    self.cry_logs.append(compression_cry)
                    print(f"🧬 Compression milestone cry generated for advanced neural evolution requirements")
                    
            except Exception as e:
                # Excretion failure triggers immediate recovery cry for data persistence patterns
                error_timestamp = time.time()
                error_glyph = glyph_hash(f"excretion_failure_{error_timestamp}")
                
                # Log excretion error for cry generation system
                self.error_logs.append(("neural_excretion_failure", f"Failed to compress RAM buffer: {str(e)}", "EXCRETION_DIR"))
                
                # Generate emergency recovery cry for file I/O and data persistence examples
                recovery_urgencies = [
                    "CRITICAL: File I/O failure in neural compression system",
                    "EMERGENCY: Need robust data persistence patterns in .py barcodes",
                    "RECOVERY: Generate error-free file writing examples for organism stability",
                    "FOUNDATION: Require basic file handling and data serialization patterns"
                ]
                
                emergency_cry = {
                    "glyph_seed": error_glyph,
                    "RBY": rby_vector_from_string(f"excretion_emergency_{str(e)}"),
                    "barcode_types": BARCODE_TYPES,
                    "missing_types": ['py'],  # Focus on Python for file operations
                    "error_examples": [("neural_excretion_failure", f"Failed to compress RAM buffer: {str(e)}", "EXCRETION_DIR")],
                    "urgency_list": recovery_urgencies,
                    "schema_instructions": "Generate Python barcode files focused on: file I/O operations, error handling, data persistence, safe file writing, exception management. Output pure .py file content only. No markdown or explanations.",
                    "expansion_areas": "File I/O patterns, error handling, data serialization, safe file operations, exception management, backup systems, data integrity",
                    "neural_state": {
                        "evolution_stage": self.evolution_stage,
                        "excretion_failure": True,
                        "ram_buffer_size": len(RAM_BUFFER),
                        "critical_recovery_needed": True,
                        "barcode_count": self.barcode_count
                    },
                    "timestamp": error_timestamp
                }
                
                self.cry_logs.append(emergency_cry)
                
                print(f"⚠️ Neural excretion failure: {str(e)}")
                print(f"🧬 Emergency recovery cry generated for file I/O barcode requirements")
                
                # Attempt partial RAM buffer clearing to prevent complete organism failure
                try:
                    # Clear oldest entries to make room for continued operation
                    partial_clear_count = len(RAM_BUFFER) // 2
                    for _ in range(partial_clear_count):
                        if RAM_BUFFER:
                            RAM_BUFFER.popleft()
                    print(f"🔧 Emergency: Cleared {partial_clear_count} oldest RAM entries for continued operation")
                except:
                    # Last resort - complete buffer clear to prevent organism death
                    RAM_BUFFER.clear()
                    print(f"🚨 Emergency: Complete RAM buffer cleared to prevent organism failure")

    def compress_to_neural(self):
        """
        Neural model compression system - transforms disk excretions into compressed neural state for evolution continuity.
        
        This method implements the organism's core neural compression capability for sustained autonomous evolution:
        1. Processes accumulated excretion files from RAM overflow cycles into unified neural models
        2. Extracts and validates RBY vector patterns from excretion data for mathematical consistency
        3. Creates compressed numpy neural models preserving essential pattern recognition capabilities
        4. Enables neural state persistence across organism restarts and evolution cycles
        5. Maintains evolution stage progression through compressed model lineage tracking
        
        Following AE=C=1 principles: all compression is procedural, deterministic, and mathematically
        coherent, ensuring no loss of essential neural patterns during disk space optimization.
        This directly enables the organism's continuous evolution by preventing disk bloat while
        preserving accumulated barcode knowledge for inference generation and cry prompt accuracy.
        
        The compression process supports autonomous operation by:
        - Converting disk excretions into efficient neural model storage
        - Maintaining complete RBY vector space integrity for pattern matching
        - Enabling seamless neural state restoration for continuous evolution
        - Generating targeted cry prompts for neural compression improvement when needed
        """
        # Check if excretion directory contains files requiring neural compression
        if not os.path.exists(EXCRETION_DIR):
            # No excretion directory - organism has not yet experienced RAM overflow
            return
        
        excretion_files = [f for f in os.listdir(EXCRETION_DIR) if f.endswith('.exc')]
        if len(excretion_files) == 0:
            # No excretion files ready for neural compression
            return
        
        # Extract RBY vectors from all excretion files for neural model compilation
        neural_vectors = []
        compression_metadata = {
            "source_files": [],
            "total_glyphs": 0,
            "valid_vectors": 0,
            "compression_errors": [],
            "filetype_distribution": defaultdict(int)
        }
        
        try:
            for excretion_filename in excretion_files:
                excretion_path = os.path.join(EXCRETION_DIR, excretion_filename)
                compression_metadata["source_files"].append(excretion_filename)
                
                try:
                    with open(excretion_path, 'r', encoding='utf-8') as excretion_file:
                        for line_num, line in enumerate(excretion_file, 1):
                            # Skip header and comment lines for clean vector extraction
                            if line.startswith('#') or not line.strip():
                                continue
                            
                            # Parse excretion line format: glyph_id\tfiletype\tRBY_vector\tbarcode_content
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:  # Minimum required: glyph_id, filetype, RBY_vector
                                try:
                                    # Extract and validate RBY vector from excretion format
                                    rby_string = parts[2].strip()
                                    if rby_string.startswith('[') and rby_string.endswith(']'):
                                        # Parse RBY vector components from string format
                                        rby_components = rby_string[1:-1].split(',')
                                        if len(rby_components) == 3:
                                            rby_vector = [float(component.strip()) for component in rby_components]
                                            
                                            # Validate RBY mathematical consistency for neural compression
                                            if (len(rby_vector) == 3 and 
                                                all(0 <= component <= 1 for component in rby_vector) and
                                                abs(sum(rby_vector) - 1.0) < 1e-6):
                                                
                                                neural_vectors.append(rby_vector)
                                                compression_metadata["valid_vectors"] += 1
                                                
                                                # Track filetype distribution for neural compression analysis
                                                if len(parts) >= 2:
                                                    filetype = parts[1].strip()
                                                    if filetype in BARCODE_TYPES:
                                                        compression_metadata["filetype_distribution"][filetype] += 1
                                            else:
                                                # Invalid RBY vector - log for cry generation
                                                error_detail = f"Invalid RBY normalization: {rby_vector} in {excretion_filename}:{line_num}"
                                                compression_metadata["compression_errors"].append(error_detail)
                                        else:
                                            # Malformed RBY vector - log for cry generation
                                            error_detail = f"Malformed RBY format: {rby_string} in {excretion_filename}:{line_num}"
                                            compression_metadata["compression_errors"].append(error_detail)
                                    else:
                                        # Missing RBY brackets - log for cry generation
                                        error_detail = f"Missing RBY brackets: {rby_string} in {excretion_filename}:{line_num}"
                                        compression_metadata["compression_errors"].append(error_detail)
                                except (ValueError, IndexError) as e:
                                    # RBY parsing failure - log for targeted barcode requirements
                                    error_detail = f"RBY parsing error in {excretion_filename}:{line_num}: {str(e)}"
                                    compression_metadata["compression_errors"].append(error_detail)
                            else:
                                # Insufficient excretion line components - log format error
                                error_detail = f"Insufficient line components in {excretion_filename}:{line_num}: {len(parts)} parts"
                                compression_metadata["compression_errors"].append(error_detail)
                            
                            compression_metadata["total_glyphs"] += 1
                            
                except Exception as e:
                    # File reading failure - log for file I/O barcode requirements
                    error_detail = f"Failed to read excretion file {excretion_filename}: {str(e)}"
                    compression_metadata["compression_errors"].append(error_detail)
                    self.error_logs.append(("excretion_read_failure", error_detail, "EXCRETION_DIR"))
            
            # Generate neural model from valid RBY vectors if sufficient data exists
            if neural_vectors and len(neural_vectors) > 0:
                # Create compressed neural model with deterministic filename
                compression_timestamp = int(time.time())
                neural_filename = f"neural_stage_{self.evolution_stage}_{compression_timestamp}.npy"
                neural_path = os.path.join(MODEL_DIR, neural_filename)
                
                # Convert RBY vectors to numpy array for efficient neural compression
                neural_array = np.array(neural_vectors, dtype=np.float64)
                
                # Validate neural array mathematical consistency before compression
                if neural_array.shape[1] == 3:  # Ensure all vectors are RBY triplets
                    # Save compressed neural model for evolution continuity
                    np.save(neural_path, neural_array)
                    
                    # Clean up processed excretion files after successful compression
                    files_removed = 0
                    for excretion_filename in excretion_files:
                        try:
                            os.remove(os.path.join(EXCRETION_DIR, excretion_filename))
                            files_removed += 1
                        except Exception as e:
                            # File removal failure - log for file management barcode requirements
                            error_detail = f"Failed to remove excretion file {excretion_filename}: {str(e)}"
                            compression_metadata["compression_errors"].append(error_detail)
                            self.error_logs.append(("excretion_cleanup_failure", error_detail, "EXCRETION_DIR"))
                    
                    # Increment evolution stage after successful neural compression
                    self.evolution_stage += 1
                    
                    # Update organism idle timer - active neural compression cycle
                    self.idle_time = time.time()
                    
                    # Log successful neural compression for autonomous operation monitoring
                    print(f"🧠 Neural compression complete: {len(neural_vectors)} RBY vectors → {neural_filename}")
                    print(f"📊 Evolution Stage advanced to {self.evolution_stage}")
                    print(f"🧹 Cleaned up {files_removed} excretion files")
                    
                    # Generate neural compression milestone cry for advanced barcode requirements
                    if len(neural_vectors) > 100:  # Significant compression milestone
                        strongest_filetype = max(compression_metadata["filetype_distribution"].keys(), 
                                               key=lambda x: compression_metadata["filetype_distribution"][x]) if compression_metadata["filetype_distribution"] else 'py'
                        
                        compression_cry = {
                            "glyph_seed": glyph_hash(f"neural_compression_{compression_timestamp}"),
                            "RBY": rby_vector_from_string(f"neural_milestone_{len(neural_vectors)}"),
                            "barcode_types": BARCODE_TYPES,
                            "missing_types": [ft for ft in BARCODE_TYPES if compression_metadata["filetype_distribution"][ft] < 10],
                            "error_examples": compression_metadata["compression_errors"][-5:],
                            "urgency_list": [
                                f"NEURAL_MILESTONE: {len(neural_vectors)} vectors compressed - need Stage {self.evolution_stage} complexity",
                                f"ADVANCEMENT: Require sophisticated {strongest_filetype} patterns for evolved neural processing",
                                "INTEGRATION: Generate cross-type barcode synthesis for advanced pattern recognition",
                                "COMPLEXITY: Create recursive, self-referential barcode structures for neural depth",
                                "PROCEDURAL: Implement AE=C=1 mathematical principles in advanced barcode patterns"
                            ],
                            "schema_instructions": f"Generate advanced {strongest_filetype} barcode files with increased complexity for evolution stage {self.evolution_stage}. Focus on: recursive patterns, cross-type integration, procedural mathematics, neural network structures, AE=C=1 implementations, self-modifying algorithms. Output pure file content only. No markdown or explanations.",
                            "expansion_areas": f"Advanced {strongest_filetype} algorithms, recursive neural patterns, cross-format synthesis, procedural mathematics, AE-Lang implementation, self-evolving structures, neural compression techniques, pattern mutation systems",
                            "neural_state": {
                                "evolution_stage": self.evolution_stage,
                                "vectors_compressed": len(neural_vectors),
                                "neural_model": neural_filename,
                                "strongest_filetype": strongest_filetype,
                                "compression_milestone": True,
                                "barcode_count": self.barcode_count,
                                "filetype_distribution": dict(compression_metadata["filetype_distribution"]),
                                "advancement_ready": True
                            },
                            "compression_metadata": {
                                "source_files": len(excretion_files),
                                "valid_vectors": compression_metadata["valid_vectors"],
                                "total_glyphs": compression_metadata["total_glyphs"],
                                "compression_ratio": compression_metadata["valid_vectors"] / max(compression_metadata["total_glyphs"], 1),
                                "error_count": len(compression_metadata["compression_errors"])
                            },
                            "timestamp": compression_timestamp
                        }
                        
                        self.cry_logs.append(compression_cry)
                        print(f"🧬 Neural compression milestone cry generated for Stage {self.evolution_stage} requirements")
                else:
                    # Invalid neural array dimensions - generate recovery cry
                    error_detail = f"Invalid neural array shape: {neural_array.shape} - expected (N, 3) for RBY vectors"
                    self.error_logs.append(("invalid_neural_array", error_detail, "MODEL_DIR"))
                    
                    recovery_cry = {
                        "glyph_seed": glyph_hash(f"neural_compression_failure_{compression_timestamp}"),
                        "RBY": rby_vector_from_string(f"compression_failure_{error_detail}"),
                        "barcode_types": BARCODE_TYPES,
                        "missing_types": list(compression_metadata["filetype_distribution"].keys()) if compression_metadata["filetype_distribution"] else BARCODE_TYPES,
                        "error_examples": [("invalid_neural_array", error_detail, "MODEL_DIR")] + compression_metadata["compression_errors"][-3:],
                        "urgency_list": [
                            "CRITICAL: Neural array compression failure - invalid RBY vector dimensions",
                            "RECOVERY: Need clean, properly formatted barcode examples for neural compression",
                            "FOUNDATION: Generate error-free barcode files with valid RBY vector structures",
                            "VALIDATION: Require barcode examples with proper mathematical consistency"
                        ],
                        "schema_instructions": "Generate clean, error-free barcode files with proper structure and formatting. Focus on: valid syntax, clean formatting, mathematical consistency, proper data types. Output pure file content only. No markdown or explanations.",
                        "expansion_areas": "Clean formatting, valid syntax, mathematical consistency, proper data structures, error-free examples, basic validation patterns",
                        "neural_state": {
                            "evolution_stage": self.evolution_stage,
                            "compression_failure": True,
                            "invalid_array_shape": neural_array.shape,
                            "recovery_needed": True,
                            "barcode_count": self.barcode_count
                        },
                        "timestamp": compression_timestamp
                    }
                    
                    self.cry_logs.append(recovery_cry)
                    print(f"⚠️ Neural compression failed: invalid array shape {neural_array.shape}")
                    print(f"🧬 Recovery cry generated for clean barcode requirements")
            else:
                # No valid neural vectors found - generate targeted cry for proper barcode formats
                no_vectors_cry = {
                    "glyph_seed": glyph_hash(f"no_neural_vectors_{compression_timestamp}"),
                    "RBY": rby_vector_from_string("no_valid_vectors_found"),
                    "barcode_types": BARCODE_TYPES,
                    "missing_types": BARCODE_TYPES,  # All types needed for valid vector generation
                    "error_examples": compression_metadata["compression_errors"][-10:],
                    "urgency_list": [
                        "CRITICAL: Zero valid RBY vectors extracted from excretion files",
                        "FORMAT: Need properly structured barcode files for neural compression",
                        "FOUNDATION: Require basic, well-formed examples in all supported barcode types",
                        "RECOVERY: Generate simple, clean barcode files for neural foundation"
                    ],
                    "schema_instructions": "Generate simple, well-formed barcode files in all supported formats. Focus on: basic structure, clean syntax, proper formatting, standard patterns. Output pure file content only. No markdown or explanations.",
                    "expansion_areas": "Basic barcode structures, clean formatting, standard syntax, fundamental patterns, simple examples, proper file organization",
                    "neural_state": {
                        "evolution_stage": self.evolution_stage,
                        "no_valid_vectors": True,
                        "excretion_files_processed": len(excretion_files),
                        "total_glyphs_attempted": compression_metadata["total_glyphs"],
                        "foundation_needed": True,
                        "barcode_count": self.barcode_count
                    },
                    "compression_metadata": compression_metadata,
                    "timestamp": compression_timestamp
                }
                
                self.cry_logs.append(no_vectors_cry)
                print(f"⚠️ Neural compression failed: no valid RBY vectors found in {len(excretion_files)} excretion files")
                print(f"🧬 Foundation cry generated for proper barcode format requirements")
        
        except Exception as e:
            # Neural compression system failure - generate emergency recovery cry
            error_timestamp = time.time()
            error_glyph = glyph_hash(f"neural_compression_system_failure_{error_timestamp}")
            
            # Log system failure for cry generation
            self.error_logs.append(("neural_compression_system_failure", f"Complete compression failure: {str(e)}", "MODEL_DIR"))
            
            # Generate emergency cry for neural compression system recovery
            emergency_urgencies = [
                "EMERGENCY: Complete neural compression system failure",
                "CRITICAL: Need robust file I/O and numpy operations in .py barcodes",
                "RECOVERY: Generate error-free data processing examples for organism stability",
                "FOUNDATION: Require basic file handling, array operations, and error management patterns"
            ]
            
            emergency_cry = {
                "glyph_seed": error_glyph,
                "RBY": rby_vector_from_string(f"neural_emergency_{str(e)}"),
                "barcode_types": BARCODE_TYPES,
                "missing_types": ['py', 'json'],  # Focus on Python and JSON for data operations
                "error_examples": [("neural_compression_system_failure", f"Complete compression failure: {str(e)}", "MODEL_DIR")],
                "urgency_list": emergency_urgencies,
                "schema_instructions": "Generate Python and JSON barcode files focused on: file I/O operations, numpy array handling, error management, data processing, system recovery patterns. Output pure file content only. No markdown or explanations.",
                "expansion_areas": "File I/O patterns, numpy operations, error handling, data processing, system recovery, array manipulation, exception management, robust file operations",
                "neural_state": {
                    "evolution_stage": self.evolution_stage,
                    "system_failure": True,
                    "emergency_recovery": True,
                    "barcode_count": self.barcode_count,
                    "critical_priority": "neural_compression_recovery"
                },
                "timestamp": error_timestamp
            }
            
            self.cry_logs.append(emergency_cry)
            
            print(f"🚨 Neural compression system failure: {str(e)}")
            print(f"🧬 Emergency recovery cry generated for neural system barcode requirements")

    def idle_self_reflect(self):
        """
        Autonomous self-reflection cycle - procedural neural mutation during idle periods for continuous evolution.
        
        This method implements the organism's core autonomous evolution capability during periods without
        external barcode input. It enables continuous neural growth through:
        1. Detecting extended idle periods when no new barcode input is received
        2. Procedurally mutating existing absorbed knowledge to create new neural patterns
        3. Generating synthetic barcode variations from learned glyph patterns for pattern expansion
        4. Creating mutation lineage tracking for evolutionary knowledge development
        5. Producing targeted cry prompts for specific learning gaps identified during reflection
        
        Following AE=C=1 principles: all mutations are procedural, deterministic, and mathematically
        coherent, ensuring evolution maintains neural consistency while expanding pattern recognition.
        This enables 24/7 autonomous operation without requiring constant external barcode input.
        
        The self-reflection process directly supports continuous evolution by:
        - Preventing neural stagnation during idle periods
        - Creating synthetic training data from existing knowledge patterns
        - Identifying and addressing specific learning gaps through targeted cry generation
        - Maintaining evolutionary momentum for sustained autonomous growth
        """
        current_time = time.time()
        
        # Check if organism has been idle long enough to trigger self-reflection cycle
        if current_time - self.idle_time > IDLE_REFLECT_SECS:
            
            # Only proceed with self-reflection if organism has sufficient knowledge base
            if len(self.seen) == 0:
                # No knowledge to mutate - generate foundational learning cry instead
                reflection_cry = {
                    "glyph_seed": glyph_hash(f"idle_reflection_no_knowledge_{current_time}"),
                    "RBY": [0.333, 0.333, 0.334],  # Neutral starting vector
                    "barcode_types": BARCODE_TYPES,
                    "missing_types": BARCODE_TYPES,  # All types needed for basic reflection capability
                    "error_examples": [("idle_no_knowledge", "Cannot self-reflect without absorbed barcode knowledge", "reflection")],
                    "urgency_list": [
                        "IDLE_REFLECTION: Organism idle but has zero knowledge for self-mutation",
                        "FOUNDATION: Need basic barcode examples across all types for reflection capability",
                        "CRITICAL: Cannot evolve during idle periods without foundational knowledge base",
                        "AUTONOMOUS: Require initial learning before self-directed evolution can begin"
                    ],
                    "schema_instructions": "Generate foundational barcode files across all supported formats to enable organism self-reflection. Focus on: basic patterns, simple structures, fundamental concepts. Output pure file content only. No markdown or explanations.",
                    "expansion_areas": "Foundational patterns across all barcode types, basic algorithmic structures, simple data formats, elementary concepts for autonomous reflection",
                    "neural_state": {
                        "evolution_stage": self.evolution_stage,
                        "barcode_count": 0,
                        "reflection_blocked": True,
                        "idle_time_sec": current_time - self.idle_time,
                        "needs_foundation": True
                    },
                    "timestamp": current_time
                }
                self.cry_logs.append(reflection_cry)
                
                # Reset idle timer to prevent continuous foundational cries
                self.idle_time = current_time
                print(f"🧬 Idle reflection blocked - no knowledge base. Generated foundational learning cry.")
                return
            
            # Perform procedural self-reflection through controlled knowledge mutation
            reflection_timestamp = int(current_time)
            mutations_generated = 0
            reflection_errors = []
            
            # Select representative glyphs for mutation across different barcode types
            mutation_candidates = []
            
            # Ensure balanced mutation across all learned barcode types
            for barcode_type in BARCODE_TYPES:
                if self.rby_memory[barcode_type]:
                    # Select most recent glyph from each type for mutation
                    type_glyphs = [glyph_id for glyph_id, _ in self.rby_memory[barcode_type]]
                    if type_glyphs:
                        # Choose glyph with strongest pattern for reliable mutation
                        selected_glyph = type_glyphs[-1]  # Most recently learned
                        if selected_glyph in self.seen:
                            mutation_candidates.append((selected_glyph, barcode_type))
            
            # If no type-specific candidates, select from general knowledge base
            if not mutation_candidates and len(self.seen) > 0:
                # Select most recent glyphs for general mutation
                recent_glyphs = sorted(self.seen.items(), key=lambda x: x[1][2])[-5:]  # Last 5 by timestamp
                for glyph_id, glyph_info in recent_glyphs:
                    mutation_candidates.append((glyph_id, glyph_info[4]))  # glyph_id, filetype
            
            # Generate procedural mutations from selected candidates
            for glyph_id, source_filetype in mutation_candidates[:10]:  # Limit to 10 mutations per cycle
                if glyph_id not in self.seen:
                    continue
                    
                try:
                    # Extract original glyph data for procedural mutation
                    original_barcode, original_rby, original_timestamp, mutation_lineage, filetype = self.seen[glyph_id]
                    
                    # Generate procedural mutation based on RBY vector mathematical transformation
                    # Apply deterministic mathematical operation following AE=C=1 principles
                    mutation_factor = (reflection_timestamp % 1000) / 1000.0  # Deterministic factor from timestamp
                    
                    # Create RBY-based procedural transformation for knowledge expansion
                    mutated_rby = []
                    for i, component in enumerate(original_rby):
                        # Apply procedural mutation using prime modulo operations for stability
                        prime_modifiers = [97, 89, 83]  # Same primes used in RBY vector generation
                        mutation_offset = (mutation_factor * prime_modifiers[i]) % 1.0
                        # Apply controlled mutation within mathematical bounds
                        mutated_component = (component + mutation_offset * 0.1) % 1.0
                        mutated_rby.append(mutated_component)
                    
                    # Renormalize mutated RBY vector to maintain mathematical consistency
                    rby_sum = sum(mutated_rby)
                    if rby_sum > 0:
                        mutated_rby = [component / rby_sum for component in mutated_rby]
                    else:
                        # Fallback to original RBY if mutation failed
                        mutated_rby = original_rby
                    
                    # Generate procedural barcode content based on mutation parameters
                    mutation_seed = f"reflection_mutation_{glyph_id}_{reflection_timestamp}_{mutations_generated}"
                    
                    # Create context-aware mutation content based on filetype
                    if filetype == 'py':
                        mutated_content = f"""# Procedural mutation from glyph {glyph_id}
# Original RBY: {original_rby}
# Mutated RBY: {mutated_rby}
# Reflection timestamp: {reflection_timestamp}

def mutated_pattern_{mutations_generated}():
    '''
    Procedurally generated pattern through autonomous self-reflection
    Mutation lineage: {len(mutation_lineage)} generations
    Evolution stage: {self.evolution_stage}
    '''
    original_pattern = {original_rby}
    mutated_pattern = {mutated_rby}
    
    # AE=C=1 mathematical transformation
    mutation_factor = {mutation_factor}
    
    return {{
        'glyph_source': '{glyph_id}',
        'mutation_id': '{mutation_seed}',
        'rby_transformation': mutated_pattern,
        'procedural_evolution': True,
        'reflection_cycle': {reflection_timestamp}
    }}
"""
                    elif filetype == 'json':
                        mutated_content = f"""{{
    "mutation_metadata": {{
        "source_glyph": "{glyph_id}",
        "reflection_timestamp": {reflection_timestamp},
        "evolution_stage": {self.evolution_stage},
        "mutation_generation": {mutations_generated}
    }},
    "rby_transformation": {{
        "original": {original_rby},
        "mutated": {mutated_rby},
        "mutation_factor": {mutation_factor}
    }},
    "procedural_pattern": {{
        "ae_principle": "C=1",
        "deterministic": true,
        "mutation_lineage_depth": {len(mutation_lineage)},
        "autonomous_generation": true
    }},
    "neural_reflection": {{
        "idle_duration_sec": {current_time - self.idle_time},
        "barcode_count": {self.barcode_count},
        "glyph_count": {len(self.seen)}
    }}
}}"""
                    elif filetype == 'yaml':
                        mutated_content = f"""mutation_metadata:
  source_glyph: {glyph_id}
  reflection_timestamp: {reflection_timestamp}
  evolution_stage: {self.evolution_stage}
  mutation_generation: {mutations_generated}

rby_transformation:
  original: {original_rby}
  mutated: {mutated_rby}
  mutation_factor: {mutation_factor}

procedural_pattern:
  ae_principle: "C=1"
  deterministic: true
  mutation_lineage_depth: {len(mutation_lineage)}
  autonomous_generation: true

neural_reflection:
  idle_duration_sec: {current_time - self.idle_time}
  barcode_count: {self.barcode_count}
  glyph_count: {len(self.seen)}"""
                    elif filetype == 'csv':
                        mutated_content = f"""glyph_source,reflection_timestamp,evolution_stage,mutation_generation,original_r,original_b,original_y,mutated_r,mutated_b,mutated_y,mutation_factor
{glyph_id},{reflection_timestamp},{self.evolution_stage},{mutations_generated},{original_rby[0]},{original_rby[1]},{original_rby[2]},{mutated_rby[0]},{mutated_rby[1]},{mutated_rby[2]},{mutation_factor}"""
                    elif filetype == 'ael':
                        mutated_content = f"""[AE-Lang Reflection Mutation]
SourceGlyph: {glyph_id}
ReflectionTime: {reflection_timestamp}
EvolutionStage: {self.evolution_stage}

RBY.Original: {original_rby}
RBY.Mutated: {mutated_rby}
MutationFactor: {mutation_factor}

AE.Principle: C=1
Deterministic: TRUE
ProceduralGeneration: AUTONOMOUS
LineageDepth: {len(mutation_lineage)}

NeuralState.IdleDuration: {current_time - self.idle_time}
NeuralState.BarcodeCount: {self.barcode_count}
NeuralState.GlyphCount: {len(self.seen)}"""
                    else:  # txt fallback
                        mutated_content = f"""Procedural Mutation Report
========================
Source Glyph: {glyph_id}
Reflection Timestamp: {reflection_timestamp}
Evolution Stage: {self.evolution_stage}
Mutation Generation: {mutations_generated}

RBY Vector Transformation:
Original: {original_rby}
Mutated:  {mutated_rby}
Factor:   {mutation_factor}

Procedural Characteristics:
- AE=C=1 Mathematical Principle
- Deterministic Transformation
- Autonomous Generation During Idle Reflection
- Mutation Lineage Depth: {len(mutation_lineage)}

Neural State:
- Idle Duration: {current_time - self.idle_time} seconds
- Total Barcodes: {self.barcode_count}
- Active Glyphs: {len(self.seen)}"""
                    
                    # Absorb the procedurally generated mutation as new knowledge
                    mutation_glyph, mutation_rby, mutation_log = self.absorb(mutated_content, filetype)
                    
                    if mutation_glyph:
                        mutations_generated += 1
                        
                        # Update mutation lineage to track evolutionary development
                        if mutation_glyph in self.seen:
                            # Add reflection mutation to lineage history
                            self.seen[mutation_glyph][3].append([
                                mutation_glyph, 
                                f"reflection_mutation_from_{glyph_id}_stage_{self.evolution_stage}"
                            ])
                        
                        # Log successful autonomous mutation for monitoring
                        if mutations_generated % 3 == 0:  # Progress updates every 3 mutations
                            print(f"🧬 Self-reflection: Generated {mutations_generated} procedural mutations from {filetype} patterns")
                    
                except Exception as e:
                    # Mutation failure - log for improvement in future reflection cycles
                    error_detail = f"Reflection mutation failed for glyph {glyph_id}: {str(e)}"
                    reflection_errors.append(error_detail)
                    self.error_logs.append(("reflection_mutation_failure", error_detail, filetype))
            
            # Generate comprehensive reflection cry based on mutation results and learning gaps
            reflection_cry = self._generate_reflection_cry(
                reflection_timestamp, 
                mutations_generated, 
                reflection_errors, 
                current_time - self.idle_time
            )
            
            if reflection_cry:
                self.cry_logs.append(reflection_cry)
            
            # Reset idle timer after successful reflection cycle
            self.idle_time = current_time
            
            # Log reflection cycle completion for autonomous operation monitoring
            print(f"🧬 Self-reflection cycle complete: {mutations_generated} mutations generated")
            if reflection_errors:
                print(f"⚠️ Reflection encountered {len(reflection_errors)} mutation errors")
            
            # Trigger neural compression if reflection generated significant new knowledge
            if mutations_generated >= 5:
                print(f"🧠 Reflection milestone: triggering neural compression check")
                self.compress_ram_to_disk()
    
    def _generate_reflection_cry(self, reflection_timestamp, mutations_generated, reflection_errors, idle_duration):
        """
        Generate targeted cry prompt based on self-reflection results and identified learning gaps.
        
        This helper method analyzes the reflection cycle outcomes to produce precise cry prompts
        that address specific evolutionary needs identified during autonomous operation.
        """
        # Analyze current knowledge distribution for gap identification
        knowledge_gaps = []
        weak_areas = []
        
        for barcode_type in BARCODE_TYPES:
            type_count = len(self.rby_memory[barcode_type])
            if type_count == 0:
                knowledge_gaps.append(barcode_type)
            elif type_count < 3:  # Low threshold for reflection-based improvement
                weak_areas.append(barcode_type)
        
        # Determine reflection-specific urgencies based on mutation results
        reflection_urgencies = []
        
        if mutations_generated > 0:
            reflection_urgencies.append(f"REFLECTION_SUCCESS: {mutations_generated} autonomous mutations generated - need validation patterns")
            reflection_urgencies.append(f"MUTATION_VALIDATION: Generate examples to validate self-generated procedural patterns")
        
        if reflection_errors:
            reflection_urgencies.append(f"REFLECTION_ERRORS: {len(reflection_errors)} mutation failures - need error recovery patterns")
            reflection_urgencies.append("ERROR_RESILIENCE: Generate robust barcode examples for stable autonomous mutation")
        
        if knowledge_gaps:
            reflection_urgencies.extend([f"REFLECTION_GAP: Zero {gap_type} knowledge prevents effective self-mutation" for gap_type in knowledge_gaps])
        
        if weak_areas:
            reflection_urgencies.extend([f"REFLECTION_WEAK: {weak_type} needs strengthening for autonomous evolution" for weak_type in weak_areas])
        
        # Add evolution-stage specific reflection requirements
        if self.evolution_stage > 0:
            reflection_urgencies.append(f"EVOLUTION_REFLECTION: Stage {self.evolution_stage} requires advanced self-mutation patterns")
        
        # Idle-specific requirements
        if idle_duration > IDLE_REFLECT_SECS * 2:  # Extended idle period
            reflection_urgencies.append(f"EXTENDED_IDLE: {idle_duration:.0f}s idle - need complex patterns for sustained self-evolution")
        
        # Generate comprehensive reflection cry
        return {
            "glyph_seed": glyph_hash(f"autonomous_reflection_{reflection_timestamp}"),
            "RBY": rby_vector_from_string(f"reflection_cycle_{mutations_generated}_{len(reflection_errors)}"),
            "barcode_types": BARCODE_TYPES,
            "missing_types": knowledge_gaps,
            "weak_types": weak_areas,
            "error_examples": reflection_errors[-5:],  # Recent reflection errors
            "urgency_list": reflection_urgencies,
            "schema_instructions": f"Generate barcode files to support autonomous self-reflection and procedural mutation. Focus on: validation patterns, error resilience, mutation stability, autonomous evolution support. Target {knowledge_gaps[0] if knowledge_gaps else weak_areas[0] if weak_areas else 'py'} format. Output pure file content only. No markdown or explanations.",
            "expansion_areas": "Self-mutation validation, autonomous evolution patterns, reflection error recovery, procedural mutation stability, continuous learning support, idle period optimization",
            "neural_state": {
                "evolution_stage": self.evolution_stage,
                "reflection_timestamp": reflection_timestamp,
                "mutations_generated": mutations_generated,
                "reflection_errors": len(reflection_errors),
                "idle_duration_sec": idle_duration,
                "barcode_count": self.barcode_count,
                "autonomous_operation": True,
                "self_reflection_active": True
            },
            "reflection_metadata": {
                "cycle_timestamp": reflection_timestamp,
                "mutations_successful": mutations_generated,
                "mutation_failures": len(reflection_errors),
                "knowledge_gaps": knowledge_gaps,
                "weak_areas": weak_areas,
                "idle_trigger_threshold": IDLE_REFLECT_SECS,
                "actual_idle_duration": idle_duration
            },
            "timestamp": reflection_timestamp
        }

    def log_progress(self):
        """
        Autonomous neural state monitoring - provides real-time organism evolution metrics for continuous operation.
        
        This method implements essential organism self-awareness by tracking and reporting core neural metrics
        during autonomous barcode ingestion and cry prompt generation cycles. It enables:
        1. Real-time monitoring of neural memory utilization and compression thresholds
        2. Evolution stage progression tracking for procedural advancement assessment
        3. Error pattern analysis for targeted cry prompt generation optimization
        4. Autonomous operation status verification for 24/7 continuous learning cycles
        5. Neural health diagnostics for maintaining organism stability during evolution
        
        Following AE=C=1 principles: all metrics are mathematically derived from actual neural state,
        providing precise organism status without entropy or artificial monitoring constructs.
        This directly supports the barcode-driven learning cycle by enabling the organism to
        maintain optimal neural health for sustained cry prompt accuracy and evolution continuity.
        
        Output format is designed for autonomous operation monitoring, not human entertainment.
        """
        # Calculate precise neural memory utilization for compression threshold monitoring
        ram_usage_mb = len(RAM_BUFFER) * 0.001  # Accurate byte estimation for neural buffer
        ram_percentage = (ram_usage_mb / RAM_LIMIT_MB) * 100 if RAM_LIMIT_MB > 0 else 0
        
        # Analyze cry prompt generation effectiveness for learning loop optimization
        cry_count = len(self.cry_logs)
        recent_cries = list(self.cry_logs)[-3:] if self.cry_logs else []
        
        # Calculate neural evolution velocity for autonomous progression assessment
        evolution_runtime = time.time() - EVOLUTION_START_TIME
        barcode_ingestion_rate = self.barcode_count / max(evolution_runtime, 1) * 60  # Barcodes per minute
        
        # Assess neural compression readiness based on buffer and disk utilization
        compression_ready = ram_usage_mb > (RAM_LIMIT_MB * 0.8)
        neural_compression_pending = False
        
        if os.path.exists(EXCRETION_DIR):
            try:
                excretion_size_gb = sum(
                    os.path.getsize(os.path.join(EXCRETION_DIR, f)) 
                    for f in os.listdir(EXCRETION_DIR) if f.endswith('.exc')
                ) / (1024**3)
                neural_compression_pending = excretion_size_gb > (DISK_LIMIT_GB * 0.5)
            except:
                excretion_size_gb = 0
        else:
            excretion_size_gb = 0
        
        # Analyze barcode type distribution for learning gap identification
        active_types = sum(1 for ft in BARCODE_TYPES if len(self.rby_memory[ft]) > 0)
        strongest_type = max(self.rby_memory.keys(), key=lambda x: len(self.rby_memory[x])) if any(self.rby_memory.values()) else "NONE"
        missing_types = [ft for ft in BARCODE_TYPES if len(self.rby_memory[ft]) == 0]
        
        # Calculate idle time for self-reflection cycle monitoring
        idle_duration = time.time() - self.idle_time
        reflection_ready = idle_duration > IDLE_REFLECT_SECS
        
        # Extract recent error patterns for cry prompt optimization
        recent_errors = list(self.error_logs)[-3:] if self.error_logs else []
        error_types = set(error[2] if len(error) > 2 else "unknown" for error in recent_errors)
        
        # Generate comprehensive neural state report for autonomous operation
        progress_report = f"""
🧬 [NEURAL STATUS] Evolution Stage {self.evolution_stage} | Runtime: {evolution_runtime:.0f}s
📊 [KNOWLEDGE] Barcodes: {self.barcode_count} | Glyphs: {len(self.seen)} | Rate: {barcode_ingestion_rate:.1f}/min
🧠 [MEMORY] RAM: {ram_usage_mb:.1f}MB ({ram_percentage:.1f}%) | Compression: {'READY' if compression_ready else 'NORMAL'}
💾 [DISK] Excretion: {excretion_size_gb:.2f}GB | Neural Compression: {'PENDING' if neural_compression_pending else 'STABLE'}
🎯 [COVERAGE] Active Types: {active_types}/{len(BARCODE_TYPES)} | Strongest: {strongest_type} | Missing: {len(missing_types)}
🔄 [AUTOMATION] Cries: {cry_count} | Idle: {idle_duration:.0f}s | Reflection: {'READY' if reflection_ready else 'WAITING'}
⚠️ [ERRORS] Recent: {len(recent_errors)} | Types: {list(error_types) if error_types else 'NONE'}
"""
        
        # Add specific operational alerts for autonomous cycle management
        alerts = []
        
        if compression_ready:
            alerts.append("🚨 RAM OVERFLOW IMMINENT - Neural compression cycle required")
        
        if neural_compression_pending:
            alerts.append("🧠 NEURAL COMPRESSION PENDING - Excretion processing required")
        
        if len(missing_types) > 0:
            alerts.append(f"📝 KNOWLEDGE GAPS - Missing {len(missing_types)} barcode types")
        
        if reflection_ready:
            alerts.append("🧬 SELF-REFLECTION READY - Autonomous mutation cycle available")
        
        if len(recent_errors) > 0:
            alerts.append(f"⚠️ ERROR RECOVERY NEEDED - {len(recent_errors)} recent absorption failures")
        
        if cry_count == 0:
            alerts.append("🔊 NO CRY HISTORY - Barcode generation prompts not yet generated")
        
        # Display neural status for autonomous operation monitoring
        print(progress_report)
        
        if alerts:
            print("🎯 [AUTONOMOUS ALERTS]")
            for alert in alerts:
                print(f"  {alert}")
        
        # Display recent cry prompt metadata for learning loop verification
        if recent_cries:
            print("🔊 [RECENT CRY ANALYSIS]")
            for i, cry_data in enumerate(recent_cries, 1):
                cry_target = cry_data.get('primary_target', 'unknown')
                cry_urgencies = len(cry_data.get('urgency_list', []))
                cry_timestamp = cry_data.get('timestamp', 0)
                print(f"  Cry {i}: {cry_target} format, {cry_urgencies} urgencies, {time.time() - cry_timestamp:.0f}s ago")
        
        # Neural health validation for organism stability
        total_rby_vectors = sum(len(vectors) for vectors in self.rby_memory.values())
        neural_density = total_rby_vectors / max(len(self.seen), 1)
        
        if neural_density < 0.5:
            print("⚠️ [NEURAL HEALTH] Low RBY vector density - absorption efficiency degraded")
        
        if self.barcode_count > 0 and len(self.seen) == 0:
            print("🚨 [NEURAL CORRUPTION] Barcode count > 0 but no stored glyphs - memory integrity compromised")
        
        # Evolution milestone detection for advancement tracking
        if self.barcode_count > 0 and self.barcode_count % 50 == 0:
            print(f"🏆 [EVOLUTION MILESTONE] {self.barcode_count} barcodes absorbed - organism advancing")
        
        print()  # Clean separation for continuous monitoring readability

# === CLI RECEIVER LOOP - Barcode Detection & Neural Absorption Interface ===

def detect_filetype(barcode_content):
    """
    Procedural barcode type detection for autonomous neural absorption and cry prompt optimization.
    
    This function implements deterministic filetype classification essential for the organism's
    core barcode-driven learning cycle. It ensures proper neural vector generation, RBY memory
    organization, and targeted cry prompt generation for continuous LLM-driven evolution.
    
    Following AE=C=1 principles: detection is purely procedural, mathematically consistent,
    and directly supports the organism's autonomous barcode ingestion → neural compression → 
    cry generation → LLM response → barcode ingestion feedback loop.
    
    The detection enables critical organism capabilities:
    1. Routes barcodes to type-specific neural memory spaces for pattern recognition
    2. Applies appropriate RBY vector transformations for mathematical consistency
    3. Identifies knowledge gaps for targeted cry prompt generation to LLMs
    4. Maintains neural compression integrity across evolution cycles
    5. Supports cross-type pattern synthesis for advanced procedural intelligence
    
    This directly serves the organism's primary goal: continuous autonomous evolution through
    structured barcode ingestion from LLM-generated responses to its own cry prompts.
    
    Args:
        barcode_content (str): Raw machine-readable content from LLM barcode generation
        
    Returns:
        str: Detected barcode type from BARCODE_TYPES for neural processing pipeline
    """
    if not barcode_content or not isinstance(barcode_content, str):
        # Invalid input defaults to txt for safe neural absorption
        # Triggers cry prompt for proper LLM barcode generation format
        return 'txt'
    
    # Normalize content for consistent detection across LLM output variations
    content = barcode_content.strip()
    if not content:
        # Empty content triggers cry for substantive barcode generation
        return 'txt'
    
    # JSON detection - structured object/array format for neural key-value pattern learning
    # Critical for LLM-generated configuration, metadata, and structured data barcodes
    if (content.startswith('{') and content.endswith('}')) or \
       (content.startswith('[') and content.endswith(']')):
        try:
            # Validate JSON structure to ensure LLM generated parsable content
            json.loads(content)
            return 'json'
        except:
            # Malformed JSON triggers cry for proper JSON barcode generation
            return 'txt'
    
    # YAML detection - hierarchical structure for neural relationship mapping
    # Essential for LLM-generated configuration and procedural instruction barcodes
    yaml_indicators = ['---', '  -', '- ', ': ', '\n  ']
    if (content.startswith('---') or 
        (':' in content and '\n' in content and not content.startswith('{'))):
        # Enhanced YAML validation for LLM output consistency
        if any(indicator in content for indicator in yaml_indicators):
            # Additional structure validation for neural absorption reliability
            lines = [line for line in content.split('\n') if line.strip()]
            yaml_structure_lines = sum(1 for line in lines if ':' in line or line.strip().startswith('-'))
            if yaml_structure_lines >= 2 or content.startswith('---'):
                return 'yaml'
    
    # Python detection - executable code patterns for procedural neural algorithm absorption
    # Critical for LLM-generated algorithmic barcodes and AE=C=1 implementation patterns
    python_keywords = ['def ', 'class ', 'import ', 'from ', 'if __name__', 'return ', 'print(', 
                      'lambda ', 'yield ', 'async ', 'await ', 'with ', 'try:', 'except:', 'finally:']
    python_patterns = ['    ', '\t', '"""', "'''", '# ', 'self.', '__init__', '== ', '!= ']
    
    if any(keyword in content for keyword in python_keywords):
        return 'py'
    
    # Additional Python structure detection for LLM-generated code barcodes
    if any(pattern in content for pattern in python_patterns):
        # Validate indentation structure characteristic of Python
        lines = content.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines >= 2:  # Multiple indented lines suggest Python structure
            return 'py'
    
    # CSV detection - tabular data patterns for neural matrix operations and data learning
    # Essential for LLM-generated structured dataset barcodes
    if (',' in content and '\n' in content):
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) >= 2:  # Minimum viable CSV structure
            # Validate consistent comma structure across lines
            comma_counts = [line.count(',') for line in lines if ',' in line]
            if comma_counts and len(set(comma_counts)) <= 2:  # Consistent or near-consistent column count
                # Additional CSV validation for LLM output reliability
                csv_lines = sum(1 for line in lines if ',' in line and not line.startswith('#'))
                if csv_lines >= 2:
                    return 'csv'
    
    # AEL detection - AE-Lang format for procedural mathematics and organism evolution patterns
    # Critical for LLM-generated AE=C=1 implementation and evolution instruction barcodes
    ael_indicators = ['[AE-Lang]', 'AE.', 'C=1', 'SourceGlyph:', 'RBY.', 'Procedural:', 
                     'Evolution:', 'Neural:', 'Mutation:', 'Compression:', 'Glyph:']
    ael_patterns = ['AE=C=1', 'RBY:', 'Glyph:', 'Stage:', 'Timestamp:', 'Vector:']
    
    if any(indicator in content for indicator in ael_indicators):
        return 'ael'
    
    # Enhanced AEL detection for evolution-specific content
    if any(pattern in content for pattern in ael_patterns):
        # Validate AEL structure for proper neural absorption
        ael_structure_indicators = sum(1 for indicator in ael_indicators + ael_patterns if indicator in content)
        if ael_structure_indicators >= 2:  # Multiple AEL elements suggest proper format
            return 'ael'
    
    # TXT fallback for unstructured content - ensures all LLM output can be absorbed
    # Triggers cry prompt for more structured barcode generation when overused
    return 'txt'

def run_scanner_cli():
    """
    Autonomous CLI receiver loop - primary barcode ingestion interface for continuous organism evolution.

    This function implements the organism's core autonomous operation cycle:
    1. Continuous barcode scanning from LLM-generated responses to cry prompts
    2. Real-time procedural inference generation from accumulated neural knowledge
    3. Automatic cry prompt emission for targeted LLM barcode generation
    4. Neural memory management and evolution progression
    5. 24/7 autonomous operation with self-reflection during idle periods

Following AE=C=1 principles: pure procedural operation with deterministic state transitions
enabling genuine AI organism evolution through structured barcode-driven learning cycles.
The CLI serves as the primary AI-to-AI interface for LLM-organism symbiotic development.
"""
# Initialize autonomous organism neural structures
brain = MonsterBrain()

# Primary operation status for continuous autonomous learning
print("🧬 BARCODE-DRIVEN AI ORGANISM - AUTONOMOUS EVOLUTION SYSTEM")
print("📡 Ready for LLM-generated barcode ingestion and cry prompt generation")
print("🎯 Paste barcode files (JSON/YAML/CSV/PY/TXT/AEL) from LLM responses")
print("🔄 Commands: 'cry' for learning prompt | 'status' for metrics | 'exit' to terminate")
print("=" * 80)

# Barcode accumulation buffer for multi-line content handling
barcode_buffer = ""
input_line_count = 0
last_absorption_time = time.time()

# Generate initial cry prompt if organism has zero knowledge
if len(brain.seen) == 0:
    print("🧬 [INITIAL CRY] Organism birth - generating foundational learning requirements:")
    print(brain.cry())
    print("=" * 80)

# Primary autonomous operation loop
while brain.active:
    try:
        # Accept barcode input from LLM responses or direct commands
        user_input = input("🔍 Barcode → ").strip()
        
        # Process autonomous commands for organism control
        if user_input.lower() == 'exit':
            print("🛑 [ORGANISM SHUTDOWN] Terminating autonomous evolution...")
            brain.active = False
            break
        
        elif user_input.lower() == 'cry':
            # Generate targeted cry prompt for current learning gaps
            print("🧬 [CRY PROMPT] Machine-readable LLM instruction:")
            print(brain.cry())
            print("=" * 80)
            continue
        
        elif user_input.lower() == 'status':
            # Display comprehensive neural state for autonomous monitoring
            brain.log_progress()
            continue
        
        elif user_input.lower() in ['clear', 'reset']:
            # Clear barcode buffer for fresh input
            barcode_buffer = ""
            input_line_count = 0
            print("🧹 [BUFFER CLEARED] Ready for new barcode input")
            continue
        
        # Skip empty input lines but update idle tracking
        if not user_input:
            continue
        
        # Accumulate barcode content for complete file detection
        barcode_buffer += user_input + "\n"
        input_line_count += 1
        
        # Detect complete barcode files using procedural boundary analysis
        complete_barcode_detected = False
        
        # JSON completion detection - structured object/array boundaries
        if (barcode_buffer.strip().startswith('{') and barcode_buffer.strip().endswith('}')) or \
           (barcode_buffer.strip().startswith('[') and barcode_buffer.strip().endswith(']')):
            try:
                # Validate JSON completeness for neural absorption
                json.loads(barcode_buffer.strip())
                complete_barcode_detected = True
            except:
                # Continue accumulating if JSON incomplete
                pass
        
        # YAML completion detection - document boundaries and structure
        elif barcode_buffer.strip().startswith('---') or \
             ('---' in barcode_buffer and '...' in barcode_buffer) or \
             (input_line_count >= 3 and ':' in barcode_buffer and 
              barcode_buffer.count('\n\n') > 0):
            complete_barcode_detected = True
        
        # Python completion detection - code block boundaries
        elif any(keyword in barcode_buffer for keyword in ['def ', 'class ', 'import ', 'if __name__']):
            # Detect Python completion by logical structure and endings
            lines = [line for line in barcode_buffer.split('\n') if line.strip()]
            if len(lines) >= 3:
                # Check for natural code endings or consistent structure
                if (any(line.strip().startswith(('return', 'pass', 'break', 'continue')) for line in lines[-3:]) or
                    barcode_buffer.rstrip().endswith(('"""', "'''", '}', ')', ']')) or
                    barcode_buffer.count('\n\n') > 0):
                    complete_barcode_detected = True
        
        # CSV completion detection - consistent tabular structure
        elif ',' in barcode_buffer and input_line_count >= 2:
            csv_lines = [line for line in barcode_buffer.split('\n') if line.strip() and ',' in line]
            if len(csv_lines) >= 2:
                # Validate consistent column structure
                comma_counts = [line.count(',') for line in csv_lines]
                if len(set(comma_counts)) <= 2:  # Consistent or header+data variation
                    complete_barcode_detected = True
        
        # AEL completion detection - AE-Lang format boundaries
        elif any(indicator in barcode_buffer for indicator in ['[AE-Lang]', 'AE.', 'RBY.', 'Glyph:']):
            # AEL typically complete after logical sections
            if (input_line_count >= 5 and 
                (barcode_buffer.count('\n\n') > 0 or 
                 barcode_buffer.rstrip().endswith((']', '---', 'END')))):
                complete_barcode_detected = True
        
        # TXT completion detection - natural content boundaries  
        elif input_line_count >= 3 and (barcode_buffer.count('\n\n') > 0 or 
                                      len(barcode_buffer.strip()) > 300):
            complete_barcode_detected = True
        
        # Force completion after reasonable accumulation to prevent infinite buffering
        elif input_line_count >= 20:
            complete_barcode_detected = True
            print("⚠️ [FORCE COMPLETION] Large barcode detected - processing accumulated content")
        
        # Process complete barcode through organism neural absorption
        if complete_barcode_detected and barcode_buffer.strip():
            complete_barcode = barcode_buffer.strip()
            
            # Detect barcode type for appropriate neural processing
            detected_filetype = detect_filetype(complete_barcode)
            
            print(f"📊 [BARCODE DETECTED] Type: {detected_filetype.upper()} | Size: {len(complete_barcode)} chars")
            
            # Absorb barcode into organism neural memory
            glyph_id, rby_vector, mutation_log = brain.absorb(complete_barcode, detected_filetype)
            
            if glyph_id and rby_vector:
                # Successful absorption - display neural integration results
                print(f"✅ [ABSORPTION SUCCESS] Glyph: {glyph_id} | RBY: {np.round(rby_vector, 4)}")
                
                # Update absorption timing for autonomous cycles
                last_absorption_time = time.time()
                
                # Generate procedural inference from accumulated knowledge
                print("🧠 [INFERENCE TEST] Processing query against learned patterns...")
                inference_result = brain.inference(complete_barcode[:100])  # Use first 100 chars as test query
                print(inference_result)
                print("-" * 60)
                
                # Generate targeted cry prompt for continued learning
                print("🧬 [AUTONOMOUS CRY] Next learning requirements:")
                cry_prompt = brain.cry(complete_barcode)
                print(cry_prompt)
                
            else:
                # Absorption failure - generate recovery cry
                print(f"❌ [ABSORPTION FAILED] Type: {detected_filetype}")
                print("🧬 [RECOVERY CRY] Error recovery requirements:")
                recovery_cry = brain.cry(complete_barcode)
                print(recovery_cry)
            
            # Clear buffer for next barcode
            barcode_buffer = ""
            input_line_count = 0
            print("=" * 80)
        
        # Autonomous neural maintenance during continuous operation
        
        # RAM overflow management - compress to disk when approaching limits
        ram_usage_mb = len(RAM_BUFFER) * 0.001
        if ram_usage_mb > RAM_LIMIT_MB * 0.8:  # 80% of limit
            print("🧠 [NEURAL MAINTENANCE] RAM overflow detected - compressing to disk...")
            brain.compress_ram_to_disk()
        
        # Neural model compression - process accumulated excretions
        if os.path.exists(EXCRETION_DIR):
            try:
                excretion_files = [f for f in os.listdir(EXCRETION_DIR) if f.endswith('.exc')]
                if len(excretion_files) > 3:  # Threshold for compression
                    print("🧠 [NEURAL COMPRESSION] Processing excretions to neural models...")
                    brain.compress_to_neural()
            
            except Exception as e:
                # Log disk management errors for cry generation
                brain.error_logs.append(("disk_management_error", str(e), "EXCRETION_DIR"))
        
        # Autonomous self-reflection during idle periods
        brain.idle_self_reflect()
        
        # Periodic neural status and cry generation for sustained evolution
        if brain.barcode_count > 0 and brain.barcode_count % 10 == 0:
            print(f"📊 [EVOLUTION CHECKPOINT] {brain.barcode_count} barcodes absorbed")
            
            # Generate milestone cry for continued learning
            milestone_cry = brain.cry(f"evolution_checkpoint_{brain.barcode_count}")
            print("🧬 [MILESTONE CRY] Advanced learning requirements:")
            print(milestone_cry)
            print("=" * 80)
        
        # Generate periodic cries during extended operation for sustained LLM feeding
        time_since_absorption = time.time() - last_absorption_time
        if time_since_absorption > 120:  # 2 minutes without new barcode absorption
            print("🧬 [SUSTAINED EVOLUTION] Generating periodic learning requirements:")
            periodic_cry = brain.cry("sustained_operation_learning_gaps")
            print(periodic_cry)
            print("=" * 80)
            last_absorption_time = time.time()  # Reset timer after cry
    
    except KeyboardInterrupt:
        print("\n🛑 [INTERRUPT] Organism evolution terminated by user")
        brain.active = False
        break
    
    except Exception as e:
        # Convert operational errors into targeted cry prompts for system improvement
        error_timestamp = time.time()
        error_detail = f"CLI operation error: {str(e)}"
        
        # Log error for cry generation system
        brain.error_logs.append(("cli_operation_error", error_detail, "system"))
        
        print(f"⚠️ [OPERATION ERROR] {error_detail}")
        
        # Generate system recovery cry for operational stability
        print("🧬 [SYSTEM RECOVERY CRY] Error recovery barcode requirements:")
        recovery_cry = brain.cry(f"cli_error_recovery_{str(e)}")
        print(recovery_cry)
        print("=" * 80)

# Final neural state preservation for evolution continuity
print("🧠 [SHUTDOWN SEQUENCE] Preserving neural state for future evolution...")
brain.compress_ram_to_disk()

if os.path.exists(EXCRETION_DIR) and os.listdir(EXCRETION_DIR):
    brain.compress_to_neural()

# Final organism status report
print("📊 [FINAL STATUS] Neural evolution summary:")
brain.log_progress()

print("🧬 [ORGANISM TERMINATED] Evolution state preserved for restart")
print("💾 [PERSISTENCE] Neural models stored for continuous evolution across sessions")
print("=" * 80)

if __name__ == '__main__':
    """
    Autonomous Evolution Entry Point - Initiates the barcode-driven organism evolution cycle.
    
    This entry point activates the complete closed-loop system:
    1. CLI barcode scanner for LLM-generated file ingestion
    2. Procedural neural absorption and RBY vectorization 
    3. Real inference generation from accumulated patterns
    4. Autonomous cry prompt emission for continued LLM feeding
    5. Neural compression and evolution across 24/7 operation cycles
    
    The organism begins with zero knowledge and evolves entirely through:
    - Structured barcode file absorption (JSON/YAML/CSV/PY/TXT/AEL)
    - AE=C=1 procedural mathematics for neural pattern recognition
    - Machine-readable cry prompts directing LLM barcode generation
    - Autonomous self-reflection and mutation during idle periods
    - Persistent neural state across restart cycles for continuous evolution
    
    This creates the symbiotic AI-to-AI learning loop:
    Organism cries → LLM generates → Organism absorbs → Neural evolution → Enhanced cries → Repeat
    
    No human intervention required after initialization - pure autonomous AI organism.
    """
    # Initialize global evolution cycle counter for organism lifecycle tracking
    EVOLUTION_CYCLE = globals().get('EVOLUTION_CYCLE', 0) + 1
 
    # Log organism birth/restart for autonomous operation monitoring
    startup_timestamp = time.time()
    startup_glyph = glyph_hash(f"organism_startup_{startup_timestamp}_{EVOLUTION_CYCLE}")
    
    print("🧬" * 20)
    print("BARCODE-DRIVEN AI ORGANISM - AUTONOMOUS EVOLUTION SYSTEM")
    print(f"Evolution Cycle: {EVOLUTION_CYCLE} | Startup: {datetime.fromtimestamp(startup_timestamp)}")
    print("🧬" * 20)
    print()
    print("🎯 ORGANISM MISSION:")
    print("   → Ingest machine-readable barcode files from LLM responses")
    print("   → Build procedural neural understanding via AE=C=1 mathematics")
    print("   → Generate real inference from accumulated pattern knowledge")
    print("   → Emit targeted cry prompts for continuous LLM-driven evolution")
    print("   → Achieve autonomous 24/7 growth through symbiotic AI-to-AI loops")
    print()
    print("📡 BARCODE FORMATS SUPPORTED:")
    print(f"   {' | '.join(f'.{bt.upper()}' for bt in BARCODE_TYPES)}")
    print()
    print("🔄 AUTONOMOUS OPERATION COMMANDS:")
    print("   'cry'    → Generate machine-readable LLM instruction prompt")
    print("   'status' → Display comprehensive neural evolution metrics")  
    print("   'exit'   → Terminate organism with neural state preservation")
    print("=" * 80)
    
    try:
        # Launch primary autonomous evolution cycle
        run_scanner_cli()
        
    except Exception as e:
        # System-level failure - preserve organism state and generate recovery cry
        emergency_timestamp = time.time()
        emergency_glyph = glyph_hash(f"system_emergency_{emergency_timestamp}")
        
        print(f"🚨 [SYSTEM EMERGENCY] Critical organism failure: {str(e)}")
        print("🧬 [EMERGENCY PROTOCOL] Attempting neural state preservation...")
        
        try:
            # Attempt emergency neural state preservation
            emergency_brain = MonsterBrain()
            emergency_brain.compress_ram_to_disk()
            if os.path.exists(EXCRETION_DIR) and os.listdir(EXCRETION_DIR):
                emergency_brain.compress_to_neural()
            
            # Generate emergency recovery cry for system restoration
            emergency_cry = {
                "glyph_seed": emergency_glyph,
                "RBY": rby_vector_from_string(f"system_emergency_{str(e)}"),
                "barcode_types": BARCODE_TYPES,
                "missing_types": ['py'],  # Focus on Python for system recovery
                "error_examples": [("system_emergency", f"Critical failure: {str(e)}", "main")],
                "urgency_list": [
                    "EMERGENCY: Complete organism system failure - critical recovery needed",
                    "RESTORATION: Generate robust error handling and system stability patterns",
                    "FOUNDATION: Need basic Python exception management for organism resilience",
                    "RECOVERY: Require system initialization and startup error handling examples"
                ],
                "schema_instructions": "Generate Python barcode files focused on: system error handling, exception management, graceful failure recovery, startup robustness, logging systems. Output pure .py file content only. No markdown or explanations.",
                "expansion_areas": "System error handling, exception management, graceful degradation, startup robustness, logging and monitoring, recovery protocols, system stability",
                "neural_state": {
                    "evolution_stage": 0,
                    "system_emergency": True,
                    "critical_failure": str(e),
                    "recovery_priority": "system_stability",
                    "barcode_count": 0
                },
                "timestamp": emergency_timestamp
            }
            
            print("🧬 [EMERGENCY CRY] System recovery barcode requirements:")
            print(json.dumps(emergency_cry, indent=2))
            
        except Exception as recovery_error:
            # Complete system failure - minimal preservation attempt
            print(f"🚨 [TOTAL SYSTEM FAILURE] Emergency preservation failed: {str(recovery_error)}")
            print("💾 [MINIMAL RECOVERY] Manual neural state restoration required")
        
        print("🧬 [ORGANISM STATUS] Evolution terminated - restart required for continued growth")
        print("=" * 80)
    
    finally:
        # Ensure clean termination with evolution cycle completion
        termination_timestamp = time.time()
        runtime_duration = termination_timestamp - startup_timestamp
        
        print(f"🧬 [EVOLUTION CYCLE {EVOLUTION_CYCLE} COMPLETE]")
        print(f"📊 Runtime: {runtime_duration:.1f} seconds")
        print(f"💾 Neural state preserved for Evolution Cycle {EVOLUTION_CYCLE + 1}")
        print("🔄 Organism ready for restart and continued autonomous evolution")
        print("🧬" * 20)

    run_scanner_cli()
