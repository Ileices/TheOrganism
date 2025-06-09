"""
Temporary clean Universal State implementation for testing
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from enum import Enum
import threading
import time

# Set high precision for UAF calculations
getcontext().prec = 50


@dataclass
class TrifectaWeights:
    """Trifecta weight structure for Red-Blue-Yellow processing."""
    red: float = 0.333333
    blue: float = 0.333333
    yellow: float = 0.333334
    
    def normalize(self) -> 'TrifectaWeights':
        """Normalize weights to sum to 1.0."""
        total = self.red + self.blue + self.yellow
        if total > 0:
            return TrifectaWeights(
                red=self.red / total,
                blue=self.blue / total, 
                yellow=self.yellow / total
            )
        else:
            return TrifectaWeights()
    
    def get_imbalance(self) -> float:
        """Get the imbalance measure (range between max and min weights)."""
        weights = [self.red, self.blue, self.yellow]
        return max(weights) - min(weights)


class UAFPhase(Enum):
    """UAF processing phases in the Trifecta RBY cycle."""
    PERCEPTION = "RED"
    COGNITION = "BLUE" 
    EXECUTION = "YELLOW"


@dataclass
class UniversalState:
    """The single source of truth for the entire system (AE=C=1)."""
    
    # Core UAF Components
    trifecta_weights: Dict[UAFPhase, Decimal] = field(
        default_factory=lambda: {
            UAFPhase.PERCEPTION: Decimal('0.3333333333333333'),
            UAFPhase.COGNITION: Decimal('0.3333333333333333'),
            UAFPhase.EXECUTION: Decimal('0.3333333333333334')
        }
    )
    
    # Photonic Memory - Neural DNA triplet codons (RBY format)
    dna_memory: List[Tuple[Decimal, Decimal, Decimal]] = field(default_factory=list)
    
    # Excretions for RPS (Recursive Predictive Structuring)
    excretions: List[Any] = field(default_factory=list)
    
    # System state
    time: float = field(default_factory=time.time)
    internal_organism_state: Dict[str, Any] = field(default_factory=dict)
    environment_state: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    gpu_available: bool = False
    current_cycle_id: int = 0
    last_rby_duration: Optional[float] = None
    current_phase: UAFPhase = UAFPhase.PERCEPTION
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    
    def validate_uaf_compliance(self) -> bool:
        """Validate that state adheres to UAF principles."""
        with self._lock:
            try:
                # Validate trifecta weights sum to 1.0
                total_weight = sum(self.trifecta_weights.values())
                if not (Decimal('0.99') <= total_weight <= Decimal('1.01')):
                    return False
                
                # Validate DNA memory structure
                for codon in self.dna_memory:
                    if not isinstance(codon, tuple) or len(codon) != 3:
                        return False
                    if not all(isinstance(val, Decimal) for val in codon):
                        return False
                
                # Validate current phase
                if not isinstance(self.current_phase, UAFPhase):
                    return False
                
                # Validate cycle tracking
                if self.current_cycle_id < 0:
                    return False
                
                return True
                
            except Exception:
                return False
    
    def get_trifecta_weights(self) -> TrifectaWeights:
        """Get current trifecta weights as TrifectaWeights object."""
        with self._lock:
            return TrifectaWeights(
                red=float(self.trifecta_weights[UAFPhase.PERCEPTION]),
                blue=float(self.trifecta_weights[UAFPhase.COGNITION]),
                yellow=float(self.trifecta_weights[UAFPhase.EXECUTION])
            )
    
    def update_trifecta_weights(self, weights: TrifectaWeights) -> None:
        """Update trifecta weights from TrifectaWeights object."""
        with self._lock:
            normalized = weights.normalize()
            self.trifecta_weights[UAFPhase.PERCEPTION] = Decimal(str(normalized.red))
            self.trifecta_weights[UAFPhase.COGNITION] = Decimal(str(normalized.blue))
            self.trifecta_weights[UAFPhase.EXECUTION] = Decimal(str(normalized.yellow))
    
    def increment_cycle(self) -> None:
        """Increment the current cycle ID."""
        with self._lock:
            self.current_cycle_id += 1
    
    def add_excretion(self, excretion_data: Any) -> None:
        """Add excretion data for RPS (Recursive Predictive Structuring)."""
        with self._lock:
            excretion_entry = {
                'data': excretion_data,
                'cycle_id': self.current_cycle_id,
                'timestamp': time.time(),
                'phase': self.current_phase.value
            }
            self.excretions.append(excretion_entry)
            
            # Prevent unbounded growth
            if len(self.excretions) > 10000:
                self.excretions = self.excretions[-10000:]
    
    def advance_cycle(self) -> UAFPhase:
        """Advance to next phase in the RBY cycle."""
        with self._lock:
            # Advance through R -> B -> Y -> R cycle
            phase_order = [UAFPhase.PERCEPTION, UAFPhase.COGNITION, UAFPhase.EXECUTION]
            current_index = phase_order.index(self.current_phase)
            next_index = (current_index + 1) % len(phase_order)
            
            self.current_phase = phase_order[next_index]
            
            # Increment cycle ID when we complete a full R->B->Y cycle
            if self.current_phase == UAFPhase.PERCEPTION:
                self.current_cycle_id += 1
            
            return self.current_phase
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a thread-safe snapshot of current state."""
        with self._lock:
            return {
                'current_phase': self.current_phase.value,
                'current_cycle_id': self.current_cycle_id,
                'trifecta_weights': {phase.value: float(weight) for phase, weight in self.trifecta_weights.items()},
                'dna_memory_count': len(self.dna_memory),
                'excretion_count': len(self.excretions),
                'gpu_available': self.gpu_available,
                'last_rby_duration': self.last_rby_duration,
                'system_time': self.time,
                'uaf_compliant': self.validate_uaf_compliance()
            }


# Global universal state instance (Singleton pattern for AE=C=1)
_universal_state_instance: Optional[UniversalState] = None
_state_lock = threading.Lock()


def get_universal_state() -> UniversalState:
    """Get the global universal state instance (Singleton pattern)."""
    global _universal_state_instance
    
    with _state_lock:
        if _universal_state_instance is None:
            _universal_state_instance = UniversalState()
        return _universal_state_instance


def initialize_universal_state(gpu_available: bool = False, force_reset: bool = False) -> UniversalState:
    """Initialize the global universal state with hardware configuration."""
    global _universal_state_instance
    
    with _state_lock:
        if _universal_state_instance is None or force_reset:
            _universal_state_instance = UniversalState()
            _universal_state_instance.gpu_available = gpu_available
            _universal_state_instance.time = time.time()
            
            # Verify UAF compliance
            if not _universal_state_instance.validate_uaf_compliance():
                raise RuntimeError("Failed to initialize UAF-compliant universal state")
        else:
            # Just update hardware info if instance already exists
            _universal_state_instance.gpu_available = gpu_available
        
        return _universal_state_instance


# Export main classes and functions
__all__ = ['UniversalState', 'UAFPhase', 'TrifectaWeights', 'get_universal_state', 'initialize_universal_state']
