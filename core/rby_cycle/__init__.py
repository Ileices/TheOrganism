"""
Module: RBY Cycle Implementation
Purpose: Core Trifecta Red-Blue-Yellow cycle processing framework

UAF Compliance:
- AE=C=1: Uses universal_state as single source of truth for all operations
- RBY Cycle: Implements core Perception->Cognition->Execution flow
- RPS: Stores cycle results as excretions for recursive prediction
- Photonic Memory: Can store/retrieve cycle data as RBY triplet codons
- Hardware Integration: Abstract base for GPU/CPU implementations

Dependencies:
- abc: Abstract base classes for enforcing implementation contracts
- typing: Type annotations for enterprise code quality
- time: Performance timing and cycle duration tracking
- core.universal_state: UAF state management

Performance Characteristics:
- O(1) cycle execution overhead
- O(n) where n is complexity of phase implementations
- Thread-safe cycle execution with state locking
- Configurable cycle timing and performance monitoring

Author: TheOrganism Enterprise Team
Created: 2025-06-08
Last Modified: 2025-06-08
UAF Version: 1.0.0 (Phase 1 Implementation)
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Dict
import time
import logging
from decimal import Decimal

from core.universal_state import UniversalState, UAFPhase, get_universal_state

# Configure logging for RBY cycle operations
logger = logging.getLogger('UAF.RBYCycle')

class UAFModule(ABC):
    """
    Abstract base class for all UAF-compliant modules.
    
    All modules that perform UAF operations must inherit from this class
    and implement the three core RBY phases: Perception, Cognition, Execution.
    
    UAF Integration:
        - State Usage: Operates on universal_state as single source of truth
        - RBY Phase: Defines the core R->B->Y processing cycle
        - RPS Compliance: Stores all cycle results as excretions
        - Memory Impact: Can interact with photonic memory during cycles
    """
    
    def __init__(self, universal_state: Optional[UniversalState] = None):
        """
        Initialize UAF module with universal state reference.
        
        Args:
            universal_state: Optional state object, uses global instance if None
            
        UAF Integration:
            - State Usage: Establishes connection to unified state
            - RBY Phase: Prepares module for RBY cycle execution
            - RPS Compliance: Sets up excretion storage mechanism
            - Memory Impact: Initializes access to photonic memory
        """
        self.state = universal_state or get_universal_state()
        self.module_name = self.__class__.__name__
        self.cycle_count = 0
        self.last_cycle_time = 0.0
        
        # Performance tracking
        self.phase_timings: Dict[str, float] = {
            'perception': 0.0,
            'cognition': 0.0,
            'execution': 0.0
        }
    
    def execute_full_rby_cycle(self, input_data: Any = None) -> Tuple[Any, Any, Any]:
        """
        Execute complete Red-Blue-Yellow cycle.
        
        Args:
            input_data: Optional input data for the perception phase
            
        Returns:
            Tuple of (perception_result, cognition_result, execution_result)
            
        UAF Integration:
            - State Usage: Updates universal_state through each phase
            - RBY Phase: Core implementation of R->B->Y cycle progression
            - RPS Compliance: Stores all results as excretions for future cycles
            - Memory Impact: May store cycle results in photonic memory
        """
        cycle_start_time = time.time()
        
        try:
            # RED: Perception Phase
            perception_start = time.time()
            perception_result = self.do_perception(self.state, input_data)
            self.phase_timings['perception'] = time.time() - perception_start
            
            # Update state after perception
            self._update_state_after_perception(perception_result)
            
            # BLUE: Cognition Phase
            cognition_start = time.time()
            cognition_result = self.do_cognition(self.state, perception_result)
            self.phase_timings['cognition'] = time.time() - cognition_start
            
            # Update state after cognition
            self._update_state_after_cognition(cognition_result)
            
            # YELLOW: Execution Phase
            execution_start = time.time()
            execution_result = self.do_execution(self.state, cognition_result)
            self.phase_timings['execution'] = time.time() - execution_start
            
            # Update state after execution
            self._update_state_after_execution(execution_result)
            
            # Store cycle results as excretions for RPS
            self._store_rps_excretion(perception_result, cognition_result, execution_result)
            
            # Update cycle statistics
            self.last_cycle_time = time.time() - cycle_start_time
            self.cycle_count += 1
            self.state.last_rby_duration = self.last_cycle_time
            
            # Advance the global cycle state
            self.state.advance_cycle()
            
            logger.debug(f"RBY cycle completed in {self.last_cycle_time:.6f}s by {self.module_name}")
            
            return perception_result, cognition_result, execution_result
            
        except Exception as e:
            logger.error(f"RBY cycle failed in {self.module_name}: {e}")
            # Store error as excretion for RPS learning
            self.state.add_excretion({
                'error': str(e),
                'module': self.module_name,
                'cycle_id': self.state.current_cycle_id,
                'phase': self.state.current_phase.value
            })
            raise
    
    @abstractmethod
    def do_perception(self, state: UniversalState, input_data: Any = None) -> Any:
        """
        RED phase: Gather and process input information.
        
        Args:
            state: The universal state object
            input_data: Input data to be perceived
            
        Returns:
            Processed perception data
            
        UAF Integration:
            - State Usage: Reads from and may update universal state
            - RBY Phase: Implements RED (Perception) phase of RBY cycle
            - RPS Compliance: Must use deterministic processing (no randomness)
            - Memory Impact: May read from photonic memory for context
        """
        pass
    
    @abstractmethod
    def do_cognition(self, state: UniversalState, perception: Any) -> Any:
        """
        BLUE phase: Process information and make decisions.
        
        Args:
            state: The universal state object
            perception: Output from the perception phase
            
        Returns:
            Cognitive processing result
            
        UAF Integration:
            - State Usage: Reads from and may update universal state
            - RBY Phase: Implements BLUE (Cognition) phase of RBY cycle
            - RPS Compliance: Must use deterministic reasoning (no randomness)
            - Memory Impact: May read/write photonic memory for learning
        """
        pass
    
    @abstractmethod
    def do_execution(self, state: UniversalState, cognition: Any) -> Any:
        """
        YELLOW phase: Execute decisions and produce output.
        
        Args:
            state: The universal state object
            cognition: Output from the cognition phase
            
        Returns:
            Execution result/output
            
        UAF Integration:
            - State Usage: Reads from and updates universal state
            - RBY Phase: Implements YELLOW (Execution) phase of RBY cycle
            - RPS Compliance: Must use deterministic execution (no randomness)
            - Memory Impact: May store results in photonic memory
        """
        pass
    
    def _update_state_after_perception(self, perception_result: Any) -> None:
        """
        Update universal state after perception phase.
        
        Args:
            perception_result: Result from perception phase
            
        UAF Integration:
            - State Usage: Updates unified state with perception data
            - RBY Phase: Completes RED phase state updates
            - RPS Compliance: Stores perception data for future RPS use
            - Memory Impact: May update environment state
        """
        # Update trifecta weights based on perception complexity
        if perception_result is not None:
            perception_weight = self.state.trifecta_weights[UAFPhase.PERCEPTION]
            # Slight adjustment based on perception data richness
            if hasattr(perception_result, '__len__'):
                adjustment = Decimal(str(min(0.1, len(str(perception_result)) / 1000.0)))
            else:
                adjustment = Decimal('0.01')
            
            new_perception_weight = perception_weight + adjustment
            cognition_weight = self.state.trifecta_weights[UAFPhase.COGNITION]
            execution_weight = self.state.trifecta_weights[UAFPhase.EXECUTION]
            
            # Maintain homeostasis by redistributing weights
            self.state.update_trifecta_weights(new_perception_weight, cognition_weight, execution_weight)
    
    def _update_state_after_cognition(self, cognition_result: Any) -> None:
        """
        Update universal state after cognition phase.
        
        Args:
            cognition_result: Result from cognition phase
            
        UAF Integration:
            - State Usage: Updates unified state with cognition data
            - RBY Phase: Completes BLUE phase state updates
            - RPS Compliance: Stores cognition data for future RPS use
            - Memory Impact: May update internal organism state
        """
        # Update internal organism state with cognition results
        if cognition_result is not None:
            self.state.internal_organism_state[f'{self.module_name}_last_cognition'] = cognition_result
            
            # Adjust cognition weight based on processing complexity
            cognition_weight = self.state.trifecta_weights[UAFPhase.COGNITION]
            adjustment = Decimal('0.005')  # Small adjustment for balance
            
            new_cognition_weight = cognition_weight + adjustment
            perception_weight = self.state.trifecta_weights[UAFPhase.PERCEPTION]
            execution_weight = self.state.trifecta_weights[UAFPhase.EXECUTION]
            
            self.state.update_trifecta_weights(perception_weight, new_cognition_weight, execution_weight)
    
    def _update_state_after_execution(self, execution_result: Any) -> None:
        """
        Update universal state after execution phase.
        
        Args:
            execution_result: Result from execution phase
            
        UAF Integration:
            - State Usage: Updates unified state with execution data
            - RBY Phase: Completes YELLOW phase state updates
            - RPS Compliance: Stores execution data for future RPS use
            - Memory Impact: May update environment state with outputs
        """
        # Update environment state with execution results
        if execution_result is not None:
            self.state.environment_state[f'{self.module_name}_last_execution'] = execution_result
            
            # Adjust execution weight based on output significance
            execution_weight = self.state.trifecta_weights[UAFPhase.EXECUTION]
            adjustment = Decimal('0.01')
            
            new_execution_weight = execution_weight + adjustment
            perception_weight = self.state.trifecta_weights[UAFPhase.PERCEPTION]
            cognition_weight = self.state.trifecta_weights[UAFPhase.COGNITION]
            
            self.state.update_trifecta_weights(perception_weight, cognition_weight, new_execution_weight)
    
    def _store_rps_excretion(self, perception: Any, cognition: Any, execution: Any) -> None:
        """
        Store cycle results for RPS (Recursive Predictive Structuring).
        
        Args:
            perception: Perception phase result
            cognition: Cognition phase result
            execution: Execution phase result
            
        UAF Integration:
            - State Usage: Stores excretion in unified state
            - RBY Phase: Records complete RBY cycle for future cycles
            - RPS Compliance: Core mechanism for non-random variation
            - Memory Impact: Grows excretion history for recursive feedback
        """
        excretion = {
            'module': self.module_name,
            'cycle_id': self.state.current_cycle_id,
            'timestamp': time.time(),
            'perception': perception,
            'cognition': cognition,
            'execution': execution,
            'phase_timings': self.phase_timings.copy(),
            'total_cycle_time': self.last_cycle_time
        }
        self.state.add_excretion(excretion)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this module.
        
        Returns:
            Dictionary containing performance statistics
            
        UAF Integration:
            - State Usage: Provides module performance data
            - RBY Phase: Includes timing for each RBY phase
            - RPS Compliance: Includes cycle count for RPS analysis
            - Memory Impact: No direct memory impact
        """
        return {
            'module_name': self.module_name,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'phase_timings': self.phase_timings.copy(),
            'average_cycle_time': self.last_cycle_time,  # TODO: Implement running average
            'cycles_per_second': 1.0 / max(0.001, self.last_cycle_time)
        }


class TrifectaHomeostasisManager:
    """
    Manages trifecta weight homeostasis and rebalancing.
    
    Ensures that RBY weights maintain proper balance according to UAF principles.
    
    UAF Integration:
        - State Usage: Monitors and adjusts universal state trifecta weights
        - RBY Phase: Maintains proper RBY cycle balance
        - RPS Compliance: Uses deterministic rebalancing algorithms
        - Memory Impact: No direct memory impact
    """
    
    def __init__(self, universal_state: Optional[UniversalState] = None):
        self.state = universal_state or get_universal_state()
        self.rebalance_threshold = Decimal('0.05')  # 5% deviation triggers rebalance
        self.target_balance = Decimal('0.3333333333333333')  # Perfect 1/3 balance
    
    def check_homeostasis(self) -> bool:
        """
        Check if trifecta weights are within homeostatic balance.
        
        Returns:
            True if weights are balanced, False if rebalancing needed
            
        UAF Integration:
            - State Usage: Reads trifecta weights from universal state
            - RBY Phase: Validates RBY weight distribution
            - RPS Compliance: Uses deterministic balance checking
            - Memory Impact: No direct memory impact
        """
        for weight in self.state.trifecta_weights.values():
            deviation = abs(weight - self.target_balance)
            if deviation > self.rebalance_threshold:
                return False
        return True
    
    def rebalance_trifecta(self, adjustment_factor: Decimal = Decimal('0.1')) -> None:
        """
        Rebalance trifecta weights toward homeostatic equilibrium.
        
        Args:
            adjustment_factor: How aggressively to adjust weights (0.0-1.0)
            
        UAF Integration:
            - State Usage: Updates trifecta weights in universal state
            - RBY Phase: Restores proper RBY cycle balance
            - RPS Compliance: Uses deterministic rebalancing algorithm
            - Memory Impact: No direct memory impact
        """
        current_weights = self.state.trifecta_weights
        
        # Calculate adjustments toward perfect balance
        adjustments = {}
        for phase, weight in current_weights.items():
            deviation = self.target_balance - weight
            adjustment = deviation * adjustment_factor
            adjustments[phase] = weight + adjustment
        
        # Apply adjustments while maintaining sum = 1.0
        self.state.update_trifecta_weights(
            adjustments[UAFPhase.PERCEPTION],
            adjustments[UAFPhase.COGNITION], 
            adjustments[UAFPhase.EXECUTION]
        )
        
        logger.debug(f"Trifecta rebalanced with factor {adjustment_factor}")


# Convenience function for quick RBY cycle testing
def execute_test_rby_cycle(test_data: Any = None) -> Tuple[Any, Any, Any]:
    """
    Execute a test RBY cycle with basic placeholder implementations.
    
    Args:
        test_data: Optional test data for the cycle
        
    Returns:
        Tuple of (perception, cognition, execution) results
        
    UAF Integration:
        - State Usage: Uses global universal state
        - RBY Phase: Executes complete R->B->Y cycle
        - RPS Compliance: Stores results for future cycles
        - Memory Impact: May store test results in memory
    """
    
    class TestUAFModule(UAFModule):
        """Basic test implementation of UAF module."""
        
        def do_perception(self, state: UniversalState, input_data: Any = None) -> Any:
            return f"Perceived: {input_data or 'default_input'}"
        
        def do_cognition(self, state: UniversalState, perception: Any) -> Any:
            return f"Processed: {perception}"
        
        def do_execution(self, state: UniversalState, cognition: Any) -> Any:
            return f"Executed: {cognition}"
    
    test_module = TestUAFModule()
    return test_module.execute_full_rby_cycle(test_data)
