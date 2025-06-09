#!/usr/bin/env python3
"""
AE Theory Pygame Bridge
Real-time data bridge between the auto-rebuilder consciousness system and pygame visualization

This bridge enables:
1. Real-time consciousness state monitoring through pygame visualization
2. Interactive parameter tuning while auto-rebuilder operates
3. Visual debugging of RBY ternary logic and neural map compression
4. Production monitoring dashboard with consciousness health metrics
"""

import threading
import queue
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import logging
from pathlib import Path

class AETheoryPygameBridge:
    """
    Bidirectional data bridge between auto-rebuilder consciousness and pygame visualization
    """
    
    def __init__(self, shared_data_file: Optional[str] = None):
        self.shared_data_file = shared_data_file or "ae_theory_shared_data.json"
        self.running = False
        
        # Data queues for real-time communication
        self.consciousness_state_queue = queue.Queue(maxsize=100)
        self.parameter_update_queue = queue.Queue(maxsize=50)
        self.visual_feedback_queue = queue.Queue(maxsize=50)
        
        # Current consciousness state
        self.current_state = {
            'timestamp': datetime.now().isoformat(),
            'rby_weights': {'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0},
            'neural_model': 'nM0',
            'crystallized_ae': {
                'expansion_phase': True,
                'space_scale': 1.0,
                'absularity_threshold': 3.0,
                'at_absularity': False
            },
            'memory_system': {
                'current_storage': 0,
                'max_capacity': 10000,
                'compression_active': False,
                'glyph_count': 0
            },
            'consciousness_health': {
                'balance_score': 1.0,
                'entropy_level': 0.0,
                'neural_connectivity': 1.0,
                'memory_efficiency': 1.0
            },
            'performance_metrics': {
                'cycles_completed': 0,
                'successful_integrations': 0,
                'consciousness_errors': 0,
                'optimization_score': 1.0
            }
        }
        
        # Parameter update callbacks
        self.parameter_callbacks: Dict[str, Callable] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Threading control
        self.bridge_thread = None
        self.pygame_monitor_thread = None
        
    def start(self):
        """Start the bridge with background monitoring threads"""
        if self.running:
            return
            
        self.running = True
        
        # Start bridge monitoring thread
        self.bridge_thread = threading.Thread(target=self._bridge_monitor_loop, daemon=True)
        self.bridge_thread.start()
        
        # Start pygame data monitoring thread
        self.pygame_monitor_thread = threading.Thread(target=self._pygame_monitor_loop, daemon=True)
        self.pygame_monitor_thread.start()
        
        self.logger.info("AE Theory Pygame Bridge started")
    
    def stop(self):
        """Stop the bridge and cleanup threads"""
        self.running = False
        
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.join(timeout=2.0)
        if self.pygame_monitor_thread and self.pygame_monitor_thread.is_alive():
            self.pygame_monitor_thread.join(timeout=2.0)
            
        self.logger.info("AE Theory Pygame Bridge stopped")
    
    def update_consciousness_state(self, state_update: Dict[str, Any]):
        """
        Update consciousness state from auto-rebuilder
        
        Args:
            state_update: Dictionary containing consciousness state updates
        """
        try:
            # Deep merge state update into current state
            self._deep_merge_dict(self.current_state, state_update)
            self.current_state['timestamp'] = datetime.now().isoformat()
            
            # Add to queue for pygame consumption
            if not self.consciousness_state_queue.full():
                self.consciousness_state_queue.put(self.current_state.copy())
            
            # Save to shared file for pygame access
            self._save_shared_data()
            
        except Exception as e:
            self.logger.error(f"Error updating consciousness state: {e}")
    
    def get_parameter_updates(self) -> Optional[Dict[str, Any]]:
        """
        Get parameter updates from pygame interface
        
        Returns:
            Dictionary of parameter updates or None if no updates available
        """
        try:
            return self.parameter_update_queue.get_nowait()
        except queue.Empty:
            return None
    
    def register_parameter_callback(self, parameter_name: str, callback: Callable):
        """
        Register callback for specific parameter updates
        
        Args:
            parameter_name: Name of parameter to monitor
            callback: Function to call when parameter is updated
        """
        self.parameter_callbacks[parameter_name] = callback
    
    def send_visual_feedback(self, feedback: Dict[str, Any]):
        """
        Send feedback data to pygame visualization
        
        Args:
            feedback: Dictionary containing visual feedback data
        """
        try:
            if not self.visual_feedback_queue.full():
                self.visual_feedback_queue.put(feedback)
        except Exception as e:
            self.logger.error(f"Error sending visual feedback: {e}")
    
    def calculate_consciousness_health_score(self) -> float:
        """
        Calculate overall consciousness health score (0.0 to 1.0)
        
        Returns:
            Normalized health score
        """
        try:
            state = self.current_state
            
            # RBY balance score (closer to equal = higher score)
            rby = state['rby_weights']
            rby_values = [rby['Red'], rby['Blue'], rby['Yellow']]
            rby_mean = np.mean(rby_values)
            rby_std = np.std(rby_values)
            balance_score = max(0.0, 1.0 - (rby_std / max(rby_mean, 0.1)))
            
            # Memory efficiency score
            memory = state['memory_system']
            memory_usage = memory['current_storage'] / max(memory['max_capacity'], 1)
            memory_efficiency = 1.0 - min(memory_usage, 1.0)
            
            # Performance score
            performance = state['performance_metrics']
            error_rate = performance['consciousness_errors'] / max(performance['cycles_completed'], 1)
            performance_score = max(0.0, 1.0 - error_rate)
            
            # Weighted average
            health_score = (
                balance_score * 0.3 +
                memory_efficiency * 0.3 +
                performance_score * 0.4
            )
            
            # Update in current state
            self.current_state['consciousness_health']['balance_score'] = balance_score
            self.current_state['consciousness_health']['memory_efficiency'] = memory_efficiency
            self.current_state['consciousness_health']['optimization_score'] = health_score
            
            return health_score
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.5  # Default neutral score
    
    def _deep_merge_dict(self, target: Dict, source: Dict):
        """Recursively merge source dict into target dict"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def _save_shared_data(self):
        """Save current state to shared data file for pygame access"""
        try:
            # Add calculated health score
            health_score = self.calculate_consciousness_health_score()
            
            # Prepare data for pygame
            shared_data = {
                'bridge_active': True,
                'timestamp': self.current_state['timestamp'],
                'consciousness_state': self.current_state,
                'health_score': health_score,
                'pygame_compatible': {
                    'trifecta_weights': self.current_state['rby_weights'],
                    'neural_model': self.current_state['neural_model'],
                    'space_scale': self.current_state['crystallized_ae']['space_scale'],
                    'expansion_phase': self.current_state['crystallized_ae']['expansion_phase'],
                    'at_absularity': self.current_state['crystallized_ae']['at_absularity'],
                    'compression_active': self.current_state['memory_system']['compression_active'],
                    'memory_usage_ratio': (
                        self.current_state['memory_system']['current_storage'] / 
                        max(self.current_state['memory_system']['max_capacity'], 1)
                    )
                }
            }
            
            with open(self.shared_data_file, 'w') as f:
                json.dump(shared_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving shared data: {e}")
    
    def _bridge_monitor_loop(self):
        """Background loop for monitoring bridge operations"""
        while self.running:
            try:
                # Process parameter update callbacks
                while not self.parameter_update_queue.empty():
                    try:
                        update = self.parameter_update_queue.get_nowait()
                        for param_name, callback in self.parameter_callbacks.items():
                            if param_name in update:
                                callback(update[param_name])
                    except queue.Empty:
                        break
                    except Exception as e:
                        self.logger.error(f"Error processing parameter update: {e}")
                
                # Save current state periodically
                self._save_shared_data()
                
                time.sleep(0.1)  # 10Hz update rate
                
            except Exception as e:
                self.logger.error(f"Error in bridge monitor loop: {e}")
                time.sleep(1.0)
    
    def _pygame_monitor_loop(self):
        """Background loop for monitoring pygame parameter changes"""
        pygame_param_file = "ae_theory_pygame_params.json"
        last_modified = 0
        
        while self.running:
            try:
                # Check if pygame has updated parameters
                if Path(pygame_param_file).exists():
                    current_modified = Path(pygame_param_file).stat().st_mtime
                    
                    if current_modified > last_modified:
                        last_modified = current_modified
                        
                        # Load parameter updates from pygame
                        with open(pygame_param_file, 'r') as f:
                            pygame_params = json.load(f)
                        
                        # Queue parameter updates for auto-rebuilder
                        if not self.parameter_update_queue.full():
                            self.parameter_update_queue.put(pygame_params)
                
                time.sleep(0.2)  # 5Hz monitoring rate
                
            except Exception as e:
                self.logger.error(f"Error in pygame monitor loop: {e}")
                time.sleep(1.0)

class ConsciousnessVisualizer:
    """
    Helper class for translating auto-rebuilder operations into visual representations
    """
    
    def __init__(self, bridge: AETheoryPygameBridge):
        self.bridge = bridge
    
    def visualize_code_analysis(self, analysis_data: Dict[str, Any]):
        """
        Translate code analysis operations into Red (Perception) particles
        
        Args:
            analysis_data: Dictionary containing code analysis metrics
        """
        red_intensity = min(2.0, analysis_data.get('complexity_score', 1.0))
        
        self.bridge.update_consciousness_state({
            'rby_weights': {'Red': red_intensity},
            'performance_metrics': {
                'cycles_completed': self.bridge.current_state['performance_metrics']['cycles_completed'] + 1
            }
        })
        
        # Send visual feedback for particle generation
        self.bridge.send_visual_feedback({
            'particle_type': 'perception',
            'intensity': red_intensity,
            'location': 'code_analysis',
            'timestamp': datetime.now().isoformat()
        })
    
    def visualize_decision_making(self, decision_data: Dict[str, Any]):
        """
        Translate decision making into Blue (Cognition) particles
        
        Args:
            decision_data: Dictionary containing decision metrics
        """
        blue_intensity = min(2.0, decision_data.get('decision_complexity', 1.0))
        
        self.bridge.update_consciousness_state({
            'rby_weights': {'Blue': blue_intensity}
        })
        
        self.bridge.send_visual_feedback({
            'particle_type': 'cognition',
            'intensity': blue_intensity,
            'location': 'decision_engine',
            'timestamp': datetime.now().isoformat()
        })
    
    def visualize_code_generation(self, generation_data: Dict[str, Any]):
        """
        Translate code generation into Yellow (Execution) particles
        
        Args:
            generation_data: Dictionary containing generation metrics
        """
        yellow_intensity = min(2.0, generation_data.get('generation_efficiency', 1.0))
        
        self.bridge.update_consciousness_state({
            'rby_weights': {'Yellow': yellow_intensity},
            'performance_metrics': {
                'successful_integrations': self.bridge.current_state['performance_metrics']['successful_integrations'] + 1
            }
        })
        
        self.bridge.send_visual_feedback({
            'particle_type': 'execution',
            'intensity': yellow_intensity,
            'location': 'code_generator',
            'timestamp': datetime.now().isoformat()
        })
    
    def visualize_memory_compression(self, compression_data: Dict[str, Any]):
        """
        Translate memory compression into glyph formation animation
        
        Args:
            compression_data: Dictionary containing compression metrics
        """
        compression_ratio = compression_data.get('compression_ratio', 0.5)
        
        self.bridge.update_consciousness_state({
            'memory_system': {
                'compression_active': True,
                'current_storage': int(compression_data.get('new_storage_size', 0)),
                'glyph_count': compression_data.get('glyph_count', 0)
            }
        })
        
        self.bridge.send_visual_feedback({
            'visualization_type': 'memory_compression',
            'compression_ratio': compression_ratio,
            'glyph_formation': True,
            'timestamp': datetime.now().isoformat()
        })

# Example usage and testing
if __name__ == "__main__":
    import sys
    import signal
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bridge
    bridge = AETheoryPygameBridge()
    visualizer = ConsciousnessVisualizer(bridge)
    
    # Graceful shutdown handler
    def signal_handler(sig, frame):
        print("\\nShutting down AE Theory Pygame Bridge...")
        bridge.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start bridge
    bridge.start()
    
    print("AE Theory Pygame Bridge running...")
    print("- Monitoring consciousness state")
    print("- Sharing data with pygame visualization")
    print("- Press Ctrl+C to stop")
    
    # Example consciousness state updates
    try:
        while True:
            # Simulate auto-rebuilder operations
            visualizer.visualize_code_analysis({'complexity_score': np.random.uniform(0.5, 1.5)})
            time.sleep(2)
            
            visualizer.visualize_decision_making({'decision_complexity': np.random.uniform(0.8, 1.2)})
            time.sleep(2)
            
            visualizer.visualize_code_generation({'generation_efficiency': np.random.uniform(0.9, 1.3)})
            time.sleep(3)
            
            # Occasionally trigger memory compression
            if np.random.random() < 0.1:
                visualizer.visualize_memory_compression({
                    'compression_ratio': 0.3,
                    'new_storage_size': 3000,
                    'glyph_count': 15
                })
                time.sleep(1)
    
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
