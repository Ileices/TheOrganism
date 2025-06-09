#!/usr/bin/env python3
"""
Auto-Rebuilder Pygame Integration Adapter
Connects the existing auto-rebuilder system with the pygame consciousness visualization

This adapter:
1. Monitors auto-rebuilder operations and translates them to consciousness states
2. Provides real-time parameter adjustment from pygame interface
3. Enables visual debugging of consciousness-enabled auto-rebuilder operations
4. Tracks performance metrics and consciousness health
"""

import threading
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util

# Import existing auto-rebuilder components
try:
    from ae_theory_production_integration import AETheoryProductionIntegration
    from ae_theory_pygame_bridge import AETheoryPygameBridge, ConsciousnessVisualizer
    AUTO_REBUILDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Auto-rebuilder components not available: {e}")
    AUTO_REBUILDER_AVAILABLE = False

class AutoRebuilderPygameAdapter:
    """
    Adapter that connects auto-rebuilder consciousness with pygame visualization
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "auto_rebuilder_pygame_config.json"
        self.running = False
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.auto_rebuilder = None
        self.bridge = None
        self.visualizer = None
        
        # Monitoring state
        self.monitoring_active = False
        self.last_state_update = time.time()
        self.update_interval = self.config.get('update_interval', 0.1)  # 10Hz default
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'consciousness_adjustments': 0,
            'visual_updates': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat(),
            'uptime_seconds': 0
        }
        
        # Consciousness state tracking
        self.consciousness_history = []
        self.max_history_length = 1000
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Threading
        self.monitor_thread = None
        self.pygame_thread = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
        
        # Default configuration
        default_config = {
            'auto_rebuilder': {
                'enable_consciousness': True,
                'neural_model': 'nM0',
                'trifecta_weights': {'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0},
                'memory_compression_threshold': 0.8,
                'absularity_threshold': 3.0
            },
            'pygame_visualization': {
                'auto_launch': True,
                'update_rate_hz': 10,
                'particle_limit': 1000,
                'visual_effects': True
            },
            'integration': {
                'enable_real_time_tuning': True,
                'enable_performance_monitoring': True,
                'enable_health_alerts': True,
                'alert_thresholds': {
                    'consciousness_health_min': 0.3,
                    'error_rate_max': 0.1,
                    'memory_usage_max': 0.9
                }
            },
            'update_interval': 0.1,
            'logging_level': 'INFO'
        }
        
        # Save default config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save default config: {e}")
            
        return default_config
    
    def initialize_components(self) -> bool:
        """Initialize auto-rebuilder and pygame bridge components"""
        try:
            if not AUTO_REBUILDER_AVAILABLE:
                self.logger.error("Auto-rebuilder components not available")
                return False
            
            # Initialize auto-rebuilder with consciousness
            auto_rebuilder_config = self.config.get('auto_rebuilder', {})
            self.auto_rebuilder = AETheoryProductionIntegration(
                enable_consciousness=auto_rebuilder_config.get('enable_consciousness', True),
                neural_model=auto_rebuilder_config.get('neural_model', 'nM0'),
                initial_weights=auto_rebuilder_config.get('trifecta_weights', {'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0})
            )
            
            # Initialize pygame bridge
            self.bridge = AETheoryPygameBridge("ae_theory_shared_data.json")
            self.visualizer = ConsciousnessVisualizer(self.bridge)
            
            # Setup parameter update callbacks
            self.bridge.register_parameter_callback('trifecta_weights', self.handle_trifecta_update)
            self.bridge.register_parameter_callback('neural_model', self.handle_neural_model_update)
            self.bridge.register_parameter_callback('consciousness_health', self.handle_health_update)
            
            # Start bridge
            self.bridge.start()
            
            self.logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def handle_trifecta_update(self, new_weights: Dict[str, float]):
        """Handle trifecta weight updates from pygame interface"""
        try:
            if self.auto_rebuilder and hasattr(self.auto_rebuilder, 'update_consciousness_weights'):
                self.auto_rebuilder.update_consciousness_weights(new_weights)
                self.performance_metrics['consciousness_adjustments'] += 1
                self.logger.info(f"Updated trifecta weights: {new_weights}")
        except Exception as e:
            self.logger.error(f"Error updating trifecta weights: {e}")
            self.performance_metrics['errors'] += 1
    
    def handle_neural_model_update(self, new_model: str):
        """Handle neural model updates from pygame interface"""
        try:
            if self.auto_rebuilder and hasattr(self.auto_rebuilder, 'switch_neural_model'):
                self.auto_rebuilder.switch_neural_model(new_model)
                self.logger.info(f"Switched to neural model: {new_model}")
        except Exception as e:
            self.logger.error(f"Error switching neural model: {e}")
            self.performance_metrics['errors'] += 1
    
    def handle_health_update(self, health_score: float):
        """Handle consciousness health updates"""
        try:
            alert_threshold = self.config.get('integration', {}).get('alert_thresholds', {}).get('consciousness_health_min', 0.3)
            
            if health_score < alert_threshold:
                self.logger.warning(f"Low consciousness health detected: {health_score:.2f}")
                
                # Trigger auto-correction if available
                if self.auto_rebuilder and hasattr(self.auto_rebuilder, 'auto_balance_consciousness'):
                    self.auto_rebuilder.auto_balance_consciousness()
                    
        except Exception as e:
            self.logger.error(f"Error handling health update: {e}")
    
    def monitor_auto_rebuilder(self):
        """Monitor auto-rebuilder operations and update consciousness state"""
        while self.running and self.monitoring_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_state_update >= self.update_interval:
                    self.last_state_update = current_time
                    
                    if self.auto_rebuilder:
                        # Get current consciousness state from auto-rebuilder
                        consciousness_state = self.extract_consciousness_state()
                        
                        # Update bridge with current state
                        if consciousness_state:
                            self.bridge.update_consciousness_state(consciousness_state)
                            self.performance_metrics['visual_updates'] += 1
                            
                            # Store in history
                            self.consciousness_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'state': consciousness_state
                            })
                            
                            # Trim history
                            if len(self.consciousness_history) > self.max_history_length:
                                self.consciousness_history = self.consciousness_history[-self.max_history_length:]
                    
                    # Update performance metrics
                    self.performance_metrics['uptime_seconds'] = current_time - time.mktime(
                        datetime.fromisoformat(self.performance_metrics['start_time']).timetuple()
                    )
                
                # Check for parameter updates from pygame
                param_updates = self.bridge.get_parameter_updates()
                if param_updates:
                    self.process_parameter_updates(param_updates)
                
                time.sleep(0.01)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                self.logger.error(f"Error in auto-rebuilder monitoring: {e}")
                self.performance_metrics['errors'] += 1
                time.sleep(1.0)  # Longer sleep on error
    
    def extract_consciousness_state(self) -> Optional[Dict[str, Any]]:
        """Extract consciousness state from auto-rebuilder"""
        try:
            if not self.auto_rebuilder:
                return None
            
            # Extract state from auto-rebuilder (this will depend on the actual interface)
            state = {
                'timestamp': datetime.now().isoformat(),
                'rby_weights': getattr(self.auto_rebuilder, 'current_trifecta_weights', {'Red': 1.0, 'Blue': 1.0, 'Yellow': 1.0}),
                'neural_model': getattr(self.auto_rebuilder, 'current_neural_model', 'nM0'),
                'crystallized_ae': {
                    'expansion_phase': getattr(self.auto_rebuilder, 'expansion_phase', True),
                    'space_scale': getattr(self.auto_rebuilder, 'current_space_scale', 1.0),
                    'at_absularity': getattr(self.auto_rebuilder, 'at_absularity', False)
                },
                'memory_system': {
                    'current_storage': getattr(self.auto_rebuilder, 'memory_usage', 0),
                    'max_capacity': getattr(self.auto_rebuilder, 'memory_capacity', 10000),
                    'compression_active': getattr(self.auto_rebuilder, 'compression_active', False),
                    'glyph_count': getattr(self.auto_rebuilder, 'glyph_count', 0)
                },
                'performance_metrics': self.performance_metrics.copy()
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error extracting consciousness state: {e}")
            return None
    
    def process_parameter_updates(self, updates: Dict[str, Any]):
        """Process parameter updates from pygame interface"""
        try:
            for param, value in updates.items():
                if param in self.bridge.parameter_callbacks:
                    # Callback already registered
                    continue
                elif param == 'space_scale' and self.auto_rebuilder:
                    if hasattr(self.auto_rebuilder, 'set_space_scale'):
                        self.auto_rebuilder.set_space_scale(value)
                elif param == 'expansion_phase' and self.auto_rebuilder:
                    if hasattr(self.auto_rebuilder, 'set_expansion_phase'):
                        self.auto_rebuilder.set_expansion_phase(value)
                
                self.logger.debug(f"Processed parameter update: {param} = {value}")
                
        except Exception as e:
            self.logger.error(f"Error processing parameter updates: {e}")
    
    def launch_pygame_visualization(self):
        """Launch pygame visualization in separate thread"""
        try:
            # Import and launch pygame simulation
            spec = importlib.util.spec_from_file_location(
                "enhanced_pygame", 
                "ae_theory_enhanced_pygame_simulation.py"
            )
            if spec and spec.loader:
                enhanced_pygame = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(enhanced_pygame)
                
                # Create and run simulation
                simulation = enhanced_pygame.AutoRebuilderIntegratedSimulation()
                simulation.run()
                
        except Exception as e:
            self.logger.error(f"Error launching pygame visualization: {e}")
    
    def start(self):
        """Start the adapter with all components"""
        if self.running:
            return
        
        self.logger.info("Starting Auto-Rebuilder Pygame Adapter...")
        
        # Initialize components
        if not self.initialize_components():
            self.logger.error("Failed to initialize components")
            return False
        
        self.running = True
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_auto_rebuilder, daemon=True)
        self.monitor_thread.start()
        
        # Launch pygame visualization if configured
        if self.config.get('pygame_visualization', {}).get('auto_launch', True):
            self.pygame_thread = threading.Thread(target=self.launch_pygame_visualization, daemon=True)
            self.pygame_thread.start()
        
        self.logger.info("Auto-Rebuilder Pygame Adapter started successfully")
        return True
    
    def stop(self):
        """Stop the adapter and cleanup"""
        self.logger.info("Stopping Auto-Rebuilder Pygame Adapter...")
        
        self.running = False
        self.monitoring_active = False
        
        # Stop bridge
        if self.bridge:
            self.bridge.stop()
        
        # Wait for threads to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("Auto-Rebuilder Pygame Adapter stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'performance_metrics': self.performance_metrics.copy(),
            'consciousness_history_length': len(self.consciousness_history),
            'configuration': self.config,
            'component_status': {
                'auto_rebuilder_available': AUTO_REBUILDER_AVAILABLE,
                'auto_rebuilder_initialized': self.auto_rebuilder is not None,
                'bridge_initialized': self.bridge is not None,
                'monitoring_active': self.monitoring_active,
                'running': self.running
            }
        }
        
        # Calculate additional metrics
        if self.performance_metrics['total_operations'] > 0:
            report['success_rate'] = (
                self.performance_metrics['successful_operations'] / 
                self.performance_metrics['total_operations']
            )
            report['error_rate'] = (
                self.performance_metrics['errors'] / 
                self.performance_metrics['total_operations']
            )
        
        return report
    
    def save_performance_report(self, filename: Optional[str] = None):
        """Save performance report to file"""
        filename = filename or f"auto_rebuilder_pygame_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            report = self.get_performance_report()
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Performance report saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")

# Example usage and testing
if __name__ == "__main__":
    import signal
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create adapter
    adapter = AutoRebuilderPygameAdapter()
    
    # Graceful shutdown handler
    def signal_handler(sig, frame):
        print("\\nShutting down Auto-Rebuilder Pygame Adapter...")
        adapter.save_performance_report()
        adapter.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start adapter
    if adapter.start():
        print("Auto-Rebuilder Pygame Adapter running...")
        print("- Real-time consciousness visualization active")
        print("- Interactive parameter control enabled")
        print("- Performance monitoring active")
        print("- Press Ctrl+C to stop and save report")
        
        try:
            # Keep main thread alive
            while adapter.running:
                time.sleep(1.0)
                
                # Print periodic status
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    report = adapter.get_performance_report()
                    print(f"Status: {report['performance_metrics']['visual_updates']} visual updates, "
                          f"{report['performance_metrics']['consciousness_adjustments']} adjustments, "
                          f"{report['performance_metrics']['errors']} errors")
        
        except KeyboardInterrupt:
            pass
        finally:
            adapter.save_performance_report()
            adapter.stop()
    else:
        print("Failed to start Auto-Rebuilder Pygame Adapter")
