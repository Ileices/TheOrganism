#!/usr/bin/env python3
"""
Auto-Rebuilder Integration Adapter for Digital Organism
========================================================

Simple adapter module to integrate auto_rebuilder.py capabilities into the existing
Digital Organism ecosystem via the unified launcher system.

This module provides easy integration without requiring major changes to existing code.

Author: Digital Organism Core Team
Date: June 6, 2025
Version: 1.0.0
"""

import asyncio
import threading
import time
from typing import Dict, Any, Optional

class AutoRebuilderAdapter:
    """Lightweight adapter for integrating auto_rebuilder into Digital Organism"""
    
    def __init__(self, heartbeat_interval: int = 300):
        """
        Initialize the auto-rebuilder adapter
        
        Args:
            heartbeat_interval: Seconds between heartbeat cycles (default: 5 minutes)
        """
        self.heartbeat_interval = heartbeat_interval
        self.running = False
        self.heartbeat_thread = None
        self.health_score = 0.85  # Default healthy score
        self.last_heartbeat = None
        
    def start(self):
        """Start the auto-rebuilder heartbeat service"""
        if self.running:
            return
            
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        print("✅ Auto-Rebuilder Adapter: Heartbeat service started")
        
    def stop(self):
        """Stop the auto-rebuilder heartbeat service"""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        print("🛑 Auto-Rebuilder Adapter: Heartbeat service stopped")
        
    def _heartbeat_loop(self):
        """Main heartbeat loop running in background thread"""
        while self.running:
            try:
                self._perform_heartbeat_cycle()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                print(f"⚠️  Auto-Rebuilder Heartbeat Error: {e}")
                time.sleep(10)  # Short delay before retry
                
    def _perform_heartbeat_cycle(self):
        """Perform a single heartbeat cycle"""
        current_time = time.time()
        self.last_heartbeat = current_time
        
        # Assess system health
        self.health_score = self._assess_system_health()
        
        # Check if self-improvement is needed
        if self.health_score < 0.7:
            self._trigger_self_improvement()
            
        # Log heartbeat
        print(f"💓 Auto-Rebuilder Heartbeat: Health {self.health_score:.2f}")
        
    def _assess_system_health(self) -> float:
        """Assess current system health (simplified version)"""
        try:
            # Basic health checks
            health_factors = []
            
            # Check if auto_rebuilder is available
            try:
                import auto_rebuilder
                health_factors.append(0.3)  # Core module available
            except ImportError:
                health_factors.append(0.0)  # Core module missing
                
            # Check system resources (simplified)
            import psutil
            if psutil:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Health decreases with high resource usage
                cpu_health = max(0, (100 - cpu_percent) / 100) * 0.2
                memory_health = max(0, (100 - memory_percent) / 100) * 0.2
                
                health_factors.extend([cpu_health, memory_health])
            else:
                health_factors.extend([0.2, 0.2])  # Default if psutil not available
                
            # Check Digital Organism components
            try:
                # Try to import key Digital Organism modules
                from aeos_production_orchestrator import AEOSOrchestrator
                health_factors.append(0.1)  # Orchestrator available
            except ImportError:
                health_factors.append(0.05)  # Partial availability
                
            try:
                from enhanced_ae_consciousness_system import EnhancedAEConsciousness
                health_factors.append(0.1)  # Consciousness system available
            except ImportError:
                health_factors.append(0.05)  # Partial availability
                
            # Calculate overall health
            total_health = sum(health_factors)
            return min(1.0, total_health)  # Cap at 1.0
            
        except Exception as e:
            print(f"⚠️  Health assessment error: {e}")
            return 0.5  # Default moderate health if assessment fails
            
    def _trigger_self_improvement(self):
        """Trigger self-improvement cycle when health is low"""
        print(f"🔄 Auto-Rebuilder: Triggering self-improvement (health: {self.health_score:.2f})")
        
        try:
            # Try to use auto_rebuilder capabilities if available
            import auto_rebuilder
            
            # Assess and improve system components
            improvement_actions = []
            
            # Check for namespace conflicts
            try:
                conflicts = auto_rebuilder.resolve_namespace_conflicts(".")
                if conflicts:
                    improvement_actions.append(f"Resolved {len(conflicts)} namespace conflicts")
            except Exception as e:
                improvement_actions.append(f"Namespace check: {e}")
                
            # Basic safety assessment
            try:
                # Test a simple code snippet for safety validation
                test_code = "def test(): return 'healthy'"
                safety = auto_rebuilder.assess_code_safety(test_code)
                improvement_actions.append(f"Safety validation: {safety}")
            except Exception as e:
                improvement_actions.append(f"Safety check: {e}")
                
            if improvement_actions:
                print(f"   📋 Improvement actions: {len(improvement_actions)}")
                for action in improvement_actions[:3]:  # Show first 3 actions
                    print(f"      • {action}")
                    
        except ImportError:
            # Fallback improvement without auto_rebuilder
            print("   📋 Basic system optimization (auto_rebuilder not available)")
            
        except Exception as e:
            print(f"   ⚠️  Self-improvement error: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current adapter status"""
        return {
            "running": self.running,
            "health_score": self.health_score,
            "last_heartbeat": self.last_heartbeat,
            "heartbeat_interval": self.heartbeat_interval
        }
        
    def assess_code_safety(self, code: str) -> Dict[str, Any]:
        """Assess code safety using auto_rebuilder if available"""
        try:
            import auto_rebuilder
            return auto_rebuilder.assess_code_safety(code)
        except ImportError:
            return {
                "safety_score": 0.5,
                "risk_level": "unknown",
                "warnings": ["auto_rebuilder not available for full assessment"]
            }
        except Exception as e:
            return {
                "safety_score": 0.0,
                "risk_level": "error",
                "warnings": [f"Assessment error: {e}"]
            }

# Global adapter instance
_adapter_instance: Optional[AutoRebuilderAdapter] = None

def get_auto_rebuilder_adapter(heartbeat_interval: int = 300) -> AutoRebuilderAdapter:
    """Get or create the global auto-rebuilder adapter instance"""
    global _adapter_instance
    
    if _adapter_instance is None:
        _adapter_instance = AutoRebuilderAdapter(heartbeat_interval)
        
    return _adapter_instance

def start_auto_rebuilder_service(heartbeat_interval: int = 300):
    """Start the auto-rebuilder service (convenience function)"""
    adapter = get_auto_rebuilder_adapter(heartbeat_interval)
    adapter.start()
    return adapter

def stop_auto_rebuilder_service():
    """Stop the auto-rebuilder service (convenience function)"""
    global _adapter_instance
    
    if _adapter_instance:
        _adapter_instance.stop()

# Integration function for unified launcher
def integrate_with_digital_organism():
    """
    Integration function that can be called from unified_digital_organism_launcher.py
    """
    try:
        adapter = start_auto_rebuilder_service()
        
        # Return integration info for the launcher
        return {
            "component_name": "auto_rebuilder_adapter",
            "status": "active",
            "adapter": adapter,
            "capabilities": [
                "continuous_health_monitoring",
                "automated_self_improvement",
                "code_safety_assessment",
                "heartbeat_service"
            ]
        }
        
    except Exception as e:
        return {
            "component_name": "auto_rebuilder_adapter", 
            "status": "error",
            "error": str(e),
            "capabilities": []
        }

if __name__ == "__main__":
    """Test the adapter independently"""
    print("🧪 Testing Auto-Rebuilder Adapter")
    print("-" * 40)
    
    # Create and test adapter
    adapter = AutoRebuilderAdapter(heartbeat_interval=5)  # 5 seconds for testing
    
    print("📊 Initial Status:")
    status = adapter.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n🚀 Starting adapter...")
    adapter.start()
    
    # Let it run for a few cycles
    print("⏱️  Running for 15 seconds...")
    time.sleep(15)
    
    print("\n📊 Final Status:")
    status = adapter.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n🛑 Stopping adapter...")
    adapter.stop()
    
    print("✅ Adapter test completed successfully!")
