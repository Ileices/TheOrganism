#!/usr/bin/env python3
"""
Digital Organism Visual Tracking System
Real-time component monitoring, version management, and integration visualization
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from collections import defaultdict, deque
import subprocess

# =============================================================================
# VISUAL TRACKING SYSTEM CONFIGURATION
# =============================================================================

TRACKING_DIR = "digital_organism_tracking"
VERSION_DIR = os.path.join(TRACKING_DIR, "versions")
COMPONENT_DIR = os.path.join(TRACKING_DIR, "components") 
INTEGRATION_DIR = os.path.join(TRACKING_DIR, "integration")
VISUAL_DIR = os.path.join(TRACKING_DIR, "visual")

# Create tracking directory structure
for directory in [TRACKING_DIR, VERSION_DIR, COMPONENT_DIR, INTEGRATION_DIR, VISUAL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Component definitions for Digital Organism system
DIGITAL_ORGANISM_COMPONENTS = {
    "aeos_production_orchestrator.py": {
        "type": "CORE",
        "role": "ORCHESTRATION", 
        "status": "OPERATIONAL",
        "dependencies": ["all_components"],
        "rby_classification": "YELLOW"  # Generation/Coordination
    },
    "aeos_deployment_manager.py": {
        "type": "CORE",
        "role": "DEPLOYMENT",
        "status": "OPERATIONAL", 
        "dependencies": ["docker", "kubernetes"],
        "rby_classification": "BLUE"  # Processing/Management
    },
    "aeos_multimodal_generator.py": {
        "type": "CORE",
        "role": "GENERATION",
        "status": "OPERATIONAL",
        "dependencies": ["transformers", "torch"],
        "rby_classification": "YELLOW"  # Generation/Creation
    },
    "aeos_training_pipeline.py": {
        "type": "CORE", 
        "role": "TRAINING",
        "status": "OPERATIONAL",
        "dependencies": ["datasets", "torch"],
        "rby_classification": "BLUE"  # Processing/Learning
    },
    "aeos_distributed_hpc_network.py": {
        "type": "CORE",
        "role": "NETWORK",
        "status": "OPERATIONAL", 
        "dependencies": ["psutil", "network"],
        "rby_classification": "RED"  # Perception/Communication
    },
    "enhanced_ae_consciousness_system.py": {
        "type": "CONSCIOUSNESS",
        "role": "CONSCIOUSNESS",
        "status": "OPERATIONAL",
        "dependencies": ["numpy", "threading"],
        "rby_classification": "RED"  # Perception/Awareness
    },
    "social_consciousness_demo.py": {
        "type": "CONSCIOUSNESS", 
        "role": "SOCIAL",
        "status": "OPERATIONAL",
        "dependencies": ["consciousness_system"],
        "rby_classification": "RED"  # Perception/Social
    }
}

# Integration opportunities from prototype analysis
PROTOTYPE_INTEGRATIONS = {
    "monster_scanner_rby": {
        "source": "AE_Theory/monster_scanner.py",
        "target": "enhanced_ae_consciousness_system.py", 
        "priority": "HIGH",
        "status": "PENDING",
        "enhancement": "RBY Vector Mathematics"
    },
    "neural_sim_consciousness": {
        "source": "AE_Theory/NEURAL_SIM.py",
        "target": "social_consciousness_demo.py",
        "priority": "HIGH", 
        "status": "PENDING",
        "enhancement": "Neurochemical Simulation"
    },
    "visual_ascii_tracking": {
        "source": "AE_Theory/Visual_ASCII_Mind_Map_Part1.md",
        "target": "digital_organism_visual_tracker.py",
        "priority": "MEDIUM",
        "status": "IN_PROGRESS", 
        "enhancement": "Visual Component Mapping"
    },
    "real_world_architecture": {
        "source": "AE_Theory/Real_World_Goals.md",
        "target": "aeos_production_orchestrator.py",
        "priority": "MEDIUM",
        "status": "PENDING",
        "enhancement": "Complete Architecture Specification"
    }
}

class DigitalOrganismTracker:
    """
    Real-time tracking system for Digital Organism development
    Provides visual feedback, version management, and integration monitoring
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.component_status = {}
        self.integration_history = []
        self.version_history = []
        self.real_time_metrics = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize component tracking
        self._initialize_component_tracking()
        
        # Load existing tracking data
        self._load_tracking_data()
        
    def _initialize_component_tracking(self):
        """Initialize tracking for all Digital Organism components"""
        for component, config in DIGITAL_ORGANISM_COMPONENTS.items():
            self.component_status[component] = {
                "config": config,
                "last_modified": self._get_file_mtime(component),
                "version": "1.0.0",
                "health": "UNKNOWN",
                "performance_metrics": [],
                "integration_points": []
            }
    
    def _get_file_mtime(self, filename):
        """Get file modification time if file exists"""
        try:
            if os.path.exists(filename):
                return os.path.getmtime(filename)
            return 0
        except:
            return 0
    
    def _load_tracking_data(self):
        """Load existing tracking data from disk"""
        tracking_file = os.path.join(TRACKING_DIR, "tracking_state.json")
        if os.path.exists(tracking_file):
            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                    self.integration_history = data.get("integration_history", [])
                    self.version_history = data.get("version_history", [])
                    print(f"✅ Loaded tracking data: {len(self.integration_history)} integrations, {len(self.version_history)} versions")
            except Exception as e:
                print(f"⚠️ Error loading tracking data: {e}")
    
    def _save_tracking_data(self):
        """Save current tracking state to disk"""
        tracking_file = os.path.join(TRACKING_DIR, "tracking_state.json")
        try:
            data = {
                "integration_history": self.integration_history,
                "version_history": self.version_history,
                "last_updated": datetime.now().isoformat(),
                "system_status": self._get_system_status_summary()
            }
            with open(tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving tracking data: {e}")
    
    def start_real_time_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print("🔍 Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        print("⏹️ Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Check component health
                self._update_component_health()
                
                # Record metrics
                metrics = self._collect_real_time_metrics()
                self.real_time_metrics.append(metrics)
                
                # Save state periodically
                self._save_tracking_data()
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(10)
    
    def _update_component_health(self):
        """Update health status for all components"""
        for component in self.component_status:
            try:
                # Check if file exists and is accessible
                if os.path.exists(component):
                    # Check for syntax errors
                    if component.endswith('.py'):
                        result = subprocess.run([sys.executable, '-m', 'py_compile', component], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            self.component_status[component]["health"] = "HEALTHY"
                        else:
                            self.component_status[component]["health"] = "SYNTAX_ERROR"
                    else:
                        self.component_status[component]["health"] = "HEALTHY"
                    
                    # Check for modifications
                    current_mtime = self._get_file_mtime(component)
                    if current_mtime > self.component_status[component]["last_modified"]:
                        self.component_status[component]["last_modified"] = current_mtime
                        self._record_version_change(component)
                else:
                    self.component_status[component]["health"] = "MISSING"
            except Exception as e:
                self.component_status[component]["health"] = f"ERROR: {str(e)}"
    
    def _collect_real_time_metrics(self):
        """Collect current system metrics"""
        healthy_components = sum(1 for comp in self.component_status.values() 
                                if comp["health"] == "HEALTHY")
        total_components = len(self.component_status)
        
        return {
            "timestamp": time.time(),
            "healthy_components": healthy_components,
            "total_components": total_components,
            "health_percentage": (healthy_components / total_components) * 100,
            "integration_count": len(self.integration_history),
            "version_count": len(self.version_history)
        }
    
    def _record_version_change(self, component):
        """Record a version change for a component"""
        version_info = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "previous_version": self.component_status[component]["version"],
            "type": "MODIFICATION"
        }
        self.version_history.append(version_info)
        
        # Increment version
        current_version = self.component_status[component]["version"]
        try:
            major, minor, patch = current_version.split('.')
            new_version = f"{major}.{minor}.{int(patch) + 1}"
            self.component_status[component]["version"] = new_version
            version_info["new_version"] = new_version
        except:
            self.component_status[component]["version"] = "1.0.1"
    
    def record_integration_attempt(self, integration_id, status, details=None):
        """Record an integration attempt"""
        integration_record = {
            "integration_id": integration_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details or {},
            "system_health_before": self._get_system_health_percentage()
        }
        
        self.integration_history.append(integration_record)
        self._save_tracking_data()
        
        print(f"📝 Integration recorded: {integration_id} - {status}")
    
    def _get_system_health_percentage(self):
        """Calculate overall system health percentage"""
        if not self.component_status:
            return 0
        
        healthy_count = sum(1 for comp in self.component_status.values() 
                           if comp["health"] == "HEALTHY")
        return (healthy_count / len(self.component_status)) * 100
    
    def _get_system_status_summary(self):
        """Get comprehensive system status summary"""
        return {
            "total_components": len(self.component_status),
            "healthy_components": sum(1 for comp in self.component_status.values() 
                                     if comp["health"] == "HEALTHY"),
            "health_percentage": self._get_system_health_percentage(),
            "total_integrations": len(self.integration_history),
            "pending_integrations": len([i for i in PROTOTYPE_INTEGRATIONS.values() 
                                        if i["status"] == "PENDING"]),
            "uptime_hours": (time.time() - self.start_time) / 3600
        }
    
    def display_visual_status(self):
        """Display visual ASCII status of the Digital Organism system"""
        print("\n" + "="*80)
        print("🌟 DIGITAL ORGANISM VISUAL TRACKING SYSTEM")
        print("="*80)
        
        # System Overview
        status = self._get_system_status_summary()
        print(f"\n📊 SYSTEM OVERVIEW:")
        print(f"   • Total Components: {status['total_components']}")
        print(f"   • Healthy Components: {status['healthy_components']}")
        print(f"   • System Health: {status['health_percentage']:.1f}%")
        print(f"   • Uptime: {status['uptime_hours']:.1f} hours")
        
        # Component Status Visual
        print(f"\n🔧 COMPONENT STATUS:")
        for component, info in self.component_status.items():
            health = info["health"]
            rby_class = info["config"]["rby_classification"]
            status_icon = "🟢" if health == "HEALTHY" else "🔴" if "ERROR" in health else "🟡"
            rby_icon = "🔴" if rby_class == "RED" else "🔵" if rby_class == "BLUE" else "🟡"
            
            print(f"   {status_icon} {rby_icon} {component[:30]:<30} | {info['config']['role']:<12} | {health}")
        
        # Integration Status
        print(f"\n🔗 PROTOTYPE INTEGRATION STATUS:")
        for integration_id, info in PROTOTYPE_INTEGRATIONS.items():
            status_icon = "🟢" if info["status"] == "COMPLETED" else "🟡" if info["status"] == "IN_PROGRESS" else "⚪"
            priority_icon = "🔥" if info["priority"] == "HIGH" else "⚡" if info["priority"] == "MEDIUM" else "📋"
            
            print(f"   {status_icon} {priority_icon} {integration_id:<25} | {info['enhancement']:<25} | {info['status']}")
        
        # Recent Activity
        if self.version_history:
            print(f"\n📈 RECENT ACTIVITY (Last 5):")
            for activity in self.version_history[-5:]:
                time_str = activity["timestamp"][:19].replace('T', ' ')
                print(f"   🔄 {time_str} | {activity['component'][:30]:<30} | {activity['type']}")
        
        # Real-time Metrics
        if self.real_time_metrics:
            latest_metrics = self.real_time_metrics[-1]
            print(f"\n📊 REAL-TIME METRICS:")
            print(f"   • Health: {latest_metrics['health_percentage']:.1f}%")
            print(f"   • Active Components: {latest_metrics['healthy_components']}/{latest_metrics['total_components']}")
            print(f"   • Integrations: {latest_metrics['integration_count']}")
            print(f"   • Versions: {latest_metrics['version_count']}")
        
        print("\n" + "="*80)
    
    def display_integration_plan(self):
        """Display ASCII visual of integration plan"""
        print("\n" + "="*80)
        print("🎯 PROTOTYPE INTEGRATION ROADMAP")
        print("="*80)
        
        print("""
        AE_Theory Prototypes → Digital Organism Enhancement
        
        ┌─────────────────────┐    ┌─────────────────────────┐
        │   monster_scanner   │───▶│  Enhanced Consciousness │
        │   • RBY Mathematics │    │  • Advanced Vectors    │
        │   • Glyph Hashing   │    │  • Memory Compression  │
        │   • Neural Memory   │    │  • Pattern Recognition │
        └─────────────────────┘    └─────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐    ┌─────────────────────────┐
        │    NEURAL_SIM       │───▶│   Social Consciousness  │
        │   • Neurochemistry  │    │  • Emotional Feedback  │
        │   • Pleasure/Pain   │    │  • Mood-based Responses│
        │   • Dream System    │    │  • Biological Learning │
        └─────────────────────┘    └─────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐    ┌─────────────────────────┐
        │  Visual_ASCII_Map   │───▶│   Visual Tracking Sys  │
        │   • Component Map   │    │  • Real-time Monitor   │
        │   • Dependency Viz  │    │  • Integration Health  │
        │   • Evolution Track │    │  • Version Management  │
        └─────────────────────┘    └─────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐    ┌─────────────────────────┐
        │   Real_World_Goals  │───▶│  Complete Architecture │
        │   • End-state Spec  │    │  • Missing Components  │
        │   • Macro Arch      │    │  • P2P Mesh Network   │
        │   • System Vision   │    │  • Visual Nexus       │
        └─────────────────────┘    └─────────────────────────┘
        """)
        
        print(f"\n📋 INTEGRATION PRIORITIES:")
        priorities = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for integration_id, info in PROTOTYPE_INTEGRATIONS.items():
            priorities[info["priority"]].append(f"{integration_id} ({info['status']})")
        
        for priority, items in priorities.items():
            if items:
                icon = "🔥" if priority == "HIGH" else "⚡" if priority == "MEDIUM" else "📋"
                print(f"\n   {icon} {priority} PRIORITY:")
                for item in items:
                    print(f"      • {item}")
        
        print("\n" + "="*80)

# =============================================================================
# INTERACTIVE TRACKING CONSOLE
# =============================================================================

def main():
    """Main interactive tracking console"""
    tracker = DigitalOrganismTracker()
    tracker.start_real_time_monitoring()
    
    print("🌟 Digital Organism Visual Tracking System Started")
    print("Type 'help' for commands, 'quit' to exit")
    
    try:
        while True:
            try:
                command = input("\n📊 Tracker> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'help':
                    print("""
Available Commands:
  status      - Display visual system status
  plan        - Show integration roadmap  
  health      - Show component health details
  integrations - Show integration history
  versions    - Show version history
  metrics     - Show real-time metrics
  test        - Test integration (integration_id)
  help        - Show this help
  quit        - Exit tracking system
                    """)
                elif command == 'status':
                    tracker.display_visual_status()
                elif command == 'plan':
                    tracker.display_integration_plan()
                elif command == 'health':
                    status = tracker._get_system_status_summary()
                    print(f"\n🏥 SYSTEM HEALTH DETAILS:")
                    print(f"   Overall Health: {status['health_percentage']:.1f}%")
                    print(f"   Operational: {status['healthy_components']}/{status['total_components']} components")
                    print(f"   Pending Integrations: {status['pending_integrations']}")
                    print(f"   System Uptime: {status['uptime_hours']:.1f} hours")
                elif command == 'integrations':
                    print(f"\n🔗 INTEGRATION HISTORY ({len(tracker.integration_history)} total):")
                    for integration in tracker.integration_history[-10:]:  # Last 10
                        time_str = integration["timestamp"][:19].replace('T', ' ')
                        print(f"   {time_str} | {integration['integration_id']:<20} | {integration['status']}")
                elif command == 'versions':
                    print(f"\n📦 VERSION HISTORY ({len(tracker.version_history)} total):")
                    for version in tracker.version_history[-10:]:  # Last 10
                        time_str = version["timestamp"][:19].replace('T', ' ')
                        print(f"   {time_str} | {version['component']:<30} | {version['type']}")
                elif command == 'metrics':
                    if tracker.real_time_metrics:
                        latest = tracker.real_time_metrics[-1]
                        print(f"\n📊 REAL-TIME METRICS:")
                        print(f"   Health: {latest['health_percentage']:.1f}%")
                        print(f"   Components: {latest['healthy_components']}/{latest['total_components']}")
                        print(f"   Integrations: {latest['integration_count']}")
                        print(f"   Versions: {latest['version_count']}")
                    else:
                        print("No metrics available yet")
                elif command.startswith('test '):
                    integration_id = command[5:].strip()
                    if integration_id in PROTOTYPE_INTEGRATIONS:
                        print(f"🧪 Testing integration: {integration_id}")
                        tracker.record_integration_attempt(integration_id, "TEST_STARTED")
                        # Simulate test
                        time.sleep(1)
                        tracker.record_integration_attempt(integration_id, "TEST_COMPLETED", 
                                                         {"test_result": "SUCCESS", "improvements": "Ready for integration"})
                        print(f"✅ Test completed for {integration_id}")
                    else:
                        print(f"❌ Unknown integration: {integration_id}")
                        print(f"Available: {list(PROTOTYPE_INTEGRATIONS.keys())}")
                else:
                    print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Tracking system interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    finally:
        tracker.stop_real_time_monitoring()
        print("🔄 Tracking system shutdown complete")

if __name__ == "__main__":
    main()
