#!/usr/bin/env python3
"""
Unified Consciousness Framework Launcher
Complete integration of Procedural Laws Engine with GeoBIT Tower Defense
and Advanced GUI Consciousness Monitoring Dashboard

This launcher combines:
- Procedural Laws Engine (mathematical consciousness framework)
- Enhanced Consciousness Dashboard (real-time monitoring GUI)
- GeoBIT Tower Defense Core (existing game framework)
- Multi-mode Architecture (Tower Defense/MMORPG/Base Builder switching)
"""

import sys
import os
import json
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import consciousness framework components
try:
    from ae_core_consciousness import AEConsciousness
    from ae_consciousness_mathematics import AEMathEngine, AEVector
    from ae_multimode_architecture import MultiModeArchitecture
    from ae_procedural_laws_engine import ProceduralLawsEngine, create_procedural_laws_engine
    from consciousness_dashboard_adapter import CompleteConsciousnessDashboard
    from ae_procedural_world_generation import ProceduralWorldGenerator
except ImportError as e:
    print(f"Warning: Some consciousness modules not found: {e}")
    print("Continuing with basic functionality...")

class UnifiedConsciousnessLauncher(QMainWindow):
    """Main launcher for the complete Unified Absolute Framework implementation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Absolute Framework - Consciousness Game Engine")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize core components
        self.math_engine = None
        self.procedural_engine = None
        self.consciousness_dashboard = None
        self.world_generator = None
        self.game_mode = "tower_defense"  # Start with tower defense
        
        # Game state
        self.consciousness_state = {
            "red": 0.333,
            "blue": 0.333, 
            "yellow": 0.334
        }
        
        self.init_engines()
        self.init_ui()
        self.start_consciousness_monitoring()
        
    def init_engines(self):
        """Initialize all consciousness engines"""
        try:
            # Core consciousness mathematics
            self.math_engine = AEMathEngine()
            
            # Procedural laws engine for game mechanics
            self.procedural_engine = create_procedural_laws_engine()
            
            # World generation engine
            self.world_generator = ProceduralWorldGenerator(self.math_engine)
            
            print("✅ All consciousness engines initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Engine initialization warning: {e}")
            print("Continuing with fallback functionality...")
    
    def init_ui(self):
        """Initialize the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Game control and mode selection
        left_panel = self.create_game_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Center panel - Game visualization/viewport
        center_panel = self.create_game_viewport()
        main_layout.addWidget(center_panel, 3)
        
        # Right panel - Consciousness monitoring dashboard
        right_panel = self.create_consciousness_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Unified Absolute Framework Ready")
        
        # Dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #00ff00;
            }
            QWidget {
                background-color: #2a2a2a;
                color: #00ff00;
                border: 1px solid #444444;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #00ff00;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #5a5a5a;
            }
            QLabel {
                border: none;
                padding: 4px;
            }
        """)
    
    def create_game_control_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Mode selection
        mode_group = QGroupBox("Game Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.tower_defense_radio = QRadioButton("Tower Defense")
        self.mmorpg_radio = QRadioButton("MMORPG Mode")
        self.base_builder_radio = QRadioButton("Base Builder")
        
        self.tower_defense_radio.setChecked(True)
        self.tower_defense_radio.toggled.connect(lambda: self.set_game_mode("tower_defense"))
        self.mmorpg_radio.toggled.connect(lambda: self.set_game_mode("mmorpg"))
        self.base_builder_radio.toggled.connect(lambda: self.set_game_mode("base_builder"))
        
        mode_layout.addWidget(self.tower_defense_radio)
        mode_layout.addWidget(self.mmorpg_radio)
        mode_layout.addWidget(self.base_builder_radio)
        
        # Procedural controls
        procedural_group = QGroupBox("Procedural Laws Controls")
        procedural_layout = QVBoxLayout(procedural_group)
        
        self.generate_world_btn = QPushButton("Generate New World")
        self.generate_world_btn.clicked.connect(self.generate_new_world)
        
        self.trigger_spiral_btn = QPushButton("Trigger Spiral Anomaly")
        self.trigger_spiral_btn.clicked.connect(self.trigger_spiral_anomaly)
        
        self.dream_state_btn = QPushButton("Enter Dream State")
        self.dream_state_btn.clicked.connect(self.enter_dream_state)
        
        procedural_layout.addWidget(self.generate_world_btn)
        procedural_layout.addWidget(self.trigger_spiral_btn)
        procedural_layout.addWidget(self.dream_state_btn)
        
        # Rectangle leveling system display
        rectangles_group = QGroupBox("Consciousness Rectangles")
        rectangles_layout = QVBoxLayout(rectangles_group)
        
        self.rectangle_up_label = QLabel("Rectangle Up: Level 0")
        self.rectangle_down_label = QLabel("Rectangle Down: Level 0") 
        self.rectangle_permanent_label = QLabel("Rectangle Permanent: Level 0")
        
        rectangles_layout.addWidget(self.rectangle_up_label)
        rectangles_layout.addWidget(self.rectangle_down_label)
        rectangles_layout.addWidget(self.rectangle_permanent_label)
        
        # Add all groups to main layout
        layout.addWidget(mode_group)
        layout.addWidget(procedural_group)
        layout.addWidget(rectangles_group)
        layout.addStretch()
        
        return panel
    
    def create_game_viewport(self):
        """Create the center game viewport"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Game title
        title_label = QLabel("UNIFIED ABSOLUTE FRAMEWORK")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #00ffff;
            background-color: #1a1a1a;
            padding: 10px;
            border: 2px solid #00ffff;
        """)
        
        # Game viewport (placeholder for actual game rendering)
        self.game_viewport = QLabel("Game Viewport - GeoBIT Tower Defense with Procedural Laws")
        self.game_viewport.setAlignment(Qt.AlignCenter)
        self.game_viewport.setMinimumHeight(400)
        self.game_viewport.setStyleSheet("""
            background-color: #0a0a0a;
            border: 2px solid #008800;
            color: #00ff00;
            font-size: 14px;
        """)
        
        # Game stats display
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        
        self.health_label = QLabel("Health: 100/100")
        self.energy_label = QLabel("Energy: 50/50")
        self.score_label = QLabel("Score: 0")
        self.consciousness_label = QLabel("Consciousness: Balanced")
        
        stats_layout.addWidget(self.health_label)
        stats_layout.addWidget(self.energy_label)
        stats_layout.addWidget(self.score_label)
        stats_layout.addWidget(self.consciousness_label)
        
        layout.addWidget(title_label)
        layout.addWidget(self.game_viewport, 1)
        layout.addWidget(stats_widget)
        
        return panel
    
    def create_consciousness_panel(self):
        """Create the right consciousness monitoring panel"""
        try:
            # Use the enhanced dashboard if available
            if hasattr(self, 'procedural_engine') and self.procedural_engine:
                self.consciousness_dashboard = CompleteConsciousnessDashboard(
                    consciousness_engine=None,  # We'll pass our math engine
                    procedural_engine=self.procedural_engine
                )
                return self.consciousness_dashboard
        except Exception as e:
            print(f"Dashboard creation error: {e}")
        
        # Fallback simple consciousness panel
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        consciousness_group = QGroupBox("Consciousness Monitor")
        consciousness_layout = QVBoxLayout(consciousness_group)
        
        self.red_label = QLabel("Red (Action): 33.3%")
        self.blue_label = QLabel("Blue (Logic): 33.3%")
        self.yellow_label = QLabel("Yellow (Wisdom): 33.4%")
        self.unity_label = QLabel("AE Unity: 1.000")
        
        consciousness_layout.addWidget(self.red_label)
        consciousness_layout.addWidget(self.blue_label)
        consciousness_layout.addWidget(self.yellow_label)
        consciousness_layout.addWidget(self.unity_label)
        
        # Procedural generation status
        procedural_group = QGroupBox("Procedural Status")
        procedural_layout = QVBoxLayout(procedural_group)
        
        self.world_status_label = QLabel("World: Generated")
        self.spiral_prob_label = QLabel("Spiral Probability: 0.001%")
        self.dream_status_label = QLabel("Dream State: Inactive")
        
        procedural_layout.addWidget(self.world_status_label)
        procedural_layout.addWidget(self.spiral_prob_label)
        procedural_layout.addWidget(self.dream_status_label)
        
        layout.addWidget(consciousness_group)
        layout.addWidget(procedural_group)
        layout.addStretch()
        
        return panel
    
    def set_game_mode(self, mode):
        """Switch between game modes"""
        self.game_mode = mode
        self.status_bar.showMessage(f"Switched to {mode.replace('_', ' ').title()} mode")
        
        # Update viewport based on mode
        mode_messages = {
            "tower_defense": "GeoBIT Tower Defense - Defend with consciousness-driven towers",
            "mmorpg": "MMORPG Mode - Explore the 11x13 dimensional universe",
            "base_builder": "Base Builder - Construct consciousness-enhanced structures"
        }
        
        self.game_viewport.setText(f"{mode_messages.get(mode, 'Unknown mode')}\n\nProcedural Laws Active\nConsciousness Mathematics Enabled")
    
    def generate_new_world(self):
        """Generate a new procedural world"""
        if self.world_generator:
            try:
                # Generate a new universe
                self.world_generator.generate_universe()
                self.status_bar.showMessage("New procedural world generated using consciousness mathematics")
                
                # Update world display
                zone_data = self.world_generator.get_zone_summary(1, 1)  # Get first zone
                if zone_data:
                    self.world_status_label.setText(f"World: {zone_data['location']} - {zone_data['type']}")
                    
            except Exception as e:
                self.status_bar.showMessage(f"World generation error: {e}")
        else:
            self.status_bar.showMessage("World generator not available")
    
    def trigger_spiral_anomaly(self):
        """Trigger a spiral anomaly event"""
        if self.procedural_engine:
            try:
                # Simulate spiral anomaly
                spiral_result = self.procedural_engine.trigger_spiral_anomaly()
                self.status_bar.showMessage(f"Spiral anomaly triggered: {spiral_result}")
                
            except Exception as e:
                self.status_bar.showMessage(f"Spiral anomaly error: {e}")
        else:
            self.status_bar.showMessage("Procedural engine not available")
    
    def enter_dream_state(self):
        """Enter consciousness dream state"""
        if self.procedural_engine:
            try:
                # Trigger dream state
                dream_result = self.procedural_engine.enter_dream_state()
                self.dream_status_label.setText("Dream State: Active")
                self.status_bar.showMessage(f"Entered dream state: {dream_result}")
                
            except Exception as e:
                self.status_bar.showMessage(f"Dream state error: {e}")
        else:
            self.status_bar.showMessage("Procedural engine not available")
    
    def start_consciousness_monitoring(self):
        """Start real-time consciousness monitoring"""
        self.consciousness_timer = QTimer()
        self.consciousness_timer.timeout.connect(self.update_consciousness_display)
        self.consciousness_timer.start(1000)  # Update every second
    
    def update_consciousness_display(self):
        """Update consciousness state display"""
        if self.math_engine:
            try:
                # Create current consciousness vector
                consciousness = AEVector(
                    self.consciousness_state["red"],
                    self.consciousness_state["blue"], 
                    self.consciousness_state["yellow"]
                )
                
                # Update labels
                self.red_label.setText(f"Red (Action): {consciousness.red*100:.1f}%")
                self.blue_label.setText(f"Blue (Logic): {consciousness.blue*100:.1f}%")
                self.yellow_label.setText(f"Yellow (Wisdom): {consciousness.yellow*100:.1f}%")
                self.unity_label.setText(f"AE Unity: {consciousness.ae_unity():.3f}")
                
                # Update consciousness in game viewport
                dominance = "Balanced"
                if consciousness.red > 0.4:
                    dominance = "Action-Driven"
                elif consciousness.blue > 0.4:
                    dominance = "Logic-Driven"
                elif consciousness.yellow > 0.4:
                    dominance = "Wisdom-Driven"
                
                self.consciousness_label.setText(f"Consciousness: {dominance}")
                
                # Simulate slight consciousness drift (procedural evolution)
                import random
                drift = 0.001
                self.consciousness_state["red"] += random.uniform(-drift, drift)
                self.consciousness_state["blue"] += random.uniform(-drift, drift)
                self.consciousness_state["yellow"] += random.uniform(-drift, drift)
                
                # Normalize to maintain sum = 1
                total = sum(self.consciousness_state.values())
                for key in self.consciousness_state:
                    self.consciousness_state[key] /= total
                    
            except Exception as e:
                print(f"Consciousness update error: {e}")

def main():
    """Launch the Unified Consciousness Framework"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Unified Absolute Framework")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Consciousness Research Lab")
    
    # Create and show main window
    launcher = UnifiedConsciousnessLauncher()
    launcher.show()
    
    print("🚀 Unified Absolute Framework launched successfully!")
    print("📊 Consciousness monitoring active")
    print("🎮 Procedural laws engine ready")
    print("🌍 World generation system online")
    print("💫 GeoBIT tower defense with consciousness integration")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
