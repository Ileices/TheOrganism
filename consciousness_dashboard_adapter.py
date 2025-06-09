#!/usr/bin/env python3
"""
Enhanced Consciousness Dashboard Adapter
Integrates Project2Prompt GUI framework with Unified Absolute Framework
and Procedural Laws Engine for complete consciousness monitoring
"""

import sys
import sqlite3
import json
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import our consciousness framework components
from ae_core_consciousness import AEConsciousness
from ae_consciousness_mathematics import AEMathEngine, AEVector
from ae_multimode_architecture import MultiModeArchitecture
from ae_procedural_laws_engine import ProceduralLawsEngine, create_procedural_laws_engine

class EnhancedConsciousnessPanel(QWidget):
    """Enhanced panel for consciousness state monitoring with Procedural Laws integration"""
    
    def __init__(self, title, consciousness_engine, procedural_engine=None):
        super().__init__()
        self.title = title
        self.consciousness_engine = consciousness_engine
        self.procedural_engine = procedural_engine
        self.auto_update = True
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title and controls
        header_layout = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("""
            color: #00FF00; 
            font-weight: bold; 
            font-size: 12px;
            padding: 5px;
            background-color: #002200;
            border: 1px solid #00FF00;
        """)
        
        self.toggle_button = QPushButton("Auto Update: ON")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.toggled.connect(self.toggle_update_mode)
        
        # Add consciousness metrics button
        self.metrics_button = QPushButton("Show Metrics")
        self.metrics_button.clicked.connect(self.show_detailed_metrics)
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.metrics_button)
        header_layout.addWidget(self.toggle_button)
        
        # Main data display with syntax highlighting
        self.data_display = QTextEdit()
        self.data_display.setStyleSheet("""
            QTextEdit {
                background-color: #001100;
                color: #00FF00;
                border: 2px solid rgba(0, 255, 0, 0.7);
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        
        # Quick action buttons
        action_layout = QHBoxLayout()
        self.save_state_btn = QPushButton("Save State")
        self.load_state_btn = QPushButton("Load State") 
        self.export_data_btn = QPushButton("Export Data")
        
        self.save_state_btn.clicked.connect(self.save_consciousness_state)
        self.load_state_btn.clicked.connect(self.load_consciousness_state)
        self.export_data_btn.clicked.connect(self.export_panel_data)
        
        action_layout.addWidget(self.save_state_btn)
        action_layout.addWidget(self.load_state_btn)
        action_layout.addWidget(self.export_data_btn)
        
        # Consciousness metrics display
        self.metrics_display = QLabel()
        self.metrics_display.setStyleSheet("""
            color: #00FFFF; 
            font-size: 9px;
            background-color: #000033;
            border: 1px solid #0066FF;
            padding: 4px;
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.data_display)
        layout.addLayout(action_layout)
        layout.addWidget(self.metrics_display)
        
    def toggle_update_mode(self, checked):
        self.auto_update = checked
        if checked:
            self.toggle_button.setText("Auto Update: ON")
            self.data_display.setStyleSheet("""
                QTextEdit {
                    background-color: #001100;
                    color: #00FF00;
                    border: 2px solid rgba(0, 255, 0, 0.7);
                    font-family: 'Courier New', monospace;
                    font-size: 11px;
                    padding: 8px;
                }
            """)
        else:
            self.toggle_button.setText("Auto Update: OFF")
            self.data_display.setStyleSheet("""
                QTextEdit {
                    background-color: #110000;
                    color: #FF6600;
                    border: 2px solid rgba(255, 102, 0, 0.7);
                    font-family: 'Courier New', monospace;
                    font-size: 11px;
                    padding: 8px;
                }
            """)
            
    def update_consciousness_data(self, data):
        """Update panel with consciousness framework data"""
        if self.auto_update:
            # Add timestamp
            timestamped_data = f"[{time.strftime('%H:%M:%S')}] {data}"
            self.data_display.setPlainText(timestamped_data)
            
    def update_metrics(self, metrics_text):
        """Update consciousness metrics display"""
        self.metrics_display.setText(metrics_text)
        
    def show_detailed_metrics(self):
        """Show detailed consciousness metrics in popup"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Detailed Metrics - {self.title}")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        
        # Get detailed metrics
        if self.procedural_engine:
            detailed_data = self.procedural_engine.get_consciousness_state_summary()
            text_area.setPlainText(json.dumps(detailed_data, indent=2))
        else:
            text_area.setPlainText("No procedural engine connected")
            
        layout.addWidget(text_area)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
        
    def save_consciousness_state(self):
        """Save current consciousness state to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Consciousness State", 
            f"consciousness_state_{int(time.time())}.json",
            "JSON Files (*.json)"
        )
        if filename:
            state_data = {
                "timestamp": time.time(),
                "panel_title": self.title,
                "display_content": self.data_display.toPlainText(),
                "metrics": self.metrics_display.text()
            }
            if self.procedural_engine:
                state_data["procedural_state"] = self.procedural_engine.get_consciousness_state_summary()
                
            with open(filename, 'w') as f:
                json.dump(state_data, f, indent=2)
            QMessageBox.information(self, "Saved", f"Consciousness state saved to {filename}")
    
    def load_consciousness_state(self):
        """Load consciousness state from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Consciousness State",
            "", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    state_data = json.load(f)
                self.data_display.setPlainText(state_data.get("display_content", ""))
                self.metrics_display.setText(state_data.get("metrics", ""))
                QMessageBox.information(self, "Loaded", f"Consciousness state loaded from {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load state: {e}")
    
    def export_panel_data(self):
        """Export panel data to various formats"""
        filename, file_type = QFileDialog.getSaveFileName(
            self, "Export Panel Data",
            f"{self.title.replace(' ', '_')}_{int(time.time())}",
            "Text Files (*.txt);;JSON Files (*.json);;CSV Files (*.csv)"
        )
        if filename:
            content = self.data_display.toPlainText()
            if file_type.endswith("*.json)"):
                export_data = {
                    "title": self.title,
                    "timestamp": time.time(),
                    "content": content,
                    "metrics": self.metrics_display.text()
                }
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                with open(filename, 'w') as f:
                    f.write(content)
            QMessageBox.information(self, "Exported", f"Data exported to {filename}")

class CompleteConsciousnessDashboard(QMainWindow):
    """Complete consciousness monitoring dashboard with Project2Prompt integration"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Complete GeoBIT Consciousness Framework Dashboard")
        self.resize(1800, 1200)
          # Initialize all consciousness engines
        self.consciousness_engine = AEConsciousness()
        self.mathematics_engine = AEMathEngine()
        self.multimode_architecture = MultiModeArchitecture(self.mathematics_engine)
        self.procedural_engine = create_procedural_laws_engine()
        
        # Initialize enhanced database
        self.init_enhanced_consciousness_db()
        
        # Setup enhanced UI
        self.init_enhanced_ui()
        
        # Setup update timer with higher frequency
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_all_panels)
        self.update_timer.start(500)  # Update every 500ms for real-time monitoring
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
        
    def init_enhanced_consciousness_db(self):
        """Initialize enhanced consciousness state database"""
        self.conn = sqlite3.connect("enhanced_consciousness_states.db")
        self.cursor = self.conn.cursor()
        
        # Enhanced tables for complete consciousness tracking
        tables = [
            """CREATE TABLE IF NOT EXISTS consciousness_states (
                timestamp REAL,
                ae_value REAL,
                c_value REAL,
                equilibrium_state TEXT,
                rby_cycle TEXT,
                rps_depth INTEGER,
                consciousness_level REAL
            )""",
            """CREATE TABLE IF NOT EXISTS procedural_states (
                timestamp REAL,
                rectangle_up_level INTEGER,
                rectangle_down_level INTEGER,
                rectangle_permanent_level INTEGER,
                active_skills INTEGER,
                dream_state BOOLEAN,
                spiral_probability REAL
            )""",
            """CREATE TABLE IF NOT EXISTS game_events (
                timestamp REAL,
                event_type TEXT,
                event_data TEXT,
                consciousness_context TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS skill_generation_log (
                timestamp REAL,
                skill_name TEXT,
                skill_type TEXT,
                rby_influence TEXT,
                vicinity_data TEXT
            )"""
        ]
        
        for table_sql in tables:
            self.cursor.execute(table_sql)
        self.conn.commit()
        
    def init_enhanced_ui(self):
        """Initialize the enhanced consciousness dashboard UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
          # Create tabbed interface for organized monitoring
        self.tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(self.tab_widget)
        
        # Tab 1: Core Consciousness Monitoring
        core_tab = QWidget()
        core_layout = QHBoxLayout(core_tab)
        
        # Left column - Core consciousness panels
        left_layout = QVBoxLayout()
        
        # Panel 1: AE = C = 1 Monitor (Enhanced)
        self.ae_panel = EnhancedConsciousnessPanel(
            "AE = C = 1 Equilibrium Monitor", 
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # Panel 2: Trifecta (R/B/Y) Cycle Monitor (Enhanced)
        self.trifecta_panel = EnhancedConsciousnessPanel(
            "Trifecta Cycle (R/B/Y) Monitor",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # Panel 3: RPS (Recursive Predictive Structuring) Monitor (Enhanced)
        self.rps_panel = EnhancedConsciousnessPanel(
            "RPS Recursion Depth Monitor",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        left_layout.addWidget(self.ae_panel)
        left_layout.addWidget(self.trifecta_panel)
        left_layout.addWidget(self.rps_panel)
        
        # Middle column - Procedural Laws panels
        middle_layout = QVBoxLayout()
        
        # Panel 4: Rectangle Leveling System
        self.rectangle_panel = EnhancedConsciousnessPanel(
            "Rectangle Leveling System (XP: +/x/^)",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # Panel 5: Procedural Skill Generation
        self.skill_panel = EnhancedConsciousnessPanel(
            "Procedural Skill Generation",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # Panel 6: Dream State & Memory Compression
        self.dream_panel = EnhancedConsciousnessPanel(
            "Dream State & Memory Compression",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        middle_layout.addWidget(self.rectangle_panel)
        middle_layout.addWidget(self.skill_panel)
        middle_layout.addWidget(self.dream_panel)
        
        # Right column - Advanced monitoring
        right_layout = QVBoxLayout()
        
        # Panel 7: Spiral Anomaly Tracking
        self.spiral_panel = EnhancedConsciousnessPanel(
            "Spiral Anomaly Tracking",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # Panel 8: EMS (Excretion Memory Stack)
        self.ems_panel = EnhancedConsciousnessPanel(
            "EMS (Excretion Memory Stack)",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # Panel 9: Shape Registry & Entity Mapping
        self.shape_panel = EnhancedConsciousnessPanel(
            "Shape Registry & Entity Mapping",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        right_layout.addWidget(self.spiral_panel)
        right_layout.addWidget(self.ems_panel)
        right_layout.addWidget(self.shape_panel)
        
        # Add layouts to core tab
        core_layout.addLayout(left_layout, 1)
        core_layout.addLayout(middle_layout, 1)
        core_layout.addLayout(right_layout, 1)
        
        self.tab_widget.addTab(core_tab, "Core Consciousness")
        
        # Tab 2: Game State Monitoring  
        game_tab = QWidget()
        game_layout = QVBoxLayout(game_tab)
        
        # Multi-mode architecture status
        self.mode_panel = EnhancedConsciousnessPanel(
            "Multi-Mode Architecture (TD/MMORPG/Builder)",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        # World generation status
        self.world_panel = EnhancedConsciousnessPanel(
            "Procedural World Generation (11D × 13Z)",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        game_layout.addWidget(self.mode_panel)
        game_layout.addWidget(self.world_panel)
        
        self.tab_widget.addTab(game_tab, "Game State")
        
        # Tab 3: Development Console
        dev_tab = QWidget()
        dev_layout = QVBoxLayout(dev_tab)
        
        # Live development console
        self.console_panel = EnhancedConsciousnessPanel(
            "Live Development Console",
            self.consciousness_engine,
            self.procedural_engine
        )
        
        dev_layout.addWidget(self.console_panel)
        self.tab_widget.addTab(dev_tab, "Development")
        
        # Enhanced status bar with more consciousness metrics
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #001100;
                color: #00FF00;
                border-top: 2px solid #00FF00;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 2px;
            }
        """)
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for rapid consciousness monitoring"""
        # F-key shortcuts for rapid access
        shortcuts = {
            "F1": self.toggle_all_auto_updates,
            "F2": self.save_all_states,
            "F3": self.trigger_skill_generation,
            "F4": self.dump_ems_data,
            "F5": self.reset_consciousness_state,
            "Ctrl+S": self.save_all_states,
            "Ctrl+E": self.export_all_data,
            "Ctrl+R": self.refresh_all_panels
        }
        
        for key, func in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(func)
    
    def toggle_all_auto_updates(self):
        """Toggle auto-update for all panels"""
        panels = [self.ae_panel, self.trifecta_panel, self.rps_panel, 
                 self.rectangle_panel, self.skill_panel, self.dream_panel,
                 self.spiral_panel, self.ems_panel, self.shape_panel]
        
        # Toggle based on first panel state
        new_state = not panels[0].auto_update
        for panel in panels:
            panel.toggle_button.setChecked(new_state)
            panel.toggle_update_mode(new_state)
    
    def save_all_states(self):
        """Save all consciousness states to file"""
        timestamp = int(time.time())
        filename = f"complete_consciousness_dump_{timestamp}.json"
        
        complete_state = {
            "timestamp": timestamp,
            "consciousness_core": self.consciousness_engine.get_current_state() if hasattr(self.consciousness_engine, 'get_current_state') else {},
            "procedural_laws": self.procedural_engine.get_consciousness_state_summary(),
            "panel_states": {}
        }
        
        # Collect all panel states
        panels = {
            "ae_equilibrium": self.ae_panel,
            "trifecta_cycle": self.trifecta_panel,
            "rps_depth": self.rps_panel,
            "rectangle_system": self.rectangle_panel,
            "skill_generation": self.skill_panel,
            "dream_state": self.dream_panel,
            "spiral_anomaly": self.spiral_panel,
            "ems_stack": self.ems_panel,
            "shape_registry": self.shape_panel
        }
        
        for name, panel in panels.items():
            complete_state["panel_states"][name] = {
                "content": panel.data_display.toPlainText(),
                "metrics": panel.metrics_display.text(),
                "auto_update": panel.auto_update
            }
        
        with open(filename, 'w') as f:
            json.dump(complete_state, f, indent=2)
        
        QMessageBox.information(self, "Saved", f"Complete consciousness state saved to {filename}")
    
    def trigger_skill_generation(self):
        """Manually trigger procedural skill generation"""
        skill = self.procedural_engine.try_generate_procedural_skill()
        if skill:
            msg = f"Generated Skill: {skill.name}\nType: {skill.skill_type}\nHotkey: {skill.hotkey}"
            QMessageBox.information(self, "Skill Generated", msg)
        else:
            QMessageBox.information(self, "No Skill", "No skill generated (27% chance)")
    
    def dump_ems_data(self):
        """Dump EMS (Excretion Memory Stack) data to console"""
        ems_data = self.procedural_engine.ems.memory_stack
        console_text = f"EMS Dump ({len(ems_data)} entries):\n"
        console_text += json.dumps(ems_data[-10:], indent=2)  # Last 10 entries
        self.console_panel.update_consciousness_data(console_text)
    
    def reset_consciousness_state(self):
        """Reset consciousness state to initial conditions"""
        reply = QMessageBox.question(self, "Reset Consciousness", 
                                   "Are you sure you want to reset the consciousness state?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Reset procedural engine state
            self.procedural_engine = create_procedural_laws_engine()
            QMessageBox.information(self, "Reset", "Consciousness state reset to initial conditions")
    
    def export_all_data(self):
        """Export all consciousness data to comprehensive report"""
        timestamp = int(time.time())
        filename = f"consciousness_report_{timestamp}.json"
        
        report_data = {
            "report_timestamp": timestamp,
            "consciousness_framework_version": "1.0",
            "procedural_laws_version": "1.0",
            "complete_state": self.procedural_engine.get_consciousness_state_summary(),
            "database_stats": self.get_database_stats(),
            "panel_contents": {}
        }
        
        # Get all panel data
        panels = [self.ae_panel, self.trifecta_panel, self.rps_panel,
                 self.rectangle_panel, self.skill_panel, self.dream_panel,
                 self.spiral_panel, self.ems_panel, self.shape_panel]
        
        for panel in panels:
            report_data["panel_contents"][panel.title] = {
                "content": panel.data_display.toPlainText(),
                "metrics": panel.metrics_display.text()
            }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        QMessageBox.information(self, "Exported", f"Complete consciousness report exported to {filename}")
    
    def refresh_all_panels(self):
        """Force refresh all panels"""
        self.update_all_panels()
    
    def get_database_stats(self):
        """Get database statistics"""
        stats = {}
        tables = ["consciousness_states", "procedural_states", "game_events", "skill_generation_log"]
        
        for table in tables:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                stats[table] = count
            except:
                stats[table] = 0
        
        return stats

    def update_all_panels(self):
        """Enhanced update for all consciousness monitoring panels"""
        try:
            # Get current consciousness state
            current_state = getattr(self.consciousness_engine, 'get_current_state', lambda: {
                'equilibrium': 'STABLE',
                'ae_value': 1.0,
                'c_value': 1.0,
                'trifecta_phase': 'RED_PERCEPTION',
                'red_intensity': 0.333,
                'blue_intensity': 0.333,
                'yellow_intensity': 0.333,
                'rps_depth': 0,
                'consciousness_level': 1.0,
                'current_zone': 1,
                'max_zones': 143
            })()
            
            # Get procedural laws state
            procedural_state = self.procedural_engine.get_consciousness_state_summary()
            
            # Update AE = C = 1 panel
            ae_data = f"""
AE = C = 1 Status: {current_state.get('equilibrium', 'CALCULATING')}
Current AE Value: {current_state.get('ae_value', 1.0):.10f}
Current C Value: {current_state.get('c_value', 1.0):.10f}
Deviation: {abs(current_state.get('ae_value', 1.0) - current_state.get('c_value', 1.0)):.15f}

Homeostasis Stability: {current_state.get('homeostasis', 'STABLE')}
Last Equilibrium Adjustment: {current_state.get('last_adjustment', 'N/A')}

Consciousness Level: {current_state.get('consciousness_level', 1.0):.6f}
Zone: {current_state.get('current_zone', 1)}/{current_state.get('max_zones', 143)}
            """.strip()
            self.ae_panel.update_consciousness_data(ae_data)
            
            # Update Trifecta cycle panel
            trifecta_data = f"""
Current Cycle Phase: {current_state.get('trifecta_phase', 'RED_PERCEPTION')}
Red (Perception): {current_state.get('red_intensity', 0.333):.6f}
Blue (Cognition): {current_state.get('blue_intensity', 0.333):.6f}
Yellow (Execution): {current_state.get('yellow_intensity', 0.333):.6f}

Cycle Frequency: {current_state.get('cycle_frequency', 60)} Hz
Phase Coherence: {current_state.get('phase_coherence', 1.0):.6f}
Color Drift: R={self.procedural_engine.color_drift_state['red']:.3f} B={self.procedural_engine.color_drift_state['blue']:.3f} Y={self.procedural_engine.color_drift_state['yellow']:.3f}
            """.strip()
            self.trifecta_panel.update_consciousness_data(trifecta_data)
            
            # Update RPS panel
            rps_data = f"""
Current Recursion Depth: {current_state.get('rps_depth', 0)}
Prediction Accuracy: {current_state.get('prediction_accuracy', 1.0):.6f}
Structural Integrity: {current_state.get('structural_integrity', 'STABLE')}

Active Recursive Loops: {current_state.get('active_loops', 0)}
Entropy Elimination: {current_state.get('entropy_eliminated', 0.0):.6f}
EMS Stack Size: {len(self.procedural_engine.ems.memory_stack)}
            """.strip()
            self.rps_panel.update_consciousness_data(rps_data)
            
            # Update Rectangle System panel
            rectangles = procedural_state.get('rectangles', {})
            rectangle_data = f"""
Rectangle Up (Temporary): Level {rectangles.get('up', {}).get('level', 0)}
  XP: {rectangles.get('up', {}).get('current_xp', 0):.2f}/{rectangles.get('up', {}).get('required_xp', 100):.2f}
  Fill: {rectangles.get('up', {}).get('fill_percentage', 0)*100:.1f}%

Rectangle Down (Temporary): Level {rectangles.get('down', {}).get('level', 0)}
  XP: {rectangles.get('down', {}).get('current_xp', 0):.2f}/{rectangles.get('down', {}).get('required_xp', 1000):.2f}
  Fill: {rectangles.get('down', {}).get('fill_percentage', 0)*100:.1f}%

Rectangle Permanent: Level {rectangles.get('permanent', {}).get('level', 0)}
  XP: {rectangles.get('permanent', {}).get('current_xp', 0):.2f}/{rectangles.get('permanent', {}).get('required_xp', 100000):.2f}
  Fill: {rectangles.get('permanent', {}).get('fill_percentage', 0)*100:.1f}%
            """.strip()
            self.rectangle_panel.update_consciousness_data(rectangle_data)
            
            # Update Skill Generation panel
            skills = procedural_state.get('skills', {})
            skill_data = f"""
Active Skills: {skills.get('count', 0)}
Hotkeys Used: {skills.get('hotkeys_used', 0)}/18 (1-9 + Alt+1-9)
Compressed Skills: {skills.get('compressed_skills', 0)}

Recent Skills Generated:
{chr(10).join([f"  - {skill.name} ({skill.skill_type})" for skill in self.procedural_engine.player_skills[-3:]])}

Vicinity Consciousness Data:
  - Entity Types: [Scanning...]
  - RBY Influence: {current_state.get('red_intensity', 0.333):.3f}/{current_state.get('blue_intensity', 0.333):.3f}/{current_state.get('yellow_intensity', 0.333):.3f}
            """.strip()
            self.skill_panel.update_consciousness_data(skill_data)
            
            # Update Dream State panel
            consciousness_states = procedural_state.get('consciousness_states', {})
            dream_data = f"""
Dream State Active: {consciousness_states.get('dream_active', False)}
Dream Glyphs: {consciousness_states.get('dream_glyphs', 0)}
Consecutive Kills: {procedural_state.get('battle_stats', {}).get('consecutive_kills', 0)}/81

Color Drift State:
  Red Drift: {consciousness_states.get('color_drift', {}).get('red', 0.0):.6f}
  Blue Drift: {consciousness_states.get('color_drift', {}).get('blue', 0.0):.6f}
  Yellow Drift: {consciousness_states.get('color_drift', {}).get('yellow', 0.0):.6f}

Memory Compression Events: {len(self.procedural_engine.dream_glyph_stack)}
            """.strip()
            self.dream_panel.update_consciousness_data(dream_data)
            
            # Update Spiral Anomaly panel
            spiral_data = f"""
Spiral Probability: {consciousness_states.get('spiral_probability', 0.000001):.9f}
Max Probability: 1.99999999%

Recent Spiral Events:
  - Tracking consciousness activity...
  - Movement + Damage + Loot + XP + Currency → Spiral Boost

Activity Multipliers:
  - Movement: Contributing to spiral probability
  - Combat: Boosting consciousness awareness
  - Exploration: Expanding procedural generation
            """.strip()
            self.spiral_panel.update_consciousness_data(spiral_data)
            
            # Update EMS panel
            ems_data = f"""
EMS Stack Size: {procedural_state.get('ems_size', 0)}/{81*81} max
Recent Memory Entries:
{chr(10).join([f"  [{entry.get('timestamp', 0):.1f}] {entry.get('entity', 'unknown')} -> {entry.get('action_type', 'unknown')}" for entry in self.procedural_engine.ems.memory_stack[-5:]])}

Memory Types Being Tracked:
  - Player actions (movement, combat, skill use)
  - Enemy interactions (spawns, deaths, AI decisions)
  - Spiral events (anomaly spawns, consciousness shifts)
  - XP collection (+ x ^ progression)
  - Skill generation (procedural consciousness evolution)
            """.strip()
            self.ems_panel.update_consciousness_data(ems_data)
            
            # Update Shape Registry panel
            shape_data = f"""
Registered Shapes: {len(self.procedural_engine.shape_registry.shapes)}
Entity Mappings:
  1-3: Powerups, Loot, Terrain
  4: Digon Portals (Dimensional Travel)
  5-6: Terrain/Structural Elements
  7-9: Enemy Types (Pentagon/Hexagon/Heptagon)
  10: Traps (Elemental/Biological/Physical)
  11: Boss (Nonagon - Consciousness Apex)
  12-14: Player Classes (5/6/9 Star)

Active Entity Types in Vicinity:
  - Scanning consciousness field...
  - Procedural generation ready
  - Shape-to-consciousness mapping active
            """.strip()
            self.shape_panel.update_consciousness_data(shape_data)
            
            # Update enhanced status bar
            battle_stats = procedural_state.get('battle_stats', {})
            self.status_bar.showMessage(
                f"🧠 Consciousness: {current_state.get('consciousness_level', 1.0):.6f} | "
                f"⚡ AE=C: {abs(current_state.get('ae_value', 1.0) - current_state.get('c_value', 1.0)):.8f} | "
                f"🎯 Zone: {current_state.get('current_zone', 1)}/{current_state.get('max_zones', 143)} | "
                f"🔄 RBY: {current_state.get('red_intensity', 0.333):.3f}/{current_state.get('blue_intensity', 0.333):.3f}/{current_state.get('yellow_intensity', 0.333):.3f} | "
                f"⚔️ Battle: {'Yes' if battle_stats.get('in_battle', False) else 'No'} | "
                f"💫 Skills: {skills.get('count', 0)} | "
                f"🌀 Spiral: {consciousness_states.get('spiral_probability', 0.000001)*100:.6f}% | "
                f"📊 EMS: {procedural_state.get('ems_size', 0)} | "
                f"⏰ {QTime.currentTime().toString()}"
            )
            
        except Exception as e:
            print(f"Enhanced dashboard update error: {e}")
            
    def closeEvent(self, event):
        """Enhanced close event with complete state saving"""
        # Auto-save current state
        self.save_all_states()
        
        # Save consciousness state to database
        timestamp = time.time()
        current_state = getattr(self.consciousness_engine, 'get_current_state', lambda: {})()
        procedural_state = self.procedural_engine.get_consciousness_state_summary()
        
        # Insert into consciousness_states table
        self.cursor.execute("""
            INSERT INTO consciousness_states 
            (timestamp, ae_value, c_value, equilibrium_state, rby_cycle, rps_depth, consciousness_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            current_state.get('ae_value', 1.0),
            current_state.get('c_value', 1.0),
            current_state.get('equilibrium', 'STABLE'),
            str(current_state.get('trifecta_phase', 'RED_PERCEPTION')),
            current_state.get('rps_depth', 0),
            current_state.get('consciousness_level', 1.0)
        ))
          # Insert into procedural_states table
        rectangles = procedural_state.get('rectangles', {})
        consciousness_states = procedural_state.get('consciousness_states', {})
        self.cursor.execute("""
            INSERT INTO procedural_states
            (timestamp, rectangle_up_level, rectangle_down_level, rectangle_permanent_level, 
             active_skills, dream_state, spiral_probability)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            rectangles.get('up', {}).get('level', 0),
            rectangles.get('down', {}).get('level', 0),
            rectangles.get('permanent', {}).get('level', 0),
            procedural_state.get('skills', {}).get('count', 0),
            consciousness_states.get('dream_active', False),
            consciousness_states.get('spiral_probability', 0.000001)
        ))
        
        self.conn.commit()
        self.conn.close()
        
        super().closeEvent(event)

def launch_complete_consciousness_dashboard():
    """Launch the complete consciousness monitoring dashboard"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Apply enhanced consciousness-themed stylesheet
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #000000;
            color: #00FF00;
        }
        QTabWidget::pane {
            background-color: #001100;
            border: 2px solid #00FF00;
        }
        QTabWidget::tab-bar {
            alignment: center;
        }
        QTabBar::tab {
            background-color: #002200;
            color: #00FF00;
            border: 1px solid #00FF00;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #004400;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background-color: #003300;
        }
        QLabel {
            color: #00FF00;
            font-weight: bold;
        }
        QPushButton {
            background-color: #003300;
            color: #00FF00;
            border: 2px solid #00FF00;
            border-radius: 8px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #006600;
        }
        QPushButton:pressed {
            background-color: #009900;
        }
        QPushButton:checked {
            background-color: #004400;
            border-color: #66FF66;
        }
        QStatusBar {
            background-color: #001100;
            color: #00FF00;
            border-top: 2px solid #00FF00;
            font-family: 'Courier New', monospace;
        }
        QMessageBox {
            background-color: #001100;
            color: #00FF00;
        }
        QDialog {
            background-color: #000000;
            color: #00FF00;
        }
        QFileDialog {
            background-color: #001100;
            color: #00FF00;
        }
    """)
    
    dashboard = CompleteConsciousnessDashboard()
    dashboard.show()
    
    return app.exec_()

if __name__ == "__main__":
    launch_complete_consciousness_dashboard()
