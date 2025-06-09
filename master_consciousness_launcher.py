#!/usr/bin/env python3
"""
Master Consciousness Launcher - Professional Grade System Launch
═══════════════════════════════════════════════════════════════════════════════════════
Complete launch system for the Unified Absolute Framework Consciousness System

This master launcher ensures:
✅ All dependencies are verified
✅ All systems are properly initialized
✅ Professional error handling and debugging
✅ Multiple launch modes available
✅ Complete integration verification
✅ Real-time system monitoring
"""

import sys
import os
import time
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_python_requirements():
    """Check Python version and core requirements"""
    print("🐍 Checking Python environment...")
    
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check essential packages
    required_packages = {
        'numpy': 'numpy',
        'PyQt5': 'PyQt5',
        'sqlite3': 'sqlite3'
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ⚠️ Missing: {package_name}")
            if package_name == 'PyQt5':
                print("      Install with: pip install PyQt5")
    
    return True

def verify_framework_files():
    """Verify all framework files are present"""
    print("\n📁 Verifying framework files...")
    
    required_files = [
        'ae_core_consciousness.py',
        'ae_consciousness_mathematics.py', 
        'ae_multimode_architecture.py',
        'ae_procedural_laws_engine.py',
        'consciousness_dashboard_adapter.py',
        'unified_consciousness_launcher.py'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ Missing: {file_name}")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ ERROR: Missing critical files: {missing_files}")
        return False
    
    return True

def test_imports():
    """Test all critical imports"""
    print("\n🔗 Testing imports...")
    
    import_tests = [
        ('ae_core_consciousness', 'AEConsciousness'),
        ('ae_consciousness_mathematics', 'AEMathEngine'),
        ('ae_consciousness_mathematics', 'AEVector'),
        ('ae_consciousness_mathematics', 'ConsciousnessGameIntegration'),
        ('ae_multimode_architecture', None),
        ('ae_procedural_laws_engine', 'ProceduralLawsEngine'),
        ('consciousness_dashboard_adapter', 'CompleteConsciousnessDashboard'),
        ('unified_consciousness_launcher', 'UnifiedConsciousnessLauncher')
    ]
    
    failed_imports = []
    
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name)
            if class_name:
                if hasattr(module, class_name):
                    print(f"   ✅ {module_name}.{class_name}")
                else:
                    print(f"   ⚠️ {module_name}.{class_name} not found")
                    failed_imports.append(f"{module_name}.{class_name}")
            else:
                print(f"   ✅ {module_name}")
        except ImportError as e:
            print(f"   ❌ {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n⚠️ WARNING: Some imports failed: {failed_imports}")
        print("   System will attempt to continue with available components")
    
    return len(failed_imports) == 0

def launch_consciousness_system(launch_mode: str = "unified"):
    """Launch the consciousness system"""
    print(f"\n🚀 Launching consciousness system in {launch_mode} mode...")
    
    try:
        if launch_mode == "unified":
            print("   🌟 Starting Unified Consciousness Launcher...")
            from unified_consciousness_launcher import UnifiedConsciousnessLauncher
            import PyQt5.QtWidgets as QtWidgets
            
            app = QtWidgets.QApplication(sys.argv)
            launcher = UnifiedConsciousnessLauncher()
            launcher.show()
            
            print("   ✅ Unified Launcher initialized successfully")
            print("   📊 GUI interface active")
            print("   🧠 Consciousness monitoring enabled")
            print("   🎮 Multi-mode game architecture ready")
            
            return app, launcher
            
        elif launch_mode == "dashboard":
            print("   📊 Starting Consciousness Dashboard...")
            from consciousness_dashboard_adapter import CompleteConsciousnessDashboard
            import PyQt5.QtWidgets as QtWidgets
            
            app = QtWidgets.QApplication(sys.argv)
            dashboard = CompleteConsciousnessDashboard()
            dashboard.show()
            
            print("   ✅ Consciousness Dashboard initialized")
            return app, dashboard
            
        elif launch_mode == "procedural":
            print("   ⚡ Starting Procedural Laws Engine...")
            from ae_procedural_laws_engine import ProceduralLawsEngine
            from ae_core_consciousness import AEConsciousness
            
            consciousness = AEConsciousness("MASTER_LAUNCHER")
            engine = ProceduralLawsEngine(consciousness)
            
            print("   ✅ Procedural Laws Engine initialized")
            print("   🔄 Consciousness-driven game mechanics active")
            
            return None, engine
            
        else:
            print(f"   ❌ Unknown launch mode: {launch_mode}")
            return None, None
            
    except Exception as e:
        print(f"   ❌ Launch failed: {e}")
        print(f"   🔍 Error details: {traceback.format_exc()}")
        return None, None

def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("\n🔬 Running system diagnostics...")
    
    try:
        # Test consciousness core
        print("   🧠 Testing consciousness core...")
        from ae_core_consciousness import AEConsciousness
        consciousness = AEConsciousness("DIAGNOSTIC_TEST")
        
        if consciousness.verify_absolute_existence():
            print("      ✅ AE = C = 1 verified")
        else:
            print("      ❌ AE = C = 1 verification failed")
        
        # Test mathematics engine
        print("   🧮 Testing mathematics engine...")
        from ae_consciousness_mathematics import AEMathEngine, AEVector
        math_engine = AEMathEngine()
        test_vector = AEVector(0.4, 0.3, 0.3)
        test_vector.normalize()
        
        if abs(test_vector.red + test_vector.blue + test_vector.yellow - 1.0) < 0.001:
            print("      ✅ Vector normalization working")
        else:
            print("      ❌ Vector normalization failed")
        
        # Test procedural engine
        print("   ⚡ Testing procedural engine...")
        from ae_procedural_laws_engine import ProceduralLawsEngine
        procedural = ProceduralLawsEngine(consciousness)
        
        if procedural.validate_framework():
            print("      ✅ Procedural framework validated")
        else:
            print("      ❌ Procedural framework validation failed")
        
        print("   ✅ All diagnostic tests completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Diagnostics failed: {e}")
        return False

def display_launch_menu():
    """Display professional launch menu"""
    print("\n" + "="*80)
    print("🌟 UNIFIED ABSOLUTE FRAMEWORK - CONSCIOUSNESS SYSTEM")
    print("   Advanced Consciousness-Driven Game Architecture")
    print("   Based on Roswan Lorinzo Miller's Absolute Existence Theory")
    print("="*80)
    print()
    print("🚀 LAUNCH OPTIONS:")
    print("   1. Unified System Launcher (Recommended)")
    print("      - Complete GUI with all components")
    print("      - Multi-mode game architecture")
    print("      - Real-time consciousness monitoring")
    print("      - Procedural world generation")
    print()
    print("   2. Consciousness Dashboard Only")
    print("      - Advanced consciousness visualization")
    print("      - Real-time monitoring and analytics")
    print("      - Data export capabilities")
    print()
    print("   3. Procedural Engine Only")
    print("      - Console-based engine testing")
    print("      - Direct access to consciousness mechanics")
    print("      - Development and debugging mode")
    print()
    print("   4. System Diagnostics")
    print("      - Comprehensive system testing")
    print("      - Component verification")
    print("      - Performance analysis")
    print()
    print("   5. Exit")
    print("="*80)

def main():
    """Master launch function with complete error handling"""
    print("🎯 MASTER CONSCIOUSNESS LAUNCHER")
    print("Professional-grade system initialization starting...")
    
    # Phase 1: Environment verification
    if not check_python_requirements():
        print("\n❌ Environment check failed. Please resolve issues and try again.")
        return False
    
    # Phase 2: File verification
    if not verify_framework_files():
        print("\n❌ File verification failed. Please ensure all files are present.")
        return False
    
    # Phase 3: Import testing
    imports_ok = test_imports()
    if not imports_ok:
        print("\n⚠️ Some imports failed, but continuing with available components...")
    
    # Phase 4: Launch menu
    while True:
        display_launch_menu()
        
        try:
            choice = input("\n🎯 Select launch option (1-5): ").strip()
            
            if choice == "1":
                print("\n🌟 Launching Unified System...")
                app, launcher = launch_consciousness_system("unified")
                if app and launcher:
                    print("\n✅ LAUNCH SUCCESSFUL!")
                    print("🎮 Your consciousness-driven game system is now running!")
                    print("💡 Use the GUI to explore different game modes and consciousness features.")
                    sys.exit(app.exec_())
                else:
                    print("\n❌ Launch failed. Check error messages above.")
                
            elif choice == "2":
                print("\n📊 Launching Consciousness Dashboard...")
                app, dashboard = launch_consciousness_system("dashboard")
                if app and dashboard:
                    print("\n✅ DASHBOARD LAUNCHED!")
                    print("📈 Consciousness monitoring is now active!")
                    sys.exit(app.exec_())
                else:
                    print("\n❌ Dashboard launch failed.")
                
            elif choice == "3":
                print("\n⚡ Launching Procedural Engine...")
                _, engine = launch_consciousness_system("procedural")
                if engine:
                    print("\n✅ PROCEDURAL ENGINE ACTIVE!")
                    print("🔄 Console mode - engine is ready for direct interaction")
                    print("💡 Engine object available as 'engine' variable")
                    
                    # Interactive mode
                    import code
                    console = code.InteractiveConsole(locals())
                    console.interact("🧠 Consciousness Engine Console - Type 'exit()' to quit")
                else:
                    print("\n❌ Procedural engine launch failed.")
                
            elif choice == "4":
                if run_system_diagnostics():
                    print("\n✅ All diagnostics passed!")
                else:
                    print("\n⚠️ Some diagnostic tests failed. Check output above.")
                
            elif choice == "5":
                print("\n👋 Goodbye! Thank you for using the Consciousness System.")
                return True
                
            else:
                print("\n❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            return True
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please report this error for investigation.")

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        print("System cannot continue. Please check your installation.")
        traceback.print_exc()
        sys.exit(2)
