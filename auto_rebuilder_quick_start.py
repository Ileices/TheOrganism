#!/usr/bin/env python3
"""
Auto-Rebuilder Quick Start Guide
================================

This script demonstrates how to quickly get started with the auto-rebuilder integration
in your Digital Organism system.

Author: Digital Organism Core Team
Date: June 6, 2025
"""

def quick_start_auto_rebuilder():
    """Quick start guide for auto-rebuilder integration"""
    
    print("🚀 Auto-Rebuilder Integration - Quick Start Guide")
    print("=" * 55)
    print()
    
    print("📋 Step 1: Start the Digital Organism with Auto-Rebuilder")
    print("-" * 55)
    print("   Run the unified launcher:")
    print("   > python unified_digital_organism_launcher.py")
    print()
    print("   The auto-rebuilder will automatically start if available.")
    print()
    
    print("📋 Step 2: Verify Auto-Rebuilder Integration")
    print("-" * 55)
    print("   In the interactive mode, type 'demo' to see auto-rebuilder status:")
    print("   Digital Organism> demo")
    print()
    print("   Look for lines like:")
    print("   ✅ Auto-Rebuilder Service: ACTIVE")
    print("   ✅ System Health Score: 0.85")
    print("   ✅ Code Safety Assessment: low")
    print()
    
    print("📋 Step 3: Monitor System Health")
    print("-" * 55)
    print("   The auto-rebuilder runs health checks every 5 minutes.")
    print("   You can check status anytime with 'status' command:")
    print("   Digital Organism> status")
    print()
    
    print("📋 Step 4: Use Auto-Rebuilder Programmatically")
    print("-" * 55)
    print("   Code example:")
    print('''
   from auto_rebuilder_adapter import get_auto_rebuilder_adapter
   
   # Get the adapter
   adapter = get_auto_rebuilder_adapter()
   
   # Check system health
   status = adapter.get_status()
   print(f"Health: {status['health_score']:.2f}")
   
   # Assess code safety
   test_code = "def my_function(): return 'safe'"
   safety = adapter.assess_code_safety(test_code)
   print(f"Safety: {safety['risk_level']}")
   ''')
    print()
    
    print("📋 Step 5: Integration with Your Code")
    print("-" * 55)
    print("   To integrate auto-rebuilder into your own projects:")
    print('''
   # Import the integration
   from auto_rebuilder_adapter import integrate_with_digital_organism
   
   # Start the service
   integration = integrate_with_digital_organism()
   
   # Check status
   if integration['status'] == 'active':
       print("Auto-rebuilder ready!")
       adapter = integration['adapter']
       
       # Use the capabilities
       health = adapter.get_status()['health_score']
       print(f"System health: {health:.2f}")
   ''')
    print()
    
    print("🎯 Key Benefits You Get:")
    print("-" * 55)
    print("   ✅ Continuous health monitoring every 5 minutes")
    print("   ✅ Automatic self-improvement when efficiency drops")
    print("   ✅ Code safety assessment for new integrations")
    print("   ✅ Industrial-grade security framework")
    print("   ✅ Integration with consciousness systems")
    print()
    
    print("⚠️  Important Notes:")
    print("-" * 55)
    print("   • Auto-rebuilder requires auto_rebuilder.py to be available")
    print("   • System runs in background with minimal resource usage")
    print("   • All code changes are tested in sandbox before integration")
    print("   • Health scores below 0.7 trigger self-improvement cycles")
    print()
    
    print("🔧 Troubleshooting:")
    print("-" * 55)
    print("   If auto-rebuilder shows as 'OPTIONAL' in launcher:")
    print("   1. Ensure auto_rebuilder.py is in the same directory")
    print("   2. Check that auto_rebuilder_adapter.py is available")
    print("   3. Run: python auto_rebuilder_adapter.py (for standalone test)")
    print()
    
    print("📊 Advanced Configuration:")
    print("-" * 55)
    print("   You can customize the heartbeat interval:")
    print('''
   from auto_rebuilder_adapter import get_auto_rebuilder_adapter
   
   # Custom configuration (300 = 5 minutes, 60 = 1 minute)
   adapter = get_auto_rebuilder_adapter(heartbeat_interval=60)
   adapter.start()
   ''')
    print()
    
    print("🎉 Ready to Go!")
    print("-" * 55)
    print("   Your Digital Organism now has autonomous self-improvement!")
    print("   The system will continuously monitor and optimize itself.")
    print("   You can focus on development while the auto-rebuilder")
    print("   handles system health and integration security.")
    print()

def demonstrate_auto_rebuilder_capabilities():
    """Demonstrate what auto-rebuilder can do"""
    
    print("\n🧪 Auto-Rebuilder Capabilities Demonstration")
    print("=" * 50)
    
    try:
        from auto_rebuilder_adapter import get_auto_rebuilder_adapter
        
        # Get adapter
        adapter = get_auto_rebuilder_adapter()
        
        print("\n1️⃣ System Health Assessment:")
        print("-" * 30)
        status = adapter.get_status()
        print(f"   Health Score: {status['health_score']:.2f}")
        print(f"   Service Running: {status['running']}")
        print(f"   Heartbeat Interval: {status['heartbeat_interval']}s")
        
        print("\n2️⃣ Code Safety Assessment:")
        print("-" * 30)
        
        # Test safe code
        safe_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
        
        safety_result = adapter.assess_code_safety(safe_code)
        print(f"   Safe Code Risk Level: {safety_result.get('risk_level', 'unknown')}")
        print(f"   Safety Score: {safety_result.get('safety_score', 0.0):.2f}")
        
        # Test potentially risky code
        risky_code = """
import os
def delete_everything():
    os.system('rm -rf /')
"""
        
        risky_result = adapter.assess_code_safety(risky_code)
        print(f"   Risky Code Risk Level: {risky_result.get('risk_level', 'unknown')}")
        print(f"   Safety Score: {risky_result.get('safety_score', 0.0):.2f}")
        print(f"   Warnings: {len(risky_result.get('warnings', []))}")
        
        print("\n3️⃣ Start Heartbeat Service:")
        print("-" * 30)
        if not status['running']:
            adapter.start()
            print("   ✅ Heartbeat service started")
        else:
            print("   ✅ Heartbeat service already running")
            
        print("\n✅ Auto-Rebuilder is fully operational!")
        print("   Your system now has continuous self-improvement capabilities.")
        
    except ImportError:
        print("   ⚠️  Auto-rebuilder adapter not available")
        print("   Make sure auto_rebuilder_adapter.py is in your path")
        
    except Exception as e:
        print(f"   ❌ Error during demonstration: {e}")

if __name__ == "__main__":
    quick_start_auto_rebuilder()
    demonstrate_auto_rebuilder_capabilities()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("=" * 60)
    print("1. Run: python unified_digital_organism_launcher.py")
    print("2. Type 'demo' to see auto-rebuilder in action")
    print("3. Type 'status' to monitor system health")
    print("4. Enjoy autonomous AI evolution! 🚀")
    print()
    print("For full documentation, see:")
    print("- AUTO_REBUILDER_DEPLOYMENT_GUIDE.md")
    print("- AUTO_REBUILDER_INTEGRATION_SUCCESS_REPORT.md")
