#!/usr/bin/env python3
"""
Enterprise Visual DNA System Status Dashboard
============================================

Real-time status monitoring for the enterprise visualization system.
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

def check_component_status():
    """Check status of all system components"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'components': {},
        'overall_status': 'UNKNOWN'
    }
    
    components = [
        ('VDN Format', 'vdn_format', 'VDNFormat'),
        ('Twmrto Compression', 'twmrto_compression', 'TwmrtoCompressor'),
        ('Enterprise System', 'enterprise_visual_dna_system', 'EnterpriseVisualDNASystem'),
        ('3D Visualization', 'visual_dna_3d_system', 'VisualDNA3DSystem'),
        ('Execution Tracer', 'real_time_execution_tracer', 'RealTimeExecutionTracer'),
        ('Steganographic Security', 'steganographic_png_security', 'SteganographicPNGSecurity')
    ]
    
    working_components = 0
    
    for name, module, class_name in components:
        try:
            module_obj = __import__(module)
            class_obj = getattr(module_obj, class_name)
            status['components'][name] = {
                'status': 'OPERATIONAL',
                'module': module,
                'class': class_name
            }
            working_components += 1
        except ImportError as e:
            status['components'][name] = {
                'status': 'IMPORT_ERROR',
                'error': str(e),
                'module': module
            }
        except AttributeError as e:
            status['components'][name] = {
                'status': 'CLASS_NOT_FOUND',
                'error': str(e),
                'module': module
            }
        except Exception as e:
            status['components'][name] = {
                'status': 'ERROR',
                'error': str(e),
                'module': module
            }
    
    # Determine overall status
    total_components = len(components)
    if working_components == total_components:
        status['overall_status'] = 'FULLY_OPERATIONAL'
    elif working_components >= total_components * 0.8:
        status['overall_status'] = 'MOSTLY_OPERATIONAL'
    elif working_components >= total_components * 0.5:
        status['overall_status'] = 'PARTIALLY_OPERATIONAL'
    else:
        status['overall_status'] = 'CRITICAL_FAILURE'
    
    status['working_components'] = working_components
    status['total_components'] = total_components
    status['operational_percentage'] = (working_components / total_components) * 100
    
    return status

def display_status():
    """Display formatted status"""
    status = check_component_status()
    
    print("=" * 80)
    print("🏢 ENTERPRISE VISUAL DNA SYSTEM - STATUS DASHBOARD")
    print("=" * 80)
    print(f"⏰ Timestamp: {status['timestamp']}")
    print(f"🎯 Overall Status: {status['overall_status']}")
    print(f"📊 Operational: {status['working_components']}/{status['total_components']} ({status['operational_percentage']:.1f}%)")
    print("=" * 80)
    
    for name, details in status['components'].items():
        if details['status'] == 'OPERATIONAL':
            print(f"✅ {name}: {details['status']}")
        else:
            print(f"❌ {name}: {details['status']} - {details.get('error', 'Unknown error')}")
    
    print("=" * 80)
    
    # Provide recommendations
    if status['overall_status'] == 'FULLY_OPERATIONAL':
        print("🚀 SYSTEM READY FOR INTERSTELLAR COMMUNICATION!")
        print("   All components are operational and ready for use.")
    elif status['overall_status'] == 'MOSTLY_OPERATIONAL':
        print("⚡ SYSTEM MOSTLY READY")
        print("   Most components working, minor issues detected.")
    elif status['overall_status'] == 'PARTIALLY_OPERATIONAL':
        print("⚠️ SYSTEM PARTIALLY OPERATIONAL")
        print("   Some components failing, troubleshooting recommended.")
    else:
        print("🔧 CRITICAL SYSTEM FAILURE")
        print("   Multiple components failing, immediate attention required.")
    
    print("=" * 80)
    
    return status

def save_status_report(status):
    """Save status report to file"""
    try:
        with open('enterprise_system_status.json', 'w') as f:
            json.dump(status, f, indent=2)
        print(f"📄 Status report saved to: enterprise_system_status.json")
    except Exception as e:
        print(f"⚠️ Failed to save status report: {e}")

if __name__ == "__main__":
    status = display_status()
    save_status_report(status)
    
    # If running with --monitor flag, keep monitoring
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        print("\n🔄 CONTINUOUS MONITORING MODE")
        print("Press Ctrl+C to stop monitoring...")
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking system status...")
                status = display_status()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
