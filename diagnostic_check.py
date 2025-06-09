#!/usr/bin/env python3
"""
Diagnostic Check for Enterprise Visual DNA System
================================================

Simple diagnostic to identify crash points and continue iteration
"""

import sys
import traceback
import os

def test_step_by_step():
    """Test each component step by step"""
    print("🔍 DIAGNOSTIC CHECK - Step by Step Testing")
    print("=" * 50)
    
    # Step 1: Basic imports
    print("\n📦 Step 1: Testing Basic Imports...")
    try:
        import numpy as np
        print("✅ NumPy: OK")
        
        from PIL import Image
        print("✅ PIL: OK")
        
        import json
        print("✅ JSON: OK")
        
    except Exception as e:
        print(f"❌ Basic Import Error: {e}")
        return False
    
    # Step 2: Test VDN Format
    print("\n🗜️ Step 2: Testing VDN Format...")
    try:
        import vdn_format
        vdn = vdn_format.VDNFormat()
        print("✅ VDN Format: Imported and instantiated")
        
        # Simple test
        test_data = b"test"
        compressed = vdn.compress(test_data)
        print(f"✅ VDN Compression: {len(test_data)} -> {len(compressed)} bytes")
        
    except Exception as e:
        print(f"❌ VDN Format Error: {e}")
        traceback.print_exc()
    
    # Step 3: Test Twmrto
    print("\n🧠 Step 3: Testing Twmrto Compression...")
    try:
        import twmrto_compression
        compressor = twmrto_compression.TwmrtoCompressor()
        print("✅ Twmrto: Imported and instantiated")
        
        # Test simple compression
        result = compressor.compress_text("test")
        print(f"✅ Twmrto Compression: 'test' -> '{result}'")
        
    except Exception as e:
        print(f"❌ Twmrto Error: {e}")
        traceback.print_exc()
    
    # Step 4: Test Enterprise System Import
    print("\n🏢 Step 4: Testing Enterprise System Import...")
    try:
        import enterprise_visual_dna_system
        print("✅ Enterprise System: Module imported")
        
        # Try to get the class
        EnterpriseSystem = enterprise_visual_dna_system.EnterpriseVisualDNASystem
        Config = enterprise_visual_dna_system.EnterpriseConfig
        print("✅ Enterprise Classes: Accessible")
        
    except Exception as e:
        print(f"❌ Enterprise Import Error: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Test Enterprise System Creation
    print("\n🚀 Step 5: Testing Enterprise System Creation...")
    try:
        config = Config()
        print("✅ Enterprise Config: Created")
        
        system = EnterpriseSystem(config)
        print("✅ Enterprise System: Instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise Creation Error: {e}")
        traceback.print_exc()
        return False

def run_simple_visualization():
    """Run a simple visualization test"""
    print("\n🎨 Step 6: Testing Simple Visualization...")
    try:
        # Use existing codebase analyzer
        import codebase_relationship_analyzer as cra
        
        # Run simple analysis
        print("✅ Starting simple visualization...")
        result = cra.visualize_project_structure(".", max_files=5)
        print(f"✅ Simple Visualization: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting diagnostic check...")
    
    step1_ok = test_step_by_step()
    
    if step1_ok:
        print("\n🎯 Basic systems OK - Testing visualization...")
        step2_ok = run_simple_visualization()
        
        if step2_ok:
            print("\n✅ ALL SYSTEMS OPERATIONAL")
            print("🚀 Ready to continue enterprise iteration!")
        else:
            print("\n⚠️ Visualization needs attention")
    else:
        print("\n❌ Basic systems need fixing first")
    
    print("\nDiagnostic complete.")
