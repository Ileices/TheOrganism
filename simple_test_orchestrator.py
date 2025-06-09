import sys
import os

print("Testing basic functionality...")

# Test 1: Check if file exists
orchestrator_file = "aeos_production_orchestrator.py"
if os.path.exists(orchestrator_file):
    print(f"✅ {orchestrator_file} exists")
else:
    print(f"❌ {orchestrator_file} not found")
    sys.exit(1)

# Test 2: Check file encoding
try:
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✅ File readable as UTF-8, {len(content)} characters")
except Exception as e:
    print(f"❌ File encoding issue: {e}")
    
    # Try with different encoding
    try:
        with open(orchestrator_file, 'r', encoding='latin-1') as f:
            content = f.read()
        print(f"⚠️ File readable as latin-1, {len(content)} characters")
    except Exception as e2:
        print(f"❌ File unreadable: {e2}")
        sys.exit(1)

# Test 3: Try to compile the code
try:
    compile(content, orchestrator_file, 'exec')
    print("✅ File compiles successfully")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)

print("🎉 Basic tests passed!")
