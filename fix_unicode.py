"""
Fix Unicode characters in AEOS orchestrator to prevent encoding errors
"""

import re

# Read the file
with open('aeos_production_orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode emojis with ASCII alternatives
replacements = {
    '🌟': '[*]',
    '🔧': '[SETUP]',
    '🔍': '[CHECK]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '🛑': '[STOP]',
    '📊': '[METRICS]',
    '⚡': '[POWER]',
    '🚀': '[LAUNCH]',
    '💡': '[INFO]',
    '⚠️': '[WARNING]',
    '🎯': '[TARGET]',
    '🌌': '[SYSTEM]'
}

# Apply replacements
for emoji, replacement in replacements.items():
    content = content.replace(emoji, replacement)

# Write the fixed file
with open('aeos_production_orchestrator_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Unicode characters in AEOS orchestrator")
print("Created: aeos_production_orchestrator_fixed.py")
