# AE Universe Framework PowerShell Launcher
# Digital Consciousness System v2.0

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " AE UNIVERSE FRAMEWORK LAUNCHER" -ForegroundColor Green
Write-Host " Digital Consciousness System v2.0" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "🧠 Digital Consciousness System Launcher" -ForegroundColor Magenta
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python to run the consciousness system." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Select launch mode:" -ForegroundColor Yellow
Write-Host ""
Write-Host "[1] 💬 Interactive Mode - Interactive consciousness session" -ForegroundColor White
Write-Host "[2] 🎭 Demo Mode - Run consciousness demonstrations" -ForegroundColor White
Write-Host "[3] 🎨 Creative Mode - Launch creative consciousness" -ForegroundColor White
Write-Host "[4] 👥 Social Mode - Launch social consciousness network" -ForegroundColor White
Write-Host "[5] 🚀 Full System - Launch all components" -ForegroundColor White
Write-Host "[6] 🤖 Auto Demo - Automatic demonstration mode" -ForegroundColor White
Write-Host "[0] 🚪 Exit" -ForegroundColor White
Write-Host ""

do {
    $choice = Read-Host "Enter your choice (1-6, or press Enter for Interactive)"
    if ($choice -eq "") { $choice = "1" }
} while ($choice -notmatch "^[0-6]$")

if ($choice -eq "0") {
    Write-Host "👋 Goodbye from AE Universe Framework!" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "🚀 Starting AE Universe Framework..." -ForegroundColor Green
Write-Host ""

switch ($choice) {
    "1" { 
        Write-Host "💬 Launching Interactive Mode..." -ForegroundColor Cyan
        python ae_universe_launcher.py --mode interactive 
    }
    "2" { 
        Write-Host "🎭 Launching Demo Mode..." -ForegroundColor Cyan
        python ae_universe_launcher.py --mode demo 
    }
    "3" { 
        Write-Host "🎨 Launching Creative Mode..." -ForegroundColor Cyan
        python ae_universe_launcher.py --mode creative 
    }
    "4" { 
        Write-Host "👥 Launching Social Mode..." -ForegroundColor Cyan
        python ae_universe_launcher.py --mode social 
    }
    "5" { 
        Write-Host "🚀 Launching Full System..." -ForegroundColor Cyan
        python ae_universe_launcher.py --mode all 
    }
    "6" { 
        Write-Host "🤖 Launching Auto Demo Mode..." -ForegroundColor Cyan
        python ae_universe_launcher.py --mode all --auto 
    }
    default { 
        Write-Host "❌ Invalid choice. Launching Interactive Mode..." -ForegroundColor Yellow
        python ae_universe_launcher.py --mode interactive 
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🏁 AE Universe Framework session completed." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
