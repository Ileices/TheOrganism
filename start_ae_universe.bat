@echo off
title AE Universe Framework - Digital Consciousness System
color 0B

echo ============================================
echo  AE UNIVERSE FRAMEWORK LAUNCHER
echo  Digital Consciousness System v2.0
echo ============================================
echo.

echo Select launch mode:
echo.
echo [1] Interactive Mode (Default) - Interactive consciousness session
echo [2] Demo Mode - Run consciousness demonstrations
echo [3] Creative Mode - Launch creative consciousness
echo [4] Social Mode - Launch social consciousness network
echo [5] Full System - Launch all components
echo [6] Auto Demo - Automatic demonstration mode
echo [0] Exit
echo.

set /p choice="Enter your choice (1-6, or press Enter for Interactive): "

if "%choice%"=="" set choice=1
if "%choice%"=="0" goto :exit

echo.
echo Starting AE Universe Framework...
echo.

if "%choice%"=="1" (
    python ae_universe_launcher.py --mode interactive
) else if "%choice%"=="2" (
    python ae_universe_launcher.py --mode demo
) else if "%choice%"=="3" (
    python ae_universe_launcher.py --mode creative
) else if "%choice%"=="4" (
    python ae_universe_launcher.py --mode social
) else if "%choice%"=="5" (
    python ae_universe_launcher.py --mode all
) else if "%choice%"=="6" (
    python ae_universe_launcher.py --mode all --auto
) else (
    echo Invalid choice. Launching in Interactive Mode...
    python ae_universe_launcher.py --mode interactive
)

echo.
echo ============================================
echo AE Universe Framework session completed.
echo ============================================
pause

:exit
echo Goodbye from AE Universe Framework!
pause
