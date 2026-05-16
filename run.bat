@echo off
REM ===================================================================
REM   Stock Dashboard - Launcher
REM   Double-click to start Streamlit and open the dashboard
REM ===================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    echo [info] Using local .venv
) else (
    set "PYTHON=python"
    echo [info] Using system python. Recommend running setup.bat first.
)

echo.
echo [info] Starting Streamlit... browser will open http://localhost:8501
echo [info] Press Ctrl+C in this window to stop.
echo.

%PYTHON% -m streamlit run app.py
pause
