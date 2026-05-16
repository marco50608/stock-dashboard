@echo off
REM ===================================================================
REM   Stock Dashboard - First-time Setup
REM   Creates .venv and installs all dependencies. Run once.
REM ===================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ===================================================================
echo  Stock Dashboard - First-time Setup
echo ===================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [error] Python not found. Install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [step 1/3] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] Failed to create venv
        pause
        exit /b 1
    )
) else (
    echo [step 1/3] .venv already exists, skipping creation
)

echo.
echo [step 2/3] Upgrading pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo.
echo [step 3/3] Installing packages from requirements.txt (this may take 1-3 minutes)...
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [error] Package install failed. See messages above.
    pause
    exit /b 1
)

if not exist ".env" (
    if exist ".env.example" (
        echo.
        echo [info] Copied .env.example to .env. Edit it to add your API keys.
        copy ".env.example" ".env" >nul
    )
)

echo.
echo ===================================================================
echo  Setup complete!
echo ===================================================================
echo.
echo  Next steps:
echo    1. Open .env in Notepad and fill in API keys:
echo       - FRED_API_KEY        (required) https://fredaccount.stlouisfed.org/apikeys
echo       - FINRA_CLIENT_ID     (optional) https://developer.finra.org
echo       - FINRA_CLIENT_SECRET (optional)
echo       - SEC_USER_AGENT      (recommended) use your name and email
echo    2. (Optional) Put AAII sentiment.xls into cache\aaii_sentiment.xls
echo    3. Double-click run.bat to launch the dashboard
echo.
pause
