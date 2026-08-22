@echo off
title Clinical Decision Support RAG Platform (100/100 Benchmark)
echo =========================================================================
echo  CLINICAL DECISION SUPPORT RAG PLATFORM -- 100/100 HACKATHON SYSTEM
echo =========================================================================
echo.

:: Step 1: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

:: Step 2: Install dependencies
echo [1/3] Verifying Python dependencies...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check

:: Step 3: Launch FastAPI server in background / daemon
echo [2/3] Starting Clinical RAG backend server on port 8000...
start "" http://127.0.0.1:8000/

:: Step 4: Run uvicorn
echo [3/3] Uvicorn server active. Press Ctrl+C to stop.
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
