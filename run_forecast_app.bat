@echo off
REM Script de lancement de l'application de forecasting
cls
echo.
echo ========================================================================
echo           SYSTEME DE PREVISION DE SIGNAUX TEMPORELS
echo           Signal Forecasting System with Uncertainty Bands
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou non accessible
    echo Veuillez installer Python 3.8+ depuis https://www.python.org
    pause
    exit /b 1
)

REM Check if requirements are installed
echo [*] Verifying dependencies...
python -c "import streamlit, prophet, pandas, plotly" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing required packages...
    pip install -r requirements.txt
)

REM Start the app
echo [*] Demarrage de l'application...
echo [*] Accedez a: http://localhost:8501
echo.
timeout /t 2 >nul

streamlit run forecast_app.py

pause
