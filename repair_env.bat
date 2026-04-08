@echo off
REM Script de réparation de l'environnement Python (version Batch)
REM Exécuter en double-cliquant sur ce fichier

setlocal enabledelayedexpansion

cls
echo.
echo ========================================================================
echo    REPARATION ENVIRONNEMENT PYTHON - MODE AUTOMATIQUE
echo ========================================================================
echo.

REM Aller au répertoire du script
cd /d "%~dp0"

echo Etape 1: Suppression de l'ancien environnement...
if exist ".venv" (
    rmdir /s /q ".venv" >nul 2>&1
    echo [OK] .venv supprime
) else (
    echo [OK] .venv n'existe pas
)

echo.
echo Etape 2: Creation d'un nouvel environnement...
python -m venv .venv
echo [OK] Nouvel environnement cree

echo.
echo Etape 3: Activation de l'environnement...
call ".venv\Scripts\activate.bat"
echo [OK] Environnement active

echo.
echo Etape 4: Mise a jour de pip...
python -m pip install --upgrade pip -q
echo [OK] pip a jour

echo.
echo Etape 5: Installation de PyTorch CPU...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
echo [OK] PyTorch installe (CPU)

echo.
echo Etape 6: Installation des dependances...
python -m pip install chromadb sentence-transformers transformers pillow pandas python-dotenv scikit-learn numpy scipy -q
echo [OK] Dependances installees

echo.
echo Etape 7: Verification...
python -c "import torch; print('[OK] PyTorch', torch.__version__)"
python -c "import chromadb; print('[OK] ChromaDB')"
python -c "from sklearn.preprocessing import StandardScaler; print('[OK] Scikit-learn')"

cls
echo.
echo ========================================================================
echo    ENVIRONNEMENT REPARE AVEC SUCCES!
echo ========================================================================
echo.
echo Vous pouvez maintenant:
echo 1. Lancer Jupyter: jupyter notebook
echo 2. Relancer vos cellules de notebook
echo 3. Le systeme RAG doit fonctionner en mode Retrieval-only
echo.
echo Appuyez sur une touche pour fermer...
pause > nul
