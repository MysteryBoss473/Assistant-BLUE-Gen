# Script de reparation de l'environnement Python
# Executer avec: powershell -ExecutionPolicy Bypass -File repair_env_fixed.ps1

Write-Host "===== REPARATION ENVIRONNEMENT PYTHON =====" -ForegroundColor Cyan

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

Write-Host "`nEtape 1: Suppression de l'ancien environnement..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "OK: .venv supprime" -ForegroundColor Green
} else {
    Write-Host "OK: .venv n'existe pas" -ForegroundColor Green
}

Write-Host "`nEtape 2: Creation d'un nouvel environnement..." -ForegroundColor Yellow
python -m venv .venv
Write-Host "OK: Nouvel environnement cree" -ForegroundColor Green

Write-Host "`nEtape 3: Activation de l'environnement..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"
Write-Host "OK: Environnement active" -ForegroundColor Green

Write-Host "`nEtape 4: Mise a jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "OK: pip a jour" -ForegroundColor Green

Write-Host "`nEtape 5: Installation de PyTorch CPU..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "OK: PyTorch installe (CPU)" -ForegroundColor Green

Write-Host "`nEtape 6: Installation des dependances RAG..." -ForegroundColor Yellow
python -m pip install chromadb sentence-transformers transformers pillow pandas python-dotenv scikit-learn numpy scipy --quiet
Write-Host "OK: Dependances installees" -ForegroundColor Green

Write-Host "`nEtape 7: Verification de l'installation..." -ForegroundColor Yellow
python -c "import torch; print(f'OK: PyTorch {torch.__version__}')"
python -c "import chromadb; print('OK: ChromaDB OK')"
python -c "from sklearn.feature_extraction.text import TfidfVectorizer; print('OK: Scikit-learn')"

Write-Host "`n===== ENVIRONNEMENT REPARE AVEC SUCCES =====" -ForegroundColor Green
Write-Host "Vous pouvez maintenant relancer Jupyter!" -ForegroundColor Cyan

pause
