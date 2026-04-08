# Script de réparation de l'environnement Python
# Exécuter avec: powershell -ExecutionPolicy Bypass -File repair_env.ps1

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     RÉPARATION ENVIRONNEMENT PYTHON - MODE AUTOMATIQUE    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectPath

Write-Host "`nÉtape 1: Suppression de l'ancien environnement..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "✓ .venv supprimé" -ForegroundColor Green
} else {
    Write-Host "✓ .venv n'existe pas (normal)" -ForegroundColor Green
}

Write-Host "`nÉtape 2: Création d'un nouvel environnement..." -ForegroundColor Yellow
python -m venv .venv
Write-Host "✓ Nouvel environnement créé" -ForegroundColor Green

Write-Host "`nÉtape 3: Activation de l'environnement..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"
Write-Host "✓ Environnement activé" -ForegroundColor Green

Write-Host "`nÉtape 4: Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ pip à jour" -ForegroundColor Green

Write-Host "`nÉtape 5: Installation de PyTorch CPU..." -ForegroundColor Yellow
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "✓ PyTorch installé (CPU)" -ForegroundColor Green

Write-Host "`nÉtape 6: Installation des dépendances RAG..." -ForegroundColor Yellow
python -m pip install chromadb sentence-transformers transformers pillow pandas python-dotenv scikit-learn numpy scipy --quiet
Write-Host "✓ Dépendances installées" -ForegroundColor Green

Write-Host "`nÉtape 7: Vérification de l'installation..." -ForegroundColor Yellow
python -c "import torch; print(f'✓ PyTorch {torch.__version__} OK')"
python -c "import chromadb; print('✓ ChromaDB OK')"
python -c "from sklearn.feature_extraction.text import TfidfVectorizer; print('✓ Scikit-learn OK')"

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║          ENVIRONNEMENT RÉPARÉ AVEC SUCCÈS! ✓              ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`nVous pouvez maintenant:" -ForegroundColor Cyan
Write-Host "1. Lancer Jupyter: jupyter notebook" -ForegroundColor White
Write-Host "2. Relancer vos cellules de notebook" -ForegroundColor White
Write-Host "3. Le systeme RAG devrait fonctionner en mode Retrieval-only" -ForegroundColor White

pause
