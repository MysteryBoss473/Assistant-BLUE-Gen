#!/usr/bin/env python3
"""
run_all_init.py - Lance tous les scripts d'initialisation (RAG, LoRA, Forecasting)
"""

import logging
import subprocess
import sys
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_init_script(script_name: str) -> bool:
    """Exécute un script d'initialisation."""
    script_path = Path(__file__).parent / f"{script_name}_init.py"
    
    if not script_path.exists():
        logger.error(f"❌ Script non trouvé: {script_path}")
        return False
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Exécution: {script_name}_init.py")
    logger.info(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Erreur: {e}")
        return False

def main():
    """Fonction principale."""
    logger.info("\n")
    logger.info("╔" + "="*68 + "╗")
    logger.info("║" + " "*10 + "🚀 LANCEMENT COMPLET - TOUS LES PIPELINES 🚀" + " "*14 + "║")
    logger.info("║" + " "*20 + "RAG • LoRA • Forecasting" + " "*21 + "║")
    logger.info("╚" + "="*68 + "╝")
    
    results = {}
    
    # 1. RAG
    logger.info("\n[1/3] Initialisation RAG (Texte)...")
    results["rag"] = run_init_script("rag")
    
    # 2. LoRA
    logger.info("\n[2/3] Initialisation LoRA (Images)...")
    results["lora"] = run_init_script("lora")
    
    # 3. Forecasting
    logger.info("\n[3/3] Initialisation Forecasting (Séries)...")
    results["forecast"] = run_init_script("forecast")
    
    # Résumé
    logger.info("\n" + "="*70)
    logger.info("📊 RÉSUMÉ DU LANCEMENT")
    logger.info("="*70)
    
    for name, success in results.items():
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        logger.info(f"  {name.upper():12} → {status}")
    
    all_success = all(results.values())
    
    logger.info("\n" + "="*70)
    if all_success:
        logger.info("✨ TOUS LES PIPELINES INITIALISÉS AVEC SUCCÈS!")
        logger.info("🎯 Prochaines étapes: Consulter README.md dans chaque dossier")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.info(f"⚠️  {len(failed)} pipeline(s) ont échoué: {', '.join(failed)}")
        logger.info("🔧 Vérifiez les logs ci-dessus pour détails")
    logger.info("="*70 + "\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
