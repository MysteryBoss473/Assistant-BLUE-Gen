#!/usr/bin/env python3
"""
rag_init.py - Lancement du pipeline RAG (Retrieval-Augmented Generation)
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

# FIX: Protobuf compatibility with ChromaDB
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

# Ajouter le parent directory au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.text_rag import TextRAGPipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_rag_pipeline() -> Dict[str, Any]:
    """Exécute l'ingestion et le test de génération RAG."""
    logger.info("🚀 EXÉCUTION DU PIPELINE RAG")

    results = {
        "ingestion": False,
        "sample_query": False
    }

    text_dir = Path(__file__).parent.parent / "dataset" / "texte"
    if text_dir.exists():
        logger.info(f"⏳ Ingestion des PDFs depuis: {text_dir}")
        rag = TextRAGPipeline()
        ingestion_results = rag.ingest_directory(str(text_dir))
        success_count = sum(1 for r in ingestion_results if r.get("status") == "success")
        skipped_count = sum(1 for r in ingestion_results if r.get("status") == "skipped")
        error_count   = sum(1 for r in ingestion_results if r.get("status") == "error")
        logger.info(
            f"✓ Ingestion terminée: {success_count} indexés | "
            f"{skipped_count} déjà présents | {error_count} erreurs "
            f"(sur {len(ingestion_results)} fichiers)"
        )

        # La collection est prête si des fichiers ont été indexés maintenant
        # OU s'ils l'avaient déjà été lors d'une session précédente
        results["ingestion"] = rag.is_collection_ready()

        if results["ingestion"]:
            try:
                sample_query = "Quels sont les principaux enjeux de la gestion des ressources hydriques ?"
                logger.info(f"⏳ Génération d'une réponse d'exemple pour la question: {sample_query}")
                response = rag.generate_response(sample_query, use_docs=True)
                logger.info("✓ Réponse générée")
                logger.info(f"   → {response['response'][:300].replace(chr(10), ' ')}...")
                logger.info(f"   Sources: {response.get('sources', [])}")
                results["sample_query"] = True
            except Exception as e:
                logger.warning(f"⚠️  Impossible de générer une réponse d'exemple: {e}")
    else:
        logger.warning("⚠️  Répertoire dataset/texte/ manquant. Aucune action de pipeline effectuée.")

    return results

def main() -> int:
    """Fonction principale."""
    logger.info("🌊 LANCEMENT DU PIPELINE RAG")

    try:
        results = run_rag_pipeline()

        logger.info("✅ RÉSUMÉ DU PIPELINE RAG")
        logger.info(f"   - Ingestion: {'✅' if results['ingestion'] else '❌'}")
        logger.info(f"   - Test requête: {'✅' if results['sample_query'] else '⚠️'}")

        return 0

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DU LANCEMENT: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())