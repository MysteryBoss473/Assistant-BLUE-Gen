#!/usr/bin/env python3
"""
lora_init.py - Lancement du pipeline LoRA avec Stable Diffusion v1.5

Pipeline:
1. Charger et prétraiter les images du dataset (redimensionner à 512×512)
2. Appliquer l'augmentation des données (rotation, flip, brightness, contrast)
3. Fine-tuner Stable Diffusion UNet avec LoRA
4. Valider le modèle fine-tuné avec prompts de test
5. Sauvegarder l'adaptateur LoRA

Configuration par défaut:
- Modèle: runwayml/stable-diffusion-v1-5
- LoRA Rank: 8
- LoRA Alpha: 16
- Learning Rate: 1e-4
- Epochs: 15
- Batch Size: 4
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import os

# Ajouter le parent directory au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image_lora import ImageLoRAPipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_lora_pipeline(skip_validation: bool = False) -> Dict[str, Any]:
    """
    Prépare, entraîne et sauvegarde le modèle LoRA avec Stable Diffusion.
    
    Args:
        skip_validation: Si True, skip la validation et les générations de test
    """
    logger.info("🚀 LANCEMENT DU PIPELINE LoRA STABLE DIFFUSION V1.5")

    results = {
        "dataset_prepared": False,
        "training_completed": False,
        "validation_completed": False,
        "adapter_saved": False,
        "model": "Stable Diffusion v1.5 (UNet + LoRA)"
    }

    # Chemins
    image_dir = Path(__file__).parent.parent / "dataset" / "images"
    adapter_output_dir = Path(__file__).parent / "lora_adapter"
    
    logger.info(f"📁 Répertoires:")
    logger.info(f"   - Images: {image_dir}")
    logger.info(f"   - Adaptateur sortie: {adapter_output_dir}")
    
    if image_dir.exists():
        captions_file = image_dir / "captions.json"
        captions_path = str(captions_file) if captions_file.exists() else None
        
        # Initialiser le pipeline LoRA avec Stable Diffusion
        logger.info("⏳ Initialisation pipeline Stable Diffusion...")
        try:
            lora = ImageLoRAPipeline(
                base_model="runwayml/stable-diffusion-v1-5",  # Stable Diffusion v1.5
                lora_rank=8,
                lora_alpha=16,
                learning_rate=1e-4,
                batch_size=2
            )
            logger.info("✓ Pipeline initialisé avec Stable Diffusion v1.5")
        except Exception as e:
            logger.error(f"✗ Impossible d'initialiser le pipeline: {e}")
            logger.info("💡 Assurez-vous que diffusers est installé: pip install diffusers>=0.25.0")
            return results
        
        # Étape 1: Préparation du dataset avec preprocessing et augmentation
        logger.info("\n📊 ÉTAPE 1: Préparation du dataset")
        logger.info("   - Prétraitement: Redimensionnement à 512×512")
        logger.info("   - Augmentation: Rotation ±15°, Flip 50%, Brightness/Contrast ±0.2")
        
        prep = lora.prepare_dataset(str(image_dir), captions_file=captions_path, augmentation=True)
        results["dataset_prepared"] = prep.get("status") == "success"
        
        if results["dataset_prepared"]:
            logger.info(f"✓ Dataset préparé: {prep['num_images']} images prétraitées et augmentées")
        else:
            logger.error("✗ Erreur préparation dataset")
            return results
        
        # Étape 2: Fine-tuning
        if results["dataset_prepared"] and prep.get("requires_training"):
            logger.info("\n🎯 ÉTAPE 2: Fine-tuning Stable Diffusion UNet")
            
            # Dossier pour les checkpoints
            checkpoint_dir = Path(__file__).parent / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Vérifier s'il existe un checkpoint
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoint_files:
                logger.info(f"✓ Checkpoint trouvé - reprise depuis: {checkpoint_files[-1].name}")
            
            train_res = lora.train(prep["dataset"], epochs=15, checkpoint_dir=str(checkpoint_dir))
            results["training_completed"] = train_res.get("status") == "success"
            
            if results["training_completed"]:
                logger.info(f"✓ Fine-tuning complété")
                if "history" in train_res:
                    for entry in train_res["history"]:
                        logger.info(f"   - Epoch {entry['epoch']}: Train Loss={entry['train_loss']:.4f}, Val Loss={entry['val_loss']:.4f}")
            else:
                logger.error(f"✗ Erreur fine-tuning: {train_res.get('error', 'Unknownfine-tuning')}")
                return results
            
            # Étape 3: Validation (optionnel)
            if results["training_completed"] and not skip_validation:
                logger.info("\n✅ ÉTAPE 3: Validation du modèle fine-tuné")
                validation_prompts = [
                    "water_management reservoir imagery sxs_style",
                    "hydro_ technical overview water distribution h2o_tech"
                ]
                val_res = lora.validate(validation_prompts, num_images=2)
                results["validation_completed"] = bool(val_res.get("test_results"))
                
                if results["validation_completed"]:
                    logger.info(f"✓ Validation complétée")
                    if "metrics" in val_res:
                        logger.info(f"   - Modèle: {val_res['metrics'].get('model', 'Unknown')}")
                        logger.info(f"   - Images générées: {val_res['metrics'].get('total_images_generated', 0)}")
            elif skip_validation:
                logger.info("\n⏭️  ÉTAPE 3: Validation skippée (--skip-validation)")
            
            # Étape 4: Sauvegarde de l'adaptateur
            logger.info("\n💾 ÉTAPE 4: Sauvegarde de l'adaptateur LoRA")
            if hasattr(lora, 'lora_adapter_loaded') and lora.lora_adapter_loaded:
                adapter_output_dir.mkdir(parents=True, exist_ok=True)
                save_res = lora.save_lora_adapter(str(adapter_output_dir))
                results["adapter_saved"] = save_res.get("status") == "success"
                
                if results["adapter_saved"]:
                    logger.info(f"✓ Adaptateur LoRA sauvegardé: {adapter_output_dir}")
                    
                    # Liste les fichiers
                    adapter_files = list(adapter_output_dir.glob("*"))
                    for f in adapter_files[:5]:  # Montrer les 5 premiers fichiers
                        logger.info(f"   - {f.name}")
                    if len(adapter_files) > 5:
                        logger.info(f"   - ... et {len(adapter_files) - 5} autres fichiers")
                else:
                    logger.error(f"✗ Erreur sauvegarde: {save_res.get('error', 'Unknown error')}")
            else:
                logger.warning("⚠️ LoRA non configuré - sauvegarde de l'adaptateur ignorée")
                results["adapter_saved"] = False
        else:
            logger.warning("⚠️ Aucune image disponible pour le fine-tuning. Vérifiez dataset/images/")
    else:
        logger.warning(f"⚠️ Répertoire dataset/images/ manquant: {image_dir}")

    return results

def main():
    """Fonction principale."""
    logger.info("=" * 70)
    logger.info("🖼️  PIPELINE LoRA - FINE-TUNING STABLE DIFFUSION V1.5")
    logger.info("=" * 70)
    
    # Vérifier les dépendances
    try:
        import diffusers
        logger.info(f"✓ diffusers {diffusers.__version__} disponible")
    except ImportError:
        logger.error("✗ diffusers non installé")
        logger.info("Installation: pip install diffusers>=0.25.0 safetensors accelerate")
        return 1
    
    # Vérifier CUDA
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CUDA non disponible - utilisation CPU (plus lent)")
    except:
        pass
    
    try:
        # Déterminer si on skip la validation basée sur variable d'env
        skip_validation = os.getenv("SKIP_VALIDATION", "false").lower() == "true"
        
        results = run_lora_pipeline(skip_validation=skip_validation)

        logger.info("\n" + "=" * 70)
        logger.info("📋 RÉSUMÉ DU PIPELINE LoRA")
        logger.info("=" * 70)
        logger.info(f"   - Dataset préparé (preprocessing + augmentation): {'✅' if results['dataset_prepared'] else '❌'}")
        logger.info(f"   - Fine-tuning Stable Diffusion UNet: {'✅' if results['training_completed'] else '❌'}")
        logger.info(f"   - Validation du modèle: {'✅' if results['validation_completed'] else '⚠️'}")
        logger.info(f"   - Adaptateur LoRA sauvegardé: {'✅' if results['adapter_saved'] else '⚠️'}")
        logger.info(f"   - Modèle utilisé: {results['model']}")
        
        if all([results['dataset_prepared'], results['training_completed'], results['adapter_saved']]):
            logger.info("\n✅ PIPELINE COMPLÉTÉ AVEC SUCCÈS")
            logger.info("   Le modèle Stable Diffusion fine-tuné est prêt pour la génération!")
            return 0
        else:
            logger.warning("\n⚠️ PIPELINE COMPLÉTÉ AVEC AVERTISSEMENTS")
            return 0

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DU LANCEMENT: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())