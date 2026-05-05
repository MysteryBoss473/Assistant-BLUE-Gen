#!/usr/bin/env python3
"""
merge.py - Fusion des poids LoRA avec le modèle de base Stable Diffusion v1.5

Ce script prend un adaptateur LoRA fine-tuné et fusionne ses poids avec
le modèle de base pour créer un modèle final optimisé pour la génération d'images.

Usage:
    python lancement/merge.py                    # Utilise lancement/lora_adapter/ par défaut
    python lancement/merge.py --adapter-dir path # Spécifie un répertoire d'adaptateur
    python lancement/merge.py --output-dir path  # Spécifie le répertoire de sortie
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import argparse

# Ajouter le parent directory au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_lora(adapter_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Fusionne un adaptateur LoRA avec le modèle de base Stable Diffusion v1.5.
    
    Args:
        adapter_dir: Répertoire contenant l'adaptateur LoRA
        output_dir: Répertoire de sortie (optionnel, utilise adapter_dir par défaut)
    
    Returns:
        Résultat de la fusion avec statut et chemin
    """
    adapter_dir = Path(adapter_dir).resolve()  # Résoudre le chemin absolu
    
    if not adapter_dir.exists():
        logger.error(f"✗ Répertoire d'adaptateur non trouvé: {adapter_dir}")
        return {"status": "error", "error": "Adapter directory not found"}
    
    # Vérifier que c'est un adaptateur LoRA valide
    adapter_files = list(adapter_dir.glob("*.bin")) + list(adapter_dir.glob("*.safetensors"))
    config_file = adapter_dir / "adapter_config.json"
    
    if not config_file.exists() or not adapter_files:
        logger.error(f"✗ Aucun adaptateur LoRA valide dans: {adapter_dir}")
        logger.info("   Fichiers attendus: adapter_config.json + adapter_model.safetensors/bin")
        logger.info(f"   Fichiers présents: {[f.name for f in adapter_dir.iterdir()]}")
        return {"status": "error", "error": "Invalid LoRA adapter - missing adapter_config.json"}
    
    output_dir = output_dir or str(adapter_dir)
    
    logger.info("=" * 70)
    logger.info("🔗 FUSION DES POIDS LoRA - STABLE DIFFUSION V1.5")
    logger.info("=" * 70)
    logger.info(f"📁 Adaptateur: {adapter_dir}")
    logger.info(f"📁 Sortie: {output_dir}")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from peft import PeftModel
        
        # Initialiser le modèle de base
        logger.info("\n⏳ Chargement du modèle de base Stable Diffusion v1.5...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        if device == "cpu":
            logger.warning("⚠️ CUDA non disponible - fusion sur CPU (plus lent)")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        logger.info("✓ Modèle de base chargé")
        
        # Charger l'UNet fine-tuné avec LoRA
        logger.info("\n⏳ Chargement de l'UNet fine-tuné avec LoRA...")
        # Important: utiliser le chemin absolu et is_trainable=False pour charger depuis un chemin local
        unet_with_lora = PeftModel.from_pretrained(
            pipeline.unet,
            str(adapter_dir),
            is_trainable=False
        )
        logger.info("✓ UNet avec LoRA chargé")
        
        # Fusionner les poids LoRA avec l'UNet
        logger.info("\n⏳ Fusion des poids LoRA...")
        merged_unet = unet_with_lora.merge_and_unload()
        logger.info("✓ Fusion réussie!")
        
        # Créer le pipeline fusionné
        logger.info("\n⏳ Création du pipeline Stable Diffusion fusionné...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        merged_pipeline = StableDiffusionPipeline(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=merged_unet,
            scheduler=pipeline.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        
        # Sauvegarder le pipeline complet
        logger.info(f"\n⏳ Sauvegarde du pipeline fusionné...")
        merged_pipeline_dir = output_path / "merged_pipeline"
        merged_pipeline.save_pretrained(str(merged_pipeline_dir))
        logger.info(f"✓ Pipeline sauvegardé: {merged_pipeline_dir}")
        
        # Sauvegarder également l'UNet seul
        logger.info(f"⏳ Sauvegarde de l'UNet fusionné...")
        merged_unet_dir = output_path / "merged_model"
        merged_unet.save_pretrained(str(merged_unet_dir))
        logger.info(f"✓ UNet sauvegardé: {merged_unet_dir}")
        
        # Créer les métadonnées
        import json
        metadata = {
            "model": "runwayml/stable-diffusion-v1-5",
            "lora_rank": 8,
            "lora_alpha": 16,
            "is_merged": True,
            "merged_at": __import__('datetime').datetime.now().isoformat()
        }
        
        metadata_path = output_path / "merged_model_metadata.json"
        with open(str(metadata_path), 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Métadonnées sauvegardées: {metadata_path.name}")
        
        logger.info(f"\n📦 Fichiers créés:")
        logger.info(f"   - merged_pipeline/      (Pipeline complet Stable Diffusion)")
        logger.info(f"   - merged_model/         (UNet fusionné)")
        logger.info(f"   - merged_model_metadata.json (Métadonnées)")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ FUSION COMPLÉTÉE AVEC SUCCÈS")
        logger.info("=" * 70)
        logger.info("\nLe modèle fusionné est prêt pour la génération d'images!")
        logger.info("Le frontend peut maintenant charger et utiliser ce modèle.")
        
        return {
            "status": "success",
            "path": str(output_path),
            "merged_pipeline_dir": str(output_path / "merged_pipeline"),
            "merged_model_dir": str(output_path / "merged_model")
        }
        
    except Exception as e:
        logger.error(f"❌ ERREUR LORS DE LA FUSION: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Fusionne les poids LoRA avec Stable Diffusion v1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python lancement/merge.py
  python lancement/merge.py --adapter-dir lancement/lora_adapter
  python lancement/merge.py --output-dir custom/output
        """
    )
    
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="lancement/lora_adapter",
        help="Répertoire contenant l'adaptateur LoRA (défaut: lancement/lora_adapter)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Répertoire de sortie (défaut: même que adapter-dir)"
    )
    
    args = parser.parse_args()
    
    # Convertir les chemins relatifs en chemins absolus
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = Path(__file__).parent.parent / adapter_dir
    
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).parent.parent / output_dir
        output_dir = str(output_dir)
    
    # Vérifier les dépendances
    try:
        import diffusers
        logger.info(f"✓ diffusers {diffusers.__version__} disponible")
    except ImportError:
        logger.error("✗ diffusers non installé")
        logger.info("Installation: pip install diffusers>=0.25.0 safetensors")
        return 1
    
    try:
        import peft
        logger.info(f"✓ peft {peft.__version__} disponible")
    except ImportError:
        logger.error("✗ peft non installé")
        logger.info("Installation: pip install peft")
        return 1
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CUDA non disponible - fusion sur CPU (plus lent)")
    except:
        pass
    
    # Exécuter la fusion
    result = merge_lora(str(adapter_dir), output_dir)
    
    if result.get("status") == "success":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
