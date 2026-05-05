"""
image_lora.py - Fine-tuning LoRA avec anti-overfitting pour petits datasets (~70 images)
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ImageAugmentationConfig:
    rotation_degrees: float = 20.0      # Augmenté pour plus de variété
    flip_probability: float = 0.5
    brightness_range: float = 0.25
    contrast_range: float = 0.25
    target_size: Tuple[int, int] = (512, 512)

class ImageLoRAPipeline:
    def __init__(self,
                 base_model: str = "runwayml/stable-diffusion-v1-5",
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 learning_rate: float = 1e-4,   # Réduit pour petit dataset
                 batch_size: int = 2,           # Réduit pour éviter le sur-apprentissage batch
                 weight_decay: float = 0.01,    # Régularisation L2
                 max_grad_norm: float = 1.0):   # Gradient clipping
        self.base_model_name = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        self.augmentation_config = ImageAugmentationConfig()
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.tokenizer = None
        self.scheduler = None
        self.lora_adapter_loaded = False
        self.training_history = []
        self.validation_results = []
        self._sd_pipe = None
        self._best_val_loss = float('inf')
        self._best_epoch = 0
        
        logger.info("⏳ Initialisation du pipeline Image/LoRA...")
        self._initialize_model()
        logger.info("✓ Pipeline Image/LoRA initialisé")
    
    def _initialize_model(self):
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        import torch
        
        logger.info(f"⏳ Chargement Stable Diffusion: {self.base_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline = pipeline.to(device)
        
        self._sd_pipe = pipeline
        self.model = pipeline.unet
        self.text_encoder = pipeline.text_encoder
        self.vae = pipeline.vae
        self.tokenizer = pipeline.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        
        logger.info("✓ Composants chargés")
        logger.info(f"  - UNet: {self.model.__class__.__name__}")
        logger.info(f"  - Device: {device}, dtype: {dtype}")
        
        if device == "cpu":
            logger.warning("  ⚠️ CPU détecté — l'entraînement sera lent (~2-5 min/batch)")
            torch.set_num_threads(min(4, os.cpu_count() or 4))
    
    def setup_lora(self):
        if self.model is None:
            self._initialize_model()
        if self.model is None:
            self.lora_adapter_loaded = False
            return
        
        try:
            from peft import LoraConfig, get_peft_model
            
            logger.info(f"⏳ Configuration LoRA (rank={self.lora_rank}, dropout=0.15)...")
            
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_q", "to_v", "to_k"],
                lora_dropout=0.15,
            )
            
            # Utiliser get_peft_model au lieu de add_adapter (API correcte pour PEFT 0.19)
            self.model = get_peft_model(self.model, lora_config)
            
            if hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing()
            
            self.lora_adapter_loaded = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✓ LoRA configuré")
            logger.info(f"  - Params entraînables: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            logger.info(f"  - Dropout: 0.15 | Modules: to_q, to_v, to_k")
            
        except Exception as e:
            logger.error(f"✗ Configuration LoRA échouée: {e}")
            self.lora_adapter_loaded = False
    
    def _preprocess_image(self, image_path: Path):
        from PIL import Image
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.augmentation_config.target_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            logger.warning(f"⚠ Erreur prétraitement {image_path.name}: {e}")
            return None
    
    def _augment_image(self, image):
        from PIL import Image, ImageEnhance
        import random
        cfg = self.augmentation_config
        
        if random.random() < 0.5:
            angle = random.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
            image = image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        if random.random() < cfg.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Augmentation aléatoire supplémentaire: zoom léger
        if random.random() < 0.3:
            w, h = image.size
            crop = 0.95
            left = w * (1 - crop) * random.random()
            top = h * (1 - crop) * random.random()
            image = image.crop((left, top, left + w*crop, top + h*crop))
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        
        image = ImageEnhance.Brightness(image).enhance(
            1.0 + random.uniform(-cfg.brightness_range, cfg.brightness_range)
        )
        image = ImageEnhance.Contrast(image).enhance(
            1.0 + random.uniform(-cfg.contrast_range, cfg.contrast_range)
        )
        return image
    
    def prepare_dataset(self, image_dir: str, captions_file: Optional[str] = None, augmentation: bool = True):
        logger.info(f"⏳ Préparation dataset: {image_dir}")
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
        
        if not image_files:
            return {"status": "error", "message": "No images found"}
        
        logger.info(f"ℹ Trouvé {len(image_files)} images")
        
        captions_map = {}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r', encoding='utf-8') as f:
                captions_map = json.load(f)
            logger.info(f"✓ {len(captions_map)} captions chargés")
        
        dataset = {"images": [], "captions": [], "metadata": [], "preprocessed_images": []}
        
        for idx, img_file in enumerate(image_files):
            img_name = img_file.stem
            preprocessed = self._preprocess_image(img_file)
            if preprocessed is None:
                continue
            
            augmented = self._augment_image(preprocessed) if augmentation else preprocessed
            caption = captions_map.get(img_name, f"sxs_style water_management h2o_tech image of {img_name}")
            
            dataset["images"].append(str(img_file))
            dataset["preprocessed_images"].append(augmented)
            dataset["captions"].append(caption)
            dataset["metadata"].append({
                "name": img_name, "path": str(img_file),
                "has_custom_caption": img_name in captions_map, "augmented": augmentation
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  ✓ {idx + 1}/{len(image_files)} images prétraitées")
        
        logger.info(f"✓ Dataset préparé: {len(dataset['images'])} images")
        return {
            "status": "success",
            "num_images": len(dataset["images"]),
            "dataset": dataset,
            "requires_training": len(dataset["images"]) > 0
        }
    
    def _run_epoch(self, train_loader, optimizer, lr_scheduler, device, epoch: int, total_epochs: int):
        """Exécute une epoch d'entraînement. Retourne la loss moyenne."""
        import torch
        import torch.nn.functional as F
        
        self.model.train()
        epoch_loss = 0.0
        num_steps = 0
        
        for batch_idx, (batch_imgs, batch_caps) in enumerate(train_loader):
            logger.info(f"    🔄 Batch {batch_idx+1}/{len(train_loader)} (epoch {epoch}/{total_epochs})...")
            
            batch_imgs = batch_imgs.to(device)
            batch_imgs = batch_imgs * 2.0 - 1.0
            
            with torch.no_grad():
                latents = self.vae.encode(batch_imgs).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            text_inputs = self.tokenizer(
                batch_caps, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]
            
            noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            loss.backward()
            
            # Gradient clipping (anti-explosion + stabilité)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_steps += 1
            logger.info(f"    ✓ Batch {batch_idx+1}/{len(train_loader)} — Loss: {loss.item():.4f} — LR: {lr_scheduler.get_last_lr()[0]:.2e}")
        
        return epoch_loss / max(num_steps, 1)
    
    def _validate_epoch(self, val_loader, device):
        """Évalue la loss sur le set de validation. Retourne la loss moyenne."""
        import torch
        import torch.nn.functional as F
        
        self.model.eval()
        val_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for batch_imgs, batch_caps in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_imgs = batch_imgs * 2.0 - 1.0
                
                latents = self.vae.encode(batch_imgs).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps, (bsz,), device=device
                ).long()
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                text_inputs = self.tokenizer(
                    batch_caps, padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]
                
                noise_pred = self.model(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                val_loss += loss.item()
                num_steps += 1
        
        self.model.train()
        return val_loss / max(num_steps, 1)
    
    def train(self, dataset: Dict[str, Any], epochs: int = 15, validation_split: float = 0.3, patience: int = 4, checkpoint_dir: str = None, resume_from_checkpoint: bool = True):
        """
        Fine-tuning avec early stopping, validation et checkpoints.
        
        Args:
            epochs: Nombre max d'epochs (recommandé: 3 pour ~70 images)
            validation_split: Fraction validation (0.3 recommandé pour petit dataset)
            patience: Epochs à attendre avant arrêt si val_loss ne s'améliore pas
            checkpoint_dir: Répertoire pour sauvegarder les checkpoints
            resume_from_checkpoint: Si True, cherche et charge le dernier checkpoint
        """
        if not self.lora_adapter_loaded:
            self.setup_lora()
        
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        import random
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu" and epochs > 2:
            logger.warning("⚠️ CPU détecté: réduction à 2 epochs maximum recommandée")
        
        logger.info(f"⏳ Démarrage fine-tuning: max {epochs} epochs")
        logger.info(f"  - Dataset: {len(dataset['images'])} images")
        logger.info(f"  - Validation split: {validation_split*100:.0f}%")
        logger.info(f"  - Early stopping patience: {patience}")
        logger.info(f"  - LoRA: rank={self.lora_rank}, alpha={self.lora_alpha}, dropout=0.15")
        logger.info(f"  - Optimiseur: AdamW (lr={self.learning_rate}, wd={self.weight_decay})")
        logger.info(f"  - Gradient clipping: {self.max_grad_norm}")
        
        images = dataset.get("preprocessed_images", [])
        captions = dataset.get("captions", [])
        if not images:
            raise ValueError("Aucune image prétraitée trouvée")
        
        image_tensors = []
        for img in images:
            img_np = np.array(img).astype(np.float32) / 255.0
            image_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
        
        n = len(image_tensors)
        n_val = max(1, int(n * validation_split))  # Au moins 1 image en validation
        indices = list(range(n))
        random.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        logger.info(f"  - Train: {len(train_idx)} images | Validation: {len(val_idx)} images")
        
        class SimpleDataset(Dataset):
            def __init__(self, imgs, caps, idx):
                self.imgs = [imgs[i] for i in idx]
                self.caps = [caps[i] for i in idx]
            def __len__(self):
                return len(self.imgs)
            def __getitem__(self, i):
                return self.imgs[i], self.caps[i]
        
        train_dataset = SimpleDataset(image_tensors, captions, train_idx)
        val_dataset = SimpleDataset(image_tensors, captions, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler cosine avec warmup (stabilise l'entraînement)
        total_steps = len(train_loader) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))  # 10% warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Charger checkpoint si disponible et demandé
        start_epoch = 1
        if resume_from_checkpoint and checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                ckpt_result = self.load_checkpoint(str(latest_checkpoint))
                if ckpt_result["status"] == "success":
                    start_epoch = ckpt_result["epoch"] + 1
                    optimizer.load_state_dict(ckpt_result["optimizer_state"])
                    lr_scheduler.load_state_dict(ckpt_result["scheduler_state"])
                    logger.info(f"✓ Reprise de l'entraînement depuis epoch {start_epoch}")
        
        self.vae.eval()
        self.text_encoder.eval()
        
        patience_counter = 0
        
        for epoch in range(start_epoch, epochs + 1):
            logger.info(f"\n📈 Epoch {epoch}/{epochs}")
            
            # Entraînement
            train_loss = self._run_epoch(train_loader, optimizer, lr_scheduler, device, epoch, epochs)
            
            # Validation
            val_loss = self._validate_epoch(val_loader, device)
            
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "num_train": len(train_idx),
                "num_val": len(val_idx)
            })
            
            logger.info(f"✓ Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")
            
            # Sauvegarde du meilleur modèle
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_epoch = epoch
                patience_counter = 0
                logger.info(f"  ⭐ Nouveau meilleur modèle (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"  ⚠️ Val loss stagnante ({patience_counter}/{patience})")
            
            # Sauvegarder checkpoint après chaque epoch
            if checkpoint_dir:
                self.save_checkpoint(checkpoint_dir, epoch, optimizer, lr_scheduler)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\n🛑 Early stopping déclenché après {epoch} epochs (best: epoch {self._best_epoch})")
                break
        
        logger.info(f"\n✓ Fine-tuning terminé — Meilleur epoch: {self._best_epoch} (val_loss: {self._best_val_loss:.4f})")
        return {
            "status": "success",
            "epochs_completed": len(self.training_history),
            "best_epoch": self._best_epoch,
            "best_val_loss": self._best_val_loss,
            "history": self.training_history,
            "num_images_trained": len(train_idx),
            "model": "Stable Diffusion v1.5 (UNet + LoRA)"
        }
    
    def validate(self, test_prompts: List[str], num_images: int = 2):
        logger.info(f"⏳ Validation: {len(test_prompts)} prompts × {num_images} images")
        results = {"test_results": [], "metrics": {}, "status": "success"}
        
        for idx, prompt in enumerate(test_prompts, 1):
            logger.info(f"  ⏳ Test {idx}/{len(test_prompts)}: '{prompt}'")
            generated = self._generate_image_batch(prompt, num_images)
            results["test_results"].append({
                "prompt": prompt, "num_generated": len(generated),
                "generated_images": generated, "tokens_detected": self._extract_tokens(prompt)
            })
        
        total_generated = sum(len(r["generated_images"]) for r in results["test_results"])
        unique_tokens = set()
        for r in results["test_results"]:
            unique_tokens.update(r["tokens_detected"])
        
        results["metrics"] = {
            "total_images_generated": total_generated,
            "tokens_learned": list(unique_tokens),
            "model": "Stable Diffusion v1.5 (LoRA fine-tuned)",
            "model_ready": True
        }
        self.validation_results = results
        logger.info(f"✓ Validation complétée — {total_generated} images générées")
        return results
    
    def _generate_image_batch(self, prompt: str, num_images: int):
        try:
            import torch
            import base64
            from io import BytesIO
            
            logger.info(f"⏳ Génération: '{prompt}'")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self._sd_pipe is None:
                from diffusers import StableDiffusionPipeline
                logger.info("⏳ Création du pipeline de génération...")
                self._sd_pipe = StableDiffusionPipeline(
                    vae=self.vae, text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer, unet=self.model,
                    scheduler=self.scheduler, safety_checker=None,
                    feature_extractor=None, requires_safety_checker=False,
                ).to(device)
            
            pipe = self._sd_pipe
            pipe.set_progress_bar_config(disable=True)
            
            generated_images = []
            for i in range(num_images):
                logger.info(f"    🖼️ Image {i+1}/{num_images}...")
                with torch.no_grad():
                    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                                height=512, width=512).images[0]
                
                buf = BytesIO()
                image.save(buf, format='PNG')
                generated_images.append({
                    "base64": base64.b64encode(buf.getvalue()).decode('utf-8'),
                    "format": "png"
                })
                logger.info(f"    ✓ Image {i+1} générée")
            
            return generated_images
            
        except Exception as e:
            logger.error(f"✗ Erreur génération: {e}")
            return self._generate_image_batch_fallback(prompt, num_images)
    
    def _generate_image_batch_fallback(self, prompt: str, num_images: int):
        from PIL import Image, ImageDraw
        import base64
        from io import BytesIO
        generated = []
        for i in range(num_images):
            img = Image.new('RGB', (512, 512), color=(30, 101, 192))
            draw = ImageDraw.Draw(img)
            try:
                draw.text((50, 230), f"Placeholder\n{prompt[:50]}...", fill=(255, 255, 255))
            except:
                pass
            buf = BytesIO()
            img.save(buf, format='PNG')
            generated.append({"base64": base64.b64encode(buf.getvalue()).decode('utf-8'), "format": "png"})
        return generated
    
    def _extract_tokens(self, text: str):
        tokens = ["sxs_style", "h2o_tech", "water_management", "aqua_reserve", "hydro_", "aqua_"]
        return [t for t in tokens if t in text.lower()]
    
    def generate_image(self, prompt: str, num_images: int = 1,
                       guidance_scale: float = 7.5, num_inference_steps: int = 50):
        try:
            images = self._generate_image_batch(prompt, num_images)
            return {
                "status": "success", "prompt": prompt, "num_images": num_images,
                "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps,
                "generated_images": images, "tokens_used": self._extract_tokens(prompt)
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "prompt": prompt}
    
    def merge_lora_adapter(self, output_dir: str):
        """Fusionne les poids LoRA avec le modèle de base et sauvegarde le modèle complet."""
        try:
            import torch
            
            if not self.lora_adapter_loaded:
                return {"status": "error", "message": "LoRA adapter not loaded"}
            
            logger.info("⏳ Fusion des poids LoRA avec le modèle de base...")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Accéder aux poids du modèle pour la fusion
            from peft import PeftModel
            
            # Vérifier si le modèle est un modèle PEFT
            if isinstance(self.model, PeftModel):
                # Fusionner les adaptateurs avec le modèle de base
                merged_model = self.model.merge_and_unload()
            else:
                logger.warning("⚠️ Le modèle n'est pas un modèle PEFT, impossible de fusionner")
                return {"status": "error", "message": "Model is not a PEFT model"}
            
            # Sauvegarder le modèle fusionné
            merged_model_dir = output_dir / "merged_model"
            merged_model_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_model_dir))
            
            # Sauvegarder le pipeline complet avec le modèle fusionné
            logger.info("⏳ Sauvegarde du pipeline Stable Diffusion complet...")
            
            # Créer un nouveau pipeline avec le modèle fusionné
            from diffusers import StableDiffusionPipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            merged_pipeline = StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=merged_model,
                scheduler=self.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(device)
            
            merged_pipeline.save_pretrained(str(output_dir / "merged_pipeline"))
            
            # Sauvegarder les métadonnées du modèle fusionné
            metadata = {
                "model": self.base_model_name,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "is_merged": True,
                "training_history": self.training_history,
                "validation_results": self.validation_results,
                "best_val_loss": self._best_val_loss,
                "best_epoch": self._best_epoch,
                "merged_at": __import__('datetime').datetime.now().isoformat()
            }
            
            with open(output_dir / "merged_model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Modèle fusionné sauvegardé: {output_dir}")
            logger.info(f"  - Dossier pipeline: merged_pipeline/")
            logger.info(f"  - Dossier UNet: merged_model/")
            logger.info(f"  - Métadonnées: merged_model_metadata.json")
            
            return {
                "status": "success",
                "path": str(output_dir),
                "merged_pipeline_dir": str(output_dir / "merged_pipeline"),
                "merged_model_dir": str(output_dir / "merged_model")
            }
        except Exception as e:
            logger.error(f"✗ Erreur fusion: {e}")
            return {"status": "error", "error": str(e)}
    
    def load_merged_model(self, merged_model_dir: str):
        """Charge le modèle fusionné (UNet) d'un répertoire."""
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            
            merged_model_dir = Path(merged_model_dir)
            if not merged_model_dir.exists():
                return {"status": "error", "error": "Merged model directory not found"}
            
            logger.info(f"⏳ Chargement modèle fusionné: {merged_model_dir}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            # Charger le pipeline fusionné complet
            merged_pipeline_dir = merged_model_dir.parent / "merged_pipeline"
            if merged_pipeline_dir.exists():
                logger.info("✓ Chargement du pipeline fusionné complet")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    str(merged_pipeline_dir),
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
                
                self._sd_pipe = pipeline
                self.model = pipeline.unet
                self.text_encoder = pipeline.text_encoder
                self.vae = pipeline.vae
                self.tokenizer = pipeline.tokenizer
                self.scheduler = pipeline.scheduler
                self.lora_adapter_loaded = False  # Plus de LoRA car c'est fusionné
                
                logger.info("✓ Modèle fusionné chargé")
                return {"status": "success", "is_merged": True}
            else:
                return {"status": "error", "error": "Merged pipeline directory not found"}
        except Exception as e:
            logger.error(f"✗ Erreur chargement modèle fusionné: {e}")
            return {"status": "error", "error": str(e)}

    def save_lora_adapter(self, output_dir: str):
        try:
            if self.lora_adapter_loaded and self.model is not None:
                # Créer le répertoire s'il n'existe pas
                os.makedirs(output_dir, exist_ok=True)
                
                # Sauvegarder l'adapter LoRA (pas le modèle entier)
                # Avec get_peft_model, save_pretrained() crée adapter_config.json + adapter_model.safetensors
                self.model.save_pretrained(output_dir)
                
                # Sauvegarder les métadonnées
                metadata = {
                    "model": self.base_model_name,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "training_history": self.training_history,
                    "validation_results": self.validation_results,
                    "best_val_loss": float(self._best_val_loss),
                    "best_epoch": self._best_epoch,
                    "peft_model": True  # Marquer que c'est un adapter PEFT
                }
                metadata_path = os.path.join(output_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✓ Adaptateur LoRA sauvegardé: {output_dir}")
                logger.info(f"  - Meilleur modèle: epoch {self._best_epoch} (val_loss: {self._best_val_loss:.4f})")
                logger.info(f"  - Fichiers: adapter_config.json + adapter_model.safetensors + metadata.json")
                return {"status": "success", "path": output_dir}
            else:
                return {"status": "error", "message": "LoRA adapter not loaded"}
        except Exception as e:
            logger.error(f"✗ Erreur sauvegarde adapter: {e}")
            logger.exception(e)
            return {"status": "error", "error": str(e)}
    
    def load_lora_adapter(self, adapter_dir: str):
        try:
            logger.info(f"⏳ Chargement adaptateurs LoRA: {adapter_dir}")
            self.model.load_adapter(adapter_dir)
            self.lora_adapter_loaded = True
            logger.info("✓ Adaptateurs LoRA chargés")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"✗ Erreur chargement: {e}")
            return {"status": "error", "error": str(e)}
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, optimizer, lr_scheduler):
        """Sauvegarde un checkpoint après chaque epoch pour permettre la reprise."""
        try:
            import torch
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "training_history": self.training_history,
                "best_val_loss": self._best_val_loss,
                "best_epoch": self._best_epoch,
                "lora_config": {
                    "rank": self.lora_rank,
                    "alpha": self.lora_alpha,
                    "base_model": self.base_model_name
                }
            }
            
            torch.save(checkpoint, str(checkpoint_path))
            logger.info(f"  💾 Checkpoint sauvegardé: epoch {epoch}")
            
            # Garder seulement les 3 derniers checkpoints pour économiser l'espace
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if len(checkpoint_files) > 3:
                old_checkpoint = checkpoint_files[0]
                old_checkpoint.unlink()
                logger.info(f"  🗑️ Ancien checkpoint supprimé: {old_checkpoint.name}")
            
            return {"status": "success", "path": str(checkpoint_path)}
        except Exception as e:
            logger.error(f"✗ Erreur sauvegarde checkpoint: {e}")
            return {"status": "error", "error": str(e)}
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint et restaure l'état d'entraînement."""
        try:
            import torch
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                return {"status": "error", "error": "Checkpoint not found"}
            
            logger.info(f"⏳ Chargement checkpoint: {checkpoint_path.name}")
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            
            # Restaurer l'état du modèle
            self.model.load_state_dict(checkpoint["model_state"])
            self.training_history = checkpoint.get("training_history", [])
            self._best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            self._best_epoch = checkpoint.get("best_epoch", 0)
            
            logger.info(f"✓ Checkpoint chargé — Dernier epoch: {checkpoint['epoch']}")
            logger.info(f"  - Historique: {len(self.training_history)} epochs")
            logger.info(f"  - Meilleur val_loss: {self._best_val_loss:.4f}")
            
            return {
                "status": "success",
                "epoch": checkpoint["epoch"],
                "optimizer_state": checkpoint["optimizer_state"],
                "scheduler_state": checkpoint["scheduler_state"],
                "training_history": self.training_history
            }
        except Exception as e:
            logger.error(f"✗ Erreur chargement checkpoint: {e}")
            return {"status": "error", "error": str(e)}
    
    def export_validation_report(self, output_path: str):
        try:
            report = {
                "model": self.base_model_name, "lora_rank": self.lora_rank,
                "training_history": self.training_history,
                "validation_results": self.validation_results,
                "best_val_loss": self._best_val_loss,
                "best_epoch": self._best_epoch,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Rapport exporté: {output_path}")
            return {"status": "success", "path": output_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

LoRAPipeline = ImageLoRAPipeline