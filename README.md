# BLUE-Gen 🌊 - Assistant IA Multimodal pour la Gestion de l'Eau

**BLUE-Gen** est un assistant IA avancé et multimodal conçu pour la gestion intelligente et la prévision des ressources en eau. Il intègre trois pipelines d'intelligence artificielle complémentaires :

1. **📊 Signal Forecast** - Prévisions de séries temporelles pour les données hydrologiques
2. **🖼️ Image LoRA** - Génération d'images personnalisées via fine-tuning LoRA
3. **📚 Text RAG** - Recherche augmentée par génération pour répondre aux questions

---

## 🎯 Fonctionnalités Principales

### 1. **Prévision de Signaux (Signal Forecast)**
- 🔮 Prévisions de séries temporelles avec intervalles de confiance
- 📈 Support de multiples méthodes (Prophet, Holt-Winters, linéaire)
- 📊 Visualisation interactive des données historiques et prévisions
- 💧 Traitement de signaux hydrologiques (débits, niveaux, précipitations)
- 📋 Analyse résiduelle et validation statistique

### 2. **Génération d'Images avec Fine-tuning (Image LoRA)**
- 🎨 Fine-tuning LoRA de Stable Diffusion v1.5
- 💾 Checkpoint automatique pendant l'entraînement
- 🔄 Fusion des poids LoRA avec le modèle de base
- ⚡ Support de l'accélération GPU/CPU
- 🎯 Génération d'images thématiques liées à l'eau

### 3. **Recherche Augmentée par Génération (Text RAG)**
- 🔍 Indexation vectorielle avec ChromaDB
- 🌍 Embeddings multilingues via Sentence Transformers
- 🤖 Génération de réponses avec Gemini API
- 🎓 Réranking intelligent avec Cross-Encoders
- 📖 Extraction de documents PDF

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│        Interface Streamlit (app_integrated.py)  │
│            Chat Web UI interactif               │
└────────────────────┬────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Signal  │  │  Image   │  │  Text    │
│Forecast  │  │  LoRA    │  │  RAG     │
│Pipeline  │  │Pipeline  │  │Pipeline  │
└──────────┘  └──────────┘  └──────────┘
     │               │               │
     ▼               ▼               ▼
 Prophet      Diffusers     ChromaDB + Gemini
 Holt-W.      PEFT LoRA     Transformers
 Linear       SafeTensors   LangChain
```

### Flux de Données

1. **Entrée utilisateur** → Chat Streamlit
2. **Routage** → Détection du type de demande (signal/image/texte)
3. **Traitement** → Pipeline approprié
4. **Réponse** → Rendu HTML avec visualisations
5. **Affichage** → Chat formaté avec bulles de message

---

## 📦 Structure du Projet

```
.
├── app_integrated.py              # Application Streamlit principale
├── requirements.txt               # Dépendances Python
├── .env                          # Variables d'environnement
│
├── core/                         # Pipelines d'IA
│   ├── signal_forecast.py       # Prévisions de séries temporelles
│   ├── image_lora.py            # Fine-tuning LoRA pour génération
│   ├── text_rag.py              # Recherche augmentée par génération
│   └── utils.py                 # Fonctions utilitaires
│
├── lancement/                    # Scripts d'initialisation et outils
│   ├── forecast_init.py         # Initialisation et test du pipeline signal
│   ├── lora_init.py             # Initialisation du fine-tuning LoRA
│   ├── rag_init.py              # Initialisation du RAG
│   ├── run_all_init.py          # Exécution de tous les pipelines
│   ├── merge.py                 # Fusion des poids LoRA
│   └── lora_adapter/            # Modèle LoRA entraîné
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── metadata.json
│
├── dataset/                      # Données
│   ├── images/                  # Images d'entraînement + captions
│   ├── signaux/                 # Données de signaux hydrologiques
│   └── texte/                   # Documents textes pour RAG
│
└── .venv/                        # Environnement virtuel Python
```

---

## 🚀 Démarrage Rapide

### 1. **Installation**

```bash
# Cloner le projet
cd c:\Users\PC\Downloads\testing

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### 2. **Configuration**

Créer un fichier `.env` avec les variables suivantes :

```bash
# Google Generative AI (Gemini)
GOOGLE_API_KEY=your_google_api_key_here

# ChromaDB (optionnel - utilise SQLite par défaut)
CHROMA_DB_URL=http://api.trychroma.com

# Chemins des données
SIGNAL_DATA_PATH=dataset/signaux
IMAGE_DATA_PATH=dataset/images
TEXT_DATA_PATH=dataset/texte

# Modèles
DIFFUSERS_MODEL_ID=runwayml/stable-diffusion-v1-5
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large
RAG_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

### 3. **Lancer l'Application**

```bash
# Démarrer le serveur Streamlit (port 8502)
streamlit run app_integrated.py --server.port 8502

# Accéder à l'interface
# http://localhost:8502
```

### 4. **Initialiser les Pipelines** (optionnel)

```bash
# Tester tous les pipelines
python lancement/run_all_init.py

# Ou tester individuellement
python lancement/forecast_init.py   # Test des prévisions
python lancement/lora_init.py       # Fine-tuning LoRA
python lancement/rag_init.py        # Indexation RAG
```

---

## 📊 Pipelines en Détail

### Signal Forecast Pipeline

**Fichier :** `core/signal_forecast.py`

#### Fonctionnalités
- Prévisions avec intervalles de confiance (90%, 95%)
- Plusieurs méthodes de fallback automatique
- Analyse résiduelle pour validation
- Support de données manquantes

#### Méthodes Disponibles
1. **Prophet** (primaire) - Modèle de séries temporelles robuste
2. **Holt-Winters** (fallback) - Lissage exponentiel
3. **Linéaire** (dernier recours) - Régression simple

#### Utilisation

```python
from core.signal_forecast import SignalForecastPipeline

pipeline = SignalForecastPipeline()
result = pipeline.run(signal_key='MY_SIGNAL', historical_data=data)

# Résultat
{
    'status': 'success',
    'signal_key': 'MY_SIGNAL',
    'analysis': {
        'trend': 'increasing',
        'volatility': 0.45,
        'method_used': 'prophet'
    },
    'historical': [
        {'label': '2020', 'value': 125.5},
        ...
    ],
    'forecast': [
        {'label': '2025', 'value': 145.2, 'lower': 138.1, 'upper': 152.3},
        ...
    ]
}
```

### Image LoRA Pipeline

**Fichier :** `core/image_lora.py`

#### Fonctionnalités
- Fine-tuning LoRA efficace en paramètres
- Stable Diffusion v1.5
- Sauvegarde de checkpoints
- Fusion de modèles

#### Configurations
- **Rank LoRA :** 16
- **Alpha LoRA :** 32
- **Learning Rate :** 1e-4
- **Batch Size :** 2
- **Gradient Checkpointing :** Activé

#### Utilisation

```python
from core.image_lora import ImageLoRAPipeline

pipeline = ImageLoRAPipeline()

# Fine-tuning
result = pipeline.train(
    image_dir='dataset/images',
    captions_file='dataset/images/captions.json',
    num_epochs=10,
    output_dir='lancement/lora_adapter'
)

# Génération d'images
image = pipeline.generate(
    prompt='Une rivière cristalline',
    adapter_path='lancement/lora_adapter',
    num_images=1
)

# Fusion des poids
from lancement.merge import merge_lora
merge_lora('lancement/lora_adapter', 'merged_model')
```

### Text RAG Pipeline

**Fichier :** `core/text_rag.py`

#### Fonctionnalités
- Indexation vectorielle avec ChromaDB
- Embeddings multilingues
- Réranking intelligent
- Intégration Gemini pour génération
- Support PDF

#### Configuration Modèles
- **Embedding :** `intfloat/multilingual-e5-large`
- **Réranker :** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **LLM :** Google Gemini API

#### Utilisation

```python
from core.text_rag import TextRAGPipeline

pipeline = TextRAGPipeline()

# Indexer des documents
pipeline.index_documents('dataset/texte')

# Interroger
response = pipeline.query(
    question='Comment gérer les ressources en eau?',
    top_k=3
)

print(response['answer'])        # Réponse générée
print(response['sources'])       # Documents utilisés
```

---

## 🔧 Stack Technologique

### Frontend
- **Streamlit** ≥1.28.0 - Interface web interactive
- **Plotly** ≥5.14.0 - Visualisations interactives

### Deep Learning & IA
- **PyTorch** ≥2.0.0 - Framework de deep learning
- **Diffusers** ≥0.38.0 - Pipelines de génération
- **PEFT** ≥0.19.0 - Parameter-Efficient Fine-Tuning (LoRA)
- **Transformers** ≥4.35.0 - Modèles pré-entraînés

### Séries Temporelles & Statistiques
- **Prophet** ≥1.1.0 - Prévisions de séries temporelles
- **Statsmodels** ≥0.14.0 - Outils statistiques
- **Scikit-learn** ≥1.3.0 - Machine Learning

### Recherche Augmentée
- **ChromaDB** ≥0.4.0 - Base de données vectorielle
- **Sentence-Transformers** ≥2.2.0 - Embeddings multilingues
- **LangChain** ≥0.1.0 - Framework RAG
- **Google Generative AI** ≥0.5.0 - Gemini API

### Données
- **Pandas** ≥2.0.0 - Manipulation de données
- **NumPy** ≥1.24.0 - Calcul numérique
- **OpenPyXL** ≥3.1.0 - Lecture Excel

### Utilitaires
- **SafeTensors** ≥0.4.0 - Sérialisation de modèles
- **Python-dotenv** ≥1.0.0 - Variables d'environnement
- **Tqdm** ≥4.66.0 - Barres de progression

---

## 💻 Utilisation de l'Application

### Interface Utilisateur

#### Chat Principal
1. **Saisir une demande** dans la zone de texte
2. **Sélectionner le mode** (auto-détecte, ou choix manuel)
3. **Visualiser la réponse** avec les éléments multimédias

#### Modes Disponibles
- 🤖 **Auto** - Détection automatique du type de demande
- 📊 **Signal** - Prévisions de séries temporelles
- 🖼️ **Image** - Génération d'images
- 📚 **RAG** - Questions-réponses

#### Exemples de Demandes

**Signal Forecast :**
```
"Prédis le niveau d'eau pour les 5 prochaines années"
"Analyse la tendance des débits de la rivière"
```

**Image Generation :**
```
"Génère une image d'une centrale hydroélectrique"
"Crée une image montrant la gestion de l'eau"
```

**Text RAG :**
```
"Comment fonctionne la gestion intégrée des ressources en eau?"
"Quelles sont les meilleures pratiques pour les irrigations?"
```

---

## 🐛 Troubleshooting

### Problème : Streamlit n'apparaît pas
```bash
# Vérifier les processus
tasklist | findstr streamlit

# Tuer les anciens processus
taskkill /F /IM streamlit.exe

# Relancer
streamlit run app_integrated.py --server.port 8502
```

### Problème : "ModuleNotFoundError"
```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall

# Vérifier les packages
pip list | findstr streamlit torch diffusers
```

### Problème : ChromaDB connection error
```bash
# Vérifier la configuration .env
# Assurez-vous que GOOGLE_API_KEY est définie
# ChromaDB utilise SQLite par défaut (pas de serveur requis)
```

### Problème : GPU non reconnu
```python
# Vérifier PyTorch
import torch
print(torch.cuda.is_available())  # Devrait être False en CPU

# Pour GPU NVIDIA, installer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problème : Prévisions non affichées
- Vérifier les données d'entrée (pas de NaN)
- Vérifier le format des données historiques
- Consulter les logs Streamlit

---

## 📈 Performance & Optimisation

### Benchmarks Référence
- **Signal Forecast** : ~500ms pour 44 signaux
- **Image Generation** : ~5-10s par image (CPU)
- **RAG Query** : ~2-3s pour recherche + génération

### Recommandations d'Optimisation
1. **GPU NVIDIA** : Utiliser torch+CUDA pour 3-5x speedup
2. **Parallélisation** : Utiliser le cache Streamlit avec `@st.cache_resource`
3. **Batch Processing** : Traiter plusieurs signaux ensemble
4. **Quantization** : Réduire la taille des modèles LoRA

---

## 🔐 Sécurité & Configuration

### Clés API
- **Google API** : Garder `GOOGLE_API_KEY` secrète dans `.env`
- **Ne pas commiter** le fichier `.env` au Git

### Données Sensibles
- Données d'eau : Vérifier la conformité des régulations locales
- Logs : ChromaDB stocke localement par défaut

---

## 📝 Workflow Typique

### 1. Préparation des Données
```bash
# Placer les données dans dataset/
├── images/        # Images JPG/PNG + captions.json
├── signaux/       # Fichiers Excel avec données
└── texte/         # Documents PDF pour RAG
```

### 2. Entraînement des Modèles (Optionnel)
```bash
python lancement/lora_init.py    # Fine-tune LoRA
python lancement/rag_init.py     # Index RAG
```

### 3. Lancer l'Application
```bash
streamlit run app_integrated.py --server.port 8502
```

### 4. Utiliser l'Interface
- Chat avec demandes multimodales
- Visualiser prévisions, images, réponses
- Affiner les prompts selon les résultats

---

## 🤝 Contribution & Développement

### Structure du Code
- **Séparation des concerns** : Chaque pipeline indépendant
- **Interfaces cohérentes** : Toutes les pipelines retournent `{status, data}`
- **Caching** : Utiliser `@st.cache_resource` pour les pipelines

### Ajouter un Nouveau Pipeline
1. Créer `core/new_pipeline.py`
2. Implémenter classe avec méthode `run()`
3. Intégrer dans `app_integrated.py`
4. Ajouter les dépendances à `requirements.txt`

### Débogage
```bash
# Logs Streamlit
streamlit run app_integrated.py --logger.level=debug

# Tests individuels des pipelines
python -c "from core.signal_forecast import SignalForecastPipeline; print('OK')"
```

---

## 📚 Ressources

### Documentation Officielle
- [Streamlit Docs](https://docs.streamlit.io/)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [PyTorch PEFT](https://github.com/huggingface/peft)
- [ChromaDB](https://www.trychroma.com/)
- [Prophet Forecasting](https://facebook.github.io/prophet/)

### Modèles Utilisés
- **Stable Diffusion v1.5** : [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Embeddings** : [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- **Reranker** : [mmarco mMiniLMv2](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)

---

## 📄 Licence & Attribution

**BLUE-Gen** © 2026

### Modèles Open Source Utilisés
- Stable Diffusion (CompVis, Runway, LAION)
- Sentence Transformers (Hugging Face)
- Prophet (Facebook Research)
- ChromaDB (Chroma)

---

## 🎯 Roadmap Futur

- [ ] Support multi-GPU avec DistributedDataParallel
- [ ] Dashboard analytics des prévisions
- [ ] Export des résultats (PDF, CSV)
- [ ] API REST pour intégration externe
- [ ] Historique de conversation persistant
- [ ] Fine-tuning multi-utilisateur
- [ ] Support de modèles alternatifs (Llama, Mistral)

---

## 💬 Support

Pour les problèmes ou questions :
1. Consulter la section **Troubleshooting**
2. Vérifier les logs Streamlit
3. Valider les dépendances avec `pip list`
4. Tester les pipelines individuels

---

**Version :** 1.0.0  
**Dernière mise à jour :** Mai 2026  
**Statut :** Production ✅

```
   ╔═══════════════════════════════════╗
   ║     BLUE-Gen Ready to Deploy      ║
   ║   Multimodal AI for Water Mgmt    ║
   ╚═══════════════════════════════════╝
```
