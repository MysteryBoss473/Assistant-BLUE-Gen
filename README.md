# 🤖 Système RAG Multi-Modal avec ChromaDB et Phi-4

Système complet de Retrieval-Augmented Generation (RAG) supportant **texte, images et signaux** avec ChromaDB Cloud et Phi-4-multimodal-instruct.

## 🎯 Vision

Un système d'IA générative capable de:
- **Ingérer** des données multi-modales (texte, images, signaux)
- **Vectoriser** avec des embeddings spécialisés par modalité
- **Indexer** dans ChromaDB Cloud (scalable, distribué)
- **Retriever** les contextes les plus pertinents
- **Générer** des réponses augmentées avec Phi-4

## 🏗️ Architecture

```
USER QUERY
    ↓
[Vectoriser avec embedder_text]
    ↓
┌─────────────────────────┐
│ ChromaDB Collections    │
│ ├─ blueGen_texts        │
│ ├─ blueGen_images       │
│ └─ blueGen_signals      │
└─────────────────────────┘
    ↓
[Top-K Retrieval (cosine similarity)]
    ↓
[Augment Context with Retrieved Docs]
    ↓
[Phi-4-multimodal-instruct Generate]
    ↓
FINAL ANSWER
```

## 🔧 Installation et Configuration

### 1. Dépendances
```bash
pip install chromadb sentence-transformers transformers pillow torch torchvision torchaudio accelerate pymupdf pandas python-dotenv scikit-learn numpy scipy
```

### 2. Credentials ChromaDB Cloud
Créer un fichier `.env`:
```
CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=your_api_key_here
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
```

### 3. Exécuter le notebook
```bash
jupyter notebook .ipynb
```

## 📊 Collections ChromaDB

### blueGen_texts
- **Modèle**: intfloat/multilingual-e5-large (1024 dims)
- **Contenu**: Chunks textuels de documents (français/anglais)
- **Métadonnées**: source, filename, page, type

### blueGen_images
- **Modèle**: Google ViT-Large (1024 dims)
- **Contenu**: Images, graphiques, visualisations
- **Métadonnées**: source, size (width×height), type

### blueGen_signals
- **Modèle**: Features statistiques + SentenceTransformer
- **Contenu**: Données temporelles, signaux, séries temporelles
- **Métadonnées**: signal_length, mean, std, top_features

## 🚀 Utilisation

### Recherche simple
```python
results = retrieve_similar_documents(
    "votre requête",
    collections["texts"],
    embedder_text,
    n_results=5
)

for result in results:
    print(f"Score: {result['similarity']:.2%}")
    print(f"Document: {result['document'][:200]}...")
```

### RAG complet (Retrieval + Génération)
```python
response = rag_system.query("Comment vont les précipitations en 2016?")
print(response["answer"])
```

### Ajouter des données

**Textes:**
```python
new_chunks = [
    {
        "text": "Contenu du texte...",
        "metadata": {"source": "source_url", "page": 1}
    }
]
index_text_chunks(new_chunks, collections["texts"], embedder_text)
```

**Images:**
```python
new_images = [
    {
        "image_path": "path/to/image.jpg",
        "description": "Graphique de précipitations",
        "metadata": {"type": "chart"}
    }
]
index_image_chunks(new_images, collections["images"], image_processor, embedder_image)
```

**Signaux:**
```python
import numpy as np
new_signals = [
    {
        "signal_data": np.array([1, 2, 3, ..., 1000]),
        "description": "Série temporelle de température",
        "metadata": {"sampling_rate": 1000}
    }
]
index_signal_chunks(new_signals, collections["signals"], embedder_text)
```

## 📈 Structures de données

### Chunk texte
```python
{
    "chunk_id": "unique_id",
    "text": "Contenu textuel...",
    "metadata": {
        "source": "url",
        "filename": "document.pdf",
        "page": 1,
        "type": "text"
    }
}
```

### Résultat de retrieval
```python
{
    "id": "doc_id",
    "document": "Contenu du document...",
    "metadata": {...},
    "distance": 0.15,        # Cosine distance
    "similarity": 0.85,      # Cosine similarity (1 - distance)
    "modal_type": "text"
}
```

### Réponse RAG
```python
{
    "query": "Votre question",
    "context": "=== CONTEXTES TEXTUELS ===\n[85%] Premier résultat...",
    "answer": "Réponse générée par Phi-4..."
}
```

## 🔑 Objets disponibles après initialisation

| Objet | Type | Description |
|-------|------|-------------|
| `client` | ChromaDB Client | Connection à ChromaDB Cloud |
| `collections` | dict | Collections texte/image/signal |
| `embedder_text` | SentenceTransformer | Vectorisation texte |
| `embedder_image` | AutoModel (ViT) | Vectorisation images |
| `image_processor` | AutoImageProcessor | Prétraitement images |
| `phi4` | AutoModelForCausalLM | LLM génération |
| `tokenizer` | AutoTokenizer | Tokenizer pour Phi-4 |
| `rag_system` | RAGSystem | Pipeline RAG complet |

## ⚙️ Configuration avancée

### Paramètres de génération
```python
response = rag_system.generate_answer(
    query="Question",
    context="Contexte",
    max_length=512,        # Longueur max réponse (tokens)
    temperature=0.7        # Créativité (0=déterministe, 1=créatif)
)
```

### Batch processing
Les tailles de batch sont optimisées par modalité:
- **Textes**: 100 par batch
- **Images**: 20 par batch  
- **Signaux**: 50 par batch

## 🆘 Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| ConnectionError ChromaDB | Authentification | Vérifier `.env` et credentials |
| CUDA out of memory | Batch trop large | Réduire batch_size |
| Model not found | Pas de connexion HF | Vérifier internet, ou télécharger hors-ligne |
| Faibles scores | Requête mal formulée | Reformuler ou ajouter plus de contexte |

## 📚 Ressources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Microsoft Phi-4](https://huggingface.co/microsoft/phi-4)
- [Vision Transformers](https://huggingface.co/google/vit-large-patch16-224-in21k)

## 📝 License

Projet académique - S8 IA Générative

---

**Créé**: 2026 | **Dernière mise à jour**: Avril 2026
