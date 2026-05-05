"""
text_rag.py - Module RAG (Retrieval Augmented Generation) pour l'analyse de documents
"""
import os

# ── FIX protobuf conflict with ChromaDB ────────────────────────────────────────
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json

from .utils import (
    get_env_vars, setup_chroma_client, create_collection,
    chunk_text, format_citations, extract_metadata_from_pdf,
    load_or_create_embedding_model, get_timestamp,
    TextPreprocessor, preprocessor as default_preprocessor
)

logger = logging.getLogger(__name__)

class TextRAGPipeline:
    """Pipeline RAG complète pour analyse de documents texte/PDF."""
    
    def __init__(self, 
                 embedding_model: str = "intfloat/multilingual-e5-large",
                 retrieval_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                 temperature: float = 0.2,
                 max_memory: int = 5,
                 preprocessor: Optional[TextPreprocessor] = None,
                 llm_timeout: int = 60,
                 gemini_model: str = "gemini-2.5-flash"):
        """
        Initialise le pipeline RAG.
        
        Args:
            embedding_model:  Modèle pour les embeddings
            retrieval_model:  Modèle pour le re-ranking des résultats
            temperature:      Température du LLM (faible = moins d'hallucinations)
            max_memory:       Nombre de messages à conserver en historique
            preprocessor:     Instance de TextPreprocessor (utilise le singleton par défaut)
            llm_timeout:      Secondes avant d'abandonner la génération LLM (défaut 60)
            gemini_model:     Nom du modèle Gemini à utiliser (défaut: gemini-2.5-flash)
        """
        self.embedding_model_name = embedding_model
        self.retrieval_model_name = retrieval_model
        self.temperature = temperature
        self.max_memory = max_memory
        self.preprocessor = preprocessor or default_preprocessor
        self.llm_timeout = llm_timeout
        self.gemini_model = gemini_model
        
        # Initialisation des composants
        logger.info("⏳ Initialisation du pipeline RAG...")
        self.embedding_model = load_or_create_embedding_model(embedding_model)
        self.chroma_client = setup_chroma_client()
        self.collection = create_collection(self.chroma_client, "water_documents")
        
        # Chargement du modèle de re-ranking
        self._load_retrieval_model()
        
        # Historique de conversation
        self.conversation_history: List[Dict] = []
        
        # Client Gemini (nouveau SDK google-genai)
        self._init_gemini_client()
        
        logger.info("✓ Pipeline RAG initialisée")
    
    def _init_gemini_client(self):
        """Initialise le client Gemini avec le nouveau SDK."""
        try:
            from google import genai
            from google.genai import types
            
            env = get_env_vars()
            api_key = env.get('GEMINI_API_KEY') or env.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY ou GOOGLE_API_KEY non configurée dans .env")
            
            self.gemini_client = genai.Client(api_key=api_key)
            self.gemini_types = types
            logger.info("✓ Client Gemini initialisé (SDK google-genai)")
        except ImportError:
            logger.warning("⚠️  SDK google-genai non installé. Installez-le avec: pip install google-genai")
            self.gemini_client = None
            self.gemini_types = None
        except Exception as e:
            logger.warning(f"⚠️  Impossible d'initialiser le client Gemini: {e}")
            self.gemini_client = None
            self.gemini_types = None
    
    def _load_retrieval_model(self):
        """Charge le modèle de re-ranking pour améliorer la pertinence."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"⏳ Chargement du retriever: {self.retrieval_model_name}")
            self.retrieval_model = CrossEncoder(self.retrieval_model_name)
            logger.info("✓ Retriever chargé")
        except Exception as e:
            logger.warning(f"⚠ Retriever non disponible: {e}. Utilisation fallback.")
            self.retrieval_model = None
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Ingère un PDF dans la base vectorielle.
        
        Args:
            pdf_path: Chemin du fichier PDF
            chunk_size: Taille des chunks
            chunk_overlap: Chevauchement entre chunks
        """
        try:
            from PyPDF2 import PdfReader
            
            logger.info(f"⏳ Chargement PDF: {os.path.basename(pdf_path)}")
            logger.info(f"   Prétraitement actif — options: "
                        f"unicode={self.preprocessor.normalize_unicode}, "
                        f"hyphenation={self.preprocessor.fix_hyphenation}, "
                        f"lowercase={self.preprocessor.lowercase}, "
                        f"collapse_repeated={self.preprocessor.collapse_repeated_chars}, "
                        f"min_line={self.preprocessor.min_line_length}")
            
            file_name = os.path.basename(pdf_path)
            doc_id    = file_name.replace('.pdf', '').replace(' ', '_')

            # ── Protection anti-doublons ─────────────────────────────────
            try:
                existing = self.collection.get(
                    where={"doc_id": {"$eq": doc_id}},
                    limit=1,
                    include=[]
                )
                already_indexed = bool(existing and existing.get("ids"))
            except Exception as check_err:
                logger.warning(f"⚠ Vérification doublon impossible : {check_err} — on continue.")
                already_indexed = False

            if already_indexed:
                logger.info(
                    f"⏭  '{file_name}' déjà indexé (doc_id='{doc_id}' trouvé dans ChromaDB) — ignoré."
                )
                return {"status": "skipped", "file": file_name, "reason": "already_indexed"}
            # ────────────────────────────────────────────────────────────

            # Extraction du texte
            with open(pdf_path, 'rb') as f:
                reader    = PdfReader(f)
                num_pages = len(reader.pages)

                chunks_by_page = []
                for page_num, page in enumerate(reader.pages, 1):
                    raw_text = page.extract_text()
                    if not raw_text or not raw_text.strip():
                        continue

                    # ── Prétraitement du texte brut ──────────────────────
                    clean_text = self.preprocessor.preprocess(raw_text)

                    if not clean_text:
                        logger.debug(f"   Page {page_num}: vide après prétraitement, ignorée")
                        continue

                    stats = self.preprocessor.get_stats(raw_text, clean_text)
                    logger.debug(
                        f"   Page {page_num}: {stats['original_chars']} → "
                        f"{stats['cleaned_chars']} chars (-{stats['reduction_pct']}%)"
                    )
                    # ────────────────────────────────────────────────────

                    chunks = chunk_text(clean_text, chunk_size, chunk_overlap)
                    if chunks:
                        chunks_by_page.append((page_num, chunks))

            # Génération des embeddings et stockage via upsert (idempotent)
            all_ids = []
            for page_num, chunks in chunks_by_page:
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id  = f"{doc_id}_p{page_num}_c{chunk_idx}"
                    embedding = self.embedding_model.encode(chunk, convert_to_tensor=False).tolist()

                    self.collection.upsert(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "file":        file_name,
                            "page":        page_num,
                            "doc_id":      doc_id,
                            "chunk_idx":   chunk_idx,
                            "ingested_at": get_timestamp()
                        }]
                    )
                    all_ids.append(chunk_id)

            logger.info(f"✓ PDF ingéré: {len(all_ids)} chunks indexés ({num_pages} pages)")
            return {
                "status":         "success",
                "file":           file_name,
                "chunks_indexed": len(all_ids),
                "pages":          num_pages
            }
            
        except Exception as e:
            logger.error(f"✗ Erreur ingestion PDF: {e}")
            return {"status": "error", "error": str(e)}
    
    def is_collection_ready(self) -> bool:
        """Retourne True si la collection ChromaDB contient au moins un chunk."""
        try:
            count = self.collection.count()
            logger.info(f"✓ Collection prête — {count} chunks indexés")
            return count > 0
        except Exception as e:
            logger.error(f"✗ Impossible de vérifier la collection : {e}")
            return False

    def ingest_directory(self, directory: str, chunk_size: int = 512):
        """
        Ingère tous les PDFs d'un répertoire.
        Les fichiers déjà indexés sont ignorés (status='skipped').
        Retourne une liste de résultats par fichier (rétrocompatible).
        Utilisez is_collection_ready() pour savoir si le pipeline peut répondre.

        Args:
            directory:  Chemin du répertoire contenant les PDFs
            chunk_size: Taille des chunks
        """
        results   = []
        pdf_dir   = Path(directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"⚠ Aucun PDF trouvé dans : {directory}")

        for pdf_file in pdf_files:
            result = self.ingest_pdf(str(pdf_file), chunk_size)
            results.append(result)

        n_success = sum(1 for r in results if r.get("status") == "success")
        n_skipped = sum(1 for r in results if r.get("status") == "skipped")
        n_error   = sum(1 for r in results if r.get("status") == "error")

        if n_skipped and not n_success:
            logger.info(
                f"⏭  Tous les PDFs étaient déjà indexés "
                f"({n_skipped} ignorés, {n_error} erreurs). "
                f"Appelez is_collection_ready() pour continuer."
            )
        else:
            logger.info(
                f"✓ Répertoire traité — "
                f"indexés: {n_success} | ignorés: {n_skipped} | erreurs: {n_error}"
            )

        return results
    
    def retrieve(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        Récupère les chunks les plus pertinents.
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats
            threshold: Seuil de pertinence
        """
        query = self.preprocessor.preprocess(query) or query

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k*2,
            include=["documents", "distances", "metadatas"]
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        retrieved = [
            {
                'document': doc,
                'distance': dist,
                'metadata': meta,
                'file': meta.get('file', 'Unknown'),
                'page': meta.get('page', 1)
            }
            for doc, dist, meta in zip(documents, distances, metadatas)
        ]
        
        if self.retrieval_model and retrieved:
            pairs = [(query, item['document']) for item in retrieved]
            scores = self.retrieval_model.predict(pairs)
            
            max_score = max(scores) if max(scores) > 0 else 1
            for item, score in zip(retrieved, scores):
                item['rerank_score'] = score / max_score
        else:
            for item in retrieved:
                item['rerank_score'] = 1 - item['distance']
        
        retrieved = sorted(retrieved, key=lambda x: x['rerank_score'], reverse=True)
        retrieved = [r for r in retrieved if r['rerank_score'] >= threshold][:k]
        
        logger.info(f"✓ {len(retrieved)} documents récupérés (pertinence >= {threshold})")
        return retrieved
    
    def generate_response(self, query: str, use_docs: bool = True, k: int = 5) -> Dict[str, Any]:
        """
        Génère une réponse RAG complète avec citations.
        
        Args:
            query: Question de l'utilisateur
            use_docs: Utiliser les documents pour la réponse
            k: Nombre de documents à récupérer
        """
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": get_timestamp()
        })
        
        if len(self.conversation_history) > self.max_memory * 2:
            self.conversation_history = self.conversation_history[-self.max_memory*2:]
        
        retrieved_docs = self.retrieve(query, k=k) if use_docs else []
        
        context_text = ""
        if retrieved_docs:
            context_text = "Contexte des documents :\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                context_text += f"[{i}] (Page {doc['page']}, {doc['file']}):\n{doc['document']}\n\n"
        
        prompt = self._build_prompt(query, context_text)
        
        response_text = self._generate_with_llm(prompt)
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": get_timestamp()
        })
        
        citations = ""
        if retrieved_docs:
            citations = format_citations([
                {'file': d['file'], 'page': d['page'], 'distance': d['distance']}
                for d in retrieved_docs
            ])
        
        return {
            "query": query,
            "response": response_text,
            "citations": citations,
            "sources": [{"file": d['file'], "page": d['page']} for d in retrieved_docs],
            "num_sources": len(retrieved_docs),
            "temperature": self.temperature
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Construit le prompt avec instructions anti-hallucination."""
        system_prompt = """Tu es BLUE-Gen, un assistant IA spécialisé en gestion et ressources hydriques.
        
INSTRUCTIONS CRITIQUES:
1. Réponds EXCLUSIVEMENT en français et sans émojis
2. Si un contexte de documents est fourni, utilise-le obligatoirement pour construire une réponse claire et cohérente.
3. Si aucun document n'est disponible, le dire explicitement: "Je n'ai pas d'informations dans mes documents"
4. Ne fais JAMAIS de suppositions ou hallucinations"""

        if context:
            prompt = f"""{system_prompt}

CONTEXTE FOURNI:
{context}

REGLE: Tu DOIS baser ta réponse EXCLUSIVEMENT sur le contexte ci-dessus pour construire une réponse claire.
Ne fais pas d'ajouts, d'interprétations personnelles ou de connaissances externes.

QUESTION: {query}

REPONSE:"""
        else:
            prompt = f"""{system_prompt}

QUESTION (Pas de contexte de documents disponible): {query}

REPONSE: [Indique que tu n'as pas d'informations relatives aux documents indexés]"""
        
        return prompt
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Génère une réponse avec Gemini (Google AI) via le nouveau SDK google-genai.
        Utilise GenerateContentConfig pour system_instruction et generation_config.
        """
        if self.gemini_client is None:
            logger.warning("⚠️  Client Gemini non initialisé — bascule sur le mode fallback.")
            return self._simple_synthesis(prompt)
        
        try:
            # Construction du contenu avec historique conversationnel
            contents = []
            
            # Ajout de l'historique récent (limité pour ne pas dépasser les tokens)
            recent_history = self.conversation_history[-(self.max_memory * 2):-1]  # Exclut le message actuel
            for msg in recent_history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    self.gemini_types.Content(
                        role=role,
                        parts=[self.gemini_types.Part.from_text(text=msg["content"])]
                    )
                )
            
            # Ajout du prompt actuel
            contents.append(
                self.gemini_types.Content(
                    role="user",
                    parts=[self.gemini_types.Part.from_text(text=prompt)]
                )
            )
            
            # Configuration de génération avec system_instruction séparée
            config = self.gemini_types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=1024,  # Augmenté pour des réponses plus complètes
                top_p=0.95,
                top_k=40,
                system_instruction="Tu es BLUE-Gen, un assistant IA spécialisé en gestion et ressources hydriques. Réponds en français, de manière factuelle et scientifique."
            )
            
            logger.info(f"⏳ Appel Gemini (modèle: {self.gemini_model})...")
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=config
            )
            
            # Vérification de la réponse
            if not response.candidates:
                logger.warning("⚠️  Réponse vide de Gemini (candidates vides)")
                return self._simple_synthesis(prompt)
            
            # Extraction du texte (gestion des candidates et parts)
            text_parts = []
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
            
            result_text = " ".join(text_parts).strip()
            
            if not result_text:
                logger.warning("⚠️  Texte vide dans la réponse Gemini")
                return self._simple_synthesis(prompt)
            
            logger.info("✓ Réponse Gemini générée avec succès")
            return result_text

        except Exception as e:
            logger.warning(f"⚠️  Erreur Gemini: {e} — bascule sur le mode fallback.")
            return self._simple_synthesis(prompt)
    
    def _simple_synthesis(self, prompt: str) -> str:
        """Synthèse simple quand le LLM n'est pas disponible."""
        if "CONTEXTE FOURNI:" in prompt:
            context_start = prompt.find("CONTEXTE FOURNI:") + len("CONTEXTE FOURNI:")
            context_end = prompt.find("REGLE:")
            context = prompt[context_start:context_end].strip()
            
            sentences = context.split('\n')
            summary = '\n'.join([s for s in sentences if s.strip()][:3])
            return f"Basé sur les documents indexés :\n\n{summary}"
        else:
            return "Je n'ai pas d'informations relatives à cette question dans les documents indexés."
    
    def add_to_memory(self, role: str, content: str):
        """Ajoute manuellement un message à la mémoire."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": get_timestamp()
        })
        
        if len(self.conversation_history) > self.max_memory * 2:
            self.conversation_history = self.conversation_history[-self.max_memory*2:]
    
    def get_memory(self) -> List[Dict]:
        """Retourne l'historique de conversation."""
        return self.conversation_history
    
    def clear_memory(self):
        """Vide l'historique de conversation."""
        self.conversation_history = []
        logger.info("✓ Historique effacé")
    
    def export_collection(self, output_path: str):
        """Exporte les données indexées en JSON."""
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": self.embedding_model_name,
                    "num_documents": len(data['ids']),
                    "documents": data,
                    "exported_at": get_timestamp()
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Collection exportée: {output_path}")
        except Exception as e:
            logger.error(f"✗ Erreur export: {e}")


# Alias pour compatibility
RAGPipeline = TextRAGPipeline