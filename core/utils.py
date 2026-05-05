"""
Utils.py - Utilitaires communs pour BLUE-Gen
"""
import os

# ── FIX protobuf conflict with ChromaDB ────────────────────────────────────────
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import re
import unicodedata
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_env_vars() -> Dict[str, str]:
    """Charge les variables d'environnement nécessaires."""
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
        logger.info(f"✓ Chargement .env depuis: {dotenv_path}")
    else:
        logger.warning("⚠ .env introuvable ; utilisation des variables d'environnement déjà définies")
    
    return {
        'CHROMA_HOST': os.getenv('CHROMA_HOST', 'api.trychroma.com'),
        'CHROMA_API_KEY': os.getenv('CHROMA_API_KEY', ''),
        'CHROMA_TENANT': os.getenv('CHROMA_TENANT', ''),
        'CHROMA_DATABASE': os.getenv('CHROMA_DATABASE', 'proj_ia_generative'),
        'HF_TOKEN': os.getenv('HF_TOKEN', ''),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
    }

def setup_chroma_client():
    """Initialise le client Chroma pour l'indexation vectorielle."""
    import chromadb
    
    env = get_env_vars()
    
    # Afficher les paramètres de connexion (sans clé API)
    logger.info(f"⏳ Connexion ChromaDB:")
    logger.info(f"   - Host: {env['CHROMA_HOST']}")
    logger.info(f"   - Tenant: {env['CHROMA_TENANT']}")
    logger.info(f"   - Database: {env['CHROMA_DATABASE']}")
    logger.info(f"   - Auth: {'Configurée' if env['CHROMA_API_KEY'] else 'Non configurée'}")
    
    # ChromaDB HttpClient avec authentification correcte
    try:
        api_headers = [
            {"X-Chroma-Token": env['CHROMA_API_KEY']},   # ✅ Header officiel Chroma Cloud
            {"Authorization": f"Bearer {env['CHROMA_API_KEY']}"},
            {"X-API-Key": env['CHROMA_API_KEY']},
            {"apiKey": env['CHROMA_API_KEY']},
        ]

        if env.get('CHROMA_HEADER_NAME'):
            api_headers.insert(0, {env['CHROMA_HEADER_NAME']: env['CHROMA_API_KEY']})

        last_error = None
        for headers in api_headers:
            try:
                logger.info(f"⏳ Tentative de connexion ChromaDB avec header: {list(headers.keys())[0]}")
                client = chromadb.HttpClient(
                    host=env['CHROMA_HOST'],
                    port=443,
                    ssl=True,
                    headers=headers,
                    tenant=env['CHROMA_TENANT'],
                    database=env['CHROMA_DATABASE']
                )
                logger.info(f"✓ Client ChromaDB connecté avec succès (header {list(headers.keys())[0]})")
                return client
            except Exception as e:
                logger.warning(f"⚠️  Connexion ChromaDB échouée avec header {list(headers.keys())[0]}: {e}")
                last_error = e

        # Si la première tentative échoue sur TypeError parce que la version de chromadb est ancienne,
        # retenter avec la même logique de headers.
        if isinstance(last_error, TypeError):
            raise
        raise last_error
    except TypeError as e:
        if "tenant_name" in str(e):
            # Fallback pour anciennes versions de chromadb
            logger.warning("⚠️  Utilisation du fallback ChromaDB (version ancienne)")
            client = chromadb.HttpClient(
                host=env['CHROMA_HOST'],
                port=443,
                ssl=True,
                headers={
                    "Authorization": f"Bearer {env['CHROMA_API_KEY']}"
                },
                tenant=env['CHROMA_TENANT'],
                database=env['CHROMA_DATABASE']
            )
            logger.info("✓ Client ChromaDB connecté (mode compatible)")
            return client
        else:
            raise

def create_collection(client, collection_name: str, metadata: Dict = None) -> Any:
    """Crée ou récupère une collection ChromaDB."""
    if metadata is None:
        metadata = {"hnsw:space": "cosine"}
    
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata=metadata
        )
        logger.info(f"✓ Collection '{collection_name}' prête")
        return collection
    except Exception as e:
        logger.error(f"✗ Erreur création collection: {e}")
        raise

def get_timestamp() -> str:
    """Retourne un timestamp formaté."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class TextPreprocessor:
    """
    Prétraitement et nettoyage de texte extrait de PDFs.

    Étapes appliquées dans l'ordre :
      1. Normalisation Unicode (NFKC)
      2. Remplacement des ligatures typographiques
      3. Reconstruction des mots coupés en fin de ligne (PDF hyphenation)
      4. Suppression des artefacts PDF (en-têtes, pieds de page, numéros de page…)
      5. Suppression des caractères de contrôle et non-imprimables
      6. Normalisation des espaces blancs et sauts de ligne
      7. Suppression du bruit résiduel (URLs, emails, longues séquences de symboles)
      8. Filtrage des lignes trop courtes ou non informatives
    """

    # ------------------------------------------------------------------ #
    #  Patterns de nettoyage compilés une seule fois                      #
    # ------------------------------------------------------------------ #

    # Ligatures courantes (LaTeX, Word, OCR)
    _LIGATURES: Dict[str, str] = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi',
        'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st',
        '\u0153': 'oe', '\u0152': 'OE',
        '\u00e6': 'ae', '\u00c6': 'AE',
    }

    # Mots coupés par un tiret en fin de ligne  →  mot reconstitué
    _RE_HYPHEN_BREAK = re.compile(r'(\w)-\n(\w)')

    # Numéros de page isolés (ligne ne contenant qu'un entier, éventuellement
    # entouré de tirets ou de points)
    _RE_PAGE_NUMBER = re.compile(r'^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$', re.MULTILINE)

    # En-têtes / pieds de page typiques des rapports PDF
    # (chaînes très courtes répétées, souvent en majuscules)
    _RE_HEADER_FOOTER = re.compile(
        r'^.{0,80}(?:confidentiel|©|copyright|all rights reserved|page\s+\d|'
        r'www\.|http|rapport\s+\d{4}|version\s+\d).{0,80}$',
        re.IGNORECASE | re.MULTILINE
    )

    # Caractères de contrôle (sauf tabulation \t et saut de ligne \n)
    _RE_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

    # Séquences répétitives de ponctuation ou symboles (ligne de tableau, séparateur)
    _RE_SYMBOL_LINES = re.compile(r'^[\s\-_=*#|~.]{4,}$', re.MULTILINE)

    # Longues successions de caractères identiques (ex: ".....", "-----", "     ")
    _RE_REPEATED_CHARS = re.compile(r'(.)\1{2,}')

    # URLs et adresses e-mail
    _RE_URL   = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    _RE_EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}')

    # Espaces multiples (hors sauts de ligne)
    _RE_MULTI_SPACE = re.compile(r'[^\S\n]+')

    # Sauts de ligne multiples  →  deux au maximum
    _RE_MULTI_NEWLINE = re.compile(r'\n{3,}')

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        min_line_length: int = 15,
        fix_hyphenation: bool = True,
        normalize_unicode: bool = True,
        lowercase: bool = True,
        collapse_repeated_chars: bool = True,
    ):
        """
        Args:
            remove_urls:             Supprime les URLs du texte.
            remove_emails:           Supprime les adresses e-mail.
            min_line_length:         Longueur minimale d'une ligne pour la conserver.
            fix_hyphenation:         Reconstruit les mots coupés en fin de ligne.
            normalize_unicode:       Applique la normalisation NFKC.
            lowercase:               Convertit tout le texte en minuscules.
            collapse_repeated_chars: Réduit les successions de caractères identiques à un seul.
        """
        self.remove_urls              = remove_urls
        self.remove_emails            = remove_emails
        self.min_line_length          = min_line_length
        self.fix_hyphenation          = fix_hyphenation
        self.normalize_unicode        = normalize_unicode
        self.lowercase                = lowercase
        self.collapse_repeated_chars  = collapse_repeated_chars

    # ------------------------------------------------------------------ #
    #  Étapes de nettoyage individuelles                                  #
    # ------------------------------------------------------------------ #

    def _normalize_unicode(self, text: str) -> str:
        """NFKC : décompose puis recompose, normalise les formes de compatibilité."""
        return unicodedata.normalize('NFKC', text)

    def _replace_ligatures(self, text: str) -> str:
        """Remplace les ligatures typographiques par leurs équivalents ASCII."""
        for ligature, replacement in self._LIGATURES.items():
            text = text.replace(ligature, replacement)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        """Reconstruit les mots coupés en fin de ligne par un tiret (artefact PDF)."""
        return self._RE_HYPHEN_BREAK.sub(r'\1\2', text)

    def _remove_control_chars(self, text: str) -> str:
        """Supprime les caractères de contrôle non-imprimables."""
        return self._RE_CONTROL_CHARS.sub('', text)

    def _remove_pdf_artifacts(self, text: str) -> str:
        """Supprime numéros de page, en-têtes, pieds de page et lignes de symboles."""
        text = self._RE_PAGE_NUMBER.sub('', text)
        text = self._RE_HEADER_FOOTER.sub('', text)
        text = self._RE_SYMBOL_LINES.sub('', text)
        return text

    def _remove_noise(self, text: str) -> str:
        """Supprime URLs et e-mails si activé."""
        if self.remove_urls:
            text = self._RE_URL.sub('', text)
        if self.remove_emails:
            text = self._RE_EMAIL.sub('', text)
        return text

    def _collapse_repeated_chars(self, text: str) -> str:
        """
        Réduit les longues successions de caractères identiques à un seul exemplaire.
        Ex: "..........page 3.........." → ".page 3."
            "------------------------------" → "-"
            "bonjooooooour" → "bonjour"
        Les espaces répétés sont exclus (gérés par _normalize_whitespace).
        """
        def replace(m: re.Match) -> str:
            char = m.group(1)
            return char if char != ' ' else m.group(0)   # espaces délégués à l'étape suivante
        return self._RE_REPEATED_CHARS.sub(replace, text)

    def _to_lowercase(self, text: str) -> str:
        """Convertit l'intégralité du texte en minuscules."""
        return text.lower()

    def _normalize_whitespace(self, text: str) -> str:
        """Normalise espaces multiples et sauts de ligne excessifs."""
        text = self._RE_MULTI_SPACE.sub(' ', text)
        text = self._RE_MULTI_NEWLINE.sub('\n\n', text)
        return text.strip()

    def _filter_short_lines(self, text: str) -> str:
        """
        Supprime les lignes trop courtes (numéros isolés, entêtes résiduels…).
        Les lignes vides servant de séparateurs sont conservées.
        """
        lines = text.split('\n')
        kept = []
        for line in lines:
            stripped = line.strip()
            # Conserver les lignes vides (séparateurs de paragraphes)
            if not stripped or len(stripped) >= self.min_line_length:
                kept.append(line)
        return '\n'.join(kept)

    # ------------------------------------------------------------------ #
    #  Point d'entrée principal                                           #
    # ------------------------------------------------------------------ #

    def preprocess(self, text: str) -> str:
        """
        Applique toutes les étapes de nettoyage dans l'ordre optimal.

        Args:
            text: Texte brut extrait d'un PDF ou d'une autre source.

        Returns:
            Texte nettoyé, prêt pour le chunking et l'embedding.
        """
        if not text or not text.strip():
            return ""

        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        text = self._replace_ligatures(text)

        if self.fix_hyphenation:
            text = self._fix_hyphenation(text)

        text = self._remove_control_chars(text)
        text = self._remove_pdf_artifacts(text)
        text = self._remove_noise(text)
        text = self._normalize_whitespace(text)

        if self.collapse_repeated_chars:
            text = self._collapse_repeated_chars(text)

        if self.lowercase:
            text = self._to_lowercase(text)

        text = self._filter_short_lines(text)
        text = self._normalize_whitespace(text)   # second passage après filtre

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Prétraite une liste de textes, en ignorant silencieusement les vides.

        Args:
            texts: Liste de textes bruts.

        Returns:
            Liste de textes nettoyés (les textes vides après nettoyage sont exclus).
        """
        results = []
        for text in texts:
            cleaned = self.preprocess(text)
            if cleaned:
                results.append(cleaned)
        logger.info(f"✓ Prétraitement : {len(texts)} textes → {len(results)} conservés")
        return results

    def get_stats(self, original: str, cleaned: str) -> Dict[str, Any]:
        """
        Calcule des statistiques de nettoyage pour le monitoring.

        Args:
            original: Texte avant nettoyage.
            cleaned:  Texte après nettoyage.

        Returns:
            Dictionnaire de métriques.
        """
        original_len = len(original)
        cleaned_len  = len(cleaned)
        reduction    = round((1 - cleaned_len / original_len) * 100, 1) if original_len else 0

        return {
            "original_chars":  original_len,
            "cleaned_chars":   cleaned_len,
            "reduction_pct":   reduction,
            "original_lines":  original.count('\n') + 1,
            "cleaned_lines":   cleaned.count('\n') + 1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Instance partagée (singleton léger) – peut être importée directement
# ─────────────────────────────────────────────────────────────────────────────
preprocessor = TextPreprocessor()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Découpe le texte en chunks avec chevauchement.
    
    Args:
        text: Texte à découper
        chunk_size: Taille des chunks
        overlap: Chevauchement entre chunks
        
    Raises:
        ValueError: Si chunk_size <= 0
        TypeError: Si chunk_size n'est pas un entier
    """
    # Validation
    if not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size doit être un entier, reçu {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size doit être positif, reçu {chunk_size}")
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 20:  # Ignore chunks vides
            chunks.append(chunk)
    
    return chunks

def format_citations(sources: List[Dict[str, Any]]) -> str:
    """
    Formate les citations à partir des sources récupérées.
    
    Args:
        sources: Liste des sources avec {file, page, distance}
    """
    citations = "\n\n---\n**📚 Sources :**\n"
    
    for i, src in enumerate(sources, 1):
        file_name = src.get('file', 'Unknown')
        page = src.get('page', 'N/A')
        distance = src.get('distance', 1.0)
        
        # Distance inversée comme confiance (0 très proche, 1 très loin)
        confidence = max(0, int((1 - distance) * 100))
        
        citations += f"{i}. **{file_name}** (page {page}) — Pertinence: {confidence}%\n"
    
    return citations

def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Extrait les métadonnées d'un PDF."""
    try:
        from PyPDF2 import PdfReader
        
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            num_pages = len(reader.pages)
            
            return {
                'file': os.path.basename(pdf_path),
                'path': pdf_path,
                'num_pages': num_pages,
                'created_at': get_timestamp()
            }
    except Exception as e:
        logger.error(f"✗ Erreur extraction métadonnées: {e}")
        return {'file': os.path.basename(pdf_path), 'error': str(e)}

def load_or_create_embedding_model(model_name: str = "intfloat/multilingual-e5-large"):
    """
    Charge le modèle d'embeddings HuggingFace.
    
    Args:
        model_name: Nom du modèle HuggingFace
    """
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"⏳ Chargement modèle d'embeddings: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"✓ Modèle d'embeddings chargé")
    
    return model