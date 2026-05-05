import streamlit as st
import time
import random
import math
import os
import sys
import logging
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import plotly.graph_objects as go

# ── AUTO-INSTALL missing dependencies ──────────────────────────────────────────
def ensure_dependencies():
    """Install missing dependencies if needed."""
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'diffusers': 'diffusers',
        'torch': 'torch',
        'transformers': 'transformers',
        'chromadb': 'chromadb',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        logging.info(f"Installing missing packages: {missing}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-q'
            ] + missing)
            logging.info(f"Successfully installed: {missing}")
        except Exception as e:
            logging.warning(f"Could not auto-install packages: {e}")

ensure_dependencies()

# ── FIX protobuf conflict with ChromaDB ────────────────────────────────────────
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# ── Configuration logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Import des modules BLUE-Gen ────────────────────────────────────────────────
try:
    from core.text_rag import TextRAGPipeline
    from core.image_lora import ImageLoRAPipeline
    from core.signal_forecast import SignalForecastPipeline
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠ Modules core non disponibles: {e}")
    MODULES_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant BLUE-Gen",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Tokens ── */
:root {
  --blue-deep:   #0A2540;
  --blue-mid:    #1565C0;
  --blue-bright: #1E88E5;
  --blue-light:  #64B5F6;
  --blue-pale:   #E3F2FD;
  --blue-sky:    #BBDEFB;
  --white:       #FFFFFF;
  --gray-soft:   #F0F4F8;
  --gray-text:   #4A5568;
  --accent:      #00BCD4;
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--blue-deep);
}
.main { background: var(--white); }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--blue-deep) 0%, #0D3461 60%, #0A2540 100%);
  border-right: 2px solid var(--blue-mid);
  min-width: 220px;
  max-width: 420px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg, var(--blue-deep) 0%, var(--blue-mid) 50%, var(--blue-bright) 100%);
  border-radius: 20px;
  padding: 2.5rem 3rem;
  margin-bottom: 1.8rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(10,37,64,0.25);
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 200'%3E%3Cpath fill='rgba(255,255,255,0.06)' d='M0,80 C360,160 720,0 1080,80 C1260,120 1380,60 1440,80 L1440,200 L0,200Z'/%3E%3Cpath fill='rgba(255,255,255,0.04)' d='M0,120 C480,40 960,180 1440,100 L1440,200 L0,200Z'/%3E%3C/svg%3E") bottom/cover no-repeat;
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 900;
  color: var(--white);
  line-height: 1.1;
  margin: 0 0 0.4rem;
  text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-sub {
  font-size: 1rem;
  color: var(--blue-sky);
  margin: 0;
  letter-spacing: 0.04em;
  font-weight: 300;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.3);
  color: white; font-size: 0.75rem; font-weight: 500;
  padding: 4px 12px; border-radius: 20px;
  margin-top: 1rem; backdrop-filter: blur(8px);
}
.hero-drop {
  position: absolute; right: 3rem; top: 50%; transform: translateY(-50%);
  font-size: 7rem; opacity: 0.12; user-select: none;
}

/* ── Mode selector : empêcher la saisie dans le selectbox ── */
.stSelectbox div[data-baseweb="select"] input {
    pointer-events: none !important;
    caret-color: transparent !important;
    user-select: none !important;
}

/* ── Chat container ── */
.chat-wrapper {
  background: var(--gray-soft);
  border-radius: 20px;
  padding: 1.5rem;
  min-height: 320px;
  max-height: 680px;
  overflow-y: auto;
  overflow-x: hidden;
  margin-bottom: 1.2rem;
  border: 1px solid var(--blue-sky);
  scroll-behavior: smooth;
}
.msg-row { display: flex; gap: 12px; margin-bottom: 1rem; align-items: flex-start; }
.msg-row.user { flex-direction: row-reverse; }
.avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; flex-shrink: 0;
}
.avatar.bot { background: linear-gradient(135deg, var(--blue-mid), var(--accent)); color: white; }
.avatar.user { background: var(--blue-pale); color: var(--blue-deep); }
.bubble {
  max-width: 72%; padding: 0.85rem 1.1rem;
  border-radius: 18px; font-size: 0.93rem; line-height: 1.55;
  box-shadow: 0 2px 10px rgba(0,0,0,0.07);
  word-break: break-word;
}
.bubble.bot {
  background: var(--white);
  color: var(--blue-deep);
  border-bottom-left-radius: 4px;
  border: 1px solid var(--blue-pale);
}
.bubble.user {
  background: linear-gradient(135deg, var(--blue-mid), var(--blue-bright));
  color: white;
  border-bottom-right-radius: 4px;
}
.msg-time { font-size: 0.68rem; color: #999; margin-top: 4px; text-align: right; }

/* ── Input bar ── */
.stTextInput > div > div > input {
  border: 2px solid var(--blue-sky) !important;
  border-radius: 50px !important;
  padding: 0.75rem 1.4rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  background: white !important;
  color: var(--blue-deep) !important;
  box-shadow: 0 2px 12px rgba(21,101,192,0.08) !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--blue-bright) !important;
  box-shadow: 0 4px 20px rgba(30,136,229,0.2) !important;
  outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--blue-mid), var(--blue-bright)) !important;
  color: white !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 0.65rem 2rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  cursor: pointer !important;
  transition: all .25s ease !important;
  box-shadow: 0 4px 15px rgba(21,101,192,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(21,101,192,0.4) !important;
}

/* ── Select / Radio ── */
.stRadio > div { gap: 0.6rem !important; }
.stRadio label { font-family: 'DM Sans', sans-serif !important; }
.stSelectbox > div > div {
  border: 2px solid var(--blue-sky) !important;
  border-radius: 12px !important;
  background: white !important;
}

/* ── Sidebar elements ── */
.sidebar-logo {
  text-align: center; padding: 1.6rem 1rem 1rem;
}
.sidebar-logo-text {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem; font-weight: 900; color: white;
  letter-spacing: 0.03em;
}
.sidebar-logo-sub { font-size: 0.72rem; color: var(--blue-light); letter-spacing: 0.1em; text-transform: uppercase; }
.sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 0.8rem 1rem; }
.sidebar-section { padding: 0.4rem 1rem 0.2rem; font-size: 0.7rem; color: var(--blue-light); text-transform: uppercase; letter-spacing: 0.12em; }
.stat-card {
  background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px; padding: 0.9rem 1rem; margin: 0.4rem 0;
  color: white; text-align: center;
}
.stat-num { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: var(--blue-light); }
.stat-label { font-size: 0.72rem; color: rgba(255,255,255,0.6); margin-top: 2px; }

/* ── Signal viz ── */
.signal-bar {
  display: inline-block;
  width: 8px;
  border-radius: 3px 3px 0 0;
  background: linear-gradient(to top, var(--blue-mid), var(--accent));
  margin: 0 2px;
  vertical-align: bottom;
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }

/* ── Image placeholder ── */
.img-placeholder {
  background: linear-gradient(135deg, var(--blue-pale), var(--blue-sky));
  border: 2px dashed var(--blue-bright);
  border-radius: 16px;
  padding: 3rem 2rem; text-align: center;
  color: var(--blue-mid);
}

/* ── Tooltip / chip ── */
.chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--blue-pale); color: var(--blue-mid);
  border: 1px solid var(--blue-sky);
  border-radius: 20px; padding: 3px 10px;
  font-size: 0.75rem; font-weight: 500;
  margin: 2px;
}

/* ── Footer ── */
.app-footer {
  background: linear-gradient(90deg, rgba(10,37,64,0.04) 0%, rgba(21,101,192,0.07) 50%, rgba(10,37,64,0.04) 100%);
  border-top: 1px solid var(--blue-sky);
  border-radius: 10px;
  padding: 0.35rem 1rem;
  margin-top: 0.4rem;
  text-align: center;
  color: #7a90a8;
  font-size: 0.75rem;
}

.info-box {
  background: linear-gradient(135deg, rgba(30,136,229,0.1), rgba(0,188,212,0.1));
  border-left: 4px solid var(--blue-bright);
  border-radius: 8px;
  padding: 0.8rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: var(--blue-deep);
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
LORA_ADAPTER_DIR = Path("lancement/lora_adapter")
SIGNAL_DATA_DIR = Path("dataset/signaux")

# Fonction en cache pour charger le pipeline RAG
@st.cache_resource
def get_rag_pipeline():
    """Charge le pipeline RAG une seule fois et le met en cache."""
    try:
        logger.info("⏳ Initialisation du pipeline RAG (première exécution)...")
        pipeline = TextRAGPipeline()
        logger.info("✓ Pipeline RAG chargé en cache")
        return pipeline, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Erreur RAG: {error_msg}")
        logger.exception("Traceback complet:")
        return None, error_msg

@st.cache_resource
def get_image_pipeline():
    """Charge le pipeline Image/LoRA une seule fois et le met en cache."""
    try:
        logger.info("⏳ Initialisation du pipeline Image (première exécution)...")

        # Vérifier d'abord s'il existe un modèle fusionné (priorité)
        merged_pipeline_dir = LORA_ADAPTER_DIR / "merged_pipeline"
        if merged_pipeline_dir.exists() and any(merged_pipeline_dir.iterdir()):
            logger.info("✓ Modèle fusionné trouvé - chargement du modèle final...")
            pipeline = ImageLoRAPipeline()
            load_res = pipeline.load_merged_model(str(LORA_ADAPTER_DIR / "merged_model"))
            if load_res.get("status") == "success":
                logger.info("✓ Modèle Stable Diffusion fusionné (LoRA intégré) chargé")
                logger.info("✓ Pipeline Image chargé en cache")
                return pipeline, None
            else:
                logger.warning(f"⚠ Chargement modèle fusionné échoué: {load_res.get('error')}")
                # Fallback sur l'adaptateur LoRA

        # Fallback: charger l'adaptateur LoRA s'il existe
        pipeline = ImageLoRAPipeline()
        if LORA_ADAPTER_DIR.exists() and any(LORA_ADAPTER_DIR.iterdir()):
            # Vérifier si c'est un ancien adaptateur ou un nouveau avec fusion
            metadata_file = LORA_ADAPTER_DIR / "merged_model_metadata.json"
            if not metadata_file.exists():
                load_res = pipeline.load_lora_adapter(str(LORA_ADAPTER_DIR))
                if load_res.get("status") == "success":
                    logger.info("✓ Adaptateur LoRA fine-tuné chargé")
                else:
                    logger.warning(f"⚠ Chargement adaptateur LoRA échoué: {load_res.get('error')}")

        logger.info("✓ Pipeline Image chargé en cache")
        return pipeline, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Erreur Image: {error_msg}")
        logger.exception("Traceback complet:")
        return None, error_msg

@st.cache_resource
def get_signal_pipeline():
    """Charge le pipeline Signal/Forecast une seule fois et le met en cache."""
    try:
        logger.info("⏳ Initialisation du pipeline Signal/Forecast (première exécution)...")
        pipeline = SignalForecastPipeline()

        if SIGNAL_DATA_DIR.exists():
            first_signal = next(SIGNAL_DATA_DIR.glob("*.xlsx"), None)
            if first_signal:
                load_res = pipeline.load_signal(str(first_signal))
                if load_res.get("status") == "success":
                    pipeline.loaded_signal_file = str(first_signal)
                    logger.info(f"✓ Données de signal historiques chargées depuis {first_signal.name}")
                else:
                    logger.warning(f"⚠ Chargement signal initial échoué: {load_res.get('error')}")

        logger.info("✓ Pipeline Signal/Forecast chargé en cache")
        return pipeline, None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Erreur Signal: {error_msg}")
        logger.exception("Traceback complet:")
        return None, error_msg

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Bonjour ! Je suis BLUE-Gen, votre assistant intelligent spécialisé en eau. Choisissez un mode ci-dessus et posez-moi votre question.", "time": "maintenant", "type": "text"},
    ]
if "mode" not in st.session_state:
    st.session_state.mode = "Texte"
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0
if "img_count" not in st.session_state:
    st.session_state.img_count = 0
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "image_pipeline" not in st.session_state:
    st.session_state.image_pipeline = None
if "signal_pipeline" not in st.session_state:
    st.session_state.signal_pipeline = None
if "rag_error" not in st.session_state:
    st.session_state.rag_error = None
if "image_error" not in st.session_state:
    st.session_state.image_error = None
if "signal_error" not in st.session_state:
    st.session_state.signal_error = None
if "last_generated_image" not in st.session_state:
    st.session_state.last_generated_image = None
if "last_image_prompt" not in st.session_state:
    st.session_state.last_image_prompt = None
if "last_image_tokens" not in st.session_state:
    st.session_state.last_image_tokens = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "pending_mode" not in st.session_state:
    st.session_state.pending_mode = None


def init_rag_pipeline():
    """Initialise le pipeline RAG depuis le cache."""
    if st.session_state.rag_pipeline is None:
        pipeline, error = get_rag_pipeline()
        st.session_state.rag_pipeline = pipeline
        st.session_state.rag_error = error
        if error:
            logger.error(f"Pipeline RAG échoué: {error}")


def init_image_pipeline():
    """Initialise le pipeline Image/LoRA depuis le cache."""
    if st.session_state.image_pipeline is None:
        pipeline, error = get_image_pipeline()
        st.session_state.image_pipeline = pipeline
        st.session_state.image_error = error
        if error:
            logger.error(f"Pipeline Image échoué: {error}")


def init_signal_pipeline():
    """Initialise le pipeline Signal/Forecast depuis le cache."""
    if st.session_state.signal_pipeline is None:
        pipeline, error = get_signal_pipeline()
        st.session_state.signal_pipeline = pipeline
        st.session_state.signal_error = error
        if error:
            logger.error(f"Pipeline Signal échoué: {error}")

# ── INIT PIPELINES AT STARTUP (one time only) ──────────────────────────────────
# Initialise les pipelines au démarrage, pas à chaque rerun
if not hasattr(st.session_state, 'pipelines_initialized'):
    st.session_state.pipelines_initialized = True
    st.info("⏳ Initialisation des pipelines... (première exécution)")
    init_rag_pipeline()
    init_image_pipeline()
    init_signal_pipeline()
    st.success("✓ Pipelines prêts !")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div style="font-size:2.4rem;margin-bottom:6px;">💧</div>
      <div class="sidebar-logo-text">BLUE-Gen</div>
      <div class="sidebar-logo-sub">Assistant Eau Intelligent</div>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Statistiques de session</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{st.session_state.msg_count}</div>
          <div class="stat-label">Messages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{st.session_state.img_count}</div>
          <div class="stat-label">Générations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Thèmes eau disponibles</div>', unsafe_allow_html=True)
    topics = ["Hydrologie", "Eau potable", "Irrigation", "Océanographie", "Traitement des eaux", "Signaux hydro"]
    topics_html = ''.join([f'<div class="chip">{t}</div>' for t in topics])
    st.markdown(f'<div class="sidebar-themes">{topics_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = [
            {"role": "bot", "text": "Conversation réinitialisée. Comment puis-je vous aider ?", "time": "maintenant", "type": "text"},
        ]
        st.session_state.msg_count = 0
        st.session_state.img_count = 0
        st.session_state.pending_prompt = None
        st.session_state.pending_mode = None
        if "user_input" in st.session_state:
            del st.session_state.user_input
        st.rerun()

# ── MAIN ───────────────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class="hero">
  <div class="hero-drop">💧</div>
  <div class="hero-title">Assistant BLUE-Gen</div>
  <p class="hero-sub">Intelligence artificielle dédiée à l'eau — Texte · Image · Signal</p>
  <div class="hero-badge">🟢 IA connectée &nbsp;|&nbsp; Modèle multimodal v2.4</div>
</div>
""", unsafe_allow_html=True)

# Mode selector
mode_options = {
    "💬 Génération Texte": "Texte",
    "🖼️ Génération Image": "Image",
    "📡 Génération Signal": "Signal",
}

# Créer les colonnes pour le selectbox
col1, col2 = st.columns([2, 3])
with col1:
    selected_mode_display = st.selectbox(
        "Type de génération",
        list(mode_options.keys()),
        index=list(mode_options.values()).index(st.session_state.mode),
        label_visibility="collapsed",
        key="mode_selector"
    )
    new_mode = mode_options[selected_mode_display]
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.rerun()

# ── Chat area ──────────────────────────────────────────────────────────────────
chat_html = '<div class="chat-wrapper">'
for m in st.session_state.messages:
    role_cls = "user" if m["role"] == "user" else "bot"
    avatar_icon = "👤" if m["role"] == "user" else "💧"
    msg_type = m.get("type", "text")

    if msg_type == "image":
        data = m.get("html_data", {})
        img_base64 = data.get("base64", "")
        prompt = data.get("prompt", "")
        tokens = data.get("tokens", [])
        is_placeholder = data.get("is_placeholder", False)
        source_label = "Image placeholder (diffusers non disponible)" if is_placeholder else "Image générée avec LoRA Fine-tuned"
        prompt_display = prompt[:80] + ("…" if len(prompt) > 80 else "")
        tokens_html = ""
        if tokens:
            tokens_str = ", ".join(tokens)
            tokens_html = '<div style="font-size: 0.8rem; color: #666; margin-top: 8px;">Tokens reconnus: ' + tokens_str + '</div>'

        inner_html = (
            '<div style="border-radius: 14px; padding: 12px; background: #E3F2FD; border: 2px solid #1565C0; max-width: 440px;">'
            '<div style="font-size: 0.9rem; color: #0A2540; margin-bottom: 8px;">'
            '<strong>🎨 ' + source_label + '</strong><br>'
            '<em style="font-size: 0.85rem;">Prompt: ' + prompt_display + '</em>'
            '</div>'
            '<img src="data:image/png;base64,' + img_base64 + '" style="width: 100%; max-width: 400px; border-radius: 10px; border: 1px solid #1E88E5; display: block;">'
            + tokens_html +
            '</div>'
        )
        chat_html += (
            '<div class="msg-row ' + role_cls + '">'
            '<div class="avatar ' + role_cls + '">' + avatar_icon + '</div>'
            '<div>' + inner_html + '<div class="msg-time">' + m["time"] + '</div></div>'
            '</div>'
        )

    elif msg_type == "signal":
        data = m.get("html_data", {})
        fig_html = data.get("fig_html", "")
        meta_html = data.get("meta_html", "")
        inner_html = (
            '<div style="background:#E3F2FD;border:2px solid #1565C0;border-radius:14px;'
            'padding:14px;margin:8px 0;">'
            '<div style="font-size:0.95rem;color:#0A2540;font-weight:bold;margin-bottom:10px;">'
            '📡 Analyse &amp; Prédiction de Signal'
            '</div>'
            + fig_html + meta_html +
            '</div>'
        )
        chat_html += (
            '<div class="msg-row ' + role_cls + '">'
            '<div class="avatar ' + role_cls + '">' + avatar_icon + '</div>'
            '<div>' + inner_html + '<div class="msg-time">' + m["time"] + '</div></div>'
            '</div>'
        )

    else:
        chat_html += (
            '<div class="msg-row ' + role_cls + '">'
            '<div class="avatar ' + role_cls + '">' + avatar_icon + '</div>'
            '<div>'
            '<div class="bubble ' + role_cls + '">' + m["text"] + '</div>'
            '<div class="msg-time">' + m["time"] + '</div>'
            '</div>'
            '</div>'
        )

chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Input row ──────────────────────────────────────────────────────────────────
current_mode = st.session_state.mode

placeholders = {
    "Texte": "Ex : Quels sont les enjeux de la gestion de l'eau en Afrique subsaharienne ?",
    "Image": "Ex : Carte de distribution des ressources en eau souterraine au Sahel…",
    "Signal": "Ex : Générer un signal de débit du fleuve Niger sur 30 jours…",
}

inp_col, btn_col = st.columns([4, 1])
with inp_col:
    user_input = st.text_input(
        label="Votre question",
        placeholder=placeholders[current_mode],
        key="user_input",
        label_visibility="collapsed"
    )
with btn_col:
    send_btn = st.button("Envoyer ➤", use_container_width=True)

# ── Response logic ─────────────────────────────────────────────────────────────
def get_timestamp():
    return datetime.now().strftime("%H:%M")

def handle_text_mode(query: str):
    """Traite les requêtes en mode Texte (RAG)."""
    if st.session_state.rag_pipeline is None:
        error_detail = st.session_state.rag_error or "Erreur d'initialisation inconnue"
        return f"⚠ Pipeline RAG non initialisé.\n\nErreur: {error_detail}"

    try:
        retrieved_docs = st.session_state.rag_pipeline.retrieve(query, k=5)

        if not retrieved_docs:
            return "⚠ Aucun document pertinent trouvé dans la base. Assurez-vous d'avoir indexé des PDFs."

        result = st.session_state.rag_pipeline.generate_response(query)
        response = f"{result['response']}"
        if result['citations']:
            response += result['citations']
        return response
    except Exception as e:
        logger.exception(f"Erreur lors du traitement RAG: {e}")
        return f"✗ Erreur: {str(e)}"


def handle_image_mode(query: str):
    """Traite les requêtes en mode Image (LoRA)."""
    if st.session_state.image_pipeline is None:
        error_detail = st.session_state.image_error or "Erreur d'initialisation inconnue"
        return {"type": "text", "text": f"⚠ Pipeline Image/LoRA non initialisé.\n\nErreur: {error_detail}"}

    try:
        result = st.session_state.image_pipeline.generate_image(query)

        if result.get('status') == 'error':
            return {"type": "text", "text": f"✗ Erreur génération d'image: {result.get('error', 'Erreur inconnue')}"}

        generated_images = result.get('generated_images', [])
        tokens_used = result.get('tokens_used', [])

        if not generated_images:
            return {"type": "text", "text": "⚠ Aucune image générée"}

        image_data = generated_images[0]
        img_base64 = image_data.get('base64')

        if not img_base64:
            return {"type": "text", "text": "✗ Erreur : données image vides (base64 manquant)"}

        st.session_state.last_generated_image = img_base64
        st.session_state.last_image_prompt = query
        st.session_state.last_image_tokens = tokens_used

        is_placeholder = image_data.get('format') == 'png' and len(img_base64) < 20000

        return {
            "type": "image",
            "base64": img_base64,
            "prompt": query,
            "tokens": tokens_used,
            "is_placeholder": is_placeholder
        }

    except Exception as e:
        logger.exception(f"Erreur lors de la génération d'image: {e}")
        return {"type": "text", "text": f"✗ Erreur: {str(e)}"}


def parse_signal_request(query: str) -> Dict[str, Any]:
    """Parse une requête utilisateur pour extraire les paramètres de forecasting."""
    query_lower = query.lower()

    parsed = {
        'data_type': None,
        'region': None,
        'periods': None,
        'time_unit': 'annuelle'
    }

    if any(term in query_lower for term in ['vente', 'ventes']):
        parsed['data_type'] = 'ventes'
    elif any(term in query_lower for term in ['production', 'productions']):
        parsed['data_type'] = 'production'
    elif any(term in query_lower for term in ['abonne', 'abonnes', 'abonnement']):
        parsed['data_type'] = 'abonnes'
    elif any(term in query_lower for term in ['qualite', 'qualité']):
        parsed['data_type'] = 'qualite'

    regions = ['centre', 'nord', 'sud', 'est', 'ouest', 'centre-est', 'centre-nord',
               'centre-ouest', 'centre-sud', 'sud-ouest', 'hauts-bassins', 'plateau central', 'sahel']

    for region in regions:
        if region in query_lower:
            parsed['region'] = region.title()
            break

    import re
    duration_match = re.search(r'(\d+)\s*(prochaines?\s*)?(ans?|années?|mois|semaines?|jours?)', query_lower)
    if duration_match:
        num = int(duration_match.group(1))
        unit = duration_match.group(3) if duration_match.group(3) else duration_match.group(2)

        if unit and any(term in str(unit) for term in ['an', 'ans', 'année']):
            parsed['periods'] = num
            parsed['time_unit'] = 'annuelle'
        elif unit and 'mois' in str(unit):
            parsed['periods'] = num
            parsed['time_unit'] = 'mensuelle'
        elif unit and any(term in str(unit) for term in ['semaine']):
            parsed['periods'] = num
            parsed['time_unit'] = 'hebdomadaire'
        elif unit and 'jour' in str(unit):
            parsed['periods'] = num
            parsed['time_unit'] = 'quotidienne'

    logger.info(f"✓ Requête parsée: {parsed}")
    return parsed


def format_signal_response(query: str, parsed_request: Dict, run_result: Dict):
    """
    Formate la réponse du module Signal avec :
      - courbe historique (données réelles)
      - courbe de projection
      - enveloppe d'incertitude (intervalles de confiance)
    Retourne un dict avec les données HTML préparées.
    """
    if run_result.get('status') != 'success':
        return {"type": "text", "text": f"✗ Erreur prédiction: {run_result.get('error', 'Inconnu')}"}

    try:
        forecast_info = run_result.get('forecast', {})
        historical = run_result.get('historical', [])
        analysis = run_result.get('analysis', {})
        forecast_points = forecast_info.get('forecast', [])

        logger.info(f"[DEBUG] forecast_info keys: {forecast_info.keys()}")
        logger.info(f"[DEBUG] forecast_points type: {type(forecast_points)}, len: {len(forecast_points) if isinstance(forecast_points, list) else 'N/A'}")
        
        if not forecast_points:
            return {"type": "text", "text": "⚠ Aucune prédiction disponible"}

        if historical:
            hist_x_labels = [str(p.get('label', i)) for i, p in enumerate(historical)]
            hist_y = [float(p.get('value', 0)) for p in historical]
        else:
            hist_x_labels, hist_y = [], []

        logger.info(f"[DEBUG] hist_x_labels: {hist_x_labels}")
        logger.info(f"[DEBUG] Building fc_x_labels...")
        fc_x_labels = [str(p.get('label', len(hist_x_labels) + i)) for i, p in enumerate(forecast_points)]
        logger.info(f"[DEBUG] fc_x_labels: {fc_x_labels}")
        fc_y = [float(p.get('value', 0)) for p in forecast_points]
        fc_lower = [float(p.get('lower', 0)) for p in forecast_points]
        fc_upper = [float(p.get('upper', 0)) for p in forecast_points]

        logger.info(f"[DEBUG] fc_y created: {fc_y}")
        logger.info(f"[DEBUG] fc_lower created: {fc_lower}")
        logger.info(f"[DEBUG] fc_upper created: {fc_upper}")

        if hist_x_labels and fc_x_labels:
            link_x = [hist_x_labels[-1], fc_x_labels[0]]
            link_y = [hist_y[-1], fc_y[0]]
        else:
            link_x, link_y = [], []

        logger.info(f"[DEBUG] Creating Plotly figure...")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fc_x_labels + fc_x_labels[::-1],
            y=fc_upper + fc_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 188, 212, 0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f"Intervalle de confiance ({float(forecast_info.get('confidence_level', 0.95))*100:.0f}%)",
            hoverinfo='skip',
            showlegend=True,
        ))

        if hist_x_labels:
            fig.add_trace(go.Scatter(
                x=hist_x_labels,
                y=hist_y,
                name='Historique',
                mode='lines+markers',
                line=dict(color='#1565C0', width=2.5),
                marker=dict(size=5, color='#1565C0'),
                hovertemplate='<b>%{x}</b><br>Valeur réelle: %{y:,.2f}<extra></extra>',
            ))

        if link_x:
            fig.add_trace(go.Scatter(
                x=link_x,
                y=link_y,
                mode='lines',
                line=dict(color='#00BCD4', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.add_trace(go.Scatter(
            x=fc_x_labels,
            y=fc_y,
            name=f"Projection ({forecast_info.get('model', 'Auto')})",
            mode='lines+markers',
            line=dict(color='#00BCD4', width=3),
            marker=dict(size=6, symbol='diamond', color='#00BCD4'),
            hovertemplate='<b>%{x}</b><br>Projection: %{y:,.2f}<extra></extra>',
        ))

        signal_key = run_result.get('signal_key', '')
        short_key = signal_key.split('__')[-1] if '__' in signal_key else signal_key

        fig.update_layout(
            title=dict(
                text=f"📊 {short_key}",
                font=dict(family="DM Sans", size=13, color='#0A2540'),
            ),
            xaxis=dict(title="Période", tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(title="Valeur", tickformat=",.0f"),
            hovermode='x unified',
            height=380,
            template='plotly_white',
            margin=dict(l=60, r=20, t=50, b=60),
            font=dict(family="DM Sans", size=11),
            plot_bgcolor='#F0F4F8',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                font=dict(size=10),
            ),
        )

        if hist_x_labels and fc_x_labels:
            last_hist_x = hist_x_labels[-1]
            # Plotly add_vline cannot handle plain string labels in some axis modes.
            if isinstance(last_hist_x, (int, float)):
                fig.add_vline(
                    x=last_hist_x,
                    line_dash="dash",
                    line_color="rgba(21,101,192,0.4)",
                    annotation_text="Présent",
                    annotation_position="top right",
                    annotation_font=dict(size=9, color="#1565C0"),
                )

        fig_html = fig.to_html(
            include_plotlyjs='cdn',
            div_id=f"sig_{abs(hash(signal_key))}",
            config={"displayModeBar": False},
        )
        
        logger.info(f"[DEBUG] Plotly HTML generated, length: {len(fig_html)}")
        
        # Extract only the <script> and plot <div> from the Plotly HTML
        # This removes the outer HTML/HEAD/BODY structure
        
        # Find the script tag for Plotly JS (if it exists)
        script_match = re.search(r'<script[^>]*>.*?</script>', fig_html, re.DOTALL)
        plotly_script = script_match.group(0) if script_match else ""
        
        # Find the plot div
        div_match = re.search(r'<div id="sig_[^"]*"[^>]*>.*?</div>\s*</body>', fig_html, re.DOTALL)
        if not div_match:
            # Fallback: find any closing div
            div_match = re.search(r'<div id="sig_[^"]*".*?</div>\s*$', fig_html, re.DOTALL)
        
        if div_match:
            fig_html = plotly_script + div_match.group(0).replace('</body>', '')
            logger.info(f"[DEBUG] Extracted plot content, length: {len(fig_html)}")
        else:
            logger.warning("[DEBUG] Could not extract plot div")
            # Use a simple heuristic: find first <div id="sig_ and last </div>
            start_idx = fig_html.find('<div id="sig_')
            end_idx = fig_html.rfind('</div>')
            if start_idx >= 0 and end_idx > start_idx:
                fig_html = fig_html[start_idx:end_idx + 6]
                logger.info(f"[DEBUG] Using simple extraction, length: {len(fig_html)}")

        trend_icon = {"croissante": "📈", "décroissante": "📉"}.get(
            analysis.get('trend', 'stable'), "➡️"
        )
        
        logger.info(f"[DEBUG] analysis dict: {analysis}")
        logger.info(f"[DEBUG] forecast_info dict: {forecast_info}")
        
        # Ensure all values are properly typed before string concatenation
        n_points = str(analysis.get('n_points', len(hist_x_labels)))
        trend_val = str(analysis.get('trend', 'stable'))
        model_val = str(forecast_info.get('model', 'Auto'))
        periods_val = str(forecast_info.get('periods', len(forecast_points)))
        
        logger.info(f"[DEBUG] n_points: {n_points} (type: {type(n_points)})")
        logger.info(f"[DEBUG] trend_val: {trend_val} (type: {type(trend_val)})")
        logger.info(f"[DEBUG] model_val: {model_val} (type: {type(model_val)})")
        logger.info(f"[DEBUG] periods_val: {periods_val} (type: {type(periods_val)})")
        
        # Calculate confidence percentage with explicit float conversion
        confidence_level = forecast_info.get('confidence_level', 0.95)
        try:
            confidence_level = float(confidence_level)
            confidence_pct = f"{confidence_level*100:.0f}%"
        except (ValueError, TypeError):
            confidence_pct = "95%"
        
        logger.info(f"[DEBUG] confidence_pct: {confidence_pct} (type: {type(confidence_pct)})")

        meta_html = (
            '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;padding-top:10px;'
            'border-top:1px solid #BBDEFB;font-size:0.82rem;color:#4A5568;">'
            '<span>🔢 <b>Points historiques :</b> ' + n_points + '</span> '
            '<span>' + str(trend_icon) + ' <b>Tendance :</b> ' + trend_val + '</span> '
            '<span>🤖 <b>Modèle :</b> ' + model_val + '</span> '
            '<span>🎯 <b>Confiance :</b> ' + confidence_pct + '</span> '
            '<span>⏱ <b>Périodes prévues :</b> ' + periods_val + '</span>'
            '</div>'
        )

        return {
            "type": "signal",
            "fig_html": fig_html,
            "meta_html": meta_html
        }

    except Exception as e:
        logger.exception(f"Erreur formatage signal: {e}")
        return {"type": "text", "text": f"✗ Erreur formatage: {str(e)}"}


def handle_signal_mode(query: str):
    """
    Traite les requêtes en mode Signal (Forecast).
    """
    if st.session_state.signal_pipeline is None:
        return {"type": "text", "text": "⚠ Pipeline Forecast non initialisé."}

    pipeline = st.session_state.signal_pipeline

    try:
        parsed_request = parse_signal_request(query)

        if not pipeline.available_signals:
            signal_dir = Path("dataset/signaux")
            if not signal_dir.exists():
                return {"type": "text", "text": "⚠ Répertoire 'dataset/signaux' introuvable."}
            ingest = pipeline.ingest_directory(str(signal_dir))
            if ingest.get("status") != "success" or not pipeline.available_signals:
                return {"type": "text", "text": f"⚠ Aucun signal chargé : {ingest.get('error', 'répertoire vide')}"}

        signal_key = pipeline.find_best_signal(
            data_type=parsed_request.get('data_type'),
            region=parsed_request.get('region'),
        )

        if not signal_key:
            avail = ", ".join(pipeline.available_signals[:5])
            return {"type": "text", "text": f"⚠ Aucun signal trouvé pour cette requête.\n\nSignaux disponibles : {avail}…"}

        logger.info(f"✓ Signal sélectionné : {signal_key}")

        periods = parsed_request.get('periods') or 5
        run_result = pipeline.run(signal_key=signal_key, periods=periods)

        if run_result.get('status') != 'success':
            return {"type": "text", "text": f"✗ Erreur run() : {run_result.get('error', 'Inconnu')}"}

        return format_signal_response(
            query=query,
            parsed_request=parsed_request,
            run_result=run_result,
        )

    except Exception as e:
        logger.exception(f"Erreur handle_signal_mode: {e}")
        return {"type": "text", "text": f"✗ Erreur: {str(e)}"}


# ── Gestion de l'envoi (2 étapes pour afficher le prompt avant génération) ───
if send_btn and user_input.strip():
    ts = get_timestamp()
    st.session_state.messages.append({"role": "user", "text": user_input, "time": ts, "type": "text"})
    st.session_state.msg_count += 1
    st.session_state.pending_prompt = user_input
    st.session_state.pending_mode = current_mode
    if "user_input" in st.session_state:
        del st.session_state.user_input
    st.rerun()

# Traitement du prompt en attente
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pending_prompt
    mode = st.session_state.pending_mode
    st.session_state.pending_prompt = None
    st.session_state.pending_mode = None

    with st.spinner("🌊 BLUE-Gen analyse votre requête…"):
        time.sleep(0.5)

        if mode == "Texte":
            bot_reply = handle_text_mode(prompt)
        elif mode == "Image":
            bot_reply = handle_image_mode(prompt)
        elif mode == "Signal":
            bot_reply = handle_signal_mode(prompt)
        else:
            bot_reply = "Mode inconnu"

    if isinstance(bot_reply, dict):
        st.session_state.messages.append({
            "role": "bot",
            "type": bot_reply.get("type", "text"),
            "text": bot_reply.get("text", ""),
            "html_data": {k: v for k, v in bot_reply.items() if k not in ["type", "text"]},
            "time": get_timestamp()
        })
    else:
        st.session_state.messages.append({
            "role": "bot",
            "type": "text",
            "text": bot_reply,
            "time": get_timestamp()
        })

    if mode == "Image":
        st.session_state.img_count += 1
    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  💧 <strong style="color:#1565C0;">BLUE-Gen</strong> · Assistant Eau Intelligent · Powered by AI Multimodale
  &nbsp;|&nbsp; Langue : Français &nbsp;|&nbsp; v2.4.0
</div>""", unsafe_allow_html=True)
