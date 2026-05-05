import streamlit as st
import time
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
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        except Exception as e:
            logging.warning(f"Could not auto-install packages: {e}")

ensure_dependencies()

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODULES_AVAILABLE = True
try:
    from core.signal_forecast import SignalForecastPipeline
except ImportError as e:
    logger.warning(f"⚠ SignalForecastPipeline non disponible: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Assistant BLUE-Gen",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--blue-deep); }
.main { background: var(--white); }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--blue-deep) 0%, #0D3461 60%, #0A2540 100%);
  border-right: 2px solid var(--blue-mid);
  min-width: 220px; max-width: 420px;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }
.hero {
  background: linear-gradient(135deg, var(--blue-deep) 0%, var(--blue-mid) 50%, var(--blue-bright) 100%);
  border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 1.8rem;
  position: relative; overflow: hidden; box-shadow: 0 12px 40px rgba(10,37,64,0.25);
}
.hero::before {
  content:''; position:absolute; inset:0;
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 200'%3E%3Cpath fill='rgba(255,255,255,0.06)' d='M0,80 C360,160 720,0 1080,80 C1260,120 1380,60 1440,80 L1440,200 L0,200Z'/%3E%3C/svg%3E") bottom/cover no-repeat;
}
.hero-title { font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900; color:var(--white); line-height:1.1; margin:0 0 0.4rem; text-shadow:0 2px 20px rgba(0,0,0,0.3); }
.hero-sub { font-size:1rem; color:var(--blue-sky); margin:0; letter-spacing:0.04em; font-weight:300; }
.hero-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3); color:white; font-size:0.75rem; font-weight:500; padding:4px 12px; border-radius:20px; margin-top:1rem; backdrop-filter:blur(8px); }
.hero-drop { position:absolute; right:3rem; top:50%; transform:translateY(-50%); font-size:7rem; opacity:0.12; user-select:none; }
.stSelectbox div[data-baseweb="select"] input { pointer-events:none !important; caret-color:transparent !important; }
.msg-row { display:flex; gap:12px; margin-bottom:0.8rem; align-items:flex-start; }
.msg-row.user { flex-direction:row-reverse; }
.avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0; }
.avatar.bot { background:linear-gradient(135deg,var(--blue-mid),var(--accent)); color:white; }
.avatar.user { background:var(--blue-pale); color:var(--blue-deep); }
.bubble { max-width:72%; padding:0.85rem 1.1rem; border-radius:18px; font-size:0.93rem; line-height:1.55; box-shadow:0 2px 10px rgba(0,0,0,0.07); word-break:break-word; }
.bubble.bot { background:var(--white); color:var(--blue-deep); border-bottom-left-radius:4px; border:1px solid var(--blue-pale); }
.bubble.user { background:linear-gradient(135deg,var(--blue-mid),var(--blue-bright)); color:white; border-bottom-right-radius:4px; }
.msg-time { font-size:0.68rem; color:#999; margin-top:4px; }
.signal-header { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
.signal-avatar { width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#1565C0,#00BCD4); display:flex; align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0; color:white; }
.signal-title { font-size:0.95rem; color:#0A2540; font-weight:bold; }
.signal-meta-bar { display:flex; gap:16px; flex-wrap:wrap; padding-top:8px; border-top:1px solid #BBDEFB; font-size:0.82rem; color:#4A5568; margin-top:6px; }
.signal-time { font-size:0.68rem; color:#999; margin-top:4px; margin-left:46px; }
.stTextInput > div > div > input { border:2px solid var(--blue-sky) !important; border-radius:50px !important; padding:0.75rem 1.4rem !important; font-family:'DM Sans',sans-serif !important; font-size:0.95rem !important; background:white !important; color:var(--blue-deep) !important; box-shadow:0 2px 12px rgba(21,101,192,0.08) !important; }
.stTextInput > div > div > input:focus { border-color:var(--blue-bright) !important; box-shadow:0 4px 20px rgba(30,136,229,0.2) !important; outline:none !important; }
.stButton > button { background:linear-gradient(135deg,var(--blue-mid),var(--blue-bright)) !important; color:white !important; border:none !important; border-radius:50px !important; padding:0.65rem 2rem !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; cursor:pointer !important; transition:all .25s ease !important; box-shadow:0 4px 15px rgba(21,101,192,0.3) !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 25px rgba(21,101,192,0.4) !important; }
.stSelectbox > div > div { border:2px solid var(--blue-sky) !important; border-radius:12px !important; background:white !important; }
.sidebar-logo { text-align:center; padding:1.6rem 1rem 1rem; }
.sidebar-logo-text { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:900; color:white; }
.sidebar-logo-sub { font-size:0.72rem; color:var(--blue-light); letter-spacing:0.1em; text-transform:uppercase; }
.sidebar-divider { border:none; border-top:1px solid rgba(255,255,255,0.12); margin:0.8rem 1rem; }
.sidebar-section { padding:0.4rem 1rem 0.2rem; font-size:0.7rem; color:var(--blue-light); text-transform:uppercase; letter-spacing:0.12em; }
.stat-card { background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:0.9rem 1rem; margin:0.4rem 0; color:white; text-align:center; }
.stat-num { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:700; color:var(--blue-light); }
.stat-label { font-size:0.72rem; color:rgba(255,255,255,0.6); margin-top:2px; }
.chip { display:inline-flex; align-items:center; gap:5px; background:var(--blue-pale); color:var(--blue-mid); border:1px solid var(--blue-sky); border-radius:20px; padding:3px 10px; font-size:0.75rem; font-weight:500; margin:2px; }
.app-footer { background:linear-gradient(90deg,rgba(10,37,64,0.04) 0%,rgba(21,101,192,0.07) 50%,rgba(10,37,64,0.04) 100%); border-top:1px solid var(--blue-sky); border-radius:10px; padding:0.35rem 1rem; margin-top:0.4rem; text-align:center; color:#7a90a8; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
LORA_ADAPTER_DIR = Path("lancement/lora_adapter")
SIGNAL_DATA_DIR  = Path("dataset/signaux")


# ── Cached pipelines ───────────────────────────────────────────────────────────
@st.cache_resource
def get_rag_pipeline():
    try:
        from core.text_rag import TextRAGPipeline
        return TextRAGPipeline(), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_image_pipeline():
    try:
        from core.image_lora import ImageLoRAPipeline
        pipeline = ImageLoRAPipeline()
        merged = LORA_ADAPTER_DIR / "merged_pipeline"
        if merged.exists() and any(merged.iterdir()):
            res = pipeline.load_merged_model(str(LORA_ADAPTER_DIR / "merged_model"))
            if res.get("status") == "success":
                return pipeline, None
        if LORA_ADAPTER_DIR.exists() and any(LORA_ADAPTER_DIR.iterdir()):
            if not (LORA_ADAPTER_DIR / "merged_model_metadata.json").exists():
                pipeline.load_lora_adapter(str(LORA_ADAPTER_DIR))
        return pipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_signal_pipeline():
    try:
        from core.signal_forecast import SignalForecastPipeline
        pipeline = SignalForecastPipeline()
        if SIGNAL_DATA_DIR.exists():
            first = next(SIGNAL_DATA_DIR.glob("*.xlsx"), None)
            if first:
                res = pipeline.load_signal(str(first))
                if res.get("status") == "success":
                    pipeline.loaded_signal_file = str(first)
        return pipeline, None
    except Exception as e:
        return None, str(e)


# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "messages": [
        {"role": "bot", "text": "Bonjour ! Je suis BLUE-Gen, votre assistant intelligent spécialisé en eau. Choisissez un mode ci-dessus et posez-moi votre question.", "time": "maintenant", "type": "text"},
    ],
    "mode": "Texte",
    "msg_count": 0,
    "img_count": 0,
    "rag_pipeline": None, "image_pipeline": None, "signal_pipeline": None,
    "rag_error": None,    "image_error": None,    "signal_error": None,
    "last_generated_image": None, "last_image_prompt": None, "last_image_tokens": [],
    "pending_prompt": None, "pending_mode": None,
    "pipelines_initialized": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.pipelines_initialized:
    st.session_state.pipelines_initialized = True
    if st.session_state.rag_pipeline is None:
        p, e = get_rag_pipeline()
        st.session_state.rag_pipeline = p; st.session_state.rag_error = e
    if st.session_state.image_pipeline is None:
        p, e = get_image_pipeline()
        st.session_state.image_pipeline = p; st.session_state.image_error = e
    if st.session_state.signal_pipeline is None:
        p, e = get_signal_pipeline()
        st.session_state.signal_pipeline = p; st.session_state.signal_error = e


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
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
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{st.session_state.msg_count}</div><div class="stat-label">Messages</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{st.session_state.img_count}</div><div class="stat-label">Générations</div></div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Thèmes eau disponibles</div>', unsafe_allow_html=True)
    topics = ["Hydrologie", "Eau potable", "Irrigation", "Océanographie", "Traitement des eaux", "Signaux hydro"]
    st.markdown(''.join([f'<div class="chip">{t}</div>' for t in topics]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = [{"role": "bot", "text": "Conversation réinitialisée. Comment puis-je vous aider ?", "time": "maintenant", "type": "text"}]
        st.session_state.msg_count = 0; st.session_state.img_count = 0
        st.session_state.pending_prompt = None; st.session_state.pending_mode = None
        if "user_input" in st.session_state: del st.session_state.user_input
        st.rerun()


# ── MAIN HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-drop">💧</div>
  <div class="hero-title">Assistant BLUE-Gen</div>
  <p class="hero-sub">Intelligence artificielle dédiée à l'eau — Texte · Image · Signal</p>
  <div class="hero-badge">🟢 IA connectée &nbsp;|&nbsp; Modèle multimodal v2.4</div>
</div>
""", unsafe_allow_html=True)

mode_options = {"💬 Génération Texte": "Texte", "🖼️ Génération Image": "Image", "📡 Génération Signal": "Signal"}
col1, col2 = st.columns([2, 3])
with col1:
    sel = st.selectbox("Type de génération", list(mode_options.keys()),
                       index=list(mode_options.values()).index(st.session_state.mode),
                       label_visibility="collapsed", key="mode_selector")
    new_mode = mode_options[sel]
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT DISPLAY
#
#  SOLUTION DÉFINITIVE pour st.plotly_chart dans une boucle :
#
#  On N'utilise PAS de st.container() global autour de tous les messages.
#  Chaque message signal crée ses propres éléments Streamlit directement
#  dans le scope principal, sans aucun wrapper HTML ouvert.
#
#  La carte bleue est simulée via st.columns() (indentation) + CSS appliqué
#  sur le container Streamlit natif avec st.markdown + border trick.
# ══════════════════════════════════════════════════════════════════════════════

for idx, m in enumerate(st.session_state.messages):
    msg_type = m.get("type", "text")

    # ── Utilisateur ───────────────────────────────────────────────────────
    if m["role"] == "user":
        st.markdown(
            f'<div class="msg-row user">'
            f'  <div class="avatar user">👤</div>'
            f'  <div>'
            f'    <div class="bubble user">{m["text"]}</div>'
            f'    <div class="msg-time">{m["time"]}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Bot : image ───────────────────────────────────────────────────────
    elif msg_type == "image":
        data           = m.get("html_data", {})
        img_base64     = data.get("base64", "")
        prompt         = data.get("prompt", "")
        tokens         = data.get("tokens", [])
        is_placeholder = data.get("is_placeholder", False)
        source_label   = "Image placeholder" if is_placeholder else "Image générée avec LoRA Fine-tuned"
        prompt_display = prompt[:80] + ("…" if len(prompt) > 80 else "")
        tokens_html    = f'<div style="font-size:0.8rem;color:#666;margin-top:8px;">Tokens reconnus : {", ".join(tokens)}</div>' if tokens else ""
        st.markdown(
            f'<div class="msg-row bot">'
            f'  <div class="avatar bot">💧</div>'
            f'  <div>'
            f'    <div style="border-radius:14px;padding:12px;background:#E3F2FD;border:2px solid #1565C0;max-width:440px;">'
            f'      <div style="font-size:0.9rem;color:#0A2540;margin-bottom:8px;"><strong>🎨 {source_label}</strong><br>'
            f'        <em style="font-size:0.85rem;">Prompt : {prompt_display}</em></div>'
            f'      <img src="data:image/png;base64,{img_base64}" style="width:100%;max-width:400px;border-radius:10px;border:1px solid #1E88E5;display:block;">'
            f'      {tokens_html}'
            f'    </div>'
            f'    <div class="msg-time">{m["time"]}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Bot : signal ──────────────────────────────────────────────────────
    # STRATÉGIE FINALE :
    # • En-tête  → st.markdown() fermé
    # • Graphique → st.columns() + st.plotly_chart() sans aucun HTML ouvert
    # • Méta      → st.markdown() fermé
    # • Timestamp → st.markdown() fermé
    # Zéro div ouverte avant plotly_chart dans le même appel markdown.
    elif msg_type == "signal":
        data      = m.get("html_data", {})
        meta_html = data.get("meta_html", "")
        fig_dict  = data.get("fig_dict")

        # 1) En-tête — entièrement fermé
        st.markdown(
            '<div class="signal-header">'
            '  <div class="signal-avatar">💧</div>'
            '  <div class="signal-title">📡 Analyse &amp; Prédiction de Signal</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # 2) Graphique dans une colonne indentée
        #    Pas de st.markdown() ouvert avant cet appel
        gap_col, chart_col = st.columns([0.05, 0.95])
        with chart_col:
            if fig_dict:
                try:
                    fig = go.Figure(fig_dict)
                    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=360)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"sig_{idx}",
                        config={"displayModeBar": False},
                    )
                except Exception as e:
                    st.warning(f"⚠ Graphique indisponible : {e}")
            else:
                st.info("⚠ Aucune donnée graphique.")

            # 3) Métadonnées — div fermée, dans la même colonne
            st.markdown(
                f'<div class="signal-meta-bar">{meta_html}</div>',
                unsafe_allow_html=True,
            )

        # 4) Timestamp — hors de la colonne, div fermée
        st.markdown(
            f'<div class="signal-time">{m["time"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Bot : texte ───────────────────────────────────────────────────────
    else:
        st.markdown(
            f'<div class="msg-row bot">'
            f'  <div class="avatar bot">💧</div>'
            f'  <div>'
            f'    <div class="bubble bot">{m["text"]}</div>'
            f'    <div class="msg-time">{m["time"]}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Input row ──────────────────────────────────────────────────────────────────
current_mode = st.session_state.mode
placeholders = {
    "Texte":  "Ex : Quels sont les enjeux de la gestion de l'eau en Afrique subsaharienne ?",
    "Image":  "Ex : Carte de distribution des ressources en eau souterraine au Sahel…",
    "Signal": "Ex : Générer un signal de débit du fleuve Niger sur 30 jours…",
}
inp_col, btn_col = st.columns([4, 1])
with inp_col:
    user_input = st.text_input("Votre question", placeholder=placeholders[current_mode],
                               key="user_input", label_visibility="collapsed")
with btn_col:
    send_btn = st.button("Envoyer ➤", use_container_width=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_timestamp():
    return datetime.now().strftime("%H:%M")


def handle_text_mode(query: str):
    if st.session_state.rag_pipeline is None:
        return f"⚠ Pipeline RAG non initialisé.\n\nErreur : {st.session_state.rag_error}"
    try:
        docs = st.session_state.rag_pipeline.retrieve(query, k=5)
        if not docs:
            return "⚠ Aucun document pertinent trouvé. Assurez-vous d'avoir indexé des PDFs."
        result = st.session_state.rag_pipeline.generate_response(query)
        return result['response'] + (result.get('citations') or "")
    except Exception as e:
        logger.exception(e); return f"✗ Erreur : {e}"


def handle_image_mode(query: str):
    if st.session_state.image_pipeline is None:
        return {"type": "text", "text": f"⚠ Pipeline Image non initialisé.\n\nErreur : {st.session_state.image_error}"}
    try:
        result = st.session_state.image_pipeline.generate_image(query)
        if result.get('status') == 'error':
            return {"type": "text", "text": f"✗ {result.get('error', 'Erreur inconnue')}"}
        imgs = result.get('generated_images', [])
        if not imgs:
            return {"type": "text", "text": "⚠ Aucune image générée"}
        b64 = imgs[0].get('base64')
        if not b64:
            return {"type": "text", "text": "✗ Données image vides"}
        st.session_state.last_generated_image = b64
        st.session_state.last_image_prompt    = query
        st.session_state.last_image_tokens    = result.get('tokens_used', [])
        return {"type": "image", "base64": b64, "prompt": query,
                "tokens": result.get('tokens_used', []),
                "is_placeholder": imgs[0].get('format') == 'png' and len(b64) < 20000}
    except Exception as e:
        logger.exception(e); return {"type": "text", "text": f"✗ Erreur : {e}"}


def parse_signal_request(query: str) -> Dict[str, Any]:
    q = query.lower()
    parsed: Dict[str, Any] = {'data_type': None, 'region': None, 'periods': None, 'time_unit': 'annuelle'}
    if any(t in q for t in ['vente', 'ventes']):           parsed['data_type'] = 'ventes'
    elif any(t in q for t in ['production']):              parsed['data_type'] = 'production'
    elif any(t in q for t in ['abonne', 'abonnement']):    parsed['data_type'] = 'abonnes'
    elif any(t in q for t in ['qualite', 'qualité']):      parsed['data_type'] = 'qualite'
    for r in ['centre-est','centre-nord','centre-ouest','centre-sud','sud-ouest',
              'hauts-bassins','plateau central','sahel','centre','nord','sud','est','ouest']:
        if r in q:
            parsed['region'] = r.title(); break
    match = re.search(r'(\d+)\s*(prochaines?\s*)?(ans?|années?|mois|semaines?|jours?)', q)
    if match:
        n, unit = int(match.group(1)), match.group(3) or ""
        if any(t in unit for t in ['an','ans','année']):  parsed['periods'] = n; parsed['time_unit'] = 'annuelle'
        elif 'mois' in unit:                              parsed['periods'] = n; parsed['time_unit'] = 'mensuelle'
        elif 'semaine' in unit:                           parsed['periods'] = n; parsed['time_unit'] = 'hebdomadaire'
        elif 'jour' in unit:                              parsed['periods'] = n; parsed['time_unit'] = 'quotidienne'
    return parsed


def _safe_float(v, default=0.0):
    """Convertit tout type numérique (numpy, pandas…) en float Python natif."""
    try:
        return float(v)
    except Exception:
        return default


def format_signal_response(query: str, parsed_request: Dict, run_result: Dict):
    if run_result.get('status') != 'success':
        return {"type": "text", "text": f"✗ Erreur prédiction : {run_result.get('error', 'Inconnu')}"}
    try:
        forecast_info   = run_result.get('forecast', {})
        historical      = run_result.get('historical', [])
        analysis        = run_result.get('analysis', {})
        forecast_points = forecast_info.get('forecast', [])

        if not forecast_points:
            return {"type": "text", "text": "⚠ Aucune prédiction disponible"}

        # Conversion explicite → types Python natifs (évite erreurs numpy/pandas)
        hist_x = [str(p.get('label', i)) for i, p in enumerate(historical)]
        hist_y = [_safe_float(p.get('value', 0)) for p in historical]
        fc_x   = [str(p.get('label', len(hist_x) + i)) for i, p in enumerate(forecast_points)]
        fc_y   = [_safe_float(p.get('value', 0)) for p in forecast_points]
        fc_lo  = [_safe_float(p.get('lower', 0)) for p in forecast_points]
        fc_hi  = [_safe_float(p.get('upper', 0)) for p in forecast_points]
        link_x = [hist_x[-1], fc_x[0]] if hist_x and fc_x else []
        link_y = [hist_y[-1], fc_y[0]] if hist_x and fc_x else []

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc_x + fc_x[::-1], y=fc_hi + fc_lo[::-1],
            fill='toself', fillcolor='rgba(0,188,212,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f"Intervalle {_safe_float(forecast_info.get('confidence_level', 0.95))*100:.0f}%",
            hoverinfo='skip', showlegend=True,
        ))
        if hist_x:
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y, name='Historique', mode='lines+markers',
                line=dict(color='#1565C0', width=2.5), marker=dict(size=5, color='#1565C0'),
                hovertemplate='<b>%{x}</b><br>Valeur : %{y:,.2f}<extra></extra>',
            ))
        if link_x:
            fig.add_trace(go.Scatter(
                x=link_x, y=link_y, mode='lines',
                line=dict(color='#00BCD4', width=2, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ))
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            name=f"Projection ({forecast_info.get('model', 'Auto')})",
            mode='lines+markers',
            line=dict(color='#00BCD4', width=3),
            marker=dict(size=6, symbol='diamond', color='#00BCD4'),
            hovertemplate='<b>%{x}</b><br>Projection : %{y:,.2f}<extra></extra>',
        ))

        signal_key = run_result.get('signal_key', '')
        short_key  = signal_key.split('__')[-1] if '__' in signal_key else signal_key
        fig.update_layout(
            title=dict(text=f"📊 {short_key}", font=dict(family="DM Sans", size=13, color='#0A2540')),
            xaxis=dict(title="Période", tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(title="Valeur", tickformat=",.0f"),
            hovermode='x unified', height=360, template='plotly_white',
            margin=dict(l=60, r=20, t=50, b=60),
            font=dict(family="DM Sans", size=11),
            plot_bgcolor='#F0F4F8', paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        )

        if hist_x and fc_x:
            try:
                fig.add_vline(x=float(hist_x[-1]), line_dash="dash",
                              line_color="rgba(21,101,192,0.4)",
                              annotation_text="Présent", annotation_position="top right",
                              annotation_font=dict(size=9, color="#1565C0"))
            except (ValueError, TypeError):
                pass

        fig_dict = fig.to_dict()

        trend_icon = {"croissante": "📈", "décroissante": "📉"}.get(analysis.get('trend', ''), "➡️")
        try:
            conf_pct = f"{_safe_float(forecast_info.get('confidence_level', 0.95))*100:.0f}%"
        except Exception:
            conf_pct = "95%"

        meta_html = (
            f'<span>🔢 <b>Points historiques :</b> {analysis.get("n_points", len(hist_x))}</span> '
            f'<span>{trend_icon} <b>Tendance :</b> {analysis.get("trend", "stable")}</span> '
            f'<span>🤖 <b>Modèle :</b> {forecast_info.get("model", "Auto")}</span> '
            f'<span>🎯 <b>Confiance :</b> {conf_pct}</span> '
            f'<span>⏱ <b>Périodes prévues :</b> {forecast_info.get("periods", len(forecast_points))}</span>'
        )

        return {"type": "signal", "fig_dict": fig_dict, "meta_html": meta_html}

    except Exception as e:
        logger.exception(e)
        return {"type": "text", "text": f"✗ Erreur formatage : {e}"}


def handle_signal_mode(query: str):
    if st.session_state.signal_pipeline is None:
        return {"type": "text", "text": "⚠ Pipeline Forecast non initialisé."}
    pipeline = st.session_state.signal_pipeline
    try:
        parsed = parse_signal_request(query)
        if not pipeline.available_signals:
            if not SIGNAL_DATA_DIR.exists():
                return {"type": "text", "text": "⚠ Répertoire 'dataset/signaux' introuvable."}
            ingest = pipeline.ingest_directory(str(SIGNAL_DATA_DIR))
            if ingest.get("status") != "success" or not pipeline.available_signals:
                return {"type": "text", "text": f"⚠ Aucun signal chargé : {ingest.get('error', 'répertoire vide')}"}
        signal_key = pipeline.find_best_signal(data_type=parsed.get('data_type'), region=parsed.get('region'))
        if not signal_key:
            avail = ", ".join(pipeline.available_signals[:5])
            return {"type": "text", "text": f"⚠ Aucun signal trouvé.\n\nDisponibles : {avail}…"}
        run_result = pipeline.run(signal_key=signal_key, periods=parsed.get('periods') or 5)
        if run_result.get('status') != 'success':
            return {"type": "text", "text": f"✗ Erreur run() : {run_result.get('error', 'Inconnu')}"}
        return format_signal_response(query=query, parsed_request=parsed, run_result=run_result)
    except Exception as e:
        logger.exception(e)
        return {"type": "text", "text": f"✗ Erreur : {e}"}


# ── Envoi ──────────────────────────────────────────────────────────────────────
if send_btn and user_input.strip():
    st.session_state.messages.append({"role": "user", "text": user_input, "time": get_timestamp(), "type": "text"})
    st.session_state.msg_count += 1
    st.session_state.pending_prompt = user_input
    st.session_state.pending_mode   = current_mode
    if "user_input" in st.session_state: del st.session_state.user_input
    st.rerun()

# ── Traitement pending ─────────────────────────────────────────────────────────
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pending_prompt
    mode   = st.session_state.pending_mode
    st.session_state.pending_prompt = None
    st.session_state.pending_mode   = None

    with st.spinner("🌊 BLUE-Gen analyse votre requête…"):
        time.sleep(0.3)
        if mode == "Texte":    bot_reply = handle_text_mode(prompt)
        elif mode == "Image":  bot_reply = handle_image_mode(prompt)
        elif mode == "Signal": bot_reply = handle_signal_mode(prompt)
        else:                  bot_reply = "Mode inconnu"

    if isinstance(bot_reply, dict):
        st.session_state.messages.append({
            "role":      "bot",
            "type":      bot_reply.get("type", "text"),
            "text":      bot_reply.get("text", ""),
            "html_data": {k: v for k, v in bot_reply.items() if k not in ["type", "text"]},
            "time":      get_timestamp(),
        })
    else:
        st.session_state.messages.append({"role": "bot", "type": "text", "text": bot_reply, "time": get_timestamp()})

    if mode == "Image": st.session_state.img_count += 1
    st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  💧 <strong style="color:#1565C0;">BLUE-Gen</strong> · Assistant Eau Intelligent · Powered by AI Multimodale
  &nbsp;|&nbsp; Langue : Français &nbsp;|&nbsp; v2.4.0
</div>""", unsafe_allow_html=True)