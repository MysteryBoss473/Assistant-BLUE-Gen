import streamlit as st
import time
import random
import math

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant BLUE-Gen",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
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

/* ── Mode cards ── */
.mode-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin-bottom: 1.6rem; }
.mode-card {
  background: var(--white);
  border: 2px solid var(--blue-pale);
  border-radius: 16px;
  padding: 1.4rem 1.2rem;
  text-align: center;
  cursor: pointer;
  transition: all .25s ease;
  box-shadow: 0 2px 12px rgba(21,101,192,0.07);
}
.mode-card:hover, .mode-card.active {
  border-color: var(--blue-bright);
  background: var(--blue-pale);
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(21,101,192,0.18);
}
.mode-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }
.mode-label {
  font-family: 'Playfair Display', serif;
  font-size: 1rem; font-weight: 700;
  color: var(--blue-deep); margin-bottom: 0.2rem;
}
.mode-desc { font-size: 0.78rem; color: var(--gray-text); }

/* ── Chat container ── */
.chat-wrapper {
  background: var(--gray-soft);
  border-radius: 20px;
  padding: 1.5rem;
  min-height: 320px;
  max-height: 420px;
  overflow-y: auto;
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

/* ── Sidebar themes horizontal ── */
.sidebar-themes {
  display: flex; flex-wrap: wrap; gap: 0.4rem;
}

/* ── Sidebar & main: smooth resize transition ── */
[data-testid="stAppViewContainer"] {
  display: flex;
  min-height: 100vh;
}
section[data-testid="stSidebar"] {
  flex-shrink: 0;
  transition: width 0.25s ease, transform 0.25s ease;
}
.main {
  flex: 1;
  min-width: 0;
  overflow-x: hidden;
  transition: margin-left 0.25s ease;
}

/* ── Mode card click overlay ── */
.mode-card {
  cursor: pointer !important;
}

/* ── Button no wrap (general) ── */
.stButton > button {
  white-space: nowrap !important;
  min-width: fit-content !important;
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

/* ── Scrollbar ── */
.chat-wrapper::-webkit-scrollbar { width: 5px; }
.chat-wrapper::-webkit-scrollbar-track { background: transparent; }
.chat-wrapper::-webkit-scrollbar-thumb { background: var(--blue-sky); border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "👋 Bonjour ! Je suis **BLUE-Gen**, votre assistant intelligent spécialisé en eau. Choisissez un mode ci-dessus et posez-moi votre question.", "time": "maintenant"},
    ]
if "mode" not in st.session_state:
    st.session_state.mode = "Texte"
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0
if "img_count" not in st.session_state:
    st.session_state.img_count = 0

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
            {"role": "bot", "text": "Conversation réinitialisée. Comment puis-je vous aider ?", "time": "maintenant"},
        ]
        st.session_state.msg_count = 0
        st.session_state.img_count = 0
        st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class="hero">
  <div class="hero-drop">💧</div>
  <div class="hero-title">Assistant BLUE-Gen</div>
  <p class="hero-sub">Intelligence artificielle dédiée à l'eau — Texte · Image · Signal</p>
  <div class="hero-badge">🟢 IA connectée &nbsp;|&nbsp; Modèle multimodal v2.4</div>
</div>
""", unsafe_allow_html=True)

# Mode cards
mode_info = {
    "Texte": ("💬", "Génération Texte", "Analyse, rapports, réponses expertes sur l'eau"),
    "Image": ("🖼️", "Génération Image", "Visualisations, cartes, illustrations hydriques"),
    "Signal": ("📡", "Génération Signal", "Signaux, capteurs, données temporelles eau"),
}

st.markdown('<div id="mode-cards-wrapper"></div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, (k, (icon, label, desc)) in enumerate(mode_info.items()):
    active = "active" if st.session_state.mode == k else ""
    with cols[i]:
        st.markdown(f"""
        <div class="mode-card {active}" data-mode="{k}" onclick="document.getElementById('mode_{k}').click()">
          <span class="mode-icon">{icon}</span>
          <div class="mode-label">{label}</div>
          <div class="mode-desc">{desc}</div>
        </div><input type="hidden" id="mode_{k}" />""", unsafe_allow_html=True)

# Charger les valeurs depuis query params
if "mode" in st.query_params:
    new_mode = st.query_params["mode"]
    if new_mode in mode_info and new_mode != st.session_state.mode:
        st.session_state.mode = new_mode

# Ajouter JavaScript pour les clics
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
  const modes = ['Texte', 'Image', 'Signal'];
  modes.forEach(mode => {
    const hidden = document.getElementById('mode_' + mode);
    if (hidden) {
      hidden.addEventListener('click', function() {
        window.location.href = '?mode=' + mode;
      });
    }
  });
});
</script>
""", unsafe_allow_html=True)

# ── Chat area ─────────────────────────────────────────────────────────────────
chat_html = '<div class="chat-wrapper">'
for m in st.session_state.messages:
    role_cls = "user" if m["role"] == "user" else "bot"
    avatar_icon = "👤" if m["role"] == "user" else "💧"
    chat_html += f"""
    <div class="msg-row {role_cls}">
      <div class="avatar {role_cls}">{avatar_icon}</div>
      <div>
        <div class="bubble {role_cls}">{m["text"]}</div>
        <div class="msg-time">{m["time"]}</div>
      </div>
    </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
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
RESPONSES = {
    "Texte": [
        "💧 **Analyse hydrologique complète :** L'eau douce représente moins de 3 % des ressources mondiales, dont les deux tiers sont emprisonnés dans les glaciers. En Afrique subsaharienne, les défis incluent la variabilité climatique, le manque d'infrastructures et la croissance démographique. Des solutions intégrées combinant gestion des bassins versants, technologies de collecte d'eau de pluie et gouvernance participative sont essentielles.",
        "🌊 **Réponse BLUE-Gen :** Le cycle hydrologique est un système dynamique où l'évaporation, la condensation, les précipitations et le ruissellement s'enchaînent en boucle. Les changements climatiques perturbent ce cycle en intensifiant les événements extrêmes — sécheresses prolongées et crues soudaines. La modélisation hydrologique aide à anticiper ces risques.",
        "📘 **Information technique :** La qualité de l'eau se mesure par de nombreux paramètres : pH, turbidité, conductivité, teneur en nitrates, coliformes fécaux, métaux lourds. Les normes OMS recommandent un pH entre 6,5 et 8,5 et une turbidité inférieure à 1 NTU pour l'eau potable.",
    ],
    "Image": [
        "🖼️ **Image générée :** La visualisation de vos ressources hydriques est en cours de rendu. Une carte thermique en fausses couleurs montrant la distribution spatiale des précipitations annuelles et des aquifères souterrains sera disponible. *(Connectez le backend pour activer la génération réelle.)*",
        "🗺️ **Rendu cartographique :** Le schéma du bassin versant avec les isohyètes, les zones de recharge des nappes et les points de prélèvement est généré. La palette bleue indique les zones à forte disponibilité hydrique.",
    ],
    "Signal": [
        "📡 **Signal généré — Débit fluvial :** Les données temporelles montrent une série de 720 points sur 30 jours avec une valeur moyenne de **142 m³/s**, un pic à **387 m³/s** (jour 12, crue saisonnière) et un étiage à **68 m³/s** (jour 28). Fréquence d'échantillonnage : 1 h. Format : JSON/CSV disponibles.",
        "📊 **Signal hydrométrique :** Analyse spectrale FFT réalisée — fréquence dominante à **0,034 Hz** (cycle semi-diurne maréal). Corrélation avec les précipitations amont : **r = 0,87**. Anomalies détectées : 2 pics de crue impulsionnelle aux jours 8 et 19.",
    ],
}

def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%H:%M")

if send_btn and user_input.strip():
    ts = get_timestamp()
    st.session_state.messages.append({"role": "user", "text": user_input, "time": ts})
    st.session_state.msg_count += 1

    with st.spinner("🌊 BLUE-Gen analyse votre requête…"):
        time.sleep(1.2)

    bot_reply = random.choice(RESPONSES[current_mode])
    st.session_state.messages.append({"role": "bot", "text": bot_reply, "time": get_timestamp()})
    st.session_state.img_count += 1
    st.rerun()

# ── Signal visualizer (shown when Signal mode) ────────────────────────────────
if current_mode == "Signal" and len(st.session_state.messages) > 1:
    st.markdown("---")
    st.markdown("**📡 Aperçu Signal en direct**")
    bars_html = '<div style="display:flex;align-items:flex-end;height:80px;gap:3px;background:var(--blue-pale);padding:12px;border-radius:14px;border:1px solid var(--blue-sky);">'
    for i in range(40):
        h = int(20 + 45 * abs(math.sin(i * 0.5)) + random.randint(-5, 5))
        delay = f"{(i * 0.05):.2f}s"
        bars_html += f'<div class="signal-bar" style="height:{h}px;animation-delay:{delay};"></div>'
    bars_html += "</div>"
    st.markdown(bars_html, unsafe_allow_html=True)

# ── Image preview (shown when Image mode) ────────────────────────────────────
if current_mode == "Image" and len(st.session_state.messages) > 1:
    st.markdown("---")
    st.markdown("**🖼️ Zone de prévisualisation**")
    st.markdown("""
    <div class="img-placeholder">
      <div style="font-size:3rem;margin-bottom:0.8rem;">🗺️</div>
      <div style="font-size:1rem;font-weight:600;color:#1565C0;">Saisissez une description pour générer une image</div>
      <div style="font-size:0.8rem;color:#666;margin-top:0.4rem;">Cartes · Schémas · Infographies · Visualisations hydriques</div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  💧 <strong style="color:#1565C0;">BLUE-Gen</strong> · Assistant Eau Intelligent · Powered by AI Multimodale
  &nbsp;|&nbsp; Langue : Français &nbsp;|&nbsp; v2.4.0
</div>""", unsafe_allow_html=True)