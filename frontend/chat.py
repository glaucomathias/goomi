import os
import uuid
import base64
import json
import re
from collections import Counter
import requests
import streamlit as st
from datetime import datetime, date  # >>> já presente

# =========================
# Page config
# =========================
BASE_DIR = os.path.dirname(__file__)
IMG_PATH = os.path.join(BASE_DIR, "goomi.png")
PAGE_TITLE = "GOOMIH"
PAGE_ICON  = "🤖"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# =========================
# Configs básicas
# =========================
API_BASE   = os.getenv("GOOMI_API_BASE", "http://127.0.0.1:5000")
ASK_URL    = f"{API_BASE}/ask"
HEALTH_URL = f"{API_BASE}/health"

APP_NAME   = "Goomih"
SUBTITLE   = "Assistente virtual da nossa família"
HERO_WIDTH = 120  # logo um pouco menor para não forçar rolagem

USERS = {
    "giulia":    {"label": "Giulia",    "avatar": "😊", "color": "pink"},
    "guilherme": {"label": "Guilherme", "avatar": "😎", "color": "blue"},
    "giovanna":  {"label": "Giovanna",  "avatar": "😊", "color": "pink"},
    "helena":    {"label": "Helena",    "avatar": "😊", "color": "pink"},
    "glauco":    {"label": "Glauco",    "avatar": "😎", "color": "blue"},
}

# ====== Perguntas rápidas (por usuário) ======
DEFAULT_QUICK = {
    "glauco": [
        "Quais são os jogos da Série B hoje?",
        "Faça as projeções da Série B",
        "Quero as projeções da Série B do dia 30/08/2025",
        "Liste minhas últimas consultas (Helena)",
    ],
    "guilherme": [
        "Quais são os melhores animes?",
        "Quais são os melhores jogos do ano?",
        "Quais são as notícias de tecnologia?",
        "Mostre curiosidades de ciência",
    ],
    "giulia": [
        "Me conte uma curiosidade legal!",
        "Ideias de filmes para assistir em família",
        "Qual é a capital de cada estado do Brasil?",
        "Explique frações de forma simples",
    ],
    "giovanna": [
        "Dicas para organizar os estudos",
        "Me ajude com um resumo de história",
        "Filmes/series de suspense recomendados",
        "Curiosidades de biologia",
    ],
    "helena": [
        "Liste minhas últimas consultas (Helena)",
        "Resumo das notícias do dia",
        "Dicas de receitas para o jantar",
        "Como está a previsão do tempo hoje?",
    ],
}
QUICK_STATS_PATH = os.path.join(BASE_DIR, "quick_stats.json")

def _load_quick_stats() -> dict:
    if not os.path.exists(QUICK_STATS_PATH):
        return {}
    try:
        with open(QUICK_STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {u: {q: int(c) for q, c in d.items()} for u, d in data.items()}
    except Exception:
        return {}

def _save_quick_stats(stats: dict) -> None:
    try:
        with open(QUICK_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _top4_for(user: str) -> list[str]:
    """Retorna top-4 perguntas do usuário, caindo para as sementes se não houver histórico."""
    stats = _load_quick_stats()
    user_counts = Counter(stats.get(user, {}))
    if not user_counts:
        return DEFAULT_QUICK.get(user, DEFAULT_QUICK["glauco"])[:4]
    top = [q for q, _ in user_counts.most_common(4)]
    seeds = DEFAULT_QUICK.get(user, DEFAULT_QUICK["glauco"])
    for s in seeds:
        if len(top) >= 4:
            break
        if s not in top:
            top.append(s)
    return top[:4]

def _bump_quick_usage(user: str, question: str) -> None:
    stats = _load_quick_stats()
    stats.setdefault(user, {})
    stats[user][question] = int(stats[user].get(question, 0)) + 1
    _save_quick_stats(stats)

# =========================
# Estado de sessão
# =========================
def init_state():
    ss = st.session_state
    ss.setdefault("logged_in", False)
    ss.setdefault("client_id", None)
    ss.setdefault("session_id", str(uuid.uuid4()))
    ss.setdefault("show_user_switcher", False)
    ss.setdefault("chats", {})
    ss.setdefault("current_chat_id", None)

    if not ss["chats"]:
        cid = str(uuid.uuid4())
        ss["chats"][cid] = {"name": "Bem-vindo 👋", "messages": []}
        ss["current_chat_id"] = cid

init_state()

# =========================
# Estilos (centralização real + login estreito)
# =========================
st.markdown(f"""
<style>
:root {{
  --bg: #FFFFFF;
  --fg: #111827;
  --sub: #6B7280;
  --card: #F3F4F6;
  --ok: #22C55E;
  --bad:#EF4444;
  --content-max: 900px;
  --content-narrow: 420px;
}}

.stApp {{
  background: var(--bg) !important;
  color: var(--fg) !important;
}}

.block-container {{ padding-top: 1.2rem; }}

/* containers centrais */
.center-wrap {{ max-width: var(--content-max); margin: 0 auto; }}
.narrow-wrap {{ max-width: var(--content-narrow); margin: 0 auto; }}

/* HERO */
.hero-wrap {{ max-width: var(--content-max); margin: 0 auto; text-align: center; }}
.hero-title {{ font-size: 40px; margin: 0 0 .25rem 0; }}
.hero-sub   {{ font-size: 20px; font-weight: 600; color: var(--sub); margin: 0; }}
.hero-logo  {{ margin-top: 8px; }}
.hero-logo img {{
  width: {HERO_WIDTH}px !important;
  max-width: {HERO_WIDTH}px !important;
  display: inline-block !important;
}}

/* Card de status */
.pill {{
  display:inline-block; padding:6px 10px; border-radius:999px;
  background: var(--card); color: var(--fg);
  border: 1px solid rgba(0,0,0,0.05);
  font-weight:600; font-size:12px;
}}
.pill.ok {{ background: rgba(34,197,94,0.15); color: var(--ok); }}
.pill.bad{{ background: rgba(239,68,68,0.15); color: var(--bad); }}

/* Banner */
.shadow-card {{
  background: var(--card);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 10px 14px;
  color: var(--fg);
  text-align:center;
}}

/* Sidebar compacta */
[data-testid="stSidebar"] * {{ font-size: 14px !important; }}
[data-testid="stSidebar"] .stButton button {{
  font-size: 12px !important; padding: 6px 10px !important; white-space: nowrap;
}}

/* Quick prompts menores */
.quickbar button {{
  margin: 4px; border-radius: 999px;
  font-size: 12px !important; padding: 6px 10px !important;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}

/* Garantir que o primeiro título da página (topo) fique centralizado */
.block-container h1:first-of-type {{ text-align: center; }}
</style>
""", unsafe_allow_html=True)

# =========================
# Utils
# =========================
def api_health_ok() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=4)
        return r.ok
    except Exception:
        return False

def ask_backend(question: str) -> str:
    payload = {"client_id": st.session_state.client_id, "question": question}
    try:
        r = requests.post(ASK_URL, json=payload, timeout=60)
        if r.ok:
            return r.json().get("answer", "(sem resposta)")
        return f"Erro {r.status_code} - {r.text}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Não consegui conectar no servidor Flask. Verifique se o goomi_app.py está rodando."
    except Exception as e:
        return f"⚠️ Erro ao falar com o servidor: {e}"

def get_current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]

def avatar_for(user_key: str) -> str:
    return USERS.get(user_key, USERS["glauco"])["avatar"]

def _img_b64(path: str) -> str:
    """Retorna tag <img> base64 (centra sem depender de st.image)."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' alt='logo'/>"

def render_hero():
    """Título, subtítulo e logo — alinhados ao mesmo eixo do conteúdo central."""
    st.markdown(
        f"<div class='hero-wrap'>"
        f"<h1 class='hero-title'>{APP_NAME}</h1>"
        f"<h3 class='hero-sub'>{SUBTITLE}</h3>"
        f"<div class='hero-logo' style='width:{HERO_WIDTH}px;margin:8px auto 0;'>"
        f"{_img_b64(IMG_PATH)}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ---------- Formatter de NOTAS ----------
SUBJECT_ORDER = ["Português", "Matemática", "Ciências", "História", "Geografia", "Inglês"]

def _normalize_subject(name: str) -> str:
    n = " ".join(name.split()).strip().lower()
    mapping = {
        "português": "Português",
        "matemática": "Matemática",
        "ciências": "Ciências",
        "história": "História",
        "geografia": "Geografia",
        "inglês": "Inglês",
    }
    return mapping.get(n, n.title())

def _parse_grades_text(raw: str):
    """
    Converte bullets + 'Médias por período' em estrutura.
    """
    data = {}

    def ensure(subj):
        if subj not in data:
            data[subj] = {"P1": "", "P2": "", "Media": ""}

    lines = raw.splitlines()

    bullet_re = re.compile(
        r"^[\s•\-\*]*([A-Za-zÀ-ÿ\s]+?)\s+P([1-4])\s*[—\-–]\s*.*?:\s*([0-9]+(?:[.,][0-9]+)?)\s*$"
    )
    for line in lines:
        m = bullet_re.match(line.strip())
        if m:
            subj = _normalize_subject(m.group(1))
            per  = m.group(2)
            val  = m.group(3).replace(",", ".")
            ensure(subj)
            if per in {"1", "2", "3", "4"}:
                data[subj][f"P{per}"] = val

    if "Médias por período" in raw:
        tail = raw.split("Médias por período", 1)[1]
        tokens = re.split(r"[•\u2022]", tail)
        media_re = re.compile(r"^\s*([A-Za-zÀ-ÿ\s]+?)\s+P([1-4])\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*$")
        last_media = {}
        for t in tokens:
            t = t.strip(" .:;")
            m2 = media_re.match(t)
            if not m2:
                continue
            subj = _normalize_subject(m2.group(1))
            per  = int(m2.group(2))
            val  = m2.group(3).replace(",", ".")
            if subj not in last_media or per > last_media[subj][0]:
                last_media[subj] = (per, val)

        for subj, (per, val) in last_media.items():
            ensure(subj)
            data[subj]["Media"] = val

    ordered = {}
    for s in SUBJECT_ORDER:
        if s in data:
            ordered[s] = data[s]
    for s in sorted(set(data) - set(SUBJECT_ORDER)):
        ordered[s] = data[s]
    return ordered

def render_grades_table_if_possible(raw: str) -> str | None:
    trigger = ("Aqui estão suas notas", "Aqui estão suas notas organizadas")
    if not any(t in raw for t in trigger):
        return None
    parsed = _parse_grades_text(raw)
    if not parsed:
        return None

    lines = ["| Matéria | P1 | P2 | Média |", "|:--|:--:|:--:|:--:|"]
    for subj, vals in parsed.items():
        p1 = vals.get("P1", "")
        p2 = vals.get("P2", "")
        media = vals.get("Media", "")
        lines.append(f"| {subj} | {p1} | {p2} | {media} |")
    return "\n".join(lines)

# ---------- FUTEBOL (tabelas jogos/projeções) ----------
def _mk_table_html(headers, rows):
    th = "".join(
        f"<th style='padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:left'>{h}</th>"
        for h in headers
    )
    trs = []
    for r in rows:
        tds = "".join(
            f"<td style='padding:8px 10px;border-bottom:1px solid #f1f5f9'>{c}</td>" for c in r
        )
        trs.append(f"<tr>{tds}</tr>")
    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;min-width:560px'>"
        f"<thead><tr>{th}</tr></thead>"
        f"<tbody>{''.join(trs)}</tbody>"
        "</table></div>"
    )

def _try_render_fixtures(raw: str) -> str | None:
    lines = [l.strip("•- ").strip() for l in raw.splitlines() if l.strip()]
    rows = []
    pat = re.compile(
        r"^(?P<casa>.+?)\s+vs\s+(?P<fora>.+?)\s+[—-]\s+(?P<camp>.+?)\s+[—-]\s+(?P<data>\d{2}/\d{2}/\d{4})\s+(?P<hora>\d{2}:\d{2})$",
        re.IGNORECASE
    )
    for l in lines:
        m = pat.search(l)
        if not m:
            continue
        d = m.groupdict()
        rows.append([d["data"], d["camp"], d["casa"], d["fora"], d["hora"]])
    if not rows:
        return None
    headers = ["Data", "Campeonato", "Casa", "Visitante", "Hora"]
    return _mk_table_html(headers, rows)

def _try_render_projections(raw: str) -> str | None:
    lines = [l.strip("•- ").strip() for l in raw.splitlines() if l.strip()]
    rows = []
    headers = ["Jogo", "U3.5", "Conf. U3.5", "Dupla chance", "Conf. DC", "Gols médios", "%U3.5"]
    head_pat = re.compile(r"^(?P<casa>.+?)\s+x\s+(?P<fora>.+?)\s+[—-]\s+(?P<trail>.+)$", re.IGNORECASE)
    re_under   = re.compile(r"Under\s*3\.5\s*(?P<ok>✅|❌)\s*\((?P<pct>\d{1,3})%\)", re.IGNORECASE)
    re_dc      = re.compile(r"Dupla\s+chance(?:\s+sugerida)?\s*:\s*(?P<label>1X|X2)\s*\((?P<team>[^)]+)\)\s*\((?P<pct>\d{1,3})%\)", re.IGNORECASE)
    re_avg     = re.compile(r"gols\s*m[eé]dios.*?:\s*(?P<avg>\d+(?:[.,]\d+)?)", re.IGNORECASE)
    re_pct_u35 = re.compile(r"%Under3\.5.*?:\s*(?P<pct>\d{1,3})%", re.IGNORECASE)

    for l in lines:
        m = head_pat.search(l)
        if not m:
            continue
        casa, fora, trail = m.group("casa"), m.group("fora"), m.group("trail")
        jogo = f"{casa} x {fora}"
        u35_flag = "—"; u35_conf = "—"; dc_lab = "—"; dc_conf = "—"; avg_g = "—"; pct_u35 = "—"
        mu = re_under.search(trail)
        if mu:
            u35_flag = "✅" if mu.group("ok") == "✅" else "❌"
            u35_conf = f"{mu.group('pct')}%"
        md = re_dc.search(trail)
        if md:
            dc_lab  = f"{md.group('label')} ({md.group('team')})"
            dc_conf = f"{md.group('pct')}%"
        ma = re_avg.search(trail)
        if ma:
            avg_g = ma.group("avg").replace(",", ".")
        mp = re_pct_u35.search(trail)
        if mp:
            pct_u35 = f"{mp.group('pct')}%"
        rows.append([jogo, u35_flag, u35_conf, dc_lab, dc_conf, avg_g, pct_u35])

    if not rows:
        return None
    return _mk_table_html(headers, rows)

def try_render_football_pretty(raw: str) -> bool:
    """
    Renderiza tabela para jogos ou projeções, se qualquer parser conseguir extrair dados.
    Remove o gating por 'hint' para cobrir casos com 'x' (projeções) além de 'vs'.
    """
    html = _try_render_fixtures(raw)
    if not html:
        html = _try_render_projections(raw)
    if not html:
        return False
    st.markdown(
        "<div class='center-wrap'><div class='shadow-card' style='text-align:left'>"
        "<b>Aqui está organizado:</b><br><br>"
        f"{html}"
        "</div></div>",
        unsafe_allow_html=True,
    )
    return True

# ---------- Consultas Médicas (Helena) – formatação no front ----------
def _parse_iso(dtxt: str) -> date | None:
    try:
        return datetime.strptime(dtxt.strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def _fmt_br(d: date | None) -> str:
    return d.strftime("%d/%m/%Y") if d else "-"

def _format_consulta_line(line: str) -> str | None:
    """
    Aceita os dois formatos do backend:
      A) "#3 - Consulta (Dermatologia) | realizado: 2025-09-20 | retorno: 2025-10-01"
      B) "#24 — Consulta (Odontologia; Dra. Gabriela Dia) | Agendado: 23/10/2025"
    - Suporta "-" e "—" após o ID.
    - Suporta rótulos em minúsculas/maiúsculas: realizado/agendado/retorno.
    - Suporta datas em YYYY-MM-DD e DD/MM/YYYY.
    """
    line = line.strip()
    if not line.startswith("#"):
        return None

    # ID e Título (tipo + (extras))
    m_head = re.match(r"#(?P<id>\d+)\s*[—-]\s*(?P<tipo>[A-Za-zÀ-ÿ]+)\s*(\((?P<extras>[^)]+)\))?", line)
    if not m_head:
        return None
    rid   = m_head.group("id")
    tipo  = (m_head.group("tipo") or "Consulta").capitalize()
    extra = m_head.group("extras")  # pode conter "Dermatologia" ou "Odontologia; Dra. ..."

    # Datas (qualquer ordem), capturando rótulo + data nos dois formatos
    labels = []
    for lab, dt in re.findall(r"(?i)\b(agendado|realizado|retorno)\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{2}/[0-9]{2}/[0-9]{4})", line):
        labels.append( (lab.lower(), dt) )

    # Normaliza para DD/MM/AAAA
    def to_br(d):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            y, m, dd = d.split("-"); return f"{dd}/{m}/{y}"
        return d  # já está em BR

    parts = [f"#{rid} — {tipo}"]
    if extra:
        parts[0] += f" ({extra})"

    shown_any = False
    # Prioriza ordem: Agendado/Realizado, depois Retorno, mantendo o que vier do backend
    for key in ["agendado", "realizado", "retorno"]:
        for lab, d in labels:
            if lab == key:
                parts.append(f"{lab.capitalize()}: {to_br(d)}")
                shown_any = True

    # Fallback: se não veio label explícito mas veio uma data de realização no formato antigo
    if not shown_any:
        m_real_old = re.search(r"(?i)\brealizado\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", line)
        m_ret_old  = re.search(r"(?i)\bretorno\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", line)
        if m_ret_old:
            parts.append(f"Retorno: {to_br(m_ret_old.group(1))}")
            shown_any = True
        if m_real_old:
            parts.append(f"Realizado: {to_br(m_real_old.group(1))}")
            shown_any = True

    return " | ".join(parts) if parts else None
def render_health_if_possible(raw: str) -> bool:
    triggers = ["Próximo compromisso", "Próximo retorno", "Último compromisso", "Últimos registros",
                "Proximo compromisso", "Proximo retorno", "Ultimo compromisso", "Ultimos registros"]
    if not any(t in raw for t in triggers):
        return False

    lines = [l for l in raw.splitlines() if l.strip()]
    header = None
    items = []

    for i, l in enumerate(lines):
        if any(l.startswith(t) for t in triggers):
            header = l.strip().replace(":", "")
            continue
        fmt = _format_consulta_line(l)
        if fmt:
            items.append(fmt)

    if not header and not items:
        return False
    if not header:
        header = "Registros de saúde"

    body = "<br>".join(items) if items else "—"
    st.markdown(
        "<div class='center-wrap'><div class='shadow-card' style='text-align:left'>"
        f"<b>🩺 {header}</b><br><br>"
        f"{body}"
        "</div></div>",
        unsafe_allow_html=True,
    )
    return True
# ---------- FIM (Consultas Médicas) ----------

# =========================
# Login (estreito e centralizado)
# =========================
if not st.session_state.logged_in:
    render_hero()
    st.write("")
    st.markdown("<div class='narrow-wrap'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Escolha quem vai conversar</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3,2,3])
    with c2:
        who = st.selectbox("Usuário", list(USERS.keys()), format_func=lambda k: USERS[k]["label"])
        if st.button("Entrar", use_container_width=True):
            st.session_state.client_id = who
            st.session_state.logged_in = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =========================
# Sidebar — controles
# (AGORA USANDO APENAS O BOTÃO PADRÃO DE EXPANDIR/RECOLHER DO STREAMLIT)
# =========================
with st.sidebar:
    api_ok = api_health_ok()
    st.markdown(
        f"<span class='pill {'ok' if api_ok else 'bad'}'>● API {'online' if api_ok else 'offline'}</span>",
        unsafe_allow_html=True
    )

    if st.button("👤 Trocar usuário", use_container_width=True):
        st.session_state.show_user_switcher = not st.session_state.show_user_switcher
    if st.session_state.show_user_switcher:
        who = st.selectbox(
            "Selecione o usuário",
            list(USERS.keys()),
            index=list(USERS.keys()).index(st.session_state.client_id),
            format_func=lambda k: USERS[k]["label"]
        )
        c1, c2 = st.columns(2)
        if c1.button("Confirmar"):
            st.session_state.client_id = who
            st.session_state.show_user_switcher = False
            new_cid = str(uuid.uuid4())
            st.session_state.chats = {new_cid: {"name": "Bem-vindo 👋", "messages": []}}
            st.session_state.current_chat_id = new_cid
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        if c2.button("Cancelar"):
            st.session_state.show_user_switcher = False

    if st.button("➕ Novo chat", use_container_width=True):
        cid = str(uuid.uuid4())
        st.session_state.chats[cid] = {"name": "Novo chat", "messages": []}
        st.session_state.current_chat_id = cid
        st.rerun()

    st.write("---")
    st.subheader("Conversas")

    chat_ids = list(st.session_state.chats.keys())
    names = [st.session_state.chats[c]["name"] for c in chat_ids]
    idx = chat_ids.index(st.session_state.current_chat_id)
    sel = st.selectbox("Selecionar", names, index=idx)
    new_id = chat_ids[names.index(sel)]
    st.session_state.current_chat_id = new_id

    st.caption("Renomear")
    new_name = st.text_input(" ", value=get_current_chat()["name"], label_visibility="collapsed")
    cols = st.columns(2)
    if cols[0].button("Salvar", use_container_width=True):
        get_current_chat()["name"] = new_name
        st.success("Nome atualizado!", icon="✅")
    if cols[1].button("Apagar chat", use_container_width=True):
        if len(st.session_state.chats) > 1:
            st.session_state.chats.pop(st.session_state.current_chat_id, None)
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            st.rerun()
        else:
            st.warning("Deixe ao menos um chat.")

    st.caption(f"session_id: {st.session_state.session_id}")

    if st.button("🩺 Testar API", use_container_width=True):
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            st.success("API ok" if r.ok else f"Erro: {r.status_code}")
        except Exception as e:
            st.error(f"Falha: {e}")

# =========================
# Header central
# =========================
render_hero()
st.markdown(
    f"<div class='center-wrap'><div class='shadow-card'>"
    f"💬 Você está conversando como <b>{st.session_state.client_id.capitalize()}</b>"
    f"</div></div>",
    unsafe_allow_html=True
)
st.write("")

# =========================
# Quick prompts
# =========================
st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)
st.markdown("##### Sugestões rápidas")
qcol1, qcol2, qcol3, qcol4 = st.columns(4)
quick_list = _top4_for(st.session_state.client_id)

def send_quick(q):
    if not q:
        return
    chat = get_current_chat()
    chat["messages"].append({"role": "user", "content": q})
    with st.spinner("Perguntando ao Goomih…"):
        answer = ask_backend(q)
    chat["messages"].append({"role": "assistant", "content": answer})
    if answer and not answer.startswith("⚠️"):
        _bump_quick_usage(st.session_state.client_id, q)
    st.rerun()

for i, col in enumerate([qcol1, qcol2, qcol3, qcol4]):
    if i < len(quick_list):
        with col:
            if st.button(quick_list[i], use_container_width=True):
                send_quick(quick_list[i])
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# =========================
# Conversa
# =========================
chat = get_current_chat()
for msg in chat["messages"]:
    who = "user" if msg["role"] == "user" else "assistant"
    avatar = avatar_for(st.session_state.client_id) if who == "user" else "🤖"
    with st.chat_message(who, avatar=avatar):
        content = msg["content"]
        if who == "assistant":
            # 1) consultas médicas primeiro
            if render_health_if_possible(content):
                continue
            # 2) notas
            tbl = render_grades_table_if_possible(content)
            if tbl:
                st.markdown("**Aqui estão suas notas organizadas:**")
                st.markdown(tbl)
            else:
                # 3) futebol (jogos/projeções)
                if not try_render_football_pretty(content):
                    st.markdown(content)
        else:
            st.markdown(content)

# =========================
# Entrada
# =========================
st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)
prompt = st.chat_input(f"Converse com o {APP_NAME}…")
st.markdown("</div>", unsafe_allow_html=True)

if prompt:
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_for(st.session_state.client_id)):
        st.markdown(prompt)

    with st.spinner("Pensando…"):
        answer = ask_backend(prompt)

    chat["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🤖"):
        # 1) consultas médicas primeiro
        if not render_health_if_possible(answer):
            # 2) notas
            tbl = render_grades_table_if_possible(answer)
            if tbl:
                st.markdown("**Aqui estão suas notas organizadas:**")
                st.markdown(tbl)
            else:
                # 3) futebol (jogos/projeções)
                if not try_render_football_pretty(answer):
                    st.markdown(answer)

    _bump_quick_usage(st.session_state.client_id, prompt)
    st.rerun()
