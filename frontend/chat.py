import os
import uuid
import base64
import json
import re
import hashlib
from collections import Counter
import requests
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, date

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
# Padrão QA: 127.0.0.1:5000. Em produção, GOOMI_API_BASE é definido via /etc/goomi.env
API_BASE   = os.getenv("GOOMI_API_BASE", "http://127.0.0.1:5000")
ASK_URL    = f"{API_BASE}/ask"
UPLOAD_URL = f"{API_BASE}/upload"
UPLOADS_URL = f"{API_BASE}/uploads"
# TRANSCRIBE_URL não é mais usado (voz transcreve via OpenAI no frontend)

APP_NAME   = "Goomih"
SUBTITLE   = "Assistente virtual da nossa família"
HERO_WIDTH = 120  # logo
CONTENT_MAX = 900

USERS = {
    "giulia":    {"label": "Giulia",    "avatar": "😊", "color": "pink"},
    "guilherme": {"label": "Guilherme", "avatar": "😎", "color": "blue"},
    "giovanna":  {"label": "Giovanna",  "avatar": "😊", "color": "pink"},
    "helena":    {"label": "Helena",    "avatar": "😊", "color": "pink"},
    "glauco":    {"label": "Glauco",    "avatar": "😎", "color": "blue"},
    "rayane":   {"label": "Rayane",   "avatar": "😊", "color": "pink"},
}

# ====== Perguntas rápidas (por usuário) ======
DEFAULT_QUICK = {
    # Pais (cadastro / gestão)
    "glauco": [
        "Manual escola",
        "Giulia tem aula de que amanhã?",
        "Agenda escolar da Giulia esse mês",
        "Notas da Giulia no 1B",
    ],
    "helena": [
        "Manual escola",
        "Giovanna tem aula de que hoje?",
        "Agenda escolar da Giovanna esse mês",
        "Notas da Giovanna no 1B",
    ],
    "rayane": [
        "Manual escola",
        "Giulia tem aula de que hoje?",
        "Agenda escolar da Giulia esse mês",
        "Notas da Giulia no 1B",
    ],

    # Crianças (leitura + estudo)
    "giulia": [
        "Que aula a Giulia tem hoje?",
        "Que aula a Giulia tem amanhã?",
        "Mostra a grade semanal da Giulia",
        "Me ajuda a estudar matemática (explica bem simples)",
    ],
    "giovanna": [
        "Que aula a Giovanna tem hoje?",
        "Que aula a Giovanna tem amanhã?",
        "Mostra a grade semanal da Giovanna",
        "Me ajuda a estudar português (explica bem simples)",
    ],
    "guilherme": [
        "Que aula o Guilherme tem hoje?",
        "Que aula o Guilherme tem amanhã?",
        "Mostra a grade semanal do Guilherme",
        "Me ajuda a estudar ciências (explica bem simples)",
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
    ss.setdefault("chats", {})
    ss.setdefault("current_chat_id", None)

    if not ss["chats"]:
        cid = str(uuid.uuid4())
        ss["chats"][cid] = {"name": "Bem-vindo 👋", "messages": []}
        ss["current_chat_id"] = cid

init_state()

# =========================
# Estilos
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
  --content-max: {CONTENT_MAX}px;
}}

.stApp {{ background: var(--bg) !important; color: var(--fg) !important; }}
.block-container {{ padding-top: 1.2rem; }}
.center-wrap {{ max-width: var(--content-max); margin: 0 auto; }}

.hero-wrap {{ max-width: var(--content-max); margin: 0 auto; text-align: center; }}
.hero-title {{ font-size: 40px; margin: 0 0 .25rem 0; }}
.hero-sub   {{ font-size: 20px; font-weight: 600; color: var(--sub); margin: 0; }}
.hero-logo  {{ margin-top: 8px; }}
.hero-logo img {{ width: {HERO_WIDTH}px !important; max-width: {HERO_WIDTH}px !important; display: inline-block !important; }}

.pill {{
  display:inline-block; padding:6px 10px; border-radius:999px;
  background: var(--card); color: var(--fg);
  border: 1px solid rgba(0,0,0,0.05);
  font-weight:600; font-size:12px;
}}
.pill.ok {{ background: rgba(34,197,94,0.15); color: var(--ok); }}
.pill.bad{{ background: rgba(239,68,68,0.15); color: var(--bad); }}

.shadow-card {{
  background: var(--card);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 10px 14px;
  color: var(--fg);
  text-align:center;
}}

[data-testid="stSidebar"] * {{ font-size: 14px !important; }}

.stButton > button, button[kind],
a[data-testid="baseLinkButton-primary"], a[data-testid="baseLinkButton-secondary"] {{
  background-color: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #e5e7eb !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.04) !important;
  border-radius: 12px !important;
}}
.stButton > button:hover,
button[kind]:hover,
a[data-testid="baseLinkButton-primary"]:hover,
a[data-testid="baseLinkButton-secondary"]:hover {{
  background-color: #f8fafc !important;
  border-color: #dbe1ea !important;
}}
.stButton > button:active,
button[kind]:active {{ background-color: #f1f5f9 !important; }}

.stTextInput input, .stTextArea textarea {{
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;
}}
div[role="combobox"] {{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;
}}

[data-testid="stChatInput"] {{ background: #ffffff !important; }}

@media (prefers-color-scheme: dark) {{
  .stButton > button, button[kind],
  a[data-testid="baseLinkButton-primary"], a[data-testid="baseLinkButton-secondary"],
  .stTextInput input, .stTextArea textarea, div[role="combobox"],
  [data-testid="stChatInput"] {{
    background: #ffffff !important;
    color: #0f172a !important;
  }}
}}
</style>
""", unsafe_allow_html=True)



def upload_to_backend(file_bytes: bytes, filename: str, description: str) -> dict:
    try:
        files = {"file": (filename, file_bytes)}
        data = {"client_id": st.session_state.client_id, "description": description or ""}
        r = requests.post(UPLOAD_URL, files=files, data=data, timeout=180)
        if r.ok:
            return r.json()
        return {"ok": False, "error": f"Erro {r.status_code}: {r.text}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================
# Utils
# =========================

def ask_backend(question: str) -> str:
    ql = (question or "").strip().lower()

    payload = {"client_id": st.session_state.client_id, "question": question}
    try:
        r = requests.post(ASK_URL, json=payload, timeout=120)
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
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' alt='logo'/>"

def render_hero():
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

# ---------- Formatação genérica para respostas "cruas" do Goomih ----------
def _looks_markdownish(txt: str) -> bool:
    """Heurística simples para não retrabalhar textos já em Markdown rico (ex.: OpenAI)."""
    if any(h in txt for h in ["\n- ", "\n* ", "\n1. ", "```", "__", "**", "# "]):
        return True
    if re.search(r"\[[^\]]+\]\([^)]+\)", txt):
        return True
    return False

def _normalize_bullets(txt: str) -> str:
    """Converte bullets '•' ou '-' em lista markdown; quebra ' • ' inline em nova linha."""
    txt = re.sub(r"\s+•\s+", "\n• ", txt)
    txt = re.sub(r"([^\n])\s*•\s+", r"\1\n• ", txt)
    lines = []
    for raw in txt.splitlines():
        s = raw.strip()
        if s.startswith("•"):
            s = s.lstrip("•").strip()
            lines.append(f"- {s}")
        elif re.match(r"^[-–—]\s+\S", s):
            s = re.sub(r"^[-–—]\s+", "", s)
            lines.append(f"- {s}")
        else:
            lines.append(s)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

def _bold_labels(txt: str) -> str:
    """Destaca 'Titulo: valor' como **Titulo:** valor."""
    def repl(m):
        label = m.group(1).strip()
        value = m.group(2).strip()
        return f"**{label}:** {value}"
    pattern = re.compile(r"(?m)^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]+?):\s*(.+)$")
    return pattern.sub(repl, txt)

def format_goomi_plaintext(raw: str) -> str | None:
    """
    Deixa bonito textos do 'Manual — ...' e outras respostas cruas do Goomih.
    Não mexe em respostas que já parecem Markdown.
    """
    if not raw or _looks_markdownish(raw):
        return None

    txt = raw.strip()

    manual_m = re.match(r"(?s)^\s*(Manual\s+—\s+[^\n]+)\s*\n+(.*)$", txt)
    if manual_m:
        title = manual_m.group(1).strip()
        body  = manual_m.group(2).strip()
        body  = _normalize_bullets(body)
        body  = _bold_labels(body)
        md = f"### {title}\n\n{body}"
        return md

    if "•" in txt or re.search(r"(?m)^\s*[-–—]\s+\S", txt):
        body = _normalize_bullets(txt)
        body = _bold_labels(body)
        return body

    labels = _bold_labels(txt)
    if labels != txt:
        return labels

    return None


# =========================
# LOGIN COM SENHA POR USUÁRIO
# =========================
USERS_PASS_SHA256 = {
    "giulia":    "d0a28ee5acfcd6f70942dfc57a71418469062a92b380036e5f1b53848bc6e0c2",
    "giovanna":  "bf14dbb338eeb960b694a01b3d66d4a13f9c4c5b12a2a43f15b628811957524d",
    "guilherme": "0a9ad7b5557b663db0dcde8160043f5a7873c441aef3da4037690992dfeb4b31",
    "glauco":    "e4d8e2c97976e3e0ddeae407fd54987f0b4f8d6792284742b51399a078765319",
    "helena":    "4b9a7f50c0bb198c6f5414c5a8459f5d216d34ab521ea94c060ea35cac66f900",
}

# Permite criar/atualizar senha sem editar o código: salva hashes em um JSON local
PASSWORDS_PATH = os.path.join(BASE_DIR, "user_passwords.json")

def _load_password_overrides() -> dict:
    if not os.path.exists(PASSWORDS_PATH):
        return {}
    try:
        with open(PASSWORDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except Exception:
        return {}

def _save_password_overrides(overrides: dict) -> None:
    try:
        with open(PASSWORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# aplica overrides (ex: primeiro acesso da Rayane)
USERS_PASS_SHA256.update(_load_password_overrides())


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# Login (centralizado estreito; senha abaixo; botão largura dos inputs)
if not st.session_state.logged_in:
    render_hero()
    st.write("")
    st.markdown("<h3 style='text-align:center;'>Escolha quem vai conversar</h3>", unsafe_allow_html=True)

    left, mid, right = st.columns([1, 1.25, 1])  # ~33% da largura
    with mid:
        who = st.selectbox("Usuário", list(USERS.keys()), format_func=lambda k: USERS[k]["label"])
        pwd = st.text_input("Senha", type="password")
        # Primeiro acesso: se for um usuário novo (ex.: Rayane), você pode definir uma senha aqui.
        # Dica: depois a gente pode migrar isso para um painel de configurações.
        pending_user = st.session_state.get("_pending_user")
        if pending_user:
            st.info(f"Primeiro acesso de **{USERS.get(pending_user, {}).get('label', pending_user).title()}**: defina uma senha.")
            new_pwd1 = st.text_input("Nova senha", type="password", key="new_pwd1")
            new_pwd2 = st.text_input("Confirmar nova senha", type="password", key="new_pwd2")
            cols_pwd = st.columns(2)
            if cols_pwd[0].button("Salvar senha", use_container_width=True):
                if not new_pwd1 or len(new_pwd1) < 4:
                    st.error("Use uma senha com pelo menos 4 caracteres.")
                elif new_pwd1 != new_pwd2:
                    st.error("As senhas não conferem.")
                else:
                    overrides = _load_password_overrides()
                    overrides[pending_user] = _sha256_hex(new_pwd1)
                    _save_password_overrides(overrides)
                    USERS_PASS_SHA256[pending_user] = overrides[pending_user]
                    st.session_state.pop("_pending_user", None)
                    st.session_state.pop("_pending_pwd", None)
                    st.success("Senha salva! Agora é só entrar.")
                    st.rerun()
            if cols_pwd[1].button("Cancelar", use_container_width=True):
                st.session_state.pop("_pending_user", None)
                st.session_state.pop("_pending_pwd", None)
                st.rerun()
        if st.button("Entrar", use_container_width=True):
            # Se o usuário ainda não tem senha definida (ex.: Rayane), permitir criar no primeiro acesso
            if who not in USERS_PASS_SHA256:
                st.session_state["_pending_user"] = who
                st.session_state["_pending_pwd"] = pwd
                st.rerun()
            elif _sha256_hex(pwd) != USERS_PASS_SHA256[who]:
                st.error("Senha incorreta. Tente novamente.")
            else:
                st.session_state.client_id = who
                st.session_state.logged_in = True
                st.rerun()

    st.stop()

# =========================
# Sidebar — sem troca de usuário
# =========================
with st.sidebar:

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

    st.write("---")
    st.subheader("📎 Upload")
    up_file = st.file_uploader("Enviar arquivo (boletim, aviso, etc.)", type=None, label_visibility="collapsed")
    up_desc = st.text_input("Descrição do arquivo (pra achar depois)", key="up_desc")
    if st.button("Salvar upload", use_container_width=True, disabled=(up_file is None)):
        if up_file is None:
            st.warning("Selecione um arquivo primeiro.")
        else:
            resp = upload_to_backend(up_file.getvalue(), up_file.name, up_desc)
            if resp.get("ok"):
                st.success(f"Upload salvo! ID #{resp.get('upload_id')}", icon="✅")
                # registra no chat atual como mensagem do sistema
                get_current_chat()["messages"].append({"role":"assistant","content": f"📎 Upload salvo: #{resp.get('upload_id')} — {resp.get('original_name')}\nDescrição: {resp.get('description') or '—'}\nDica: 'Resumir arquivo #{resp.get('upload_id')}'"})
                st.rerun()
            else:
                st.error(resp.get("error","Falha no upload."))



# =========================
# Header central (mantém como v8)
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
            # 1) notas (quando o backend devolver no formato conhecido)
            tbl = render_grades_table_if_possible(content)
            if tbl:
                st.markdown("**Aqui estão suas notas organizadas:**")
                st.markdown(tbl)
            else:
                # 2) formatação genérica para respostas cruas
                md = format_goomi_plaintext(content)
                if md:
                    st.markdown(md)
                else:
                    st.markdown(content)
        else:
            st.markdown(content)



# =========================
# Entrada
# =========================
st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)






# ─────────────────────────────────────────────






# Input row (texto + voz) — layout refinado






# ─────────────────────────────────────────────






st.markdown("""






<style>






/* Deixa o input e o mic na mesma linha */






div[data-testid="stHorizontalBlock"]{ align-items: center !important; }













/* Limpa o visual do audio_input (sem caixa) */






div[data-testid="stAudioInput"]{ background: transparent !important; padding: 0 !important; margin: 0 !important; }






div[data-testid="stAudioInput"] > div{ background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }






div[data-testid="stAudioInput"] label, div[data-testid="stAudioInput"] p{ display:none !important; height:0 !important; margin:0 !important; padding:0 !important; }






div[data-testid="stAudioInput"] button{ width: 38px !important; height: 38px !important; border-radius: 999px !important; padding: 0 !important; }






div[data-testid="stAudioInput"] span{ font-size: 11px !important; }






</style>






""", unsafe_allow_html=True)













col_chat, col_mic = st.columns([0.88, 0.12], gap="small")






with col_chat:






    prompt = st.chat_input("Converse com o Goomih…")






with col_mic:






    voice = st.audio_input("", key="voice_row_v12")













# Auto-envia quando chega um áudio novo






if "last_voice_hash" not in st.session_state:






    st.session_state["last_voice_hash"] = ""













if voice is not None:






    audio_bytes = voice.getvalue()






    if audio_bytes:






        try:






            import hashlib, tempfile






            from openai import OpenAI













            h = hashlib.md5(audio_bytes).hexdigest()






            if h != st.session_state["last_voice_hash"]:






                st.session_state["last_voice_hash"] = h













                client = OpenAI()






                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:






                    tmp.write(audio_bytes)






                    tmp_path = tmp.name













                with open(tmp_path, "rb") as f:






                    tr = client.audio.transcriptions.create(






                        model="gpt-4o-mini-transcribe",






                        file=f






                    )













                texto = (getattr(tr, "text", "") or "").strip()






                if texto:






                    get_current_chat()["messages"].append({"role":"user","content": f"(Áudio) {texto}"})






                    with st.spinner("Perguntando ao Goomih…"):






                        answer = ask_backend(texto)






                    get_current_chat()["messages"].append({"role":"assistant","content": answer})






                    _bump_quick_usage(st.session_state.client_id, texto)






                    st.rerun()






        except Exception:






            try:






                st.toast("Falha ao transcrever (verifique OPENAI_API_KEY).", icon="⚠️")






            except Exception:






                pass
st.markdown("</div>", unsafe_allow_html=True)


if prompt:
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_for(st.session_state.client_id)):
        st.markdown(prompt)

    with st.spinner("Pensando…"):
        answer = ask_backend(prompt)

    chat["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🤖"):
        # 1) notas
        tbl = render_grades_table_if_possible(answer)
        if tbl:
            st.markdown("**Aqui estão suas notas organizadas:**")
            st.markdown(tbl)
        else:
            # 2) formatação genérica para respostas cruas
            md = format_goomi_plaintext(answer)
            if md:
                st.markdown(md)
            else:
                st.markdown(answer)

    _bump_quick_usage(st.session_state.client_id, prompt)
    st.rerun()

# =========================
# Auto-scroll (mantém a caixa de texto visível)
# =========================
# A âncora fica NO FIM DA PÁGINA (depois do input). Assim, ao rolar até o fim,
# a última mensagem fica visível e a caixa de texto também.
# Para não atrapalhar quando você estiver lendo mensagens antigas, só auto-rola
# quando o tamanho do histórico muda.

_current_len = len(get_current_chat()["messages"])
_last_len = st.session_state.get("_last_chat_len", -1)

st.markdown("<div id='page-bottom' style='height:1px;'></div>", unsafe_allow_html=True)

if _current_len != _last_len:
    st.session_state["_last_chat_len"] = _current_len
    components.html(
        """
        <script>
          const el = window.parent.document.getElementById('page-bottom');
          if (el) {
            // Pequeno atraso para garantir que o DOM já renderizou a última mensagem e o input
            setTimeout(() => { el.scrollIntoView({behavior: 'smooth', block: 'end'}); }, 80);
          }
        </script>
        """,
        height=0,
    )
