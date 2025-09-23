# (código completo — GOOMI, com ajustes)
"""
GOOMI — Agente de IA da Família G (backend)
-------------------------------------------
- Lê config.yaml e OPENAI_API_KEY
- Conecta ao banco goomi.db (tabelas: familia, notas, medias, consultas_exames, projecoes)
- Mantém memória por integrante (client_id)
- Endpoints:
    /health                               -> status da API (GET)
    /ask                                  -> chat geral + notas + esportes + saúde + cumprimentos + projeções (POST)
    /v1/football/league/day               -> jogos por liga+data/when (GET)
    /v1/football/serie-b/projections      -> projeções heurísticas Série B (GET)
    /v1/football/obvious                  -> jogos “óbvios” do dia (GET)
"""

# ==== 1) Importações =========================================================
import os
import re
import json
import time
import sqlite3
from typing import Optional, List, Dict, Any

import requests
import yaml
from flask import Flask, request, jsonify

# IA (LangChain)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage

from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo

# (opcional) silenciar deprecations do langchain
import warnings
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning

# ---- helper: remove 'dia' terminal capturado em frases como '... com Dr. X dia 12/09' ----

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass


# ==== 2) Configurações =======================================================
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# OPENAI

def _strip_trailing_dia(text):
    import re as _re
    if text is None:
        return None
    t = str(text).strip()
    # remove apenas 'dia' isolado no final (case-insensitive), preserva 'Diaz' etc.
    t = _re.sub(r"[ \t,;:]*\bdia\b$", "", t, flags=_re.IGNORECASE).strip()
    return t

os.environ["OPENAI_API_KEY"] = (
    os.getenv("OPENAI_API_KEY") or config.get("api_key", {}).get("key", "")
)

DB_PATH       = config["project"]["db_path"]
MODEL_NAME    = config["model"]["name"]
TEMPERATURE   = float(config["model"].get("temperature", 0))

PROJECT_TZ = ZoneInfo(config.get("project", {}).get("timezone", "America/Sao_Paulo"))

# Esportes
SPORTS_ENABLED  = bool(config.get("sports", {}).get("enabled", False))
SPORTS_PROVIDER = (config.get("sports", {}).get("provider", "") or "").lower().strip()
SPORTS_KEY      = (config.get("sports", {}).get("api_key", "") or os.getenv("APISPORTS_KEY", "") or "").strip()

# Projeções — limiares (podem ser ajustados no config.yaml; defaults sensatos)
PROJ = config.get("projections", {}) or {}
UNDER35_AVG_MAX = float(PROJ.get("UNDER35_AVG_MAX", 2.7))
UNDER35_PCT_MIN = float(PROJ.get("UNDER35_PCT_MIN", 0.8))
FORM_WEIGHT     = float(PROJ.get("FORM_WEIGHT", 1.0))
DEF_WEIGHT      = float(PROJ.get("DEF_WEIGHT", 0.8))
HOME_ADV        = float(PROJ.get("HOME_ADV", 0.3))
DC_THRESHOLD    = float(PROJ.get("DC_THRESHOLD", 0.8))
H2H_LAST        = int(PROJ.get("H2H_LAST", 5))
H2H_UNDER_BOOST = float(PROJ.get("H2H_UNDER_BOOST", 0.15))
H2H_DC_BOOST    = float(PROJ.get("H2H_DC_BOOST", 0.15))

# Log
DEBUG_LOG = True

# Cache simples em memória (TTL 60s)
CACHE_TTL_SECONDS = 60
_cache: Dict[str, Dict[str, Any]] = {}  # { key: {"ts": float, "data": Any} }

def _cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    if time.time() - item["ts"] <= CACHE_TTL_SECONDS:
        return item["data"]
    _cache.pop(key, None)
    return None

def _cache_set(key: str, data: Any):
    _cache[key] = {"ts": time.time(), "data": data}


# ==== 3) Flask ===============================================================
app = Flask(__name__)
user_memories: Dict[str, ConversationBufferMemory] = {}  # memória RAM por usuário

# memorizar último time consultado por usuário
LAST_TEAM_BY_USER: Dict[str, str] = {}


# ==== 4) Banco de dados ======================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def get_user_by_client_id(conn, client_id: str):
    cur = conn.cursor()
    cur.execute("SELECT * FROM familia WHERE client_id = ?", (client_id,))
    return cur.fetchone()

def get_user_id_by_client_id(conn, client_id: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM familia WHERE client_id = ?", (client_id,))
    row = cur.fetchone()
    return int(row["id"]) if row else None

def ensure_health_schema(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS consultas_exames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aluno_id INTEGER NOT NULL,
        tipo TEXT NOT NULL,                -- consulta, exame, procedimento
        especialidade TEXT,                -- dermatologia, cardiologia...
        descricao TEXT,
        data_realizacao TEXT,              -- YYYY-MM-DD
        data_retorno TEXT,                 -- YYYY-MM-DD
        criado_em TEXT,
        FOREIGN KEY (aluno_id) REFERENCES familia(id)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consultas_aluno    ON consultas_exames(aluno_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consultas_tipo     ON consultas_exames(tipo);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consultas_retorno  ON consultas_exames(data_retorno);")
    conn.commit()

def ensure_projections_schema(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS projecoes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        date_ref TEXT,
        league_id INTEGER,
        home_id INTEGER,
        away_id INTEGER,
        home TEXT,
        away TEXT,
        under35 INTEGER,                 -- 0/1
        conf_under REAL,
        double_chance TEXT,              -- '1X', 'X2' ou NULL
        conf_dc REAL,
        avg_total REAL,
        pct_under REAL,
        raw_json TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_proj_date ON projecoes(date_ref);")
    conn.commit()

# >>> NOVO: garantir schema de notas/medias <<<
def ensure_grades_schema(conn):
    cur = conn.cursor()
    # notas
    cur.execute("""
    CREATE TABLE IF NOT EXISTS notas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aluno_id INTEGER NOT NULL,
        disciplina TEXT NOT NULL,
        periodo TEXT NOT NULL,           -- P1..P4 ou T1..T3
        descricao TEXT,
        nota REAL NOT NULL,
        criado_em TEXT,
        FOREIGN KEY (aluno_id) REFERENCES familia(id)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notas_aluno ON notas(aluno_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notas_disc  ON notas(disciplina);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notas_per   ON notas(periodo);")

    # medias (com UNIQUE para permitir UPSERT)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS medias (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aluno_id INTEGER NOT NULL,
        disciplina TEXT NOT NULL,
        periodo TEXT NOT NULL,
        media REAL NOT NULL,
        atualizado_em TEXT,
        FOREIGN KEY (aluno_id) REFERENCES familia(id),
        UNIQUE (aluno_id, disciplina, periodo)
    );
    """)
    # redundante mas ajuda em selects
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medias_aluno ON medias(aluno_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medias_disc  ON medias(disciplina);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medias_per   ON medias(periodo);")
    conn.commit()


# ==== 5) Modelo de IA ========================================================
chat = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

SYSTEM_PROMPT = SystemMessage(content=(
    "Você é o GOOMI, assistente da Família G (Glauco, Helena, Giulia, Giovanna, Guilherme). "
    "Seja objetivo e gentil. "
    "Para perguntas sobre NOTAS ESCOLARES do próprio aluno, use os dados do sistema. "
    "Para outros temas, responda normalmente. "
    "Só personalize o perfil da pessoa quando ela perguntar sobre si."
))


# ==== 6) NOTAS (layout com período + médias) ================================
DISCIPLINA_MAP = {
    "portugues": "Português",
    "português": "Português",
    "matematica": "Matemática",
    "matemática": "Matemática",
    "historia": "História",
    "história": "História",
    "geografia": "Geografia",
    "ciencias": "Ciências",
    "ciências": "Ciências",
    "ingles": "Inglês",
    "inglês": "Inglês",
}
PERIODO_REGEX = r"\b((?:p|t)\s*[1-4])\b"  # P1..P4 ou T1..T3

def normaliza_disciplina(txt: str) -> Optional[str]:
    if not txt:
        return None
    t = txt.strip().lower()
    if t in DISCIPLINA_MAP:
        return DISCIPLINA_MAP[t]
    t = (t.replace("á","a").replace("â","a").replace("ã","a")
            .replace("é","e").replace("ê","e")
            .replace("í","i")
            .replace("ó","o").replace("ô","o")
            .replace("ú","u").replace("ç","c"))
    return DISCIPLINA_MAP.get(t)

def normaliza_periodo(texto: str) -> Optional[str]:
    if not texto:
        return None
    m = re.search(PERIODO_REGEX, texto.lower())
    if not m:
        return None
    val = m.group(1).upper().replace(" ", "")
    if val[0] == "P" and val[1] in "1234":
        return f"P{val[1]}"
    if val[0] == "T" and val[1] in "123":
        return f"T{val[1]}"
    return None

def extrair_filtros_notas(question: str) -> dict:
    sys = SystemMessage(content=(
        "Analise a pergunta e responda APENAS um JSON. "
        "Responda is_notas=true se a pergunta for sobre notas escolares do próprio aluno. "
        "Se possível, extraia 'disciplina' e 'periodo' (P1..P4 ou T1..T3). "
        'Formato: {"is_notas": true/false, "disciplina": <str|null>, "periodo": <str|null>}'
    ))
    hum = HumanMessage(content=f"Pergunta: {question}")
    try:
        resp = chat.invoke([sys, hum]).content.strip()
        data = json.loads(resp)
        disc = normaliza_disciplina(data.get("disciplina"))
        per = normaliza_periodo(data.get("periodo") or "")
        return {"is_notas": bool(data.get("is_notas")), "disciplina": disc, "periodo": per}
    except Exception:
        q = question.lower()
        is_notas = any(k in q for k in [
            "nota","boletim","minhas notas","quanto tirei","qual foi a minha nota","média","media"
        ])
        return {"is_notas": is_notas, "disciplina": None, "periodo": normaliza_periodo(q)}

def consultar_notas(conn, aluno_id: int, disciplina: Optional[str], periodo: Optional[str]):
    sql = """
        SELECT disciplina, periodo, descricao, nota
        FROM notas
        WHERE aluno_id = ?
    """
    params = [aluno_id]
    if disciplina:
        sql += " AND disciplina = ?"
        params.append(disciplina)
    if periodo:
        sql += " AND periodo = ?"
        params.append(periodo)
    sql += " ORDER BY disciplina, periodo, id"
    rows = conn.execute(sql, params).fetchall()

    sqlm = """
        SELECT disciplina, periodo, media
          FROM medias
         WHERE aluno_id = ?
    """
    pm = [aluno_id]
    if disciplina:
        sqlm += " AND disciplina = ?"
        pm.append(disciplina)
    if periodo:
        sqlm += " AND periodo = ?"
        pm.append(periodo)
    medias = conn.execute(sqlm, pm).fetchall()

    return rows, medias

def formatar_resposta_notas(nome: str, rows, medias, disciplina: Optional[str], periodo: Optional[str]) -> str:
    if not rows and not medias:
        partes = []
        if disciplina: partes.append(f"de {disciplina}")
        if periodo:    partes.append(f"em {periodo}")
        suf = f" {' '.join(partes)}" if partes else ""
        return f"{nome}, não encontrei notas{suf} no meu registro."

    linhas = []
    if rows:
        for r in rows:
            desc = r["descricao"] or "Avaliação"
            linhas.append(f"- {r['disciplina']} {r['periodo']} — {desc}: {r['nota']}")
    if medias:
        linhas.append("")
        linhas.append("Médias por período:")
        for m in medias:
            linhas.append(f"• {m['disciplina']} {m['periodo']}: {round(m['media'], 2)}")

    head = "Aqui estão suas notas"
    if disciplina: head += f" de {disciplina}"
    if periodo:    head += f" em {periodo}"
    return head + ":\n" + "\n".join(linhas)

def upsert_media(conn, client_id_alvo: str, disciplina: str, periodo: str, media_val: float) -> bool:
    aluno_id = get_user_id_by_client_id(conn, client_id_alvo)
    if not aluno_id:
        return False
    conn.execute("""
        INSERT INTO medias (aluno_id, disciplina, periodo, media, atualizado_em)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(aluno_id, disciplina, periodo) DO UPDATE SET
            media = excluded.media,
            atualizado_em = datetime('now')
    """, (aluno_id, disciplina, periodo, float(media_val)))
    conn.commit()
    return True

def inserir_nota_conversa(conn, client_id_alvo: str, disciplina: str, periodo: str, descricao: Optional[str], nota: float) -> bool:
    aluno_id = get_user_id_by_client_id(conn, client_id_alvo)
    if not aluno_id:
        return False
    agora = datetime.now().isoformat(timespec="seconds")
    conn.execute("""
        INSERT INTO notas (aluno_id, disciplina, periodo, descricao, nota, criado_em)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (aluno_id, disciplina, periodo, (descricao or periodo), nota, agora))
    conn.commit()
    return True

def extrair_comando_salvar(question: str, client_id_padrao: str) -> Dict[str, Any]:
    q = question.lower()

    # >>> NOVO: não confundir com saúde
    if any(w in q for w in ["consulta", "exame", "procediment", "retorno"]):
        return {"is_action": False}
    if any(k in q for k in ["dermatolog","cardiolog","ginecolog","ortoped","oftalmolog","endocrin","psiquiat","psicol","odont","clínico","clinico"]):
        return {"is_action": False}

    if not any(w in q for w in ["salva", "grava", "registr", "adiciona", "anota"]):
        return {"is_action": False}

    alvo = client_id_padrao
    for who in ["giulia","guilherme","giovanna","helena","glauco"]:
        if who in q:
            alvo = who
            break

    disc = None
    for k, v in DISCIPLINA_MAP.items():
        if re.search(rf"\b{k}\b", q):
            disc = v
            break

    periodo = normaliza_periodo(q)

    desc = None
    m_desc = re.search(r"\bprova\s*([1-4])\b", q)
    if m_desc and not periodo:
        desc = f"P{m_desc.group(1)}"
        periodo = desc

    # números: tomar cuidado para não confundir com anos/datas
    nums = [m.group(1) for m in re.finditer(r"(?:(?:nota|m[eé]dia|ficou|tirou|deu)[^0-9]{0,15})?(\d{1,2}(?:[.,]\d{1,2})?)", q)]
    media = None
    media_match = re.search(r"m[eé]dia[^0-9]*?(\d{1,2}(?:[.,]\d{1,2})?)", q)
    if media_match:
        try: media = float(media_match.group(1).replace(",", "."))
        except: media = None

    nota = None
    if nums:
        first = float(nums[0].replace(",", "."))
        if media is not None and abs(first - media) < 1e-6 and len(nums) > 1:
            nota = float(nums[1].replace(",", "."))
        else:
            nota = first

    return {
        "is_action": True,
        "client_id": alvo,
        "disciplina": disc,
        "periodo": periodo,
        "descricao": desc,
        "nota": nota,
        "media": media,
    }


# ==== 7) Perfis (apenas quando perguntarem) =================================
USER_PROFILES = {
    "giulia": {
        "nome": "Giulia",
        "idade": 9,
        "descricao": (
            "Oi, eu sou a Giulia! Tenho 9 anos, adoro ver vídeos engraçados, viajar e inventar brincadeiras "
            "com as minhas amigas. Torço pro Flamengo, curto um pagodinho e amo conversar sobre qualquer coisa — "
            "vale futebol, histórias, curiosidades… o que você quiser!"
        ),
    },
}
def is_profile_question(q: str) -> bool:
    q = q.lower()
    gatilhos = [
        "você me conhece", "voce me conhece",
        "quem sou eu", "quem eu sou",
        "sobre mim", "o que eu gosto", "minhas preferências", "minhas preferencias",
        "me descreve", "me descreva", "fale sobre mim"
    ]
    return any(g in q for g in gatilhos)


# ==== 8) Esportes — API-Football ============================================
FOOTBALL_BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"br_serie_a": 71, "br_serie_b": 72}

TEAM_ALIASES = {"flemango": "flamengo", "atlético": "atletico", "grêmio": "gremio"}  # correções comuns
# ---- Ligas por apelidos (normaliza variações -> id fixo) --------------------
LIGA_MAP = {
    # Brasil
    "serie a": 71, "série a": 71, "brasileirao serie a": 71, "brasileirão série a": 71,
    "serie b": 72, "série b": 72, "brasileirao serie b": 72, "brasileirão série b": 72,
    "copa do brasil": 73,

    # Inglaterra
    "premier league": 39, "premiere league": 39, "pl": 39, "epl": 39, "inglaterra": 39,

    # Espanha
    "la liga": 140, "laliga": 140, "espanha": 140,

    # Alemanha
    "bundesliga": 78, "alemanha": 78,

    # França
    "ligue 1": 61, "franca": 61, "frança": 61,

    # Europa / Conmebol
    "champions league": 2, "ucl": 2, "champions": 2,
    "libertadores": 13,
}

def _norm(txt: str) -> str:
    return (txt or "").strip().lower().replace("  ", " ")


def _api_headers():
    return {"x-apisports-key": SPORTS_KEY}

def _tz_now_date() -> datetime.date:
    return datetime.now(PROJECT_TZ).date()

def _date_to_str(d: datetime.date) -> str:
    return d.strftime("%Y-%m-%d")

def _season_from_date(date_: datetime.date) -> int:
    return date_.year

def _today() -> datetime.date: return _tz_now_date()
def _yesterday() -> datetime.date: return _tz_now_date() - timedelta(days=1)
def _tomorrow() -> datetime.date: return _tz_now_date() + timedelta(days=1)


# ---- interpretar_periodo_saude (humanized date ranges) ----
def interpretar_periodo_saude(q: str):
    """Return dict with date_from and date_to in YYYY-MM-DD or None."""
    from datetime import timedelta as _td
    qs = (q or '').lower()
    try:
        today = datetime.now(PROJECT_TZ).date()
    except Exception:
        today = datetime.now().date()
    df = None
    dt = None
    if any(k in qs for k in ['proxima','próxima','proxim','proím','futuras','futura','depois de hoje','apos hoje','após hoje']):
        df = today + _td(days=1)
    if any(k in qs for k in ['realiz','realizada','realizadas','anteriores','antes de hoje','passadas','passado']):
        dt = today - _td(days=1)
    if 'amanh' in qs or 'amanhã' in qs:
        d = today + _td(days=1); df = d; dt = d
    if 'ontem' in qs:
        d = today - _td(days=1); df = d; dt = d
    if 'hoje' in qs and df is None and dt is None:
        d = today; df = d; dt = d
    if 'semana que vem' in qs or 'proxima semana' in qs or 'próxima semana' in qs:
        dow = today.weekday()
        next_monday = today + _td(days=(7 - dow))
        next_sunday = next_monday + _td(days=6)
        df = next_monday; dt = next_sunday
    if 'mes que vem' in qs or 'mês que vem' in qs:
        y = today.year + (1 if today.month == 12 else 0)
        m = 1 if today.month == 12 else today.month + 1
        first = datetime(y, m, 1).date()
        if m == 12:
            last = datetime(y, 12, 31).date()
        else:
            next_first = datetime(y, m+1, 1).date()
            last = next_first - _td(days=1)
        df = first; dt = last
    if 'mes passado' in qs or 'mês passado' in qs:
        if today.month == 1:
            y = today.year - 1; m = 12
        else:
            y = today.year; m = today.month - 1
        first = datetime(y, m, 1).date()
        if m == 12:
            last = datetime(y, 12, 31).date()
        else:
            next_first = datetime(y, m+1, 1).date()
            last = next_first - _td(days=1)
        df = first; dt = last
    if 'ano que vem' in qs:
        y = today.year + 1; df = datetime(y,1,1).date(); dt = datetime(y,12,31).date()
    if 'ano passado' in qs:
        y = today.year - 1; df = datetime(y,1,1).date(); dt = datetime(y,12,31).date()
    return {
        'date_from': df.strftime('%Y-%m-%d') if df else None,
        'date_to': dt.strftime('%Y-%m-%d') if dt else None
    }

def _parse_pt_date_in_text(q: str) -> Optional[datetime.date]:
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", q)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)), tzinfo=PROJECT_TZ).date()
        except: return None
    meses = {
        "janeiro":1,"fevereiro":2,"marco":3,"março":3,"abril":4,"maio":5,"junho":6,
        "julho":7,"agosto":8,"setembro":9,"outubro":10,"novembro":11,"dezembro":12
    }
    m2 = re.search(r"\b(\d{1,2})\s+de\s+([a-zçãé]+)\b", q, flags=re.IGNORECASE)
    if m2:
        try:
            d = int(m2.group(1)); mes = meses.get(m2.group(2).lower()); year = _today().year
            if mes: return date(year, mes, d)
        except: pass
    if "hoje" in q: return _today()
    if "amanha" in q or "amanhã" in q: return _tomorrow()
    if "ontem" in q: return _yesterday()
    return None

def fb_fixture_generic(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{FOOTBALL_BASE}/fixtures", params=params, headers=_api_headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception as e:
        if DEBUG_LOG: print("[FOOTBALL][fixtures] erro:", e)
        return []

def _dt_local_from_fixture(fx: dict) -> Optional[datetime]:
    ts = fx.get("fixture", {}).get("timestamp")
    if isinstance(ts, int):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(PROJECT_TZ)
        except: pass
    iso = fx.get("fixture", {}).get("date")
    if not iso: return None
    try:
        return datetime.fromisoformat(iso.replace("Z","+00:00")).astimezone(PROJECT_TZ)
    except:
        return None

def fb_fixtures_by_league_date(league_id: int, d: datetime.date) -> List[Dict[str, Any]]:
    cache_key = f"fb:league:{league_id}:date:{d.isoformat()}"
    cached = _cache_get(cache_key)
    if cached is not None: return cached
    params = {"league": league_id, "date": _date_to_str(d), "season": _season_from_date(d), "timezone": "America/Sao_Paulo"}
    data = fb_fixture_generic(params)
    _cache_set(cache_key, data)
    return data

def fb_team_id(team_name: str) -> Optional[int]:
    try:
        tnorm = TEAM_ALIASES.get(team_name.lower(), team_name)
        r = requests.get(f"{FOOTBALL_BASE}/teams", params={"search": tnorm}, headers=_api_headers(), timeout=15)
        r.raise_for_status()
        for item in r.json().get("response", []):
            if item.get("team", {}).get("id"):
                return int(item["team"]["id"])
    except Exception as e:
        if DEBUG_LOG: print("[FOOTBALL][team_id] erro:", e)
    return None

def fb_fixture_last(team_id: int) -> Optional[dict]:
    arr = fb_fixture_generic({"team": team_id, "last": 1, "season": _tz_now_date().year, "timezone":"America/Sao_Paulo"})
    return arr[0] if arr else None

def fb_fixture_next(team_id: int) -> Optional[dict]:
    arr = fb_fixture_generic({"team": team_id, "next": 1, "season": _tz_now_date().year, "timezone":"America/Sao_Paulo"})
    return arr[0] if arr else None

# ordenar por data DESC (mais recente primeiro) e deduplicar
def fb_fixtures_last_n(team_id: int, n: int) -> List[dict]:
    n = max(1, min(int(n), 20))
    arr = fb_fixture_generic({"team": team_id, "last": n, "season": _tz_now_date().year, "timezone":"America/Sao_Paulo"})
    arr = sorted(arr, key=lambda x: (_dt_local_from_fixture(x) or datetime.min.replace(tzinfo=PROJECT_TZ)), reverse=True)
    seen, dedup = set(), []
    for it in arr:
        fid = it.get("fixture", {}).get("id")
        if fid and fid in seen: continue
        if fid: seen.add(fid)
        dedup.append(it)
    return dedup

def fb_fixtures_next_n(team_id: int, n: int) -> List[dict]:
    n = max(1, min(int(n), 20))
    arr = fb_fixture_generic({"team": team_id, "next": n, "season": _tz_now_date().year, "timezone":"America/Sao_Paulo"})
    arr = sorted(arr, key=lambda x: (_dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ)))
    seen, dedup = set(), []
    for it in arr:
        fid = it.get("fixture", {}).get("id")
        if fid and fid in seen: continue
        if fid: seen.add(fid)
        dedup.append(it)
    return dedup

def fb_fixture_on_date(team_id: int, date_: datetime.date) -> Optional[dict]:
    arr = fb_fixture_generic({"team": team_id, "date": _date_to_str(date_), "season": _season_from_date(date_), "timezone":"America/Sao_Paulo"})
    if not arr: return None
    if date_ < _tz_now_date():
        for fx in arr:
            g = fx.get("goals", {})
            if g.get("home") is not None and g.get("away") is not None:
                return fx
    return arr[0]

def fb_format(fx: dict) -> str:
    ldt = _dt_local_from_fixture(fx)
    data_txt = ldt.strftime("%d/%m/%Y %H:%M") if ldt else "-"
    home = fx.get("teams", {}).get("home", {}).get("name", "Time A")
    away = fx.get("teams", {}).get("away", {}).get("name", "Time B")
    hs, as_ = fx.get("goals", {}).get("home"), fx.get("goals", {}).get("away")
    league = fx.get("league", {}).get("name", "")
    return f"{home} {hs} x {as_} {away} — {league} — {data_txt}" if hs is not None and as_ is not None else f"{home} vs {away} — {league} — {data_txt}"

def format_list_football(fixtures: List[Dict[str, Any]]) -> str:
    if not fixtures: return "Nenhum jogo encontrado para esta data."
    return "\n".join(fb_format(fx) for fx in fixtures)

def normalize_football_fixture(item: Dict[str, Any]) -> Dict[str, Any]:
    fixture = item.get("fixture", {})
    league = item.get("league", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})
    score = item.get("score", {})
    return {
        "provider": "api-football",
        "sport": "football",
        "fixture_id": fixture.get("id"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "season": league.get("season"),
        "round": league.get("round"),
        "datetime_utc": fixture.get("date"),
        "status_long": fixture.get("status", {}).get("long"),
        "status_short": fixture.get("status", {}).get("short"),
        "elapsed": fixture.get("status", {}).get("elapsed"),
        "home": {
            "id": teams.get("home", {}).get("id"),
            "name": teams.get("home", {}).get("name"),
            "winner": teams.get("home", {}).get("winner"),
            "goals": goals.get("home"),
        },
        "away": {
            "id": teams.get("away", {}).get("id"),
            "name": teams.get("away", {}).get("name"),
            "winner": teams.get("away", {}).get("winner"),
            "goals": goals.get("away"),
        },
        "score": score,
    }

# ======= UTILITÁRIOS ADICIONAIS DE FUTEBOL (NOVOS) ==========================
def fixtures_to_table(fixtures: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for fx in fixtures or []:
        ldt = _dt_local_from_fixture(fx)
        data = ldt.strftime("%d/%m/%Y") if ldt else "-"
        hora = ldt.strftime("%H:%M") if ldt else "-"
        league = (fx.get("league", {}) or {}).get("name") or ""
        home = (fx.get("teams", {}) or {}).get("home", {}) or {}
        away = (fx.get("teams", {}) or {}).get("away", {}) or {}
        rows.append({
            "Data": data,
            "Campeonato": league,
            "Casa": home.get("name") or "",
            "Visitante": away.get("name") or "",
            "Hora": hora,
        })
    return rows

def table_markdown(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "Nenhum jogo encontrado."
    header = "| Data | Campeonato | Casa | Visitante | Hora |"
    sep    = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['Data']} | {r['Campeonato']} | {r['Casa']} | {r['Visitante']} | {r['Hora']} |")
    return "\n".join(lines)

def fb_fixtures_by_team_date(team_id: int, d: datetime.date) -> List[Dict[str, Any]]:
    cache_key = f"fb:team:{team_id}:date:{d.isoformat()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    arr = fb_fixture_generic({"team": team_id, "date": _date_to_str(d), "season": _season_from_date(d), "timezone":"America/Sao_Paulo"})
    _cache_set(cache_key, arr)
    return arr

def fb_team_ids_from_names(names: List[str]) -> List[int]:
    ids = []
    for nm in names:
        try:
            r = requests.get(f"{FOOTBALL_BASE}/teams", params={"search": nm}, headers=_api_headers(), timeout=15)
            r.raise_for_status()
            for item in r.json().get("response", []):
                tid = (item.get("team") or {}).get("id")
                if tid:
                    ids.append(int(tid))
                    break
        except Exception as e:
            if DEBUG_LOG: print("[FOOTBALL][team_ids_from_names] erro:", nm, e)
    # dedup
    out, seen = [], set()
    for t in ids:
        if t in seen: continue
        seen.add(t); out.append(t)
    return out

def fb_search_league_id(name: str) -> Optional[int]:
    """
    1º tenta no LIGA_MAP (apelidos/variações); se não achar, busca na API.
    """
    key = _norm(name)
    if key in LIGA_MAP:
        return LIGA_MAP[key]

    # tenta quebrar por palavras úteis
    for token in ["premier league", "premiere league", "pl", "epl", "la liga", "laliga",
                  "bundesliga", "ligue 1", "copa do brasil", "libertadores", "champions", "ucl",
                  "serie a", "série a", "serie b", "série b"]:
        if token in key and token in LIGA_MAP:
            return LIGA_MAP[token]

    # fallback: API
    try:
        r = requests.get(f"{FOOTBALL_BASE}/leagues", params={"search": name}, headers=_api_headers(), timeout=20)
        r.raise_for_status()
        resp = r.json().get("response", []) or []
        # prioriza a primeira com season disponível
        for item in resp:
            league = item.get("league") or {}
            seasons = item.get("seasons") or []
            if league.get("id") and seasons:
                return int(league["id"])
        for item in resp:
            league = item.get("league") or {}
            if league.get("id"):
                return int(league["id"])
    except Exception as e:
        if DEBUG_LOG: print("[FOOTBALL][search_league] erro:", e)
    return None
# ============================================================================

# ==== 8B) Parser de Saúde (revisado) ========================================
def extrair_comando_saude(question: str, client_id: str) -> dict:
    """
    Analisa a pergunta e retorna uma intent de saúde com campos úteis.
    PRIORIDADE DE INTENT (fixo):
      1) save (salvar/agendar/marcar/registrar/anotar)
      2) delete
      3) update
      4) next_return
      5) next
      6) last
      7) list
      8) none
    Campos:
      intent: "consulta" | "exame" | "procedimento" | "next" | "next_return" | "list" | "save" | "update" | "delete" | "last" | "none"
      tipo, especialidade, descricao, data_realizacao, data_retorno, data_ref, por_id, ultimo, set_fields
    """
    q = (question or "").lower()

    # tipo
    tipo = None
    if "procedim" in q:
        tipo = "procedimento"
    elif "exame" in q:
        tipo = "exame"
    elif "consulta" in q:
        tipo = "consulta"

    # especialidade
    espec = None
    for k, v in {
        "dermatolog": "Dermatologia",
        "cardiolog": "Cardiologia",
        "ginecolog": "Ginecologia",
        "ortoped": "Ortopedia",
        "oftalmolog": "Oftalmologia",
        "endocrin": "Endocrinologia",
        "psiquiat": "Psiquiatria",
        "psicol": "Psicologia",
        "odont": "Odontologia",
        "clinico": "Clínico Geral",
        "clínico": "Clínico Geral",
    }.items():
        if k in q:
            espec = v
            break

    # datas
    d = _parse_pt_date_in_text(q)
    data_realizacao = d.strftime("%Y-%m-%d") if d else None
    data_retorno = None
    if "retorno" in q and d:
        data_retorno = data_realizacao
        data_realizacao = None

    # id explícito
    por_id = None
    m_id = re.search(r"#(\d+)", q)
    if m_id:
        try:
            por_id = int(m_id.group(1))
        except:
            por_id = None

    ultimo_flag = any(k in q for k in ["último", "ultimo", "mais recente", "recente"])

    # ===== PRIORIDADES =====
    # 1) SAVE verbs
    if any(k in q for k in ["salvar","salva","registrar","registra","adicionar","anotar","agendar","marcar"]):
        return {
            "intent": "save",
            "tipo": (tipo or "consulta"),
            "especialidade": espec,
            "descricao": None,
            "data_realizacao": data_realizacao,
            "data_retorno": data_retorno,
            "data_ref": data_realizacao or data_retorno,
            "por_id": None,
            "ultimo": False,
            "set_fields": {}
        }

    # 2) DELETE
    if any(k in q for k in ["apagar","deletar","remover","excluir"]):
        return {
            "intent": "delete",
            "tipo": tipo,
            "especialidade": espec,
            "descricao": None,
            "data_realizacao": None,
            "data_retorno": None,
            "data_ref": data_realizacao or data_retorno,
            "por_id": por_id,
            "ultimo": ultimo_flag,
            "set_fields": {}
        }

    # 3) UPDATE
    if any(k in q for k in ["atualizar","editar","corrigir","mudar"]):
        return {
            "intent": "update",
            "tipo": tipo,
            "especialidade": espec,
            "descricao": None,
            "data_realizacao": data_realizacao,
            "data_retorno": data_retorno,
            "data_ref": data_realizacao or data_retorno,
            "por_id": por_id,
            "ultimo": ultimo_flag,
            "set_fields": {}
        }

    # 4) Próximo retorno
    if any(k in q for k in ["próximo retorno","proximo retorno"]):
        return {"intent": "next_return", "tipo": tipo, "especialidade": espec, "descricao": None,
                "data_realizacao": None, "data_retorno": None, "data_ref": None,
                "por_id": None, "ultimo": False, "set_fields": {}}

    # 5) Próximo compromisso
    if any(k in q for k in ["próxima consulta","proxima consulta","próximo exame","proximo exame",
                             "próximo procedimento","proximo procedimento","minha próxima consulta","minha proxima consulta"]):
        return {"intent": "next", "tipo": tipo or "consulta", "especialidade": espec, "descricao": None,
                "data_realizacao": None, "data_retorno": None, "data_ref": None,
                "por_id": None, "ultimo": False, "set_fields": {}}

    # 6) Último
    if any(k in q for k in ["última consulta","ultima consulta","último exame","ultimo exame","último procedimento","ultimo procedimento"]):
        return {"intent": "last", "tipo": tipo or "consulta", "especialidade": espec, "descricao": None,
                "data_realizacao": None, "data_retorno": None, "data_ref": None,
                "por_id": None, "ultimo": True, "set_fields": {}}

    # 7) Listar (fallback quando mencionar 'consulta' e não houver outra intent)
    if any(k in q for k in ["listar","liste","histórico","historico","últimas","ultimas","últimos","ultimos"]) or "consulta" in q or "exame" in q or "procedimento" in q:
        return {"intent": "list", "tipo": tipo or "consulta", "especialidade": espec, "descricao": None,
                "data_realizacao": None, "data_retorno": None, "data_ref": None,
                "por_id": None, "ultimo": ultimo_flag, "set_fields": {}}

    # 8) none
    return {"intent": "none", "tipo": tipo, "especialidade": espec, "descricao": None,
            "data_realizacao": data_realizacao, "data_retorno": data_retorno,
            "data_ref": data_realizacao or data_retorno, "por_id": por_id, "ultimo": ultimo_flag, "set_fields": {}}

# ==== 9) SAÚDE — Helena (CRUD) ==============================================
HEALTH_TYPES = ["consulta","exame","procedimento"]
SPECIALTY_HINTS = {
    "dermatolog": "Dermatologia",
    "cardiolog": "Cardiologia",
    "ginecolog": "Ginecologia",
    "ortoped": "Ortopedia",
    "oftalmolog": "Oftalmologia",
    "endocrin": "Endocrinologia",
    "psiquiat": "Psiquiatria",
    "psicol": "Psicologia",
    "odont": "Odontologia",
    "clinico": "Clínico Geral",
    "clínico": "Clínico Geral",
}

def ensure_only_helena(client_id: str) -> bool:
    return (client_id or "").lower() == "helena"

def parse_date_any_pt(text: str) -> Optional[str]:
    d = _parse_pt_date_in_text(text.lower())
    return d.strftime("%Y-%m-%d") if d else None

def format_evento(row: sqlite3.Row) -> str:
    """Formata um registro de saúde com: #id — Tipo (Especialidade; Descrição) | Agendado/Realizado/Retorno: dd/mm/aaaa"""
    def _fmt(d: str | None) -> str:
        if not d:
            return ""
        try:
            y, m, d2 = d.split("-")
            return f"{d2}/{m}/{y}"
        except Exception:
            return d  # fallback

    tipo = (row["tipo"] or "").capitalize()

    # Extras: junta especialidade e descrição dentro de parênteses (se houver)
    extras = []
    try:
        if row["especialidade"]:
            extras.append(str(row["especialidade"]).strip())
    except Exception:
        pass
    try:
        if row["descricao"]:
            extras.append(str(row["descricao"]).strip())
    except Exception:
        pass
    extra_str = f" ({'; '.join(extras)})" if extras else ""

    data_real = row["data_realizacao"]
    data_ret  = row["data_retorno"]

    hoje = _tz_now_date().strftime("%Y-%m-%d")

    partes_status = []
    try:
        if data_real:
            if data_real >= hoje and not data_ret:
                partes_status.append(f"Agendado: {_fmt(data_real)}")
            else:
                partes_status.append(f"Realizado: {_fmt(data_real)}")
        if data_ret:
            partes_status.append(f"Retorno: {_fmt(data_ret)}")
    except Exception:
        if data_real: partes_status.append(f"Realizado: {_fmt(data_real)}")
        if data_ret:  partes_status.append(f"Retorno: {_fmt(data_ret)}")

    sufixo = f" | {' | '.join(partes_status)}" if partes_status else ""
    return f"#{row['id']} — {tipo}{extra_str}{sufixo}"


def inserir_evento_saude(conn, client_id_alvo: str, tipo: str, especialidade: Optional[str], descricao: Optional[str],
                         data_realizacao: Optional[str], data_retorno: Optional[str]) -> bool:
    aluno_id = get_user_id_by_client_id(conn, client_id_alvo)
    if not aluno_id:
        return False
    # sanitize trailing 'dia' em campos capturados
    especialidade = _strip_trailing_dia(especialidade)
    descricao = _strip_trailing_dia(descricao)
    agora = datetime.now().isoformat(timespec="seconds")
    conn.execute("""
        INSERT INTO consultas_exames (aluno_id, tipo, especialidade, descricao, data_realizacao, data_retorno, criado_em)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (aluno_id, tipo, especialidade, descricao, data_realizacao, data_retorno, agora))
    conn.commit()
    return True

def proximo_evento_saude(conn, client_id: str, tipo: Optional[str], so_retorno: bool=False,
                         especialidade: Optional[str] = None, descricao: Optional[str] = None) -> Optional[sqlite3.Row]:
    aluno_id = get_user_id_by_client_id(conn, client_id)
    if not aluno_id:
        return None
    today = _today().strftime("%Y-%m-%d")
    cur = conn.cursor()

    base = "FROM consultas_exames WHERE aluno_id = ?"
    params = [aluno_id]

    if so_retorno:
        sql = f"SELECT * {base} AND data_retorno IS NOT NULL AND data_retorno >= ?"
        params.append(today)
    else:
        sql = f"SELECT * {base} AND data_retorno IS NOT NULL AND data_retorno >= ?"
        params.append(today)

    if tipo:
        sql += " AND tipo = ?"; params.append(tipo)
    if especialidade:
        sql += " AND LOWER(COALESCE(especialidade,'')) = LOWER(?)"; params.append(especialidade)
    if descricao:
        sql += " AND LOWER(COALESCE(descricao,'')) LIKE LOWER(?)"; params.append(f"%{descricao}%")

    sql += " ORDER BY data_retorno ASC, id ASC LIMIT 1"
    cur.execute(sql, params)
    row = cur.fetchone()
    if row or so_retorno:
        return row

    # fallback: busca por realização futura
    sql2 = f"SELECT * {base} AND data_realizacao IS NOT NULL AND data_realizacao >= ?"
    params2 = [aluno_id, today]
    if tipo:
        sql2 += " AND tipo = ?"; params2.append(tipo)
    if especialidade:
        sql2 += " AND LOWER(COALESCE(especialidade,'')) = LOWER(?)"; params2.append(especialidade)
    if descricao:
        sql2 += " AND LOWER(COALESCE(descricao,'')) LIKE LOWER(?)"; params2.append(f"%{descricao}%")
    sql2 += " ORDER BY data_realizacao ASC, id ASC LIMIT 1"
    cur.execute(sql2, params2)
    return cur.fetchone()
def ultimo_evento_saude(conn, client_id: str, tipo: Optional[str]) -> Optional[sqlite3.Row]:
    aluno_id = get_user_id_by_client_id(conn, client_id)
    if not aluno_id:
        return None
    today = _today().strftime("%Y-%m-%d")
    cur = conn.cursor()
    filtro_tipo = " AND tipo = ?" if tipo else ""
    params = [aluno_id, today] + ([tipo] if tipo else [])
    # último por retorno <= hoje
    cur.execute(f"""
        SELECT * FROM consultas_exames
         WHERE aluno_id = ?
           AND (
                (data_retorno IS NOT NULL AND data_retorno <= ?)
                OR (data_realizacao IS NOT NULL AND data_realizacao <= ?)
           )
           {filtro_tipo}
         ORDER BY COALESCE(data_retorno, data_realizacao) DESC, id DESC
         LIMIT 1
    """, [aluno_id, today, today] + ([tipo] if tipo else []))
    return cur.fetchone()

def listar_eventos_saude(conn, client_id: str, tipo: Optional[str], limite: int = 50,
                         especialidade: Optional[str] = None, descricao: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[sqlite3.Row]:
    aluno_id = get_user_id_by_client_id(conn, client_id)
    if not aluno_id:
        return []
    cur = conn.cursor()
    sql = """
        SELECT *
          FROM consultas_exames
         WHERE aluno_id = ?
    """
    params = [aluno_id]
    # filtros de período (date_from/date_to)
    if date_from:
        sql += " AND ( (data_retorno >= ?) OR (data_realizacao >= ?) )"
        params.extend([date_from, date_from])
    if date_to:
        sql += " AND ( (data_retorno <= ?) OR (data_realizacao <= ?) )"
        params.extend([date_to, date_to])

    if tipo:
        sql += " AND tipo = ?"
        params.append(tipo)
    if especialidade:
        sql += " AND LOWER(COALESCE(especialidade,'')) = LOWER(?)"
        params.append(especialidade)
    if descricao:
        sql += " AND LOWER(COALESCE(descricao,'')) LIKE LOWER(?)"
        params.append(f"%{descricao}%")
    sql += " ORDER BY COALESCE(data_retorno, data_realizacao) DESC, id DESC"
    sql += f" LIMIT {int(limite)}"
    cur.execute(sql, params)
    return cur.fetchall()
def buscar_eventos_para_editar(conn, client_id: str, tipo: Optional[str], especialidade: Optional[str],
                               data_ref: Optional[str], ultimo: bool, por_id: Optional[int]) -> List[sqlite3.Row]:
    aluno_id = get_user_id_by_client_id(conn, client_id)
    if not aluno_id: return []
    cur = conn.cursor()
    if por_id:
        cur.execute("SELECT * FROM consultas_exames WHERE id=? AND aluno_id=?", (por_id, aluno_id))
        row = cur.fetchone()
        return [row] if row else []
    sql = "SELECT * FROM consultas_exames WHERE aluno_id=?"
    params = [aluno_id]
    if tipo:
        sql += " AND tipo=?"; params.append(tipo)
    if especialidade:
        sql += " AND (LOWER(especialidade)=LOWER(?) )"; params.append(especialidade)
    if data_ref:
        sql += " AND (data_realizacao=? OR data_retorno=?)"; params.extend([data_ref, data_ref])
    sql += " ORDER BY id DESC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    if ultimo and rows:
        return [rows[0]]
    return rows

def atualizar_evento_saude(conn, row_id: int, aluno_id: int, fields: Dict[str, Any]) -> bool:
    sets = []
    vals = []
    for k in ["tipo","especialidade","descricao","data_realizacao","data_retorno"]:
        if fields.get(k) is not None:
            sets.append(f"{k}=?"); vals.append(fields[k])
    if not sets: return False
    vals.extend([row_id, aluno_id])
    sql = f"UPDATE consultas_exames SET {', '.join(sets)} WHERE id=? AND aluno_id=?"
    conn.execute(sql, vals)
    conn.commit()
    return True

def deletar_evento_saude(conn, row_id: int, aluno_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("DELETE FROM consultas_exames WHERE id=? AND aluno_id=?", (row_id, aluno_id))
    conn.commit()
    return cur.rowcount > 0


# ==== 10) Projeções — Série B (Heurístico + Log) ============================
def api_f(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.get(f"{FOOTBALL_BASE}{path}", params=params, headers=_api_headers(), timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if DEBUG_LOG: print("[API_F] erro:", e)
        return {"errors": [repr(e)], "response": []}

def fb_headtohead(home_id: int, away_id: int, last_n: int) -> List[Dict[str, Any]]:
    params = {"h2h": f"{home_id}-{away_id}", "last": max(1, min(int(last_n), 20)), "timezone": "America/Sao_Paulo"}
    data = api_f("/fixtures/headtohead", params)
    return data.get("response", []) if isinstance(data, dict) else []

def team_stats_from_last(fixtures: List[Dict[str, Any]], team_id: int) -> Dict[str, Any]:
    games = 0
    goals_for = goals_against = total_goals = conceded_sum = 0.0
    under35_count = 0
    form_points = 0  # vitória=3, empate=1

    for fx in fixtures:
        teams = fx.get("teams", {})
        goals = fx.get("goals", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        gh = goals.get("home"); ga = goals.get("away")
        if gh is None or ga is None: continue
        if home.get("id") == team_id:
            my_goals = gh; op_goals = ga; my_winner = home.get("winner")
        elif away.get("id") == team_id:
            my_goals = ga; op_goals = gh; my_winner = away.get("winner")
        else:
            continue
        games += 1
        goals_for += my_goals
        goals_against += op_goals
        total_goals += (gh + ga)
        conceded_sum += op_goals
        if (gh + ga) < 4: under35_count += 1
        if my_winner is True: form_points += 3
        elif my_winner is None: form_points += 1

    if games == 0:
        return {"games": 0, "avg_for": 0.0, "avg_against": 0.0, "avg_total": 0.0, "pct_under35": 0.0, "form_points": 0.0, "conceded_avg": 0.0}
    return {
        "games": games,
        "avg_for": goals_for / games,
        "avg_against": goals_against / games,
        "avg_total": total_goals / games,
        "pct_under35": under35_count / games,
        "form_points": form_points / games,  # 0..3
        "conceded_avg": conceded_sum / games,
    }

def h2h_stats(fixtures_h2h: List[Dict[str, Any]], home_id: int, away_id: int) -> Dict[str, Any]:
    games = under35 = home_wins = away_wins = draws = 0
    for fx in fixtures_h2h:
        teams = fx.get("teams", {})
        goals = fx.get("goals", {})
        gh = goals.get("home"); ga = goals.get("away")
        if gh is None or ga is None: continue
        games += 1
        if gh + ga < 4: under35 += 1
        hwin = teams.get("home", {}).get("winner")
        awin = teams.get("away", {}).get("winner")
        if hwin is True: home_wins += 1
        elif awin is True: away_wins += 1
        else: draws += 1
    dominance = "home" if home_wins > away_wins else ("away" if away_wins > home_wins else "none")
    return {"games": games, "pct_under35": (under35 / games) if games else 0.0, "home_wins": home_wins, "away_wins": away_wins, "draws": draws, "dominance": dominance}

def make_projection(stats_home: Dict[str, Any], stats_away: Dict[str, Any], h2h: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    # combinação para Under 3.5
    if stats_home["games"] and stats_away["games"]:
        combined_avg_total = (stats_home["avg_total"] + stats_away["avg_total"]) / 2
        combined_pct_under = (stats_home["pct_under35"] + stats_away["pct_under35"]) / 2
    else:
        combined_avg_total = 0.0
        combined_pct_under = 0.0

    under_ok = (
        combined_avg_total <= UNDER35_AVG_MAX
        and combined_pct_under >= UNDER35_PCT_MIN
        and max(stats_home["avg_total"], stats_away["avg_total"]) < 3.2
    )

    conf_under = 0.0
    if stats_home["games"] and stats_away["games"]:
        conf_under = min(
            1.0,
            max(
                0.0,
                0.5
                + (UNDER35_AVG_MAX - combined_avg_total) * 0.3
                + (combined_pct_under - UNDER35_PCT_MIN) * 0.4
            ),
        )

    # força relativa p/ dupla chance
    home_score = stats_home["form_points"] * FORM_WEIGHT - stats_home["conceded_avg"] * DEF_WEIGHT + HOME_ADV
    away_score = stats_away["form_points"] * FORM_WEIGHT - stats_away["conceded_avg"] * DEF_WEIGHT
    delta = home_score - away_score  # + favorece mandante; - favorece visitante

    # decisão principal (com threshold)
    dc_label, dc_conf, dc_side = None, 0.0, None
    if delta >= DC_THRESHOLD:
        dc_label = "1X"; dc_side = "home"
        dc_conf = min(1.0, 0.5 + (delta) / 2.0)
    elif -delta >= DC_THRESHOLD:
        dc_label = "X2"; dc_side = "away"
        dc_conf = min(1.0, 0.5 + (-delta) / 2.0)

    # reforço H2H
    if h2h and h2h.get("games", 0) > 0:
        if under_ok and h2h.get("pct_under35", 0.0) >= UNDER35_PCT_MIN:
            conf_under = min(1.0, conf_under + H2H_UNDER_BOOST)
        if dc_label == "1X" and h2h.get("dominance") == "home":
            dc_conf = min(1.0, dc_conf + H2H_DC_BOOST)
        elif dc_label == "X2" and h2h.get("dominance") == "away":
            dc_conf = min(1.0, dc_conf + H2H_DC_BOOST)

    # "lean" (sinal fraco) quando não bate threshold
    lean_label = "1X" if delta >= 0 else "X2"
    lean_side = "home" if delta >= 0 else "away"
    lean_conf = round(min(0.49, max(0.10, abs(delta) / 2.0)), 2)

    return {
        "under35": bool(under_ok),
        "conf_under35": round(conf_under, 2),
        "double_chance": dc_label,               # só preenchido se passou o threshold
        "conf_double_chance": round(dc_conf, 2),
        "dc_side": dc_side,                      # 'home' | 'away' | None
        "lean_double_chance": lean_label,        # sempre 1X ou X2 (sinal fraco)
        "lean_conf_double_chance": lean_conf,    # 0.10 .. 0.49
        "lean_dc_side": lean_side,               # 'home' | 'away'
        "combined": {
            "avg_total_goals": round(combined_avg_total, 2),
            "pct_under35": round(combined_pct_under, 2),
        },
    }

def save_projection_row(conn, dref: str, league_id: int, home_id: int, away_id: int,
                        home: str, away: str, proj: Dict[str, Any]) -> None:
    conn.execute("""
        INSERT INTO projecoes (
            created_at, date_ref, league_id, home_id, away_id, home, away,
            under35, conf_under, double_chance, conf_dc, avg_total, pct_under, raw_json
        )
        VALUES (
            datetime('now'), ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        dref, league_id, home_id, away_id, home, away,
        1 if proj.get("under35") else 0,
        float(proj.get("conf_under35", 0.0)),
        proj.get("double_chance"),
        float(proj.get("conf_double_chance", 0.0)),
        float(proj.get("combined", {}).get("avg_total_goals", 0.0)),
        float(proj.get("combined", {}).get("pct_under35", 0.0)),
        json.dumps(proj, ensure_ascii=False)
    ))
    conn.commit()

def serie_b_projections_calc(date_str: Optional[str]=None, last:int=5, h2h_last:int=5) -> Dict[str, Any]:
    # data alvo
    dref = (date_str or _today().strftime("%Y-%m-%d"))
    d_obj = datetime.strptime(dref, "%Y-%m-%d").date()
    season = _tz_now_date().year

    # jogos da Série B no dia
    fixtures = fb_fixtures_by_league_date(LEAGUE_IDS["br_serie_b"], d_obj)

    games = []
    for fx in fixtures:
        league_name = fx.get("league", {}).get("name")
        iso = fx.get("fixture", {}).get("date")
        home = fx.get("teams", {}).get("home", {}) or {}
        away = fx.get("teams", {}).get("away", {}) or {}
        home_id, away_id = home.get("id"), away.get("id")
        home_name, away_name = home.get("name"), away.get("name")
        if not (home_id and away_id and home_name and away_name):
            continue

        # últimos N de cada equipe
        last_home = fb_fixtures_last_n(home_id, last)
        last_away = fb_fixtures_last_n(away_id, last)
        s_home = team_stats_from_last(last_home, home_id)
        s_away = team_stats_from_last(last_away, away_id)

        # H2H
        h2h_list = fb_headtohead(home_id, away_id, h2h_last)
        h2h = h2h_stats(h2h_list, home_id, away_id) if h2h_list else {"games": 0}

        # projeção
        proj = make_projection(s_home, s_away, h2h=h2h)

        games.append({
            "match": {
                "league": league_name,
                "date_utc": iso,
                "home_id": home_id, "away_id": away_id,
                "home": home_name, "away": away_name
            },
            "projection": proj,
            "metrics": {"recent": {"home": s_home, "away": s_away}, "h2h": h2h},
            "notes": "Heurística determinística (recentes) + reforço H2H."
        })

    return {
        "date": dref,
        "season": season,
        "league_id": LEAGUE_IDS["br_serie_b"],
        "games": games,
        "thresholds": {
            "UNDER35_AVG_MAX": UNDER35_AVG_MAX,
            "UNDER35_PCT_MIN": UNDER35_PCT_MIN,
            "FORM_WEIGHT": FORM_WEIGHT,
            "DEF_WEIGHT": DEF_WEIGHT,
            "HOME_ADV": HOME_ADV,
            "DC_THRESHOLD": DC_THRESHOLD,
            "H2H_LAST": H2H_LAST,
            "H2H_UNDER_BOOST": H2H_UNDER_BOOST,
            "H2H_DC_BOOST": H2H_DC_BOOST
        }
    }

def formatar_resposta_projecoes(proj_payload: Dict[str, Any]) -> str:
    jogos = proj_payload.get("games", [])
    if not jogos:
        return "Não encontrei jogos da Série B nessa data."

    linhas = [f"Projeções Série B — {proj_payload.get('date')}:"]
    for g in jogos:
        m = g["match"]
        p = g["projection"]
        comb = p.get("combined", {}) or {}
        home_name = m.get("home") or "Mandante"
        away_name = m.get("away") or "Visitante"

        # Under 3.5
        under_txt = "Under 3.5 ✅" if p["under35"] else "Under 3.5 ❌"
        under_txt += f" ({int(p['conf_under35'] * 100)}%)"

        # Dupla chance
        if p.get("double_chance"):
            team_side = home_name if p.get("dc_side") == "home" else away_name
            dc_txt = f"Dupla chance: {p['double_chance']} ({team_side}) ({int(p['conf_double_chance'] * 100)}%)"
        else:
            team_side = home_name if p.get("lean_dc_side") == "home" else away_name
            dc_txt = f"Dupla chance sugerida: {p['lean_double_chance']} ({team_side}) ({int(p['lean_conf_double_chance'] * 100)}%)"

        extra = f" | gols médios (comb.): {comb.get('avg_total_goals', 0)} | %Under3.5 (comb.): {int(round(comb.get('pct_under35', 0)*100))}%"
        linhas.append(f"- {home_name} x {away_name} — {under_txt}, {dc_txt}{extra}")

    linhas.append("\nObs.: estimativa estatística (não garante resultado).")
    return "\n".join(linhas)

# ==== 11A) Jogos óbvios (novo modelo) =======================================
def _obv_make_strength(stats_home: Dict[str, Any], stats_away: Dict[str, Any], h2h: Optional[Dict[str, Any]]=None):
    """
    Calcula lado favorito e 'força' (0..1) com base na mesma família de sinais do
    modelo de Série B (form vs defesa), mas simplificado e voltado para 'jogos óbvios'.
    - Se delta >= 0  -> favorito = mandante
    - Se delta < 0   -> favorito = visitante
    - strength: mapeia |delta| para [0..1], com limite superior em ~2.0
    - reforço leve por H2H quando há dominância do mesmo lado
    """
    # Pesos: reaproveitamos FORM_WEIGHT/DEF_WEIGHT/HOME_ADV já existentes
    home_score = stats_home["form_points"] * FORM_WEIGHT - stats_home["conceded_avg"] * DEF_WEIGHT + HOME_ADV
    away_score = stats_away["form_points"] * FORM_WEIGHT - stats_away["conceded_avg"] * DEF_WEIGHT
    delta = home_score - away_score  # >0 favorece mandante

    side = "home" if delta >= 0 else "away"
    base_strength = min(1.0, max(0.0, abs(delta) / 2.0))  # normaliza |delta| para 0..1 (2.0 ~ muito forte)

    # reforço leve por H2H
    if h2h and h2h.get("games", 0) > 0:
        dom = h2h.get("dominance")
        if (side == "home" and dom == "home") or (side == "away" and dom == "away"):
            base_strength = min(1.0, base_strength + 0.1)

    return side, round(base_strength, 2), delta


def _obv_format_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Nenhum jogo óbvio encontrado."
    header = "| Data | Campeonato | Casa | Visitante | Hora | Favorito | Força |"
    sep    = "|---|---|---|---|---|---|---|"
    out = [header, sep]
    for r in rows:
        out.append(f"| {r['data']} | {r['league']} | {r['home']} | {r['away']} | {r['hora']} | {r['fav']} | {r['strength']} |")
    return "\n".join(out)


def obvious_games_calc(date_str: Optional[str]=None,
                       leagues: Optional[List[int]]=None,
                       last:int=5,
                       h2h_last:int=3,
                       min_strength:float=0.75) -> Dict[str, Any]:
    """
    Seleciona 'jogos óbvios' para a data/ligas:
      - Ligas padrão (se não vier): Brasileirão Série A (71) + Série B (72)
      - Para cada jogo do dia, coleta últimos 'last' de cada time e H2H (curto)
      - Calcula lado favorito + força (0..1)
      - Mantém apenas os que tiverem força >= min_strength
    Retorna payload com lista ordenada por força (desc) + tabela markdown.
    """
    dref = (date_str or _today().strftime("%Y-%m-%d"))
    d_obj = datetime.strptime(dref, "%Y-%m-%d").date()
    leagues = leagues or [LEAGUE_IDS["br_serie_a"], LEAGUE_IDS["br_serie_b"]]

    all_fixtures: List[Dict[str, Any]] = []
    for lid in leagues:
        all_fixtures += fb_fixtures_by_league_date(lid, d_obj) or []

    results = []
    for fx in all_fixtures:
        league_name = (fx.get("league") or {}).get("name") or ""
        iso = (fx.get("fixture") or {}).get("date")
        ldt = _dt_local_from_fixture(fx)
        data_txt = ldt.strftime("%d/%m/%Y") if ldt else "-"
        hora_txt = ldt.strftime("%H:%M") if ldt else "-"

        t_home = (fx.get("teams") or {}).get("home") or {}
        t_away = (fx.get("teams") or {}).get("away") or {}
        hid, aid = t_home.get("id"), t_away.get("id")
        hname, aname = t_home.get("name") or "", t_away.get("name") or ""
        if not (hid and aid):  # segurança
            continue

        last_home = fb_fixtures_last_n(hid, last)
        last_away = fb_fixtures_last_n(aid, last)
        s_home = team_stats_from_last(last_home, hid)
        s_away = team_stats_from_last(last_away, aid)

        h2h_list = fb_headtohead(hid, aid, h2h_last)
        h2h = h2h_stats(h2h_list, hid, aid) if h2h_list else {"games": 0}

        side, strength, delta = _obv_make_strength(s_home, s_away, h2h)

        if strength >= float(min_strength):
            fav_name = hname if side == "home" else aname
            results.append({
                "league": league_name,
                "date_utc": iso,
                "data_local": data_txt,
                "hora_local": hora_txt,
                "home": hname,
                "away": aname,
                "favorite_side": side,     # 'home' | 'away'
                "favorite_name": fav_name,
                "strength": strength,      # 0..1
                "delta_raw": round(delta, 3),
                "recent": {"home": s_home, "away": s_away},
                "h2h": h2h,
            })

    results.sort(key=lambda x: x["strength"], reverse=True)

    # tabela amigável
    table_rows = [{
        "data": r["data_local"],
        "league": r["league"],
        "home": r["home"],
        "away": r["away"],
        "hora": r["hora_local"],
        "fav": r["favorite_name"],
        "strength": f"{int(r['strength']*100)}%"
    } for r in results]

    return {
        "date": dref,
        "count": len(results),
        "min_strength": float(min_strength),
        "games": results,
        "table_markdown": _obv_format_table(table_rows),
        "notes": "Heurística rápida: form vs defesa + bônus leve de H2H. Strength 0..1."
    }

# ==== 12) Responder Esportes (por time) =====================================
def responder_esporte(team: str, action: str, date_str: Optional[str], last_n: Optional[int]=None) -> Optional[str]:
    if not SPORTS_ENABLED or not SPORTS_KEY:
        return "Esportes desabilitado ou sem API key configurada."
    team_id = fb_team_id(team)
    if not team_id:
        return "Não encontrei o time solicitado."

    if last_n is not None and action == "last":
        fixtures = fb_fixtures_last_n(team_id, last_n)
        if not fixtures:
            return "Não encontrei jogos recentes."
        txt = "\n".join(f"{i+1}. {fb_format(fx)}" for i, fx in enumerate(fixtures))
        return f"Aqui estão os últimos {last_n} jogos do {team.title()}:\n\n{txt}\n\nFonte: API-Football"

    if action == "last":
        fx = fb_fixture_last(team_id)
    elif action == "next":
        fx = fb_fixture_next(team_id)
    elif action == "on_date":
        alvo = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else _today()
        fx = fb_fixture_on_date(team_id, alvo)
        if not fx:
            fx = fb_fixture_next(team_id) or fb_fixture_last(team_id)
    else:
        fx = None

    if not fx:
        return "Não encontrei um jogo correspondente."
    frase = fb_format(fx)
    return f"{frase}\n\nFonte: API-Football"


# ==== 13) Endpoints ==========================================================
@app.get("/health")
def health():
    conn = get_db()
    ensure_health_schema(conn)
    ensure_projections_schema(conn)
    ensure_grades_schema(conn)  # <<< garantir notas/medias
    try:
        return jsonify({
            "status": "ok",
            "db": DB_PATH,
            "model": MODEL_NAME,
            "web_search": bool(config.get("features", {}).get("web_search", False)),
            "sports_enabled": SPORTS_ENABLED,
            "sports_key_present": bool(SPORTS_KEY),
            "sports_provider": SPORTS_PROVIDER,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "health_tables": True,
            "grades_tables": True,
            "projections": {
                "UNDER35_AVG_MAX": UNDER35_AVG_MAX,
                "UNDER35_PCT_MIN": UNDER35_PCT_MIN,
                "FORM_WEIGHT": FORM_WEIGHT,
                "DEF_WEIGHT": DEF_WEIGHT,
                "HOME_ADV": HOME_ADV,
                "DC_THRESHOLD": DC_THRESHOLD,
                "H2H_LAST": H2H_LAST,
                "H2H_UNDER_BOOST": H2H_UNDER_BOOST,
                "H2H_DC_BOOST": H2H_DC_BOOST
            }
        })
    finally:
        conn.close()


@app.get("/v1/football/league/day")
def football_league_day():
    if not SPORTS_ENABLED or not SPORTS_KEY:
        return jsonify({"error": "sports desabilitado ou sem API key"}), 400
    league_id = request.args.get("league_id", type=int)
    when = (request.args.get("when") or "today").lower()
    date_str = request.args.get("date")
    if not league_id:
        return jsonify({"error": "param 'league_id' é obrigatório"}), 400
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "date deve ser YYYY-MM-DD"}), 400
    else:
        if when in ("yesterday", "ontem"):
            target_date = _yesterday()
        elif when in ("tomorrow", "amanha", "amanhã"):
            target_date = _tomorrow()
        else:
            target_date = _today()
    fixtures = fb_fixtures_by_league_date(league_id, target_date)
    normalized = [normalize_football_fixture(it) for it in fixtures]
    fixtures = sorted(fixtures, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ))
    text_list = format_list_football(fixtures)
    rows = fixtures_to_table(fixtures)
    league_name = None
    if fixtures:
        league_name = (fixtures[0].get("league") or {}).get("name")
    return jsonify({
        "league_id": league_id,
        "league_name": league_name,
        "date": target_date.strftime("%Y-%m-%d"),
        "count": len(normalized),
        "goomi": normalized,
        "text": text_list,
        "table": rows,
        "table_markdown": table_markdown(rows),
        "source": "API-Football",
        "cache_ttl_seconds": CACHE_TTL_SECONDS
    })


@app.get("/v1/football/serie-b/projections")
def serie_b_projections_http():
    if not SPORTS_ENABLED or not SPORTS_KEY:
        return jsonify({"error": "sports desabilitado ou sem API key"}), 400
    d = request.args.get("date") or _today().strftime("%Y-%m-%d")
    last = int(request.args.get("last") or 5)
    h2h_last = int(request.args.get("h2h_last") or H2H_LAST)

    payload = serie_b_projections_calc(d, last=last, h2h_last=h2h_last)

    # log em SQLite
    conn = get_db()
    ensure_projections_schema(conn)
    try:
        for g in payload.get("games", []):
            m = g["match"]; p = g["projection"]
            save_projection_row(
                conn,
                dref=payload["date"],
                league_id=payload["league_id"],
                home_id=m.get("home_id") or 0,
                away_id=m.get("away_id") or 0,
                home=m.get("home") or "",
                away=m.get("away") or "",
                proj=p
            )
    finally:
        conn.close()

    return jsonify(payload)

@app.get("/v1/football/obvious")
def football_obvious_games():
    if not SPORTS_ENABLED or not SPORTS_KEY:
        return jsonify({"error": "sports desabilitado ou sem API key"}), 400

    d = request.args.get("date") or _today().strftime("%Y-%m-%d")
    leagues_param = request.args.get("leagues")  # ex.: "71,72"
    last = int(request.args.get("last") or 5)
    h2h_last = int(request.args.get("h2h_last") or 3)
    min_strength = float(request.args.get("min_strength") or 0.75)

    leagues = None
    if leagues_param:
        try:
            leagues = [int(x.strip()) for x in leagues_param.split(",") if x.strip().isdigit()]
        except Exception:
            leagues = None

    payload = obvious_games_calc(
        date_str=d,
        leagues=leagues,
        last=last,
        h2h_last=h2h_last,
        min_strength=min_strength
    )

    return jsonify(payload)

# ==== 14) /ask ===============================================================
def _sanitize_league_phrase(txt: str) -> str:
    """Remove palavras de tempo/sufixos comuns de uma possível frase de liga."""
    if not txt:
        return txt
    t = txt.lower()
    # corta em conectores comuns
    cut_tokens = [" hoje", " amanha", " amanhã", " ontem", " agora", " desse", " desta", " dessa", " de ", " do "]
    for tok in cut_tokens:
        if tok in t:
            t = t.split(tok)[0]
    return t.strip(" ,.;:-")



# ==== Manual / Ajuda =========================================================
def _manual_saude() -> str:
    return (
        "Manual — Saúde\n\n"
        "• Salvar consulta/exame: \"salvar minha próxima consulta no dermatologista dia 15/09/2025\"\n"
        "• Salvar retorno: \"registrar meu retorno de cardiologia para 10/10/2025\"\n"
        "• Listar últimas: \"liste minhas últimas consultas\"\n"
        "• Próximo compromisso: \"qual é minha próxima consulta?\"\n"
        "• Próximo retorno: \"qual é meu próximo retorno?\"\n"
        "• Último que ocorreu: \"qual foi minha última consulta?\"\n"
        "• Apagar por ID: \"apague a consulta #12\"\n"
        "Observação: por privacidade, só a Helena consegue ver e salvar dados de saúde."
    )

def _manual_notas() -> str:
    return (
        "Manual — Notas\n\n"
        "• Ver todas: \"quais são as minhas notas?\"\n"
        "• Por disciplina: \"minhas notas de Matemática\"\n"
        "• Por período: \"minhas notas do P2\"\n"
        "• Salvar nota: \"salva Português P2 com 8,5\"\n"
        "• Salvar média: \"minha média de Ciências no P1 é 9.2\""
    )

def _manual_esportes() -> str:
    return (
        "Manual — Esportes\n\n"
        "• Série B hoje: \"Quais são os jogos da Série B hoje?\"\n"
        "• Jogos óbvios: \"Quais os jogos óbvios de hoje?\"\n"
        "• Projeções Série B: \"Faça as projeções da Série B\"\n"
        "• Time específico: \"Flamengo joga hoje?\"\n"
        "• Liga específica: \"Quais os jogos da Premier League hoje?\""
    )

def _manual_geral() -> str:
    return (
        "Manual — GOOMI (resumo)\n\n"
        "• saúde — veja: \"Goomi, quais são as funções da saúde?\"\n"
        "• notas — veja: \"Goomi, quais são as funcionalidades das notas?\"\n"
        "• esportes — veja: \"Goomi, ajuda de esportes\""
    )

def manual_router(q: str) -> str | None:
    ql = (q or "").lower()
    if any(k in ql for k in ["quais são as funções da saúde","quais sao as funcoes da saude","ajuda de saude","manual de saude","como eu salvo as consultas","como salvar consulta","como registrar consulta"]):
        return _manual_saude()
    if any(k in ql for k in ["funcionalidades das notas","funções das notas","ajuda de notas","manual de notas","como salvar nota"]):
        return _manual_notas()
    if any(k in ql for k in ["ajuda de esportes","manual de esportes","funções dos esportes","quais são as funções dos esportes","jogos obvios ajuda","projeções ajuda","projecoes ajuda"]):
        return _manual_esportes()
    if any(k in ql for k in ["ajuda","manual","como funciona o goomi","o que você faz","quais suas funções","quais sao suas funcoes"]):
        return _manual_geral()
    return None



# ==== OVERRIDES: Saúde — parser e buscadores (ajuste de descrição/médico) ===
def _health_extract_details_pt(text: str):
    """Extrai (tipo_override, especialidade, descricao) a partir do texto.
    - Detecta médico: 'doutor/dr/doutora/dra' + nome -> descricao='Dr. Nome'
    - Detecta procedimentos comuns (ex.: botox, chip, medicamento, aplicação, cirurgia) -> tipo='procedimento', descricao capitalizada
    - Mantém especialidade se presente (Dermatologia etc.)"""
    q = (text or '').lower()

    # Procedimentos-chave
    proc_map = {
        'botox': 'Botox',
        'preenchimento': 'Preenchimento',
        'chip': 'Colocação de chip',
        'chip hormonal': 'Colocação de chip hormonal',
        'chip anticoncepcional': 'Colocação de chip anticoncepcional',
        'aplicação': 'Aplicação',
        'aplicacao': 'Aplicação',
        'medicamento': 'Medicamento',
        'medicação': 'Medicação',
        'medicacao': 'Medicação',
        'cirurgia': 'Cirurgia',
        'exodontia': 'Exodontia',
        'limpeza': 'Limpeza',
        'raio x': 'Raio X',
        'raio-x': 'Raio X'
    }
    tipo_override = None
    descricao = None

    # 1) Médico (Dr./Dra./Doutor(a) Nome)
    mdoc = re.search(r'\b(dr\.?|dra\.?|doutor(?:a)?)\s+([a-záàâãéêíóôõúç]+(?:\s+[a-záàâãéêíóôõúç]+)*)', q, flags=re.IGNORECASE)
    if mdoc:
        nome = mdoc.group(2).strip()
        nome_cap = ' '.join([w.capitalize() for w in nome.split()])
        prefix = mdoc.group(1)
        prefix_norm = 'Dr.' if prefix.lower().startswith('dr') else ('Dra.' if 'ra' in prefix.lower() else 'Dr.')
        descricao = f"{prefix_norm} {nome_cap}"

    # 2) Procedimentos
    for key, label in proc_map.items():
        if key in q:
            tipo_override = 'procedimento'
            # só define descricao se ainda não tem (médico) -> preserva médico se informado
            if not descricao:
                descricao = label
            break

    return tipo_override, descricao

def extrair_comando_saude(question: str, client_id: str) -> dict:  # override
    q = (question or '').lower()

    # tipo
    tipo = None
    if 'procedim' in q:
        tipo = 'procedimento'
    elif 'exame' in q:
        tipo = 'exame'
    elif 'consulta' in q:
        tipo = 'consulta'

    # especialidade (hints)
    espec = None
    for k, v in {
        'dermatolog': 'Dermatologia',
        'cardiolog': 'Cardiologia',
        'ginecolog': 'Ginecologia',
        'ortoped': 'Ortopedia',
        'oftalmolog': 'Oftalmologia',
        'endocrin': 'Endocrinologia',
        'psiquiat': 'Psiquiatria',
        'psicol': 'Psicologia',
        'odont': 'Odontologia',
        'clinico': 'Clínico Geral',
        'clínico': 'Clínico Geral',
    }.items():
        if k in q:
            espec = v
            break

    # data (realização ou retorno)
    d = _parse_pt_date_in_text(q)
    data_realizacao = d.strftime('%Y-%m-%d') if d else None
    data_retorno = None
    if 'retorno' in q and d:
        data_retorno = data_realizacao
        data_realizacao = None

    # id explícito (#123)
    por_id = None
    m_id = re.search(r'#(\d+)', q)
    if m_id:
        try: por_id = int(m_id.group(1))
        except: por_id = None

    ultimo_flag = any(k in q for k in ['último', 'ultimo', 'mais recente', 'recente'])

    # Detalhes (médico/procedimento)
    tipo_over, descricao = _health_extract_details_pt(q)
    if tipo_over and not tipo:
        tipo = tipo_over
    # se não setou tipo, default consulta
    if not tipo:
        tipo = 'consulta'

    # ===== PRIORIDADES =====
    if any(k in q for k in ['salvar','salva','registrar','registra','adicionar','anotar','agendar','marcar']):
        return {'intent':'save','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':data_realizacao,'data_retorno':data_retorno,'data_ref':data_realizacao or data_retorno,
                'por_id':None,'ultimo':False,'set_fields':{}}

    if any(k in q for k in ['apagar','deletar','remover','excluir']):
        return {'intent':'delete','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':None,'data_retorno':None,'data_ref':data_realizacao or data_retorno,
                'por_id':por_id,'ultimo':ultimo_flag,'set_fields':{}}

    if any(k in q for k in ['atualizar','editar','corrigir','mudar']):
        return {'intent':'update','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':data_realizacao,'data_retorno':data_retorno,'data_ref':data_realizacao or data_retorno,
                'por_id':por_id,'ultimo':ultimo_flag,'set_fields':{}}

    if any(k in q for k in ['próximo retorno','proximo retorno']):
        return {'intent':'next_return','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':None,'data_retorno':None,'data_ref':None,'por_id':None,'ultimo':False,'set_fields':{}}

    if any(k in q for k in ['próxima consulta','proxima consulta','próximo exame','proximo exame','próximo procedimento','proximo procedimento','minha próxima consulta','minha proxima consulta']):
        return {'intent':'next','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':None,'data_retorno':None,'data_ref':None,'por_id':None,'ultimo':False,'set_fields':{}}

    if any(k in q for k in ['última consulta','ultima consulta','último exame','ultimo exame','último procedimento','ultimo procedimento']):
        return {'intent':'last','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':None,'data_retorno':None,'data_ref':None,'por_id':None,'ultimo':True,'set_fields':{}}

    if any(k in q for k in ['listar','liste','histórico','historico','últimas','ultimas','últimos','ultimos']) or any(t in q for t in ['consulta','exame','procedimento']):
        return {'intent':'list','tipo':tipo,'especialidade':espec,'descricao':descricao,
                'data_realizacao':None,'data_retorno':None,'data_ref':None,'por_id':None,'ultimo':ultimo_flag,'set_fields':{}}

    return {'intent':'none','tipo':tipo,'especialidade':espec,'descricao':descricao,
            'data_realizacao':data_realizacao,'data_retorno':data_retorno,'data_ref':data_realizacao or data_retorno,
            'por_id':por_id,'ultimo':ultimo_flag,'set_fields':{}}

# Reescreve proximo_evento_saude para aceitar filtros por especialidade/descrição
def proximo_evento_saude(conn, client_id: str, tipo: Optional[str], so_retorno: bool=False, especialidade: Optional[str]=None, descricao: Optional[str]=None) -> Optional[sqlite3.Row]:  # override
    aluno_id = get_user_id_by_client_id(conn, client_id)
    if not aluno_id:
        return None
    today = _today().strftime('%Y-%m-%d')
    cur = conn.cursor()
    filtro_tipo = ' AND tipo = ?' if tipo else ''
    params = [aluno_id, today]
    if tipo: params.append(tipo)
    filtro_extra = ''
    if especialidade:
        filtro_extra += ' AND LOWER(COALESCE(especialidade, "")) = LOWER(?)'
        params.append(especialidade)
    if descricao:
        filtro_extra += ' AND LOWER(COALESCE(descricao, "")) LIKE LOWER(?)'
        params.append(f'%{descricao}%')

    if so_retorno:
        cur.execute(f"""
            SELECT * FROM consultas_exames
             WHERE aluno_id = ?
               AND data_retorno IS NOT NULL
               AND data_retorno >= ?
               {filtro_tipo}
               {filtro_extra}
             ORDER BY data_retorno ASC, id ASC
             LIMIT 1
        """, params)
        return cur.fetchone()

    # primeiro por retorno futuro, senão por realização futura
    cur.execute(f"""
        SELECT * FROM consultas_exames
         WHERE aluno_id = ?
           AND data_retorno IS NOT NULL
           AND data_retorno >= ?
           {filtro_tipo}
           {filtro_extra}
         ORDER BY data_retorno ASC, id ASC
         LIMIT 1
    """, params)
    row = cur.fetchone()
    if row: return row

    cur.execute(f"""
        SELECT * FROM consultas_exames
         WHERE aluno_id = ?
           AND data_realizacao IS NOT NULL
           AND data_realizacao >= ?
           {filtro_tipo}
           {filtro_extra}
         ORDER BY data_realizacao ASC, id ASC
         LIMIT 1
    """, params)
    return cur.fetchone()

@app.post("/ask")
def ask():
    try:
        data = request.get_json(force=True) or {}
        if DEBUG_LOG: print(f"[ASK] {data}")

        client_id = data.get("client_id")
        question  = data.get("question")
        if not client_id or not question:
            return jsonify({"error": "Envie 'client_id' e 'question'."}), 400

        conn = get_db()
        ensure_health_schema(conn)
        ensure_projections_schema(conn)
        ensure_grades_schema(conn)  # <<< garantir notas/medias antes de usar

        user = get_user_by_client_id(conn, client_id)
        if not user: return jsonify({"error": f"client_id não encontrado: {client_id}"}), 404

        if client_id not in user_memories:
            user_memories[client_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory = user_memories[client_id]

        ql = question.lower().strip()
        # ==== Manual / Ajuda (antes de qualquer outra lógica) ====
        manual_txt = manual_router(question)
        if manual_txt:
            memory = user_memories.setdefault(client_id, ConversationBufferMemory(memory_key="chat_history", return_messages=True))
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(manual_txt)
            return jsonify({"answer": manual_txt})

        # Cumprimentos simples
        cumprimentos = ["oi","olá","ola","bom dia","boa tarde","boa noite","hello","hi"]
        if any(ql.startswith(c) for c in cumprimentos):
            answer = f"Oi {user['nome']}! Tudo bem? Em que posso te ajudar hoje?"
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return jsonify({"answer": answer})

        # Perfil (apenas se perguntar)
        if is_profile_question(ql):
            perfil = USER_PROFILES.get((client_id or "").lower())
            answer = perfil['descricao'] if perfil else f"{user['nome']}, como posso te ajudar hoje?"
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return jsonify({"answer": answer})

        # ===================== SAÚDE (Helena) — PRIORIDADE SOBRE NOTAS =====================
        health = extrair_comando_saude(question, client_id)
        if health.get("intent") != "none":
            if not ensure_only_helena(client_id):
                answer = "Estas informações de saúde são privadas e só a Helena pode acessar."
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            aluno_id = get_user_id_by_client_id(conn, client_id)

            if health["intent"] == "save":
                tipo = health.get("tipo") or "consulta"
                data_real = health.get("data_realizacao")
                data_ret  = health.get("data_retorno")
                espec     = health.get("especialidade")
                miss = []
                if tipo not in HEALTH_TYPES: miss.append("tipo (consulta/exame/procedimento)")
                if not (data_real or data_ret): miss.append("data (ex.: 15/09/2025)")
                if miss:
                    answer = "Para registrar, preciso de: " + ", ".join(miss) + "."
                else:
                    ok = inserir_evento_saude(conn, client_id, tipo, espec, health.get("descricao"), data_real, data_ret)
                    answer = "Registrado com sucesso." if ok else "Não consegui registrar."
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            if health["intent"] == "next_return":
                row = proximo_evento_saude(conn, client_id, health.get("tipo"), so_retorno=True, especialidade=health.get("especialidade"), descricao=health.get("descricao"))
                answer = "Não encontrei próximos retornos." if not row else "Próximo retorno:\n" + format_evento(row)
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            if health["intent"] == "next":
                row = proximo_evento_saude(conn, client_id, health.get("tipo"), so_retorno=False, especialidade=health.get("especialidade"), descricao=health.get("descricao"))
                answer = "Não encontrei próximos compromissos de saúde." if not row else "Próximo compromisso:\n" + format_evento(row)
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            if health["intent"] == "last":
                row = ultimo_evento_saude(conn, client_id, health.get("tipo"))
                answer = "Não encontrei registros anteriores." if not row else "Último compromisso:\n" + format_evento(row)
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            if health["intent"] == "list":
                rows = listar_eventos_saude(conn, client_id, health.get("tipo"), limite=50, especialidade=health.get("especialidade"), descricao=health.get("descricao"), date_from=interpretar_periodo_saude(question)['date_from'], date_to=interpretar_periodo_saude(question)['date_to'])
                answer = "Não encontrei registros de saúde." if not rows else "Últimos registros:\n" + "\n".join(format_evento(r) for r in rows)
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            if health["intent"] == "update":
                candidatos = buscar_eventos_para_editar(conn, client_id, health.get("tipo"), health.get("especialidade"), health.get("data_ref"), health.get("ultimo"), health.get("por_id"))
                if not candidatos:
                    answer = "Não encontrei registro compatível para atualizar. Peça 'listar' para ver IDs."
                elif len(candidatos) > 1:
                    answer = "Encontrei mais de um. Especifique com ID (#id) ou data/especialidade."
                else:
                    row = candidatos[0]
                    ok = atualizar_evento_saude(conn, row["id"], aluno_id, health.get("set_fields", {}))
                    answer = "Atualizado." if ok else "Nada para atualizar."
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})

            if health["intent"] == "delete":
                candidatos = buscar_eventos_para_editar(conn, client_id, health.get("tipo"), health.get("especialidade"), health.get("data_ref"), health.get("ultimo"), health.get("por_id"))
                if not candidatos:
                    answer = "Não encontrei registro para apagar. Peça 'listar' para ver IDs."
                elif len(candidatos) > 1:
                    answer = "Encontrei mais de um. Especifique com ID (#id) ou data/especialidade."
                else:
                    row = candidatos[0]
                    ok = deletar_evento_saude(conn, row["id"], aluno_id)
                    answer = f"Registro #{row['id']} removido." if ok else "Não consegui apagar o registro."
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
                return jsonify({"answer": answer})
        # ===================== FIM SAÚDE (Helena) =====================

        # Salvar NOTA / MÉDIA
        save_grade = extrair_comando_salvar(question, client_id_padrao=client_id)
        if save_grade.get("is_action"):
            missing = []
            if (save_grade.get("nota") is not None) or (save_grade.get("media") is not None):
                if not save_grade.get("disciplina"): missing.append("disciplina")
                if not save_grade.get("periodo"):    missing.append("período (P1..P4 ou T1..T3)")
            else:
                missing.append("nota ou média")

            msgs, did_any = [], False
            if save_grade.get("nota") is not None and not missing:
                ok = inserir_nota_conversa(conn, save_grade["client_id"], save_grade["disciplina"], save_grade["periodo"], save_grade.get("descricao"), float(save_grade["nota"]))
                msgs.append("Anotado!" if ok else "Não consegui salvar a nota (aluno não encontrado).")
                did_any = did_any or ok
            if save_grade.get("media") is not None and not missing:
                okm = upsert_media(conn, save_grade["client_id"], save_grade["disciplina"], save_grade["periodo"], float(save_grade["media"]))
                msgs.append("Média atualizada." if okm else "Não consegui salvar a média (aluno não encontrado).")
                did_any = did_any or okm
            answer = " ".join(msgs) if did_any else ("Pra salvar, preciso de: " + ", ".join(missing) + ".")
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return jsonify({"answer": answer})

        # Consulta NOTAS/MÉDIAS
        filtros = extrair_filtros_notas(question)
        if filtros["is_notas"]:
            if DEBUG_LOG: print("[NOTAS] consulta banco")
            rows, medias = consultar_notas(conn, user["id"], filtros["disciplina"], filtros["periodo"])
            answer = formatar_resposta_notas(user["nome"], rows, medias, filtros["disciplina"], filtros["periodo"])
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return jsonify({"answer": answer})

        # ===================== PRIORIDADE: PROJEÇÕES DA SÉRIE B =====================
        proj_triggers = [
            "projeção", "projecao", "projeções", "projecoes",
            "prognóstico", "prognostico",
            "modelo", "estimativa",
            "faça as projeções", "faça a projeção", "faca as projecoes", "faca a projecao"
        ]
        if any(k in ql for k in proj_triggers):
            target_date = _parse_pt_date_in_text(ql)
            if not target_date:
                m_dia = re.search(r"\bdia\s+(\d{1,2})\b", ql)
                if m_dia:
                    try:
                        dnum = int(m_dia.group(1))
                        today = _today()
                        last_day = 28
                        for dd in [31, 30, 29, 28]:
                            try:
                                _ = date(today.year, today.month, dd)
                                last_day = dd
                                break
                            except:
                                continue
                        dnum = max(1, min(dnum, last_day))
                        target_date = date(today.year, today.month, dnum)
                    except Exception:
                        target_date = None

            if not target_date:
                target_date = _today()

            payload = serie_b_projections_calc(
                date_str=target_date.strftime("%Y-%m-%d"),
                last=5,
                h2h_last=H2H_LAST
            )

            if payload.get("games"):
                for g in payload["games"]:
                    m = g["match"]; p = g["projection"]
                    save_projection_row(
                        conn,
                        dref=payload["date"],
                        league_id=payload["league_id"],
                        home_id=m.get("home_id") or 0,
                        away_id=m.get("away_id") or 0,
                        home=m.get("home") or "",
                        away=m.get("away") or "",
                        proj=p
                    )

            if not payload.get("games"):
                ans = (
                    f"Não encontrei jogos da Série B em {payload.get('date')}. "
                    "Quer que eu verifique outra liga também?"
                )
            else:
                ans = formatar_resposta_projecoes(payload)

            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(ans)
            return jsonify({"answer": ans})
        # ============================================================================

        # ===================== JOGOS ÓBVIOS (novo) ===============================
        obv_triggers = [
            "jogos óbvios", "jogos obvios", "jogo óbvio", "jogo obvio",
            "análise óbvia", "analise obvia", "favoritos do dia", "jogos favoritos"
        ]
        if any(k in ql for k in obv_triggers):
            if not SPORTS_ENABLED or not SPORTS_KEY:
                ans = "Esportes desabilitado ou sem API key configurada."
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(ans)
                return jsonify({"answer": ans})

            # data alvo: hoje / amanhã / ontem / dd/mm/aaaa
            target_date = _parse_pt_date_in_text(ql) or _today()
            if "ontem" in ql:    target_date = _yesterday()
            if "amanh" in ql:    target_date = _tomorrow()  # amanha/amanhã

            # liga específica? (ex.: premier league, pl, epl, copa do brasil...)
            liga_txt = None
            m = re.search(r"(jogos\s+óbvios\s+(?:da|do)|jogos\s+obvios\s+(?:da|do)|\bda\b|\bdo\b)\s+(.+)", ql)
            if m:
                liga_txt = _sanitize_league_phrase(m.group(2))
            else:
                for hint in ["premier league","premiere league","pl","epl",
                            "la liga","laliga","bundesliga","ligue 1",
                            "libertadores","champions","ucl",
                            "série a","serie a","série b","serie b","copa do brasil"]:
                    if hint in ql:
                        liga_txt = hint
                        break

            leagues = None
            liga_label = None   # só para mostrar no cabeçalho

            if liga_txt:
                key = _norm(liga_txt)
                lid = fb_search_league_id(key)   # usa nosso mapa global + fallback de API
                if lid:
                    leagues = [lid]
                    liga_label = key

            payload = obvious_games_calc(
                date_str=target_date.strftime("%Y-%m-%d"),
                leagues=leagues,         # None => A+B
                last=5,
                h2h_last=3,
                min_strength=0.75
            )

            if payload.get("count", 0) == 0:
                rot = "hoje" if target_date == _today() else target_date.strftime("%d/%m/%Y")
                liga_str = f" ({liga_label.title()})" if liga_label else ""
                ans = f"Não encontrei jogos óbvios{liga_str} {rot} com força ≥ 75%."
            else:
                rot = "hoje" if target_date == _today() else target_date.strftime("%d/%m/%Y")
                header = f"Jogos óbvios — {rot}"
                if liga_label:
                    header += f" — {liga_label.title()}"
                ans = f"{header}\n\n{payload['table_markdown']}\n\nObs.: força = heurística form/defesa + H2H. Fonte: API-Football"

            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(ans)
            return jsonify({"answer": ans})
        # =================== FIM JOGOS ÓBVIOS (novo) =============================

        # ===================== FUTEBOL: LISTAS GERAIS POR DIA (NOVO) ================
        # 1) "Quem joga hoje/amanhã/ontem?"  -> reúne Série A + Série B
        if any(p in ql for p in ["quem joga hoje","quem joga amanha","quem joga amanhã","jogos de hoje","jogos hoje","jogos amanhã","jogos amanha","jogos de amanha","jogos de amanhã","quem joga ontem","jogos de ontem","jogos ontem"]):
            if "ontem" in ql:
                target_date = _yesterday()
                titulo = "ontem"
            elif "amanh" in ql:  # amanha/amanhã
                target_date = _tomorrow()
                titulo = "amanhã"
            else:
                target_date = _today()
                titulo = "hoje"

            a = fb_fixtures_by_league_date(LEAGUE_IDS["br_serie_a"], target_date)
            b = fb_fixtures_by_league_date(LEAGUE_IDS["br_serie_b"], target_date)
            fixtures = (a or []) + (b or [])
            fixtures = sorted(fixtures, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ))
            rows = fixtures_to_table(fixtures)
            md = table_markdown(rows)
            ans = f"Jogos do Brasileirão ({titulo})\n\n{md}\n\nFonte: API-Football" if fixtures else f"Não encontrei jogos do Brasileirão {titulo}."
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(ans)
            return jsonify({"answer": ans})

        # 2) Brasileirão — hoje/amanhã/ontem (Série A, Série B, ou ambos quando só 'brasileirão')
        if any(k in ql for k in ["brasileirão","brasileirao","série a","serie a","série b","serie b"]):
            if "ontem" in ql:
                target_date = _yesterday(); titulo = "ontem"
            elif "amanh" in ql:
                target_date = _tomorrow();  titulo = "amanhã"
            else:
                target_date = _today();     titulo = "hoje"

            want_a = any(k in ql for k in ["série a","serie a"])
            want_b = any(k in ql for k in ["série b","serie b"])
            if not (want_a or want_b):
                want_a = want_b = True  # 'brasileirão' genérico -> A + B

            fixtures = []
            if want_a:
                fixtures += fb_fixtures_by_league_date(LEAGUE_IDS["br_serie_a"], target_date) or []
            if want_b:
                fixtures += fb_fixtures_by_league_date(LEAGUE_IDS["br_serie_b"], target_date) or []
            fixtures = sorted(fixtures, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ))
            rows = fixtures_to_table(fixtures)
            md = table_markdown(rows)
            serie_txt = "Série A" if (want_a and not want_b) else ("Série B" if (want_b and not want_a) else "Série A + Série B")
            ans = f"Brasileirão — {serie_txt} — {titulo}\n\n{md}\n\nFonte: API-Football" if fixtures else f"Não encontrei jogos do Brasileirão ({serie_txt}) {titulo}."
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(ans)
            return jsonify({"answer": ans})

        # 3) "Flamengo joga hoje/amanhã/ontem?" (ou outro time)
        if (any(k in ql for k in ["joga hoje","joga amanha","joga amanhã","jogos hoje do","jogos de hoje do"])
                and ("proximo" not in ql and "próximo" not in ql and "ultimo" not in ql and "último" not in ql)
            ):
            team = None
            for tname in ["flamengo","flemango","vasco","corinthians","botafogo","palmeiras","santos","fluminense","gremio","grêmio","cruzeiro","atletico","atlético","athletico","internacional","bahia","coritiba","ceara","fortaleza","mirassol","criciuma","criciúma"]:
                if tname in ql:
                    team = TEAM_ALIASES.get(tname, tname); break
            if team:
                if "ontem" in ql:
                    dref = _yesterday()
                elif "amanh" in ql:
                    dref = _tomorrow()
                else:
                    dref = _today()
                tid = fb_team_id(team)
                if tid:
                    fixtures = fb_fixtures_by_team_date(tid, dref)
                    fixtures = sorted(fixtures, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ))
                    rows = fixtures_to_table(fixtures)
                    md = table_markdown(rows)
                    if fixtures:
                        ans = f"Jogos do {team.title()} em {dref.strftime('%d/%m/%Y')}\n\n{md}\n\nFonte: API-Football"
                    else:
                        ans = f"O {team.title()} não tem jogo em {dref.strftime('%d/%m/%Y')}."
                    memory.chat_memory.add_user_message(question)
                    memory.chat_memory.add_ai_message(ans)
                    return jsonify({"answer": ans})

        # 4) Seleções — “quais são os jogos das seleções hoje?”
        if any(k in ql for k in ["seleção","selecao","seleções","selecoes"]):
            if "ontem" in ql:
                dref = _yesterday()
            elif "amanh" in ql:
                dref = _tomorrow()
            else:
                dref = _today()
            possiveis = []
            for name in ["brasil","argentina","portugal","espanha","inglaterra","franca","frança","alemanha","italia","itália","uruguai","colombia","colômbia","holanda","pais(es) baixos","países baixos","mexico","méxico","chile","peru","paraguai"]:
                if name in ql:
                    possiveis.append(name)
            if not possiveis:
                possiveis = ["brasil"]

            team_ids = fb_team_ids_from_names(possiveis)
            all_fx = []
            for tid in team_ids:
                all_fx.extend(fb_fixtures_by_team_date(tid, dref))

            seen, unique_fx = set(), []
            for fx in all_fx:
                fid = (fx.get("fixture", {}) or {}).get("id")
                if fid and fid in seen: continue
                if fid: seen.add(fid)
                unique_fx.append(fx)

            unique_fx = sorted(unique_fx, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ))
            rows = fixtures_to_table(unique_fx)
            md = table_markdown(rows)
            titulo = f"Seleções — {dref.strftime('%d/%m/%Y')}"
            ans = f"{titulo}\n\n{md}\n\nFonte: API-Football" if unique_fx else f"Não encontrei jogos das seleções em {dref.strftime('%d/%m/%Y')}."
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(ans)
            return jsonify({"answer": ans})

        # 5) Outras ligas por nome — “quais são os jogos da Premier League hoje?”
        if any(k in ql for k in ["jogos da","jogos do","quais são os jogos da","quais sao os jogos da","quais os jogos da","liga","champions","libertadores","premier league","la liga","bundesliga","ligue 1","serie a italiana","coppa","copa do brasil","sul-americana","sulamericana"]):
            liga_txt = None
            m = re.search(r"(jogos\s+(?:da|do)\s+)(.+)", ql)
            if m:
                liga_txt = _sanitize_league_phrase(m.group(2))
            else:
                for hint in ["premier league","la liga","bundesliga","ligue 1","champions","libertadores","copa do brasil","sul-americana","sulamericana","serie a italiana","serie a tim"]:
                    if hint in ql:
                        liga_txt = hint
                        break

            if liga_txt:
                if "ontem" in ql:
                    target_date = _yesterday()
                elif "amanh" in ql:
                    target_date = _tomorrow()
                else:
                    target_date = _today()

                league_id = fb_search_league_id(liga_txt)
                if league_id:
                    fixtures = fb_fixtures_by_league_date(league_id, target_date)
                    fixtures = sorted(fixtures, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ))
                    rows = fixtures_to_table(fixtures)
                    md = table_markdown(rows)
                    ans = (f"Jogos — {liga_txt.title()} — {target_date.strftime('%d/%m/%Y')}\n\n{md}\n\nFonte: API-Football"
                           if fixtures else f"Não encontrei jogos de {liga_txt.title()} em {target_date.strftime('%d/%m/%Y')}.")
                    memory.chat_memory.add_user_message(question)
                    memory.chat_memory.add_ai_message(ans)
                    return jsonify({"answer": ans})
        # =================== FIM LISTAS GERAIS POR DIA (NOVO) =======================

        # Esportes — consultas amplas (fallback geral)
        if SPORTS_ENABLED and SPORTS_KEY:

            def _when_to_date(qtext: str) -> date:
                dnat = _parse_pt_date_in_text(qtext)
                if dnat:
                    return dnat
                if any(w in qtext for w in ["amanha", "amanhã", "tomorrow"]):
                    return _tomorrow()
                if "ontem" in qtext:
                    return _yesterday()
                return _today()

            def _fmt_hhmm_local(fx: dict) -> str:
                ldt = _dt_local_from_fixture(fx)
                return ldt.strftime("%H:%M") if ldt else "--:--"

            def _fixtures_to_table(fixtures: List[Dict[str, Any]]) -> str:
                if not fixtures:
                    return "Nenhum jogo encontrado."
                linhas = ["Data\tCampeonato\tCasa\tVisitante\tHora"]
                for fx in sorted(fixtures, key=lambda x: _dt_local_from_fixture(x) or datetime.max.replace(tzinfo=PROJECT_TZ)):
                    ldt = _dt_local_from_fixture(fx)
                    data_txt = ldt.strftime("%d/%m/%Y") if ldt else "--/--/----"
                    liga = (fx.get("league", {}) or {}).get("name", "")
                    home = (fx.get("teams", {}) or {}).get("home", {}).get("name", "—")
                    away = (fx.get("teams", {}) or {}).get("away", {}).get("name", "—")
                    hora = _fmt_hhmm_local(fx)
                    linhas.append(f"{data_txt}\t{liga}\t{home}\t{away}\t{hora}")
                return "\n".join(linhas)

            def _find_league_id_by_name(name: str) -> Optional[int]:
                try:
                    r = requests.get(
                        f"{FOOTBALL_BASE}/leagues",
                        params={"search": name},
                        headers=_api_headers(),
                        timeout=15
                    )
                    r.raise_for_status()
                    for item in r.json().get("response", []):
                        lg = item.get("league", {}) or {}
                        if lg.get("id") and not lg.get("type", "").lower().startswith("cup youth"):
                            return int(lg["id"])
                except Exception as e:
                    if DEBUG_LOG: print("[FOOTBALL][league_search] erro:", e)
                return None

            def _list_by_league_name_or_id(qtext: str, default_ids: List[int]) -> Optional[str]:
                target_date = _when_to_date(qtext)
                liga_explicita = None
                for key in ["série a", "serie a", "brasileirão série a", "brasileirao serie a",
                            "série b", "serie b", "brasileirão série b", "brasileirao serie b"]:
                    if key in qtext:
                        liga_explicita = key
                        break

                fixtures_all: List[Dict[str, Any]] = []

                if liga_explicita:
                    if "a" in liga_explicita and "série" in liga_explicita or "serie a" in liga_explicita:
                        lid = LEAGUE_IDS.get("br_serie_a")
                    elif "b" in liga_explicita:
                        lid = LEAGUE_IDS.get("br_serie_b")
                    else:
                        lid = None
                    if lid:
                        fixtures = fb_fixtures_by_league_date(lid, target_date)
                        if not fixtures:
                            return (f"Não há jogos nesta liga em {target_date.strftime('%d/%m/%Y')}.")
                        tabela = _fixtures_to_table(fixtures)
                        head = f"Jogos — {target_date.strftime('%d/%m/%Y')}"
                        return f"{head}\n\n{tabela}\nFonte: API-Football"

                for lid in default_ids:
                    fixtures_all += fb_fixtures_by_league_date(lid, target_date) or []

                if fixtures_all:
                    tabela = _fixtures_to_table(fixtures_all)
                    head = f"Brasileirão — {('hoje' if target_date==_today() else target_date.strftime('%d/%m/%Y'))}"
                    return f"{head}\n\n{tabela}\nFonte: API-Football"

                liga_nominal = None
                m_nom = re.search(r"(premier league|la liga|bundesliga|ligue 1|serie a italiana|libertadores|champions|eredivisie|mls|liga mx)", qtext)
                if m_nom:
                    liga_nominal = m_nom.group(1)
                if liga_nominal:
                    lid = _find_league_id_by_name(liga_nominal)
                    if lid:
                        fixtures = fb_fixtures_by_league_date(lid, target_date)
                        if fixtures:
                            tabela = _fixtures_to_table(fixtures)
                            head = f"{liga_nominal.title()} — {target_date.strftime('%d/%m/%Y')}"
                            return f"{head}\n\n{tabela}\nFonte: API-Football"

                return None

            def _list_national_teams(qtext: str) -> Optional[str]:
                target_date = _when_to_date(qtext)
                países = {
                    "brasil": "Brazil",
                    "argentina": "Argentina",
                    "portugal": "Portugal",
                    "espanha": "Spain",
                    "frança": "France",
                    "alemanha": "Germany",
                    "inglaterra": "England",
                    "italia": "Italy",
                    "uruguai": "Uruguay",
                    "holanda": "Netherlands",
                }
                selecionados = []
                for k in países.keys():
                    if k in qtext:
                        selecionados = [k]; break
                if not selecionados:
                    selecionados = list(países.keys())

                jogos: List[Dict[str, Any]] = []
                for k in selecionados:
                    name = países[k]
                    tid = fb_team_id(name)
                    if not tid:
                        continue
                    arr = fb_fixture_generic({"team": tid, "date": _date_to_str(target_date),
                                             "season": _season_from_date(target_date),
                                             "timezone": "America/Sao_Paulo"})
                    jogos += arr or []

                if not jogos:
                    return f"Não encontrei jogos de seleções em {target_date.strftime('%d/%m/%Y')}."

                tabela = _fixtures_to_table(jogos)
                rotulo = "Seleções (Brasil e principais)" if len(selecionados) > 1 else f"Seleção ({selecionados[0].title()})"
                return f"{rotulo} — {target_date.strftime('%d/%m/%Y')}\n\n{tabela}\nFonte: API-Football"

            def _answer_next_or_today_for_team(qtext: str, team_hint: Optional[str]) -> Optional[str]:
                known = ["flamengo","flemango","vasco","corinthians","botafogo","palmeiras","santos",
                         "fluminense","gremio","grêmio","cruzeiro","atletico","atlético","athletico",
                         "internacional","bahia","coritiba","ceara","fortaleza","mirassol","criciuma","criciúma"]
                team = None
                for tname in known:
                    if tname in qtext:
                        team = TEAM_ALIASES.get(tname, tname); break
                if not team:
                    team = team_hint or LAST_TEAM_BY_USER.get(client_id)

                if not team:
                    return None

                LAST_TEAM_BY_USER[client_id] = team
                tid = fb_team_id(team)
                if not tid:
                    return f"Não encontrei o time '{team}'."

                mN = re.search(r"últimos?\s*(\d+)", qtext) or re.search(r"ultimos?\s*(\d+)", qtext)
                if mN:
                    n = int(mN.group(1))
                    fixtures = fb_fixtures_last_n(tid, n)
                    if not fixtures:
                        return "Não encontrei jogos recentes."
                    linhas = [f"Aqui estão os últimos {n} jogos do {team.title()}:", ""]
                    for i, fx in enumerate(fixtures):
                        linhas.append(f"{i+1}. {fb_format(fx)}")
                    linhas += ["", "Fonte: API-Football"]
                    return "\n".join(linhas)

                target_date = _parse_pt_date_in_text(qtext)
                if target_date:
                    fx = fb_fixture_on_date(tid, target_date)
                    if fx:
                        return f"{fb_format(fx)}\n\nFonte: API-Football"
                    next_fx = fb_fixture_next(tid)
                    if next_fx:
                        return f"O {team.title()} não joga em {target_date.strftime('%d/%m/%Y')}. Próximo: {fb_format(next_fx)}\n\nFonte: API-Football"
                    return f"O {team.title()} não tem jogo nessa data e não encontrei próximos."

                if "próximo" in qtext or "proximo" in qtext:
                    fx = fb_fixture_next(tid)
                    if fx:
                        return f"{fb_format(fx)}\n\nFonte: API-Football"
                    return f"Não encontrei próximo jogo do {team.title()}."

                if "último" in qtext or "ultimo" in qtext:
                    fx = fb_fixture_last(tid)
                    if fx:
                        return f"{fb_format(fx)}\n\nFonte: API-Football"
                    return f"Não encontrei jogo recente do {team.title()}."

                target_date = _when_to_date(qtext)
                fx_today = fb_fixture_on_date(tid, target_date)
                if fx_today:
                    return f"{fb_format(fx_today)}\n\nFonte: API-Football"
                next_fx = fb_fixture_next(tid)
                if next_fx:
                    rot = "hoje" if target_date==_today() else target_date.strftime("%d/%m/%Y")
                    return f"O {team.title()} não joga {rot}. Próximo: {fb_format(next_fx)}\n\nFonte: API-Football"
                return f"O {team.title()} não tem jogo na data consultada e não encontrei próximos."

            # Roteamento de esportes (fallback)
            if any(k in ql for k in ["seleções", "selecoes", "seleção", "selecao"]):
                ans = _list_national_teams(ql)
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(ans)
                return jsonify({"answer": ans})

            if re.search(r"\bquem joga\b", ql) or "jogos do brasileirão" in ql or "jogos do brasileirao" in ql:
                ans = _list_by_league_name_or_id(ql, default_ids=[LEAGUE_IDS["br_serie_a"], LEAGUE_IDS["br_serie_b"]])
                if not ans:
                    dref = _when_to_date(ql)
                    ans = f"Não há jogos do Brasileirão em {dref.strftime('%d/%m/%Y')}. Quer que eu verifique outras ligas (ex.: Premier League, La Liga, Libertadores)?"
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(ans)
                return jsonify({"answer": ans})

            if any(k in ql for k in ["série a","serie a","série b","serie b","brasileirão","brasileirao",
                                     "premier league","la liga","bundesliga","ligue 1","libertadores","champions"]):
                ans = _list_by_league_name_or_id(ql, default_ids=[LEAGUE_IDS["br_serie_a"], LEAGUE_IDS["br_serie_b"]])
                if not ans:
                    dref = _when_to_date(ql)
                    ans = f"Não encontrei jogos dessa liga em {dref.strftime('%d/%m/%Y')}."
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(ans)
                return jsonify({"answer": ans})

            if any(word in ql for word in ["jogo","joga","placar","partida","resultado","próximo","proximo","ultimo","último","últimos","ultimos","próxima","proxima"]):
                ans = _answer_next_or_today_for_team(ql, LAST_TEAM_BY_USER.get(client_id))
                if ans:
                    memory.chat_memory.add_user_message(question)
                    memory.chat_memory.add_ai_message(ans)
                    return jsonify({"answer": ans})

        # Chat normal
        history = memory.load_memory_variables({}).get("chat_history", [])
        messages = [SYSTEM_PROMPT] + history + [HumanMessage(content=question)]
        resp = chat.invoke(messages)
        answer = resp.content
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        return jsonify({"answer": answer})

    except Exception as e:
        if DEBUG_LOG: print("[ERROR] /ask ->", repr(e))
        return jsonify({"error": f"Falha interna: {e}"}), 500
    finally:
        try: conn.close()
        except: pass


# ==== 15) Inicialização ======================================================
if __name__ == "__main__":
    if DEBUG_LOG:
        print("[BOOT] Rotas registradas:")
        for rule in app.url_map.iter_rules():
            print(" -", rule, "| methods:", ",".join(sorted(rule.methods)))
        print()
    app.run(host='0.0.0.0', port=5000, debug=True)