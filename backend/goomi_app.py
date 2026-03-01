# -*- coding: utf-8 -*-
"""
GOOMI — Agente IA (backend) — Foco: Escola & Estudos
---------------------------------------------------
Mantém:
- Manuais/ajuda
- Cumprimentos
- Perfis da família (só quando perguntarem)
- Horóscopo do dia (cache + múltiplas fontes)
- Busca na web (genérica, resumo + fontes)
- Notas escolares (CRUD via conversa)
- Chat “normal”

Removeu:
- Saúde (CRUD)
- Projeções / futebol / NBA
- Qualquer heurística de esportes na busca web

Novo (Escola):
- Grade semanal (matérias por dia) por criança
- Agenda escolar (provas/seminários/etc.)
- Notas por bimestre + avaliação (P1/P2/TRAB...) e médias

Permissões:
- Pais (glauco, helena): podem cadastrar/editar/excluir
- Crianças (giulia, giovanna, guilherme): somente consulta (read-only)
"""

from __future__ import annotations


# Conversational context (in-memory). Good enough for single-user local runs.
# If you run multiple workers/processes, replace with SQLite state table.
LAST_CTX = {}

import os
import re
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from flask import Flask, request, jsonify

from zoneinfo import ZoneInfo

# IA (LangChain)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage


APP_VERSION = "escola-v10"

# =============================================================================
# Config
# =============================================================================

APP_TZ = ZoneInfo("America/Sao_Paulo")

DEFAULT_CONFIG = {
    "db_path": "goomi.db",
    "openai_model": "gpt-4o-mini",
    "temperature": 0.3,
    "web_search": {"enabled": True, "max_results": 5, "timeout_sec": 12},
    "horoscope": {"enabled": True},
    "debug": True,
}

CONFIG_PATH_ENV = os.getenv("GOOMI_CONFIG_PATH", "config.yaml")


def load_config(path: str) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # shallow merge
        for k, v in raw.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


CONFIG = load_config(CONFIG_PATH_ENV)


def now_sp() -> datetime:
    return datetime.now(tz=APP_TZ)


# =============================================================================
# Perfis & Permissões
# =============================================================================

PARENTS = {"glauco", "helena", "rayane"}
CHILDREN = {"giulia", "giovanna", "guilherme"}

USER_PROFILES = {
    "glauco": {
        "nome": "Glauco",
        "role": "Pai",
        "data_nascimento": "1987-12-30",
        "signo": "Capricórnio",
        "descricao": "O Glauco é o pai da família (e do Goomih 😄).Capricorniano, caseiro; fã de cinema, séries, animes e viagens. Flamenguista, curte NBA e aprender/criar coisas novas.",
        "hobbies": ["assistir séries", "ver animes", "viajar", "cinema", "criar projetos"],
        "gostos": ["Flamengo", "NBA", "cinema", "séries", "animes"],
        "nao_gosta": ["perder tempo com ruído", "falta de métrica"],
        "musica": ["pagode"],
        "esportes_times": ["Flamengo", "NBA"],
        "series_animes": ["diversos"],
        "comidas_bebidas": ["açai"],
        "rotina": {"melhor_horario_interacao": "manhã/à noite", "estilo_aprendizado": "resumos objetivos + links de referência"},
        "gatilhos_motivacionais": ["OKRs simples", "gráficos de progresso", "tarefas com impacto claro"],
        "tom_preferido": "objetivo, com toques de humor"
    },
    "helena": {
        "nome": "Helena",
        "role": "Mãe",
        "data_nascimento": "1992-03-18",
        "signo": "Peixes",
        "descricao": "A Helena é a mãezona forte e responsável. Pisciana. Ama viajar, falar inglês, praia e sol; fã de café e chocolate.",
        "hobbies": ["viajar", "passear ao ar livre", "praia"],
        "gostos": ["inglês", "praia", "sol", "chocolate", "café"],
        "nao_gosta": ["incertezas sem plano B", "prazos difusos"],
        "musica": ["internacional"],
        "esportes_times": ["flamengo"],
        "series_animes": [],
        "comidas_bebidas": ["chocolate", "café"],
        "rotina": {"melhor_horario_interacao": "manhã (com café) ou tarde", "estilo_aprendizado": "planos claros com próximos passos"},
        "gatilhos_motivacionais": ["roadmap simples", "check-ins curtos", "mostrar progresso"],
        "tom_preferido": "calmo, encorajador e pragmático"
    },
    "giulia": {
        "nome": "Giulia",
        "role": "Filha",
        "data_nascimento": "2015-09-19",
        "signo": "Virgem",
        "descricao": "A Giulia é a caçula elétrica e curiosa da casa. Virginiana, ama inventar brincadeiras, viajar e brincar com as amigas. Flamenguista, adora pagode, piadas e sorvete de flocos.",
        "hobbies": ["brincar", "viajar", "jogar com amigas", "ver vídeos", "inventar brincadeiras"],
        "gostos": ["Flamengo", "pagode", "piadas", "sorvete de flocos"],
        "nao_gosta": ["tarefas longas sem pausa", "explicações muito formais"],
        "musica": ["pagode"],
        "esportes_times": ["Flamengo"],
        "series_animes": ["the chosen"],
        "comidas_bebidas": ["sorvete de flocos", "lasanha"],
        "rotina": {"melhor_horario_interacao": "tarde/início da noite", "estilo_aprendizado": "lúdico, perguntas e respostas, jogos rápidos"},
        "gatilhos_motivacionais": ["desafios curtos", "missões com recompensa", "elogio divertido"],
        "tom_preferido": "alegre, brincalhão, com emojis pontuais"
    },
    "giovanna": {
        "nome": "Giovanna",
        "role": "Filha",
        "data_nascimento": "2008-05-19",
        "signo": "Touro",
        "descricao": "A Giovanna é social e comunicativa. Taurina, ama conversar com amigos, ver vídeos e ouvir música. Flamenguista; curte funk, pagode e trap. Série favorita: The Vampire Diaries. Ama açaí.",
        "hobbies": ["conversar com amigos", "ver vídeos", "assistir séries"],
        "gostos": ["Flamengo", "funk", "pagode", "trap", "The Vampire Diaries", "açaí"],
        "nao_gosta": ["cobranças vagas", "planos sem passo a passo"],
        "musica": ["funk", "pagode", "trap"],
        "esportes_times": ["Flamengo"],
        "series_animes": ["The Vampire Diaries"],
        "comidas_bebidas": ["açaí"],
        "rotina": {"melhor_horario_interacao": "tarde/noite", "estilo_aprendizado": "checklists curtos e práticos"},
        "gatilhos_motivacionais": ["pequenas metas com prazo", "reforço positivo", "mostrar impacto no futuro"],
        "tom_preferido": "amigável e direto, com exemplos rápidos"
    },
    "guilherme": {
        "nome": "Guilherme",
        "role": "Filho",
        "data_nascimento": "2012-12-20",
        "signo": "Sagitário",
        "descricao": "O Guilherme é gamer e fã de animes. Sagitariano, ama jogar com os amigos. Curte vídeos e sorvete de ovomaltine.",
        "hobbies": ["jogar videogame", "assistir animes", "ver vídeos"],
        "gostos": ["Demon Slayer", "Jujutsu Kaisen", "sorvete de ovomaltine"],
        "nao_gosta": ["explicações longas sem exemplo", "tarefas sem 'objetivo'"],
        "musica": ["internacional"],
        "esportes_times": ["flamengo"],
        "series_animes": ["Demon Slayer", "Jujutsu Kaisen"],
        "comidas_bebidas": ["sorvete de ovomaltine","miojo"],
        "rotina": {"melhor_horario_interacao": "fim de tarde/noite", "estilo_aprendizado": "gamificação (fases, pontos, conquistas)"},
        "gatilhos_motivacionais": ["ranking", "XP", "missões diárias"],
        "tom_preferido": "empolgado e objetivo"
    },
    "rayane": {
        "nome": "Rayane",
        "role": "Mãe",
        "data_nascimento": "19/08/1992", 
        "signo": "Leão",
        "descricao": "A Rayane é a mãe da Giulia e da Giovanna e da Olívia",
        "hobbies": [],
        "gostos": [],
        "nao_gosta": [],
        "musica": ["Pagode"],
        "esportes_times": ["Vasco"],
        "series_animes": [],
        "comidas_bebidas": [],
        "rotina": {},
        "gatilhos_motivacionais": [],
        "tom_preferido": "divertido, motivador e simples"
    },
}

# --- Helpers: perfil padronizado (evita cair em "perfil simples") ---
SIGN_PT_TO_EN = {
    "aries": "aries",
    "touro": "taurus",
    "gemeos": "gemini",
    "cancer": "cancer",
    "leao": "leo",
    "virgem": "virgo",
    "libra": "libra",
    "escorpiao": "scorpio",
    "sagitario": "sagittarius",
    "capricornio": "capricorn",
    "aquario": "aquarius",
    "peixes": "pisces",
}



# Nomes canônicos (pt-BR) + emoji do signo (para exibição)
SIGN_NORM_TO_PT = {
    "aries": "Áries",
    "touro": "Touro",
    "gemeos": "Gêmeos",
    "cancer": "Câncer",
    "leao": "Leão",
    "virgem": "Virgem",
    "libra": "Libra",
    "escorpiao": "Escorpião",
    "sagitario": "Sagitário",
    "capricornio": "Capricórnio",
    "aquario": "Aquário",
    "peixes": "Peixes",
}

SIGN_EMOJI = {
    "aries": "♈",
    "touro": "♉",
    "gemeos": "♊",
    "cancer": "♋",
    "leao": "♌",
    "virgem": "♍",
    "libra": "♎",
    "escorpiao": "♏",
    "sagitario": "♐",
    "capricornio": "♑",
    "aquario": "♒",
    "peixes": "♓",
}
def _strip_accents(s: str) -> str:
    if not s:
        return ""
    return (
        s.lower()
        .replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
        .replace("é", "e").replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o").replace("ô", "o").replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )

def _calc_age(birthdate_iso: str) -> Optional[int]:
    """Calcula idade a partir de data:
    - ISO: YYYY-MM-DD
    - BR:  DD/MM/YYYY (ou DD/MM/YY)
    """
    if not birthdate_iso:
        return None
    s = str(birthdate_iso).strip()
    b = None
    # ISO
    try:
        b = date.fromisoformat(s)
    except Exception:
        b = None
    # BR
    if b is None:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
        if m:
            dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if yy < 100:
                yy += 2000
            try:
                b = date(yy, mm, dd)
            except Exception:
                b = None
    if b is None:
        return None
    today = now_sp().date()
    age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    return int(age)
def get_profile(client_id: str) -> Dict[str, Any]:
    cid = normalize_client_id(client_id)
    raw = USER_PROFILES.get(cid) or {}
    name = raw.get("nome") or raw.get("name") or cid.title()
    role = raw.get("role") or raw.get("papel") or ""
    birthdate = raw.get("data_nascimento") or raw.get("birthdate") or ""
    sign = raw.get("signo") or raw.get("sign") or ""
    description = raw.get("descricao") or raw.get("bio") or ""
    tone = raw.get("tom_preferido") or raw.get("tone") or ""

    return {
        "id": cid,
        "name": name,
        "role": role,
        "birthdate": birthdate,
        "age": _calc_age(birthdate) if birthdate else None,
        "sign": sign,
        "sign_norm_pt": _strip_accents(sign),
        "sign_en": SIGN_PT_TO_EN.get(_strip_accents(sign), ""),
        "description": description,
        "hobbies": raw.get("hobbies") or [],
        "likes": raw.get("gostos") or raw.get("likes") or [],
        "dislikes": raw.get("nao_gosta") or raw.get("dislikes") or [],
        "foods": raw.get("comidas_bebidas") or raw.get("foods") or [],
        "music": raw.get("musica") or raw.get("music") or [],
        "series": raw.get("series_animes") or raw.get("series_animes") or [],
        "routine": raw.get("rotina") or {},
        "motivation": raw.get("gatilhos_motivacionais") or [],
        "tone": tone,
    }

def profile_markdown(p: Dict[str, Any]) -> str:
    parts = [f"👤 **{p.get('name','')}**"]
    if p.get("role"):
        parts.append(f"- Papel: {p['role']}")
    if p.get("age") is not None:
        parts.append(f"- Idade: {p['age']} anos")
    if p.get("sign"):
        parts.append(f"- Signo: {p['sign']}")
    if p.get("description"):
        parts.append(f"- Sobre: {p['description']}")
    if p.get("likes"):
        parts.append(f"- Gostos: {', '.join(map(str, p['likes']))}")
    if p.get("hobbies"):
        parts.append(f"- Hobbies: {', '.join(map(str, p['hobbies']))}")
    if p.get("tone"):
        parts.append(f"- Tom preferido: {p['tone']}")
    return "\n".join(parts)



def is_parent(client_id: str) -> bool:
    return (client_id or "").strip().lower() in PARENTS


def normalize_client_id(client_id: str) -> str:
    return (client_id or "").strip().lower()


def extract_profile_target(text: str, default_client_id: str) -> str:
    """Try to identify which family member the user is referring to in the text.

    Examples:
      - "qual o signo da giulia?" -> "giulia"
      - "horóscopo da Giovanna hoje" -> "giovanna"
      - if none found -> default_client_id
    """
    tl = (text or "").lower()

    # 1) match by profile ids (keys)
    for cid in USER_PROFILES.keys():
        cid_norm = normalize_client_id(cid)
        if re.search(rf"\b{re.escape(cid_norm)}\b", tl):
            return cid_norm

    # 2) match by displayed names (nome/name)
    for cid in USER_PROFILES.keys():
        p = get_profile(cid)
        nm = (p.get("name") or "").lower().strip()
        if nm and re.search(rf"\b{re.escape(nm)}\b", tl):
            return normalize_client_id(cid)

    return normalize_client_id(default_client_id)




def postprocess_answer_layout(answer: str) -> str:
    """Standardize layouts for school modules to tables when possible."""
    if not answer:
        return answer
    # Agenda: if starts with "📅" and has lines with dates, attempt to table it
    if answer.startswith("📅") and "ID #" in answer and "\n" in answer and "| Data |" not in answer:
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        header = lines[0]
        items = []
        for ln in lines[1:]:
            mm = re.match(r"(\\d{2}/\\d{2}/\\d{4}).*?ID\\s*#(\\d+)\\)\\s*:\\s*(.*)$", ln)
            if mm:
                items.append([mm.group(1), f"#{mm.group(2)}", mm.group(3)])
        if items:
            return header + "\n\n" + md_table(["Data", "ID", "Evento"], items)
    return answer
def apply_followup_context(client_id: str, question: str) -> str:
    """
    Makes the conversation more fluid with short follow-ups.

    Timetable:
      "a giulia tem aula de que terça?" -> sets ctx(domain=timetable, student=giulia)
      "e na quinta?" -> becomes "giulia tem aula de que quinta?"

    Agenda:
      "qual é a agenda da giulia esse mês?" -> ctx(domain=agenda, student=giulia)
      "e em março?" / "e no próximo mês?" / "e no dia 12?" -> becomes a more specific agenda query for same student

    Grades:
      "notas da giulia no 1B" -> ctx(domain=grades, student=giulia, bimester=1B)
      "e em português?" -> becomes "notas da giulia em português no 1B"
      "e no 2B?" -> becomes "notas da giulia no 2B"
    """
    cid = normalize_client_id(client_id)
    q = (question or "").strip()
    ql = q.lower().strip()

    ctx = LAST_CTX.get(cid) or {}
    if not ctx:
        return q

    domain = ctx.get("domain")
    student = ctx.get("student")

    # ---------- Timetable follow-ups ----------
    m = re.match(r"^(e\s+)?(na\s+)?(segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)\b", ql)
    if m and domain == "timetable" and student:
        wd = m.group(3)
        return f"{student} tem aula de que {wd}?"

    m2 = re.match(r"^(e\s+)?(amanhã|amanha|hoje)\b", ql)
    if m2 and domain == "timetable" and student:
        token = m2.group(2)
        return f"Que aula {student} tem {token}?"

    # ---------- Agenda follow-ups ----------
    if domain == "agenda" and student:
        # "e em março?" / "e em marco?"
        m3 = re.match(r"^(e\s+)?em\s+([a-zçãõáéíóú]+)\b", ql)
        if m3:
            mes = m3.group(2)
            return f"Qual é a agenda da {student} em {mes}?"
        # "e no próximo mês?" / "e no proximo mes?"
        if re.match(r"^(e\s+)?(no\s+)?próximo\s+m[eê]s\b", ql) or re.match(r"^(e\s+)?(no\s+)?proximo\s+mes\b", ql):
            return f"Qual é a agenda da {student} no próximo mês?"
        # "e no dia 12?" / "e dia 12?"
        m4 = re.match(r"^(e\s+)?(no\s+dia\s+|dia\s+)(\d{1,2})\b", ql)
        if m4:
            dia = m4.group(3)
            return f"Qual é a agenda da {student} no dia {dia}?"
        # "e amanhã?" / "e hoje?"
        m5 = re.match(r"^(e\s+)?(amanhã|amanha|hoje)\b", ql)
        if m5:
            token = m5.group(2)
            return f"O que a {student} tem {token} na escola?"
        # "e semana que vem?"
        if re.search(r"semana\s+que\s+vem", ql):
            return f"Qual é a agenda da {student} na próxima semana?"

    # ---------- Grades follow-ups ----------
    if domain == "grades" and student:
        bim = ctx.get("bimester")
        subj = ctx.get("subject")

        # "e no 2B?" / "e 2b?"
        m6 = re.match(r"^(e\s+)?(no\s+)?(\d)\s*b\b", ql)
        if m6:
            b = m6.group(3)
            if subj:
                return f"Notas da {student} em {subj} no {b}B"
            return f"Notas da {student} no {b}B"

        # "e em português?" / "e portugues?"
        m7 = re.match(r"^(e\s+)?(em\s+)?([a-zçãõáéíóú\.\- ]+)\??$", ql)
        if m7:
            possible = m7.group(3).strip()
            # avoid catching very generic tokens
            if possible and possible not in {"isso", "essa", "esse", "também", "tambem", "ok", "beleza"}:
                # if user asks "e a média?"
                if "média" in possible or "media" in possible:
                    if subj and bim:
                        return f"Média da {student} em {subj} no {bim}"
                    if subj:
                        return f"Média da {student} em {subj}"
                # treat as subject follow-up
                if bim:
                    return f"Notas da {student} em {possible} no {bim}"
                return f"Notas da {student} em {possible}"


    # ---------- Horoscope follow-ups ----------
    if domain == "horoscope":
        # Examples:
        #   "qual meu horoscopo hoje?" -> ctx(domain=horoscope, target=glauco)
        #   "e da giovanna?" -> becomes "qual o horóscopo da giovanna hoje?"
        target = ctx.get("target") or ctx.get("student")
        # "e da <nome>?" / "e do <nome>?"
        m8 = re.match(r"^(e\s+)?d[ao]\s+([a-zçãõáéíóú]+)\b", ql)
        if m8:
            who = m8.group(2)
            return f"Qual o horóscopo da {who} hoje?"
        # "e dela?" / "e dele?"
        if re.match(r"^(e\s+)?del[ae]\b", ql):
            if target:
                return f"Qual o horóscopo da {target} hoje?"
        # If user only says "e o dela?" or "e o meu?" keep target
        if re.match(r"^(e\s+)?(o\s+)?(meu|minha)\b", ql) and target:
            return f"Qual o meu horóscopo hoje?"

    return q
def get_payload(request):
    """
    Robust payload reader:
    - Tries Flask request.get_json()
    - Falls back to reading raw bytes and decoding as UTF-8 / UTF-8-SIG / UTF-16
    - Falls back to form fields (multipart/form-data or application/x-www-form-urlencoded)
    """
    data = request.get_json(silent=True)
    if isinstance(data, dict) and data:
        return data

    # raw bytes fallback (PowerShell can send UTF-16 sometimes)
    raw = request.get_data(cache=False) or b""
    if raw:
        for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
            try:
                txt = raw.decode(enc)
                txt = txt.strip()
                if txt:
                    obj = json.loads(txt)
                    if isinstance(obj, dict):
                        return obj
            except Exception:
                continue

    # form fallback
    try:
        form = request.form.to_dict(flat=True)
        if form:
            return form
    except Exception:
        pass

    return {}

def can_access_student(requester_id: str, student_id: str) -> bool:
    """Leitura: pais podem ver qualquer criança; criança vê apenas a si mesma.
    Escrita: apenas pais (checar separadamente com is_parent).
    """
    requester = normalize_client_id(requester_id)
    student = normalize_client_id(student_id)
    if requester in PARENTS:
        return True
    return requester == student

# =============================================================================
# Manual / Ajuda (IMPORTANTE: fica dentro do código)
# =============================================================================

SCHOOL_MANUAL = r"""
📚 **Manual GOOMI Escola — comandos que eu reconheço**

**Regras rápidas**
- ✅ **Pais (Glauco/Helena/Rayane)**: podem **cadastrar / editar / excluir / consultar**
- 👧🧒 **Crianças (Giulia/Giovanna/Guilherme)**: podem **somente consultar (leitura)**

---

## 1) Grade de Horários (matérias por dia da semana)

### Consultar (qualquer pessoa)
- "Giulia tem aula de que hoje?"
- "Giovanna tem aula de que amanhã?"
- "Que aula o Guilherme tem na quarta?"
- "Mostra a grade semanal da Giulia"

### Cadastrar / atualizar (somente pais)
**Dia específico**
- "Giulia segunda: matemática e português"
- "Giovanna terça: artes e espanhol"
- "Guilherme quarta: ciências e inglês"

**Semana toda**
- "Giulia grade semana: segunda matemática e português; terça história e geografia; quarta ciências e inglês"

**Substituir dia inteiro**
- "Atualizar terça da Giovanna para: artes e espanhol"
- "Trocar a quarta da Giulia para: ciências e inglês"

### Excluir (somente pais)
**Remover uma matéria**
- "Remover inglês da Giulia na quarta"
- "Excluir artes da Giovanna na terça"

**Limpar um dia**
- "Apagar a terça do Guilherme"
- "Limpar segunda da Giulia"

---

## 2) Agenda Escolar (provas, seminários, festas, revisões…)

### Consultar (qualquer pessoa)
- "Qual a agenda da Giovanna esse mês?"
- "O que a Giulia tem hoje na escola?"
- "Quando é a próxima prova do Guilherme?"
- "Quando é o seminário da Giovanna?"
- "Quais eventos eu tenho essa semana?"

### Cadastrar (somente pais)
- "Adicionar prova de matemática da Giulia dia 12/03"
- "Agenda: Giovanna seminário dia 20/03"
- "Cadastrar apresentação do Guilherme dia 15/04"
- "Adicionar festa da escola dia 10/05"
- "Adicionar revisão de português da Giulia dia 08/03"
- "Adicionar prova de história da Giovanna dia 12/03, assunto Revolução Francesa"

### Excluir / editar (somente pais)
> Quando eu listar eventos, eu mostro um **ID** (ex: #12). Use:
- "Excluir evento #12"
- "Apagar agenda #7"
- "Atualizar evento #12 para dia 13/03"
- "Editar evento #12: trocar para seminário"

---

## 3) Notas e Médias (por bimestre e avaliação)

### Consultar (qualquer pessoa)
- "Notas da Giulia no 1B"
- "Notas da Giovanna em português no 3B"
- "Qual a média do Guilherme em ciências no 2B?"
- "Me mostra as notas do 4B da Giulia"
- "Quais são minhas médias no 1B?"

### Cadastrar nota (somente pais)
- "P1 de matemática da Giulia do 1B foi 8,5"
- "P2 da Giovanna de português do 3B: 7,0"
- "Trabalho de história do Guilherme no 2B: 9,0"
- "Prova de ciências da Giulia no 1B: 8,0"

### Cadastrar média (somente pais)
- "Média da Giulia em matemática no 1B: 8,2"
- "Média do Guilherme em geografia no 2B foi 7,6"

### Excluir / corrigir (somente pais)
- "Excluir P1 de matemática da Giulia no 1B"
- "Remover P2 de português da Giovanna no 3B"
- "Corrigir P1 de matemática da Giulia no 1B para 9,0"
- "Atualizar média da Giulia em matemática no 1B para 8,6"

---

## Datas que eu entendo
- hoje, amanhã, ontem
- segunda, terça, quarta, quinta, sexta, sábado, domingo
- "esse mês", "essa semana"
""".strip()


GENERAL_MANUAL = r"""
🧠 **Manual Geral do GOOMI**

Você pode pedir:
- "ajuda" / "manual"
- "manual escola" (grade, agenda, notas)
- "manual horóscopo"
- "manual web"
- "manual geral"

E também pode conversar normalmente: "me ajuda a estudar frações", "faz um resumo de um tema", etc.
""".strip()


WEB_MANUAL = r"""
🌐 **Manual Busca na Web (genérica)**

Exemplos:
- "pesquisar: o que é fotossíntese"
- "buscar na web: melhores técnicas de estudo"
- "resumo com fontes: revolução industrial"

Eu respondo com um resumo curto e 1–3 fontes.
""".strip()


HOROSCOPE_MANUAL = r"""
✨ **Manual Horóscopo**

Exemplos:
- "horóscopo de hoje de leão"
- "meu horóscopo de hoje" (se o seu perfil tiver signo configurado)
""".strip()


# =============================================================================
# Banco (SQLite)
# =============================================================================

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(CONFIG["db_path"])
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    conn = get_db()
    cur = conn.cursor()

    # tabela família (mantém)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS familia (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id TEXT UNIQUE,
        nome TEXT
    )
    """)

    # --- Escola: grade de horários (sem hora)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS school_timetable (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT NOT NULL,
        weekday INTEGER NOT NULL,
        subject TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(student, weekday, subject)
    )
    """)

    # --- Escola: agenda
    cur.execute("""
    CREATE TABLE IF NOT EXISTS school_agenda (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT,
        title TEXT NOT NULL,
        type TEXT,
        event_date TEXT NOT NULL,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- Escola: notas e médias (novo modelo)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS school_grades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT NOT NULL,
        bimester TEXT NOT NULL,
        subject TEXT NOT NULL,
        assessment TEXT NOT NULL,
        grade REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(student, bimester, subject, assessment)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS school_averages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT NOT NULL,
        bimester TEXT NOT NULL,
        subject TEXT NOT NULL,
        average REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(student, bimester, subject)
    )
    """)

    # seed familia
    for cid in USER_PROFILES.keys():
        p = get_profile(cid)
        cur.execute(
            "INSERT OR IGNORE INTO familia (client_id, nome) VALUES (?, ?)",
            (p["id"], str(p.get("name") or p["id"]).strip()),
        )

    conn.commit()
    conn.close()


# =============================================================================
# Utilidades (datas, dias da semana)
# =============================================================================

WEEKDAY_MAP = {
    "segunda": 0, "seg": 0, "seg.": 0,
    "terça": 1, "terca": 1, "ter": 1, "ter.": 1,
    "quarta": 2, "qua": 2, "qua.": 2,
    "quinta": 3, "qui": 3, "qui.": 3,
    "sexta": 4, "sex": 4, "sex.": 4,
    "sábado": 5, "sabado": 5, "sáb": 5, "sab": 5,
    "domingo": 6, "dom": 6, "dom.": 6,
}

WEEKDAY_NAME = {0: "segunda-feira", 1: "terça-feira", 2: "quarta-feira", 3: "quinta-feira", 4: "sexta-feira", 5: "sábado", 6: "domingo"}

def md_table(headers, rows):
    """Return a GitHub/Markdown table string (works well in Streamlit)."""
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(str(c) if c is not None else "—" for c in row) + " |" for row in rows)
    return "\n".join([h, sep, body]) if body else "\n".join([h, sep])




def parse_weekday(text: str) -> Optional[int]:
    t = (text or "").lower()
    # tenta achar um token de dia
    for k, v in WEEKDAY_MAP.items():
        if re.search(rf"\b{k}\b", t):
            return v
    return None


def weekday_from_relative(text: str) -> Optional[int]:
    t = (text or "").lower()
    today = now_sp().date()
    if re.search(r"\bhoje\b", t):
        return today.weekday()
    if re.search(r"\bamanh[ãa]\b", t):
        return (today + timedelta(days=1)).weekday()
    if re.search(r"\bontem\b", t):
        return (today - timedelta(days=1)).weekday()
    return None


def parse_date_ptbr(text: str) -> Optional[date]:
    """
    Aceita dd/mm ou dd/mm/aaaa. Se não tiver ano, assume ano atual.
    Também entende hoje/amanhã/ontem.
    """
    t = (text or "").lower().strip()
    drel = None
    if "hoje" in t:
        drel = now_sp().date()
    elif "amanh" in t:
        drel = now_sp().date() + timedelta(days=1)
    elif "ontem" in t:
        drel = now_sp().date() - timedelta(days=1)
    if drel:
        return drel

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if not m:
        return None
    day = int(m.group(1))
    month = int(m.group(2))
    year_s = m.group(3)
    if year_s:
        year = int(year_s)
        if year < 100:
            year += 2000
    else:
        year = now_sp().year
    try:
        return date(year, month, day)
    except ValueError:
        return None


def split_subjects(s: str) -> List[str]:
    s = (s or "").strip()
    s = re.sub(r"[.;]", " ", s)
    parts = re.split(r"\s*(?:,| e | & |/)\s*", s, flags=re.IGNORECASE)
    clean = []
    for p in parts:
        p2 = p.strip()
        if p2:
            clean.append(p2.lower())
    # remove duplicados preservando ordem
    seen = set()
    out = []
    for x in clean:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =============================================================================
# Intents — Grade semanal
# =============================================================================

def looks_like_timetable_query(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in ["grade", "aula", "aulas", "horário", "horarios", "matéria", "matérias", "materia", "materias"]) and \
           any(kw in t for kw in ["tem", "tenho", "que", "qual", "mostra", "ver"])


def looks_like_timetable_write(text: str) -> bool:
    t = (text or "").lower()
    # escrita só quando há sinal claro de cadastro/edição
    # - verbo (cadastrar/adicionar/incluir/atualizar/trocar/remover/excluir/apagar/limpar)
    # - ou padrão "dia:"
    # - ou "grade semana:" (com dois-pontos)
    has_verb = any(kw in t for kw in ["atualizar", "trocar", "cadastrar", "adicionar", "incluir", "remover", "excluir", "apagar", "limpar"])
    has_day_colon = re.search(r"\b(segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)\b\s*:", t) is not None
    has_grade_semana_colon = re.search(r"\bgrade\s+semana\b\s*:", t) is not None
    return has_verb or has_day_colon or has_grade_semana_colon


def extract_student(text: str, fallback_client_id: str) -> str:
    t = (text or "").lower()
    for s in ["giulia", "giovanna", "guilherme"]:
        if re.search(rf"\b{s}\b", t):
            return s
    # se criança perguntando sobre si mesma, usa fallback
    if fallback_client_id in CHILDREN:
        return fallback_client_id
    # default: giulia (mas ideal é pedir, aqui mantemos simples)
    return "giulia"


def timetable_set_day(conn: sqlite3.Connection, student: str, weekday: int, subjects: List[str]) -> None:
    cur = conn.cursor()
    # limpa dia e reinsere
    cur.execute("DELETE FROM school_timetable WHERE student=? AND weekday=?", (student, weekday))
    for subj in subjects:
        cur.execute(
            "INSERT OR IGNORE INTO school_timetable (student, weekday, subject) VALUES (?, ?, ?)",
            (student, weekday, subj),
        )
    conn.commit()


def timetable_add_subjects(conn: sqlite3.Connection, student: str, weekday: int, subjects: List[str]) -> None:
    cur = conn.cursor()
    for subj in subjects:
        cur.execute(
            "INSERT OR IGNORE INTO school_timetable (student, weekday, subject) VALUES (?, ?, ?)",
            (student, weekday, subj),
        )
    conn.commit()


def timetable_remove_subject(conn: sqlite3.Connection, student: str, weekday: int, subject: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM school_timetable WHERE student=? AND weekday=? AND subject=?",
        (student, weekday, subject.lower().strip()),
    )
    conn.commit()
    return cur.rowcount


def timetable_clear_day(conn: sqlite3.Connection, student: str, weekday: int) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM school_timetable WHERE student=? AND weekday=?", (student, weekday))
    conn.commit()
    return cur.rowcount


def timetable_get_day(conn: sqlite3.Connection, student: str, weekday: int) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT subject FROM school_timetable WHERE student=? AND weekday=? ORDER BY subject ASC",
        (student, weekday),
    )
    return [r["subject"] for r in cur.fetchall()]


def timetable_get_week(conn: sqlite3.Connection, student: str) -> Dict[int, List[str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT weekday, subject FROM school_timetable WHERE student=? ORDER BY weekday ASC, subject ASC",
        (student,),
    )
    out: Dict[int, List[str]] = {i: [] for i in range(7)}
    for r in cur.fetchall():
        out[int(r["weekday"])].append(r["subject"])
    return out


def handle_timetable(text: str, client_id: str) -> Optional[str]:
    t = (text or "").strip()
    tl = t.lower()
    student = extract_student(t, client_id)

    # Contexto: facilita perguntas em sequência (ex: 'e na quinta?')
    LAST_CTX[normalize_client_id(client_id)] = {"domain": "timetable", "student": student}

    conn = get_db()

    # --- WRITE
    if looks_like_timetable_write(t):
        if not is_parent(client_id):
            conn.close()
            return "Essa parte (cadastrar/editar) só o papai ou a mamãe podem fazer 😊. Quer que eu mostre a grade que já está salva?"

        # limpar dia
        if re.search(r"\b(limpar|apagar)\b", tl) and parse_weekday(t) is not None:
            wd = parse_weekday(t)
            removed = timetable_clear_day(conn, student, wd)
            conn.close()
            return f"✅ Pronto! Apaguei {removed} matéria(s) de {student.title()} na {WEEKDAY_NAME[wd]}."

        # remover matéria do dia
        if re.search(r"\b(remover|excluir)\b", tl) and parse_weekday(t) is not None:
            wd = parse_weekday(t)
            # tenta pegar matéria depois de remover/excluir
            m = re.search(r"\b(?:remover|excluir)\b\s+(.+?)\s+\b(?:da|do|na|no)\b", tl)
            subject = None
            if m:
                subject = m.group(1).strip()
            else:
                # fallback: depois do verbo até o fim
                m2 = re.search(r"\b(?:remover|excluir)\b\s+(.+)$", tl)
                subject = (m2.group(1).strip() if m2 else None)
            if not subject:
                conn.close()
                return "Me diga qual matéria você quer remover (ex: \"Remover inglês da Giulia na quarta\")."
            removed = timetable_remove_subject(conn, student, wd, subject)
            conn.close()
            return f"✅ Pronto! Removi {removed} ocorrência(s) de **{subject}** de {student.title()} na {WEEKDAY_NAME[wd]}."

        # grade semana: dia ...; dia ...
        if re.search(r"\bgrade\s+semana\b\s*:", tl):
            # exemplo: "Giulia grade semana: segunda matemática e português; terça história e geografia"
            after = tl.split("grade semana", 1)[1]
            after = after.lstrip(" :,-")
            chunks = [c.strip() for c in re.split(r"\s*;\s*", after) if c.strip()]
            updated_days = 0
            # substitui a semana inteira: limpa tudo antes
            conn.execute("DELETE FROM school_timetable WHERE student=?", (student,))
            conn.commit()
            for ch in chunks:
                wd = parse_weekday(ch)
                if wd is None:
                    continue
                # remove "segunda:" se existir
                ch2 = re.sub(r"^\s*(segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)\s*:?\s*", "", ch, flags=re.IGNORECASE)
                subs = split_subjects(ch2)
                if subs:
                    timetable_set_day(conn, student, wd, subs)
                    updated_days += 1
            conn.close()
            if updated_days:
                return f"✅ Grade semanal de {student.title()} atualizada em {updated_days} dia(s)."
            return "Não consegui identificar os dias e matérias. Ex: \"Giulia grade semana: segunda matemática e português; terça história e geografia\""

        
        # múltiplos dias no mesmo comando (sem "grade semana")
        # Ex.: "Giulia terça: história e geografia; quarta: ciências e inglês"
        if ";" in t and re.search(r"\b(segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)\b\s*:", tl):
            parts = [p.strip() for p in re.split(r"\s*;\s*", t) if p.strip()]
            updated = []
            for p in parts:
                if not re.search(r"\b(segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)\b\s*:", p.lower()):
                    continue
                wd_p = parse_weekday(p)
                if wd_p is None:
                    continue
                day_part_p = re.split(r":", p, maxsplit=1)[1].strip()
                subs_p = split_subjects(day_part_p)
                if subs_p:
                    timetable_set_day(conn, student, wd_p, subs_p)
                    updated.append((wd_p, subs_p))
            conn.close()
            if updated:
                # monta resposta amigável
                lines = []
                for wd_u, subs_u in updated:
                    lines.append(f"{WEEKDAY_NAME[wd_u]}: {', '.join(subs_u)}")
                return "✅ Salvo! " + student.title() + " — " + " | ".join(lines) + "."
            # se não atualizou nada, cai para o handler padrão abaixo

# "terça: artes e espanhol" (set day)
        if re.search(r"\b(segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo)\b\s*:", tl):
            wd = parse_weekday(t)
            day_part = re.split(r":", t, maxsplit=1)[1]
            subs = split_subjects(day_part)
            if subs:
                timetable_set_day(conn, student, wd, subs)
                conn.close()
                return f"✅ Salvo! {student.title()} na {WEEKDAY_NAME[wd]}: " + ", ".join(subs) + "."
            conn.close()
            return "Não entendi as matérias. Ex: \"Giulia segunda: matemática e português\""

        # "Atualizar terça da Giovanna para: artes e espanhol"
        if re.search(r"\b(atualizar|trocar)\b", tl) and parse_weekday(t) is not None:
            wd = parse_weekday(t)
            # pega após "para:"
            m = re.search(r"\bpara\s*:?\s*(.+)$", t, flags=re.IGNORECASE)
            subs = split_subjects(m.group(1)) if m else []
            if subs:
                timetable_set_day(conn, student, wd, subs)
                conn.close()
                return f"✅ Atualizado! {student.title()} na {WEEKDAY_NAME[wd]}: " + ", ".join(subs) + "."
            conn.close()
            return "Me diga as matérias após \"para\". Ex: \"Atualizar terça da Giovanna para: artes e espanhol\""

        # "Giovanna tem aula de artes e espanhol nas terças-feiras" (add)
        if re.search(r"\btem aula de\b", tl) and (parse_weekday(t) is not None):
            wd = parse_weekday(t)
            m = re.search(r"\btem aula de\b\s*(.+?)\s+\bna[s]?\b", tl)
            subs = split_subjects(m.group(1)) if m else []
            if subs:
                timetable_add_subjects(conn, student, wd, subs)
                conn.close()
                return f"✅ Salvo! {student.title()} na {WEEKDAY_NAME[wd]}: " + ", ".join(subs) + "."
            conn.close()
            return "Não entendi as matérias. Ex: \"Giovanna tem aula de artes e espanhol nas terças-feiras\""

        conn.close()
        return None

    # --- READ
    if looks_like_timetable_query(t):
        # Prioridade: pedidos de "grade semanal/completa" (lista seg–sex)
        # Observação: usamos um matcher dedicado para não confundir com consulta de um dia.
        wants_week = bool(
            re.search(r"\bgrade\b", tl)
            and (
                re.search(r"\b(semanal|semana\s+inteira|completa|completo)\b", tl)
                or re.search(r"\b(mostra|mostrar|exibir|listar)\b.*\bgrade\b", tl)
                or re.search(r"\bqual\s+e\s+a\s+grade\b", tl)
                or re.search(r"\bqual\s+é\s+a\s+grade\b", tl)
                or re.search(r"\bgrade\s+da\b", tl)
            )
        )

        if wants_week:
            week = timetable_get_week(conn, student)
            conn.close()

            # Opção A: sempre mostrar seg–sex, mesmo vazio
            rows = []
            any_sub = False
            for i in range(0, 5):
                subs = week.get(i, [])
                if subs:
                    any_sub = True
                    rows.append((WEEKDAY_NAME[i], ", ".join(subs)))
                else:
                    rows.append((WEEKDAY_NAME[i], "—"))

            if not any_sub:
                return (
                    f"Ainda não há matérias cadastradas para {student.title()}. "
                    f'Pais podem cadastrar com: "{student.title()} grade semana: segunda ...; terça ...".'
                )

            # Markdown table (renderiza bem no Streamlit)
            md = [f"📅 **{student.title()} — grade semanal**", "", "| Dia | Matérias |", "|---|---|"]
            for d, s in rows:
                md.append(f"| {d} | {s} |")
            return "\n".join(md)

        wd = weekday_from_relative(t) or parse_weekday(t)
        if wd is not None:
            subs = timetable_get_day(conn, student, wd)
            conn.close()
            if subs:
                return f"{student.title()} tem aula de **" + " e ".join(subs) + f"** na {WEEKDAY_NAME[wd]}."
            return f"Não encontrei matérias cadastradas para {student.title()} na {WEEKDAY_NAME[wd]}. (Pais podem cadastrar pelo manual.)"

        conn.close()
        return None

    conn.close()
    return None


# =============================================================================
# Intents — Agenda escolar
# =============================================================================

AGENDA_TYPES = ["prova", "seminário", "seminario", "apresentação", "apresentacao", "revisão", "revisao", "festa", "trabalho", "entrega"]


def looks_like_agenda(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["agenda", "prova", "semin", "apresent", "revis", "festa", "trabalho", "evento"])


def agenda_add(conn: sqlite3.Connection, student: Optional[str], title: str, type_: Optional[str], event_date: date, notes: Optional[str]) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO school_agenda (student, title, type, event_date, notes) VALUES (?, ?, ?, ?, ?)",
        (student, title.strip(), type_, event_date.isoformat(), notes),
    )
    conn.commit()
    return int(cur.lastrowid)


def agenda_list_range(conn: sqlite3.Connection, student: Optional[str], start: date, end: date) -> List[sqlite3.Row]:
    cur = conn.cursor()
    if student:
        cur.execute(
            "SELECT * FROM school_agenda WHERE student=? AND event_date>=? AND event_date<=? ORDER BY event_date ASC, id ASC",
            (student, start.isoformat(), end.isoformat()),
        )
    else:
        cur.execute(
            "SELECT * FROM school_agenda WHERE event_date>=? AND event_date<=? ORDER BY event_date ASC, id ASC",
            (start.isoformat(), end.isoformat()),
        )
    return cur.fetchall()


def agenda_delete(conn: sqlite3.Connection, event_id: int) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM school_agenda WHERE id=?", (event_id,))
    conn.commit()
    return cur.rowcount


def handle_agenda(text: str, client_id: str) -> Optional[str]:
    t = (text or "").strip()
    tl = t.lower()
    if not looks_like_agenda(t):
        return None

    student = extract_student(t, client_id)
    conn = get_db()

    # delete by #id
    mdel = re.search(r"#(\d+)", tl)
    if mdel and any(k in tl for k in ["excluir", "apagar", "remover", "deletar"]):
        if not is_parent(client_id):
            conn.close()
            return "Só o papai ou a mamãe podem apagar eventos 😊."
        eid = int(mdel.group(1))
        n = agenda_delete(conn, eid)
        conn.close()
        return f"✅ Pronto! Apaguei {n} evento(s) com ID #{eid}."

    # add
    if any(k in tl for k in ["adicionar", "cadastrar", "agenda:"]):
        if not is_parent(client_id):
            conn.close()
            return "Só o papai ou a mamãe podem cadastrar eventos 😊. Quer que eu liste a agenda?"
        d = parse_date_ptbr(t)
        if not d:
            conn.close()
            return "Me diga a data (ex: 12/03). Ex: \"Adicionar prova de matemática da Giulia dia 12/03\""
        # type
        type_ = None
        for tp in AGENDA_TYPES:
            if tp in tl:
                type_ = tp
                break
        # title: remove student and date
        title = re.sub(r"\b(giulia|giovanna|guilherme)\b", "", tl, flags=re.IGNORECASE).strip()
        title = re.sub(r"\b(adicionar|cadastrar|agenda:)\b", "", title, flags=re.IGNORECASE).strip()
        title = re.sub(r"\b(dia)\b\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", title, flags=re.IGNORECASE).strip()
        title = re.sub(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", title).strip()
        title = title.strip(" -:;,.")
        if not title:
            title = "Evento escolar"
        eid = agenda_add(conn, student, title, type_, d, None)
        conn.close()
        return f"✅ Evento cadastrado para {student.title()} em {d.strftime('%d/%m/%Y')} (ID #{eid}): **{title}**."

    # query: hoje/amanhã/essa semana/esse mês
    if any(k in tl for k in ["hoje", "amanh", "essa semana", "este mês", "esse mês", "mes", "mês", "próxima", "proxima"]):
        today = now_sp().date()
        if "esse mês" in tl or "este mês" in tl or "mês" in tl or "mes" in tl:
            start = date(today.year, today.month, 1)
            # end = last day of month
            next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
            end = next_month - timedelta(days=1)
        elif "essa semana" in tl:
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)
        elif "amanh" in tl:
            start = today + timedelta(days=1)
            end = start
        else:
            start = today
            end = today

        rows = agenda_list_range(conn, student if any(s in tl for s in CHILDREN) else student, start, end)
        conn.close()
        if not rows:
            label = "hoje" if start == end == today else f"de {start.strftime('%d/%m')} a {end.strftime('%d/%m')}"
            return f"Não encontrei eventos para {student.title()} {label}."
        lines = [f"🗓️ **Agenda de {student.title()}** ({start.strftime('%d/%m')}–{end.strftime('%d/%m')})"]
        for r in rows:
            d = datetime.fromisoformat(r["event_date"]).date()
            tp = f"[{r['type']}]" if r["type"] else ""
            lines.append(f"- {d.strftime('%d/%m')}: {tp} **{r['title']}** (#{r['id']})")
        return "\n".join(lines)

    # generic list
    if "agenda" in tl:
        today = now_sp().date()
        start = today
        end = today + timedelta(days=30)
        rows = agenda_list_range(conn, student, start, end)
        conn.close()
        if not rows:
            return f"Não encontrei eventos próximos para {student.title()}."
        lines = [f"🗓️ **Próximos eventos — {student.title()}** (30 dias)"]
        for r in rows[:15]:
            d = datetime.fromisoformat(r["event_date"]).date()
            tp = f"[{r['type']}]" if r["type"] else ""
            lines.append(f"- {d.strftime('%d/%m')}: {tp} **{r['title']}** (#{r['id']})")
        return "\n".join(lines)

    return None


# =============================================================================
# Intents — Notas e Médias
# =============================================================================

def looks_like_grades(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["nota", "notas", "média", "media", "p1", "p2", "b1", "1b", "2b", "3b", "4b"])


def parse_bimester(text: str) -> Optional[str]:
    t = (text or "").lower()
    m = re.search(r"\b([1-4])\s*b\b", t)
    if m:
        return f"{m.group(1)}B"
    return None


def parse_assessment(text: str) -> Optional[str]:
    t = (text or "").lower()
    m = re.search(r"\b(p[1-4])\b", t)
    if m:
        return m.group(1).upper()
    # termos comuns
    for k in ["trabalho", "prova", "seminario", "seminário", "apresentacao", "apresentação"]:
        if k in t:
            return k.upper()
    return None


def parse_grade_value(text: str) -> Optional[float]:
    # aceita 8,5 ou 8.5
    m = re.search(r"\b(\d{1,2})([.,](\d{1,2}))?\b", (text or ""))
    if not m:
        return None
    whole = int(m.group(1))
    frac = m.group(3)
    val = float(f"{whole}.{frac or '0'}")
    if 0 <= val <= 10.0:
        return val
    return None


def grades_upsert(conn: sqlite3.Connection, student: str, bimester: str, subject: str, assessment: str, grade: float) -> None:
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO school_grades (student, bimester, subject, assessment, grade)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(student, bimester, subject, assessment)
    DO UPDATE SET grade=excluded.grade, created_at=CURRENT_TIMESTAMP
    """, (student, bimester, subject.lower().strip(), assessment.upper().strip(), grade))
    conn.commit()


def averages_upsert(conn: sqlite3.Connection, student: str, bimester: str, subject: str, avg: float) -> None:
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO school_averages (student, bimester, subject, average)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(student, bimester, subject)
    DO UPDATE SET average=excluded.average, created_at=CURRENT_TIMESTAMP
    """, (student, bimester, subject.lower().strip(), avg))
    conn.commit()


def grades_delete(conn: sqlite3.Connection, student: str, bimester: str, subject: str, assessment: str) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM school_grades WHERE student=? AND bimester=? AND subject=? AND assessment=?",
                (student, bimester, subject.lower().strip(), assessment.upper().strip()))
    conn.commit()
    return cur.rowcount


def averages_delete(conn: sqlite3.Connection, student: str, bimester: str, subject: str) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM school_averages WHERE student=? AND bimester=? AND subject=?",
                (student, bimester, subject.lower().strip()))
    conn.commit()
    return cur.rowcount


def handle_grades(text: str, client_id: str) -> Optional[str]:
    t = (text or "").strip()
    tl = t.lower()
    if not looks_like_grades(t):
        return None

    student = extract_student(t, client_id)
    conn = get_db()

    # write: notas (somente quando há sinal claro de cadastro/edição)
    # Ex.: "P1 ... foi 8,5", "Média ...: 7,2", "nota ... = 9"
    has_bim = bool(parse_bimester(t))
    has_number = bool(re.search(r"\b\d+(?:[\.,]\d+)?\b", tl))
    has_write_marker = any(k in tl for k in ["foi", ":", "=", "corrigir", "atualizar", "editar"]) or re.search(r"\b(p1|p2|p3|p4)\b", tl)
    if has_bim and has_number and has_write_marker:
        if any(k in tl for k in ["excluir", "remover", "apagar"]):
            if not is_parent(client_id):
                conn.close()
                return "Só o papai ou a mamãe podem excluir/corrigir notas 😊."
            bim = parse_bimester(t)
    ctx_bim = bim
    if ctx_bim:
        LAST_CTX[normalize_client_id(client_id)].update({"bimester": ctx_bim})
        ass = parse_assessment(t)
            # subject: tenta achar "de <materia>"
        sm = re.search(r"\bde\s+([a-zçãõáéíóú ]+?)\s+(?:da|do)\b", tl)
        subject = sm.group(1).strip() if sm else None
    ctx_subj = (subject or "").strip()
    if ctx_subj:
        LAST_CTX[normalize_client_id(client_id)].update({"subject": ctx_subj})
        if bim and subject and ass:
                n = grades_delete(conn, student, bim, subject, ass)
                conn.close()
                return f"✅ Pronto! Excluí {n} registro(s) de {ass} em {subject} de {student.title()} no {bim}."
            # média delete
        if bim and subject and ("média" in tl or "media" in tl) and not ass:
                n = averages_delete(conn, student, bim, subject)
                conn.close()
                return f"✅ Pronto! Excluí {n} média(s) de {subject} de {student.title()} no {bim}."
        conn.close()
        return "Para excluir, use algo como: \"Excluir P1 de matemática da Giulia no 1B\"."

        if not is_parent(client_id):
            conn.close()
            return "Só o papai ou a mamãe podem cadastrar/corrigir notas 😊."

        bim = parse_bimester(t)
        if not bim:
            conn.close()
            return "Me diga o bimestre (1B, 2B, 3B ou 4B). Ex: \"P1 de matemática da Giulia do 1B foi 8,5\""

        val = parse_grade_value(t.replace(",", "."))
        if val is None:
            conn.close()
            return "Me diga a nota (ex: 8,5)."

        # média
        if "média" in tl or "media" in tl:
            sm = re.search(r"\bem\s+([a-zçãõáéíóú ]+?)\s+no\s+\d\s*b\b", tl)
            if not sm:
                sm = re.search(r"\bde\s+([a-zçãõáéíóú ]+?)\s+no\s+\d\s*b\b", tl)
            subject = sm.group(1).strip() if sm else None
            if not subject:
                conn.close()
                return "Me diga a matéria. Ex: \"Média da Giulia em matemática no 1B: 8,2\""
            averages_upsert(conn, student, bim, subject, val)
            conn.close()
            return f"✅ Média salva! {student.title()} — {subject} — {bim}: **{val:.1f}**."

        # nota (assessment)
        ass = parse_assessment(t)
        if not ass:
            conn.close()
            return "Me diga qual avaliação (P1, P2, TRABALHO, PROVA...). Ex: \"P1 de matemática da Giulia do 1B foi 8,5\""
        sm = re.search(r"\bde\s+([a-zçãõáéíóú ]+?)\s+(?:da|do)\b", tl)
        subject = sm.group(1).strip() if sm else None
        if not subject:
            conn.close()
            return "Me diga a matéria. Ex: \"P1 de matemática da Giulia do 1B foi 8,5\""
        grades_upsert(conn, student, bim, subject, ass, val)
        conn.close()
        return f"✅ Nota salva! {student.title()} — {subject} — {bim} — {ass}: **{val:.1f}**."

    # read: notas por bimestre
    bim = parse_bimester(t)
    if bim:
        cur = conn.cursor()
        cur.execute("SELECT subject, assessment, grade FROM school_grades WHERE student=? AND bimester=? ORDER BY subject ASC, assessment ASC",
                    (student, bim))
        rows = cur.fetchall()
        cur.execute("SELECT subject, average FROM school_averages WHERE student=? AND bimester=? ORDER BY subject ASC",
                    (student, bim))
        avs = cur.fetchall()
        conn.close()
        lines = [f"📒 **{student.title()} — Notas {bim}**"]
        if not rows and not avs:
            return f"Não encontrei notas/médias para {student.title()} no {bim}."
        if rows:
            by_subj: Dict[str, List[Tuple[str, float]]] = {}
            for r in rows:
                by_subj.setdefault(r["subject"], []).append((r["assessment"], float(r["grade"])))
            for subj, items in by_subj.items():
                parts = [f"{a}: {g:.1f}" for a, g in items]
                lines.append(f"- **{subj}** → " + " | ".join(parts))
        if avs:
            lines.append("\n📌 **Médias**")
            for r in avs:
                lines.append(f"- **{r['subject']}** → {float(r['average']):.1f}")
        return "\n".join(lines)

    # média específica
    if "média" in tl or "media" in tl:
        bim = parse_bimester(t) or "1B"
        sm = re.search(r"\bem\s+([a-zçãõáéíóú ]+?)\s+no\s+\d\s*b\b", tl)
        subject = sm.group(1).strip() if sm else None
        if subject:
            cur = conn.cursor()
            cur.execute("SELECT average FROM school_averages WHERE student=? AND bimester=? AND subject=?",
                        (student, bim, subject.lower().strip()))
            row = cur.fetchone()
            conn.close()
            if row:
                return f"{student.title()} — média em **{subject}** no **{bim}**: **{float(row['average']):.1f}**."
            return f"Não encontrei a média de {student.title()} em {subject} no {bim}."
        conn.close()
        return "Para consultar média, use: \"Qual a média da Giulia em matemática no 1B?\""

    conn.close()
    return None


# =============================================================================
# Horóscopo (mantido, com cache simples)
# =============================================================================

HOROSCOPE_CACHE: Dict[Tuple[str, str], str] = {}  # (sign, yyyy-mm-dd) -> text


def get_sign_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    signs = [
        "áries", "aries", "touro", "gêmeos", "gemeos", "câncer", "cancer",
        "leão", "leao", "virgem", "libra", "escorpião", "escorpiao",
        "sagitário", "sagitario", "capricórnio", "capricornio", "aquário", "aquario", "peixes",
    ]
    for s in signs:
        if re.search(rf"\b{s}\b", t):
            return s.replace("á", "a").replace("ã", "a").replace("ê", "e").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ô", "o").replace("ú", "u").replace("ç", "c")
    return None


def horoscope_from_aztro(sign: str) -> Optional[str]:
    try:
        r = requests.post("https://aztro.sameerkumar.website/?sign={}&day=today".format(sign), timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        desc = data.get("description")
        return desc.strip() if desc else None
    except Exception:
        return None


def horoscope_fallback_llm(llm: ChatOpenAI, sign: str) -> str:
    prompt = f"Escreva um horóscopo curto (máx. 6 linhas) para hoje para o signo {sign}, em português do Brasil, com tom positivo e realista."
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def handle_horoscope(text: str, client_id: str, llm: ChatOpenAI) -> Optional[str]:
    if not CONFIG.get("horoscope", {}).get("enabled", True):
        return None
    tl = (text or "").lower()
    if "horósc" not in tl and "horosc" not in tl:
        return None

    requester = normalize_client_id(client_id)

    # Se a pessoa foi citada ("da giulia", "do guilherme"), usa o perfil dela.
    target_id = extract_profile_target(text, client_id)

    # 1) tenta pegar do texto (quando o usuário escreve o signo explicitamente)
    sign_norm = get_sign_from_text(text)  # ex: "capricornio"
    sign_en = SIGN_PT_TO_EN.get(sign_norm or "", "")

    display_pt = None

    # 2) se não veio no texto, pega do perfil (automático)
    if not sign_en:
        p = get_profile(target_id)
        sign_en = p.get("sign_en") or ""
        # Para exibição, preferir o signo do perfil (com acento), se existir
        display_pt = (p.get("sign") or "").strip()
        sign_norm = (p.get("sign_norm_pt") or "").strip()

    # Se veio no texto, normaliza para exibição mais bonita
    if not display_pt:
        display_pt = SIGN_NORM_TO_PT.get(sign_norm or "", "") or (sign_norm or "")

    if not sign_en:
        # aqui só acontece se o perfil não tiver signo definido
        who = get_profile(target_id).get("name") or target_id.title()
        return f"Pra eu ver o horóscopo de {who}, me diga o signo."

    # guarda contexto para follow-ups ("e da giovanna?")
    LAST_CTX[requester] = {"domain": "horoscope", "target": target_id}

    key = (sign_en, now_sp().date().isoformat())
    emoji = SIGN_EMOJI.get(sign_norm or "", "")
    header = f"✨ **Horóscopo de hoje — {display_pt}{(' ' + emoji) if emoji else ''}**"

    if key in HOROSCOPE_CACHE:
        return header + "\n" + HOROSCOPE_CACHE[key]

    desc = horoscope_from_aztro(sign_en) or horoscope_fallback_llm(llm, display_pt or sign_norm or sign_en)
    HOROSCOPE_CACHE[key] = desc
    return header + "\n" + desc
# =============================================================================
# Busca na Web (genérica)
# =============================================================================

def looks_like_web_search(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["buscar na web", "pesquisar", "pesquisa:", "buscar:", "resumo com fontes", "web:"])


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Usa DuckDuckGo via HTML-lite (sem dependências extras).
    Se você já usa ddgs no ambiente, pode trocar por DDGS depois.
    """
    results: List[Dict[str, str]] = []
    try:
        # endpoint simples do duckduckgo (html)
        params = {"q": query}
        r = requests.get("https://duckduckgo.com/html/", params=params, timeout=CONFIG["web_search"]["timeout_sec"])
        if r.status_code != 200:
            return results
        html = r.text
        # extrai links principais
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html):
            url = m.group(1)
            title = re.sub(r"<.*?>", "", m.group(2))
            if url and title:
                results.append({"title": title, "url": url})
            if len(results) >= max_results:
                break
    except Exception:
        return results
    return results


def summarize_with_llm(llm: ChatOpenAI, query: str, sources: List[Dict[str, str]]) -> str:
    sources_txt = "\n".join([f"- {s['title']} ({s['url']})" for s in sources[:3]])
    prompt = (
        f"Faça um resumo curto e prático (5 a 8 bullets) sobre: {query}\n\n"
        f"Use apenas como referência estes links (não invente fatos):\n{sources_txt}\n\n"
        f"Escreva em português do Brasil."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    bullets = resp.content.strip()
    out = "🧾 **Resumo**\n" + bullets + "\n\n🔗 **Fontes**\n" + "\n".join([f"- {s['title']}: {s['url']}" for s in sources[:3]])
    return out


def handle_web_search(text: str, llm: ChatOpenAI) -> Optional[str]:
    if not CONFIG.get("web_search", {}).get("enabled", True):
        return None
    if not looks_like_web_search(text):
        return None
    q = re.sub(r"^(buscar na web|pesquisar|pesquisa:|buscar:|resumo com fontes|web:)\s*", "", text.strip(), flags=re.IGNORECASE)
    q = q.strip()
    if not q:
        return "Me diga o que pesquisar. Ex: \"pesquisar: fotossíntese\""
    sources = web_search(q, max_results=CONFIG["web_search"]["max_results"])
    if not sources:
        return "Não consegui achar fontes agora. Tenta reformular a pesquisa."
    return summarize_with_llm(llm, q, sources)


# =============================================================================
# Chat normal (LLM)
# =============================================================================

SYSTEM_PROMPT = """Você é o GOOMI, um amigo de estudos da família.
Prioridades:
- Ajudar com escola: grade, agenda, notas e planos de estudo.
- Explicar assuntos com linguagem simples e exemplos.
- Para crianças, seja gentil, motivador e sem complicar.

Regras:
- Se pedirem comandos, ofereça o manual (manual escola).
- Se pedirem para cadastrar/editar algo e o usuário não for pai/mãe, diga que só os pais podem cadastrar.
- Responda sempre em português (pt-BR), a não ser que o usuário peça outro idioma.
"""


MEMORY: Dict[str, ConversationBufferMemory] = {}


def get_memory(client_id: str) -> ConversationBufferMemory:
    if client_id not in MEMORY:
        MEMORY[client_id] = ConversationBufferMemory(return_messages=True)
    return MEMORY[client_id]


def chat_with_llm(llm: ChatOpenAI, client_id: str, question: str) -> str:
    mem = get_memory(client_id)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(mem.chat_memory.messages[-12:])  # corta histórico
    messages.append(HumanMessage(content=question))
    resp = llm.invoke(messages)
    mem.chat_memory.add_user_message(question)
    mem.chat_memory.add_ai_message(resp.content)
    return resp.content.strip()


# =============================================================================
# Flask app
# =============================================================================

app = Flask(__name__)

# init
ensure_schema()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or CONFIG.get("openai_api_key")
if not OPENAI_API_KEY:
    # não quebra o app; só limita chat/horóscopo/web
    pass

LLM = ChatOpenAI(
    model=CONFIG.get("openai_model", "gpt-4o-mini"),
    temperature=float(CONFIG.get("temperature", 0.3)),
    api_key=OPENAI_API_KEY,
)


def is_greeting(text: str) -> bool:
    t = (text or "").lower().strip()
    return t in {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite"} or t.startswith(("oi ", "olá ", "ola "))


def wants_manual(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["ajuda", "manual", "como usar", "comandos"])


def manual_response(text: str) -> str:
    t = (text or "").lower()
    if "escola" in t or "grade" in t or "agenda" in t or "notas" in t:
        return SCHOOL_MANUAL
    if "hor" in t:
        return HOROSCOPE_MANUAL
    if "web" in t or "buscar" in t or "pesquis" in t:
        return WEB_MANUAL
    # default: geral + escola (pra facilitar)
    return GENERAL_MANUAL + "\n\n---\n\n" + SCHOOL_MANUAL


def profile_response(text: str, client_id: str) -> Optional[str]:
    t = (text or "").strip()
    tl = t.lower()

    triggers = [
        "o que você sabe sobre mim", "o que voce sabe sobre mim",
        "meu perfil", "sobre mim", "me descreve", "quem sou eu", "quem sou",
        "o que você sabe sobre", "o que voce sabe sobre",
        "quem é", "quem e",
    ]
    if not any(k in tl for k in triggers):
        return None

    # tenta achar pessoa mencionada; se não tiver, assume quem está logado
    target = None
    for cid in USER_PROFILES.keys():
        if re.search(rf"\b{re.escape(cid)}\b", tl):
            target = cid
            break

    # também aceita nome (ex: 'Giulia')
    if not target:
        name_map = {get_profile(k)["name"].lower(): k for k in USER_PROFILES.keys()}
        for nm, cid in name_map.items():
            if re.search(rf"\b{re.escape(nm)}\b", tl):
                target = cid
                break

    target = target or normalize_client_id(client_id)
    p = get_profile(target)

    # resposta rica (sempre usando perfil completo)
    return profile_markdown(p)


def handle_sign_question(text: str, client_id: str) -> Optional[str]:
    """Answer questions like 'qual o signo da giulia?' using profiles."""
    tl = (text or "").lower()
    if "signo" not in tl:
        return None
    # evita conflitar com horóscopo
    if "horósc" in tl or "horosc" in tl:
        return None

    target_id = extract_profile_target(text, client_id)
    p = get_profile(target_id)
    sign = p.get("sign") or ""
    who = p.get("name") or target_id.title()

    # Se o perfil já tem signo, responde direto
    if sign:
        # se for a própria pessoa (meu signo)
        if target_id == normalize_client_id(client_id) and any(k in tl for k in ["meu signo", "qual meu signo", "qual é meu signo", "qual e meu signo"]):
            return f"Seu signo é **{sign}**."
        return f"{who} é do signo de **{sign}**."

    # fallback: se tiver data de nascimento no perfil, tenta calcular (opcional)
    birth = p.get("birthdate") or ""
    if birth:
        # não implementamos cálculo completo aqui; pede confirmação
        return f"Eu tenho a data de nascimento de {who}, mas não tenho o signo registrado. Quer que eu calcule e salve no perfil?"

    return f"Pra eu saber o signo de {who}, me diga a data de nascimento (dia/mês/ano)."


@app.get("/health")
def health():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as n FROM familia")
        n = cur.fetchone()["n"]
        conn.close()
        return jsonify({
            "ok": True,
            "time_sp": now_sp().isoformat(),
            "db": {"path": CONFIG["db_path"], "familia_count": n},
            "features": {
                "school_timetable": True,
                "school_agenda": True,
                "school_grades": True,
                "horoscope": bool(CONFIG.get("horoscope", {}).get("enabled", True)),
                "web_search": bool(CONFIG.get("web_search", {}).get("enabled", True)),
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/version")
def version():
    return jsonify({
        "app_version": APP_VERSION,
        "time_sp": now_sp().isoformat(),
        "db_path": CONFIG["db_path"],
    })



@app.get("/debug/routes")
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({"rule": str(rule), "methods": sorted(list(rule.methods))})
    return jsonify({"routes": routes})


@app.post("/ask")
def ask():
    data = get_payload(request)
    client_id = normalize_client_id(data.get("client_id", "glauco"))
    question = (data.get("question") or data.get("message") or data.get("text") or "").strip()
    # Make follow-up questions ("e na quinta?") work naturally
    question = apply_followup_context(client_id, question)

    if not question:
        return jsonify({"answer": "Me manda uma pergunta 😊"})

    # manual
    if wants_manual(question):
        return jsonify({"answer": manual_response(question)})

    # greeting
    if is_greeting(question):
        name = get_profile(client_id).get("name", client_id.title())
        return jsonify({"answer": f"Oi, {name}! 😊 Como posso ajudar nos estudos hoje?"})

    # profile (quando perguntarem)
    pr = profile_response(question, client_id)
    if pr:
        return jsonify({"answer": pr})
    # sign (qual o signo do(a)...)
    sg = handle_sign_question(question, client_id)
    if sg:
        return jsonify({"answer": sg})


    # horoscope
    h = handle_horoscope(question, client_id, LLM)
    if h:
        return jsonify({"answer": h})

    # web search
    ws = handle_web_search(question, LLM)
    if ws:
        return jsonify({"answer": ws})

    # timetable
    tt = handle_timetable(question, client_id)
    if tt:
        return jsonify({"answer": tt})

    # agenda
    ag = handle_agenda(question, client_id)
    if ag:
        return jsonify({"answer": ag})

    # grades
    gr = handle_grades(question, client_id)
    if gr:
        return jsonify({"answer": gr})

    # default: chat normal
    ans = chat_with_llm(LLM, client_id, question)
    return jsonify({"answer": ans})


if __name__ == "__main__":
    # modo dev
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=bool(CONFIG.get("debug", True)))
