# cria_db.py — GOOMI (Família G)
# Cria o banco goomi.db com as tabelas: familia e notas
import sqlite3
from datetime import datetime

DB_PATH = "goomi.db"

def create_tables(conn):
    cur = conn.cursor()

    # --- Tabela de integrantes da família ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS familia (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        papel TEXT NOT NULL,         -- pai, mae, filha, enteado
        idade INTEGER,
        apelido TEXT,
        client_id TEXT NOT NULL UNIQUE,
        criado_em TEXT
    );
    """)

    # --- Tabela de notas escolares ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS notas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aluno_id INTEGER NOT NULL,
        disciplina TEXT NOT NULL,
        bimestre INTEGER,
        descricao TEXT,
        nota REAL,
        data_avaliacao TEXT,
        criado_em TEXT,
        FOREIGN KEY (aluno_id) REFERENCES familia(id)
    );
    """)

    # Índices para consultas rápidas
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notas_aluno ON notas(aluno_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notas_disc ON notas(disciplina);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notas_bimestre ON notas(bimestre);")

    conn.commit()

def seed_family(conn):
    """Insere integrantes da Família G com client_id fixo (id de login)."""
    now = datetime.now().isoformat(timespec="seconds")
    dados = [
        ("Glauco",   "pai",     None,  None,   "glauco",    now),
        ("Helena",   "mae",     None,  None,   "helena",    now),
        ("Giulia",   "filha",   10,     "Giu",  "giulia",    now),
        ("Giovanna", "filha",   17,    "Gio",  "giovanna",  now),
        ("Guilherme","enteado", 12,    "Gui",  "guilherme", now),
    ]
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR IGNORE INTO familia (nome, papel, idade, apelido, client_id, criado_em)
        VALUES (?, ?, ?, ?, ?, ?);
    """, dados)
    conn.commit()

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        create_tables(conn)
        seed_family(conn)
        print(f"Banco criado/atualizado: {DB_PATH}")
    finally:
        conn.close()
