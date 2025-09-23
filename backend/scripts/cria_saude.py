# cria_saude.py — cria tabelas de saúde para Helena dentro do goomi.db
import sqlite3
from datetime import datetime

DB_PATH = "goomi.db"

def create_tables(conn):
    cur = conn.cursor()

    # --- Tabela de consultas/exames/procedimentos ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS consultas_exames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aluno_id INTEGER NOT NULL,
        tipo TEXT NOT NULL,                -- consulta, exame, procedimento
        especialidade TEXT,                -- ex: dermatologia, cardiologia
        descricao TEXT,
        data_realizacao TEXT,
        data_retorno TEXT,
        criado_em TEXT,
        FOREIGN KEY (aluno_id) REFERENCES familia(id)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_consultas_aluno ON consultas_exames(aluno_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consultas_tipo ON consultas_exames(tipo);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consultas_retorno ON consultas_exames(data_retorno);")

    conn.commit()

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        create_tables(conn)
        print(f"[OK] Tabelas de saúde criadas/atualizadas no banco {DB_PATH}")
    finally:
        conn.close()
