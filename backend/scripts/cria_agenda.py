
# cria_agenda.py — cria/atualiza tabelas de agenda e motivação
import sqlite3
from datetime import datetime
import os

DB_PATH = os.environ.get("GOOMI_DB", "goomi.db")

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS motivacao_diaria(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id TEXT NOT NULL,
        data_ref TEXT NOT NULL,
        mensagem TEXT NOT NULL,
        criado_em TEXT,
        UNIQUE(client_id, data_ref)
    );""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS agenda_atividades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dono TEXT NOT NULL,
        titulo TEXT NOT NULL,
        categoria TEXT,
        data_inicio TEXT,
        hora TEXT,
        rrule TEXT,
        dias_semana TEXT,
        materias TEXT,
        meta TEXT,
        criado_em TEXT
    );""")

    conn.commit()
    conn.close()
    print("[OK] Tabelas motivacao_diaria e agenda_atividades criadas/atualizadas.")

if __name__ == "__main__":
    main()
