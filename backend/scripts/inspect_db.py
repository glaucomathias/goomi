# backend/scripts/inspect_goomi_dbdb.py
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "goomi_backup.db"
print("Inspecionando:", DB)

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]
print("Tabelas:", tables)

for t in tables:
    try:
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"- {t}: {n} registros")
    except Exception as e:
        print(f"- {t}: erro -> {e}")

if "familia" in tables:
    rows = cur.execute("SELECT id, nome, client_id FROM familia").fetchall()
    print("\nConteúdo da tabela 'familia':")
    for r in rows:
        print(" ", r)

conn.close()
