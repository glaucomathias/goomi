import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "goomi.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print(f"Banco: {DB_PATH}")

# mostra as colunas da tabela
print("\n--- Estrutura da tabela consultas_exames ---")
for row in cur.execute("PRAGMA table_info(consultas_exames);"):
    print(row)

# mostra os registros
print("\n--- Registros em consultas_exames ---")
for row in cur.execute("SELECT * FROM consultas_exames;"):
    print(row)

conn.close()
