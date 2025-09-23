
# limpa_saude_dia.py — limpeza retroativa do “Dia” colado
import sqlite3
import os

DB_PATH = os.environ.get("GOOMI_DB", "goomi.db")

def clean_trailing_dia(t):
    if not t:
        return t
    s = str(t).strip()
    for marker in (" dia", " Dia"):
        if s.endswith(marker):
            core = s[:-len(marker)]
            if not core.endswith(("Dia","dia","Diaz","diaz")):
                return core.strip()
    return s

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, especialidade, descricao FROM consultas_exames")
    rows = cur.fetchall()
    upd = 0
    for r in rows:
        e2 = clean_trailing_dia(r["especialidade"])
        d2 = clean_trailing_dia(r["descricao"])
        if e2 != r["especialidade"] or d2 != r["descricao"]:
            cur.execute("UPDATE consultas_exames SET especialidade=?, descricao=? WHERE id=?",
                        (e2, d2, r["id"]))
            upd += 1
    conn.commit()
    conn.close()
    print(f"[OK] Registros atualizados: {upd}")

if __name__ == "__main__":
    main()
