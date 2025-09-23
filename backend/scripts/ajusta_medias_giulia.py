# ajusta_medias_giulia.py — atualiza as médias oficiais da Giulia (sobrescreve valores)
import sqlite3
from datetime import datetime

DB_PATH = "goomi.db"

GIULIA_MEDIAS = {
    ("Português", "P2"): 9.7,
    ("Matemática", "P2"): 9.6,
    ("História",   "P2"): 9.2,
    ("Geografia",  "P2"): 9.0,
    ("Ciências",   "P2"): 10.0,
    ("Inglês",     "P2"): 10.0,
}

def get_aluno_id(conn, client_id: str):
    cur = conn.cursor()
    cur.execute("SELECT id FROM familia WHERE client_id=?", (client_id,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"client_id '{client_id}' não encontrado na tabela familia.")
    return int(row[0])

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    aluno_id = get_aluno_id(conn, "giulia")
    agora = datetime.now().isoformat(timespec="seconds")

    for (disciplina, periodo), media_val in GIULIA_MEDIAS.items():
        cur.execute("""
            INSERT INTO medias (aluno_id, disciplina, periodo, media, atualizado_em)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(aluno_id, disciplina, periodo) DO UPDATE SET
              media = excluded.media,
              atualizado_em = excluded.atualizado_em
        """, (aluno_id, disciplina, periodo, media_val, agora))
        print(f"[OK] {disciplina} {periodo} => média oficial {media_val}")

    conn.commit()
    conn.close()
    print("[OK] Todas as médias da Giulia foram atualizadas.")

if __name__ == "__main__":
    main()
