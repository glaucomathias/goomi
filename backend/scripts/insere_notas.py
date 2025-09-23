# insere_notas.py — insere algumas notas de teste
import sqlite3
from datetime import datetime

DB_PATH = "goomi.db"

def inserir_nota(client_id, disciplina, bimestre, descricao, nota, data_avaliacao):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # pega o id do aluno a partir do client_id (ex: "giulia")
    cur.execute("SELECT id FROM familia WHERE client_id = ?", (client_id,))
    row = cur.fetchone()
    if not row:
        print(f"[ERRO] client_id não encontrado: {client_id}")
        conn.close()
        return

    aluno_id = row["id"]
    agora = datetime.now().isoformat(timespec="seconds")

    cur.execute("""
        INSERT INTO notas (aluno_id, disciplina, bimestre, descricao, nota, data_avaliacao, criado_em)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (aluno_id, disciplina, bimestre, descricao, nota, data_avaliacao, agora))
    conn.commit()
    conn.close()
    print(f"[OK] Nota inserida para {client_id}: {disciplina} ({descricao}) = {nota}")

if __name__ == "__main__":
    inserir_nota("giulia",    "Matemática", 1, "Prova 1", 9.5, "2025-03-15T10:00:00")
    inserir_nota("giulia",    "Ciências",   1, "Trabalho", 8.8, "2025-03-20T09:00:00")
    inserir_nota("guilherme", "Português",  1, "Prova 1", 7.2, "2025-03-18T08:30:00")
    inserir_nota("giovanna",  "História",   1, "Prova 1", 9.0, "2025-03-17T11:00:00")
