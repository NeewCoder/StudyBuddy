import sqlite3
import json
from datetime import datetime
import os

DB_PATH = "studybuddy.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(table_name: str, column_name: str) -> bool:
    conn = _connect()
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    rows = c.fetchall()
    conn.close()
    return any(r["name"] == column_name for r in rows)


def init_db():
    conn = _connect()
    c = conn.cursor()

    # users
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # analyses
    c.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_name TEXT,
        ocr_text TEXT,
        summary TEXT,
        simplified TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    # quizzes
    c.execute("""
    CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL,
        quiz_json TEXT NOT NULL,
        created_at TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

    _migrate_old_tables()
    ensure_admin_user()


def _migrate_old_tables():
    conn = _connect()
    c = conn.cursor()

    # ако analyses е стара таблица без user_id -> добавяме колоната
    c.execute("PRAGMA table_info(analyses)")
    columns = [row["name"] for row in c.fetchall()]

    if "user_id" not in columns:
        c.execute("ALTER TABLE analyses ADD COLUMN user_id INTEGER")
        conn.commit()

    conn.close()


def ensure_admin_user():
    u = os.environ.get("ADMIN_USERNAME", "admin")
    p = os.environ.get("ADMIN_PASSWORD", "admin123")

    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (u,))
    row = c.fetchone()

    if not row:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (u, p, "admin")
        )
        conn.commit()

    conn.close()


# ---------- USERS ----------

def create_user(username: str, password: str, role: str = "user"):
    conn = _connect()
    c = conn.cursor()

    try:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, password, role)
        )
        conn.commit()
        new_id = c.lastrowid
        conn.close()
        return new_id
    except sqlite3.IntegrityError:
        conn.close()
        return None


def get_user(username: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- ANALYSES ----------

def save_analysis(user_id: int, image_name: str, ocr_text: str, summary: str, simplified: str) -> int:
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        INSERT INTO analyses (user_id, image_name, ocr_text, summary, simplified, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, image_name, ocr_text, summary, simplified, datetime.utcnow().isoformat()))

    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def list_analyses(limit: int = 200):
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        SELECT id,
               user_id,
               image_name,
               substr(ocr_text, 1, 120) AS ocr_preview,
               substr(summary, 1, 200) AS summary_preview,
               created_at
        FROM analyses
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def list_analyses_for_user(user_id: int, limit: int = 200):
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        SELECT id,
               user_id,
               image_name,
               substr(ocr_text, 1, 120) AS ocr_preview,
               substr(summary, 1, 200) AS summary_preview,
               created_at
        FROM analyses
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))

    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_analysis(analysis_id: int):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_analysis_for_user(analysis_id: int, user_id: int):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id)
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_analysis(analysis_id: int):
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM quizzes WHERE analysis_id = ?", (analysis_id,))
    c.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()


# ---------- QUIZZES ----------

def save_quiz(analysis_id: int, quiz_list) -> int:
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        INSERT INTO quizzes (analysis_id, quiz_json, created_at)
        VALUES (?, ?, ?)
    """, (analysis_id, json.dumps(quiz_list, ensure_ascii=False), datetime.utcnow().isoformat()))

    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def list_quizzes_for_analysis(analysis_id: int):
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        SELECT id, quiz_json, created_at
        FROM quizzes
        WHERE analysis_id = ?
        ORDER BY id DESC
    """, (analysis_id,))

    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows