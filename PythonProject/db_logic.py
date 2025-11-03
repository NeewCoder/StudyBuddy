# db_logic.py
import sqlite3
import json
from datetime import datetime
import os

DB_PATH = "studybuddy.db"

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _connect()
    c = conn.cursor()

    # analyses
    c.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        ocr_text TEXT,
        summary TEXT,
        simplified TEXT,
        created_at TEXT
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

    # users (само за админ)
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,           -- за демо: plain (в реален проект -> хеш)
        role TEXT                -- 'admin' или 'user'
    )
    """)

    conn.commit()
    conn.close()
    ensure_admin_user()

def ensure_admin_user():
    """Създава админ от .env, ако липсва."""
    import os
    u = os.environ.get("ADMIN_USERNAME", "admin")
    p = os.environ.get("ADMIN_PASSWORD", "admin123")

    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (u,))
    row = c.fetchone()
    if not row:
        c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)", (u, p, "admin"))
        conn.commit()
    conn.close()

# --- Анализи ---
def save_analysis(image_name: str, ocr_text: str, summary: str, simplified: str) -> int:
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses (image_name, ocr_text, summary, simplified, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (image_name, ocr_text, summary, simplified, datetime.utcnow().isoformat()))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id

def list_analyses(limit: int = 200):
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT id, image_name, substr(ocr_text,1,120) AS ocr_preview,
               substr(summary,1,200) AS summary_preview,
               created_at
        FROM analyses
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
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

def delete_analysis(analysis_id: int):
    """Каскадно трие тестовете за този анализ, след това самия анализ."""
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM quizzes WHERE analysis_id = ?", (analysis_id,))
    c.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()

# --- Тестове ---
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

# --- Потребители (за админ) ---
def get_user(username: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None
