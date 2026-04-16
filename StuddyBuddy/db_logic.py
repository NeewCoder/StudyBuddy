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
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )""")
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
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL,
        quiz_json TEXT NOT NULL,
        created_at TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
    )""")
    conn.commit()
    conn.close()
    ensure_admin_user()

def ensure_admin_user():
    u, p = os.environ.get("ADMIN_USERNAME", "admin"), os.environ.get("ADMIN_PASSWORD", "admin123")
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (u,))
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (u, p, "admin"))
        conn.commit()
    conn.close()

def create_user(username, password, role="user"):
    conn = _connect()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        res = c.lastrowid
    except: res = None
    conn.close()
    return res

def get_user(username):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def save_analysis(user_id, image_name, ocr_text, summary, simplified):
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses (user_id, image_name, ocr_text, summary, simplified, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, image_name, ocr_text, summary, simplified, datetime.utcnow().isoformat()))
    conn.commit()
    res = c.lastrowid
    conn.close()
    return res

# ТАЗИ ФУНКЦИЯ ЛИПСВАШЕ:
def list_analyses(limit=200):
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT id, user_id, image_name, substr(ocr_text, 1, 120) AS ocr_preview,
               substr(summary, 1, 200) AS summary_preview, created_at
        FROM analyses ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def list_analyses_for_user(user_id, limit=200):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_analysis(analysis_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_analysis_for_user(analysis_id, user_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_analysis(analysis_id, user_id=None):
    conn = _connect()
    c = conn.cursor()
    if user_id:
        c.execute("SELECT id FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))
        if not c.fetchone():
            conn.close()
            return False
    c.execute("DELETE FROM quizzes WHERE analysis_id = ?", (analysis_id,))
    if user_id:
        c.execute("DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))
    else:
        c.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()
    return True

def save_quiz(analysis_id, quiz_list):
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT INTO quizzes (analysis_id, quiz_json, created_at) VALUES (?, ?, ?)",
              (analysis_id, json.dumps(quiz_list, ensure_ascii=False), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def list_quizzes_for_analysis(analysis_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM quizzes WHERE analysis_id = ? ORDER BY id DESC", (analysis_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows