import os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ai_logic import extract_text_from_image, simplify_with_chatgpt, quiz_with_chatgpt
from db_logic import (
    init_db, save_analysis, list_analyses, get_analysis, save_quiz, list_quizzes_for_analysis,
    get_user, delete_analysis
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "studybuddy-demo-secret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# init DB
init_db()

# --------- helpers (auth) ----------
def is_admin():
    return session.get("role") == "admin"

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_admin():
            flash("Трябва да си админ.", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrapper

# --------- public ----------
@app.route("/")
def index():
    return render_template(
        "index.html",
        image_filename=session.get("image_filename"),
        summary=session.get("summary"),
        simple_text=session.get("simple_text"),
        ocr_text=session.get("ocr_text"),
        analysis_id=session.get("analysis_id"),
        is_admin=is_admin()
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Моля, избери файл.", "warning")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        ocr_text = extract_text_from_image(save_path, lang="bul+eng")
        summary, simple_text = simplify_with_chatgpt(ocr_text)

        analysis_id = save_analysis(filename, ocr_text, summary, simple_text)

        session["image_filename"] = filename
        session["ocr_text"] = ocr_text
        session["summary"] = summary
        session["simple_text"] = simple_text
        session["analysis_id"] = analysis_id

        flash(f"Анализът е записан (ID: {analysis_id}).", "info")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Грешка при анализ: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    source_text = session.get("simple_text") or session.get("ocr_text") or ""
    analysis_id = session.get("analysis_id")
    if not source_text.strip():
        return jsonify({"ok": False, "error": "Няма текст за тест."}), 400
    try:
        quiz = quiz_with_chatgpt(source_text, num_questions=5)
        if analysis_id:
            save_quiz(analysis_id, quiz)
        return jsonify({"ok": True, "quiz": quiz})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Грешка: {e}"}), 500

@app.route("/uploads/<path:fname>")
def uploaded_file(fname):
    return send_from_directory(UPLOAD_FOLDER, fname)

# --------- history / detail ----------
@app.route("/history")
def history():
    items = list_analyses(limit=200)
    return render_template("history.html", items=items, is_admin=is_admin())

@app.route("/analysis/<int:analysis_id>")
def analysis_detail(analysis_id):
    item = get_analysis(analysis_id)
    if not item:
        flash("Анализът не е намерен.", "warning")
        return redirect(url_for("history"))
    quizzes = list_quizzes_for_analysis(analysis_id)
    return render_template("analysis.html", item=item, quizzes=quizzes, is_admin=is_admin())

# --------- admin ----------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        user = get_user(u)
        if user and user["password"] == p and user["role"] == "admin":
            session["role"] = "admin"
            session["admin_name"] = u
            flash("Влезе като админ.", "info")
            return redirect(url_for("admin_dashboard"))
        flash("Грешни данни.", "danger")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("role", None)
    session.pop("admin_name", None)
    flash("Излезе от админ.", "info")
    return redirect(url_for("index"))

@app.route("/admin")
@admin_required
def admin_dashboard():
    items = list_analyses(limit=500)
    return render_template("admin_dashboard.html", items=items, admin=session.get("admin_name"))

@app.post("/admin/delete/<int:analysis_id>")
@admin_required
def admin_delete_analysis(analysis_id):
    delete_analysis(analysis_id)
    flash(f"Анализ #{analysis_id} е изтрит.", "info")
    return redirect(url_for("admin_dashboard"))

if __name__ == "__main__":
    app.run(debug=True)
