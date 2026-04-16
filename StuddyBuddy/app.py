import os
import uuid
import re
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ai_logic import extract_text_from_image, simplify_with_chatgpt, quiz_with_chatgpt
from db_logic import (
    init_db, save_analysis, list_analyses, list_analyses_for_user,
    get_analysis, get_analysis_for_user, save_quiz,
    list_quizzes_for_analysis, get_user, get_user_by_id, create_user, delete_analysis
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "studybuddy-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

init_db()


def is_admin():
    return session.get("role") == "admin"


def is_logged_in():
    return session.get("user_id") is not None


def current_user():
    user_id = session.get("user_id")
    return get_user_by_id(user_id) if user_id else None


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            flash("Трябва първо да влезеш в профила си.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not is_admin():
            flash("Нямате права за тази страница.", "warning")
            return redirect(url_for("index"))
        return f(*args, **kwargs)

    return wrapper


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_unique_filename(filename):
    safe_name = secure_filename(filename)
    ext = safe_name.rsplit(".", 1)[1].lower() if "." in safe_name else "jpg"
    return f"{uuid.uuid4().hex[:12]}.{ext}"


@app.route("/")
def index():
    session.pop("image_filename", None)
    session.pop("ocr_text", None)
    session.pop("summary", None)
    session.pop("simple_text", None)
    session.pop("analysis_id", None)
    return render_template("index.html", is_admin=is_admin(), user=current_user())


@app.route("/analysis")
def show_analysis():
    return render_template("index.html",
                           image_filename=session.get("image_filename"),
                           summary=session.get("summary"),
                           simple_text=session.get("simple_text"),
                           ocr_text=session.get("ocr_text"),
                           analysis_id=session.get("analysis_id"),
                           is_admin=is_admin(),
                           user=current_user())


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()

        if len(username) < 3 or len(password) < 8:
            flash("Невалидно име или парола.", "warning")
            return redirect(url_for("signup"))

        if not (re.search(r"[A-Z]", password) and re.search(r"\d", password) and re.search(r"[@$!%*?&]", password)):
            flash("Паролата трябва да съдържа главна буква, цифра и специален знак.", "danger")
            return redirect(url_for("signup"))

        if password != confirm:
            flash("Паролите не съвпадат.", "danger")
            return redirect(url_for("signup"))

        if get_user(username):
            flash("Потребителското име е заето.", "warning")
            return redirect(url_for("signup"))

        user_id = create_user(username, password)
        session.update({"user_id": user_id, "username": username, "role": "user"})
        return redirect(url_for("index"))
    return render_template("signup.html", user=current_user(), is_admin=is_admin())


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = get_user(username)
        if user and user["password"] == password:
            session.update({"user_id": user["id"], "username": user["username"], "role": user["role"]})
            return redirect(url_for("index"))
        flash("Грешни данни.", "danger")
    return render_template("login.html", user=current_user(), is_admin=is_admin())


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    file = request.files.get("image")
    if file and allowed_file(file.filename):
        filename = make_unique_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        ocr = extract_text_from_image(path)
        summ, simple = simplify_with_chatgpt(ocr)
        a_id = save_analysis(session["user_id"], filename, ocr, summ, simple)
        session.update(
            {"image_filename": filename, "ocr_text": ocr, "summary": summ, "simple_text": simple, "analysis_id": a_id})
        return redirect(url_for("show_analysis"))
    flash("Моля, изберете валидна снимка.", "warning")
    return redirect(url_for("index"))


@app.route("/delete/<int:analysis_id>", methods=["POST"])
@login_required
def delete_user_analysis(analysis_id):
    if delete_analysis(analysis_id, user_id=session["user_id"]):
        flash("Записът е изтрит.", "info")
    return redirect(url_for("history"))


@app.route("/generate_quiz", methods=["POST"])
@login_required
def generate_quiz():
    text = session.get("simple_text") or session.get("ocr_text") or ""
    if not text.strip(): return jsonify({"ok": False}), 400
    quiz = quiz_with_chatgpt(text)
    if session.get("analysis_id"): save_quiz(session["analysis_id"], quiz)
    return jsonify({"ok": True, "quiz": quiz})


@app.route("/history")
@login_required
def history():
    items = list_analyses_for_user(session["user_id"])
    return render_template("history.html", items=items, is_admin=is_admin(), user=current_user())


@app.route("/uploads/<path:fname>")
def uploaded_file(fname):
    return send_from_directory(UPLOAD_FOLDER, fname)


if __name__ == "__main__":
    app.run(debug=True)