import os
import uuid
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from ai_logic import extract_text_from_image, simplify_with_chatgpt, quiz_with_chatgpt
from db_logic import (
    init_db,
    save_analysis,
    list_analyses,
    list_analyses_for_user,
    get_analysis,
    get_analysis_for_user,
    save_quiz,
    list_quizzes_for_analysis,
    get_user,
    get_user_by_id,
    create_user,
    delete_analysis
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "studybuddy-demo-secret")

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
    if not user_id:
        return None
    return get_user_by_id(user_id)


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
            flash("Трябва да си админ.", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrapper


def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def make_unique_filename(filename):
    safe_name = secure_filename(filename)
    if "." in safe_name:
        ext = safe_name.rsplit(".", 1)[1].lower()
    else:
        ext = "jpg"
    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}.{ext}"


@app.route("/")
def index():
    session.pop("image_filename", None)
    session.pop("ocr_text", None)
    session.pop("summary", None)
    session.pop("simple_text", None)
    session.pop("analysis_id", None)

    return render_template(
        "index.html",
        image_filename=None,
        summary=None,
        simple_text=None,
        ocr_text=None,
        analysis_id=None,
        is_admin=is_admin(),
        user=current_user()
    )


@app.route("/analysis")
def show_analysis():
    return render_template(
        "index.html",
        image_filename=session.get("image_filename"),
        summary=session.get("summary"),
        simple_text=session.get("simple_text"),
        ocr_text=session.get("ocr_text"),
        analysis_id=session.get("analysis_id"),
        is_admin=is_admin(),
        user=current_user()
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if len(username) < 3:
            flash("Потребителското име трябва да е поне 3 символа.", "warning")
            return redirect(url_for("signup"))

        if len(password) < 3:
            flash("Паролата трябва да е поне 3 символа.", "warning")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Паролите не съвпадат.", "danger")
            return redirect(url_for("signup"))

        existing = get_user(username)
        if existing:
            flash("Това потребителско име вече съществува.", "warning")
            return redirect(url_for("signup"))

        new_user_id = create_user(username, password, role="user")
        if not new_user_id:
            flash("Неуспешна регистрация.", "danger")
            return redirect(url_for("signup"))

        session["user_id"] = new_user_id
        session["username"] = username
        session["role"] = "user"

        flash("Регистрацията е успешна. Добре дошъл!", "info")
        return redirect(url_for("index"))

    return render_template("signup.html", user=current_user(), is_admin=is_admin())


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = get_user(username)

        if not user or user["password"] != password:
            flash("Грешно потребителско име или парола.", "danger")
            return redirect(url_for("login"))

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        flash("Успешен вход.", "info")
        return redirect(url_for("index"))

    return render_template("login.html", user=current_user(), is_admin=is_admin())


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("role", None)
    session.pop("admin_name", None)
    session.pop("image_filename", None)
    session.pop("ocr_text", None)
    session.pop("summary", None)
    session.pop("simple_text", None)
    session.pop("analysis_id", None)

    flash("Излезе успешно от профила си.", "info")
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    try:
        file = request.files.get("image")

        if not file or file.filename == "":
            flash("Моля, избери файл.", "warning")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Позволени са само изображения.", "warning")
            return redirect(url_for("index"))

        filename = make_unique_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        ocr_text = extract_text_from_image(save_path, lang="bul+eng")
        summary, simple_text = simplify_with_chatgpt(ocr_text)

        user_id = session.get("user_id")
        analysis_id = save_analysis(user_id, filename, ocr_text, summary, simple_text)

        session["image_filename"] = filename
        session["ocr_text"] = ocr_text
        session["summary"] = summary
        session["simple_text"] = simple_text
        session["analysis_id"] = analysis_id

        flash(f"Анализът е записан успешно. ID: {analysis_id}", "info")
        return redirect(url_for("show_analysis"))

    except Exception as e:
        flash(f"Грешка при анализ: {e}", "danger")
        return redirect(url_for("index"))


@app.route("/generate_quiz", methods=["POST"])
@login_required
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


@app.route("/history")
@login_required
def history():
    user_id = session.get("user_id")
    items = list_analyses_for_user(user_id, limit=10)
    return render_template(
        "history.html",
        items=items,
        is_admin=is_admin(),
        user=current_user()
    )


@app.route("/analysis/<int:analysis_id>")
@login_required
def analysis_detail(analysis_id):
    user_id = session.get("user_id")

    if is_admin():
        item = get_analysis(analysis_id)
    else:
        item = get_analysis_for_user(analysis_id, user_id)

    if not item:
        flash("Анализът не е намерен или нямаш достъп до него.", "warning")
        return redirect(url_for("history"))

    quizzes = list_quizzes_for_analysis(analysis_id)

    return render_template(
        "analysis.html",
        item=item,
        quizzes=quizzes,
        is_admin=is_admin(),
        user=current_user()
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        user = get_user(u)

        if user and user["password"] == p and user["role"] == "admin":
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["admin_name"] = u
            flash("Влезе като админ.", "info")
            return redirect(url_for("admin_dashboard"))

        flash("Грешни данни.", "danger")

    return render_template("admin_login.html", user=current_user(), is_admin=is_admin())


@app.route("/admin/logout")
def admin_logout():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("role", None)
    session.pop("admin_name", None)

    flash("Излезе от админ.", "info")
    return redirect(url_for("index"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    items = list_analyses(limit=500)
    return render_template(
        "dashboard.html",
        items=items,
        admin=session.get("admin_name"),
        user=current_user(),
        is_admin=is_admin()
    )


@app.post("/admin/delete/<int:analysis_id>")
@admin_required
def admin_delete_analysis(analysis_id):
    delete_analysis(analysis_id)
    flash(f"Анализ #{analysis_id} е изтрит.", "info")
    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    app.run(debug=True)