# ai_logic.py
import os
import re
import json
import time
import requests
from PIL import Image
import pytesseract

# 👉 Път до твоя Tesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\teseract\tesseract.exe"

# Опитай OpenCV (ако го имаш ще махнем вертикални линии/ръбове)
try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:
    HAS_CV = False

# ---------------- OCR ПРЕДОБРАБОТКА ----------------

def _preprocess_cv(image_path: str) -> Image.Image | None:
    """
    Предобработка с OpenCV:
    - изправя контраст
    - маха тънки вертикални линии (източник на '|')
    - дава чисто изображение за Tesseract
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Грейскейл + лека филтрация
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    # Изостряне (леко)
    sharp = cv2.addWeighted(gray, 1.2, cv2.GaussianBlur(gray, (0, 0), 3), -0.2, 0)

    # Бинаризация (Otsu)
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- Махане на вертикални линии ----
    # kernel за вертикали (~ 40 пиксела височина)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vert = cv2.morphologyEx(255 - bw, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    # маска на вертикали: праг
    _, mask = cv2.threshold(detect_vert, 200, 255, cv2.THRESH_BINARY)
    # "избелване" на вертикалите
    cleaned = bw.copy()
    cleaned[mask == 255] = 255

    # Ако текстът е дребен — увеличи
    h, w = cleaned.shape[:2]
    if max(w, h) < 1600:
        cleaned = cv2.resize(cleaned, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Към PIL image
    return Image.fromarray(cleaned)

def _preprocess_pillow(image_path: str) -> Image.Image | None:
    try:
        img = Image.open(image_path).convert("L")
        w, h = img.size
        if max(w, h) < 1600:
            img = img.resize((w * 2, h * 2))
        return img
    except Exception:
        return None

# ---------------- ПОЧИСТВАНЕ СЛЕД OCR ----------------

# латиница → кирилица (чести OCR грешки)
LAT_TO_CYR = {
    "A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У","W":"Ш",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у","m":"м","w":"ш",
    "b":"ъ","u":"и","q":"я","g":"д"
}

def _fix_mixed_words(text: str) -> str:
    def repl(m): return "".join(LAT_TO_CYR.get(ch, ch) for ch in m.group(0))
    return re.sub(r"[A-Za-zА-Яа-я]+", repl, text)

def _normalize_bulgarian_punctuation(text: str) -> str:
    text = text.replace("\r", "").replace("-\n", "")
    # залепи счупени думи с '|' в средата: "За| кланият" -> "Закланият"
    text = re.sub(r"(\w)\|\s*(\w)", r"\1\2", text)
    # махни всички самотни вертикални черти
    text = text.replace("|", " ")
    # пряка реч
    text = re.sub(r"(?m)^\s*-\s+", "— ", text)
    text = text.replace("–", "—")
    text = re.sub(r"\s*—\s*", " — ", text)
    # двойни двоеточия
    text = re.sub(r":\s*:", ":", text)
    # интервали преди/след пунктуация
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", text)
    # раздвоени кавички/шлаши/бекслеши
    text = re.sub(r"[\\\/`´‘’“”\"́]+", "", text)
    # мн. интервали и празни редове
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _final_touchups(text: str) -> str:
    # „с ъМ / с ЪМ / с ъм“ -> „съм“
    text = re.sub(r"\b[Сс]\s*[ЪъЬ]\s*[Мм]\b", "съм", text)
    text = re.sub(r"\bАз\s+съм\b", "Аз съм", text)

    # премахни мини латиница шум между думи (1–3 букви)
    text = re.sub(r"(?<=\s)[a-zA-Z]{1,3}(?=\s)", " ", text)

    # премахни самотни номера на страници (ред с 1–3 цифри)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

    # още веднъж нормализация на интервали
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ---------------- OCR PIPE ----------------

def extract_text_from_image(image_path: str, lang: str = "bul") -> str:
    if not os.path.exists(image_path):
        return ""
    try:
        # 1) предобработка
        if HAS_CV:
            img = _preprocess_cv(image_path)
        else:
            img = _preprocess_pillow(image_path)
        if img is None:
            img = Image.open(image_path)

        # 2) OCR
        raw = pytesseract.image_to_string(img, lang=lang, config=r"--oem 3 --psm 6 --dpi 300")

        # 3) почистване
        text = _normalize_bulgarian_punctuation(raw)
        text = _fix_mixed_words(text)
        text = _normalize_bulgarian_punctuation(text)
        text = _final_touchups(text)

        return text.strip()
    except Exception as e:
        return f"[OCR грешка] {e}"

# ---------------- OpenAI (с retries) ----------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def _chat(messages, max_tokens=800, retries=3):
    if not OPENAI_API_KEY:
        return None
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    backoff = 2.0
    for _ in range(retries):
        try:
            r = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError:
            break
        except requests.exceptions.RequestException:
            time.sleep(backoff); backoff *= 2
    return None

# ---------------- FALLBACK-и ----------------

def _first_sentences(text: str, n=3):
    parts = [p.strip() for p in re.split(r"[.!?]\s+", text) if p.strip()]
    res = ". ".join(parts[:n])
    if res and not res.endswith("."): res += "."
    return res or "Няма извлечен текст."

def _simplify_fallback(text: str):
    if not text.strip(): return ""
    t = re.sub(r"\([^)]*\)", "", text)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    out = []
    for s in sents:
        if len(s) > 220:
            out.append(s[:200].rsplit(" ", 1)[0] + "…")
        else:
            out.append(s)
    return "\n".join(out)

def _quiz_fallback(text: str, n: int = 5):
    keywords = re.findall(r"[А-Яа-яA-Za-z]{5,}", text)[:n] or ["текст", "урок", "идея", "пример", "понятие"]
    return [{
        "q": f"{i}. Какво е най-близко до значението на „{kw}“ в контекста?",
        "options": ["Определение", "Пример", "Причина", "Не е свързано"],
        "answer": 0
    } for i, kw in enumerate(keywords, 1)]

# ---------------- ПУБЛИЧНИ AI ФУНКЦИИ ----------------

def simplify_with_chatgpt(text: str):
    if not text.strip():
        return ("Няма извлечен текст.", "")

    system = "Ти си учител, който преразказва сложно съдържание на лесен български."
    user = f"""
Текст (на български). Направи:
1) кратко обобщение (1-3 изречения);
2) „лесен за учене“ вариант: преформулирай, раздели на кратки изречения, подреди логично.

Отговори САМО като валиден JSON с ключове:
{{
  "summary": "...",
  "simple": "..."
}}

Текст:
\"\"\"{text}\"\"\"
"""
    out = _chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_tokens=700
    )
    if out is None:
        return (_first_sentences(text, 3), _simplify_fallback(text))

    try:
        j = json.loads(out)
        return ((j.get("summary") or _first_sentences(text, 3)).strip(),
                (j.get("simple") or _simplify_fallback(text)).strip())
    except Exception:
        return (_first_sentences(text, 3), out.strip() or _simplify_fallback(text))

def quiz_with_chatgpt(text: str, num_questions: int = 5):
    system = "Ти си учител. Правиш логични тестове по даден учебен текст на български."
    user = f"""
От следния текст създай {num_questions} смислени въпроса с четири опции (A–D), като само една е вярна.
Покрий важни понятия/връзки/следствия. Дай само валиден JSON:
[
  {{"q":"въпрос","options":["A","B","C","D"],"answer":0}}
]
Текст:
\"\"\"{text}\"\"\"
"""
    out = _chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_tokens=900
    )
    if out is None:
        return _quiz_fallback(text, num_questions)

    try:
        data = json.loads(out)
        cleaned = []
        for it in data:
            q = str(it.get("q", "")).strip()
            options = it.get("options", []) or []
            ans = int(it.get("answer", 0))
            if q and isinstance(options, list) and len(options) == 4 and 0 <= ans <= 3:
                cleaned.append({"q": q, "options": [str(x) for x in options][:4], "answer": ans})
        return cleaned[:num_questions] or _quiz_fallback(text, num_questions)
    except Exception:
        return _quiz_fallback(text, num_questions)
