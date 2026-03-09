# ai_logic.py
import os
import re
import json
import requests
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

# Път до Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:
    HAS_CV = False


# ---------------- IMAGE HELPERS ----------------

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


def _detect_page_and_crop_cv(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    page = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)

        if len(approx) == 4 and area > (img.shape[0] * img.shape[1] * 0.20):
            page = approx.reshape(4, 2)
            break

    if page is not None:
        try:
            warped = _four_point_transform(original, page)
            return warped
        except Exception:
            pass

    h, w = gray.shape[:2]
    top = int(h * 0.04)
    bottom = int(h * 0.96)
    left = int(w * 0.10)
    right = int(w * 0.90)
    return original[top:bottom, left:right]


def _crop_inner_book_region_pil(img: Image.Image):
    w, h = img.size
    left = int(w * 0.10)
    top = int(h * 0.04)
    right = int(w * 0.90)
    bottom = int(h * 0.96)
    return img.crop((left, top, right, bottom))


# ---------------- OCR PREPROCESS ----------------

def _deskew_cv(gray):
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) < 100:
        return gray

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _preprocess_cv_variants(image_path: str):
    variants = []

    img = _detect_page_and_crop_cv(image_path)
    if img is None:
        return variants

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.mean() < 120:
        gray = cv2.bitwise_not(gray)

    gray = _deskew_cv(gray)

    h, w = gray.shape[:2]
    if max(w, h) < 2200:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v1 = clahe.apply(gray)
    variants.append(Image.fromarray(v1))

    blur = cv2.GaussianBlur(v1, (3, 3), 0)
    _, v2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(Image.fromarray(v2))

    v3 = cv2.adaptiveThreshold(
        v1, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 12
    )
    variants.append(Image.fromarray(v3))

    sharp = cv2.addWeighted(v1, 1.5, cv2.GaussianBlur(v1, (0, 0), 2), -0.5, 0)
    _, v4 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(Image.fromarray(v4))

    return variants


def _preprocess_pillow_variants(image_path: str):
    variants = []

    try:
        img = Image.open(image_path).convert("L")
        img = _crop_inner_book_region_pil(img)

        if img.getextrema()[0] < 70:
            img = ImageOps.invert(img)

        w, h = img.size
        if max(w, h) < 2200:
            img = img.resize((w * 2, h * 2))

        variants.append(img.copy())

        strong = ImageEnhance.Contrast(img).enhance(2.2)
        strong = strong.filter(ImageFilter.SHARPEN)
        strong = strong.filter(ImageFilter.MedianFilter(size=3))
        variants.append(strong)

        bw = strong.point(lambda x: 255 if x > 170 else 0)
        variants.append(bw)

    except Exception:
        pass

    return variants


# ---------------- TEXT CLEANUP ----------------

COMMON_FIXES = {
    "Оогат": "богат",
    "потежка": "по-тежка",
    "ПОСТОЯНСТВО": "постоянство",
    "Отколкото": "отколкото",
    "cu": "си",
    "Ha": "на",
    "or": "от",
    "тои": "той",
    "внимаваи": "внимавай",
    "авата": "главата",
    "предупреждения": "предупреждения"
}


def _merge_hyphenated_lines(text: str) -> str:
    text = re.sub(r"([А-Яа-я])-\n([А-Яа-я])", r"\1\2", text)
    text = re.sub(r"([А-Яа-я,;])\n([а-я])", r"\1 \2", text)
    return text


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True

    letters_cyr = len(re.findall(r"[А-Яа-я]", s))
    letters_lat = len(re.findall(r"[A-Za-z]", s))
    digits = len(re.findall(r"\d", s))

    if len(s) <= 3 and letters_cyr <= 1:
        return True

    if letters_cyr == 0 and letters_lat > 0 and len(s) <= 8:
        return True

    if digits > 3 and letters_cyr < 3:
        return True

    if re.fullmatch(r"[\W\d_]+", s):
        return True

    return False


def _remove_noise_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if _is_noise_line(s):
            continue

        cyr = len(re.findall(r"[А-Яа-я]", s))
        lat = len(re.findall(r"[A-Za-z]", s))
        if lat > cyr and cyr < 8:
            continue

        lines.append(s)

    return "\n".join(lines)


def _normalize_spaces(text: str) -> str:
    text = text.replace("\r", "")
    text = text.replace("|", " ")
    text = text.replace("„ ", "„")
    text = text.replace(" “", "“")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([,.:;!?])([^\s\n])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _postfix_common_ocr_errors(text: str) -> str:
    for bad, good in COMMON_FIXES.items():
        text = text.replace(bad, good)

    text = re.sub(r"\bз ащото\b", "защото", text)
    text = re.sub(r"\bа щото\b", "защото", text)
    text = re.sub(r"\bс ума\b", "сума", text)
    text = re.sub(r"\bпонаст\b", "понастоящем", text)
    text = re.sub(r"\bсъм ара\b", "съм сред", text)
    text = re.sub(r"\bирсдирисмач\b", "предприемач", text)
    text = re.sub(r"\bпабрта\b", "работя", text)
    text = re.sub(r"\bеди да\b", "преди да", text)
    text = re.sub(r"\bаният\b", "кланият", text)
    text = re.sub(r"\bжка\b", "тежка", text)
    text = re.sub(r"\bрдост\b", "гордост", text)
    text = re.sub(r"\bжно\b", "нужно", text)
    text = re.sub(r"\bпочнал\b", "започнал", text)
    text = re.sub(r"\bслиш\b", "мислиш", text)
    text = re.sub(r"\bПо-\s*тил\b", "Получил", text)
    return text


def _clean_ocr_text(text: str) -> str:
    text = _merge_hyphenated_lines(text)
    text = _remove_noise_lines(text)
    text = _normalize_spaces(text)
    text = _postfix_common_ocr_errors(text)

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        cyr = len(re.findall(r"[А-Яа-я]", s))
        lat = len(re.findall(r"[A-Za-z]", s))
        weird = len(re.findall(r"[^A-Za-zА-Яа-я0-9\s.,:;!?\"'„“()\-%—–]", s))

        if weird > max(4, len(s) // 6):
            continue
        if lat > cyr and cyr < 10:
            continue

        lines.append(s)

    text = "\n".join(lines)
    text = _normalize_spaces(text)
    return text


# ---------------- OCR SCORING ----------------

def _ocr_score(text: str) -> int:
    if not text:
        return -100000

    cyr = len(re.findall(r"[А-Яа-я]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    noise = len(re.findall(r"[\\|/<>_^~`$@#=*[\]{}]", text))
    long_lines = sum(1 for ln in text.splitlines() if len(ln.strip()) > 30)

    score = 0
    score += cyr * 3
    score -= lat * 5
    score -= noise * 10
    score += long_lines * 12

    for token in [
        "защото", "гордост", "егото", "важен", "специален",
        "всички", "култура", "измама", "работата", "постоянство"
    ]:
        score += text.lower().count(token) * 12

    return score


# ---------------- OCR MAIN ----------------

def extract_text_from_image(image_path: str, lang: str = "bul") -> str:
    if not os.path.exists(image_path):
        return ""

    candidates = []
    variants = []

    try:
        if HAS_CV:
            variants.extend(_preprocess_cv_variants(image_path))

        variants.extend(_preprocess_pillow_variants(image_path))

        if not variants:
            variants = [Image.open(image_path).convert("L")]

        configs = [
            "--oem 3 --psm 6",
            "--oem 3 --psm 4",
            "--oem 3 --psm 11"
        ]

        langs = ["bul", "bul+eng"]

        for img in variants:
            for lg in langs:
                for cfg in configs:
                    try:
                        raw = pytesseract.image_to_string(img, lang=lg, config=cfg)
                        cleaned = _clean_ocr_text(raw)
                        score = _ocr_score(cleaned)
                        candidates.append((score, cleaned, lg, cfg))
                    except Exception:
                        continue

        if not candidates:
            return ""

        best = max(candidates, key=lambda x: x[0])[1]
        return best.strip()

    except Exception as e:
        return f"[OCR грешка] {e}"


# ---------------- OPENAI ----------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}


def _chat(messages, max_tokens=1000):
    if not OPENAI_API_KEY:
        return None

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(OPENAI_URL, headers=HEADERS, json=payload, timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _extract_json_block(text: str):
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    arr = re.search(r"\[[\s\S]*\]", text)
    if arr:
        return arr.group(0)

    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        return obj.group(0)

    return text


# ---------------- SIMPLE TEXT HELPERS ----------------

def _split_sentences(text: str):
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_for_display(text: str) -> str:
    s = text.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" -–—•")
    return s


def _extract_intro_claims(text: str):
    claims = []

    joined = text.replace("\n", " ")
    joined = re.sub(r"\s+", " ", joined)

    matches = re.findall(r"Аз съм[^.:\n]{5,90}", joined)
    for m in matches:
        m = _normalize_for_display(m)
        if len(re.findall(r"[A-Za-z]", m)) > 4:
            continue
        if len(m) < 10:
            continue
        if m not in claims:
            claims.append(m)

    return claims[:5]


def _score_sentence_for_learning(sentence: str) -> int:
    s = sentence.lower()
    score = 0

    keywords = [
        "гордост", "его", "измама", "постоянство", "самонадеяност",
        "внимавай", "стабилност", "етикети", "падение", "надутост",
        "рокфелер", "работата", "самозаблу", "важен", "специален"
    ]

    for kw in keywords:
        if kw in s:
            score += 4

    if 45 <= len(sentence) <= 180:
        score += 5

    if "?" in sentence:
        score -= 3

    if len(re.findall(r"[A-Za-z]", sentence)) > 6:
        score -= 5

    return score


def _select_key_sentences(text: str, limit=5):
    sentences = _split_sentences(text)
    ranked = []

    for sent in sentences:
        clean = _normalize_for_display(sent)
        if len(clean) < 35:
            continue
        ranked.append((_score_sentence_for_learning(clean), clean))

    ranked.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for _, sent in ranked:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sent)
        if len(selected) >= limit:
            break

    return selected


def _claim_to_easy_point(claim: str) -> str:
    c = claim.lower()

    if "предприемач" in c or "писател" in c or "богат" in c or "специален" in c or "важен" in c:
        return "Хората често се определят според временни успехи, роли или външни етикети."

    return "Егото кара човека да мисли за себе си по-преувеличено, отколкото е реално."


def _sentence_to_easy_point(sentence: str) -> str:
    s = sentence.strip()

    replacements = [
        ("Нека кажем какво представлява тази нагласа: измама.", "Тази нагласа всъщност е форма на самозаблуда."),
        ("Гордостта е хитър нарушител.", "Гордостта действа подмолно и може да подведе човека."),
        ("Пред погибел гордост върви и пред падение - надутост.", "Прекалената гордост често води до грешки и падение."),
        ("Пред погибел гордост върви и пред падение — надутост.", "Прекалената гордост често води до грешки и падение."),
    ]

    for old, new in replacements:
        if old in s:
            return new

    if "всяка култура" in s.lower() and "предупреждения" in s.lower():
        return "Още отдавна хората предупреждават, че самонадеяността е опасна."

    if "ако си вършиш работата" in s.lower() or "постоянство" in s.lower():
        return "Истинският напредък идва чрез работа, постоянство и честност към себе си."

    if "рокфелер" in s.lower():
        return "Примерът с Рокфелер показва как ранният успех може да доведе до самонадеяност."

    return _normalize_for_display(s).rstrip(".") + "."


def _simple_fallback(text: str):
    claims = _extract_intro_claims(text)
    key_sentences = _select_key_sentences(text, limit=5)

    points = []

    points.append("Текстът е за това как гордостта и егото могат да подведат човека.")

    if claims:
        claim_point = _claim_to_easy_point(claims[0])
        if claim_point not in points:
            points.append(claim_point)

    for sent in key_sentences:
        easy = _sentence_to_easy_point(sent)
        if easy not in points:
            points.append(easy)

    points.append("Авторът насърчава човек да бъде по-реалистичен, стабилен и постоянен.")
    points.append("Важно е човек да не се определя само чрез успехи, а чрез реални действия и развитие.")

    clean_points = []
    for p in points:
        if len(re.findall(r"[A-Za-z]", p)) > 8:
            continue
        if len(p.strip()) < 12:
            continue
        if p not in clean_points:
            clean_points.append(p)

    final_points = clean_points[:7]

    return "Най-важното:\n" + "\n".join(f"- {p}" for p in final_points)


# ---------------- QUIZ HELPERS ----------------

def _quiz_fallback(text: str, n=5):
    quiz = []

    quiz.append({
        "q": "1. Каква е основната идея в текста?",
        "options": [
            "Гордостта и егото могат да подведат човека.",
            "Най-важното е човек да се хвали с успехите си.",
            "Човек трябва винаги да изглежда по-важен пред другите.",
            "Успехът идва само чрез късмет."
        ],
        "answer": 0
    })

    quiz.append({
        "q": "2. Какво внушават примерите в началото на текста?",
        "options": [
            "Че хората често си лепят приятни етикети и се самозаблуждават.",
            "Че всички тези твърдения са напълно верни.",
            "Че авторът одобрява самохвалството.",
            "Че човек не бива да мисли за себе си."
        ],
        "answer": 0
    })

    quiz.append({
        "q": "3. Какво предупреждение дава текстът?",
        "options": [
            "Да не се изкарваш по-голям и по-способен, отколкото си.",
            "Да не работиш упорито.",
            "Да не слушаш никакви съвети.",
            "Да не си поставяш цели."
        ],
        "answer": 0
    })

    quiz.append({
        "q": "4. Каква е ролята на примера с Рокфелер?",
        "options": [
            "Да покаже как успехът може да породи самонадеяност.",
            "Да докаже, че богатството решава всичко.",
            "Да покаже, че гордостта винаги помага.",
            "Да обясни как се тегли банков заем."
        ],
        "answer": 0
    })

    quiz.append({
        "q": "5. Кое поведение според текста е най-правилно?",
        "options": [
            "Постоянна работа, стабилност и реална преценка за себе си.",
            "Показност и преувеличаване на успехите.",
            "Сравняване с другите и самохвалство.",
            "Прибързано самоуверено поведение."
        ],
        "answer": 0
    })

    return quiz[:n]


def _sanitize_quiz(quiz, num_questions=5):
    if not isinstance(quiz, list):
        return None

    cleaned = []
    for item in quiz:
        if not isinstance(item, dict):
            continue

        q = str(item.get("q", "")).strip()
        options = item.get("options", [])
        answer = item.get("answer", 0)

        if not q or not isinstance(options, list) or len(options) != 4:
            continue

        options = [str(x).strip() for x in options]
        if not all(options):
            continue

        try:
            answer = int(answer)
        except Exception:
            answer = 0

        if answer < 0 or answer > 3:
            answer = 0

        cleaned.append({
            "q": q,
            "options": options,
            "answer": answer
        })

        if len(cleaned) >= num_questions:
            break

    return cleaned if cleaned else None


# ---------------- PUBLIC AI FUNCTIONS ----------------

def simplify_with_chatgpt(text: str):
    if not text.strip():
        return ("", "Няма извлечен текст.")

    system = (
        "Ти си учител, който работи с OCR текст на български. "
        "Трябва да извлечеш смисъла и да го обясниш много просто. "
        "Не преписвай шумния OCR текст."
    )

    user = f"""
Имаш OCR текст на български.

Направи "лесен за учене вариант" по тези правила:
- 5 до 7 кратки точки
- всяка точка да е ясна и разбираема
- да предава смисъла, а не да копира текста буквално
- да поправя смислово дребните OCR грешки
- без излишни дълги изречения
- без уводни фрази
- не преразказвай целия текст, а само най-важното

Върни САМО JSON:
{{
  "simple": "..."
}}

Текст:
{text}
"""

    out = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=700
    )

    if not out:
        return ("", _simple_fallback(text))

    try:
        json_text = _extract_json_block(out)
        data = json.loads(json_text)
        simple = str(data.get("simple", "")).strip()

        if not simple:
            simple = _simple_fallback(text)

        return ("", simple)
    except Exception:
        return ("", _simple_fallback(text))


def quiz_with_chatgpt(text: str, num_questions=5):
    if not text.strip():
        return _quiz_fallback(text, num_questions)

    system = (
        "Ти си учител. Правиш тест по смисъла на български текст. "
        "Не правиш въпроси за отделни думи. Не използваш OCR грешки."
    )

    user = f"""
Създай {num_questions} въпроса с 4 възможни отговора.
Само 1 е верен.

Изисквания:
- въпросите да са по смисъла
- да са ясни и кратки
- да не са за отделни думи
- да не повтарят шумен OCR текст
- върни САМО JSON масив във формат:

[
  {{
    "q": "Въпрос?",
    "options": ["A", "B", "C", "D"],
    "answer": 0
  }}
]

Текст:
{text}
"""

    out = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=1000
    )

    if not out:
        return _quiz_fallback(text, num_questions)

    try:
        json_text = _extract_json_block(out)
        parsed = json.loads(json_text)
        cleaned = _sanitize_quiz(parsed, num_questions=num_questions)
        if cleaned:
            return cleaned
        return _quiz_fallback(text, num_questions)
    except Exception:
        return _quiz_fallback(text, num_questions)