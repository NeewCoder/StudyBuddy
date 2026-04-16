import os
import re
import json
from dotenv import load_dotenv
import requests
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def _clean_ocr_text(text: str) -> str:
    text = re.sub(r"([А-Яа-я])-\n([А-Яа-я])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_image(image_path: str, lang: str = "bul+eng") -> str:
    if not os.path.exists(image_path):
        return ""
    try:
        img = Image.open(image_path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        raw_text = pytesseract.image_to_string(img, lang=lang)
        return _clean_ocr_text(raw_text)
    except Exception as e:
        return f"Грешка: {e}"

def _chat(messages, max_tokens=1000):
    if not OPENAI_API_KEY:
        return None
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(OPENAI_URL,
                          headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                          json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except:
        return None

def simplify_with_chatgpt(text: str):
    if not text.strip() or len(text) < 10:
        return ("", "Недостатъчен текст за анализ.")

    sys_prompt = "Ти си асистент за обучение. Извличаш основните точки от текст на български и ги подреждаш в списък."
    usr_prompt = f"Направи резюме в 5-7 точки на следния текст. ВАЖНО: Започвай всяка точка на НОВ РЕД с тире (-). Текст:\n{text}\n\nВърни JSON: {{\"simple\": \"- Първа точка\\n- Втора точка\"}}"

    response = _chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}])

    if response:
        try:
            clean_json = re.search(r'\{.*\}', response, re.DOTALL).group(0)
            data = json.loads(clean_json)
            return ("", data.get("simple", "Няма данни."))
        except:
            pass
    return ("", "Грешка при връзка с AI.")

def quiz_with_chatgpt(text: str, num_questions=5):
    if not text.strip():
        return []

    sys_prompt = "Ти си учител. Генерираш тестове с 4 отговора въз основа на предоставения текст."
    usr_prompt = f"Създай {num_questions} въпроса върху този текст:\n{text}\n\nВърни JSON масив: [{{ \"q\": \"Въпрос?\", \"options\": [\"отг1\", \"отг2\", \"отг3\", \"отг4\"], \"answer\": 0 }}]"

    response = _chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}])

    if response:
        try:
            clean_json = re.search(r'\[.*\]', response, re.DOTALL).group(0)
            return json.loads(clean_json)
        except:
            return []
    return []