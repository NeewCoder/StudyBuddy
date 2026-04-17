import os
import re
import json
from dotenv import load_dotenv
import requests
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

load_dotenv()

# Път до Tesseract - провери дали е същият на новия компютър!
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# API Конфигурация
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _clean_ocr_text(text: str) -> str:
    """Изчиства текста от излишни символи и нов ред при пренасяне."""
    text = re.sub(r"([А-Яа-я])-\n([А-Яа-я])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_image(image_path: str, lang: str = "bul+eng") -> str:
    """Извлича текст от изображение с предварителна обработка за по-добър OCR."""
    if not os.path.exists(image_path):
        return ""
    try:
        img = Image.open(image_path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        raw_text = pytesseract.image_to_string(img, lang=lang)
        return _clean_ocr_text(raw_text)
    except Exception as e:
        print(f"OCR ГРЕШКА: {e}")
        return ""


def _chat(messages, max_tokens=1000):
    """Помощна функция за комуникация с OpenAI API."""
    if not OPENAI_API_KEY or "sk-proj" not in OPENAI_API_KEY:
        print("ГРЕШКА: Невалиден API ключ!")
        return None

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY.strip()}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )

        if r.status_code != 200:
            print(f"!!! OpenAI API Error {r.status_code}: {r.text}")
            return None

        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"!!! КРИТИЧНА ГРЕШКА ПРИ ВРЪЗКА: {e}")
        return None


def simplify_with_chatgpt(text: str):
    """Генерира резюме на текста в JSON формат."""
    if not text.strip() or len(text) < 10:
        return ("", "Недостатъчен текст за анализ.")

    sys_prompt = "Ти си асистент за обучение. Връщай САМО JSON обект."
    usr_prompt = f"Направи резюме в 5-7 точки на български. Използвай тирета (-). Текст:\n{text}\n\nВърни ТОЧНО този формат: {{\"simple\": \"- точка 1\\n- точка 2\"}}"

    response = _chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}])

    if response:
        try:
            # Търсим JSON структурата в отговора, за да избегнем NoneType грешки
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                clean_json = match.group(0)
                data = json.loads(clean_json)
                return ("", data.get("simple", "Няма данни в JSON."))
            else:
                # Ако няма JSON скоби, връщаме целия отговор като текст
                return ("", response.strip())
        except Exception as e:
            print(f"ГРЕШКА ПРИ ПАРСВАНЕ НА JSON: {e}")
            return ("", response)

    return ("", "Грешка при връзка с AI.")


def quiz_with_chatgpt(text: str, num_questions=5):
    """Генерира тест с въпроси въз основа на текста."""
    if not text.strip():
        return []

    sys_prompt = "Ти си учител. Връщай САМО JSON масив."
    usr_prompt = f"Създай {num_questions} въпроса върху този текст:\n{text}\n\nВърни САМО JSON масив: [{{ \"q\": \"Въпрос?\", \"options\": [\"1\", \"2\", \"3\", \"4\"], \"answer\": 0 }}]"

    response = _chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}])

    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except Exception as e:
            print(f"ГРЕШКА ПРИ ПАРСВАНЕ НА ТЕСТ: {e}")
            return []
    return []