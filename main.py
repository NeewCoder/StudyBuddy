import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def test_ocr(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), lang="bul+eng")
        print("Разпознат текст:")
        print(text)
    except Exception as e:
        print(f"Грешка: {e}")

if __name__ == "__main__":
    # Примерно извикване
    # test_ocr("test.jpg")
    pass