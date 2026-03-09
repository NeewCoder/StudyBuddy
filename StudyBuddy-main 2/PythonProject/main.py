import pytesseract
from PIL import Image

# 👉 тук сложи пътя, където е инсталиран Tesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\teseract\tesseract.exe"

# 👉 тестова снимка (сложи една снимка в същата папка и напиши името ѝ)
image_path = "mazna.jpg"

try:
    text = pytesseract.image_to_string(Image.open(image_path), lang="bul+eng")

    print("✅ Разпознат текст:")
    print("---------------------------")
    print(text)
    print("---------------------------")

except Exception as e:
    print("⚠️ Грешка при разчитането на снимката:")
    print(e)
