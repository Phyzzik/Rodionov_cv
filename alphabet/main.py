from collections import Counter
import os
import cv2
import numpy as np

# Список символов в порядке их расположения на alphabet_ext.png
ALPHABET_CHARS = ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/", "P", "D"]


def imread_unicode(file_path):
    """Считывает изображение с поддержкой кириллицы в пути (Windows)."""
    try:
        # Читаем файл как массив байтов через numpy, затем декодируем через OpenCV
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def get_alphabet_templates(alphabet_image_path):
    img = imread_unicode(alphabet_image_path)

    if img is None:
        print(f"Ошибка: не удалось прочитать файл {alphabet_image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[0])

    templates = {}

    for idx, (x, y, w, h) in enumerate(boxes):
        if idx >= len(ALPHABET_CHARS):
            break

        roi = thresh[y : y + h, x : x + w]
        resized_roi = cv2.resize(roi, (20, 20))
        char_name = ALPHABET_CHARS[idx]
        templates[char_name] = resized_roi

    return templates


def recognize_symbol(roi, templates):
    resized = cv2.resize(roi, (20, 20))

    best_match = None
    min_difference = float("inf")

    for char_name, template in templates.items():
        diff = np.sum(cv2.absdiff(resized, template))
        if diff < min_difference:
            min_difference = diff
            best_match = char_name

    return best_match


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    alphabet_path = os.path.join(script_dir, "alphabet_ext.png")
    symbols_path = os.path.join(script_dir, "symbols.png")

    templates = get_alphabet_templates(alphabet_path)
    if templates is None:
        return

    
    img_symbols = imread_unicode(symbols_path)
    if img_symbols is None:
        print(f"Ошибка: не удалось прочитать файл {symbols_path}")
        return

    gray_symbols = cv2.cvtColor(img_symbols, cv2.COLOR_BGR2GRAY)
    _, thresh_symbols = cv2.threshold(gray_symbols, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh_symbols, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    found_symbols = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 3 or h < 3:
            continue

        roi = thresh_symbols[y : y + h, x : x + w]
        char_name = recognize_symbol(roi, templates)
        found_symbols.append(char_name)

    dict_counts = Counter(found_symbols)

    print("=== Частотный словарь символов ===")
    for char in ALPHABET_CHARS:
        count = dict_counts.get(char, 0)
        print(f"'{char}': {count}")


if __name__ == "__main__":
    main()