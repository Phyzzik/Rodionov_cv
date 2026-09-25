from collections import defaultdict
import os
import cv2
import numpy as np

# 1. Загрузка файла
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "balls_and_rects.png")

image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Файл не найден по пути: {image_path}")

# 2. Функция для перевода значения Hue в понятное название цвета
def get_color_name(hue_val):
    if (0 <= hue_val < 10) or (170 <= hue_val <= 179):
        return "Красный"
    elif 10 <= hue_val < 25:
        return "Оранжевый"
    elif 25 <= hue_val < 35:
        return "Желтый"
    elif 35 <= hue_val < 85:
        return "Зеленый"
    elif 85 <= hue_val < 125:
        return "Синий / Голубой"
    elif 125 <= hue_val < 150:
        return "Фиолетовый"
    elif 150 <= hue_val < 170:
        return "Розовый"
    return "Неопределенный"

# 3. Подготовка и бинаризация
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# 4. Поиск контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

total_shapes = 0
total_rects = 0
total_circles = 0

shades_total = defaultdict(int)
shades_rects = defaultdict(int)
shades_circles = defaultdict(int)

# 5. Анализ фигур
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 5:
        continue

    total_shapes += 1
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    # Круглость
    circularity = 4 * np.pi * area / (perimeter**2)

    # Определение цвета
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_val = cv2.mean(hsv, mask=mask)
    raw_hue = mean_val[0]

    # Получаем понятное имя цвета
    color_name = get_color_name(raw_hue)

    shades_total[color_name] += 1

    if circularity > 0.78:
        total_circles += 1
        shades_circles[color_name] += 1
    else:
        total_rects += 1
        shades_rects[color_name] += 1

# 6. Компактный вывод
print("=" * 60)
print(f"ОБЩЕЕ КОЛИЧЕСТВО ФИГУР: {total_shapes}")
print(f"Прямоугольников: {total_rects} | Кругов: {total_circles}")
print("=" * 60)
print("ДЕТАЛИЗАЦИЯ ПО ЦВЕТАМ:")

all_colors = sorted(shades_total.keys())
for color in all_colors:
    tot = shades_total[color]
    rects = shades_rects[color]
    circs = shades_circles[color]
    print(f"Цвет: {color:<15} | Всего: {tot:<3} | Прямоугольников: {rects:<3} | Кругов: {circs:<3}")

print("=" * 60)