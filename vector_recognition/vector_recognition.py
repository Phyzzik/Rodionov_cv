import cv2
import numpy as np
import os

# путь
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'alphabet (1).png')

# загрузка с русскими символами ( если есть в пути)
img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

# Переводим в оттенки серого и делаем изображение бинарным (ч/б)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Находим все контуры (символы)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

symbols_features = []

# Функция вычисления вектора признаков для одного символа
def get_features(contour):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    
    # 3 параметра для нахождения векторов
    # 1) Отношение ширины к высоте
    aspect_ratio = w / float(h)
    # 2) Плотность (заполненность прямоугольника)
    extent = area / float(w * h)
    # 3) Площадь
    
    return [aspect_ratio, extent, area], (x, y, w, h)

# Собираем векторы признаков для всех символов
for cnt in contours:
    if cv2.contourArea(cnt) > 5:  # Игнорируем мелкий шум
        features, bbox = get_features(cnt)
        symbols_features.append({
            'vector': features,
            'bbox': bbox
        })

print(f" Успешно! Найдено и обработано символов: {len(symbols_features)}")

if len(symbols_features) >= 2:
    v1 = symbols_features[0]['vector']
    v2 = symbols_features[1]['vector']
    
    # Евклидово расстояние между векторами признаков
    dist = np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)
    print(f"Вектор 1-го символа: {v1}")
    print(f"Вектор 2-го символа: {v2}")
    print(f"Расстояние между ними: {dist:.4f}")

for item in symbols_features:
    x, y, w, h = item['bbox']
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Сохраняем результат 
output_path = os.path.join(script_dir, 'result.png')
is_success, buffer = cv2.imencode(".png", img)
if is_success:
    buffer.tofile(output_path)
    print(f" Результат с рамками сохранен в файл: {output_path}")