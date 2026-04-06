import numpy as np
import os
from pathlib import Path
from scipy.ndimage import label, binary_erosion

TARGET_DIR = r"C:\Users\bqgk6\OneDrive\Документы\Компьютерное зрение\5_Провода\wires\wires 1-5"

def analyze_wires(file_path):
    try:
        data = np.load(file_path)
        
        struct = np.ones((3, 1))
        eroded_data = binary_erosion(data, structure=struct)
        
        labeled_array, num_features = label(eroded_data)
        
        wire_groups = {}
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_array == i)
            avg_y = int(np.mean(coords[:, 0]))
            
            found = False
            for key in wire_groups.keys():
                if abs(key - avg_y) <= 2: 
                    wire_groups[key].append(i)
                    found = True
                    break
            if not found:
                wire_groups[avg_y] = [i]

        print(f"Файл: {os.path.basename(file_path)}")
        print(f"  Всего проводов: {len(wire_groups)}")
        
        for idx, y_coord in enumerate(sorted(wire_groups.keys()), 1):
            parts_count = len(wire_groups[y_coord])
            status = "целый" if parts_count == 1 else f"порван на {parts_count} ч."
            print(f"    Провод {idx}: {status}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Ошибка в файле {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    folder = Path(TARGET_DIR)

    if folder.exists() and folder.is_dir():
        npy_files = sorted(list(folder.glob("*.npy")))
        
        if not npy_files:
            print(f"В папке {TARGET_DIR} не найдено .npy файлов.")
        else:
            for f in npy_files:
                analyze_wires(f)
    else:
        print(f"Система не видит путь: {TARGET_DIR}")
