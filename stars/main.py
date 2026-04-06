import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

file = np.load(r'C:\Users\bqgk6\OneDrive\Документы\Компьютерное зрение\7_звезды\stars\stars.npy')
binary_mask = file > 0

label_image = label(binary_mask)
regions = regionprops(label_image)

plus_count = 0
cross_count = 0

for prop in regions:
    angle = np.degrees(prop.orientation)
    if abs(angle) < 20 or abs(angle) > 70:
        plus_count += 1
    else:
        cross_count += 1

print(f"Плюсов {plus_count}")
print(f"Крестов {cross_count}")
print(f"Всего {len(regions)}")

plt.figure(figsize= (10, 10))
plt.imshow(file)
plt.axis('off')
plt.show()