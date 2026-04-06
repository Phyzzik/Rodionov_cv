import numpy as np
import matplotlib.pyplot as plt
import glob

path = 'C:/Users/bqgk6/OneDrive/Документы/Компьютерное зрение/4_траектория/out/*.npy'
files = sorted(glob.glob(path)) 
y_coords = []
x_coords = []

for file in files:
    img = np.load(file)
    indices = np.argwhere(img > 0)
    
    if indices.size > 0:
        y_center, x_center = indices.mean(axis=0)
        x_coords.append(x_center)
        y_coords.append(y_center)

plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, marker='o', markersize=2, linestyle='-', color='blue')

plt.xlim(0, 600)
plt.ylim(600, 0) 
plt.title("Траектория движения объекта")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()