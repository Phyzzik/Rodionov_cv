import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure
from scipy.spatial import distance

folder_path = r'C:\Users\bqgk6\OneDrive\Документы\Компьютерное зрение\4_траектория\out'
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

trajectories = [[], [], []]
last_points = None

for file_name in file_names:
    img = np.load(os.path.join(folder_path, file_name))
    
    labels = measure.label(img > 0)
    regions = measure.regionprops(labels)
    
    current_points = [np.array(r.centroid) for r in regions] # центры
    
    if last_points is None:
        current_points.sort(key=lambda x: x[0])
        last_points = current_points
    else:
        new_order = []
        dists = distance.cdist(last_points, current_points)
        
        for row in dists:
            idx = np.argmin(row)
            new_order.append(current_points[idx])
        
        current_points = new_order
        last_points = current_points

    for i in range(len(current_points)):
        if i < 3: 
            trajectories[i].append([current_points[i][1], current_points[i][0]])

plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
labels = ['Траектория 1', 'Траектория 2', 'Траектория 3']

for i in range(3):
    traj = np.array(trajectories[i])
    plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, linestyle='-', color=colors[i], label=labels[i])

plt.xlabel("X")
plt.ylabel("Y") 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
