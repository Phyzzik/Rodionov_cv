import numpy as np 
from skimage.measure import label, regionprops

coins_matrix = np.load('C:/Users/bqgk6/OneDrive/Рабочий стол/coins/coins.npy')

def calculated_coins(binary_image):
    binary_mask = (binary_image > 0).astype(int)
    labeled_image = label(binary_mask)

    total_value = 0

    for region in regionprops(labeled_image):
        area = region.area

        if area < 500:
            total_value += 1
        elif area < 1000:
            total_value += 2
        elif area < 2000:
            total_value += 5
        else:
            total_value += 10
    return total_value

result = calculated_coins(coins_matrix)
print("Общая сумма монет", result)

# import numpy as np
# import matplotlib.pyplot as plt

# coins_matrix = np.load('C:/Users/bqgk6/OneDrive/Рабочий стол/coins/coins.npy')

# print("размер матрицы", coins_matrix.shape)

# plt.imshow(coins_matrix, cmap='gray')
# plt.axis('off')
# plt.show()
