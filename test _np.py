import numpy as np

# 创建一个多维数组
array = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

# 获取同一列的最大值和最小值
max_values = np.max(array, axis=0)
min_values = np.min(array, axis=0)

print("每列的最大值:", max_values)
print("每列的最小值:", min_values)
