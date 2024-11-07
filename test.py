import numpy as np
import matplotlib.pyplot as plt

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def find_nearest_point(point, points):
    min_distance = float('inf')
    nearest_point = None
    for p in points:
        dist = distance(point, p)
        if dist < min_distance:
            min_distance = dist
            nearest_point = p
    return nearest_point, min_distance

def replace_points(x1, x2, threshold=500):
    replaced_x1 = x1.copy()
    for i, point1 in enumerate(x1):
        nearest_point, min_distance = find_nearest_point(point1, x2)
        if min_distance < threshold:
            replaced_x1[i] = nearest_point
    return replaced_x1

def plot_points(points):
    points = np.array(points)
    plt.scatter(points[:,0], points[:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points')
    plt.show()

x1=[[1,2],[3,4],[5,6],[7,8]]
x2=[[8,9],[10,11],[12,13],[14,15],[16,17]]

replaced_x1 = x1
while True:
    new_x1 = replace_points(replaced_x1, x2)
    if np.array_equal(replaced_x1, new_x1):
        break
    replaced_x1 = new_x1

plot_points(replaced_x1)