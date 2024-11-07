import numpy as np
import pylab as plt
import os
from skimage import io

import PIL.Image as image
from sklearn.cluster import KMeans

# img = plt.imread('E:/图片/壁纸/001.jpg')
path = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/define_img/'
path2 = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/kmeans/'
a = os.listdir(path)

for m in a:

    img = plt.imread(path + str('/') + m)

    img1 = img.reshape((img.shape[0]*img.shape[1], 3))

    k = 3
    kmeans = KMeans(n_clusters=k)

    kmeans.fit(img1)

    height = img.shape[0]
    width = img.shape[1]

    pic_new = image.new("RGB", (width, height))

    center = np.zeros([k, 3])

    for i in range(k):
        for j in range(3):
            center[i, j] = kmeans.cluster_centers_[i, j]
    center = center.astype(np.int32)

    label = kmeans.labels_.reshape((height, width))

    for i in range(height):
        for j in range(width):
            pic_new.putpixel((j, i), tuple((center[label[i][j]])))

    pic_new.save("D:/K=3.jpg", "JPEG")

    # io.imsave(path2 + m, pic_new)

