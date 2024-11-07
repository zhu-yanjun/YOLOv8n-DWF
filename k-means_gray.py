import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image


def kmeans(image, num_clusters, seed=0, max_iter=10000):
    np.random.seed(seed)
    img = image.reshape(-1, 1)
    # print(max(img))
    index = 0

    clus_value = np.array(np.random.rand(num_clusters) * max(img), dtype=float)
    # print(clus_value[0])

    while True:
        index += 1
        flag = clus_value.copy()

        cs = np.array([np.square(img - clus_value[0]), np.square(img - clus_value[1])])

        labels = np.argmin(cs, axis=0)

        for j in range(num_clusters):
            clus_value[j] = np.mean(img[labels == j])
        if np.sum(np.abs(clus_value - flag)) < 1e-8 or index == max_iter:
            break

    segmented_image = np.array(clus_value[labels].reshape(image.shape), dtype=np.uint16)
    return segmented_image

save_out = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/kmeans/"
file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/define_img/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
for img_name in file_list:
    img_path = file_root + img_name
    img = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # PIL 2 opencv
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # opencv 2 PIL

    segment_image1 = np.zeros(img.shape, dtype="uint16")
    segment_image = kmeans(img, num_clusters=2)
    # print(segment_image.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if segment_image[i][j] == np.max(segment_image):
                segment_image1[i][j] = 65535
    out_name = img_name.split('.')[0]
    save_path = save_out + out_name + '.jpg'
    cv2.imwrite(save_path, segment_image1)
