import cv2
from PIL import Image
import numpy as np
from skimage import morphology
import os
import matplotlib.pyplot as plt

file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/skeleton/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)

for img_name in file_list:
    img_path = file_root + img_name
    # img = cv2.imread(img_path, -1)
    img = Image.open(img_path)


    img = np.array(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 255:
                point = [i,j]



    img = img





