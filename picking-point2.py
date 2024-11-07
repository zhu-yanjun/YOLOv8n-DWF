# -*- coding: utf-8 -*-
import os
import cv2
import shutil
import datetime
import matplotlib.pyplot as plt
import numpy as np
# from skimage import io
from PIL import Image
import sys

starttime = datetime.datetime.now()

path_result = "runs/detect/yolov8n-det-grape-+-dcn-wiou-other"  # yolov8识别结果
path_final_result = "runs/detect/yolov8n-det-grape-+-dcn-wiou-other/final_result" # 采摘点识别结果存储路径
path = "data/dataset/images/iamges_grape"  # 原始jpg图片
path3 = "runs/detect/yolov8n-det-grape-+-dcn-wiou-other/crops/axis"  # 裁剪出来的小图保存的根目录
# w = 640                      # 原始图片resize
# h = 640
img_total = []
txt_total = []

old_path = "runs/detect/yolov8n-det-grape-+-dcn-wiou-other/labels"   # txt读取路径
new_path = "data/dataset/images/iamges_grape" # 复制一份txt，并与原始图片放在一起

filelist_t = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
for file in filelist_t:
    src = os.path.join(old_path, file)
    dst = os.path.join(new_path, file)
    shutil.copy(src, dst)

file = os.listdir(path)
for filename in file:
    first, last = os.path.splitext(filename)
    if last == ".jpg":  # 图片的后缀名
        img_total.append(first)
    else:
        txt_total.append(first)

for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_ + ".jpg"  # 图片的后缀名
        path1 = os.path.join(path, filename_img)
        img = cv2.imread(path1)
        size = img.shape
        h = size[0]
        w = size[1]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  # resize 图像大小，否则roi区域可能会报错
        filename_txt = img_ + ".txt"

        n = 1
        with open(os.path.join(path, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
            count = len(open(os.path.join(path, filename_txt), 'rb').readlines())
            ca_x = []
            ca_y = []
            h_a = []
            w_a = []

            for line in f:
                aa = line.split(" ")
                if int(aa[0]) == 1:
                    x_center_a = w * float(aa[1])  # aa[1]中心点的x坐标
                    ca_x.append(x_center_a)
                    y_center_a = h * float(aa[2])  # aa[2]中心点的y坐标
                    ca_y.append(y_center_a)
                    width_a = int(w * float(aa[3]))  # aa[3]图片width
                    w_a.append(width_a)
                    height_a = int(h * float(aa[4]))  # aa[4]图片height
                    h_a.append(height_a)
                elif int(aa[0]) == 0:
                    x_center_g = w * float(aa[1])  # aa[1]中心点的x坐标
                    y_center_g = h * float(aa[2])  # aa[2]中心点的y坐标
                    width_g = int(w * float(aa[3]))  # aa[3]图片width
                    height_g = int(h * float(aa[4]))  # aa[4]图片height

                n = n + 1

            x_add = []
            y_add = []

            if count == n - 1:
                for i in range(len(h_a)):
                    if h_a[i] / w_a[i] > 3:
                        point_x = int(ca_x[i])
                        point_y = int(ca_y[i])
                    else:
                        point_x = int(ca_x[i])
                        point_y = int(ca_y[i] - h_a[i] / 3)
                    x_add.append(point_x)
                    y_add.append(point_y)

                path_final = os.path.join(path_result, filename_img)
                img_final = cv2.imread(path_final)

                # 如果 img_final 为空，跳过当前图片
                if img_final is None:
                    continue

                for j in range(len(x_add)):
                    img_final_result = cv2.circle(img_final, (x_add[j], y_add[j]), 25, (0, 0, 255), -1)

                filename = img_ + ".jpg"
                pic_add = os.path.join(path_final_result, filename)

                # 确保 img_final_result 已定义
                if 'img_final_result' in locals():
                    cv2.imwrite(pic_add, img_final_result)
            else:
                continue

endtime = datetime.datetime.now()
print(endtime - starttime)
