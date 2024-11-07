# -*- coding: utf-8 -*-
import os
import cv2
import shutil
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from PIL import Image
import sys



path_result = "runs/detect/yolov8n-det-grape-+-dcn-wiou/predict"  # yolov8识别结果
path_final_result = "runs/detect/yolov8n-det-grape-+-dcn-wiou/predict/picking_points" # 采摘点识别结果存储路径
path = "data/dataset/images/test"  # 原始jpg图片
path3 = "runs/detect/yolov8n-det-grape-+-dcn-wiou/predict/crops/axis__self"  # 裁剪出来的小图保存的根目录
# w = 640                      # 原始图片resize
# h = 640
img_total = []
txt_total = []

old_path = "runs/detect/yolov8n-det-grape-+-dcn-wiou/predict/labels"   # txt读取路径
new_path = "data/dataset/images/test" # 复制一份txt，并与原始图片放在一起

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
    # print(img_total)
    else:
        txt_total.append(first)



for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_ + ".jpg"  # 图片的后缀名
        # print('filename_img:', filename_img)
        path1 = os.path.join(path, filename_img)
        img = cv2.imread(path1)
        size = img.shape
        # print(size)
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
            left_tx_a = []
            right_tx_a = []
            left_ty_a = []


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
                    lefttopx_a = int(x_center_a - width_a / 2.0)    # aa[1]左上点的x坐标
                    left_tx_a.append(lefttopx_a)
                    lefttopy_a = int(y_center_a - height_a / 2.0)   # aa[2]左上点的y坐标
                    left_ty_a.append(lefttopy_a)
                    righttopx_a = int(x_center_a + width_a / 2.0)
                    right_tx_a.append(righttopx_a)
                    righttopy_a = int(y_center_a - height_a / 2.0)
                    # [左上y:右下y,左上x:右下x] (y1:y2,x1:x2)需要调参，否则裁剪出来的小图可能不太好
                    roi_a = img[lefttopy_a + int(height_a/2)+1:lefttopy_a + height_a + 3, lefttopx_a + 1:lefttopx_a + width_a + 1]

                    filename_last = img_ + "_" + str(n) + ".jpg"  # 裁剪出来的小图文件名

                    cv2.imwrite(os.path.join(path3, filename_last), roi_a)
                    path_4 = os.path.join(path3, filename_last)

                    n = n + 1
                    x_add = []
                    y_add = []
                    # if len(h_a)>1:
                    for i in range(0,len(h_a)):
                        gray_value = 0
                        if h_a[i]/w_a[i]<3:
                            # roi_b = cv2.imread(path1)
                            # gray_pil = Image.fromarray(cv2.cvtColor(roi_b, cv2.COLOR_BGR2RGB))
                            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            h = h_a[i]
                            a = left_tx_a[i]
                            b = right_tx_a[i]
                            c = left_ty_a[i]
                            gray_value = max(gray_img[int(left_ty_a[i] + int(3 * h / 4)), left_tx_a[i]:right_tx_a[i]])
                            xxx = int(left_ty_a[i] + int(3 * h / 4))
                            index = np.where(gray_img[int(left_ty_a[i] + int(3 * h / 4)),left_tx_a[i]:right_tx_a[i]] == int(gray_value) )
                            if len(index[0]) >= 2:
                                point_x = int(index[0][int(len(index[0]) / 2)])+left_tx_a[i]
                                point_y = int(int(left_ty_a[i]) + int(3 * h / 4))
                            else:
                                point_x = int(index[0][0])+left_tx_a[i]
                                point_y = int(left_ty_a[i]+ int(3 * h / 4))
                        else:
                            point_x = int(ca_x[i])
                            point_y = int(ca_y[i]+int(h_a[i]/4))
                        x_add.append(point_x)
                        y_add.append(point_y)
                        path_final = os.path.join(path_result, filename_img)
                        img_final = cv2.imread(path_final)
                    for j in range(len(x_add)):
                        img_final_result = cv2.circle(img_final, (x_add[j], y_add[j]), 25, (0, 0, 255), -1)
                    filename_all = img_ + ".jpg"
                    pic_add = os.path.join(path_final_result, filename_all)
                    cv2.imwrite(pic_add, img_final_result)
            else:
                continue