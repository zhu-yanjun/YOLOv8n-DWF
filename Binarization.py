import cv2
from PIL import Image
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square, diamond
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from skimage import morphology
import os
import matplotlib.pyplot as plt

# 图像灰度延展、直方图均衡化
def linear_trans(img, k, b=0):
    # 计算灰度线性变换的映射表
    trans_list = [(np.float32(x)*k+b) for x in range(256)]
    # 将列表转换为np.array
    trans_table = np.array(trans_list)
    # 将超过[0,245]灰度范围的数值进行调整，并指定数据类型为uint8
    trans_table[trans_table>255] = 255
    trans_table[trans_table<0] = 0
    trans_table = np.round(trans_table).astype(np.uint8)
    # 使用opencv的look up table函数修改图像的灰度值
    return cv2.LUT(img, trans_table)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region

# 锐化算子
sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])


file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/axis/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu')
save_out = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu/" # 保存图片的文件夹名称


for img_name in file_list:
    img_path = file_root + img_name
    img = Image.open(img_path)

#   图像裁剪一半
    imgSize = img.size
    w = img.width
    h = img.height
    # print(imgSize)
    # area = (0, 0, w, (h / 2))
    # img = img.crop(area)

    # if h/w > 1.2:
    #     area = (0, 0, w, (h/2))
    #     img = img.crop(area)
    # elif h/w <= 1.2:
    #     area = (0, (h/2), w, h)
    #     img = img.crop(area)



    # HSV
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # PIL 2 Opencv
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # 去模糊
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # PIL 2 Opencv
    clear = cv2.bilateralFilter(img, 9, 75, 75)


    # # Canny 边缘检测
    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # PIL 2 Opencv
    # edges = cv2.Canny(img, 50, 255)




    # sobel 算子边缘检测
    grayImage = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
    # t, dst = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO)  # t表示返回的阈值

    t, otsu = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, binary = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY_INV)
    # binary[binary == 255] = 0



    # # Sobel算子
    # x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)  # 对x求一阶导
    # y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)  # 对y求一阶导
    # absX = cv2.convertScaleAbs(x)
    # absY = cv2.convertScaleAbs(y)
    # Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)



    # 形态学运算

    selem = disk(1)  # square(2)  # #square(4) #disk(6)
    # 膨胀
    dilated = dilation(otsu, selem)
    # 腐蚀
    img = erosion(dilated, selem)


    # kernel = np.ones((2,2), np.uint8)
    # dilation = cv2.morphologyEx(otsu, cv2.MORPH_DILATE, kernel)
    # mask = select_max_region(dilation)

    # skeleton0 = morphology.skeletonize(mask)
    # skeleton = skeleton0.astype(np.uint8) * 255


    out_name = img_name.split('.')[0]
    save_path = save_out + out_name + '.jpg'
    cv2.imwrite(save_path, img)   # 伽马变换  HSV











