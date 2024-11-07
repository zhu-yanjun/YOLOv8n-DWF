import cv2
from PIL import Image
import numpy as np
import os



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

file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/axis/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu')
save_out = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu/" # 保存图片的文件夹名称

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/B'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/B')
save_out1 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/B/"
if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/G'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/G')
save_out2 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/G/"
if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/R'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/R')
save_out3 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/R/"

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/define_img'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/define_img')
save_out4 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/define_img/"

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/H'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/H')
save_out5 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/H/"

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/S'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/S')
save_out6 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/S/"

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/V'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/V')
save_out7 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/V/"

for img_name in file_list:
    img_path = file_root + img_name
    img = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # PIL 2 opencv
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # opencv 2 PIL



    # kernel_size = 5
    # scale = 1
    # delta = 1
    # ddepth = cv2.CV_16S
    #
    # dst = cv2.Laplacian(img, ddepth, ksize=kernel_size, scale=scale, delta=delta)
    # img = cv2.convertScaleAbs(dst)

    (B, G, R) = cv2.split(img)  # 分离图像的RBG分量

    # define_img = 0.8*G - 0.4*R + 0.8*B

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转换为灰度图
    (H,S,V) = cv2.split(HSV)  # 将HSV格式的图片分解为3个通道

    define_img = 1.2 * V + 0.1 * S - 1.9 * H




    out_name = img_name.split('.')[0]
    save_path = save_out1 + out_name + '.jpg'
    cv2.imwrite(save_path, B)

    out_name = img_name.split('.')[0]
    save_path = save_out2 + out_name + '.jpg'
    cv2.imwrite(save_path, G)   # 伽马变换  HSV

    out_name = img_name.split('.')[0]
    save_path = save_out3 + out_name + '.jpg'
    cv2.imwrite(save_path, R)   # 伽马变换  HSV


    out_name = img_name.split('.')[0]
    save_path = save_out4 + out_name + '.jpg'
    cv2.imwrite(save_path, define_img)   # 伽马变换  HSV

    out_name = img_name.split('.')[0]
    save_path = save_out5 + out_name + '.jpg'
    cv2.imwrite(save_path, H)   # 伽马变换  HSV

    out_name = img_name.split('.')[0]
    save_path = save_out6 + out_name + '.jpg'
    cv2.imwrite(save_path, S)   # 伽马变换  HSV

    out_name = img_name.split('.')[0]
    save_path = save_out7 + out_name + '.jpg'
    cv2.imwrite(save_path, V)   # 伽马变换  HSV











