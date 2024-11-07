import cv2
import numpy as np
import os
from PIL import Image

def invert_image(im):
    im_arr = np.array(im)
    im_arr[im_arr == 0] = 255
    im_arr[im_arr == 255] = 0
    new_im = Image.fromarray(im_arr.astype('uint8'))
    return new_im

# if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu'):  # os模块判断并创建
#     os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu')
# save_out = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu/"
#
# if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu1'):  # os模块判断并创建
#     os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu1')
# save_out1 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu1/"

if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu2'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu2')
save_out2 = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu2/"

file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/define_img'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
for img_name in file_list:
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(file_root, img_name)
        img = cv2.imread(img_path,-1)

        # 中值滤波  做不做滤波处理等对图像分割影响也比较大，感兴趣的可以自己测试一下。
        img = cv2.medianBlur(img,5)

        # ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
        # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,1)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,1)

        # imgSize = img.size
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # opencv 2 PIL
        w = img.width
        h = img.height

        for i in w:
            for j in h:
                if img[i,j]==0:
                    img[i,j]=255
                else:
                    img[i,j]=0

    out_name = img_name.split('.')[0]
    save_path = save_out2 + out_name + '.jpg'
    cv2.imwrite(save_path, img)   # 伽马变换  HSV

