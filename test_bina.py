import os
import cv2

file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/otsu2'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
for img_name in file_list:
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(file_root, img_name)
        img = cv2.imread(img_path,0)
        imgSize = img.size
        w = img.width
        h = img.height