import cv2 as cv
import numpy as np
import os
from PIL import Image

def line_detection(image):  # 直线检测
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 120, 255, apertureSize=3)  # apertureSize是sobel算子窗口大小
    lines = cv.HoughLines(edges, 1, np.pi / 180, 300)  # 指定步长为1的半径和步长为π/180的角来搜索所有可能的直线

# HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None)
# 第一个参数image：是canny边缘检测后的图像
#
# 第二个参数rho和第三个参数theta：对应直线搜索的步长。在本例中，函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线。
#
# 最后一个参数threshold：是经过某一点曲线的数量的阈值，超过这个阈值，就表示这个交点所代表的参数对(rho, theta)在原图像中为一条直线
#
# 将求得交点坐标反代换进行直线绘制
    for line in lines:
        #  print(type(lines))
        rho, theta = line[0]  # 获取极值ρ长度和θ角度
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho  # 获取x轴值
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 获取这条直线最大值点x1
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))  # 获取这条直线最小值点y2　　其中*1000是内部规则/
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 划线
    cv.imshow("image-lines", image)


def line_detect_possible_demo(image):  # 检测出可能的线段
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 120, 255, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=10)

# HoughLinesP概率霍夫变换（是加强版）使用简单，效果更好，检测图像中分段的直线（而不是贯穿整个图像的直线)
# 第一个参数是需要处理的原图像，该图像必须为cannay边缘检测后的图像；

# 第二和第三参数：步长为1的半径和步长为π/180的角来搜索所有可能的直线
#
# 第四个参数是阈值，概念同霍夫变换
#
# 第五个参数：minLineLength-线的最短长度，比这个线短的都会被忽略。
#
# 第六个参数：maxLineGap-两条线之间的最大间隔，如果小于此值，这两条线就会被看成一条线

    for line in lines:
        #   print(type(line))
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo", image)


file_root = 'runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/axis/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
# save_dir_name = 'Histogram'
if not os.path.exists('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/houghb'):  # os模块判断并创建
    os.mkdir('E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/houghb')
save_out = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-ciou/predict/crops/houghb/"

for img_name in file_list:
    img_path = file_root + img_name
    src = cv.imread(img_path, -1)
    # src = Image.open(img_path)


    # src = cv.imread("qipan.jpg")
    # cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("input image", src)
    # line_detection(src.copy())
    # line_detect_possible_demo(src.copy())
    # cv.waitKey(0)
    #
    # cv.destroyAllWindows()
    out_name = img_name.split('.')[0]
    save_path = save_out + out_name + '.jpg'
    cv.imwrite(save_path, line_detection(src.copy()))   # 伽马变换  HSV


