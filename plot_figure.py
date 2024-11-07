import pandas as pd
import matplotlib.pyplot as plt
import os
# 读取 CSV 文件
data = pd.read_csv('runs/detect/yolov8n-det-grape-+-dcn-wiou/results.csv',header=0)
figure_save_path = "E:/ZYJ/yolov8/ultralytics-main-detect/runs/detect/yolov8n-det-grape-+-dcn-wiou/result figure/"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
x = data.iloc[:,0]
y = data.iloc[:,4:8]

for col in y.columns:
    plt.plot(x,y[col],label=col)
    # plt.legend(loc = 'lower right')   # upper right
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(col)
    plt.savefig(os.path.join(figure_save_path, '%d.jpg' %col))
    # plt.show()



