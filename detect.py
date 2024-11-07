import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/yolov8n-det-grape-+-dcn-wiou/weights/best.pt') # select your model.pt path
    model.predict(source='data/dataset/images/iamges_grape',
                project='runs/detect/yolov8n-det-grape-+-dcn-wiou/predict-other/',
                name='yolov8n-det-grape-+-dcn-wiou-other',
                save=True,
                save_txt = True,
                save_crop = True,
                line_thickness = 6,
                # visualize=True # visualize model features maps
                )