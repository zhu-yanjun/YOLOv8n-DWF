import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/yolov8n-seg-g-p2-mdinner-1.1/weights/best.pt') # select your model.pt path
    model.track(source='dataset/0619.mp4',
                imgsz=320,
                project='runs/track',
                name='0619',
                save=True,
                save_txt=True,
                save_crop=True,
                )