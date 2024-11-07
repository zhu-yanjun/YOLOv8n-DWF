
from ultralytics import YOLO
#Load a model
model = YOLO("ultralytics/yolo/cfg/yolov8n-det-grape-+-DCN.yaml")

# model.train(**{'cfg':'ultralytics/yolo/cfg/train-seg.yaml'})
model.val(data="data/grape.yaml",epochs=300, imgsz=640,workers=0,batch=32,device=0,lr0=0.01,lrf=0.01)



