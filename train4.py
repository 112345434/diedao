import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:\\Download\\ultralytics-main\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-LSKNet.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='D:\\Download\\ultralytics-main\\ultralytics-main\\diedao2.yaml',

                imgsz=640,
                epochs=300,
                batch=32,

                device='1',
                # using SGD
                # res'person', 'falling', 'fall', '10+', 'dog'ume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project="C:\\Users\\jsj_gjj\\Desktop\\result\\mix-fall",
                name='yolov8-LSKNet',
                )