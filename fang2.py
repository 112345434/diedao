import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:\\Download\\ultralytics-main\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-damoyolo-EMA.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='C:\\ultralytics-main\\data.yaml',

                imgsz=640,
                epochs=300,
                batch=32,

                device='0',
                # using SGD
                # res'person', 'falling', 'fall', '10+', 'dog'ume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='D:\\Download\\ultralytics-main\\ultralytics-main\\result',
                name='yolov8-damoyolo-EMA',
                )