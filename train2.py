import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:\\Download\\ultralytics-main\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-ContextGuided-damoyolo-iRMB.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='D:\\Download\\ultralytics-main\\ultralytics-main\\diedao.yaml',

                imgsz=640,
                epochs=300,
                batch=32,

                device='0',
                # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='C:\\Users\\jsj_gjj\\Desktop\\result',
                name='yolov8-ContextGuided-damoyolo-iRMB',
                )