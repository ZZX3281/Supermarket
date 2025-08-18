from ultralytics import YOLO
import os
from partition import partition_img
import cv2
if __name__ == '__main__':
    paths=r'D:\datasets\dataset_Single goal\dataset_Single goal\can'
    idx=-6
    for path_name in os.listdir(paths):
        id=0
        label_path=os.path.join(paths,path_name)
        for label in os.listdir(label_path):
            littel_label=os.path.join(label_path,label)
            model = YOLO(r'/runs/segment/train3/weights/best.pt')
            # Train the model
            results=model(littel_label)
            frame=cv2.imread(littel_label)
            processing_img=partition_img(frame,results)
            name=os.path.join(paths,str(idx))
            name=os.path.join(name,f'{id}.jpg')
            id+=1
            cv2.imwrite(name,processing_img)
        idx+=1


