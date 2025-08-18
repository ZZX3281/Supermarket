from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"D:\projects\PythonProject\smart supermarket\yolo11m-seg.pt")
    # Train the model
    results = model.train(data=r"D:\datasets\shangcao_data2\data.yaml", epochs=100,batch=8 ,imgsz=640,workers=0)
