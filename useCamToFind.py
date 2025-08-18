from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model(source=1, show=True, conf=0.4, save=True)