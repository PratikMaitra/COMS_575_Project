from ultralytics import YOLO

model = YOLO("runs/detect/my_custom_save/yolov8m_saved_weights.pt")

image_path = "DATASET/sample1.jpg"

results = model(image_path,classes=32,save=True)


for result in results:
    result.show()  

    