import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("runs/detect/my_custom_save/yolov8m_saved_weights.pt")
box_annotator = sv.BoundingBoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    return box_annotator.annotate(frame.copy(), detections=detections)

sv.process_video(
    source_path="samples/vid1.mp4",
    target_path="result.mp4",
    callback=callback
)

