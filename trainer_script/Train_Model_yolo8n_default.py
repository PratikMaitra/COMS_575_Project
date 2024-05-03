import os
os.environ["WANDB_MODE"] = "dryrun"  # Disables W&B online syncing
os.environ["WANDB_SILENT"] = "true"  
from ultralytics import YOLO
import torch  
torch.cuda.empty_cache()  

model = YOLO("yolov8n.pt")  


model.train(
    data="./dataset/Dataset1/data.yaml",  
    epochs=5,  
    batch=1,  
    imgsz=640, 
    patience=10,  
    device="0",  
    save_dir="runs/detect/my_custom_save"  
)

metrics = model.val()  

results_dict = metrics.results_dict

# Print extracted metrics for the entire model
print(f"Precision: {results_dict['metrics/precision(B)']}")
print(f"Recall: {results_dict['metrics/recall(B)']}")
print(f"mAP@50: {results_dict['metrics/mAP50(B)']}")
print(f"mAP@50-95: {results_dict['metrics/mAP50-95(B)']}")
print(f"Fitness: {results_dict['fitness']}")

# Optionally, save metrics to a file
metrics_output_path = "runs/detect/my_custom_save/metrics.txt"
with open(metrics_output_path, "w") as f:
    for key, value in results_dict.items():
        f.write(f"{key}: {value}\n")
print(f"Metrics saved to {metrics_output_path}")

model.save("runs/detect/my_custom_save/yolov8n_saved_weights.pt")  


torch.cuda.empty_cache()  
