import os
os.environ["WANDB_MODE"] = "dryrun"  # Disables W&B online syncing
os.environ["WANDB_SILENT"] = "true"  
from ultralytics import YOLO
import torch  
torch.cuda.empty_cache()  

model = YOLO("yolov8l.pt")  


model.train(
    data="./dataset/Dataset1/data.yaml",  
    epochs=25,  
    batch=1,  
    imgsz=640, 
    patience=10,  
    device="0",  
    save_dir="runs/detect/my_custom_save"  
)

metrics = model.val()  

model.save("runs/detect/my_custom_save/yolov8l_saved_weights.pt")  



torch.cuda.empty_cache()  
