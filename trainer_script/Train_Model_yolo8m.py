import os
os.environ["WANDB_MODE"] = "dryrun"  # Disables W&B online syncing
os.environ["WANDB_SILENT"] = "true"  
from ultralytics import YOLO
import torch  
torch.cuda.empty_cache()  

model = YOLO("yolov8m.pt")  


model.train(
    data="./DATASET/Dataset1/data.yaml",  
    epochs=25,  
    batch=1,  
    imgsz=640, 
    patience=10,  
    device="0",  
    save_dir="runs/detect/my_custom_save"  
)

metrics = model.val()  

model.save("runs/detect/my_custom_save/yolov8m_saved_weights.pt")  


results_predict = model("DATASET/sample1.jpg") 

if isinstance(results_predict, list):
    for result in results_predict:
        result.show()  
else:
    
    results_predict.show()
    

torch.cuda.empty_cache()  
