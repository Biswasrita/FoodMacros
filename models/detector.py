import os
import urllib.request
import torch
from ultralytics import YOLO
import numpy as np

class FoodDetector:
    def __init__(self, weights_path='weights/gfsd_yolo_weights.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Check if custom local weights exist
        if os.path.exists(weights_path):
            print(f"Loading custom GFSD weights from {weights_path}...")
            self.model = YOLO(weights_path)
        else:
            # 2. Attempt to download ISSAI GFSD YOLOv8 XL weights directly
            issai_url = "https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/GFSD/yolov8x.pt"
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            print(f"Custom weights file not found at {weights_path}.")
            print("Downloading ISSAI GFSD YOLOv8 XL pre-trained weights...")
            try:
                urllib.request.urlretrieve(issai_url, weights_path)
                self.model = YOLO(weights_path)
                print("Successfully downloaded and loaded GFSD YOLOv8 XL weights.")
            except Exception as e:
                print(f"Direct GFSD weight download failed ({e}). Falling back to standard YOLOv8 XL ('yolov8x.pt')...")
                self.model = YOLO('yolov8x.pt')
                
        self.model.to(self.device)
        
    def detect(self, image_pil):
        results = self.model(image_pil)
        if len(results) == 0:
            return [], []
            
        boxes_data = results[0].boxes.data.cpu().numpy()
        boxes = []
        crops = []
        img_np = np.array(image_pil)
        
        image_height, image_width = img_np.shape[:2]
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls = box
            x1 = max(0, min(int(x1), image_width))
            y1 = max(0, min(int(y1), image_height))
            x2 = max(0, min(int(x2), image_width))
            y2 = max(0, min(int(y2), image_height))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append({
                'box': [x1, y1, x2, y2],
                'conf': float(conf),
                'cls': int(cls),
                'name': self.model.names[int(cls)]
            })
            crops.append(img_np[y1:y2, x1:x2])
            
        return boxes, crops