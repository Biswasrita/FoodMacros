import torch
import cv2
import numpy as np

class DepthVolumeEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load MiDaS small for balanced inference speed on CPU/GPU
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def get_depth_map(self, img_np):
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        # Normalize between 0 and 1 (where 1 is closest, 0 is furthest)
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)
        return depth_map
        
    def estimate_mass(self, depth_map, box, density_g_cm3):
        # Extract the depth profile for the specific food bounding box
        x1, y1, x2, y2 = box
        food_depths = depth_map[y1:y2, x1:x2]
        
        food_mean_depth = np.mean(food_depths)
        
        # Calculate background plate pixels (proxy: regions outside bounding boxes)
        h, w = depth_map.shape
        bg_mask = np.ones((h, w), dtype=bool)
        bg_mask[y1:y2, x1:x2] = False
        bg_mean_depth = np.mean(depth_map[bg_mask])
        
        # Relative height from background
        relative_height = max(0.01, food_mean_depth - bg_mean_depth)
        pixel_area = (x2 - x1) * (y2 - y1)
        
        # Scaled spatial constant (assuming standardized focal length/distance for this prototype)
        spatial_scale_constant = 0.05
        
        volume_cm3 = pixel_area * relative_height * spatial_scale_constant
        mass_g = volume_cm3 * density_g_cm3
        
        return mass_g, volume_cm3
