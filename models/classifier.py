import torch
import torch.nn as nn
import torchvision.transforms as T
import timm

class HybridFoodClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # EfficientNet-V2 branch (using b0 for local/CPU-friendly inference if v2 isn't strictly requested in timm)
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        effnet_features = self.effnet.num_features
        
        # ViT branch
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_features = self.vit.num_features
        
        # Fusion Head for concatenated feature embeddings
        self.fc = nn.Linear(effnet_features + vit_features, num_classes)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.to(self.device)
        
    def forward(self, x):
        f_eff = self.effnet(x)
        f_vit = self.vit(x)
        f_fused = torch.cat((f_eff, f_vit), dim=1)
        return self.fc(f_fused)
        
    def predict(self, crop_np):
        self.eval()
        with torch.no_grad():
            tensor_img = self.transform(crop_np).unsqueeze(0).to(self.device)
            logits = self.forward(tensor_img)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            
            # Example mock classes mapped for this prototype (would be GFSD classes normally)
            classes = ['apple', 'chicken', 'rice', 'broccoli', 'burger']
            label = classes[pred.item() % len(classes)]
            return label, conf.item()
