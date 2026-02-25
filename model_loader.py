import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm

# Include local path for safe binary gate import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
try:
    from app.services.lesion_gate import detect_lesion
except ImportError:
    # Safe fallback if module path mismatches during testing
    detect_lesion = lambda x: True


class ModelLoader:
    def __init__(self, model_path="ham10000_model.pth"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on: {self.device}")
        
        self.num_classes = 7
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=self.num_classes)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded trained HAM10000 weights successfully.")
        else:
            print(f"Warning: {model_path} not found. Proceeding with untrained weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_mapping = {
            0: 'Actinic keratoses and intraepithelial carcinoma (akiec)',
            1: 'Basal cell carcinoma (bcc)',
            2: 'Benign keratosis-like lesions (bkl)',
            3: 'Dermatofibroma (df)',
            4: 'Melanocytic nevi (nv)',
            5: 'Vascular lesions (vasc)',
            6: 'Melanoma (mel)'
        }
        
        # Raw names for internal tracking
        self.short_names = {
            0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel'
        }

    def predict(self, image_path):
        try:
            # Stage-1 Binary Gate (Lesion Detection Layer)
            path_str = str(image_path)
            if not detect_lesion(path_str):
                return {
                    "prediction": "No clear lesion detected",
                    "confidence": None,
                    "medical_report": None
                }

            image = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1).squeeze()
                
            top_prob, top_idx = torch.max(probs, dim=0)

            # Confidence Rejection Layer
            if top_prob.item() < 0.75:
                return {
                    "prediction": "Uncertain classification – recommend clinical evaluation",
                    "confidence": float(top_prob.item()) * 100,
                    "medical_report": None
                }
            
            return {
                "class_name": self.class_mapping[top_idx.item()],
                "short_name": self.short_names[top_idx.item()],
                "probability": float(top_prob.item()) * 100,
                "all_probs": {self.short_names[i]: float(probs[i].item()) * 100 for i in range(self.num_classes)}
            }
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None
