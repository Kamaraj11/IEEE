import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm

class DermAIPredictor:
    def __init__(self, model_path, num_classes=7):
        # The number of classes in HAM10000 is 7
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pipeline on device: {self.device}")
        
        # 1. Load EfficientNet-B0 architecture
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        
        # 2. Load saved weights from ham10000_model.pth
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"Warning: Could not find weights at {model_path}. Using untrained weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Same preprocessing (224x224 resize + normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Standard HAM10000 diagnostic labels mapping
        self.class_mapping = {
            0: 'akiec', # Actinic keratoses and intraepithelial carcinoma
            1: 'bcc',   # Basal cell carcinoma
            2: 'bkl',   # Benign keratosis-like lesions
            3: 'df',    # Dermatofibroma
            4: 'nv',    # Melanocytic nevi
            5: 'vasc',  # Vascular lesions
            6: 'mel'    # Melanoma
        }

    # 3. Create predict(image_path) function
    def predict(self, image_source):
        """
        Accepts either a string path to an image file or a direct PIL image (e.g. from FastAPI memory)
        """
        if isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
        else:
            image = image_source.convert("RGB")
            
        # Apply transformation and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1).squeeze()
            
        # 5. Return Predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        top3 = []
        for i in range(3):
            idx = top_indices[i].item()
            prob_percent = float(top_probs[i].item()) * 100
            top3.append({
                "class": self.class_mapping.get(idx, f"Class {idx}"),
                "probability": f"{prob_percent:.1f}%"
            })
            
        # Return Predicted class name, Probability, Top-3 predictions
        primary_class = top3[0]["class"]
        primary_prob = top3[0]["probability"]
        
        return {
            "disease": primary_class,
            "probability": primary_prob,
            "top_3": top3,
            "raw_confidence": float(top_probs[0].item())
        }

# Example Usage
if __name__ == "__main__":
    predictor = DermAIPredictor("ham10000_model.pth")
    # result = predictor.predict("/path/to/test_image.jpg")
    # print(result)
