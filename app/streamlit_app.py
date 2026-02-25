import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
import os

# 8. Show disclaimer at the bottom (page config allows setting title)
st.set_page_config(page_title="AI Skin Disease Detector", page_icon="🩺")

# -- Separate Grad-CAM utility function -- #
class GradCAM:
    """
    Implements a simple Gradient-weighted Class Activation Mapping (Grad-CAM)
    for analyzing the 'attention' regions of the AI model.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # For timm efficientnet_b0, conv_head is the last convolution layer before pooling
        target_layer = self.model.conv_head
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Calculate Global Average Pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Multiply activations by weights
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        return cam

def overlay_heatmap(img_pil, heatmap):
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay original image and heatmap
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

# -- Separate model loading function -- #
@st.cache_resource
def load_model(model_path="ham10000_model.pth"):
    """
    Load EfficientNet-B0 architecture and weights from ham10000_model.pth.
    Uses MPS device if available.
    """
    # 3. Use MPS device if available, otherwise CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    
    # 1. Load EfficientNet-B0 architecture
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    
    # 2. Load saved weights
    path_to_check = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
    try:
        model.load_state_dict(torch.load(path_to_check, map_location=device))
    except FileNotFoundError:
        try:
             model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            st.warning(f"Model file '{model_path}' not found. Showing architecture only with untrained weights.")
    
    model.to(device)
    model.eval()
    return model, device

# -- Data Preprocessing -- #
def preprocess_image(image):
    # 5. Preprocess image: Resize to 224x224, Normalize same as training, Convert to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    # 4. Streamlit interface
    st.title("AI Skin Disease Detector")
    
    # Class mapping for HAM10000
    class_mapping = {
        0: 'Actinic keratoses (akiec)', 
        1: 'Basal cell carcinoma (bcc)',   
        2: 'Benign keratosis-like lesions (bkl)',   
        3: 'Dermatofibroma (df)',    
        4: 'Melanocytic nevi (nv)',    
        5: 'Vascular lesions (vasc)',  
        6: 'Melanoma (mel)'
    }

    st.markdown("### Upload an image to analyze")
    # Image upload option (jpg/png)
    uploaded_file = st.file_uploader("Choose a skin image (jpg/png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        model, device = load_model()
        input_tensor = preprocess_image(image).to(device)
        
        with st.spinner("Analyzing image..."):
            # Turn off gradients temporarily for raw prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1).squeeze()
            
            top_probs, top_indices = torch.topk(probs, 3)
            
            predicted_class_idx = top_indices[0].item()
            predicted_class = class_mapping.get(predicted_class_idx, "Unknown")
            probability = float(top_probs[0].item()) * 100
            
            with col2:
                # 6. Predict: Output predicted class name & Show probability
                st.markdown(f"**Predicted Class:** {predicted_class}")
                st.markdown(f"**Probability:** {probability:.1f}%")
                
                # Show Top-3 predictions with probabilities
                st.markdown("#### Top-3 Predictions:")
                for i in range(3):
                    idx = top_indices[i].item()
                    prob = float(top_probs[i].item()) * 100
                    st.write(f"{i+1}. {class_mapping.get(idx)} ({prob:.1f}%)")
                    
        st.markdown("### Explainability (Grad-CAM)")
        with st.spinner("Generating heatmap..."):
            # 7. Implement Grad-CAM
            # We must enable gradients for Grad-CAM's backward pass
            input_tensor.requires_grad_(True)
            grad_cam = GradCAM(model)
            
            # Generate heatmap for predicted class
            heatmap = grad_cam.generate(input_tensor, predicted_class_idx)
            
            # Overlay heatmap on original image
            overlay_img = overlay_heatmap(image, heatmap)
            
            # Display heatmap below prediction
            st.image(overlay_img, caption="Grad-CAM Attention Heatmap", use_container_width=False, width=400)
            
    st.markdown("---")
    # 8. Show disclaimer
    st.warning("⚠️ **Disclaimer:** This is an AI-assisted tool and not a medical diagnosis.")

if __name__ == "__main__":
    main()
