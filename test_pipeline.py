import cv2
import numpy as np
import json
import os
from model_loader import ModelLoader

# 1. Create Normal Skin (Low Variance, low edge density)
normal_skin = np.ones((400, 400, 3), dtype=np.uint8) * 180  # Light beige
normal_skin[:, :, 2] += 20 # Add slight red tint
# Add very soft noise
noise = np.random.normal(0, 5, (400, 400, 3)).astype(np.uint8)
normal_skin = cv2.add(normal_skin, noise)
cv2.imwrite("test_normal.jpg", normal_skin)

# 2. Create Scraped Knee (High Variance, high edge density)
scraped_knee = np.ones((400, 400, 3), dtype=np.uint8) * 200
# Add harsh high-frequency noise (like scabs and scrapes)
for _ in range(3000):
    x, y = np.random.randint(0, 400), np.random.randint(0, 400)
    color = [np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(150, 255)] # Reddish/dark spots
    cv2.circle(scraped_knee, (x, y), np.random.randint(1, 4), color, -1)
cv2.imwrite("test_scraped_knee.jpg", scraped_knee)

# 3. Create Melanoma (Discrete smooth blob on healthy skin)
melanoma = np.ones((400, 400, 3), dtype=np.uint8) * 180
melanoma[:, :, 2] += 20
# Draw a smooth, isolated dark blob in the center
cv2.ellipse(melanoma, (200, 200), (45, 60), 10, 0, 360, (50, 60, 80), -1)
# Add some slight blurring so it's a smooth blob, not high-frequency noise
melanoma = cv2.GaussianBlur(melanoma, (7, 7), 0)
cv2.imwrite("test_melanoma.jpg", melanoma)

print("Synthesized 3 Validation Images!")
print("Loading Medical Inference Pipeline...\n")

pipeline = ModelLoader(model_path="ham10000_model.pth")

tests = [
    ("test_normal.jpg", "Expected: OOD Rejection (Normal Skin)"),
    ("test_scraped_knee.jpg", "Expected: OOD Rejection (Highly Textured Noise)"),
    ("test_melanoma.jpg", "Expected: Valid AI Inference Pipeline Evaluation")
]

print("="*60)
for path, expectation in tests:
    print(f"\nEvaluating Image: {path}")
    print(f"Goal: {expectation}")
    
    result = pipeline.predict(path)
    
    # Prune giant 'all_probs' dictionary for better terminal viewing
    if result and "all_probs" in result:
        del result["all_probs"]
        
    print(f"PIPELINE OUTPUT:")
    print(json.dumps(result, indent=2))

print("\n" + "="*60)
