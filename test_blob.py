import cv2
import numpy as np
import glob

def analyze_blob(image_path: str):
    img = cv2.imread(image_path)
    if img is None: return
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Calculate mean color inside and outside the blob
        mean_in = cv2.mean(img, mask=mask)[:3]
        
        inv_mask = cv2.bitwise_not(mask)
        mean_out = cv2.mean(img, mask=inv_mask)[:3]
        
        diff = np.linalg.norm(np.array(mean_in) - np.array(mean_out))
        print(f"[{image_path.split('/')[-1]}] Color Diff In vs Out: {diff:.2f}")

for f in glob.glob("app/static/uploads/*.jpeg"):
    analyze_blob(f)
    
analyze_blob("test_melanoma.jpg")
