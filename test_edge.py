import cv2
import numpy as np
import glob

def check_blob_touches_edge(image_path: str):
    img = cv2.imread(image_path)
    if img is None: return
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        touches_edge = (x <= 5 or y <= 5 or (x + w) >= 219 or (y + h) >= 219)
        area_ratio = cv2.contourArea(largest_contour) / (224 * 224)
        print(f"[{image_path.split('/')[-1]}]")
        print(f"  Touches Edge: {touches_edge} | Box: ({x},{y},{w},{h}) | Area Ratio: {area_ratio:.4f}\n")

for f in glob.glob("app/static/uploads/*.jpeg"):
    check_blob_touches_edge(f)
    
check_blob_touches_edge("test_melanoma.jpg")
