import cv2
import numpy as np
import glob

def print_metrics(image_path: str):
    img = cv2.imread(image_path)
    if img is None: return
    img = cv2.resize(img, (224, 224))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_std = np.std(h)
    s_std = np.std(s)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_ratio = 0
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (224 * 224)
        
    print(f"[{image_path.split('/')[-1]}]")
    print(f"  -> std_dev: {std_dev:.1f} | h_std: {h_std:.1f} | s_std: {s_std:.1f}")
    print(f"  -> edge_dense: {edge_density:.4f} | lap_var: {lap_var:.1f}")
    print(f"  -> largest blob ratio: {area_ratio:.4f}\n")

for f in glob.glob("app/static/uploads/*.jpeg"):
    print_metrics(f)
