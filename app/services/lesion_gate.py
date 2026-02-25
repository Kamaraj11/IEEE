import cv2
import numpy as np

def detect_lesion(image_path: str) -> bool:
    """
    Stage-1 Binary Gate (Lesion Detection Layer).
    Implements a lightweight heuristic gate using color variance.
    Returns False if the image is highly uniform Normal Skin.
    Returns True if a lesion is detected.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return True

        # Resize for consistent uniform metric calculation
        img = cv2.resize(img, (224, 224))
        
        # 1. Image Variance Check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Extremely smooth uniform block of skin
        # Normal skin has very low variance. Real lesions create variance in the pixels.
        if std_dev < 10.0 and edge_density < 0.01:
            return False
                
        # If it passed, assume there is a valid structural anomaly to be classified by ML
        return True
    except Exception as e:
        print(f"Lesion Gate Warning: {e}")
        # Always fail-open to the ML pipeline if CV2 maths corrupt
        return True
