import cv2
import numpy as np
from pyzbar.pyzbar import decode

def process_image(image_np):
    """Applies preprocessing fallbacks, decodes barcodes/QR codes, and draws boundaries."""
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # --- DECODING WITH PREPROCESSING FALLBACKS ---
    decoded_objects = decode(image_cv)
    
    if not decoded_objects:
        # Fallback 1: Grayscale and Otsu Thresholding
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        decoded_objects = decode(thresh)
        
    if not decoded_objects:
        # Fallback 2: Upscaling resolution via cubic interpolation
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        decoded_objects = decode(resized)
    # ---------------------------------------------
    
    annotated_image = image_np.copy()
    results = []
    
    for obj in decoded_objects:
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            n_pts = len(hull)
            for j in range(n_pts):
                pt1 = tuple(int(v) for v in hull[j][0])
                pt2 = tuple(int(v) for v in hull[(j+1) % n_pts][0])
                cv2.line(annotated_image, pt1, pt2, (0, 255, 0), 4)
        else:
            pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
            cv2.polylines(annotated_image, [pts], True, (0, 255, 0), 4)
            
        code_type = obj.type
        code_data = obj.data.decode('utf-8', errors='ignore')
        results.append({"type": code_type, "data": code_data})
        
    return annotated_image, results
