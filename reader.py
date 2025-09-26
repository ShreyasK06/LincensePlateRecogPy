import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
import numpy as np
import re

def format_license_plate(text):
    """Format license plate text with proper spacing"""
    # Remove all non-alphanumeric characters first
    clean = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    
    if len(clean) >= 7:  # Typical format: ABC 123AA
        # Insert space after first 3 characters for typical UK format
        formatted = clean[:3] + ' ' + clean[3:]
        return formatted
    
    return clean

def read_license_plate():
    """Read license plate with optimized settings"""
    
    img = cv2.imread("plateImage.jpg")
    if img is None:
        print("plateImage.jpg not found")
        return ""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use the best performing method from our tests
    # High resolution OTSU gave us 'egb626aa' with good confidence
    large = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    _, processed = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try the best configurations
    configs = [
        "--oem 3 --psm 8",  # Single word
        "--oem 3 --psm 13", # Raw line  
        "--oem 3 --psm 7",  # Single text line
    ]
    
    results = []
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(processed, config=config).strip()
            
            if text:
                # Clean and format
                clean_text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
                formatted = format_license_plate(clean_text)
                
                # Get confidence
                data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                results.append({
                    'raw': text,
                    'clean': clean_text,
                    'formatted': formatted,
                    'confidence': avg_confidence,
                    'config': config
                })
                
        except Exception as e:
            continue
    
    return results

# Main execution
results = read_license_plate()

if results:
    # Find the best match based on confidence
    best_match = max(results, key=lambda x: x['confidence'])
    final_answer = best_match['formatted']
    print(f"Plate Number: {final_answer}")
else:
    print("Could not read license plate")