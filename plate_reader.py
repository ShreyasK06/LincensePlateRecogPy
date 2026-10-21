"""Reusable version of the two-stage pipeline in plateDetection.py and reader.py.

Stage 1 finds plate-shaped regions with Otsu thresholding and connected components.
Stage 2 upscales the sharpened plate crop and reads it with Tesseract.
The thresholds and OCR settings are the same ones the scripts use.
"""
import re
import shutil

import cv2
import numpy as np
import pytesseract
from scipy import ndimage
from skimage import color, exposure, filters, measure

WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_CONFIGS = ["--oem 3 --psm 8", "--oem 3 --psm 13", "--oem 3 --psm 7"]  # word, raw line, text line

if shutil.which("tesseract") is None:
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT


def find_candidates(image):
    """Return plate candidate boxes (minr, minc, maxr, maxc) for an RGB image, in scan order."""
    gray = color.rgb2gray(image)
    binary = gray < filters.threshold_otsu(gray)
    labels = measure.label(~binary)  # invert so the light regions are labelled
    candidates = []
    for region in measure.regionprops(labels):
        minr, minc, maxr, maxc = region.bbox
        aspect_ratio = (maxc - minc) / (maxr - minr)
        if 2 < aspect_ratio < 6 and 1000 < region.area < 20000:
            candidates.append((minr, minc, maxr, maxc))
    return candidates


def enhance_plate(image, box):
    """Crop a candidate and sharpen it with a per-channel unsharp mask."""
    minr, minc, maxr, maxc = box
    plate = image[minr:maxr, minc:maxc]
    out = np.zeros(plate.shape, dtype=np.float64)
    for i in range(3):
        channel = plate[:, :, i] / 255.0
        blurred = ndimage.gaussian_filter(channel, sigma=1.0)
        out[:, :, i] = exposure.rescale_intensity(channel + 1.5 * (channel - blurred), out_range=(0, 1))
    return (out * 255).astype(np.uint8)


def format_plate(text):
    """Lower-case, strip punctuation, and put a space after the third character of 7+ character plates."""
    clean = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
    return clean[:3] + " " + clean[3:] if len(clean) >= 7 else clean


def read_plate(plate_rgb):
    """OCR an enhanced plate crop with three page-segmentation modes; best average confidence first."""
    gray = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)
    large = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    _, processed = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = []
    for config in OCR_CONFIGS:
        text = pytesseract.image_to_string(processed, config=config).strip()
        if not text:
            continue
        data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        confs = [int(float(c)) for c in data["conf"] if int(float(c)) > 0]
        results.append({
            "config": config,
            "raw": text,
            "plate": format_plate(text),
            "confidence": float(np.mean(confs)) if confs else 0.0,
        })
    return sorted(results, key=lambda r: r["confidence"], reverse=True)
