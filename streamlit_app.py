"""Browser front end for the license plate reader.

Run locally:  streamlit run streamlit_app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import plate_reader as pr

HERE = Path(__file__).parent
SAMPLES = {"Sample: car.jpg": HERE / "car.jpg", "Sample: car2.jpg": HERE / "car2.jpg"}

st.set_page_config(page_title="License Plate Reader", page_icon="🚗", layout="wide")
st.title("License Plate Reader")
st.write(
    "Two stages, no deep learning: **(1)** Otsu thresholding and connected regions find plate-shaped "
    "candidates, **(2)** the sharpened crop is upscaled and read with Tesseract OCR."
)

with st.sidebar:
    source = st.radio("Image", [*SAMPLES, "Upload your own"])
    upload = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"]) if source == "Upload your own" else None
    st.caption(
        "The detector uses fixed size and aspect-ratio thresholds (plate area 1,000 to 20,000 px, "
        "aspect ratio 2 to 6), so it works best on close, level photos of roughly 500 to 800 px "
        "and can miss or misread others. Uploaded images are processed in memory and not stored."
    )

if source in SAMPLES:
    image = np.array(Image.open(SAMPLES[source]).convert("RGB"))
elif upload is not None:
    image = np.array(Image.open(upload).convert("RGB"))
else:
    st.info("Upload a photo of a car to try it, or pick a sample in the sidebar.")
    st.stop()

candidates = pr.find_candidates(image)

annotated = image.copy()
for i, (minr, minc, maxr, maxc) in enumerate(candidates, 1):
    cv2.rectangle(annotated, (minc, minr), (maxc, maxr), (255, 0, 0), 3)
    cv2.putText(annotated, str(i), (minc, max(minr - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

left, right = st.columns(2)
left.subheader("1. Detection")
left.image(annotated, caption=f"{len(candidates)} plate candidate(s) found", width="stretch")

if not candidates:
    right.subheader("2. Reading")
    right.warning("No plate-shaped region found. Try a closer, level photo of the plate.")
    st.stop()

choice = right.selectbox("Candidate", range(1, len(candidates) + 1), help="The original script always uses candidate 1.")
plate = pr.enhance_plate(image, candidates[choice - 1])
right.subheader("2. Reading")
right.image(plate, caption="Sharpened crop passed to OCR", width="stretch")

with st.spinner("Running Tesseract..."):
    results = pr.read_plate(plate)

if results:
    best = results[0]
    right.metric("Plate", best["plate"].upper(), help="Highest average OCR confidence of the three modes")
    right.caption(f"Tesseract average confidence: {best['confidence']:.0f} / 100")
    with right.expander("All OCR modes"):
        st.table(
            [{"mode": r["config"], "reading": r["plate"], "confidence": round(r["confidence"])} for r in results]
        )
else:
    right.warning("Tesseract could not read any text from this candidate.")
