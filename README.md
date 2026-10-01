# LincensePlateRecogPy

A two-stage license plate reader in Python: locate the plate in a photo with classical image processing, then read the characters with Tesseract OCR.

## How it works

**1. Plate detection (`plateDetection.py`)**
- Loads a car photo, converts it to grayscale, and applies Otsu thresholding.
- Labels connected regions and keeps candidates whose bounding boxes look like a plate: aspect ratio between 2 and 6, area between 1,000 and 20,000 pixels.
- Displays the candidates, the grayscale and binary images, and the best candidate.
- Sharpens the first candidate with an unsharp mask and writes it to `plateImage.jpg`.

**2. Plate reading (`reader.py`)**
- Loads `plateImage.jpg`, upscales it 6x, and applies Otsu thresholding.
- Runs Tesseract with three page-segmentation modes (single word, raw line, single text line).
- Picks the result with the highest average confidence and formats it with a space after the third character (a UK-style layout).

## Requirements

- Python 3.9+
- Packages: `numpy`, `scipy`, `matplotlib`, `scikit-image`, `opencv-python`, `pytesseract`
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed. `reader.py` currently points at the Windows default path (`C:\Program Files\Tesseract-OCR\tesseract.exe`); edit that line for your system.

```bash
pip install numpy scipy matplotlib scikit-image opencv-python pytesseract
```

## Usage

```bash
python plateDetection.py   # edit the image filename near the top; writes plateImage.jpg
python reader.py           # prints the best OCR result
```

Sample inputs are included (`car.jpg`, `car2.jpg`), along with `plateImage.jpg`, `test.png`, and a `segments/` folder of character crops from an earlier segmentation experiment (not used by the current scripts).

## Limitations

- Image filenames are hard-coded and the detector always uses the first candidate, so results depend on the photo.
- The detector uses fixed size and aspect-ratio thresholds, so it is sensitive to image resolution and camera angle.
- The scikit-learn and joblib imports in `plateDetection.py` are placeholders for a trained character classifier that has not been added.
- The formatting rule assumes a 7-character UK-style plate.

## Ideas for next steps

Replace the hand-tuned filter with a trained detector (for example, a small CNN or YOLO), add character segmentation plus a classifier, take the image path from the command line, and add a test set with accuracy measurement.
