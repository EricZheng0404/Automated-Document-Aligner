# Automated Document Aligner — `rectify.py`

This project implements an automatic computer vision pipeline that takes distorted images of letter-sized documents and produces perfectly rectified, frontal images. The system processes all images inside a given input folder and outputs 550×425 rectified results to a separate directory. The entire process requires no manual tuning once executed.


---

## Command-Line Usage

```bash
python3 rectify.py <folder_with_inputs>
```

- `<folder_with_inputs>` is a directory containing `.jpg` or `.png` images
- A new folder `rectified_outputs/` is automatically created in the current directory
- Each output is named `outputN.jpg` based on input ordering

---

## High-Level Pipeline Description

`rectify.py` performs the following automated steps for every input image:

### 1. Load & Preprocess
- Reads the image from disk
- Converts to grayscale
- Applies Gaussian or Bilateral filtering to remove noise while preserving edges

### 2. Binarization
- Uses Otsu's global thresholding
- Falls back to adaptive thresholding when lighting is uneven
- Produces a clean binary mask separating document from background

### 3. Feature / Contour Extraction
- Canny edge detection finds document boundaries
- `cv2.findContours` extracts all contours
- The largest quadrilateral contour is selected as the document
- Ramer–Douglas–Peucker polygon simplification reduces boundary to four main vertices

### 4. Corner Detection
- The simplified polygon provides the four document corners
- Corners are sorted consistently: top-left → top-right → bottom-right → bottom-left
- This prevents flipping or mirrored warps

### 5. Geometric Rectification
- **Source:** detected 4 corners
- **Destination:** a flat 11×8.5" document at 50 DPI → 550×425 px
- Computes homography with `cv2.getPerspectiveTransform`
- Rectifies perspective using `cv2.warpPerspective`

### 6. Output
- Saves every output to `rectified_outputs/outputN.jpg`
- All results are exactly 550×425 pixels
- Color preserved from original image

---

## Methods Used

### Binarization
- **Otsu thresholding** (primary)
- **Adaptive Gaussian** (backup when shadows interfere)

### Edge & Contour Detection
- **Canny edge detector**
- `cv2.findContours` to isolate document
- **Ramer–Douglas–Peucker (RDP) simplification** forces shape to 4 dominant points

### Corner Localization
- Corners taken from simplified contour
- Sorted by geometric rules:
  - Smallest `(x + y)` → top-left
  - Largest `(x + y)` → bottom-right

### Homography + Warping
- `cv2.getPerspectiveTransform` forms the 3×3 matrix
- `cv2.warpPerspective` creates rectified output

---

## Parameter Choices

| Component | Parameter | Reason |
|-----------|-----------|--------|
| Gaussian Blur | 5×5 kernel | Smooths noise before edge detection |
| Canny Thresholds | 75 → 200 | Reliable on synthetic dataset |
| Polygon Approx. | ~2–3% perimeter | Produces clean 4-vertex contour |
| Output Resolution | 550×425 px | Required by assignment (50 DPI on 11×8.5") |

All parameters are fixed and identical for all images (automation requirement).

---

## Input & Output Format

### Input
Folder contains `.jpg` / `.png` images, for example:

```
./inputs/
    img1.jpg
    img2.png
    img3.jpg
```

### Output
Automatically created:

```
./rectified_outputs/
    output1.jpg
    output2.jpg
    output3.jpg
```

If a document cannot be detected, the script logs a warning and continues.

---

## Dependencies

- Python 3
- OpenCV
- NumPy

Install with:

```bash
pip install opencv-python numpy
```


