# Automated Document Aligner - Midterm Project

## Overview
This project implements an automated document rectification pipeline that detects documents in images with complex backgrounds and warps them to a canonical frontal view. The pipeline handles various challenging scenarios including textured backgrounds, varying lighting conditions, and documents at arbitrary orientations.

## High-Level Pipeline Description

The rectification pipeline follows this sequence of operations:

1. **Preprocessing & Contrast Normalization**
   - Convert input image to grayscale
   - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for robust contrast enhancement
   - Apply Gaussian blur to reduce noise while preserving edges

2. **Dual-Path Binarization**
   - **Path A (Edge-based)**: Generate edge map using adaptive Canny edge detection
   - **Path B (Region-based)**: Generate binary mask using adaptive thresholding

3. **Morphological Operations**
   - Apply closing operation to connect gaps in edges/regions
   - Perform dilation to strengthen detected features
   - Kernel size is dynamically scaled to 1% of image dimensions

4. **Contour Detection & Filtering**
   - Extract contours from both binary paths
   - Filter out contours that:
     - Are too small (< 5% of image area)
     - Touch the image border (within 5px)
     - Don't approximate to 4-sided polygons

5. **Corner Localization & Candidate Scoring**
   - Approximate each contour to a quadrilateral using Douglas-Peucker algorithm
   - Score each candidate based on:
     - Normalized area (larger is better)
     - Rectangularity (how close corner angles are to 90°)
   - Select the highest-scoring quadrilateral

6. **Fallback Strategy**
   - If no valid quad found, use minAreaRect on largest contour (still avoiding border-touching)
   - As last resort, use the absolute largest contour

7. **Perspective Transformation**
   - Order detected corners as: Top-Left, Top-Right, Bottom-Right, Bottom-Left
   - Compute homography to map to fixed output dimensions (550×425)
   - Warp using bicubic interpolation for smooth results

## Methods Chosen

### Binarization
**Primary Methods (Dual-Path Approach):**

1. **Adaptive Canny Edge Detection** (`auto_canny`)
   - Automatically computes thresholds based on image median
   - Lower threshold: `max(0, (1 - σ) × median)`
   - Upper threshold: `min(255, (1 + σ) × median)`
   - Rationale: Adapts to varying lighting conditions without manual tuning

2. **Adaptive Threshold** (`cv2.adaptiveThreshold`)
   - Method: Gaussian-weighted mean
   - Inverted binary mode (document interior becomes white)
   - Block size: Dynamically computed as `max(15, (min(h,w) // 20) // 2 * 2 + 1)`
   - Constant: 5
   - Rationale: Handles textured/gradient backgrounds where global thresholding fails

**Why Dual-Path?**
- Edge detection excels with clean backgrounds and strong document boundaries
- Adaptive thresholding handles complex backgrounds with texture/patterns
- Combining both increases robustness across diverse input conditions

### Edge/Contour Finding
**Method:** `cv2.findContours` with `RETR_LIST` and `CHAIN_APPROX_SIMPLE`

**Process:**
1. Apply morphological closing with dynamically-sized kernel (1% of image dimensions)
2. Apply single-iteration dilation with 3×3 kernel
3. Extract all contours without hierarchy (`RETR_LIST`)
4. Filter contours by:
   - Minimum area threshold (5% of image area)
   - Border proximity test (rejects if within 5px of edge)
   - Perimeter-based approximation to 4 vertices

**Rationale:**
- Morphological operations connect broken edges while preserving shape
- Border rejection prevents false positives from image frame
- Simple chain approximation reduces memory while maintaining shape accuracy

### Corner Localization
**Method:** Douglas-Peucker Polygon Approximation (`cv2.approxPolyDP`)

**Parameters:**
- Epsilon: `0.02 × perimeter`
- Closed contour: True

**Scoring System:**
Each 4-point candidate receives a composite score:
```
score = (area / image_area) + 0.5 × rectangularity_score
```

**Rectangularity Computation:**
- For each corner, compute vectors to adjacent corners
- Calculate absolute dot product of normalized vectors (cosine of angle)
- Average all four corner cosines
- Score = `1.0 - mean(cosines)` (closer to 1.0 = more rectangular)

**Corner Ordering:**
After selection, corners are consistently ordered as TL, TR, BR, BL using:
- TL: minimum sum of coordinates (x + y)
- BR: maximum sum of coordinates
- TR: minimum difference (y - x)
- BL: maximum difference (y - x)

**Rationale:**
- 2% epsilon balances shape fidelity with noise tolerance
- Composite scoring prefers large, rectangular regions (typical documents)
- Geometric corner ordering ensures correct perspective transform regardless of input orientation

## Key Parameter Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| `OUT_W` | 550 | Output width in pixels |
| `OUT_H` | 425 | Output height in pixels |
| `MIN_AREA_FRAC` | 0.05 | Minimum contour area as fraction of image (5%) |
| `CANNY_SIGMA` | 0.33 | Sensitivity for auto-Canny thresholding |
| `BORDER_PAD` | 5 | Pixel distance from edge to consider "border-touching" |
| `CLAHE clipLimit` | 2.0 | Contrast limiting for adaptive histogram equalization |
| `CLAHE tileGridSize` | (8, 8) | Grid size for localized contrast enhancement |
| `Gaussian kernel` | (5, 5) | Blur kernel size for noise reduction |
| `Morph kernel size` | `max(3, int(min(w,h) × 0.01) \| 1)` | Dynamically scaled to 1% of image dimension (odd) |
| `approxPolyDP epsilon` | `0.02 × perimeter` | Polygon approximation tolerance (2% of perimeter) |
| `Rectangularity weight` | 0.5 | Weight of angle-score in composite scoring |
| `Adaptive threshold block` | `max(15, (min(h,w) // 20) // 2 * 2 + 1)` | Dynamic block size (≈5% of image, odd, min 15) |
| `Adaptive threshold C` | 5 | Constant subtracted from weighted mean |

## Usage

```bash
python3 rectify.py <input_folder>
```

Outputs will be saved to `./outputs/` with format `output (N).jpg` where N is extracted from the input filename.

### Optional Debug Mode
Set environment variable to save detection overlays:
```bash
DEBUG_DIR=./debug python3 rectify.py <input_folder>
```

## Dependencies
- OpenCV (cv2)
- NumPy
- Python 3.6+

