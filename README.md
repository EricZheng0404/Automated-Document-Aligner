# Automated Document Aligner - Midterm Project

## Overview
This project implements a robust automated document rectification pipeline that detects documents in images with complex backgrounds and warps them to a canonical frontal view.

## High-Level Pipeline Description

The rectification pipeline performs the following sequence of operations:

1. **Preprocessing**: Convert input to grayscale and LAB color space; apply bilateral filtering
2. **Binarization**: Apply cascading detection strategies (LAB+Otsu → Canny edges → Adaptive threshold → Global threshold)
3. **Morphological Operations**: Apply closing to connect gaps, then erosion to shrink back to true edges
4. **Contour Detection**: Extract external contours and filter by area and border proximity
5. **Polygon Approximation**: Try multiple epsilon values to find 4-sided quadrilaterals
6. **Validation & Scoring**: Score candidates by area, corner angles, and aspect ratio
7. **Subpixel Refinement**: Refine corner positions to subpixel accuracy using `cv2.cornerSubPix`
8. **Corner Ordering**: Order corners as Top-Left, Top-Right, Bottom-Right, Bottom-Left
9. **Warping**: Apply 1% inward bias, compute homography, and warp with bicubic interpolation

## Specific Methods Chosen

### Binarization

**Multi-Strategy Cascading Approach** - Each strategy only runs if previous ones fail:

1. **LAB Color Space + Otsu's Thresholding**
   - Convert to LAB color space and extract L (lightness) channel
   - Apply automatic Otsu thresholding: `cv2.threshold(l_channel, 0, 255, THRESH_BINARY + THRESH_OTSU)`
   - **Why**: Ignores color variations in textured backgrounds (e.g., green/yellow grass) and focuses only on brightness differences

2. **Bilateral Filter + Canny Edge Detection**
   - Bilateral filter parameters: diameter=11, sigmaColor=60, sigmaSpace=60
   - Canny thresholds: lower=40, upper=120
   - **Why**: Smooths texture noise while preserving document edges; good for clear boundaries

3. **Adaptive Gaussian Thresholding**
   - Block size: dynamically computed as ~4% of image dimension (minimum 11, must be odd)
   - Constant: 10
   - **Why**: Handles non-uniform lighting and gradient backgrounds

4. **Simple Global Thresholding**
   - Fixed threshold: 127
   - **Why**: Fast fallback for high-contrast cases

### Edge/Contour Finding

**Method**: `cv2.findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`

**Process**:
1. Apply morphological closing (5×5 kernel, 1 iteration) to connect broken edges
2. Apply erosion (3×3 kernel, 1 iteration) to shrink back to actual document edges
3. Extract only external contours (avoids nested shapes from texture)
4. Sort contours by area (largest first)
5. Filter out contours that:
   - Are smaller than 8% of image area
   - Touch the image border (within 10 pixels)
   - Cannot be approximated to 4 vertices

**Why**: 
- `RETR_EXTERNAL` prevents false positives from internal texture/text
- Dilate-then-erode balances edge connectivity with precision
- Border rejection prevents selecting the image frame

### Corner Localization

**Method**: Multi-Epsilon Douglas-Peucker Approximation + Subpixel Refinement + Scoring

**Polygon Approximation**:
- Try multiple epsilon values: [0.005, 0.008, 0.01, 0.012, 0.015, 0.02] (fraction of perimeter)
- Start with tightest approximation for better accuracy
- Accept first 4-vertex approximation that passes validation

**Validation**:
- Corner angles must be between 50° and 130°
- Quadrilateral must not touch border
- Must be convex

**Scoring Formula**:
```
score = (area × 0.5) - (angle_deviation × 150) - (aspect_diff × 5000)
```
- `angle_deviation`: Sum of |90° - actual_angle| for all 4 corners
- `aspect_diff`: Absolute difference from target aspect ratio (550/425 = 1.294)
- Select highest-scoring candidate

**Subpixel Refinement**:
- Apply `cv2.cornerSubPix` with 5×5 window, 30 max iterations, 0.001 epsilon
- Achieves corner accuracy within ~0.1 pixels

**Corner Ordering**:
- Sort points by y-coordinate to separate top and bottom pairs
- Sort each pair by x-coordinate to get left and right

**Inward Bias**:
- Move each corner 1% toward document center: `corner_new = corner + (center - corner) × 0.01`
- Prevents including shadows and edge artifacts

**Why**:
- Tighter epsilon values (0.5%-2% vs typical 2%) capture corners more accurately
- Composite scoring prioritizes rectangularity and correct aspect ratio over just size
- Subpixel refinement eliminates residual distortion
- Inward bias excludes background transitions from final output

## Key Parameter Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Output Dimensions** | | |
| `OUT_W` | 550 | Output width in pixels |
| `OUT_H` | 425 | Output height (aspect ratio: 1.294) |
| **Filtering** | | |
| `MIN_AREA_FRAC` | 0.08 | Minimum contour area (8% of image) |
| `BORDER_PAD` | 10 | Border exclusion distance (pixels) |
| **Bilateral Filter** | | |
| Diameter | 11 | Neighborhood size |
| Sigma color/space | 60 / 60 | Filter strength |
| **Canny Edge Detection** | | |
| Lower threshold | 40 | Minimum edge strength |
| Upper threshold | 120 | Strong edge threshold |
| **Morphological Operations** | | |
| Closing kernel | 5×5 | Rectangle kernel |
| Closing iterations | 1 | Single pass |
| Erosion kernel | 3×3 | Shrink kernel |
| Erosion iterations | 1 | Pull back to edges |
| **Adaptive Threshold** | | |
| Block size | `max(11, (min(h,w)//25)\|1)` | Dynamic, ~4% of image (odd) |
| Constant C | 10 | Offset from mean |
| **Polygon Approximation** | | |
| Epsilon values | [0.005, 0.008, 0.01, 0.012, 0.015, 0.02] | Fraction of perimeter |
| Angle tolerance | 50° - 130° | Valid corner angles |
| **Scoring Weights** | | |
| Area weight | 0.5 | Normalized area factor |
| Angle penalty | 150 | Per degree from 90° |
| Aspect ratio penalty | 5000 | Per unit from target |
| **Subpixel Refinement** | | |
| Window size | 5×5 | Search neighborhood |
| Max iterations | 30 | Convergence limit |
| Epsilon | 0.001 | Convergence threshold |
| **Perspective Transform** | | |
| Inward bias | 0.01 (1%) | Corner adjustment factor |
| Interpolation | `INTER_CUBIC` | Bicubic for smoothness |
| Border mode | `BORDER_REPLICATE` | Edge pixel handling |

## Usage

```bash
python3 rectify.py <input_folder>
```

Outputs will be saved to `./outputs/` directory.

## Dependencies
```bash
pip install opencv-python numpy
```
