#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path
import cv2
import numpy as np

# fixed output size
OUT_W, OUT_H = 550, 425
# drop tiny contours
MIN_AREA_FRAC = 0.08
# px distance from frame to consider "touching"
BORDER_PAD = 10


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    
    # Sort by y-coordinate to get top and bottom pairs
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_pts = sorted_by_y[:2]
    bottom_pts = sorted_by_y[2:]
    
    # Sort top points by x to get TL and TR
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    tl, tr = top_pts[0], top_pts[1]
    
    # Sort bottom points by x to get BL and BR
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    bl, br = bottom_pts[0], bottom_pts[1]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def touches_border(pts, w, h, pad=BORDER_PAD):
    """Check if any point in the quad touches the border."""
    pts = np.array(pts).reshape(-1, 2)
    return np.any(pts[:, 0] < pad) or np.any(pts[:, 0] > w - pad) or \
           np.any(pts[:, 1] < pad) or np.any(pts[:, 1] > h - pad)


def find_quad_from_contours(contours, w, h, min_area):
    """Try to find best quadrilateral from contours."""
    candidates = []
    
    for contour in contours[:15]:  # Check top 15 largest contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        peri = cv2.arcLength(contour, True)
        
        # Try multiple epsilon values for approximation - TIGHTER values
        for epsilon in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                
                # Skip if touching border
                if touches_border(pts, w, h):
                    continue
                
                ordered = order_quad(pts)
                
                # Calculate angles
                angles = []
                for i in range(4):
                    v1 = ordered[(i + 1) % 4] - ordered[i]
                    v2 = ordered[(i - 1) % 4] - ordered[i]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.abs(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    angles.append(np.degrees(angle))
                
                # Check rectangularity
                if all(50 <= angle <= 130 for angle in angles):
                    # Calculate aspect ratio
                    w1 = np.linalg.norm(ordered[1] - ordered[0])
                    w2 = np.linalg.norm(ordered[2] - ordered[3])
                    h1 = np.linalg.norm(ordered[3] - ordered[0])
                    h2 = np.linalg.norm(ordered[2] - ordered[1])
                    
                    avg_w = (w1 + w2) / 2
                    avg_h = (h1 + h2) / 2
                    aspect = avg_w / (avg_h + 1e-8)
                    
                    # Prefer aspect ratios close to our target (550/425 ≈ 1.29)
                    target_aspect = OUT_W / OUT_H
                    aspect_diff = abs(aspect - target_aspect)
                    
                    # Score: prioritize good angles and aspect ratio, then area
                    angle_deviation = sum(abs(90 - a) for a in angles)
                    score = area * 0.5 - (angle_deviation * 150) - (aspect_diff * 5000)
                    
                    candidates.append((score, ordered, epsilon))
                    break  # Found good quad for this contour
    
    if candidates:
        # Return best scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    return None


def refine_corners(gray, corners):
    """Refine corner positions to subpixel accuracy."""
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Convert to the format cornerSubPix expects
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    
    # Refine corners
    refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    
    return refined.reshape(4, 2)


def detect_document_quad(img):
    """Return 4×2 float32 points for the best document quad, or None."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    min_area = MIN_AREA_FRAC * w * h
    best_quad = None
    
    # Strategy 1: Color-based segmentation (works well for white/light documents)
    # Convert to LAB color space and threshold on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Documents are usually lighter than background
    _, binary = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up with morphology - LESS aggressive to preserve edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Erode slightly to shrink back to true edges
    morph = cv2.erode(morph, np.ones((3, 3), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        best_quad = find_quad_from_contours(contours, w, h, min_area)
    
    # Strategy 2: Edge detection with aggressive preprocessing
    if best_quad is None:
        # Bilateral filter to smooth texture while keeping edges
        bilateral = cv2.bilateralFilter(gray, 11, 60, 60)
        blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        # Erode to pull back to actual edges
        dilated = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            best_quad = find_quad_from_contours(contours, w, h, min_area)
    
    # Strategy 3: Adaptive threshold
    if best_quad is None:
        block_size = max(11, (min(h, w) // 25) | 1)  # Must be odd
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.erode(morph, np.ones((3, 3), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            best_quad = find_quad_from_contours(contours, w, h, min_area)
    
    # Strategy 4: Simple global threshold (for high contrast cases)
    if best_quad is None:
        _, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(simple_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.erode(morph, np.ones((3, 3), np.uint8), iterations=1)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            best_quad = find_quad_from_contours(contours, w, h, min_area * 0.7)
    
    # Fallback: use largest contour with minAreaRect
    if best_quad is None and 'contours' in locals() and contours:
        largest = contours[0]
        if cv2.contourArea(largest) >= min_area * 0.6:  # More lenient
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            best_quad = order_quad(box.astype(np.float32))
    
    # Refine corners to subpixel accuracy
    if best_quad is not None:
        try:
            best_quad = refine_corners(gray, best_quad)
        except:
            pass  # If refinement fails, use original corners
    
    return best_quad


def warp_document(img):
    """Detect and rectify the document; returns rectified image or None if fails."""
    quad = detect_document_quad(img)
    if quad is None:
        return None

    src = quad
    
    # Apply slight inward bias to avoid edge artifacts (shrink by 1-2 pixels)
    center = src.mean(axis=0)
    src_adjusted = src + (center - src) * 0.01  # Move 1% inward
    
    dst = np.array([[0, 0],
                    [OUT_W - 1, 0],
                    [OUT_W - 1, OUT_H - 1],
                    [0, OUT_H - 1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_adjusted, dst)
    rectified = cv2.warpPerspective(img, H, (OUT_W, OUT_H), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return rectified


def process_folder(in_dir: Path, out_dir: Path):
    ensure_dir(out_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"No images found in {in_dir}")
        return

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skipping unreadable file: {p.name}")
            continue
        # Extract number from "input (5)" -> "5"
        match = re.search(r'\((\d+)\)', p.stem)
        num = match.group(1) if match else p.stem
        
        rectified = warp_document(img)
        if rectified is None:
            print(f"[WARN] Could not find a document in: {p.name}")
            cv2.imwrite(str(out_dir / f"FAILED_{num}.jpg"), img)
            continue
        out_path = out_dir / f"output ({num}).jpg"
        cv2.imwrite(str(out_path), rectified)
        print(f"Wrote {out_path.name}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 rectify.py <folder_with_inputs>")
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input folder not found: {in_dir}")
        sys.exit(1)
    out_dir = Path.cwd() / "outputs"
    process_folder(in_dir, out_dir)


if __name__ == "__main__":
    main()
