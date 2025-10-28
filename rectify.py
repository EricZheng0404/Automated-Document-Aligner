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
MIN_AREA_FRAC = 0.05
# median heuristic
CANNY_SIGMA = 0.33
# px distance from frame to consider "touching"
BORDER_PAD = 5
# if set, write overlays here
DEBUG_DIR = os.environ.get("DEBUG_DIR")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def auto_canny(img_gray, sigma=CANNY_SIGMA):
    v = np.median(img_gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_gray, lower, upper)


def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def touches_border(bbox, w, h, pad=BORDER_PAD):
    x, y, bw, bh = bbox
    return (x <= pad) or (y <= pad) or (x + bw >= w - pad) or (y + bh >= h - pad)


def rectangularity_score(pts):
    """1.0 when angles ~90°, smaller otherwise."""
    p = np.asarray(pts, np.float32)
    # use polygon order returned by approx; we only need angle coherence
    cos_vals = []
    for i in range(4):
        v1 = p[(i + 1) % 4] - p[i]
        v2 = p[(i - 1) % 4] - p[i]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_vals.append(abs(np.dot(v1, v2) / denom))
    return float(1.0 - np.mean(cos_vals))  # closer to 1 == squarer


def find_quads(binimg, min_area, w, h):
    """Return list of (score, quad_pts) from a binary mask."""
    # Close gaps; kernel scaled to image size but derived algorithmically (no per-image tuning)
    k = max(3, int(round(min(w, h) * 0.01)) | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        bbox = cv2.boundingRect(c)
        if touches_border(bbox, w, h):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        rect_score = rectangularity_score(pts)
        score = area / float(w * h) + 0.5 * rect_score
        results.append((score, pts))
    return results


def detect_document_quad(img):
    """Return 4×2 float32 points for the best document quad, or None."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Contrast normalize: CLAHE is robust on textured backgrounds
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)

    # Branch A: edges
    edges = auto_canny(blur)

    # Branch B: adaptive threshold (inverts so page interior is white→1)
    block = max(15, (min(h, w) // 20) // 2 * 2 + 1)
    th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 5)

    candidates = []
    candidates += find_quads(edges, MIN_AREA_FRAC * w * h, w, h)
    candidates += find_quads(th,    MIN_AREA_FRAC * w * h, w, h)

    if candidates:
        best = max(candidates, key=lambda t: t[0])[1]
        return best

    # Fallback: biggest minAreaRect that doesn't touch the border
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA_FRAC * w * h:
            break
        if touches_border(cv2.boundingRect(c), w, h):
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c))
        return box.astype(np.float32)

    # Worst case: just use the absolute largest contour
    box = cv2.boxPoints(cv2.minAreaRect(cnts[0]))
    return box.astype(np.float32)


def warp_document(img):
    """Detect and rectify the document; returns rectified image or None if fails."""
    quad = detect_document_quad(img)
    if quad is None:
        return None

    src = order_quad(quad)
    dst = np.array([[0, 0],
                    [OUT_W - 1, 0],
                    [OUT_W - 1, OUT_H - 1],
                    [0, OUT_H - 1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(img, H, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)

    if DEBUG_DIR:
        ensure_dir(Path(DEBUG_DIR))
        dbg = img.copy()
        cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite(str(Path(DEBUG_DIR) / "debug_detect.jpg"), dbg)
    return rectified


# ---------------- I/O ----------------
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