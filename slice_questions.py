"""
Simple + robust OMR row slicer (two-column) using:
- adaptive binarization with auto polarity selection (prevents "everything white" failures)
- morphological extraction of vertical separators (center + right border)
- morphological extraction of horizontal row lines, then y-projection peaks
- row rectangles = between consecutive horizontal lines inside each column ROI

Outputs:
out/<stem>/questions/colL_row_###.png
out/<stem>/questions/colR_row_###.png
out/<stem>/debug/*.png and overlays + data.json

Usage:
python omr_slice_simple.py --image /path/to/crop.png --out out_dir
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np


# -------------------------
# Tunable parameters
# -------------------------

# Binarization
GAUSS_BLUR_K = 5
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
ADAPTIVE_BLOCK_FRAC = 0.02  # block size ~ 2% of min(h,w), forced odd and >= 31
ADAPTIVE_C = 7              # try 5..15
FG_RATIO_MAX = 0.20         # if foreground ratio too high -> pick the other polarity

# Vertical separator detection
VERT_KERNEL_W_FRAC = 0.006  # thin vertical kernel width ~0.6% of width
VERT_KERNEL_H_FRAC = 0.20   # tall vertical kernel height ~20% of height
CENTER_BAND = (0.30, 0.70)  # search for center divider in this x range
RIGHT_BAND = (0.80, 0.98)   # search for right border in this x range

# Column ROIs
PAGE_MARGIN_FRAC = 0.015
SEP_PAD_FRAC = 0.010        # padding away from center divider to avoid bleed
RIGHT_BORDER_PAD_FRAC = 0.010

# Horizontal row line detection
HORIZ_KERNEL_W_FRAC = 0.18  # wide horizontal kernel ~18% of width
HORIZ_KERNEL_H = 1
ROW_PEAK_THRESH = 0.35      # relative peak threshold
ROW_CLUSTER_TOL_PX = 6       # cluster close peaks
MIN_ROW_H_FRAC = 0.012       # discard ultra tiny rows

# Cropping
NUMBER_STRIP_FRAC = 0.14     # remove question-number strip on right side of each row (inside column)
ROW_INSET_Y_FRAC = 0.02      # small inset inside row boundaries
ROW_INSET_X_FRAC = 0.01

# Debug
PNG_COMPRESSION = 3


# -------------------------
# Helpers
# -------------------------

def _odd_at_least(v: int, mn: int) -> int:
    v = max(mn, v)
    if v % 2 == 0:
        v += 1
    return v

def write_png(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def binarize_auto(gray: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Adaptive threshold in BOTH polarities, then choose the one
    whose foreground ratio is reasonable (not "everything is foreground").
    """
    h, w = gray.shape[:2]
    k = _odd_at_least(int(min(h, w) * ADAPTIVE_BLOCK_FRAC), 31)
    blur = cv2.GaussianBlur(gray, (GAUSS_BLUR_K, GAUSS_BLUR_K), 0)

    bin_inv = cv2.adaptiveThreshold(
        blur, 255, ADAPTIVE_METHOD, cv2.THRESH_BINARY_INV, k, ADAPTIVE_C
    )
    bin_norm = cv2.adaptiveThreshold(
        blur, 255, ADAPTIVE_METHOD, cv2.THRESH_BINARY, k, ADAPTIVE_C
    )

    # We want "ink/lines" = 255 in our working binary.
    # If we use THRESH_BINARY_INV, ink becomes 255. If we use THRESH_BINARY, background becomes 255.
    # So compute a heuristic: ink ratio should be "smallish".
    inv_ratio = float(np.mean(bin_inv > 0))
    # For bin_norm, ink is 0, so invert for ratio:
    norm_ratio = float(np.mean((255 - bin_norm) > 0))

    # Choose the polarity with smaller (more sane) ink ratio, but not too tiny.
    # (Sheets with lots of lines still have ink ratio well below ~0.2)
    pick_inv = True
    if inv_ratio > FG_RATIO_MAX and norm_ratio < inv_ratio:
        pick_inv = False

    if pick_inv:
        chosen = bin_inv
        ink_ratio = inv_ratio
        polarity = "INV"
        # chosen already has ink=255
        work = chosen
    else:
        chosen = bin_norm
        ink_ratio = norm_ratio
        polarity = "NORM"
        # convert to ink=255
        work = 255 - chosen

    info = {
        "adaptive_block": k,
        "adaptive_C": ADAPTIVE_C,
        "inv_ink_ratio": inv_ratio,
        "norm_ink_ratio": norm_ratio,
        "chosen_polarity": polarity,
        "chosen_ink_ratio": ink_ratio,
    }
    return work, info

def morph_vertical(bin_ink: np.ndarray) -> np.ndarray:
    h, w = bin_ink.shape[:2]
    kw = max(1, int(w * VERT_KERNEL_W_FRAC))
    kh = max(15, int(h * VERT_KERNEL_H_FRAC))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    vert = cv2.morphologyEx(bin_ink, cv2.MORPH_OPEN, kernel)
    return vert

def morph_horizontal(bin_ink: np.ndarray) -> np.ndarray:
    h, w = bin_ink.shape[:2]
    kw = max(25, int(w * HORIZ_KERNEL_W_FRAC))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, HORIZ_KERNEL_H))
    horiz = cv2.morphologyEx(bin_ink, cv2.MORPH_OPEN, kernel)
    return horiz

def x_peak(vert: np.ndarray, x0f: float, x1f: float) -> int:
    h, w = vert.shape[:2]
    x0 = int(w * x0f)
    x1 = int(w * x1f)
    proj = vert.sum(axis=0).astype(np.float32)
    band = proj[x0:x1]
    if band.size == 0:
        return w // 2
    idx = int(np.argmax(band))
    return x0 + idx

def find_two_separators(vert: np.ndarray) -> Tuple[int, int, Dict]:
    """
    Returns:
      center_x: center divider x
      right_x: right border x
    """
    h, w = vert.shape[:2]
    center_x = x_peak(vert, CENTER_BAND[0], CENTER_BAND[1])
    right_x = x_peak(vert, RIGHT_BAND[0], RIGHT_BAND[1])

    # safety: if right_x accidentally equals center_x, push right_x near edge
    if abs(right_x - center_x) < w * 0.08:
        right_x = int(w * 0.98)

    info = {"center_x": int(center_x), "right_x": int(right_x)}
    return int(center_x), int(right_x), info

def y_peaks_from_horiz(horiz_roi: np.ndarray) -> List[int]:
    """
    Detect horizontal-line peaks by y-projection + simple peak picking + clustering.
    Returns y indices relative to ROI (not absolute).
    """
    proj = horiz_roi.sum(axis=1).astype(np.float32)
    if proj.max() <= 0:
        return []

    proj_norm = proj / (proj.max() + 1e-6)

    peaks = []
    for y in range(1, len(proj_norm) - 1):
        if proj_norm[y] >= ROW_PEAK_THRESH and proj_norm[y] >= proj_norm[y - 1] and proj_norm[y] >= proj_norm[y + 1]:
            peaks.append(y)

    if not peaks:
        return []

    # cluster
    clustered = [peaks[0]]
    for p in peaks[1:]:
        if p - clustered[-1] <= ROW_CLUSTER_TOL_PX:
            # keep the stronger one
            if proj_norm[p] > proj_norm[clustered[-1]]:
                clustered[-1] = p
        else:
            clustered.append(p)

    return clustered

def build_row_boxes(
    horiz: np.ndarray,
    col_x0: int,
    col_x1: int,
    y0: int,
    y1: int,
    img_w: int,
    img_h: int
) -> List[Tuple[int, int, int, int]]:
    """
    Build row boxes within [y0,y1) using horizontal line peaks.
    """
    col_x0 = max(0, min(col_x0, img_w - 1))
    col_x1 = max(0, min(col_x1, img_w))
    y0 = max(0, min(y0, img_h - 1))
    y1 = max(0, min(y1, img_h))

    roi = horiz[y0:y1, col_x0:col_x1]
    peaks = y_peaks_from_horiz(roi)

    # Need boundaries: include top and bottom
    boundaries = [0] + peaks + [roi.shape[0] - 1]
    boundaries = sorted(set(boundaries))

    boxes: List[Tuple[int, int, int, int]] = []
    min_h = int(img_h * MIN_ROW_H_FRAC)

    for i in range(len(boundaries) - 1):
        yy0 = boundaries[i]
        yy1 = boundaries[i + 1]
        if yy1 - yy0 < min_h:
            continue
        # inset a bit to avoid slicing on the line itself
        inset_y = int((yy1 - yy0) * ROW_INSET_Y_FRAC)
        inset_x = int((col_x1 - col_x0) * ROW_INSET_X_FRAC)

        ax0 = col_x0 + inset_x
        ax1 = col_x1 - inset_x
        ay0 = y0 + yy0 + inset_y
        ay1 = y0 + yy1 - inset_y

        if ay1 > ay0 and ax1 > ax0:
            boxes.append((ax0, ay0, ax1, ay1))

    return boxes

def crop_row(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return img[0:0, 0:0]
    row = img[y0:y1, x0:x1]

    # remove number strip (on the RIGHT side inside each row)
    strip = int(row.shape[1] * NUMBER_STRIP_FRAC)
    if row.shape[1] - strip > 5:
        row = row[:, : row.shape[1] - strip]
    return row


# -------------------------
# Main
# -------------------------

def slice_sheet(image_path: Path, out_root: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    gray = to_gray(img)

    stem = image_path.stem.replace("_crop", "")
    out_dir = out_root / stem
    out_q = out_dir / "questions"
    out_d = out_dir / "debug"
    out_q.mkdir(parents=True, exist_ok=True)
    out_d.mkdir(parents=True, exist_ok=True)

    # 1) Binarize with auto polarity
    bin_ink, bin_info = binarize_auto(gray)
    write_png(out_d / "01_gray.png", gray)
    write_png(out_d / "02_bin_ink.png", bin_ink)

    # 2) Extract vertical + horizontal structure
    vert = morph_vertical(bin_ink)
    horiz = morph_horizontal(bin_ink)
    write_png(out_d / "03_vert.png", vert)
    write_png(out_d / "04_horiz.png", horiz)

    # 3) Find global separators: center divider + right border
    center_x, right_x, sep_info = find_two_separators(vert)

    # 4) Define columns + table vertical span (simple: from just below header line)
    # Header line: strongest horizontal line in top 65%
    top_band = horiz[: int(h * 0.65), :]
    yproj = top_band.sum(axis=1).astype(np.float32)
    header_y = int(np.argmax(yproj)) if yproj.size else int(h * 0.12)
    header_y = min(h - 2, max(0, header_y + 2))

    margin = int(w * PAGE_MARGIN_FRAC)
    sep_pad = int(w * SEP_PAD_FRAC)
    rb_pad = int(w * RIGHT_BORDER_PAD_FRAC)

    colL = (margin, max(margin + 10, center_x - sep_pad))
    colR = (min(w - margin - 10, center_x + sep_pad), max(center_x + sep_pad + 10, right_x - rb_pad))

    # 5) Row boxes per column
    rows_L = build_row_boxes(horiz, colL[0], colL[1], header_y, h - 1, w, h)
    rows_R = build_row_boxes(horiz, colR[0], colR[1], header_y, h - 1, w, h)

    # 6) Export crops
    exported = {"colL": [], "colR": []}
    for i, box in enumerate(rows_L):
        crop = crop_row(img, box)
        if crop.size:
            name = f"colL_row_{i:03d}.png"
            cv2.imwrite(str(out_q / name), crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            exported["colL"].append({"i": i, "box": list(map(int, box)), "file": name})

    for i, box in enumerate(rows_R):
        crop = crop_row(img, box)
        if crop.size:
            name = f"colR_row_{i:03d}.png"
            cv2.imwrite(str(out_q / name), crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            exported["colR"].append({"i": i, "box": list(map(int, box)), "file": name})

    # 7) Debug overlay
    overlay = img.copy()
    cv2.line(overlay, (center_x, 0), (center_x, h - 1), (0, 255, 0), 2)
    cv2.line(overlay, (right_x, 0), (right_x, h - 1), (0, 200, 255), 2)
    cv2.line(overlay, (0, header_y), (w - 1, header_y), (0, 0, 255), 2)
    cv2.rectangle(overlay, (colL[0], header_y), (colL[1], h - 2), (255, 0, 0), 2)
    cv2.rectangle(overlay, (colR[0], header_y), (colR[1], h - 2), (255, 0, 0), 2)

    for b in rows_L:
        cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 1)
    for b in rows_R:
        cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)

    write_png(out_d / "05_overlay.png", overlay)

    # 8) Save JSON debug
    dbg = {
        "image": {"w": w, "h": h},
        "binarize": bin_info,
        "separators": {"center_x": center_x, "right_x": right_x, "header_y": header_y},
        "columns": {"L": list(map(int, colL)), "R": list(map(int, colR))},
        "counts": {"rows_L": len(rows_L), "rows_R": len(rows_R)},
        "exported": exported,
        "tuning_hints": {
            "If 02_bin_ink looks flooded (too white)": [
                "increase ADAPTIVE_C (e.g. 10..15)",
                "increase ADAPTIVE_BLOCK_FRAC (e.g. 0.03)",
                "or lower FG_RATIO_MAX slightly (e.g. 0.15) so polarity flips earlier",
            ],
            "If row splitting misses lines": [
                "lower ROW_PEAK_THRESH (e.g. 0.25)",
                "increase HORIZ_KERNEL_W_FRAC (e.g. 0.22)",
            ],
            "If center/right separators are off": [
                "increase VERT_KERNEL_H_FRAC (e.g. 0.25)",
                "adjust CENTER_BAND/RIGHT_BAND",
            ],
        },
    }
    (out_d / "data.json").write_text(json.dumps(dbg, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    slice_sheet(args.image, args.out)


if __name__ == "__main__":
    main()
