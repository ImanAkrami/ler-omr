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

# Question box contour detection
RECT_MIN_W_FRAC_IMG = 0.26   # min width relative to image width
RECT_MAX_W_FRAC_IMG = 0.70   # max width relative to image width (keeps us inside a column)
RECT_MIN_H_FRAC_IMG = 0.012  # min height relative to image height
RECT_MAX_H_FRAC_IMG = 0.20   # max height relative to image height (supports long open questions)
RECT_ASPECT_MIN = 3.0        # w/h lower bound
RECT_ASPECT_MAX = 25.0       # w/h upper bound
RECT_CLOSE_K = (5, 5)        # closing kernel before contour detection
RECT_Y_MERGE_GAP_PX = 6      # merge close rectangles
RECT_MIN_PER_COL = 18        # if fewer than this, try helper/horiz fallback

# Helper bar (thick right strip inside each question)
BAR_BAND_FRAC = 0.18         # portion (from the right of a column) to search for the helper bar
BAR_MIN_W_FRAC = 0.01        # minimum helper bar width relative to column width
BAR_MIN_H_FRAC = 0.012       # minimum helper bar height relative to column height
BAR_MIN_AREA_FRAC = 0.00025  # min area relative to column area
BAR_MERGE_GAP_FRAC = 0.004   # merge neighbouring helper fragments separated by small gaps
BAR_CORE_W_FRAC = 0.32       # keep only the densest vertical slice inside the helper band
BAR_Y_THRESH_FRAC = 0.08     # threshold relative to peak y-projection
BAR_CLOSE_KH = 13            # vertical closing kernel to seal tiny holes inside the bar

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


def merge_spans(spans: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s[0])
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def merge_boxes_vert(boxes: List[Tuple[int, int, int, int]], max_gap: int) -> List[Tuple[int, int, int, int]]:
    """
    Merge rectangles that are vertically touching/close and largely aligned horizontally.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = [list(boxes[0])]
    for b in boxes[1:]:
        last = merged[-1]
        # vertical proximity + reasonable horizontal overlap
        if b[1] - last[3] <= max_gap and not (b[2] < last[0] or b[0] > last[2]):
            last[0] = min(last[0], b[0])
            last[1] = min(last[1], b[1])
            last[2] = max(last[2], b[2])
            last[3] = max(last[3], b[3])
        else:
            merged.append(list(b))
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in merged]


def helper_bar_spans(
    bin_ink: np.ndarray,
    col_x0: int,
    col_x1: int,
    y0: int,
    y1: int
) -> Tuple[List[Tuple[int, int]], Dict, np.ndarray]:
    """
    Locate helper bar components inside the right-side band of a column.
    Returns (spans, debug_info, mask_on_full_image).
    """
    h, w = bin_ink.shape[:2]
    col_x0 = max(0, min(col_x0, w - 1))
    col_x1 = max(0, min(col_x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))

    col = bin_ink[y0:y1, col_x0:col_x1]
    col_h, col_w = col.shape[:2]
    band_w = max(8, int(col_w * BAR_BAND_FRAC))
    band_x0 = max(0, col_w - band_w)
    band = col[:, band_x0:]

    # Clean minor holes inside the thick bar without merging neighbouring rows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    band_clean = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Focus on the densest vertical slice of the band to isolate the helper bar
    x_proj = band_clean.sum(axis=0).astype(np.float32)
    peak_x_rel = int(np.argmax(x_proj)) if x_proj.size else 0
    core_w = max(max(3, int(col_w * BAR_MIN_W_FRAC)), int(band_w * BAR_CORE_W_FRAC))
    core_x0 = max(0, min(peak_x_rel - core_w // 2, band_w - core_w))
    core_x1 = core_x0 + core_w

    core = np.zeros_like(band_clean)
    core[:, core_x0:core_x1] = band_clean[:, core_x0:core_x1]

    # Close vertically to bridge tiny gaps and ink holes
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, BAR_CLOSE_KH))
    core_closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, close_k, iterations=1)

    y_proj = core_closed.sum(axis=1).astype(np.float32)
    spans: List[Tuple[int, int]] = []
    kept_components: List[Dict] = []

    if y_proj.size:
        thr = y_proj.max() * BAR_Y_THRESH_FRAC
        min_h = max(12, int(col_h * BAR_MIN_H_FRAC))

        in_run = False
        start = 0
        for i, v in enumerate(y_proj):
            if v > thr and not in_run:
                in_run = True
                start = i
            elif v <= thr and in_run:
                end = i
                if end - start >= min_h:
                    spans.append((start + y0, end + y0))
                in_run = False
        if in_run:
            end = len(y_proj) - 1
            if end - start >= min_h:
                spans.append((start + y0, end + y0))

        merge_gap = max(6, int((y1 - y0) * BAR_MERGE_GAP_FRAC))
        spans = merge_spans(spans, merge_gap)

        for sy0, sy1 in spans:
            h_box = sy1 - sy0
            area_est = int(y_proj[max(0, sy0 - y0):max(0, sy1 - y0)].sum() / max(1, core_w))
            kept_components.append(
                {
                    "y0": int(sy0),
                    "y1": int(sy1),
                    "h": int(h_box),
                    "w": int(core_w),
                    "area_est": int(area_est),
                }
            )
    else:
        merge_gap = max(6, int((y1 - y0) * BAR_MERGE_GAP_FRAC))

    mask_full = np.zeros_like(bin_ink, dtype=np.uint8)
    mask_full[y0:y1, col_x0 + band_x0 + core_x0:col_x0 + band_x0 + core_x1] = core_closed[:, core_x0:core_x1]

    debug = {
        "band_x0": int(col_x0 + band_x0),
        "band_w": int(band_w),
        "core_x0": int(core_x0),
        "core_w": int(core_w),
        "peak_x_rel": int(peak_x_rel),
        "y_threshold": float(y_proj.max() * BAR_Y_THRESH_FRAC) if y_proj.size else 0.0,
        "merge_gap": int(merge_gap),
        "candidates": kept_components,
        "spans": [(int(a), int(b)) for a, b in spans],
    }
    return spans, debug, mask_full


def rect_candidates(
    bin_ink: np.ndarray, y0: int, y1: int
) -> Tuple[List[Tuple[int, int, int, int]], Dict, np.ndarray]:
    """
    Detect rectangular question boxes over the whole sheet.
    Returns (boxes, debug_info, mask).
    """
    h, w = bin_ink.shape[:2]
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    roi = bin_ink[y0:y1, :]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECT_CLOSE_K)
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    min_w = int(w * RECT_MIN_W_FRAC_IMG)
    max_w = int(w * RECT_MAX_W_FRAC_IMG)
    min_h = int(h * RECT_MIN_H_FRAC_IMG)
    max_h = int(h * RECT_MAX_H_FRAC_IMG)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        y_abs = y + y0
        if bw < min_w or bw > max_w or bh < min_h or bh > max_h:
            continue
        aspect = bw / max(bh, 1)
        if aspect < RECT_ASPECT_MIN or aspect > RECT_ASPECT_MAX:
            continue
        boxes.append((x, y_abs, x + bw, y_abs + bh))

    boxes = merge_boxes_vert(boxes, RECT_Y_MERGE_GAP_PX)

    mask = np.zeros_like(bin_ink, dtype=np.uint8)
    for b in boxes:
        cv2.rectangle(mask, (b[0], b[1]), (b[2], b[3]), 255, 1)

    dbg = {
        "min_w": int(min_w),
        "max_w": int(max_w),
        "min_h": int(min_h),
        "max_h": int(max_h),
        "aspect_min": RECT_ASPECT_MIN,
        "aspect_max": RECT_ASPECT_MAX,
        "merge_gap_px": RECT_Y_MERGE_GAP_PX,
        "candidate_count": len(boxes),
    }
    return boxes, dbg, mask


def rect_row_spans(
    bin_ink: np.ndarray,
    col_x0: int,
    col_x1: int,
    y0: int,
    y1: int,
    img_w: int,
    img_h: int,
) -> Tuple[List[Tuple[int, int, int, int]], Dict, np.ndarray]:
    """
    Detect question rows via rectangular contour boxes inside a column ROI.
    Returns (boxes, debug_info, mask_on_full_image).
    """
    h, w = bin_ink.shape[:2]
    col_x0 = max(0, min(col_x0, w - 1))
    col_x1 = max(0, min(col_x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))

    roi = bin_ink[y0:y1, col_x0:col_x1]
    col_w = col_x1 - col_x0
    min_w = max(10, int(col_w * RECT_MIN_W_FRAC))
    max_w = int(col_w * RECT_MAX_W_FRAC)
    min_h = max(8, int(img_h * RECT_MIN_H_FRAC))
    max_h = int(img_h * RECT_MAX_H_FRAC)

    # Slightly close to connect gaps on the rectangle strokes
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand_boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < min_w or ww > max_w or hh < min_h or hh > max_h:
            continue
        aspect = ww / max(hh, 1)
        if aspect < RECT_ASPECT_MIN or aspect > RECT_ASPECT_MAX:
            continue
        # Keep only rectangles that sit well inside the ROI (avoid header line)
        cand_boxes.append((col_x0 + x, y0 + y, col_x0 + x + ww, y0 + y + hh))

    # Merge vertically-close candidates
    cand_boxes = sorted(cand_boxes, key=lambda b: (b[1], b[0]))
    merged: List[Tuple[int, int, int, int]] = []
    merge_gap = max(4, int(img_h * RECT_MERGE_GAP_FRAC))
    for b in cand_boxes:
        if not merged:
            merged.append(list(b))
            continue
        last = merged[-1]
        if b[1] - last[3] <= merge_gap and abs(b[0] - last[0]) < col_w * 0.15:
            last[0] = min(last[0], b[0])
            last[2] = max(last[2], b[2])
            last[3] = max(last[3], b[3])
        else:
            merged.append(list(b))

    merged_boxes = [(int(a), int(b), int(c), int(d)) for a, b, c, d in merged]

    # Build mask for debug
    mask = np.zeros_like(bin_ink, dtype=np.uint8)
    for (x0, y0b, x1, y1b) in merged_boxes:
        cv2.rectangle(mask, (x0, y0b), (x1, y1b), 255, 1)

    debug = {
        "min_w": int(min_w),
        "max_w": int(max_w),
        "min_h": int(min_h),
        "max_h": int(max_h),
        "aspect_min": RECT_ASPECT_MIN,
        "aspect_max": RECT_ASPECT_MAX,
        "merge_gap": int(merge_gap),
        "candidates": [list(map(int, b)) for b in cand_boxes],
        "merged": [list(map(int, b)) for b in merged_boxes],
    }
    return merged_boxes, debug, mask

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


def boxes_from_spans(
    spans: List[Tuple[int, int]],
    col_x0: int,
    col_x1: int,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    col_x0 = max(0, min(col_x0, img_w - 1))
    col_x1 = max(0, min(col_x1, img_w))
    boxes: List[Tuple[int, int, int, int]] = []
    min_h = int(img_h * MIN_ROW_H_FRAC)

    for y0, y1 in spans:
        y0 = max(0, min(y0, img_h - 1))
        y1 = max(0, min(y1, img_h))
        if y1 - y0 < min_h:
            continue
        inset_y = int((y1 - y0) * ROW_INSET_Y_FRAC)
        inset_x = int((col_x1 - col_x0) * ROW_INSET_X_FRAC)

        ay0 = y0 + inset_y
        ay1 = y1 - inset_y
        ax0 = col_x0 + inset_x
        ax1 = col_x1 - inset_x

        if ay1 > ay0 and ax1 > ax0:
            boxes.append((ax0, ay0, ax1, ay1))
    return boxes


def cluster_columns_from_boxes(boxes: List[Tuple[int, int, int, int]], img_w: int) -> Tuple[List[int], Dict]:
    """
    Cluster candidate rectangles into two columns based on x-center using k-means (k=2).
    Returns labels per box and debug info.
    """
    if not boxes:
        return [], {"labels": [], "centers": []}

    pts = np.float32([[ (b[0] + b[2]) * 0.5 ] for b in boxes])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    compactness, labels, centers = cv2.kmeans(pts, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten().tolist()
    centers = centers.flatten().tolist()
    # sort so column 0 = left
    if centers[0] > centers[1]:
        centers = centers[::-1]
        labels = [1 - l for l in labels]

    dbg = {"compactness": float(compactness), "centers": [float(c) for c in centers], "labels": labels}
    return labels, dbg


def filter_and_sort_boxes_by_column(
    boxes: List[Tuple[int, int, int, int]],
    labels: List[int],
    col_idx: int,
    header_y: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    col_boxes = [b for b, l in zip(boxes, labels) if l == col_idx and b[1] >= header_y]
    col_boxes.sort(key=lambda b: b[1])
    # drop duplicates/overlaps too close vertically
    cleaned: List[Tuple[int, int, int, int]] = []
    min_gap = max(4, int(img_h * 0.004))
    for b in col_boxes:
        if cleaned and b[1] - cleaned[-1][1] < min_gap and b[3] - cleaned[-1][3] < min_gap:
            continue
        cleaned.append(b)
    return cleaned


def detect_rows_for_column(
    bin_ink: np.ndarray,
    horiz: np.ndarray,
    col_bounds: Tuple[int, int],
    y0: int,
    y1: int,
    img_w: int,
    img_h: int,
    rect_boxes: List[Tuple[int, int, int, int]],
    rect_labels: List[int],
    col_idx: int,
) -> Tuple[List[Tuple[int, int, int, int]], Dict, np.ndarray]:
    """
    Detect row boxes for a single column given precomputed rectangle candidates + labels.
    Falls back to helper bar spans, then horizontal lines.
    """
    rect_in_col = filter_and_sort_boxes_by_column(rect_boxes, rect_labels, col_idx, y0, img_h)
    rect_mask = np.zeros_like(bin_ink, dtype=np.uint8)
    for b in rect_in_col:
        cv2.rectangle(rect_mask, (b[0], b[1]), (b[2], b[3]), 255, 1)

    if len(rect_in_col) >= RECT_MIN_PER_COL:
        info = {
            "method": "rectangles",
            "rect_debug": {"selected": [list(map(int, b)) for b in rect_in_col]},
            "count": len(rect_in_col),
        }
        return rect_in_col, info, rect_mask

    spans, bar_dbg, bar_mask = helper_bar_spans(bin_ink, col_bounds[0], col_bounds[1], y0, y1)
    if spans:
        boxes = boxes_from_spans(spans, col_bounds[0], col_bounds[1], img_w, img_h)
        info = {
            "method": "helper_bar",
            "bar_debug": bar_dbg,
            "count": len(boxes),
        }
        return boxes, info, bar_mask

    boxes = build_row_boxes(horiz, col_bounds[0], col_bounds[1], y0, y1 - 1, img_w, img_h)
    info = {
        "method": "horizontal_fallback",
        "count": len(boxes),
        "fallback": {
            "row_peak_thresh": ROW_PEAK_THRESH,
            "row_cluster_tol_px": ROW_CLUSTER_TOL_PX,
            "min_row_h_frac": MIN_ROW_H_FRAC,
        },
    }
    return boxes, info, np.zeros_like(bin_ink, dtype=np.uint8)

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

    # 5) Rectangle candidates across the page + column clustering
    rect_boxes, rect_dbg, rect_mask = rect_candidates(bin_ink, header_y, h - 1)
    rect_labels, cluster_dbg = cluster_columns_from_boxes(rect_boxes, w)

    # 6) Row boxes per column (rectangles preferred, helper/horiz fallback)
    rows_L, info_L, mask_L = detect_rows_for_column(
        bin_ink, horiz, colL, header_y, h - 1, w, h, rect_boxes, rect_labels, 0
    )
    rows_R, info_R, mask_R = detect_rows_for_column(
        bin_ink, horiz, colR, header_y, h - 1, w, h, rect_boxes, rect_labels, 1
    )

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

    # 7) Debug overlay and helper mask
    overlay = img.copy()
    cv2.line(overlay, (center_x, 0), (center_x, h - 1), (0, 255, 0), 2)
    cv2.line(overlay, (right_x, 0), (right_x, h - 1), (0, 200, 255), 2)
    cv2.line(overlay, (0, header_y), (w - 1, header_y), (0, 0, 255), 2)
    cv2.rectangle(overlay, (colL[0], header_y), (colL[1], h - 2), (255, 0, 0), 2)
    cv2.rectangle(overlay, (colR[0], header_y), (colR[1], h - 2), (255, 0, 0), 2)

    if info_L["method"] == "helper_bar":
        for y0, y1 in info_L["bar_debug"]["spans"]:
            cv2.rectangle(overlay, (info_L["bar_debug"]["band_x0"], y0), (colL[1] - 1, y1), (0, 140, 255), 1)
    if info_L["method"] == "rectangles":
        for x0, y0, x1, y1 in info_L["rect_debug"]["selected"]:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (120, 200, 255), 1)
    for b in rows_L:
        cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 1)
    if info_R["method"] == "helper_bar":
        for y0, y1 in info_R["bar_debug"]["spans"]:
            cv2.rectangle(overlay, (info_R["bar_debug"]["band_x0"], y0), (colR[1] - 1, y1), (0, 255, 140), 1)
    if info_R["method"] == "rectangles":
        for x0, y0, x1, y1 in info_R["rect_debug"]["selected"]:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (120, 255, 200), 1)
    for b in rows_R:
        cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)

    write_png(out_d / "05_overlay.png", overlay)
    helper_mask = cv2.bitwise_or(mask_L, mask_R)
    write_png(out_d / "06_helper_bars.png", helper_mask)
    write_png(out_d / "07_rect_candidates.png", rect_mask)

    # 8) Save JSON debug
    dbg = {
        "image": {"w": w, "h": h},
        "binarize": bin_info,
        "separators": {"center_x": center_x, "right_x": right_x, "header_y": header_y},
        "columns": {"L": list(map(int, colL)), "R": list(map(int, colR))},
        "counts": {"rows_L": len(rows_L), "rows_R": len(rows_R)},
        "row_detection": {
            "rect_candidates": rect_dbg,
            "cluster": cluster_dbg,
            "L": info_L,
            "R": info_R,
        },
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
