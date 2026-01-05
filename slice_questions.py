#!/usr/bin/env python3
"""
Dash-driven OMR row slicer (deterministic, dash-first, snap-to-table-borders)

What we KEEP unchanged:
- Dash detection logic (your tuned LONG_LINE_H_FRAC=0.60 and DASH_MAX_H_FRAC=0.14)
- The "avoid morphology merging into one long line" approach
- Small debug outputs

What we ADD:
1) Cluster dashes into 2 X-columns (spine/right-edge)
2) Build one row ROI per dash using the dash bbox height EXACTLY (y0..y1 = dash bbox)
3) For each row, compute a *coarse* horizontal search band that cannot leak into the other column:
   - If dash is rightmost column: search band starts just right of the spine-dash column and ends just left of this dash
   - If dash is spine column: search band starts at page left and ends just left of this dash
4) Inside that band, snap ROI width to printed table vertical borders by detecting strong vertical strokes
5) Export crops + overlay:
   - debug/dashes_overlay_small.jpg  (kept dashes)
   - debug/dashes_and_rois_overlay_small.jpg (coarse band = purple, snapped ROI = cyan, dash bbox = green)
   - debug/dash_mask_small.png (kept dashes mask)
   - debug/cleaned_small.png, debug/long_lines_small.png
6) Naming:
   - Right column is ALWAYS FIRST (constant RIGHT_COLUMN_IS_FIRST=True)
   - Global numbering starts at 1: Q1-Col-R, Q2-Col-R, ... then continues on left column: Q{n+1}-Col-L...

Notes:
- Height is strictly dash bbox height (no height padding).
- Width snapping uses vertical-line projection within the row band; very constrained so it won't grab the other column.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# =========================
# PARAMETERS (tunable, safe)
# =========================

# Output naming
RIGHT_COLUMN_IS_FIRST = True  # per your requirement

# --- Binarization ---
GAUSS_BLUR = 5
ADAPT_BLOCK_FRAC = 0.02
ADAPT_C = 7
MAX_INK_RATIO = 0.30

# --- Vertical span (ignore header/footer) ---
TOP_IGNORE_FRAC = 0.10
BOTTOM_IGNORE_FRAC = 0.02

# --- Pre-clean (do NOT connect rows) ---
CLOSE_K_FRAC = 0.004
CLOSE_ITERS = 1

# --- Long vertical line removal (prevents "one long line" mask) ---
LONG_LINE_W_FRAC = 0.008
LONG_LINE_H_FRAC = 0.60  # your tuned value
LONG_LINE_ITERS = 1

# --- Dash candidate geometry ---
DASH_MIN_H_FRAC = 0.012
DASH_MAX_H_FRAC = 0.14    # your tuned value
DASH_MIN_W_FRAC = 0.002
DASH_MAX_W_FRAC = 0.050
DASH_AR_MIN = 1.15
DASH_AR_MAX = 20.0
DASH_FILL_MIN = 0.55

# --- ROI building ---
DASH_TO_TABLE_GAP_PX = 4          # crop ends this many px before dash x0
INTER_COL_GAP_PAD_PX = 10         # for right column: start after spine-dash column + this pad
LEFT_PAGE_PAD_PX = 4              # for left column: small pad from page edge
MIN_ROI_WIDTH_FRAC = 0.18         # if snapping fails, ensure a minimum width
MAX_ROI_WIDTH_FRAC = 0.49         # if snapping goes crazy, cap width (per column)

# --- Vertical line snapping inside row band ---
# We detect strong vertical strokes in the coarse band to snap left/right borders.
VERT_OPEN_W_PX = 3
VERT_OPEN_H_FRAC_OF_ROW = 0.80    # kernel height relative to row height
VERT_PEAK_MIN_FRAC = 0.55         # x-projection threshold relative to row height
RIGHT_BORDER_SEARCH_FRAC = 0.22   # search last 22% of band for right border
LEFT_BORDER_SEARCH_FRAC = 0.22    # search first 22% of band for left border

# --- Debug outputs ---
DEBUG_SCALE = 0.35
JPG_QUALITY = 75
PNG_COMPRESSION = 6


# =========================
# HELPERS
# =========================

def odd_at_least(v, mn):
    v = max(mn, int(v))
    return v + 1 if v % 2 == 0 else v


def adaptive_binarize(gray):
    """
    Returns bin_img where ink is 255 and background 0.
    """
    h, w = gray.shape
    k = odd_at_least(min(h, w) * ADAPT_BLOCK_FRAC, 31)
    blur = cv2.GaussianBlur(gray, (GAUSS_BLUR, GAUSS_BLUR), 0)

    bin_inv = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        k, ADAPT_C
    )
    bin_norm = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        k, ADAPT_C
    )

    inv_ratio = float(np.mean(bin_inv > 0))
    norm_ratio = float(np.mean((255 - bin_norm) > 0))

    if inv_ratio <= MAX_INK_RATIO:
        chosen = bin_inv
        polarity = "INV"
        chosen_ratio = inv_ratio
    else:
        chosen = 255 - bin_norm
        polarity = "NORM"
        chosen_ratio = norm_ratio

    info = {
        "block": int(k),
        "C": int(ADAPT_C),
        "polarity": polarity,
        "inv_ink_ratio": inv_ratio,
        "norm_ink_ratio": norm_ratio,
        "chosen_ink_ratio": chosen_ratio,
        "max_ink_ratio": float(MAX_INK_RATIO),
    }
    return chosen, info


# =========================
# DASH DETECTION (UNCHANGED)
# =========================

def detect_dashes(bin_img, y0, y1):
    """
    Detect short, thick, SOLID vertical dashes.
    Returns:
      kept_dashes: list[dict]  (bbox in FULL image coords)
      kept_mask_roi: uint8 mask (ROI size) with kept components
      cleaned_roi: ROI after subtracting long_lines
      long_lines_roi: ROI of extracted long lines
      debug: dict
      rejected: list[dict] (ROI coords)
    """
    h, w = bin_img.shape
    roi = bin_img[y0:y1, :]

    # 1) Light close to fix jagged edges / tiny holes (won't connect rows if kernel is small)
    k_close = max(3, int(min(h, w) * CLOSE_K_FRAC))
    if k_close % 2 == 0:
        k_close += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, close_kernel, iterations=CLOSE_ITERS)

    # 2) Remove long continuous vertical lines (page borders / continuous strokes)
    k_ll_w = max(1, int(w * LONG_LINE_W_FRAC))
    k_ll_h = max(25, int(h * LONG_LINE_H_FRAC))
    long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_ll_w, k_ll_h))
    long_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, long_kernel, iterations=LONG_LINE_ITERS)

    # Subtract long lines
    cleaned = cv2.subtract(closed, long_lines)

    # 3) Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), connectivity=8)

    # thresholds in px (note: based on FULL image size like your version)
    min_h = int(h * DASH_MIN_H_FRAC)
    max_h = int(h * DASH_MAX_H_FRAC)
    min_w = int(w * DASH_MIN_W_FRAC)
    max_w = int(w * DASH_MAX_W_FRAC)

    kept = []
    rejected = []
    kept_mask = np.zeros_like(cleaned, dtype=np.uint8)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if ww <= 0 or hh <= 0:
            continue

        if hh < min_h or hh > max_h:
            rejected.append({"reason": "h", "bbox": [int(x), int(y), int(x + ww), int(y + hh)]})
            continue
        if ww < min_w or ww > max_w:
            rejected.append({"reason": "w", "bbox": [int(x), int(y), int(x + ww), int(y + hh)]})
            continue

        ar = hh / max(ww, 1)
        if ar < DASH_AR_MIN or ar > DASH_AR_MAX:
            rejected.append({"reason": "ar", "bbox": [int(x), int(y), int(x + ww), int(y + hh)], "ar": float(ar)})
            continue

        fill = float(area) / float(ww * hh)
        if fill < DASH_FILL_MIN:
            rejected.append({"reason": "fill", "bbox": [int(x), int(y), int(x + ww), int(y + hh)], "fill": float(fill)})
            continue

        kept_mask[labels == i] = 255
        kept.append({
            "bbox": [int(x), int(y0 + y), int(x + ww), int(y0 + y + hh)],  # FULL coords
            "cx": int(x + ww // 2),
            "cy": int(y0 + y + hh // 2),
            "w": int(ww),
            "h": int(hh),
            "fill": float(fill),
        })

    kept.sort(key=lambda d: (d["cy"], d["cx"]))

    debug = {
        "y_bounds": [int(y0), int(y1)],
        "kernels": {
            "close": [int(k_close), int(k_close)],
            "long_open": [int(k_ll_w), int(k_ll_h)],
        },
        "thresholds": {
            "min_h": int(min_h),
            "max_h": int(max_h),
            "min_w": int(min_w),
            "max_w": int(max_w),
            "ar_min": float(DASH_AR_MIN),
            "ar_max": float(DASH_AR_MAX),
            "fill_min": float(DASH_FILL_MIN),
        },
        "counts": {
            "components": int(num - 1),
            "kept": int(len(kept)),
            "rejected": int(len(rejected)),
        }
    }

    return kept, kept_mask, cleaned, long_lines, debug, rejected


# =========================
# DASH CLUSTERING (2 columns)
# =========================

def cluster_dashes_two_columns(dashes):
    """
    Returns:
      cols: dict {0: [dash...], 1: [dash...]} ordered by x center (0 = left, 1 = right)
      centers: (left_center_x, right_center_x)
      spine_col_index: which column is the "spine" (the left of the two dash columns)
      rightmost_col_index: which is the right edge dash column
    """
    if len(dashes) == 0:
        return {0: [], 1: []}, (0.0, 0.0), 0, 1
    if len(dashes) == 1:
        # can't kmeans; treat as rightmost for safety
        return {0: [], 1: [dashes[0]]}, (0.0, float(dashes[0]["cx"])), 0, 1

    xs = np.array([d["cx"] for d in dashes], dtype=np.float32).reshape(-1, 1)
    _, labels, centers = cv2.kmeans(
        xs, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        10, cv2.KMEANS_PP_CENTERS
    )

    centers = centers.flatten()
    order = np.argsort(centers)  # left then right

    cols = {0: [], 1: []}
    for d, lab in zip(dashes, labels.flatten()):
        # map actual label to ordered 0/1
        ordered_col = int(np.where(order == lab)[0][0])
        cols[ordered_col].append(d)

    cols[0].sort(key=lambda dd: dd["cy"])
    cols[1].sort(key=lambda dd: dd["cy"])

    left_center = float(centers[order[0]])
    right_center = float(centers[order[1]])
    spine_col_index = 0
    rightmost_col_index = 1
    return cols, (left_center, right_center), spine_col_index, rightmost_col_index


# =========================
# ROI WIDTH SNAPPING
# =========================

def _snap_roi_width_to_table(bin_img, y0, y1, x0_coarse, x1_coarse):
    """
    Given fixed row height [y0,y1) and a coarse x band,
    snap to the printed table's vertical borders inside that band.

    Returns (x0_snap, x1_snap, debug_dict)
    """
    h, w = bin_img.shape
    y0 = int(max(0, min(h - 1, y0)))
    y1 = int(max(y0 + 1, min(h, y1)))
    x0 = int(max(0, min(w - 1, x0_coarse)))
    x1 = int(max(x0 + 1, min(w, x1_coarse)))

    row_h = y1 - y0
    band_w = x1 - x0
    if band_w < 10 or row_h < 5:
        return x0, x1, {"snapped": False, "reason": "tiny_band"}

    band = bin_img[y0:y1, x0:x1]

    # Emphasize vertical strokes inside this band
    k_h = max(5, int(row_h * VERT_OPEN_H_FRAC_OF_ROW))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (VERT_OPEN_W_PX, k_h))
    vert = cv2.morphologyEx(band, cv2.MORPH_OPEN, k, iterations=1)

    # X-projection: count of ink pixels per x
    proj = (vert > 0).sum(axis=0).astype(np.int32)
    thresh = int(row_h * VERT_PEAK_MIN_FRAC)

    # Candidate x's where strong vertical line exists
    cand = np.where(proj >= thresh)[0]
    if cand.size == 0:
        return x0, x1, {"snapped": False, "reason": "no_peaks", "thresh": thresh}

    # Choose left border near left edge of band, and right border near right edge of band
    left_lim = int(band_w * LEFT_BORDER_SEARCH_FRAC)
    right_lim = int(band_w * (1.0 - RIGHT_BORDER_SEARCH_FRAC))

    left_cand = cand[cand <= max(1, left_lim)]
    right_cand = cand[cand >= min(band_w - 2, right_lim)]

    if left_cand.size == 0:
        left_x = int(cand.min())
    else:
        left_x = int(left_cand.min())

    if right_cand.size == 0:
        right_x = int(cand.max())
    else:
        right_x = int(right_cand.max())

    # Ensure sane ordering and width
    if right_x <= left_x + 5:
        return x0, x1, {"snapped": False, "reason": "bad_span", "left_x": left_x, "right_x": right_x}

    x0_snap = x0 + left_x
    x1_snap = x0 + right_x + 1  # inclusive->exclusive

    return x0_snap, x1_snap, {
        "snapped": True,
        "thresh": int(thresh),
        "band": [int(x0), int(x1)],
        "picked": [int(x0_snap), int(x1_snap)],
        "kernel": [int(VERT_OPEN_W_PX), int(k_h)],
        "left_x_in_band": int(left_x),
        "right_x_in_band": int(right_x),
    }


def build_row_rois_from_dashes(bin_img, dashes, cols, spine_col_index, rightmost_col_index, w_img):
    """
    Build ROIs:
      - Height is EXACT dash bbox height: y0..y1 = dash bbox
      - Width:
          coarse band = constrained per column (won't leak into other column)
          snap band -> printed table borders via vertical lines

    Returns list of items:
      {
        "col": "R" or "L",
        "dash": dash_dict,
        "coarse": [x0,y0,x1,y1],
        "roi": [x0,y0,x1,y1],
        "snap_debug": {...}
      }
    """
    # compute spine column right edge (max x1 of dash bbox in spine col)
    spine_x_right = None
    if len(cols.get(spine_col_index, [])) > 0:
        spine_x_right = int(max(d["bbox"][2] for d in cols[spine_col_index]))
    else:
        spine_x_right = int(w_img * 0.5)

    items = []
    for col_idx, ds in cols.items():
        # Map to logical label: rightmost dash column => "R" (first reading column)
        logical = "R" if col_idx == rightmost_col_index else "L"

        for d in ds:
            x0d, y0d, x1d, y1d = d["bbox"]
            # height is EXACT dash bbox
            y0 = y0d
            y1 = y1d

            # coarse band ends just left of dash
            x1_coarse = max(0, x0d - DASH_TO_TABLE_GAP_PX)

            if col_idx == rightmost_col_index:
                # right column: band starts after spine dashes (plus pad)
                x0_coarse = max(0, spine_x_right + INTER_COL_GAP_PAD_PX)
            else:
                # left column: band starts from left page (plus tiny pad)
                x0_coarse = LEFT_PAGE_PAD_PX

            # clamp coarse width sanity
            min_w = int(w_img * MIN_ROI_WIDTH_FRAC)
            max_w = int(w_img * MAX_ROI_WIDTH_FRAC)
            if x1_coarse - x0_coarse < min_w:
                # expand left if possible
                x0_coarse = max(0, x1_coarse - min_w)
            if x1_coarse - x0_coarse > max_w:
                x0_coarse = max(0, x1_coarse - max_w)

            # Snap width to table vertical borders inside [x0_coarse, x1_coarse]
            x0_snap, x1_snap, snap_dbg = _snap_roi_width_to_table(bin_img, y0, y1, x0_coarse, x1_coarse)

            # Ensure snapped ROI still respects coarse band constraints
            x0_snap = max(x0_coarse, x0_snap)
            x1_snap = min(x1_coarse, x1_snap)
            if x1_snap <= x0_snap + 5:
                # fallback to coarse
                x0_snap, x1_snap = x0_coarse, x1_coarse
                snap_dbg = {**snap_dbg, "forced_fallback_to_coarse": True}

            items.append({
                "col": logical,
                "dash": d,
                "coarse": [int(x0_coarse), int(y0), int(x1_coarse), int(y1)],
                "roi": [int(x0_snap), int(y0), int(x1_snap), int(y1)],
                "snap_debug": snap_dbg,
            })

    return items


# =========================
# DEBUG DRAWING
# =========================

def draw_dashes_overlay(img_bgr, y0, y1, kept, rejected):
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, y0), (overlay.shape[1] - 1, y1), (0, 255, 255), 2)  # ROI bounds

    for d in kept:
        x0, yy0, x1, yy1 = d["bbox"]
        cv2.rectangle(overlay, (x0, yy0), (x1, yy1), (0, 255, 0), 2)  # kept dashes

    for r in rejected[:200]:
        x0, yy0, x1, yy1 = r["bbox"]
        cv2.rectangle(overlay, (x0, y0 + yy0), (x1, y0 + yy1), (0, 0, 255), 1)

    return overlay


def draw_dashes_and_rois_overlay(img_bgr, items, split_x=None):
    overlay = img_bgr.copy()

    # optional split line (purely visual)
    if split_x is not None:
        cv2.line(overlay, (int(split_x), 0), (int(split_x), overlay.shape[0] - 1), (0, 255, 0), 2)

    for it in items:
        # coarse band = purple
        x0, y0, x1, y1 = it["coarse"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)

        # snapped ROI = cyan
        x0, y0, x1, y1 = it["roi"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)

        # dash bbox = green
        dx0, dy0, dx1, dy1 = it["dash"]["bbox"]
        cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (0, 255, 0), 2)

    return overlay


# =========================
# MAIN
# =========================

def run(image_path: Path, out_root: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    stem = image_path.stem
    out_dir = out_root / stem
    q_dir = out_dir / "questions"
    d_dir = out_dir / "debug"
    q_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)

    # Binarize
    bin_img, bin_info = adaptive_binarize(gray)

    # ROI bounds for detection
    y0 = int(h * TOP_IGNORE_FRAC)
    y1 = int(h * (1.0 - BOTTOM_IGNORE_FRAC))

    # Detect dashes
    kept, kept_mask_roi, cleaned_roi, long_lines_roi, dash_debug, rejected = detect_dashes(bin_img, y0, y1)

    # Cluster into 2 dash columns (spine + right edge)
    cols, centers, spine_col_index, rightmost_col_index = cluster_dashes_two_columns(kept)

    # (optional) visual split line between dash columns for debug
    # We choose midpoint between the two dash-column centers if available.
    split_x = None
    if centers[0] > 0 and centers[1] > 0:
        split_x = int((centers[0] + centers[1]) * 0.5)

    # Build row ROIs from dashes (height = dash bbox, width snapped to table borders)
    items = build_row_rois_from_dashes(bin_img, kept, cols, spine_col_index, rightmost_col_index, w)

    # Ordering + naming:
    # Right column first (top->bottom), then left column (top->bottom), global numbering from 1.
    right_items = [it for it in items if it["col"] == "R"]
    left_items = [it for it in items if it["col"] == "L"]
    right_items.sort(key=lambda it: it["dash"]["cy"])
    left_items.sort(key=lambda it: it["dash"]["cy"])

    ordered = right_items + left_items if RIGHT_COLUMN_IS_FIRST else left_items + right_items

    exported = []
    q_index = 1
    for it in ordered:
        x0r, y0r, x1r, y1r = it["roi"]
        crop = img[y0r:y1r, x0r:x1r]
        if crop.size == 0:
            continue

        col = it["col"]
        name = f"Q{q_index}-Col-{col}.png"
        cv2.imwrite(str(q_dir / name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        exported.append({
            "q": int(q_index),
            "col": col,
            "file": name,
            "roi": [int(x0r), int(y0r), int(x1r), int(y1r)],
            "coarse": it["coarse"],
            "dash": it["dash"]["bbox"],
            "snap_debug": it["snap_debug"],
        })
        q_index += 1

    # Build full-size masks for easier viewing
    kept_mask_full = np.zeros((h, w), dtype=np.uint8)
    kept_mask_full[y0:y1, :] = kept_mask_roi

    cleaned_full = np.zeros((h, w), dtype=np.uint8)
    cleaned_full[y0:y1, :] = cleaned_roi

    long_full = np.zeros((h, w), dtype=np.uint8)
    long_full[y0:y1, :] = long_lines_roi

    # Debug overlays
    dashes_overlay = draw_dashes_overlay(img, y0, y1, kept, rejected)
    dashes_and_rois_overlay = draw_dashes_and_rois_overlay(img, ordered, split_x=split_x)

    # Save SMALL debug images
    def save_small(path: Path, im, is_jpg: bool):
        small = cv2.resize(im, None, fx=DEBUG_SCALE, fy=DEBUG_SCALE, interpolation=cv2.INTER_AREA if im.ndim == 3 else cv2.INTER_NEAREST)
        if is_jpg:
            cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
        else:
            cv2.imwrite(str(path), small, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])

    save_small(d_dir / "dashes_overlay_small.jpg", dashes_overlay, True)
    save_small(d_dir / "dashes_and_rois_overlay_small.jpg", dashes_and_rois_overlay, True)
    save_small(d_dir / "dash_mask_small.png", kept_mask_full, False)
    save_small(d_dir / "cleaned_small.png", cleaned_full, False)
    save_small(d_dir / "long_lines_small.png", long_full, False)

    # JSON
    data = {
        "image": {"w": int(w), "h": int(h)},
        "binarization": bin_info,
        "table_bounds": {"y0": int(y0), "y1": int(y1)},
        "dash_detection": dash_debug,
        "dash_columns": {
            "centers_x": [float(centers[0]), float(centers[1])],
            "spine_col_index": int(spine_col_index),
            "rightmost_col_index": int(rightmost_col_index),
            "counts": {"col0": int(len(cols[0])), "col1": int(len(cols[1]))},
            "right_column_is_first": bool(RIGHT_COLUMN_IS_FIRST),
        },
        "exported": exported,
        "outputs": {
            "dashes_overlay": "debug/dashes_overlay_small.jpg",
            "dashes_and_rois_overlay": "debug/dashes_and_rois_overlay_small.jpg",
            "dash_mask": "debug/dash_mask_small.png",
            "cleaned": "debug/cleaned_small.png",
            "long_lines": "debug/long_lines_small.png",
            "questions_dir": "questions/",
        },
    }
    (out_dir / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    run(args.image, args.out)


if __name__ == "__main__":
    main()
