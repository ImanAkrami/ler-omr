import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
RIGHT_COLUMN_IS_FIRST = True
HEADER_SEARCH_MAX_Y_FRAC = 0.40
SEP_SEARCH_X_BAND = (0.35, 0.65)
ROW_STRONG_FRAC = 0.22
ROW_MIN_GAP = 8
NUMBER_STRIP_FRAC = 0.14
SEP_PAD_FRAC = 0.02  # Padding away from separator line


@dataclass
class Question:
    questionNumber: int
    type: str  # "MCQ" | "TF" | "DESC"


@dataclass
class Meta:
    questions: List[Question]


def load_meta(path: Path) -> Meta:
    data = json.loads(path.read_text())
    return Meta(questions=[Question(**q) for q in data["questions"]])


def binarize(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )


def extract_lines(bin_img):
    h, w = bin_img.shape[:2]
    hk = max(15, w // 40)
    vk = max(15, h // 40)

    horiz = cv2.morphologyEx(
        bin_img, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    )
    vert = cv2.morphologyEx(
        bin_img, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    )
    return horiz, vert


def find_header_y(horiz):
    h = horiz.shape[0]
    lim = int(HEADER_SEARCH_MAX_Y_FRAC * h)
    proj = horiz[:lim].sum(axis=1)
    return int(np.argmax(proj)) + 5


def find_sep_x(vert):
    h, w = vert.shape[:2]
    x0, x1 = int(w * SEP_SEARCH_X_BAND[0]), int(w * SEP_SEARCH_X_BAND[1])
    proj = vert[:, x0:x1].sum(axis=0)
    return int(np.argmax(proj)) + x0


def filter_horizontal_rows(horiz, column_width: int, debug_dir: Optional[Path] = None, stem: str = "") -> np.ndarray:
    """Filter horizontal lines using connected component analysis.

    Only keeps components that are:
    - Wide enough (>= 70% of column width)
    - Thin enough (<= 1.5 * expected line thickness)
    """
    h, w = horiz.shape[:2]
    expected_line_thickness = max(3, h // 200)  # Rough estimate

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)

    # Build filtered mask
    horiz_rows_only = np.zeros_like(horiz)
    kept_mask = np.zeros_like(horiz, dtype=np.uint8)
    rejected_mask = np.zeros_like(horiz, dtype=np.uint8)

    min_width = int(0.70 * column_width)
    max_height = int(1.5 * expected_line_thickness)

    for label in range(1, num_labels):
        comp_width = stats[label, cv2.CC_STAT_WIDTH]
        comp_height = stats[label, cv2.CC_STAT_HEIGHT]

        if comp_width >= min_width and comp_height <= max_height:
            # Keep this component
            mask = (labels == label).astype(np.uint8) * 255
            horiz_rows_only = cv2.bitwise_or(horiz_rows_only, mask)
            kept_mask = cv2.bitwise_or(kept_mask, mask)
        else:
            # Reject this component
            mask = (labels == label).astype(np.uint8) * 255
            rejected_mask = cv2.bitwise_or(rejected_mask, mask)

    # Save debug image if requested
    if debug_dir is not None:
        debug_img = cv2.cvtColor(horiz, cv2.COLOR_GRAY2BGR)
        # Overlay kept components in green
        debug_img[kept_mask > 0] = [0, 255, 0]
        # Overlay rejected components in red
        debug_img[rejected_mask > 0] = [0, 0, 255]
        cv2.imwrite(str(debug_dir / f"{stem}_horiz_rows_filtered.png"), debug_img)

    return horiz_rows_only


def group_peaks(idx):
    groups = []
    for i in sorted(idx):
        if not groups or i - groups[-1][-1] > ROW_MIN_GAP:
            groups.append([i])
        else:
            groups[-1].append(i)
    return [int(np.mean(g)) for g in groups]


def row_boundaries(horiz_filtered, y0, x0, x1):
    """Detect row boundaries from filtered horizontal lines."""
    region = horiz_filtered[y0:, x0:x1]
    proj = region.sum(axis=1)
    strong = np.where(proj > proj.max() * ROW_STRONG_FRAC)[0]
    rows = [y0 + r for r in group_peaks(strong)]
    rows = sorted(set([y0] + rows + [horiz_filtered.shape[0] - 1]))
    return rows


def build_unit_grid(boundaries: List[int], median_unit_height: float) -> List[int]:
    """Rebuild boundaries using a stable unit grid model.

    Rules:
    - If gap < 0.6 * median_unit_height → merge (skip boundary)
    - If gap > 1.8 * median_unit_height → split into N units
    - Snap boundaries to nearest multiple of median_unit_height
    """
    if len(boundaries) < 2:
        return boundaries

    if median_unit_height <= 0:
        return boundaries

    # Build new grid starting from first boundary
    new_boundaries = [boundaries[0]]

    for i in range(len(boundaries) - 1):
        gap = boundaries[i + 1] - boundaries[i]
        gap_ratio = gap / median_unit_height

        if gap_ratio < 0.6:
            # Merge: skip this boundary, continue to next
            continue
        elif gap_ratio > 1.8:
            # Split: add multiple unit boundaries
            num_units = max(1, int(round(gap_ratio)))
            unit_step = gap / num_units
            current_y = float(boundaries[i])
            for j in range(1, num_units):
                next_y = current_y + j * unit_step
                # Snap to nearest multiple of median_unit_height
                snap_y = round(next_y / median_unit_height) * median_unit_height
                new_boundaries.append(int(snap_y))
            # Add final boundary
            new_boundaries.append(boundaries[i + 1])
        else:
            # Normal: add single boundary, snapped
            next_y = float(boundaries[i + 1])
            snap_y = round(next_y / median_unit_height) * median_unit_height
            new_boundaries.append(int(snap_y))

    # Ensure last boundary is included and sorted
    if new_boundaries[-1] != boundaries[-1]:
        new_boundaries.append(boundaries[-1])

    # Remove duplicates and sort
    new_boundaries = sorted(set(new_boundaries))
    return new_boundaries


def assign_units_geometry_driven(questions: List[Question], boundaries: List[int], column_name: str) -> List[int]:
    """Assign units to questions based on geometry, not arithmetic.

    Rules:
    - MCQ / TF → always consume exactly 1 unit
    - DESC → consume consecutive units, typically 2-4 units based on available space
    """
    if len(boundaries) < 2:
        return [1] * len(questions)

    num_units = len(boundaries) - 1
    units = []
    unit_idx = 0

    # Count how many MCQ/TF vs DESC we have
    mcq_tf_count = sum(1 for q in questions if q.type in ["MCQ", "TF"])
    desc_count = sum(1 for q in questions if q.type == "DESC")

    # Calculate units available for DESC questions
    units_for_mcq_tf = mcq_tf_count
    units_for_desc = num_units - units_for_mcq_tf

    # Average units per DESC (if any)
    avg_desc_units = max(2, units_for_desc // desc_count) if desc_count > 0 else 1
    avg_desc_units = min(avg_desc_units, 4)  # Cap at 4 units per DESC

    for i, q in enumerate(questions):
        if q.type in ["MCQ", "TF"]:
            # Always exactly 1 unit
            units.append(1)
            unit_idx += 1
        elif q.type == "DESC":
            # Consume multiple units for descriptive questions
            if i == len(questions) - 1:
                # Last question: consume all remaining units
                remaining = num_units - unit_idx
                units.append(max(1, remaining))
            else:
                # Use average, but ensure we don't exceed available units
                remaining_units = num_units - unit_idx
                remaining_questions = len(questions) - i
                # Reserve at least 1 unit for each remaining question
                max_for_this = remaining_units - (remaining_questions - 1)
                consume = min(avg_desc_units, max_for_this)
                units.append(max(1, consume))
            unit_idx += units[-1]
        else:
            # Unknown type, default to 1 unit
            units.append(1)
            unit_idx += 1

    return units


def slice_sheet(crop: Path, meta: Meta, out_root: Path):
    img = cv2.imread(str(crop))
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = binarize(gray)
    horiz, vert = extract_lines(bin_img)

    top_y = find_header_y(horiz)
    sep_x = find_sep_x(vert)

    margin = int(0.02 * w)
    sep_pad = int(SEP_PAD_FRAC * w)

    # Fix column ROI to avoid separator line
    right = (margin, sep_x - sep_pad)
    left = (sep_x + sep_pad, w - margin)

    right_col_width = right[1] - right[0]
    left_col_width = left[1] - left[0]

    # Filter horizontal rows using connected components
    stem = crop.stem.replace("_crop", "")
    out_d = out_root / stem / "debug"
    out_d.mkdir(parents=True, exist_ok=True)

    # Use full stem name for debug image
    debug_stem = stem
    horiz_filtered = filter_horizontal_rows(horiz, min(right_col_width, left_col_width), out_d, debug_stem)

    # Detect initial row boundaries
    rb_initial = row_boundaries(horiz_filtered, top_y, *right)
    lb_initial = row_boundaries(horiz_filtered, top_y, *left)

    # Compute unit heights and median
    def compute_median_unit_height(boundaries):
        if len(boundaries) < 2:
            return h * 0.05  # Default estimate
        heights = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
        return float(np.median(heights))

    median_unit_height_r = compute_median_unit_height(rb_initial)
    median_unit_height_l = compute_median_unit_height(lb_initial)

    # Build stable unit grids
    rb = build_unit_grid(rb_initial, median_unit_height_r)
    lb = build_unit_grid(lb_initial, median_unit_height_l)

    # Sanity checks
    sanity_flags = []

    # Check total units vs questions
    questions = meta.questions
    mid = (len(questions) + 1) // 2

    if RIGHT_COLUMN_IS_FIRST:
        q_r, q_l = questions[:mid], questions[mid:]
    else:
        q_l, q_r = questions[:mid], questions[mid:]

    total_units_r = len(rb) - 1
    total_units_l = len(lb) - 1

    if total_units_r < len(q_r):
        sanity_flags.append(f"Right column: {total_units_r} units < {len(q_r)} questions")
    if total_units_l < len(q_l):
        sanity_flags.append(f"Left column: {total_units_l} units < {len(q_l)} questions")

    # Check median unit height
    if median_unit_height_r < 0.01 * h or median_unit_height_r > 0.10 * h:
        sanity_flags.append(f"Right column median unit height {median_unit_height_r:.1f}px is outside normal range")
    if median_unit_height_l < 0.01 * h or median_unit_height_l > 0.10 * h:
        sanity_flags.append(f"Left column median unit height {median_unit_height_l:.1f}px is outside normal range")

    # Assign units using geometry-driven approach
    ur = assign_units_geometry_driven(q_r, rb, "right")
    ul = assign_units_geometry_driven(q_l, lb, "left")

    # Check for DESC questions consuming too many units
    for i, (q, u) in enumerate(zip(q_r, ur)):
        if q.type == "DESC" and u > 6:
            sanity_flags.append(f"Right column Q{q.questionNumber} (DESC) consumes {u} units (warning)")
    for i, (q, u) in enumerate(zip(q_l, ul)):
        if q.type == "DESC" and u > 6:
            sanity_flags.append(f"Left column Q{q.questionNumber} (DESC) consumes {u} units (warning)")

    # Save debug data
    out_q = out_root / stem / "questions"
    out_q.mkdir(parents=True, exist_ok=True)

    # Save unit heights
    unit_heights_data = {
        "right": {
            "boundaries": rb,
            "heights": [rb[i+1] - rb[i] for i in range(len(rb) - 1)],
            "median_height": median_unit_height_r,
        },
        "left": {
            "boundaries": lb,
            "heights": [lb[i+1] - lb[i] for i in range(len(lb) - 1)],
            "median_height": median_unit_height_l,
        },
        "sanity_flags": sanity_flags,
    }
    (out_d / "unit_heights.json").write_text(json.dumps(unit_heights_data, indent=2))

    # Create unit grid overlay
    overlay = img.copy()
    for y in rb:
        cv2.line(overlay, (0, y), (w, y), (255, 0, 0), 1)
    for y in lb:
        cv2.line(overlay, (0, y), (w, y), (255, 0, 0), 1)
    cv2.imwrite(str(out_d / "unit_grid_overlay.png"), overlay)

    boxes = []

    def emit(col, rows, qs, us, col_name):
        k = 0
        for i, (q, u) in enumerate(zip(qs, us)):
            if k + u >= len(rows):
                # Safety check: don't go out of bounds
                u = len(rows) - k
            if u <= 0:
                break
            y0, y1 = rows[k], rows[k + u]
            k += u
            x0, x1 = col
            strip = int((x1 - x0) * NUMBER_STRIP_FRAC)

            crop_img = img[y0:y1, x0:x1 - strip]
            cv2.imwrite(str(out_q / f"q{q.questionNumber:03d}.png"), crop_img)

            boxes.append({
                "q": q.questionNumber,
                "type": q.type,
                "units": u,
                "box": [x0, y0, x1 - strip, y1],
                "column": col_name,
            })

            cv2.rectangle(overlay, (x0, y0), (x1 - strip, y1), (0, 255, 0), 2)

    emit(right, rb, q_r, ur, "right")
    emit(left, lb, q_l, ul, "left")

    cv2.imwrite(str(out_d / "overlay.png"), overlay)

    # Add sanity flags to data.json
    data_with_flags = {
        "boxes": boxes,
        "sanity_flags": sanity_flags,
        "unit_counts": {
            "right": total_units_r,
            "left": total_units_l,
        },
        "question_counts": {
            "right": len(q_r),
            "left": len(q_l),
        },
    }
    (out_d / "data.json").write_text(json.dumps(data_with_flags, indent=2))

