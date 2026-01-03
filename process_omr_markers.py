import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Tunable knobs
ROI_FRAC = 0.26
THICK_K_FRAC = 0.012
COMP_MIN_AREA = 150
ARM_DENSITY_MIN = 0.04
INNER_DENSITY_MAX = 0.25
INSET_FRAC = 0.012
KERNEL_MAX = 15  # cap to avoid erasing strokes on phone photos
ROI_PAD = 12  # slight expansion to keep L legs inside ROI
DEBUG_MAX_DIM = 1600
DEBUG_JPEG_QUALITY = 85
DEBUG_PNG_COMPRESSION = 3

Point = Tuple[float, float]


def write_debug_image(image: np.ndarray, path: Path) -> None:
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > DEBUG_MAX_DIM:
        scale = DEBUG_MAX_DIM / float(max_dim)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    params = []
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, DEBUG_JPEG_QUALITY]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, DEBUG_PNG_COMPRESSION]

    cv2.imwrite(str(path), image, params)


def preprocess(gray: np.ndarray) -> Dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try adaptive threshold first
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
    )

    min_dim = min(gray.shape[:2])
    k = int(max(7, min(KERNEL_MAX, THICK_K_FRAC * min_dim)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # If too much was removed, try a more lenient approach
    if cv2.countNonZero(opened) < cv2.countNonZero(binary) * 0.1:
        # Try with smaller kernel
        k_small = max(3, k // 2)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (k_small, k_small))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        if cv2.countNonZero(opened) == 0:
            opened = binary  # fallback if everything was removed
    elif cv2.countNonZero(opened) == 0:
        opened = binary  # fallback if everything was removed

    thick = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    return {"binary": binary, "thick": thick, "kernel_size": k}


def extract_rois(width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
    roi_w = int(ROI_FRAC * width)
    roi_h = int(ROI_FRAC * height)
    pad = ROI_PAD
    return {
        "tl": (max(0, 0 - pad), max(0, 0 - pad), min(roi_w + pad, width), min(roi_h + pad, height)),
        "tr": (
            max(0, width - roi_w - pad),
            max(0, 0 - pad),
            min(roi_w + pad, width),
            min(roi_h + pad, height),
        ),
        "br": (
            max(0, width - roi_w - pad),
            max(0, height - roi_h - pad),
            min(roi_w + pad, width),
            min(roi_h + pad, height),
        ),
        "bl": (max(0, 0 - pad), max(0, height - roi_h - pad), min(roi_w + pad, width), min(roi_h + pad, height)),
    }


def component_score(mask: np.ndarray, corner: str) -> Tuple[float, float, float, float]:
    h, w = mask.shape[:2]
    strip = max(3, int(0.12 * min(w, h)))
    fill_ratio = float(np.count_nonzero(mask)) / float(w * h)

    if corner == "tl":
        h_strip = mask[:strip, :]
        v_strip = mask[:, :strip]
        inner = mask[strip:, strip:]
    elif corner == "tr":
        h_strip = mask[:strip, :]
        v_strip = mask[:, w - strip :]
        inner = mask[strip:, : w - strip]
    elif corner == "br":
        h_strip = mask[h - strip :, :]
        v_strip = mask[:, w - strip :]
        inner = mask[: h - strip, : w - strip]
    else:  # bl
        h_strip = mask[h - strip :, :]
        v_strip = mask[:, :strip]
        inner = mask[: h - strip, strip:]

    h_density = float(np.count_nonzero(h_strip)) / float(h_strip.size)
    v_density = float(np.count_nonzero(v_strip)) / float(v_strip.size)
    inner_density = float(np.count_nonzero(inner)) / float(inner.size) if inner.size > 0 else 0.0

    # More lenient thresholds for photographed images - allow lower densities
    arm_density_threshold = ARM_DENSITY_MIN * 0.5  # Allow half the minimum for faint markers
    if h_density < arm_density_threshold or v_density < arm_density_threshold or inner_density > INNER_DENSITY_MAX:
        return -1.0, h_density, v_density, inner_density

    score = (h_density + v_density) - inner_density - abs(fill_ratio - 0.2) * 0.2
    return score, h_density, v_density, inner_density


def outer_vertex_from_component(comp_mask_full: np.ndarray, rx: int, ry: int, corner: str) -> Optional[Point]:
    """Returns the outermost vertex from the selected component.

    Args:
        comp_mask_full: ROI-space mask of the selected CC, uint8 0/255
        rx, ry: ROI origin in full image coordinates
        corner: "tl"|"tr"|"br"|"bl"

    Returns:
        (x, y) in full image coordinates, or None if no pixels found
    """
    ys, xs = np.nonzero(comp_mask_full)
    if len(xs) == 0:
        return None

    # Convert to full image coordinates
    X = xs + rx
    Y = ys + ry

    # Choose extreme point depending on corner
    if corner == "tl":
        # Minimize (X + Y)
        idx = int(np.argmin(X + Y))
    elif corner == "tr":
        # Maximize (X - Y) equivalent to minimize (Y - X)
        idx = int(np.argmax(X - Y))
    elif corner == "br":
        # Maximize (X + Y)
        idx = int(np.argmax(X + Y))
    else:  # bl
        # Maximize (Y - X) equivalent to minimize (X - Y)
        idx = int(np.argmax(Y - X))

    return (float(X[idx]), float(Y[idx]))


def find_corner_in_roi(thick_mask: np.ndarray, roi: Tuple[int, int, int, int], corner: str, image_center: Tuple[float, float], debug_dir: Path, stem: str, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Tuple[Optional[Point], Dict]:
    rx, ry, rw, rh = roi
    roi_mask = thick_mask[ry : ry + rh, rx : rx + rw]
    write_debug_image(roi_mask, debug_dir / f"{stem}_roi_{corner}_thick.png")
    _, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)

    best_score = -1.0
    best_label = None
    best_density: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    for label in range(1, stats.shape[0]):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < COMP_MIN_AREA:
            continue
        x, y, w, h, _ = stats[label]
        comp_mask = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255
        score, h_density, v_density, inner_density = component_score(comp_mask, corner)
        if score > best_score:
            best_score = score
            best_label = label
            best_density = (h_density, v_density, inner_density)

    component_debug = {
        "corner": corner,
        "score": best_score,
        "densities": {"horizontal": best_density[0], "vertical": best_density[1], "inner": best_density[2]},
    }

    if best_label is None or best_score < 0:
        return None, component_debug

    comp_mask_full = (labels == best_label).astype(np.uint8) * 255
    color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(comp_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0, 0, 255), 2)
    write_debug_image(color, debug_dir / f"{stem}_roi_{corner}_comp.png")

    # Use deterministic outer vertex method (primary)
    point = outer_vertex_from_component(comp_mask_full, rx, ry, corner)
    if point is None:
        return None, component_debug

    # Compute component pixel extremes for sanity check
    ys, xs = np.nonzero(comp_mask_full)
    if len(xs) == 0:
        return None, component_debug

    X = xs + rx
    Y = ys + ry
    minX, maxX = float(X.min()), float(X.max())
    minY, maxY = float(Y.min()), float(Y.max())
    cw = maxX - minX
    ch = maxY - minY

    # Sanity check: verify picked point is near expected extremes
    px, py = point[0], point[1]
    dx = 0.15 * cw
    dy = 0.15 * ch
    picked_method = "outer_vertex"

    sanity_passed = False
    if corner == "tl":
        sanity_passed = (px <= minX + dx) and (py <= minY + dy)
    elif corner == "tr":
        sanity_passed = (px >= maxX - dx) and (py <= minY + dy)
    elif corner == "br":
        sanity_passed = (px >= maxX - dx) and (py >= maxY - dy)
    else:  # bl
        sanity_passed = (px <= minX + dx) and (py >= maxY - dy)

    # Fallback: windowed-extrema corner pick if sanity check fails
    if not sanity_passed:
        picked_method = "fallback_windowed_extrema"
        dx_fallback = dx
        dy_fallback = dy

        # Define candidate windows based on corner type
        if corner == "tl":
            candidates_mask = (X <= minX + dx_fallback) & (Y <= minY + dy_fallback)
            ideal = (minX, minY)
        elif corner == "tr":
            candidates_mask = (X >= maxX - dx_fallback) & (Y <= minY + dy_fallback)
            ideal = (maxX, minY)
        elif corner == "br":
            candidates_mask = (X >= maxX - dx_fallback) & (Y >= maxY - dy_fallback)
            ideal = (maxX, maxY)
        else:  # bl
            candidates_mask = (X <= minX + dx_fallback) & (Y >= maxY - dy_fallback)
            ideal = (minX, maxY)

        candidates = np.where(candidates_mask)[0]

        # If no candidates, relax tolerance once
        if len(candidates) == 0:
            dx_fallback *= 1.8
            dy_fallback *= 1.8
            if corner == "tl":
                candidates_mask = (X <= minX + dx_fallback) & (Y <= minY + dy_fallback)
            elif corner == "tr":
                candidates_mask = (X >= maxX - dx_fallback) & (Y <= minY + dy_fallback)
            elif corner == "br":
                candidates_mask = (X >= maxX - dx_fallback) & (Y >= maxY - dy_fallback)
            else:  # bl
                candidates_mask = (X <= minX + dx_fallback) & (Y >= maxY - dy_fallback)
            candidates = np.where(candidates_mask)[0]

        # Pick candidate closest to ideal corner
        if len(candidates) > 0:
            ideal_x, ideal_y = ideal
            dists = (X[candidates] - ideal_x) ** 2 + (Y[candidates] - ideal_y) ** 2
            idx = int(candidates[np.argmin(dists)])
            point = (float(X[idx]), float(Y[idx]))
        # If still no candidates, keep original point (don't crash)

    # Add debug info to component_debug
    component_debug["picked_method"] = picked_method
    component_debug["component_extremes"] = {
        "minX": minX,
        "maxX": maxX,
        "minY": minY,
        "maxY": maxY,
        "cw": cw,
        "ch": ch,
        "dx": dx,
        "dy": dy,
    }

    # Debug overlay: show the chosen vertex
    picked_color = color.copy()
    px_roi = int(point[0] - rx)
    py_roi = int(point[1] - ry)
    cv2.circle(picked_color, (px_roi, py_roi), 5, (0, 255, 0), -1)

    # If fallback was used, save additional debug image
    if picked_method == "fallback_windowed_extrema":
        write_debug_image(picked_color, debug_dir / f"{stem}_roi_{corner}_picked_fallback.png")
    else:
        write_debug_image(picked_color, debug_dir / f"{stem}_roi_{corner}_picked.png")

    return point, component_debug


def compute_warp(corners: np.ndarray, inset: int) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    (tl, tr, br, bl) = corners
    inset_corners = np.array(
        [
            (tl[0] + inset, tl[1] + inset),
            (tr[0] - inset, tr[1] + inset),
            (br[0] - inset, br[1] - inset),
            (bl[0] + inset, bl[1] - inset),
        ],
        dtype=np.float32,
    )

    width_top = np.linalg.norm(inset_corners[1] - inset_corners[0])
    width_bottom = np.linalg.norm(inset_corners[2] - inset_corners[3])
    height_right = np.linalg.norm(inset_corners[2] - inset_corners[1])
    height_left = np.linalg.norm(inset_corners[3] - inset_corners[0])

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    width = max(width, 100)
    height = max(height, 100)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(inset_corners, dst)
    return matrix, (width, height), inset_corners


def preprocess_alternative(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Alternative preprocessing with more lenient thresholding for photographed images"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try Otsu thresholding as alternative
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    min_dim = min(gray.shape[:2])
    k = int(max(3, min(KERNEL_MAX // 2, THICK_K_FRAC * min_dim * 0.5)))  # Smaller kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(opened) == 0:
        opened = binary  # fallback if everything was removed

    thick = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    return {"binary": binary, "thick": thick, "kernel_size": k}


def process_image(path: Path, output_dir: Path, debug_dir: Path) -> Dict:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image {path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)
    write_debug_image(processed["binary"], debug_dir / f"{path.stem}_mask_binary.png")
    write_debug_image(processed["thick"], debug_dir / f"{path.stem}_mask_thick.png")

    rois = extract_rois(width, height)
    image_center = (width / 2.0, height / 2.0)
    corners: Dict[str, Optional[Point]] = {}
    component_logs: Dict[str, Dict] = {}

    for corner_name, roi in rois.items():
        pt, comp_dbg = find_corner_in_roi(processed["thick"], roi, corner_name, image_center, debug_dir, path.stem, width, height)
        corners[corner_name] = pt
        component_logs[corner_name] = comp_dbg

    # If some corners are missing, try alternative preprocessing
    missing_corners = [name for name, pt in corners.items() if pt is None]
    if missing_corners:
        processed_alt = preprocess_alternative(gray)
        write_debug_image(processed_alt["thick"], debug_dir / f"{path.stem}_mask_thick_alt.png")

        for corner_name in missing_corners:
            roi = rois[corner_name]
            pt, comp_dbg = find_corner_in_roi(processed_alt["thick"], roi, corner_name, image_center, debug_dir, path.stem, width, height)
            if pt is not None:  # Only update if we found something
                corners[corner_name] = pt
                component_logs[corner_name] = comp_dbg

    debug_data: Dict = {
        "image": {"width": width, "height": height, "path": str(path)},
        "rois": rois,
        "constants": {
            "ROI_FRAC": ROI_FRAC,
            "THICK_K_FRAC": THICK_K_FRAC,
            "COMP_MIN_AREA": COMP_MIN_AREA,
            "ARM_DENSITY_MIN": ARM_DENSITY_MIN,
            "INNER_DENSITY_MAX": INNER_DENSITY_MAX,
            "INSET_FRAC": INSET_FRAC,
            "KERNEL_MAX": KERNEL_MAX,
            "kernel_size": processed["kernel_size"],
        },
        "components": component_logs,
        "selected": corners,
        "status": "failed",
    }

    overlay = image.copy()
    for corner_name, (rx, ry, rw, rh) in rois.items():
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        if corners[corner_name] is not None:
            cv2.circle(overlay, (int(corners[corner_name][0]), int(corners[corner_name][1])), 10, (0, 0, 255), -1)

    found_points = [corners.get(key) for key in ["tl", "tr", "br", "bl"]]
    if any(pt is None for pt in found_points):
        found_count = sum(pt is not None for pt in found_points)
        debug_data["status"] = f"fail_missing ({found_count}/4)"
        write_debug_image(overlay, debug_dir / f"{path.stem}_debug_rois_and_points.png")
        with open(debug_dir / f"{path.stem}_data.json", "w", encoding="utf-8") as fp:
            json.dump(debug_data, fp, indent=2)
        return debug_data

    ordered_corners = np.array(found_points, dtype=np.float32)
    inset = max(10, int(INSET_FRAC * min(width, height)))
    warp_matrix, size, inset_corners = compute_warp(ordered_corners, inset)
    warped = cv2.warpPerspective(image, warp_matrix, size)

    cv2.polylines(overlay, [np.array(inset_corners, dtype=np.int32)], True, (255, 0, 255), 3)
    write_debug_image(overlay, debug_dir / f"{path.stem}_debug_rois_and_points.png")
    write_debug_image(overlay, debug_dir / f"{path.stem}_debug_final_quad.png")
    cv2.imwrite(str(output_dir / f"{path.stem}_crop.png"), warped)

    debug_data.update(
        {
            "selected": ordered_corners.tolist(),
            "status": "success",
            "warp_size": {"width": size[0], "height": size[1]},
            "inset": inset,
        }
    )

    with open(debug_dir / f"{path.stem}_data.json", "w", encoding="utf-8") as fp:
        json.dump(debug_data, fp, indent=2)

    return debug_data


def run(input_dir: Path, output_dir: Path, debug_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, str] = {}
    for path in sorted(input_dir.glob("*.jpg")):
        try:
            debug_data = process_image(path, output_dir, debug_dir)
            results[path.name] = debug_data["status"]
        except Exception as exc:  # noqa: BLE001
            results[path.name] = f"error: {exc}"
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect OMR markers, deskew, and crop using L-shaped corners.")
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"), help="Directory containing input images.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/crops"), help="Directory to write cropped outputs.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("output/debug"),
        help="Directory to write debug visualizations and metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(args.input_dir, args.output_dir, args.debug_dir)
    success = sum(1 for status in results.values() if status == "success")
    total = len(results)
    print(f"Processed {total} images. Success: {success}. Failures: {total - success}.")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
