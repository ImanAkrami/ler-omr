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

Point = Tuple[float, float]


def preprocess(gray: np.ndarray) -> Dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
    if cv2.countNonZero(opened) == 0:
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

    if h_density < ARM_DENSITY_MIN or v_density < ARM_DENSITY_MIN or inner_density > INNER_DENSITY_MAX:
        return -1.0, h_density, v_density, inner_density

    score = (h_density + v_density) - inner_density - abs(fill_ratio - 0.2) * 0.2
    return score, h_density, v_density, inner_density


def intersect_lines(line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def find_corner_in_roi(thick_mask: np.ndarray, roi: Tuple[int, int, int, int], corner: str, image_center: Tuple[float, float], debug_dir: Path, stem: str) -> Tuple[Optional[Point], Dict]:
    rx, ry, rw, rh = roi
    roi_mask = thick_mask[ry : ry + rh, rx : rx + rw]
    cv2.imwrite(str(debug_dir / f"{stem}_roi_{corner}_thick.png"), roi_mask)
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
    cv2.imwrite(str(debug_dir / f"{stem}_roi_{corner}_comp.png"), color)

    edges = cv2.Canny(comp_mask_full, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=15, maxLineGap=8)
    horizontal_line: Optional[Tuple[int, int, int, int]] = None
    vertical_line: Optional[Tuple[int, int, int, int]] = None
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            length = math.hypot(x2 - x1, y2 - y1)
            if angle < 15 or angle > 165:
                if horizontal_line is None or length > math.hypot(horizontal_line[2] - horizontal_line[0], horizontal_line[3] - horizontal_line[1]):
                    horizontal_line = (x1, y1, x2, y2)
            if 75 <= angle <= 105:
                if vertical_line is None or length > math.hypot(vertical_line[2] - vertical_line[0], vertical_line[3] - vertical_line[1]):
                    vertical_line = (x1, y1, x2, y2)

    intersection: Optional[Tuple[float, float]] = None
    if horizontal_line is not None and vertical_line is not None:
        intersection = intersect_lines(horizontal_line, vertical_line)

    if intersection is None:
        ys, xs = np.nonzero(comp_mask_full)
        if len(xs) == 0:
            return None, component_debug
        pts_global = np.stack([xs + rx, ys + ry], axis=1)
        center_arr = np.array([[image_center[0], image_center[1]]], dtype=np.float32)
        dists = np.linalg.norm(pts_global - center_arr, axis=1)
        idx = int(np.argmin(dists))
        point = (float(pts_global[idx, 0]), float(pts_global[idx, 1]))
    else:
        point = (intersection[0] + rx, intersection[1] + ry)

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


def process_image(path: Path, output_dir: Path, debug_dir: Path) -> Dict:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image {path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)
    cv2.imwrite(str(debug_dir / f"{path.stem}_mask_binary.png"), processed["binary"])
    cv2.imwrite(str(debug_dir / f"{path.stem}_mask_thick.png"), processed["thick"])

    rois = extract_rois(width, height)
    image_center = (width / 2.0, height / 2.0)
    corners: Dict[str, Optional[Point]] = {}
    component_logs: Dict[str, Dict] = {}

    for corner_name, roi in rois.items():
        pt, comp_dbg = find_corner_in_roi(processed["thick"], roi, corner_name, image_center, debug_dir, path.stem)
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
        cv2.imwrite(str(debug_dir / f"{path.stem}_debug_rois_and_points.png"), overlay)
        with open(debug_dir / f"{path.stem}_data.json", "w", encoding="utf-8") as fp:
            json.dump(debug_data, fp, indent=2)
        return debug_data

    ordered_corners = np.array(found_points, dtype=np.float32)
    inset = max(10, int(INSET_FRAC * min(width, height)))
    warp_matrix, size, inset_corners = compute_warp(ordered_corners, inset)
    warped = cv2.warpPerspective(image, warp_matrix, size)

    cv2.polylines(overlay, [np.array(inset_corners, dtype=np.int32)], True, (255, 0, 255), 3)
    cv2.imwrite(str(debug_dir / f"{path.stem}_debug_rois_and_points.png"), overlay)
    cv2.imwrite(str(debug_dir / f"{path.stem}_debug_final_quad.png"), overlay)
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
