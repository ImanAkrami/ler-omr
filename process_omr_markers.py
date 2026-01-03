import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def preprocess(gray: np.ndarray) -> Dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    combined = cv2.bitwise_and(otsu, adaptive)
    inverted = cv2.bitwise_not(combined)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    inv_otsu = cv2.bitwise_not(otsu)
    return {"otsu": otsu, "adaptive": adaptive, "binary": cleaned, "dilated": dilated, "inv_otsu": inv_otsu}


def order_indices_by_points(points: np.ndarray) -> List[int]:
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)[:, 0]
    tl = int(np.argmin(s))
    br = int(np.argmax(s))
    tr = int(np.argmin(diff))
    bl = int(np.argmax(diff))
    return [tl, tr, br, bl]


def order_points(points: np.ndarray) -> np.ndarray:
    idx = order_indices_by_points(points)
    return points[idx]


def find_inner_corner(contour: np.ndarray, image_center: np.ndarray) -> Tuple[Point, float]:
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is not None and len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None and len(defects) > 0:
            flat = defects.reshape(-1, 4)
            farthest = max(flat, key=lambda d: d[3])
            depth = float(farthest[3]) / 256.0
            point = contour[int(farthest[2])][0]
            return (float(point[0]), float(point[1])), depth

    pts = contour[:, 0, :]
    dists = np.linalg.norm(pts - image_center, axis=1)
    idx = int(np.argmin(dists))
    point = pts[idx]
    return (float(point[0]), float(point[1])), 0.0


def find_marker_candidates(binary: np.ndarray, image_shape: Tuple[int, int]) -> List[Dict]:
    height, width = image_shape
    image_area = float(height * width)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    center = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
    min_area = image_area * 0.0003
    max_area = image_area * 0.06

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(w * h)
        if bbox_area == 0:
            continue
        rectangularity = area / bbox_area
        if rectangularity < 0.1 or rectangularity > 0.9:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) < 5:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        inner_corner, concavity = find_inner_corner(contour, center)
        edge_proximity = min(cx / width, cy / height, (width - cx) / width, (height - cy) / height)

        candidates.append(
            {
                "contour": contour,
                "area": area,
                "rectangularity": rectangularity,
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "inner_corner": inner_corner,
                "concavity": concavity,
                "edge_proximity": edge_proximity,
            }
        )

    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates


def score_combination(combo: Sequence[Dict], width: int, height: int) -> float:
    centers = np.array([c["center"] for c in combo], dtype=np.float32)
    ordered = order_points(centers)
    ideal = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])
    diag = math.hypot(width, height)
    dist_score = float(np.linalg.norm(ordered - ideal, axis=1).sum() / diag)

    concavity_bonus = sum(c.get("concavity", 0.0) for c in combo) / (len(combo) * 10.0 + 1e-6)
    rectangularity_score = sum(abs(c["rectangularity"] - 0.45) for c in combo) / len(combo)
    edge_bonus = sum(1.0 - min(c["edge_proximity"], 0.4) / 0.4 for c in combo) / len(combo)

    return dist_score + rectangularity_score + edge_bonus - concavity_bonus


def choose_markers(candidates: List[Dict], width: int, height: int) -> Optional[List[Dict]]:
    if len(candidates) < 4:
        return None

    best_combo: Optional[Sequence[Dict]] = None
    best_score = float("inf")
    for combo in combinations(candidates[:12], 4):
        centers = np.array([c["center"] for c in combo], dtype=np.float32)
        span_x = np.ptp(centers[:, 0])
        span_y = np.ptp(centers[:, 1])
        if span_x < width * 0.3 or span_y < height * 0.3:
            continue

        score = score_combination(combo, width, height)
        if score < best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        return None

    ordered_indices = order_indices_by_points(np.array([c["center"] for c in best_combo], dtype=np.float32))
    ordered_combo = [list(best_combo)[i] for i in ordered_indices]
    return ordered_combo


def compute_warp(corners: List[Point]) -> Tuple[np.ndarray, Tuple[int, int]]:
    pts = np.array(corners, dtype=np.float32)
    (tl, tr, br, bl) = pts
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    width = max(width, 100)
    height = max(height, 100)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return matrix, (width, height)


def process_image(path: Path, output_dir: Path, debug_dir: Path) -> Dict:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image {path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)
    candidate_lists = [
        find_marker_candidates(processed["binary"], (height, width)),
        find_marker_candidates(processed["dilated"], (height, width)),
        find_marker_candidates(processed["inv_otsu"], (height, width)),
    ]
    candidates: List[Dict] = []
    for c_list in candidate_lists:
        for cand in c_list:
            cx, cy = cand["center"]
            if all((cx - oc["center"][0]) ** 2 + (cy - oc["center"][1]) ** 2 > 9 for oc in candidates):
                candidates.append(cand)
    candidates.sort(key=lambda c: c["area"], reverse=True)
    selected = choose_markers(candidates, width, height)

    debug_data: Dict = {
        "image": {
            "width": width,
            "height": height,
            "path": str(path),
        },
        "candidates": [
            {
                "center": candidate["center"],
                "area": candidate["area"],
                "rectangularity": candidate["rectangularity"],
                "concavity": candidate.get("concavity", 0.0),
                "bbox": candidate["bbox"],
                "edge_proximity": candidate["edge_proximity"],
            }
            for candidate in candidates
        ],
        "selected": None,
        "status": "failed",
    }

    overlay = image.copy()
    for candidate in candidates:
        cv2.drawContours(overlay, [candidate["contour"]], -1, (0, 255, 0), 2)
        cx, cy = map(int, candidate["center"])
        cv2.circle(overlay, (cx, cy), 5, (0, 200, 255), -1)

    if selected is None:
        cv2.imwrite(str(debug_dir / f"{path.stem}_binary.png"), processed["binary"])
        cv2.imwrite(str(debug_dir / f"{path.stem}_candidates.png"), overlay)
        with open(debug_dir / f"{path.stem}_data.json", "w", encoding="utf-8") as fp:
            json.dump(debug_data, fp, indent=2)
        return debug_data

    ordered_corners = [candidate["inner_corner"] for candidate in selected]
    warp_matrix, size = compute_warp(ordered_corners)
    warped = cv2.warpPerspective(image, warp_matrix, size)

    for corner, color in zip(ordered_corners, [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]):
        cv2.circle(overlay, (int(corner[0]), int(corner[1])), 10, color, -1)

    cv2.polylines(overlay, [np.array(ordered_corners, dtype=np.int32)], True, (255, 0, 255), 3)

    cv2.imwrite(str(debug_dir / f"{path.stem}_binary.png"), processed["binary"])
    cv2.imwrite(str(debug_dir / f"{path.stem}_candidates.png"), overlay)
    cv2.imwrite(str(output_dir / f"{path.stem}_crop.png"), warped)

    debug_data.update({
        "selected": ordered_corners,
        "status": "success",
        "warp_size": {"width": size[0], "height": size[1]},
    })

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
