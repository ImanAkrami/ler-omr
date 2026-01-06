#!/usr/bin/env python3
"""
MCQ processor – template-fit X placement + FIXED fill detection

Fixes:
- Threshold computed from BACKGROUND RING outside bubble (not halo)
- Fill decision uses DENSITIES (core_ratio vs halo_ratio), not raw core>=k*halo
- Handles "ink outside" correctly:
  - If ink is mostly outside, halo_ratio ~ core_ratio => rejected
  - If bubble is filled, core_ratio >> halo_ratio => accepted
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import cv2
import numpy as np


# ============================================================
# 0) Known template facts
# ============================================================

# LEFT->RIGHT labels are 4,3,2,1
BUBBLE_X_FRACS = [0.28, 0.49, 0.70, 0.91]
LABELS_LTR = ["4", "3", "2", "1"]


# ============================================================
# 1) Normalize height only
# ============================================================

CANONICAL_H = 48


# ============================================================
# 2) Bubble geometry (relative to canonical height)
# ============================================================

RY_FRAC = 0.22
RX_OVER_RY = 1.35  # horizontal bubble

CORE_SCALE = 0.50          # a bit larger than before (helps light fills)
HALO_INNER = 1.05
HALO_OUTER = 1.55

# Background ring OUTSIDE halo (used for threshold only)
BG_RING_INNER = 1.75
BG_RING_OUTER = 2.45


# ============================================================
# 3) Robust y (row) selection using vertical edges only
# ============================================================

CY_SEARCH_Y0_FRAC = 0.25
CY_SEARCH_Y1_FRAC = 0.80
BAND_HALF_H = 9


# ============================================================
# 4) X model fit params (small search)
# ============================================================

SCALE_MIN = 0.94
SCALE_MAX = 1.06
SCALE_STEP = 0.005

SHIFT_FRAC = 0.06
SHIFT_STEP_PX = 1

BORDER_WIN = 2  # scoring window around borders


# ============================================================
# 5) Fill detection (THIS IS THE FIX)
# ============================================================

# threshold = p90(background_ring) - delta
RING_P90_DELTA = 28.0

# core must have "some" ink in absolute sense
MIN_CORE_DARK_PIXELS = 16

# core must be dark-dense enough (not just a tiny dot)
CORE_DARK_RATIO_MIN = 0.22

# reject outside-only: core density must exceed halo density by margin
# (core_ratio >= halo_ratio + margin)
CORE_MINUS_HALO_MIN = 0.12

CONF_OK = 0.95
CONF_BLANK = 0.90


# ============================================================
# Helpers
# ============================================================

def _resize_height(img: np.ndarray, H: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == H:
        return img
    s = H / float(h)
    return cv2.resize(img, (int(round(w * s)), H), interpolation=cv2.INTER_AREA)


def _abs_sobel_dx(gray: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dx = np.abs(dx)
    dx = cv2.GaussianBlur(dx, (5, 5), 0)
    return dx


def _find_cy(dx_abs: np.ndarray) -> int:
    h, _ = dx_abs.shape[:2]
    y0 = int(h * CY_SEARCH_Y0_FRAC)
    y1 = int(h * CY_SEARCH_Y1_FRAC)
    y0 = max(0, min(h - 2, y0))
    y1 = max(y0 + 1, min(h, y1))
    row_energy = dx_abs[y0:y1, :].sum(axis=1)
    return int(y0 + int(np.argmax(row_energy)))


def _ellipse_mask(shape_hw: Tuple[int, int], cx: int, cy: int, rx: float, ry: float) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(rx)), max(1, int(ry))), 0, 0, 360, 255, -1)
    return m


def _annulus_mask(shape_hw: Tuple[int, int], cx: int, cy: int,
                  rx_in: float, ry_in: float, rx_out: float, ry_out: float) -> np.ndarray:
    outer = _ellipse_mask(shape_hw, cx, cy, rx_out, ry_out)
    inner = _ellipse_mask(shape_hw, cx, cy, rx_in, ry_in)
    return cv2.subtract(outer, inner)


def _ring_threshold(gray: np.ndarray, ring_mask: np.ndarray) -> float:
    vals = gray[ring_mask > 0]
    if vals.size == 0:
        return 200.0
    p90 = float(np.percentile(vals, 90))
    thr = p90 - RING_P90_DELTA
    return float(np.clip(thr, 50.0, 245.0))


def _score_alignment(dx_band: np.ndarray, w: int, scale: float, shift: float, rx: float) -> float:
    score = 0.0
    for f in BUBBLE_X_FRACS:
        cx = shift + scale * (f * w)
        lx = int(round(cx - rx))
        rxp = int(round(cx + rx))

        if lx < 2 or rxp > w - 3:
            score -= 1e6
            continue

        lo1, hi1 = lx - BORDER_WIN, lx + BORDER_WIN + 1
        lo2, hi2 = rxp - BORDER_WIN, rxp + BORDER_WIN + 1
        score += float(dx_band[:, lo1:hi1].sum())
        score += float(dx_band[:, lo2:hi2].sum())
    return score


def _fit_x_model(dx_abs: np.ndarray, cy: int, w: int, rx: float) -> Tuple[float, float, float]:
    band_y0 = max(0, cy - BAND_HALF_H)
    band_y1 = min(dx_abs.shape[0], cy + BAND_HALF_H)
    dx_band = dx_abs[band_y0:band_y1, :]

    shift_max = int(round(SHIFT_FRAC * w))
    best_score = -1e18
    best_s = 1.0
    best_sh = 0.0

    s = SCALE_MIN
    while s <= SCALE_MAX + 1e-9:
        for sh in range(-shift_max, shift_max + 1, SHIFT_STEP_PX):
            sc = _score_alignment(dx_band, w, s, float(sh), rx)
            if sc > best_score:
                best_score = sc
                best_s = s
                best_sh = float(sh)
        s += SCALE_STEP

    return best_s, best_sh, best_score


# ============================================================
# Main API (drop-in)
# ============================================================

def process(
    question_crop: Union[np.ndarray, Path],
    question_number: int,
    debug_dir: Optional[Path] = None,
) -> Dict:

    # ---- Load ----
    if isinstance(question_crop, Path):
        img0 = cv2.imread(str(question_crop))
        if img0 is None:
            return {"answer": [], "status": "error", "confidence": 0.0, "error": "failed_to_load"}
    else:
        img0 = question_crop

    if img0 is None or img0.size == 0:
        return {"answer": [], "status": "error", "confidence": 0.0, "error": "empty_crop"}

    # ---- Normalize height only ----
    img = _resize_height(img0, CANONICAL_H)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # ---- Vertical-edge energy ----
    dx_abs = _abs_sobel_dx(gray)

    # ---- Find bubble row (cy) ----
    cy = _find_cy(dx_abs)
    band_y0 = max(0, cy - BAND_HALF_H)
    band_y1 = min(h, cy + BAND_HALF_H)

    # ---- Bubble geometry ----
    ry = float(RY_FRAC * CANONICAL_H)
    rx = float(RX_OVER_RY * ry)

    # ---- Fit X model ----
    best_scale, best_shift, best_score = _fit_x_model(dx_abs, cy, w, rx)

    cxs = sorted([int(round(best_shift + best_scale * (f * w))) for f in BUBBLE_X_FRACS])

    # ---- Fill detect per bubble (FIXED) ----
    answers: List[str] = []
    bubbles_dbg: List[Dict] = []

    vis = img.copy()
    cv2.rectangle(vis, (0, band_y0), (w - 1, band_y1), (255, 0, 255), 1)

    for i in range(4):
        cx = cxs[i]

        core = _ellipse_mask((h, w), cx, cy, rx * CORE_SCALE, ry * CORE_SCALE)
        halo = _annulus_mask((h, w), cx, cy, rx * HALO_INNER, ry * HALO_INNER, rx * HALO_OUTER, ry * HALO_OUTER)

        # ✅ threshold from clean background ring (outside halo)
        bg_ring = _annulus_mask(
            (h, w), cx, cy,
            rx * BG_RING_INNER, ry * BG_RING_INNER,
            rx * BG_RING_OUTER, ry * BG_RING_OUTER
        )

        thr = _ring_threshold(gray, bg_ring)

        core_vals = gray[core > 0]
        halo_vals = gray[halo > 0]

        dark_core = int(np.count_nonzero(core_vals < thr))
        dark_halo = int(np.count_nonzero(halo_vals < thr))

        core_area = int(core_vals.size)
        halo_area = int(halo_vals.size)

        core_ratio = float(dark_core) / float(max(1, core_area))
        halo_ratio = float(dark_halo) / float(max(1, halo_area))

        # ✅ fixed decision (density-based, outside-ink aware)
        filled = (
            (dark_core >= MIN_CORE_DARK_PIXELS) and
            (core_ratio >= CORE_DARK_RATIO_MIN) and
            (core_ratio >= halo_ratio + CORE_MINUS_HALO_MIN)
        )

        if filled:
            answers.append(LABELS_LTR[i])

        # ---- Debug draw ----
        color = (0, 255, 0) if filled else (0, 0, 255)
        cv2.ellipse(vis, (cx, cy), (int(rx), int(ry)), 0, 0, 360, color, 2)
        cv2.ellipse(vis, (cx, cy), (int(rx * CORE_SCALE), int(ry * CORE_SCALE)), 0, 0, 360, (255, 0, 0), 1)
        cv2.ellipse(vis, (cx, cy), (int(rx * HALO_OUTER), int(ry * HALO_OUTER)), 0, 0, 360, (0, 255, 255), 1)

        cv2.putText(
            vis,
            f"{LABELS_LTR[i]} c={dark_core}({core_ratio:.2f}) h={dark_halo}({halo_ratio:.2f}) thr={thr:.0f}",
            (max(0, cx - int(rx)), max(12, cy - int(ry) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

        bubbles_dbg.append({
            "label": LABELS_LTR[i],
            "cx": int(cx),
            "cy": int(cy),
            "thr": float(thr),
            "dark_core": dark_core,
            "core_area": core_area,
            "core_ratio": core_ratio,
            "dark_halo": dark_halo,
            "halo_area": halo_area,
            "halo_ratio": halo_ratio,
            "filled": bool(filled),
        })

    status = "ok" if answers else "blank"
    confidence = CONF_OK if answers else CONF_BLANK

    cv2.putText(
        vis,
        f"Q{question_number} {status} ans={answers} s={best_scale:.3f} sh={best_shift:.0f}",
        (6, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )

    debug_paths: Dict[str, str] = {}
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        img_p = debug_dir / f"stage3_Q{question_number:03d}_mcq_debug.jpg"
        json_p = debug_dir / f"stage3_Q{question_number:03d}_mcq_debug.json"

        cv2.imwrite(str(img_p), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        json_p.write_text(json.dumps({
            "question": question_number,
            "status": status,
            "answer": answers,
            "w": int(w),
            "h": int(h),
            "cy": int(cy),
            "band": [int(band_y0), int(band_y1)],
            "fit": {"scale": float(best_scale), "shift": float(best_shift), "score": float(best_score)},
            "cxs": [int(x) for x in cxs],
            "params": {
                "RING_P90_DELTA": RING_P90_DELTA,
                "MIN_CORE_DARK_PIXELS": MIN_CORE_DARK_PIXELS,
                "CORE_DARK_RATIO_MIN": CORE_DARK_RATIO_MIN,
                "CORE_MINUS_HALO_MIN": CORE_MINUS_HALO_MIN,
                "CORE_SCALE": CORE_SCALE,
                "BG_RING_INNER": BG_RING_INNER,
                "BG_RING_OUTER": BG_RING_OUTER,
            },
            "bubbles": bubbles_dbg,
        }, indent=2), encoding="utf-8")

        debug_paths = {"image": str(img_p), "json": str(json_p)}

    return {
        "answer": answers,
        "status": status,
        "confidence": float(confidence),
        "debug": debug_paths,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--q", required=True, type=int)
    ap.add_argument("--debug_dir", required=True, type=Path)
    args = ap.parse_args()
    res = process(args.image, args.q, args.debug_dir)
    print(json.dumps(res, indent=2))
