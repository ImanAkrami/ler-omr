#!/usr/bin/env python3
"""
TF processor — FIXED placement from RIGHT edge for BOTH bubbles (no detection)

Left bubble  = false
Right bubble = true

No scanning, no snapping, no curves, no contours.
Just fixed coordinates + fill scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import cv2
import numpy as np


# ============================================================
# 1) FIXED PLACEMENT (FROM RIGHT EDGE) — THIS IS THE FIX
# ============================================================

# These are "how far from the RIGHT edge is the bubble center"
# cx = w * (1 - margin)

TRUE_RIGHT_MARGIN_FRAC  = 0.19   # ~0.82w
FALSE_RIGHT_MARGIN_FRAC = 0.43   # ~0.58w  <-- moved left so it won't sit on "غلط"

CY_FRAC = 0.50  # fixed vertical center


# ============================================================
# 2) FIXED SHAPES
# ============================================================

OVAL_H_FRAC = 0.55
OVAL_W_OVER_H = 1.60

CORE_SCALE = 0.45
BG_INNER_SCALE = 1.90
BG_OUTER_SCALE = 2.80


# ============================================================
# 3) DECISION CONSTANTS
# ============================================================

SCORE_MIN = 8.0
WINNER_GAP_MIN = 4.0

BOTH_SCORE_MIN = 18.0
BOTH_GAP_MAX = 3.5

CONF_OK = 0.95
CONF_BLANK = 0.90


# ============================================================
# Helpers
# ============================================================

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


def _masked_mean(gray: np.ndarray, mask: np.ndarray) -> float:
    vals = gray[mask > 0]
    if vals.size == 0:
        return 255.0
    return float(np.mean(vals))


def _score_fill(gray: np.ndarray, cx: int, cy: int, rx: float, ry: float) -> Dict:
    core = _ellipse_mask(gray.shape[:2], cx, cy, rx * CORE_SCALE, ry * CORE_SCALE)
    bg = _annulus_mask(gray.shape[:2], cx, cy,
                       rx * BG_INNER_SCALE, ry * BG_INNER_SCALE,
                       rx * BG_OUTER_SCALE, ry * BG_OUTER_SCALE)

    core_mean = _masked_mean(gray, core)
    bg_mean = _masked_mean(gray, bg)
    score = float(bg_mean - core_mean)

    return {"core_mean": core_mean, "bg_mean": bg_mean, "score": score}


# ============================================================
# Main API
# ============================================================

def process(
    question_crop: Union[np.ndarray, Path],
    question_number: int,
    debug_dir: Optional[Path] = None,
) -> Dict:

    if isinstance(question_crop, Path):
        img = cv2.imread(str(question_crop))
        if img is None:
            return {"answer": None, "status": "error", "confidence": 0.0, "debug": None}
    else:
        img = question_crop

    if img is None or img.size == 0:
        return {"answer": None, "status": "error", "confidence": 0.0, "debug": None}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cy = int(round(h * CY_FRAC))
    ry = float((h * OVAL_H_FRAC) / 2.0)
    rx = float(ry * OVAL_W_OVER_H)

    # ✅ fixed from right (independent)
    cx_true  = int(round(w * (1.0 - TRUE_RIGHT_MARGIN_FRAC)))
    cx_false = int(round(w * (1.0 - FALSE_RIGHT_MARGIN_FRAC)))

    # clamp safely (still fixed logic; prevents going out of frame)
    pad = int(round(rx * 0.65))
    cx_true = int(np.clip(cx_true, pad, w - 1 - pad))
    cx_false = int(np.clip(cx_false, pad, w - 1 - pad))

    # scores
    s_false = _score_fill(gray, cx_false, cy, rx, ry)
    s_true  = _score_fill(gray, cx_true,  cy, rx, ry)

    sf = float(s_false["score"])
    st = float(s_true["score"])
    gap = abs(sf - st)

    false_on = sf >= SCORE_MIN
    true_on = st >= SCORE_MIN

    if false_on and true_on:
        if (sf >= BOTH_SCORE_MIN) and (st >= BOTH_SCORE_MIN) and (gap <= BOTH_GAP_MAX):
            answer = "both"
        else:
            answer = "false" if sf > st else "true"
        status = "ok"
        confidence = CONF_OK

    elif false_on or true_on:
        answer = "false" if false_on else "true"
        status = "ok"
        confidence = CONF_OK

    else:
        if gap >= WINNER_GAP_MIN:
            answer = "false" if sf > st else "true"
            status = "ok"
            confidence = CONF_OK
        else:
            answer = None
            status = "blank"
            confidence = CONF_BLANK

    # debug
    debug_paths = None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = img.copy()

        def draw(cx: int, label: str, data: Dict, chosen: bool):
            color = (0, 255, 0) if chosen else (0, 0, 255)
            cv2.ellipse(vis, (cx, cy), (int(rx), int(ry)), 0, 0, 360, color, 2)
            cv2.ellipse(vis, (cx, cy), (int(rx * CORE_SCALE), int(ry * CORE_SCALE)), 0, 0, 360, (255, 0, 0), 1)
            cv2.ellipse(vis, (cx, cy), (int(rx * BG_OUTER_SCALE), int(ry * BG_OUTER_SCALE)), 0, 0, 360, (0, 255, 255), 1)
            cv2.putText(
                vis,
                f"{label} score={data['score']:.1f} core={data['core_mean']:.0f} bg={data['bg_mean']:.0f}",
                (max(0, cx - int(rx)), max(14, cy - int(ry) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        chosen_false = answer in ("false", "both")
        chosen_true  = answer in ("true", "both")

        draw(cx_false, "false", s_false, chosen_false)
        draw(cx_true,  "true",  s_true,  chosen_true)

        cv2.putText(
            vis,
            f"Q{question_number} {status} ans={answer} gap={gap:.1f}",
            (6, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

        img_p = debug_dir / f"stage3_tf_Q{question_number}_debug.jpg"
        json_p = debug_dir / f"stage3_tf_Q{question_number}_debug.json"

        cv2.imwrite(str(img_p), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        json_p.write_text(json.dumps({
            "question": question_number,
            "answer": answer,
            "status": status,
            "confidence": confidence,
            "placement": {
                "TRUE_RIGHT_MARGIN_FRAC": TRUE_RIGHT_MARGIN_FRAC,
                "FALSE_RIGHT_MARGIN_FRAC": FALSE_RIGHT_MARGIN_FRAC,
                "cx_false": cx_false,
                "cx_true": cx_true,
                "cy": cy,
            },
            "scores": {"false": sf, "true": st, "gap": gap},
            "params": {
                "SCORE_MIN": SCORE_MIN,
                "WINNER_GAP_MIN": WINNER_GAP_MIN,
                "BOTH_SCORE_MIN": BOTH_SCORE_MIN,
                "BOTH_GAP_MAX": BOTH_GAP_MAX,
            }
        }, indent=2), encoding="utf-8")

        debug_paths = {"image": str(img_p), "json": str(json_p)}

    return {
        "answer": answer,
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
    out = process(args.image, args.q, args.debug_dir)
    print(json.dumps(out, indent=2))
