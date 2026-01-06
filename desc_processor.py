# desc_processor.py

from typing import Dict, Optional, Union
from pathlib import Path
import base64
import json

import cv2
import numpy as np


def _image_to_base64_png(img: np.ndarray) -> str:
    """Encode image (BGR numpy array) as base64 PNG."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def process(
    question_crop: Union[np.ndarray, Path],
    question_number: int,
    debug_dir: Optional[Path] = None,
) -> Dict:
    """
    Process a descriptive (DESC) question.

    Strategy:
    - NO detection
    - NO interpretation
    - Just return the cropped image as base64
    """

    # ---- Load image ----
    if isinstance(question_crop, Path):
        img = cv2.imread(str(question_crop))
        if img is None:
            return {
                "answer": None,
                "status": "error",
                "confidence": 0.0,
                "error": "failed_to_load_image",
                "debug": None,
            }
    else:
        img = question_crop

    if img is None or img.size == 0:
        return {
            "answer": None,
            "status": "error",
            "confidence": 0.0,
            "error": "empty_crop",
            "debug": None,
        }

    # ---- Encode ----
    try:
        img_b64 = _image_to_base64_png(img)
    except Exception as e:
        return {
            "answer": None,
            "status": "error",
            "confidence": 0.0,
            "error": str(e),
            "debug": None,
        }

    result = {
        "answer": {
            "image": img_b64,
            "encoding": "base64_png",
        },
        "status": "ok",
        "confidence": 1.0,  # deterministic, no inference
        "debug": None,
    }

    # ---- Debug output (optional) ----
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

        img_path = debug_dir / f"stage3_Q{question_number:03d}_desc_crop.png"
        json_path = debug_dir / f"stage3_Q{question_number:03d}_desc_debug.json"

        cv2.imwrite(str(img_path), img)

        json_path.write_text(
            json.dumps(
                {
                    "question_number": question_number,
                    "type": "DESC",
                    "status": "ok",
                    "note": "Descriptive question – raw image returned",
                    "image_file": str(img_path.name),
                    "encoding": "base64_png",
                    "image_shape": {
                        "height": img.shape[0],
                        "width": img.shape[1],
                        "channels": img.shape[2] if img.ndim == 3 else 1,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        result["debug"] = {
            "image": str(img_path),
            "json": str(json_path),
        }

    return result
