"""Main processor that orchestrates Stage-1 (corner detection/warp) and Stage-2 (question slicing).

This processor accepts a single image file and processes it through both stages.
"""
from pathlib import Path
from typing import Dict

import omr_cropper
import slice_questions


def process_single_file(image_path: Path, input_dir: Path, output_dir: Path, debug_dir: Path) -> Dict:
    """Process a single image file through Stage-1 (cropping) and Stage-2 (question slicing).

    Args:
        image_path: Path to the input image file (e.g., inputs/foo.jpg)
        input_dir: Directory containing input files (for finding JSON metadata)
        output_dir: Directory for Stage-1 crop output
        debug_dir: Directory for debug outputs

    Returns:
        Dictionary with processing status and results
    """
    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Stage-1: Corner detection and warping
    crop_debug_data = omr_cropper.process_image(image_path, output_dir, debug_dir)

    if crop_debug_data["status"] != "success":
        return {
            "image": image_path.name,
            "stage1_status": crop_debug_data["status"],
            "stage2_status": "skipped",
            "overall_status": "failed",
        }

    # Stage-2: Question slicing
    try:
        crop_path = output_dir / f"{image_path.stem}_crop.png"

        if not crop_path.exists():
            return {
                "image": image_path.name,
                "stage1_status": "success",
                "stage2_status": "error",
                "error": f"Crop file not found: {crop_path}",
                "overall_status": "failed",
            }

        # Determine output root for questions
        # Output structure: output/questions/<sheet_stem>/questions/ and debug/
        sheet_stem = image_path.stem
        questions_output_root = Path("output") / "questions"

        slice_questions.slice_sheet(crop_path, questions_output_root)

        return {
            "image": image_path.name,
            "stage1_status": "success",
            "stage2_status": "success",
            "overall_status": "success",
        }
    except Exception as exc:
        return {
            "image": image_path.name,
            "stage1_status": "success",
            "stage2_status": "error",
            "error": str(exc),
            "overall_status": "failed",
        }

