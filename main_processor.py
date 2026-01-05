"""Main processor that orchestrates Stage-1 (corner detection/warp) and Stage-2 (question slicing).

This processor accepts a single image file and processes it through both stages.
All image data is passed in memory; only debug outputs are saved to disk.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import cv2

import omr_cropper
import question_processor
import slice_questions


def load_metadata(json_path: Optional[Path]) -> Optional[Dict]:
    """Load question metadata from JSON file."""
    if json_path is None or not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def process_single_file(image_path: Path, input_dir: Path, output_root: Path) -> Dict:
    """Process a single image file through Stage-1 (cropping), Stage-2 (question slicing), and Stage-3 (question processing).

    Args:
        image_path: Path to the input image file (e.g., inputs/foo.jpg)
        input_dir: Directory containing input files (for finding JSON metadata)
        output_root: Root output directory (e.g., Path("output"))

    Returns:
        Dictionary with processing status and results
    """
    # Create unified output structure: output/{filename}/
    filename_stem = image_path.stem
    file_output_dir = output_root / filename_stem

    # Create subdirectories
    crops_dir = file_output_dir / "crops"
    debug_dir = file_output_dir / "debug"
    questions_dir = file_output_dir / "questions"

    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "image": image_path.name,
            "stage1_status": "error",
            "error": f"Failed to load image: {image_path}",
            "overall_status": "failed",
        }

    # Stage-1: Corner detection and warping
    crop_debug_data, warped_image = omr_cropper.process_image(image_path, crops_dir, debug_dir)

    if crop_debug_data["status"] != "success" or warped_image is None:
        return {
            "image": image_path.name,
            "stage1_status": crop_debug_data["status"],
            "stage2_status": "skipped",
            "overall_status": "failed",
        }

    # Stage-2: Question slicing (in memory)
    try:
        # Load JSON metadata for Stage-2 (optional)
        json_path = input_dir / f"{image_path.stem}.json"
        metadata = load_metadata(json_path if json_path.exists() else None)

        # Slice questions (returns crops in memory)
        question_crops, slice_debug_data = slice_questions.slice_questions(
            warped_image,
            file_output_dir,
            filename_stem,
            metadata
        )

        # Note: stage2_data.json is saved by slice_questions itself

        # Stage-3: Process all questions
        processed_results = question_processor.process_all_questions(
            question_crops,
            file_output_dir,
            metadata
        )

        return {
            "image": image_path.name,
            "stage1_status": "success",
            "stage2_status": "success",
            "stage3_status": "success",
            "overall_status": "success",
            "questions_processed": len(processed_results),
        }
    except Exception as exc:
        return {
            "image": image_path.name,
            "stage1_status": "success",
            "stage2_status": "error",
            "error": str(exc),
            "overall_status": "failed",
        }

