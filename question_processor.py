"""
Question processor that processes all sliced questions.

Takes all question crops + metadata and processes them using type-specific processors.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import desc_processor
import mcq_processor
import tf_processor


def get_processor_for_type(question_type: str):
    processors = {
        "MCQ": mcq_processor.process,
        "TF": tf_processor.process,
        "DESC": desc_processor.process,
    }
    return processors.get(question_type.upper())


def process_question_crop(
    question_crop: np.ndarray,
    question_number: int,
    question_type: str,
    debug_dir: Optional[Path] = None,
) -> Dict:
    processor = get_processor_for_type(question_type)
    if processor is None:
        return {"answer": None, "status": "error", "error": f"Unknown question type: {question_type}"}

    try:
        # New signature supports debug_dir (MCQ uses it; TF/DESC can ignore for now)
        return processor(question_crop, question_number, debug_dir=debug_dir)
    except TypeError:
        # Backward compatibility for processors not yet updated
        return processor(question_crop, question_number)
    except Exception as e:
        return {"answer": None, "status": "error", "error": str(e)}


def process_all_questions(
    question_crops: List[Dict],
    file_output_dir: Path,
    metadata: Optional[Dict] = None
) -> List[Dict]:
    q_dir = file_output_dir / "questions"
    q_dir.mkdir(parents=True, exist_ok=True)

    # Stage 3 debug directory (all types in same folder)
    stage3_debug_dir = file_output_dir / "debug" / "stage3"
    stage3_debug_dir.mkdir(parents=True, exist_ok=True)

    processed_results: List[Dict] = []
    mcq_summary: List[Dict] = []

    for q_data in question_crops:
        q_index = int(q_data["q_index"])
        crop = q_data["crop"]
        question_type = q_data.get("question_type")
        question_meta = q_data.get("question_meta")

        # Save crop image to disk (for debug / inspection)
        crop_path = q_dir / q_data["name"]
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if question_type is None:
            continue

        # All types use the same debug directory
        debug_dir = stage3_debug_dir

        processing_result = process_question_crop(
            crop,
            q_index,
            question_type,
            debug_dir=debug_dir,
        )

        question_result = {
            "question_number": q_index,
            "type": question_type,
            "result": processing_result,
            "meta": question_meta,
        }

        # Save individual question result JSON
        question_result_path = q_dir / f"Q{q_index}_res.json"
        question_result_path.write_text(json.dumps(question_result, indent=2), encoding="utf-8")

        processed_results.append({
            "question_number": q_index,
            "type": question_type,
            "result": processing_result,
        })

        if str(question_type).upper() == "MCQ":
            mcq_summary.append({
                "question_number": q_index,
                "answer": processing_result.get("answer"),
                "status": processing_result.get("status"),
                "confidence": processing_result.get("confidence"),
                "debug": processing_result.get("debug"),
            })

    # Combined results
    if processed_results:
        (file_output_dir / "result.json").write_text(json.dumps(processed_results, indent=2), encoding="utf-8")

    # Uploadable MCQ summary
    if mcq_summary:
        (stage3_debug_dir / "stage3_mcq_summary.json").write_text(json.dumps(mcq_summary, indent=2), encoding="utf-8")

    return processed_results
