import shutil
import unittest
from pathlib import Path

import main_processor


class ProcessOmrMarkersIntegrationTest(unittest.TestCase):
    def test_all_sample_inputs_process_successfully(self) -> None:
        input_dir = Path("inputs")
        output_root = Path("output")

        # Clean up old test outputs
        if output_root.exists():
            for item in output_root.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)

        # Process each image file individually
        results = {}
        for image_path in sorted(input_dir.glob("*.jpg")):
            result = main_processor.process_single_file(image_path, input_dir, output_root)
            results[image_path.name] = result["overall_status"]

        photographed = {
            "filled_sheet_all_mcq_photographed.jpg",
            "filled_sheet_all_mcq_photographed_2.jpg",
            "filled_sheet_all_mcq_photographed_3.jpg",
        }
        failures = {name: status for name, status in results.items() if name in photographed and status != "success"}
        self.assertDictEqual(failures, {})


if __name__ == "__main__":
    unittest.main()
