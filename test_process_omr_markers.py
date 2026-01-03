import shutil
import unittest
from pathlib import Path

import process_omr_markers


class ProcessOmrMarkersIntegrationTest(unittest.TestCase):
    def test_all_sample_inputs_process_successfully(self) -> None:
        input_dir = Path("inputs")
        output_dir = Path("output/test_crops")
        debug_dir = Path("output/test_debug")

        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(debug_dir, ignore_errors=True)

        results = process_omr_markers.run(input_dir, output_dir, debug_dir)

        photographed = {
            "filled_sheet_all_mcq_photographed.jpg",
            "filled_sheet_all_mcq_photographed_2.jpg",
            "filled_sheet_all_mcq_photographed_3.jpg",
        }
        failures = {name: status for name, status in results.items() if name in photographed and status != "success"}
        self.assertDictEqual(failures, {})


if __name__ == "__main__":
    unittest.main()
