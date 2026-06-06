from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from demo import run_demo


class DemoRunnerTests(unittest.TestCase):
    def test_parse_metric_cell_extracts_mean_value(self) -> None:
        self.assertEqual(run_demo.parse_metric_cell("0.490 +/- 0.032"), 0.49)
        self.assertEqual(run_demo.parse_metric_cell("-599.108 +/- 236.114"), -599.108)
        self.assertIsNone(run_demo.parse_metric_cell(""))

    def test_build_demo_bundle_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            bundle = run_demo.build_demo_bundle(output_dir)

            report_path = output_dir / "DEMO_REPORT.md"
            summary_path = output_dir / "demo_summary.json"
            manifest_path = output_dir / "manifest.json"

            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(len(bundle["isaac_rows"]), 3)
            self.assertEqual(len(bundle["mujoco_rows"]), 3)
            self.assertEqual(len(bundle["reliability_rows"]), 3)
            self.assertTrue((output_dir / "figures" / "figure_mechanism_chain.png").exists())

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Current Thesis", report_text)
            self.assertIn("Matched MuJoCo Replay", report_text)
            self.assertIn("Reliability Snapshot", report_text)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["thesis"], run_demo.THESIS)
            self.assertIn("isaac_metric_winners", summary)
            self.assertIn("mujoco_metric_winners", summary)
            self.assertIn("data/table_full_paper_isaac_mechanism_comparison.csv", summary["source_files"])
            self.assertIn("data/upstream_sources.json", summary["source_files"])


if __name__ == "__main__":
    unittest.main()
