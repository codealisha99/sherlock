"""Integration test for Sherlock Orchestrator."""

import unittest
from pathlib import Path
import tempfile
import shutil

from config import SherlockConfig
from app.orchestrator.orchestrator import Orchestrator


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sherlock_orch_test_")
        self.root = Path(self.test_dir)
        self.workspace = self.root / "workspace"
        self.cfg = SherlockConfig(workspace_dir=self.workspace)

        # Create dummy project
        (self.root / "src").mkdir()
        with open(self.root / "src" / "app.py", "w") as f:
            f.write("def start_app():\n    # TODO: write docs\n    pass\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_analysis_end_to_end(self):
        orch = Orchestrator(sherlock_cfg=self.cfg)
        report = orch.run_analysis(self.root)

        self.assertIsNotNone(report)
        self.assertEqual(len(report.pages), 10)

        # Check generated artifacts on disk
        analysis_dir = self.workspace / report.analysis_id
        self.assertTrue(analysis_dir.is_dir())
        self.assertTrue((analysis_dir / "repository_map.json").is_file())
        self.assertTrue((analysis_dir / "flows.json").is_file())
        self.assertTrue((analysis_dir / "components.json").is_file())
        self.assertTrue((analysis_dir / "data_integrations.json").is_file())
        self.assertTrue((analysis_dir / "gaps.json").is_file())
        self.assertTrue((analysis_dir / "teaching_report.json").is_file())
        self.assertTrue((analysis_dir / "ten_page_report.json").is_file())
        self.assertTrue((analysis_dir / "README_ANALYSIS.md").is_file())


if __name__ == "__main__":
    unittest.main()
