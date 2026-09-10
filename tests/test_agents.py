"""Tests for Sherlock Specialized Agents."""

import unittest
from pathlib import Path
import tempfile
import shutil

from config import SherlockConfig
from app.analysis.context import AnalysisContext
from app.agents.repository_mapper import RepositoryMapperAgent
from app.agents.flow_tracer import FlowTracerAgent
from app.agents.component_analyst import ComponentAnalystAgent
from app.agents.data_integration_analyst import DataIntegrationAnalystAgent
from app.agents.gap_analyst import GapAnalystAgent
from app.agents.teacher import TeacherAgent


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sherlock_agent_test_")
        self.root = Path(self.test_dir)
        self.cfg = SherlockConfig(workspace_dir=self.root / "workspace")

        (self.root / "app").mkdir()
        with open(self.root / "app" / "main.py", "w") as f:
            f.write("def main():\n    # TODO: add args\n    print('hello')\n")

        with open(self.root / "requirements.txt", "w") as f:
            f.write("fastapi==0.100.0\npydantic==2.0\n")

        self.ctx = AnalysisContext(self.root)
        self.ctx.initialize_index()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_agent_1_mapper(self):
        agent = RepositoryMapperAgent(sherlock_cfg=self.cfg)
        rep_map = agent.run(self.ctx)
        self.assertIsNotNone(rep_map)
        self.assertEqual(rep_map.project_name, self.root.name)
        self.assertIn("FastAPI", rep_map.frameworks)

    def test_agent_2_flow(self):
        agent = FlowTracerAgent(sherlock_cfg=self.cfg)
        flows = agent.run(self.ctx)
        self.assertIsNotNone(flows)
        self.assertTrue(len(flows.key_flows) >= 1)

    def test_agent_3_components(self):
        agent = ComponentAnalystAgent(sherlock_cfg=self.cfg)
        comps = agent.run(self.ctx)
        self.assertIsNotNone(comps)
        self.assertTrue(len(comps.components) >= 1)

    def test_agent_4_data(self):
        agent = DataIntegrationAnalystAgent(sherlock_cfg=self.cfg)
        data_int = agent.run(self.ctx)
        self.assertIsNotNone(data_int)
        self.assertTrue(len(data_int.data_flows) >= 1)

    def test_agent_5_gaps(self):
        agent = GapAnalystAgent(sherlock_cfg=self.cfg)
        gaps = agent.run(self.ctx)
        self.assertIsNotNone(gaps)
        self.assertTrue(len(gaps.gaps) >= 1)
        # Should have found the TODO in main.py
        self.assertTrue(any("add args" in g.gap for g in gaps.gaps))

    def test_agent_6_teacher(self):
        # Run prior agents first to populate context
        RepositoryMapperAgent(sherlock_cfg=self.cfg).run(self.ctx)
        FlowTracerAgent(sherlock_cfg=self.cfg).run(self.ctx)
        ComponentAnalystAgent(sherlock_cfg=self.cfg).run(self.ctx)
        DataIntegrationAnalystAgent(sherlock_cfg=self.cfg).run(self.ctx)
        GapAnalystAgent(sherlock_cfg=self.cfg).run(self.ctx)

        teacher = TeacherAgent(sherlock_cfg=self.cfg)
        teaching_report = teacher.run(self.ctx)
        self.assertIsNotNone(teaching_report)
        self.assertEqual(len(teaching_report.answers_to_25_questions), 25)
        self.assertIsNotNone(self.ctx.ten_page_report)
        self.assertEqual(len(self.ctx.ten_page_report.pages), 10)


if __name__ == "__main__":
    unittest.main()
